#!/usr/bin/env python3
"""Cluster-side pipeline orchestrator — survives SSH/session disconnects.

Runs as a tiny Slurm job (scripts/slurm/decider.sbatch) chained with
--dependency=afterany behind eval jobs. Each stage reads finished
results, writes outputs/PIPELINE_STATUS.md (the user-readable truth),
commits small results into the repo (push happens from a login session
later), and submits the next stage's jobs plus itself.

Stages:
  confirm3x3  pick anchor-grid winner from the 3x3 confirmations,
              launch test-10 (winner vs v3) split over the fast lanes
  dev10       aggregate test-10, pick overall winner, launch the full
              220-route benchmark in <=4h chunks
  bench220    aggregate benchmark chunks into the final table

Lane choice is DYNAMIC (user rule 2026-07-29: never hardcode which
node/partition is best — audit at submission time). pick_lanes() counts
actually-free GPUs per candidate partition via sinfo Gres/GresUsed and
orders lanes by free capacity; outputs/fast_lanes.txt (one partition
per line) overrides when a human/audit knows better. Only the
*capability map* (how to request a GPU on each partition, which is
graphics-capable) is static knowledge.
"""
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import traceback

OUT = "/scratch/srahmani/ditto_av/outputs"
REPO = "/scratch/srahmani/ditto_av/DITTO_AV"
B2D = "/scratch/srahmani/ditto_av/Bench2Drive"
STATUS = os.path.join(OUT, "PIPELINE_STATUS.md")
CHAIN = os.path.join(REPO, "scripts/slurm/carla_eval_chain.sbatch")
DECIDER = os.path.join(REPO, "scripts/slurm/decider.sbatch")

DEV10_A = "3514,3255,26405,25381,25378"
DEV10_B = "25424,2091,27494,17569,28198"

CANDIDATES = {  # 3x3 confirmations; kl01's 3x3 is already committed
    "kl01": (os.path.join(REPO, "runs/carla_smoke/kl01_3x3.json"),
             "configs/diag_kl01.yaml"),
    "kl015": (os.path.join(OUT, "carla_results_kl015_3x3.json"),
              "configs/diag_kl015.yaml"),
    "kl02": (os.path.join(OUT, "carla_results_kl02_3x3.json"),
             "configs/diag_kl02.yaml"),
    "v5kl01": (os.path.join(OUT, "carla_results_v5kl01_3x3.json"),
               "configs/diag_v5kl01.yaml"),
    "kl01k16": (os.path.join(OUT, "carla_results_kl01k16_3x3.json"),
                "configs/diag_kl01k16.yaml"),
}
V3_CONF = "configs/diag_fix_norec.yaml"


def log(msg):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def records(path):
    d = json.load(open(path))
    return d.get("_checkpoint", d).get("records", [])


def agg(recs):
    if not recs:
        return None
    n = len(recs)
    comp = sum(r["scores"]["score_route"] for r in recs) / n
    pen = sum(r["scores"]["score_penalty"] for r in recs) / n
    score = sum(r["scores"]["score_composed"] for r in recs) / n
    done = sum(1 for r in recs if r["status"] == "Completed")
    return dict(n=n, completion=comp, penalty=pen, score=score, completed=done)


def sbatch(args, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    out = subprocess.run(["sbatch", "--parsable"] + args, env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


# capability map (semi-static): graphics-capable partitions usable for
# CARLA and how a GPU must be requested there. MIG partitions are
# deliberately absent (no graphics). Statuses/queues are NOT encoded
# here — pick_lanes() measures those live.
LANE_FLAGS = {
    "participation": ["--gpus-per-task=1"],
    # unmanaged GPU, no GRES flag. De-facto node exclusivity via CPU
    # pressure (>half of the 32 cores): --exclusive needs --mem=0 which
    # clashes with the chain script's --mem-per-cpu directive.
    "visual": ["--cpus-per-task=17"],
    "gpu-a100": ["--gpus-per-task=1"],
    "gpu-v100": ["--gpus-per-task=1"],
}


def free_gpus(part):
    """Actually-free GPUs right now (squeue lies under PrivateData)."""
    out = subprocess.run(
        ["sinfo", "-h", "-N", "-p", part, "-O", "Gres:40,GresUsed:40"],
        capture_output=True, text=True).stdout
    total = used = idle_nodes = 0
    for line in out.splitlines():
        m = re.findall(r"gpu[:\w.]*:(\d+)", line)
        if len(m) >= 2:
            total += int(m[0])
            used += int(m[1])
        elif "null" in line:
            idle_nodes += 1  # no GRES (visual): usable if node not busy
    if total:
        return total - used
    st = subprocess.run(["sinfo", "-h", "-p", part, "-t", "idle", "-o", "%D"],
                        capture_output=True, text=True).stdout.split()
    return sum(int(x) for x in st)


def pick_lanes():
    override = os.path.join(OUT, "fast_lanes.txt")
    if os.path.exists(override):
        lanes = [l.strip() for l in open(override) if l.strip() in LANE_FLAGS]
        if lanes:
            log(f"lanes from override file: {lanes}")
            return lanes
    scored = []
    for p in LANE_FLAGS:
        try:
            scored.append((free_gpus(p), p))
        except Exception as e:
            log(f"lane audit: {p} unreadable ({e})")
    scored.sort(reverse=True)
    log(f"lane audit (free GPUs now): {scored}")
    lanes = [p for n, p in scored if n > 0]
    return lanes[:2] if lanes else ["participation", "visual"]


def submit_chain(lane, name, variants, routes_xml=None):
    extra = {"VARIANTS": variants}
    if routes_xml:
        extra["ROUTES_XML"] = routes_xml
    args = ["-p", lane] + LANE_FLAGS[lane] + \
        ["--job-name", name, "--export=ALL", CHAIN]
    jid = sbatch(args, extra)
    log(f"submitted {name} -> {lane} job {jid}")
    return jid


def submit_next_stage(stage, dep_jobs):
    dep = "afterany:" + ":".join(dep_jobs)
    jid = sbatch(["-p", "visual", "--dependency", dep,
                  "--job-name", f"decide-{stage}", DECIDER, stage])
    log(f"decider stage {stage} queued as job {jid} (after {len(dep_jobs)} jobs)")


def git_commit(paths, msg):
    try:
        subprocess.run(["git", "-C", REPO, "add"] + paths, check=True)
        r = subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", msg])
        log(f"git commit {'ok' if r.returncode == 0 else 'nothing to commit'}"
            " (push from a login session)")
    except Exception as e:
        log(f"git commit FAILED (non-fatal): {e}")


def stage_confirm3x3():
    log("== stage confirm3x3: anchor-grid verdict ==")
    table = {}
    for name, (path, conf) in CANDIDATES.items():
        try:
            table[name] = agg(records(path))
            t = table[name]
            log(f"  {name:8s} completion {t['completion']:5.1f}%  penalty "
                f"{t['penalty']:.3f}  score {t['score']:5.2f}  ({t['n']} runs)")
        except Exception as e:
            table[name] = None
            log(f"  {name:8s} MISSING/UNREADABLE ({e})")
    live = {k: v for k, v in table.items() if v}
    if not live:
        log("NO candidate results at all — stopping; investigate eval chains")
        return
    winner = max(live, key=lambda k: live[k]["completion"])
    log(f"WINNER (mean completion): {winner}")

    t12 = None
    try:
        t12 = agg(records(os.path.join(OUT, "carla_results_t12_smoke.json")))
        log(f"  t12 smoke: completion {t12['completion']:.1f}% "
            f"({t12['n']} run) — Town12 leaderboard path "
            f"{'OK' if t12['completion'] > 0 else 'DUBIOUS'}")
    except Exception as e:
        log(f"  t12 smoke MISSING ({e}) — test-10 will still launch; "
            "Town12 was boot-verified, watch its routes")

    new_json = [p for n, (p, _) in CANDIDATES.items()
                if p.startswith(OUT) and os.path.exists(p)]
    dest = []
    for p in new_json:
        d = os.path.join(REPO, "runs/carla_smoke", os.path.basename(p)
                         .replace("carla_results_", ""))
        subprocess.run(["cp", p, d])
        dest.append(d)
    git_commit(dest, f"3x3 confirmations (H100/Quadro lanes): winner {winner} "
               + ", ".join(f"{k} {v['completion']:.1f}%" for k, v in live.items()))

    wconf = CANDIDATES[winner][1]
    lanes = pick_lanes()
    specs = [("d10-win-A", f"d10_{winner}_A:{wconf}:{DEV10_A}:3"),
             ("d10-win-B", f"d10_{winner}_B:{wconf}:{DEV10_B}:3"),
             ("d10-v3-A", f"d10_v3_A:{V3_CONF}:{DEV10_A}:3"),
             ("d10-v3-B", f"d10_v3_B:{V3_CONF}:{DEV10_B}:3")]
    jobs = [submit_chain(lanes[i % len(lanes)], name, var)
            for i, (name, var) in enumerate(specs)]
    with open(os.path.join(OUT, "dev10_winner.txt"), "w") as f:
        f.write(f"{winner}\n{wconf}\n")
    submit_next_stage("dev10", jobs)


def stage_dev10():
    log("== stage dev10: winner-vs-v3 on the 10 dev routes ==")
    winner, wconf = open(os.path.join(OUT, "dev10_winner.txt")).read().split()
    groups = {winner: [], "v3": []}
    for name, tags in ((winner, ("A", "B")), ("v3", ("A", "B"))):
        for t in tags:
            p = os.path.join(OUT, f"carla_results_d10_{name}_{t}.json")
            try:
                groups[name] += records(p)
            except Exception as e:
                log(f"  d10_{name}_{t} MISSING ({e}) — aggregate is partial")
    summary = {}
    for name, recs in groups.items():
        summary[name] = agg(recs)
        if summary[name]:
            s = summary[name]
            log(f"  {name:8s} completion {s['completion']:5.1f}%  penalty "
                f"{s['penalty']:.3f}  score {s['score']:5.2f}  "
                f"({s['n']}/30 runs, {s['completed']} full routes)")
        else:
            log(f"  {name:8s} NO RESULTS")
    dest = os.path.join(REPO, "runs/carla_smoke/dev10_results.json")
    json.dump({k: v for k, v in summary.items()}, open(dest, "w"), indent=2)
    for name in groups:
        for t in ("A", "B"):
            p = os.path.join(OUT, f"carla_results_d10_{name}_{t}.json")
            if os.path.exists(p):
                subprocess.run(["cp", p, os.path.join(
                    REPO, "runs/carla_smoke", f"d10_{name}_{t}.json")])
    git_commit([os.path.join(REPO, "runs/carla_smoke")],
               f"test-10 eval: {winner} vs v3")

    live = {k: v for k, v in summary.items() if v}
    if not live:
        log("test-10 empty — stopping before the 220 benchmark; investigate")
        return
    overall = max(live, key=lambda k: live[k]["completion"])
    oconf = wconf if overall == winner else V3_CONF
    log(f"OVERALL WINNER for the full 220-route benchmark: {overall}")

    ids = re.findall(r'<route id="(\d+)"',
                     open(os.path.join(B2D, "leaderboard/data/bench2drive220.xml")).read())
    log(f"benchmark: {len(ids)} routes, 1 rep, chunks of 12")
    jobs = []
    lanes = pick_lanes()
    for i in range(0, len(ids), 12):
        chunk = ",".join(ids[i:i + 12])
        tag = f"b220_{i // 12:02d}"
        jobs.append(submit_chain(lanes[(i // 12) % len(lanes)], tag,
                                 f"{tag}:{oconf}:{chunk}:1",
                                 routes_xml="bench2drive220.xml"))
    with open(os.path.join(OUT, "bench220_config.txt"), "w") as f:
        f.write(f"{overall}\n{oconf}\n{len(jobs)}\n")
    submit_next_stage("bench220", jobs)


def stage_bench220():
    log("== stage bench220: final benchmark aggregate ==")
    overall = open(os.path.join(OUT, "bench220_config.txt")).read().split()[0]
    recs = []
    missing = []
    for p in sorted(glob.glob(os.path.join(OUT, "carla_results_b220_*.json"))):
        try:
            recs += records(p)
        except Exception:
            missing.append(os.path.basename(p))
    s = agg(recs)
    if not s:
        log("no benchmark records — investigate chunk jobs")
        return
    log(f"FINAL full 220-route result for {overall}: driving score {s['score']:.2f}, "
        f"completion {s['completion']:.1f}%, penalty {s['penalty']:.3f} "
        f"({s['n']}/220 routes scored, {s['completed']} completed)")
    if s["n"] < 220:
        log(f"WARNING: only {s['n']}/220 routes present "
            f"(unreadable: {missing}) — resubmit the gaps before publishing")
    os.makedirs(os.path.join(REPO, "runs/bench220"), exist_ok=True)
    for p in glob.glob(os.path.join(OUT, "carla_results_b220_*.json")):
        subprocess.run(["cp", p, os.path.join(REPO, "runs/bench220/")])
    json.dump({"config": overall, **s},
              open(os.path.join(REPO, "runs/bench220/summary.json"), "w"),
              indent=2)
    git_commit([os.path.join(REPO, "runs/bench220")],
               f"full 220-route benchmark: {overall} score {s['score']:.2f} "
               f"completion {s['completion']:.1f}% ({s['n']} routes)")


if __name__ == "__main__":
    stage = sys.argv[1]
    try:
        {"confirm3x3": stage_confirm3x3,
         "dev10": stage_dev10,
         "bench220": stage_bench220}[stage]()
    except Exception:
        log(f"DECIDER CRASH in stage {stage}:\n{traceback.format_exc()}")
        raise
