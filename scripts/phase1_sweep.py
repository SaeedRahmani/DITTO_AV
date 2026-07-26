"""Phase-1 paper sweep: generate run configs / aggregate results.

Experiment matrix (PAPER_PLAN.md, Phase 1):
  main:      3 seeds x full pipeline (bc, ditto_single, ditto_multi)
  ablations: K (retrieval modes), n_negatives, horizon  [reuse seed-0
             data + world model -> policies+eval only]
             data scale, expert style ratio             [full pipeline]

Usage:
  python scripts/phase1_sweep.py generate --sweep-dir $BASE/outputs/phase1
  python scripts/phase1_sweep.py aggregate --sweep-dir $BASE/outputs/phase1 \
      --out runs/phase1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

BASE_CFG = {
    "device": "cpu",
    "collect": {"n_expert_episodes": 300, "n_noisy_episodes": 100,
                "noise_eps": 0.3, "aggressive_prob": 0.5, "seed": 0},
    "wm": {"train_steps": 4000, "batch_size": 32, "seq_len": 16},
    "ac": {"train_steps": 3000, "horizon": 15, "batch_size": 64,
           "k_modes": 8},
    "bc": {"train_steps": 3000},
    "eval": {"n_episodes": 50},
}

SEEDS = [0, 1, 2]


def deep_update(d: dict, u: dict) -> dict:
    out = {k: (v.copy() if isinstance(v, dict) else v) for k, v in d.items()}
    for k, v in u.items():
        if isinstance(v, dict):
            out[k] = deep_update(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def runs_matrix():
    """(name, overrides, reuse_parent_or_None) for every sweep run."""
    runs = []
    for s in SEEDS:
        runs.append((f"main_seed{s}",
                     {"seed": s, "collect": {"seed": s}}, None))
    # AC-only ablations: identical data + world model as main_seed0
    for k in (1, 2, 4, 16):
        runs.append((f"k{k}", {"ac": {"k_modes": k}}, "main_seed0"))
    for n in (0, 4, 32):
        runs.append((f"neg{n}", {"ac": {"n_negatives": n}}, "main_seed0"))
    for h in (5, 10):
        runs.append((f"h{h}", {"ac": {"horizon": h}}, "main_seed0"))
    # collection-level ablations: full pipeline
    for n_ep in (75, 150):
        runs.append((f"data{n_ep}",
                     {"collect": {"n_expert_episodes": n_ep,
                                  "n_noisy_episodes": n_ep // 3}}, None))
    runs.append(("style25", {"collect": {"aggressive_prob": 0.25}}, None))
    return runs


def cmd_generate(args):
    sweep = Path(args.sweep_dir)
    cfg_dir = sweep / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for name, overrides, reuse in runs_matrix():
        cfg = deep_update(BASE_CFG, overrides)
        cfg["run_dir"] = str(sweep / name)
        path = cfg_dir / f"{name}.yaml"
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
        plan.append({"name": name, "config": str(path),
                     "reuse": str(sweep / reuse) if reuse else None})
    json.dump(plan, open(sweep / "plan.json", "w"), indent=2)
    print(f"{len(plan)} configs -> {cfg_dir}")
    for p in plan:
        dep = f"  (reuse {p['reuse']})" if p["reuse"] else ""
        print(f"  {p['name']}{dep}")


def _load(sweep: Path, name: str):
    p = sweep / name / "results" / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(vals):
    m, s = float(np.mean(vals)), float(np.std(vals))
    return f"{m:.2f} ± {s:.2f}"


def cmd_aggregate(args):
    sweep = Path(args.sweep_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 1 results (highway-env)", ""]
    raw = {}

    # ---- main table: mean ± std over seeds ----
    mains = {s: _load(sweep, f"main_seed{s}") for s in SEEDS}
    have = {s: r for s, r in mains.items() if r}
    if have:
        raw["main"] = {f"seed{s}": r for s, r in have.items()}
        lines += [f"## Main comparison ({len(have)} seeds)", ""]
        for cond in ("in_distribution", "shifted"):
            lines += [f"### {cond}", "",
                      "| policy | return | collision rate | mean speed |",
                      "|---|---|---|---|"]
            policies = list(next(iter(have.values()))[cond])
            for pol in policies:
                ret = [r[cond][pol]["return_mean"] for r in have.values()]
                col = [r[cond][pol]["collision_rate"] for r in have.values()]
                spd = [r[cond][pol]["mean_speed"] for r in have.values()]
                lines.append(f"| {pol} | {_fmt(ret)} | {_fmt(col)} "
                             f"| {_fmt(spd)} |")
            lines.append("")

    # ---- ablation tables (seed 0, ditto_multi unless noted) ----
    groups = {
        "K (retrieval modes, ditto_multi)":
            [("k1", "K=1"), ("k2", "K=2"), ("k4", "K=4"),
             ("main_seed0", "K=8 (main)"), ("k16", "K=16")],
        "Contrastive negatives (ditto_multi)":
            [("neg0", "M=0 (raw)"), ("neg4", "M=4"),
             ("main_seed0", "M=16 (main)"), ("neg32", "M=32")],
        "Imagination horizon (ditto_multi)":
            [("h5", "H=5"), ("h10", "H=10"), ("main_seed0", "H=15 (main)")],
        "Expert data scale (ditto_multi)":
            [("data75", "75 eps"), ("data150", "150 eps"),
             ("main_seed0", "300 eps (main)")],
        "Expert style ratio (ditto_multi)":
            [("style25", "25/75"), ("main_seed0", "50/50 (main)")],
    }
    for title, rows in groups.items():
        table = []
        for run, label in rows:
            r = _load(sweep, run)
            if r:
                raw.setdefault("ablations", {})[run] = r
                table.append((label, r))
        if not table:
            continue
        lines += [f"## {title}", "",
                  "| setting | return (ID) | collisions (ID) "
                  "| return (shift) | collisions (shift) |",
                  "|---|---|---|---|---|"]
        for label, r in table:
            i, s = r["in_distribution"]["ditto_multi"], \
                r["shifted"]["ditto_multi"]
            lines.append(
                f"| {label} | {i['return_mean']:.2f} "
                f"| {i['collision_rate']:.2f} | {s['return_mean']:.2f} "
                f"| {s['collision_rate']:.2f} |")
        lines.append("")

    (out / "phase1_results.md").write_text("\n".join(lines))
    (out / "phase1_results.json").write_text(json.dumps(raw, indent=2))
    print(f"wrote {out}/phase1_results.md and .json")
    missing = [p["name"] for p in
               json.loads((sweep / "plan.json").read_text())
               if not _load(sweep, p["name"])]
    if missing:
        print(f"still missing: {missing}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--sweep-dir", required=True)
    g.set_defaults(fn=cmd_generate)
    a = sub.add_parser("aggregate")
    a.add_argument("--sweep-dir", required=True)
    a.add_argument("--out", default="runs/phase1")
    a.set_defaults(fn=cmd_aggregate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
