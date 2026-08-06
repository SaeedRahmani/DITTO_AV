#!/usr/bin/env python3
"""Aggregate v0.2 full 220-route chunk results and commit them.

Usage: collect_bench220.py <tag_prefix> <run_subdir>
e.g.   collect_bench220.py v02b220_v02_999t_rl bench220_v02_999t_rl
"""
import glob
import json
import os
import subprocess
import sys

OUT = f"/scratch/{os.environ.get('USER', 'srahmani')}/ditto_av/outputs"
REPO = f"/scratch/{os.environ.get('USER', 'srahmani')}/ditto_av/DITTO_AV"


def main():
    prefix, subdir = sys.argv[1], sys.argv[2]
    recs, bad = [], []
    files = sorted(glob.glob(f"{OUT}/carla_results_{prefix}_*.json"))
    for f in files:
        try:
            recs += json.load(open(f))["_checkpoint"]["records"]
        except Exception:
            bad.append(os.path.basename(f))
    n = len(recs)
    if not n:
        print(f"no records for {prefix} — investigate chunk jobs")
        return 1
    s = {
        "config": prefix, "n": n,
        "completion": sum(r["scores"]["score_route"] for r in recs) / n,
        "penalty": sum(r["scores"]["score_penalty"] for r in recs) / n,
        "score": sum(r["scores"]["score_composed"] for r in recs) / n,
        "success": 100.0 * sum(1 for r in recs
                               if r["status"] == "Completed") / n,
        "unreadable": bad,
    }
    print(f"V02 220-ROUTE {prefix}: DS {s['score']:.2f} "
          f"completion {s['completion']:.1f}% penalty {s['penalty']:.3f} "
          f"success {s['success']:.1f}% ({n}/220; unreadable {bad})")
    d = os.path.join(REPO, "runs", subdir)
    os.makedirs(d, exist_ok=True)
    json.dump(s, open(os.path.join(d, "summary.json"), "w"), indent=2)
    for f in files:
        subprocess.run(["cp", f, d + "/"])
    subprocess.run(["git", "-C", REPO, "add", f"runs/{subdir}"])
    subprocess.run(["git", "-C", REPO, "commit", "-q", "-m",
                    f"v0.2 full 220-route benchmark {prefix}: DS {s['score']:.2f}"
                    f" completion {s['completion']:.1f}% success "
                    f"{s['success']:.1f}% ({n}/220)"])
    if n < 220:
        print(f"WARNING: only {n}/220 routes present — resubmit gaps "
              "before publishing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
