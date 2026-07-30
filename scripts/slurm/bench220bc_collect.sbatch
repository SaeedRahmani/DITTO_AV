#!/bin/bash
#SBATCH --job-name=b220bc-collect
#SBATCH --partition=visual
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Aggregate gen-2 220-route chunks (tags g3bc_*) — fresh prefix so the
# committed gen-1 b220_* files can never contaminate the aggregate.
set -euo pipefail
PY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
REPO=/scratch/$USER/ditto_av/DITTO_AV
$PY - <<'PYEOF'
import glob, json, subprocess, os
recs, bad = [], []
for f in sorted(glob.glob("/scratch/srahmani/ditto_av/outputs/carla_results_g3bc_*.json")):
    try: recs += json.load(open(f))["_checkpoint"]["records"]
    except Exception: bad.append(os.path.basename(f))
n = len(recs)
s = dict(config="gen2_10x_bc", n=n,
         completion=sum(r["scores"]["score_route"] for r in recs)/n,
         penalty=sum(r["scores"]["score_penalty"] for r in recs)/n,
         score=sum(r["scores"]["score_composed"] for r in recs)/n,
         completed=sum(1 for r in recs if r["status"]=="Completed"))
print(f"GEN2 220-ROUTE FINAL: score {s['score']:.2f} completion {s['completion']:.1f}% "
      f"penalty {s['penalty']:.3f} success {100*s['completed']/n:.1f}% ({n}/220; unreadable {bad})")
repo = "/scratch/srahmani/ditto_av/DITTO_AV"
os.makedirs(repo + "/runs/bench220_gen2bc", exist_ok=True)
json.dump(s, open(repo + "/runs/bench220_gen2bc/summary.json", "w"), indent=2)
for f in glob.glob("/scratch/srahmani/ditto_av/outputs/carla_results_g3bc_*.json"):
    subprocess.run(["cp", f, repo + "/runs/bench220_gen2bc/"])
subprocess.run(["git", "-C", repo, "add", "runs/bench220_gen2bc"])
subprocess.run(["git", "-C", repo, "commit", "-q", "-m",
                f"GEN-2 FINAL 220-route benchmark: gen2_10x_bc score {s['score']:.2f} "
                f"completion {s['completion']:.1f}% success {100*s['completed']/n:.1f}% ({n} routes)"])
PYEOF
echo "[$(date '+%Y-%m-%d %H:%M')] bench220_gen2bc collector done" >> /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md
echo B220_GEN2_COLLECTED
