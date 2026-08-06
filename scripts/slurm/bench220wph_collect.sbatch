#!/bin/bash
#SBATCH --job-name=b220wph-collect
#SBATCH --partition=visual
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Aggregate gen3_wph full 220-route chunks (tags wph220_*) into
# runs/bench220_gen3wph. Fresh prefix — earlier bench220 files can
# never contaminate the aggregate.
set -euo pipefail
PY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$PY - <<'PYEOF'
import glob, json, subprocess, os
recs, bad = [], []
for f in sorted(glob.glob("/scratch/srahmani/ditto_av/outputs/carla_results_wph220_*.json")):
    try: recs += json.load(open(f))["_checkpoint"]["records"]
    except Exception: bad.append(os.path.basename(f))
n = len(recs)
s = dict(config="gen3_wph_bc_rec", n=n,
         completion=sum(r["scores"]["score_route"] for r in recs)/n,
         penalty=sum(r["scores"]["score_penalty"] for r in recs)/n,
         score=sum(r["scores"]["score_composed"] for r in recs)/n,
         completed=sum(1 for r in recs if r["status"]=="Completed"))
print(f"GEN3_WPH 220-ROUTE FINAL: score {s['score']:.2f} completion {s['completion']:.1f}% "
      f"penalty {s['penalty']:.3f} success {100*s['completed']/n:.1f}% ({n}/220; unreadable {bad})")
repo = "/scratch/srahmani/ditto_av/DITTO_AV"
os.makedirs(repo + "/runs/bench220_gen3wph", exist_ok=True)
json.dump(s, open(repo + "/runs/bench220_gen3wph/summary.json", "w"), indent=2)
for f in glob.glob("/scratch/srahmani/ditto_av/outputs/carla_results_wph220_*.json"):
    subprocess.run(["cp", f, repo + "/runs/bench220_gen3wph/"])
subprocess.run(["git", "-C", repo, "add", "runs/bench220_gen3wph"])
subprocess.run(["git", "-C", repo, "commit", "-q", "-m",
                f"GEN-3 wph FINAL full 220-route benchmark: gen3_wph_bc_rec score {s['score']:.2f} "
                f"completion {s['completion']:.1f}% success {100*s['completed']/n:.1f}% ({n} routes)"])
PYEOF
echo "[$(date '+%Y-%m-%d %H:%M')] bench220_gen3wph collector done" >> /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md
echo B220_WPH_COLLECTED
