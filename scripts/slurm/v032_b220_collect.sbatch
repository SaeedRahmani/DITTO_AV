#!/bin/bash
#SBATCH --job-name=v032-b220-collect
#SBATCH --partition=visual
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Aggregate v0.3.2 220-route chunks (tags v032b220_<LABEL>_*) into
# outputs/v032/bench220_<LABEL>.json + a PIPELINE_STATUS line.
set -euo pipefail
: "${LABEL:?set LABEL}"
PY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
LABEL=$LABEL $PY - <<'PYEOF'
import glob, json, os
label = os.environ["LABEL"]
recs, bad = [], []
for f in sorted(glob.glob(
        f"/scratch/srahmani/ditto_av/outputs/carla_results_v032b220_{label}_*.json")):
    try:
        recs += json.load(open(f))["_checkpoint"]["records"]
    except Exception:
        bad.append(os.path.basename(f))
n = len(recs)
ds = sum(r["scores"]["score_composed"] for r in recs) / max(n, 1)
rc = sum(r["scores"]["score_route"] for r in recs) / max(n, 1)
comp = sum(1 for r in recs if r["scores"]["score_route"] > 99.9)
tot = {}
for r in recs:
    for k, v in r["infractions"].items():
        if isinstance(v, list) and v:
            tot[k] = tot.get(k, 0) + len(v)
out = {"label": label, "n_runs": n, "ds_mean": ds, "rc_mean": rc,
       "full_completion": comp, "infractions": tot, "bad_files": bad}
os.makedirs("/scratch/srahmani/ditto_av/outputs/v032", exist_ok=True)
path = f"/scratch/srahmani/ditto_av/outputs/v032/bench220_{label}.json"
json.dump(out, open(path, "w"), indent=1)
print(json.dumps(out, indent=1))
from datetime import datetime
line = (f"[{datetime.now():%Y-%m-%d %H:%M}] V03.2 b220 {label}: "
        f"DS {ds:.2f} RC {rc:.1f} runs {n} full {comp} "
        f"veh {tot.get('collisions_vehicle',0)} "
        f"lay {tot.get('collisions_layout',0)} (refs @220: v0.2 "
        f"champions 75.88/76.10)\n")
open("/scratch/srahmani/ditto_av/outputs/PIPELINE_STATUS.md", "a").write(line)
PYEOF
echo "B220_COLLECT_DONE"
