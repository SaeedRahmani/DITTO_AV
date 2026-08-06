#!/bin/bash
#SBATCH --job-name=vid-finalize
#SBATCH --partition=participation
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Runs after the five per-branch video jobs (afterany): re-renders every
# 2D clip from its state log with the current renderer so all folders
# share one style, and appends the run scores to each MODEL.md.
set -eu
PY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
R=/scratch/$USER/ditto_av/DITTO_AV/scripts/render_bev_video.py
V=/scratch/$USER/ditto_av/outputs/videos
OUT=/scratch/$USER/ditto_av/outputs

for d in $V/v0.*/; do
  tag=$(basename $d)
  label=$(sed -n 's/^- label: //p' $d/MODEL.md 2>/dev/null | head -1)
  [ -n "$label" ] || label=$tag
  for s in $d/route*_state.jsonl; do
    [ -e "$s" ] || continue
    $PY $R "$s" "${s%_state.jsonl}_2d.mp4" --label "$label" | tail -1
  done
  $PY - "$tag" "$d" <<'PYEOF' >> $d/MODEL.md
import glob
import json
import sys

tag, d = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f"/scratch/srahmani/ditto_av/outputs/vid_{tag}_*.json")):
    r = f.split("_")[-1].split(".")[0]
    try:
        g = json.load(open(f))["_checkpoint"]["global_record"]["scores_mean"]
        rows.append((r, g["score_composed"], g["score_route"]))
    except Exception:
        rows.append((r, None, None))
print("\n## run scores (1 rep each)\n")
print("| route | driving score | completion |")
print("|---|---|---|")
for r, ds, rc in rows:
    print(f"| {r} | {ds if ds is None else round(ds, 2)} | "
          f"{rc if rc is None else round(rc, 1)} |")
PYEOF
done
echo "VIDEOS_FINALIZE_DONE"
