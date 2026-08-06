#!/bin/bash
#SBATCH --job-name=vid-finalize
#SBATCH --partition=participation
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Re-renders every 2D clip from its state log with the CURRENT renderer,
# so folders filmed at different times (or by a requeued half) share one
# style. Chain it behind the recording jobs:
#   sbatch --dependency=afterany:<job>:<job> scripts/slurm/videos_finalize.sbatch
# Label comes from each folder's MODEL.md; the 3D clips are untouched.
set -eu
PY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
R=/scratch/$USER/ditto_av/DITTO_AV/scripts/render_bev_video.py
V=/scratch/$USER/ditto_av/outputs/videos

for d in $V/v0.*/; do
  label=$(sed -n 's/^- label: //p' $d/MODEL.md 2>/dev/null | head -1)
  [ -n "$label" ] || label=$(basename $d)
  for s in $d/route*_state.jsonl; do
    [ -e "$s" ] || continue
    $PY $R "$s" "${s%_state.jsonl}_2d.mp4" --label "$label" | tail -1
  done
done
echo "VIDEOS_FINALIZE_DONE"
