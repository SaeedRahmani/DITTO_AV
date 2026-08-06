#!/bin/bash
#SBATCH --job-name=vid-best
#SBATCH --partition=participation
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Paired 2D/3D rollout videos for ONE model, generic over branches:
# each route is ONE CARLA run recorded two ways (frame-synchronized) —
# route<R>_3d.mp4 (chase camera) and route<R>_2d.mp4 (BEV drawn over
# the town's OpenDRIVE map from the run's state log, kept alongside as
# route<R>_state.jsonl for restyled re-renders).
#
# Lanes (carla_eval_chain.sbatch header): participation needs
# --gpus-per-task=1, visual takes no gpu flag. Both cap at 4 h, so a
# half of dev-10 (5 routes) per job.
#
# Submit as (env style, never --export with commas):
#   REPO=/scratch/$USER/ditto_av/DITTO_AV_v02 \
#   CONF=configs/diag_v02_999s.yaml \
#   VID=/scratch/$USER/ditto_av/outputs/videos/v0.2_999s_shaped \
#   LABEL="v0.2 DITTO-AV shaped (999s)" \
#   ROUTES="3514 3255 26405 25381 25378" \
#   sbatch --gpus-per-task=1 scripts/slurm/videos_best.sbatch

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$VENVPY -c "import carla, torch, cv2" || { echo "env broken"; exit 1; }
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

BASE=/scratch/$USER/ditto_av
REPO=${REPO:-$BASE/DITTO_AV}
B2D=$BASE/Bench2Drive
SIF=$BASE/carla_0915.sif
[ -s "$SIF" ] || { echo "CARLA SIF missing"; exit 1; }
OUT=$BASE/outputs
: "${CONF:?set CONF (agent config, repo-relative)}"
: "${VID:?set VID (output folder)}"
LABEL=${LABEL:-}
TAG=$(basename "$VID")
ROUTES=${ROUTES:-"3514 3255 26405 25381 25378 25424 2091 27494 17569 28198"}
mkdir -p "$VID"
export XDG_CACHE_HOME=$BASE/cache
export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$BASE/carla_pythonapi:$REPO
export CARLA_ROOT=$BASE/carla_root
export SAVE_PATH=$VID   # unlocks the >3 m camera rig radius (B2D flag)

cd $B2D
for R in $ROUTES; do
  echo "===== ROUTE $R ($TAG) ====="
  FRAMES=/tmp/$SLURM_JOB_ID/frames_$R
  rm -rf $FRAMES; mkdir -p $FRAMES
  export DITTO_VIDEO_DIR=$FRAMES
  export DITTO_AGENT_LOG=$OUT/agent_ticks_vid_${TAG}_$R.jsonl
  PORT=$((20000 + RANDOM % 10000)); TMPORT=$((PORT + 50))
  rm -f $DITTO_AGENT_LOG
  $VENVPY $B2D/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes=$B2D/leaderboard/data/${ROUTES_XML:-drivetransformer_bench2drive_dev10.xml} \
      --routes-subset=$R --repetitions=1 --track=MAP \
      --agent=$REPO/scripts/carla_team_agent.py \
      --agent-config=$REPO/$CONF \
      --checkpoint=$OUT/vid_${TAG}_$R.json \
      --port=$PORT --traffic-manager-port=$TMPORT || \
      echo "route $R evaluator nonzero (continuing)"
  $VENVPY - "$FRAMES" "$VID/route${R}_3d.mp4" "$LABEL" <<'PYEOF'
import glob
import sys

import cv2
frames = sorted(glob.glob(sys.argv[1] + "/*.jpg"))
label = sys.argv[3] if len(sys.argv) > 3 else ""
print(f"encoding {len(frames)} frames -> {sys.argv[2]}")
if frames:
    h, w = cv2.imread(frames[0]).shape[:2]
    vw = cv2.VideoWriter(sys.argv[2],
                         cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if label:
            cv2.putText(img, label, (14, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2,
                        cv2.LINE_AA)
        vw.write(img)
    vw.release()
PYEOF
  if [ -s "$FRAMES/state.jsonl" ]; then
    cp $FRAMES/state.jsonl $VID/route${R}_state.jsonl
    $VENVPY $REPO/scripts/render_bev_video.py \
        $FRAMES/state.jsonl $VID/route${R}_2d.mp4 --label "$LABEL"
  else
    echo "route $R: NO state.jsonl (agent state dump failed?)"
  fi
  rm -rf $FRAMES
done
rm -rf /tmp/$SLURM_JOB_ID

# folder provenance — which model these clips actually show
{
  echo "# $TAG"
  echo
  echo "- label: $LABEL"
  echo "- repo: $REPO ($(git -C $REPO rev-parse --short HEAD 2>/dev/null))"
  echo "- agent config: $CONF"
  echo "- routes (job $SLURM_JOB_ID): $ROUTES"
  echo "- rendered: $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "route<R>_3d.mp4 chase camera, route<R>_2d.mp4 BEV over the town"
  echo "map, route<R>_state.jsonl the per-tick log both were made from."
} >> $VID/MODEL.md
ls -la $VID
echo "VIDEOS_BEST_DONE"
