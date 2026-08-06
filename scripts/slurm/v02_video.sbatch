#!/bin/bash
#SBATCH --job-name=v02-video
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# 3D chase-cam videos of the leading v0.2 models (presentation only —
# the policy drives from the vector state). One route per model; frames
# saved by the agent (DITTO_VIDEO_DIR), encoded to mp4 in-job (cv2),
# loose frames deleted (inode rules). Outputs: outputs/videos/*.mp4

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$VENVPY -c "import carla, torch, cv2" || { echo "env broken"; exit 1; }
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

BASE=/scratch/$USER/ditto_av
REPO=$BASE/DITTO_AV
B2D=$BASE/Bench2Drive
SIF=$BASE/carla_0915.sif
[ -s "$SIF" ] || { echo "CARLA SIF missing"; exit 1; }
OUT=$BASE/outputs
VID=$OUT/videos
mkdir -p $VID
export XDG_CACHE_HOME=$BASE/cache
export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$BASE/carla_pythonapi:$REPO
export CARLA_ROOT=$BASE/carla_root
export SAVE_PATH=$VID   # unlocks the 100 m sensor rig radius (B2D flag)

ROUTE=${ROUTE:-25378}
VARIANTS=${VARIANTS:-"v02bc297:configs/diag_v02_clp_bc.yaml \
v02shaped297:configs/diag_v02_shaped.yaml \
v02rl999t:configs/diag_v02_999t_rl.yaml"}

cd $B2D
for spec in $VARIANTS; do
  IFS=: read -r TAG CONF <<< "$spec"
  echo "===== VIDEO $TAG conf=$CONF route=$ROUTE ====="
  FRAMES=/tmp/$SLURM_JOB_ID/frames_$TAG
  mkdir -p $FRAMES
  export DITTO_VIDEO_DIR=$FRAMES
  export DITTO_AGENT_LOG=$OUT/agent_ticks_video_$TAG.jsonl
  PORT=$((20000 + RANDOM % 10000)); TMPORT=$((PORT + 50))
  rm -f $DITTO_AGENT_LOG
  $VENVPY $B2D/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes=$B2D/leaderboard/data/drivetransformer_bench2drive_dev10.xml \
      --routes-subset=$ROUTE --repetitions=1 --track=MAP \
      --agent=$REPO/scripts/carla_team_agent.py \
      --agent-config=$REPO/$CONF \
      --checkpoint=$OUT/video_run_$TAG.json \
      --port=$PORT --traffic-manager-port=$TMPORT || \
      echo "variant $TAG evaluator exited nonzero (continuing)"
  $VENVPY - "$FRAMES" "$VID/${TAG}_route${ROUTE}.mp4" <<'PYEOF'
import glob, sys
import cv2
frames = sorted(glob.glob(sys.argv[1] + "/*.jpg"))
print(f"encoding {len(frames)} frames -> {sys.argv[2]}")
if frames:
    h, w = cv2.imread(frames[0]).shape[:2]
    vw = cv2.VideoWriter(sys.argv[2],
                         cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for f in frames:
        vw.write(cv2.imread(f))
    vw.release()
PYEOF
  rm -rf $FRAMES
done
rm -rf /tmp/$SLURM_JOB_ID
ls -la $VID
echo "V02_VIDEO_DONE"
