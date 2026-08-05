#!/bin/bash
#SBATCH --job-name=v02-video20
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=03:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# 20 paired videos of the best v0.2 model (DITTO-AV shaped @999):
# for each of the 10 dev-10 routes, ONE CARLA run recorded two ways —
# 3D chase-cam (camera frames) + 2D BEV (state.jsonl of the same run).
# Outputs: outputs/videos20/route<R>_{3d,2d}.mp4 (+ state logs).

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
VID=${VID:-$OUT/videos20}
mkdir -p $VID
export XDG_CACHE_HOME=$BASE/cache
export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$BASE/carla_pythonapi:$REPO
export CARLA_ROOT=$BASE/carla_root
export SAVE_PATH=$VID   # unlocks the >3 m camera rig radius (B2D flag)

CONF=${CONF:-configs/diag_v02_999s.yaml}
ROUTES=${ROUTES:-"3514 3255 26405 25381 25378 25424 2091 27494 17569 28198"}

cd $B2D
for R in $ROUTES; do
  echo "===== ROUTE $R ====="
  FRAMES=/tmp/$SLURM_JOB_ID/frames_$R
  mkdir -p $FRAMES
  export DITTO_VIDEO_DIR=$FRAMES
  export DITTO_AGENT_LOG=$OUT/agent_ticks_v20_$R.jsonl
  PORT=$((20000 + RANDOM % 10000)); TMPORT=$((PORT + 50))
  rm -f $DITTO_AGENT_LOG
  $VENVPY $B2D/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes=$B2D/leaderboard/data/drivetransformer_bench2drive_dev10.xml \
      --routes-subset=$R --repetitions=1 --track=MAP \
      --agent=$REPO/scripts/carla_team_agent.py \
      --agent-config=$REPO/$CONF \
      --checkpoint=$OUT/video20_run_$R.json \
      --port=$PORT --traffic-manager-port=$TMPORT || \
      echo "route $R evaluator nonzero (continuing)"
  $VENVPY - "$FRAMES" "$VID/route${R}_3d.mp4" <<'PYEOF'
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
  if [ -s "$FRAMES/state.jsonl" ]; then
    cp $FRAMES/state.jsonl $VID/route${R}_state.jsonl
    $VENVPY $REPO/scripts/render_bev_video.py \
        $FRAMES/state.jsonl $VID/route${R}_2d.mp4
  else
    echo "route $R: NO state.jsonl (agent state dump failed?)"
  fi
  rm -rf $FRAMES
done
rm -rf /tmp/$SLURM_JOB_ID
ls -la $VID
echo "V02_VIDEO20_DONE"
