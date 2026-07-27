#!/bin/bash
#SBATCH --job-name=carla-smoke
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Closed-loop smoke test: CARLA 0.9.15 server (apptainer, offscreen) +
# Bench2Drive leaderboard, one base-town route (dev10 subset 0, Town03),
# driven by DittoCarlaAgent. ~2 GPU-h worst case.

set -euo pipefail
# GPU nodes have NO python modules and do not mount the spack tree the
# main venv's python links against (verified, diag job 10522194: the venv
# python symlink resolves to nothing there and PATH falls through to a
# bare miniconda 3.9). Use the self-contained conda env on scratch: own
# python + libs + CPU torch (the policy is tiny; the GPU runs CARLA).
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$VENVPY -c "import carla, torch" || { echo "carla_eval env broken"; exit 1; }
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

BASE=/scratch/$USER/ditto_av
B2D=$BASE/Bench2Drive
SIF=$BASE/carla_0915.sif
[ -s "$SIF" ] || { echo "CARLA SIF missing/empty: $SIF"; exit 1; }
PORT=$((20000 + RANDOM % 10000))
TMPORT=$((PORT + 50))

export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$BASE/carla_pythonapi:$BASE/DITTO_AV
# the evaluator launches $CARLA_ROOT/CarlaUE4.sh itself (and restarts it
# on failure); point it at the apptainer shim
export CARLA_ROOT=$BASE/carla_root

export DITTO_AGENT_LOG=/scratch/$USER/ditto_av/outputs/agent_ticks.jsonl
rm -f $DITTO_AGENT_LOG
# the evaluator opens 'leaderboard/data/weather.xml' relative to cwd
cd $B2D
$VENVPY $B2D/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$B2D/leaderboard/data/drivetransformer_bench2drive_dev10.xml \
    --routes-subset=25378 \
    --repetitions=1 \
    --track=MAP \
    --agent=$BASE/DITTO_AV/scripts/carla_team_agent.py \
    --agent-config=$BASE/DITTO_AV/configs/carla_agent.yaml \
    --checkpoint=$BASE/outputs/carla_smoke_results.json \
    --port=$PORT --traffic-manager-port=$TMPORT \
    --timeout=120
echo "SMOKE EVAL DONE"
$VENVPY - <<'EOF'
import json
d = json.load(open("/scratch/srahmani/ditto_av/outputs/carla_smoke_results.json"))
rec = d.get("_checkpoint", {}).get("records", [])
for r in rec:
    print(r.get("route_id"), r.get("status"),
          "driving score:", r.get("scores", {}).get("score_composed"))
EOF
