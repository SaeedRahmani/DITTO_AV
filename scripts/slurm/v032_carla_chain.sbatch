#!/bin/bash
#SBATCH --job-name=v032-carla
#SBATCH --partition=visual
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.3.2 CARLA eval chain — carla_eval_chain.sbatch running the
# WORKTREE code (REPO=DITTO_AV_v032, agent confs from its configs/).
# Default lane: visual (unmanaged Quadro, NO gpu flag — lane facts in
# carla_eval_chain.sbatch header). Pass --gpus-per-task=1 +
# --partition=participation at submit time for the H100 lane.
# VARIANTS: "<tag>:<agent_conf>:<routes>:<reps>" space-separated
# (dev-10 A/B protocol: A=3255,3514,25378,25381,26405 x3,
#  B=2091,17569,25424,27494,28198 x3).
# SUBMIT AS:  VARIANTS='...' sbatch scripts/slurm/v032_carla_chain.sbatch
# NEVER via sbatch --export=ALL,VARIANTS=... — sbatch splits --export
# on commas, truncating the route list (burned 2026-08-05, jobs
# 10582601/02).

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$VENVPY -c "import carla, torch" || { echo "carla_eval env broken"; exit 1; }
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

BASE=/scratch/$USER/ditto_av
REPO=$BASE/DITTO_AV_v032
B2D=$BASE/Bench2Drive
SIF=$BASE/carla_0915.sif
[ -s "$SIF" ] || { echo "CARLA SIF missing/empty: $SIF"; exit 1; }
OUT=$BASE/outputs
export XDG_CACHE_HOME=$BASE/cache
export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$BASE/carla_pythonapi:$REPO
export CARLA_ROOT=$BASE/carla_root

: "${VARIANTS:?set VARIANTS at submit time}"

cd $B2D
for spec in $VARIANTS; do
    IFS=: read -r TAG CONF ROUTES REPS <<< "$spec"
    [ -n "${REPS:-}" ] || { echo "bad VARIANTS spec: '$spec'"; exit 1; }
    echo "===== VARIANT $TAG conf=$CONF routes=$ROUTES reps=$REPS ====="
    PORT=$((20000 + RANDOM % 10000)); TMPORT=$((PORT + 50))
    export DITTO_AGENT_LOG=$OUT/agent_ticks_$TAG.jsonl
    RESULTS=$OUT/carla_results_$TAG.json
    rm -f $DITTO_AGENT_LOG $RESULTS
    $VENVPY $B2D/leaderboard/leaderboard/leaderboard_evaluator.py \
        --routes=$B2D/leaderboard/data/${ROUTES_XML:-drivetransformer_bench2drive_dev10.xml} \
        --routes-subset=$ROUTES \
        --repetitions=$REPS \
        --track=MAP \
        --agent=$REPO/scripts/carla_team_agent.py \
        --agent-config=$REPO/$CONF \
        --checkpoint=$RESULTS \
        --port=$PORT --traffic-manager-port=$TMPORT \
        --timeout=120 || echo "VARIANT $TAG FAILED (continuing)"
    echo "===== VARIANT $TAG DONE ====="
done
echo "CHAIN_DONE"
$VENVPY - <<'EOF'
import glob
import json

for f in sorted(glob.glob(
        "/scratch/srahmani/ditto_av/outputs/carla_results_v032_*.json")):
    try:
        g = json.load(open(f))["_checkpoint"]["global_record"]
        sm = g["scores_mean"]
        print(f.split("/")[-1], "score:", round(sm["score_composed"], 2),
              "completion:", round(sm["score_route"], 1),
              "penalty:", round(sm["score_penalty"], 3))
    except Exception as e:
        print(f.split("/")[-1], "unreadable:", e)
EOF
