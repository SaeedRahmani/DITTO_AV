#!/bin/bash
#SBATCH --job-name=carla-eval
#SBATCH --partition=gpu-a100
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Generalized closed-loop eval (scratch-primary successor of
# carla_smoke.sbatch / carla_smoke_home.sbatch): CARLA 0.9.15 server
# (apptainer shim, offscreen) + Bench2Drive leaderboard driven by
# DittoCarlaAgent. Routes on Town11-15 need the AdditionalMaps overlay
# wired into the shim (carla_root/CarlaUE4.sh; mirror in
# scripts/patches/).
# Env knobs:
#   ROUTES_SUBSET  route ids (default 25378)
#   REPS           repetitions per route (default 1)
#   AGENT_CONF     agent-config yaml, repo-relative (default configs/carla_agent.yaml)
#   TAG            results file suffix (default $SLURM_JOB_ID)
# Example (3x3 confirmation):
#   ROUTES_SUBSET=25381,25378,27494 REPS=3 AGENT_CONF=configs/diag_kl015.yaml \
#     TAG=kl015_3x3 sbatch --time=00:59:00 --job-name=kl015x3 \
#     scripts/slurm/carla_eval.sbatch

set -euo pipefail
# GPU nodes: no python modules, no spack tree — self-contained conda env
# (see carla_smoke.sbatch header for the diagnosis)
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$VENVPY -c "import carla, torch" || { echo "carla_eval env broken"; exit 1; }
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

BASE=/scratch/$USER/ditto_av
REPO=$BASE/DITTO_AV
B2D=$BASE/Bench2Drive
SIF=$BASE/carla_0915.sif
[ -s "$SIF" ] || { echo "CARLA SIF missing/empty: $SIF"; exit 1; }
OUT=$BASE/outputs
export XDG_CACHE_HOME=$BASE/cache
PORT=$((20000 + RANDOM % 10000))
TMPORT=$((PORT + 50))

export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$BASE/carla_pythonapi:$REPO
# Bench2Drive's route_scenario.get_all_scenario_classes() globs
# ${SCENARIO_RUNNER_ROOT:-./}/srunner/scenarios/*.py — unset, it
# resolves to $B2D/srunner (nonexistent), the class registry comes
# up empty and EVERY scenario is skipped with a KeyError. Runs then
# score background traffic only. Their run_evaluation.sh exports it.
export SCENARIO_RUNNER_ROOT=$B2D/scenario_runner
# the evaluator launches $CARLA_ROOT/CarlaUE4.sh itself; apptainer shim
export CARLA_ROOT=$BASE/carla_root

TAG=${TAG:-$SLURM_JOB_ID}
export DITTO_AGENT_LOG=$OUT/agent_ticks_$TAG.jsonl
RESULTS=$OUT/carla_results_$TAG.json
rm -f $DITTO_AGENT_LOG $RESULTS
echo "agent-config: ${AGENT_CONF:-configs/carla_agent.yaml}" \
     "routes: ${ROUTES_SUBSET:-25378} reps: ${REPS:-1} tag: $TAG"

# the evaluator opens 'leaderboard/data/weather.xml' relative to cwd
cd $B2D
$VENVPY $B2D/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$B2D/leaderboard/data/drivetransformer_bench2drive_dev10.xml \
    --routes-subset=${ROUTES_SUBSET:-25378} \
    --repetitions=${REPS:-1} \
    --track=MAP \
    --agent=$REPO/scripts/carla_team_agent.py \
    --agent-config=$REPO/${AGENT_CONF:-configs/carla_agent.yaml} \
    --checkpoint=$RESULTS \
    --port=$PORT --traffic-manager-port=$TMPORT \
    --timeout=120
echo "EVAL DONE tag=$TAG"
$VENVPY - "$RESULTS" <<'EOF'
import json
import sys

d = json.load(open(sys.argv[1]))
ck = d.get("_checkpoint", {})
for r in ck.get("records", []):
    s = r.get("scores", {})
    print(r.get("route_id"), r.get("status"),
          "score:", s.get("score_composed"),
          "completion:", s.get("score_route"),
          "penalty:", s.get("score_penalty"))
g = ck.get("global_record", {})
if g:
    sm = g.get("scores_mean", {})
    print("MEAN score:", sm.get("score_composed"),
          "completion:", sm.get("score_route"),
          "penalty:", sm.get("score_penalty"))
EOF
