#!/bin/bash
#SBATCH --job-name=carla-smoke-h
#SBATCH --partition=gpu-v100
#SBATCH --time=01:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# carla_smoke.sbatch, freeze-safe variant: same closed-loop eval, but
# every write goes to $HOME (slurm log, per-tick agent log, results
# json, caches) and the agent code comes from the HOME clone
# (~/ditto_work/DITTO_AV) — the scratch clone is read-only under the
# quota freeze and predates the theta fix (2783e7e). Scratch is used
# strictly read-only: conda env, SIF, shim, Bench2Drive harness,
# trained checkpoints.
# Env knobs:
#   ROUTES_SUBSET  route ids (default 25378)
#   REPS           repetitions per route (default 1)
#   AGENT_CONF     agent-config yaml (default configs/carla_agent.yaml)
# Example (post-theta-fix diagnostic pair):
#   ROUTES_SUBSET=25381,25378,27494 REPS=3 \
#     sbatch scripts/slurm/carla_smoke_home.sbatch
#   ROUTES_SUBSET=25381,25378,27494 REPS=3 \
#     AGENT_CONF=configs/carla_agent_norec.yaml \
#     sbatch --job-name=carla-smoke-norec scripts/slurm/carla_smoke_home.sbatch

set -euo pipefail
# GPU nodes: no python modules, no spack tree — self-contained conda env
# (see carla_smoke.sbatch header for the diagnosis)
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$VENVPY -c "import carla, torch" || { echo "carla_eval env broken"; exit 1; }
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

BASE=/scratch/$USER/ditto_av
REPO=$HOME/ditto_work/DITTO_AV
B2D=$BASE/Bench2Drive
SIF=$BASE/carla_0915.sif
[ -s "$SIF" ] || { echo "CARLA SIF missing/empty: $SIF"; exit 1; }
OUT=$HOME/ditto_out
mkdir -p $OUT
export XDG_CACHE_HOME=$OUT/cache
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

export DITTO_AGENT_LOG=$OUT/agent_ticks_$SLURM_JOB_ID.jsonl
RESULTS=$OUT/carla_results_$SLURM_JOB_ID.json
echo "agent-config: ${AGENT_CONF:-configs/carla_agent.yaml}" \
     "routes: ${ROUTES_SUBSET:-25378} reps: ${REPS:-1}"

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
echo "SMOKE EVAL DONE"
$VENVPY - "$RESULTS" <<'EOF'
import json
import sys

d = json.load(open(sys.argv[1]))
for r in d.get("_checkpoint", {}).get("records", []):
    s = r.get("scores", {})
    print(r.get("route_id"), r.get("status"),
          "score:", s.get("score_composed"),
          "completion:", s.get("score_route"),
          "penalty:", s.get("score_penalty"))
EOF
