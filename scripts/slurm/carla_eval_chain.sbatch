#!/bin/bash
#SBATCH --job-name=eval-chain
#SBATCH --partition=participation
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Chained closed-loop evals in ONE job. Lanes verified 2026-07-29 by
# carla_boot_probe.sbatch (CARLA_BOOT_OK on both, Town03+Town12):
#   participation  H100 NVL, GRES-managed -> add --gpus-per-task=1
#   visual         Quadro RTX 4000, unmanaged GPU -> NO gpu flag
# Both 4 h cap, instant start while gpu-a100/gpu-v100 pend for hours
# (saturated by hidden long jobs; innovation account rejects full GPUs
# with AssocMaxGRESPerJob; MIG a100-small has no graphics; -nullrhi
# does not boot). No gpu directive here so the visual lane works;
# pass --gpus-per-task=1 for GRES-managed partitions.
# VARIANTS: "<tag>:<agent_conf>:<routes>:<reps>" space-separated.
# Default: the four 3x3 plateau/grid confirmations + Town12 overlay
# smoke. Results: outputs/carla_results_<tag>.json (same names the
# research-ceg-tp twins would write — cancel one lane when the other
# starts).

set -euo pipefail
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
export PYTHONPATH=$B2D/leaderboard:$B2D/scenario_runner:$BASE/carla_pythonapi:$REPO
export CARLA_ROOT=$BASE/carla_root

VARIANTS=${VARIANTS:-"kl015_3x3:configs/diag_kl015.yaml:25381,25378,27494:3 \
kl02_3x3:configs/diag_kl02.yaml:25381,25378,27494:3 \
v5kl01_3x3:configs/diag_v5kl01.yaml:25381,25378,27494:3 \
kl01k16_3x3:configs/diag_kl01k16.yaml:25381,25378,27494:3 \
t12_smoke:configs/diag_kl01.yaml:2091:1"}

cd $B2D
for spec in $VARIANTS; do
    IFS=: read -r TAG CONF ROUTES REPS <<< "$spec"
    echo "===== VARIANT $TAG conf=$CONF routes=$ROUTES reps=$REPS ====="
    PORT=$((20000 + RANDOM % 10000)); TMPORT=$((PORT + 50))
    export DITTO_AGENT_LOG=$OUT/agent_ticks_$TAG.jsonl
    RESULTS=$OUT/carla_results_$TAG.json
    rm -f $DITTO_AGENT_LOG $RESULTS
    # one variant failing must not kill the chain
    $VENVPY $B2D/leaderboard/leaderboard/leaderboard_evaluator.py \
        --routes=$B2D/leaderboard/data/drivetransformer_bench2drive_dev10.xml \
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

for f in sorted(glob.glob("/scratch/srahmani/ditto_av/outputs/carla_results_*_3x3.json")
                + glob.glob("/scratch/srahmani/ditto_av/outputs/carla_results_t12_smoke.json")):
    try:
        g = json.load(open(f))["_checkpoint"]["global_record"]
        sm = g["scores_mean"]
        print(f.split("/")[-1], "score:", round(sm["score_composed"], 2),
              "completion:", round(sm["score_route"], 1),
              "penalty:", round(sm["score_penalty"], 3))
    except Exception as e:
        print(f.split("/")[-1], "unreadable:", e)
EOF
