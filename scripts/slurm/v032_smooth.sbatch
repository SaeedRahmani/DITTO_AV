#!/bin/bash
#SBATCH --job-name=v032-smooth
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.3.2 axis-1: D3 rerun with the sigma_yawrate reward channel
# (V032_PLAN 1.3.1). Same protocol as v03_d3.sbatch otherwise.
# Env overrides per round: SIGMA_YR (0.5), STEPS (3500), TAG
# (v032_d3sm). Submit as: SIGMA_YR=1.0 STEPS=6000 TAG=v032_d3sm2 \
#   sbatch scripts/slurm/v032_smooth.sbatch   (env style — NOT
#   --export with commas, see v032_carla_chain.sbatch header)

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=1 XDG_CACHE_HOME=$HOME/ditto_out/cache
SIGMA_YR=${SIGMA_YR:-0.5}
W_CHURN=${W_CHURN:-0.0}
W_CONS=${W_CONS:-0.0}
GATE_D0=${GATE_D0:-0.0}
STEPS=${STEPS:-3500}
SEED=${SEED:--1}
TAG=${TAG:-v032_d3sm}
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
cd /scratch/$USER/ditto_av/DITTO_AV_v032
$PY scripts/v032_train_smooth.py \
    --data $HOME/ditto_out/v03_w0c/data \
    --w0 $HOME/ditto_out/v03_w0c \
    --champion $HOME/ditto_out/b2d_v02_999s/checkpoints/clp_rl.pt \
    --champion-cfg $HOME/ditto_out/b2d_v02_999s_cpu.yaml \
    --out $HOME/ditto_out/$TAG \
    --sigma-yawrate $SIGMA_YR --w-churn $W_CHURN \
    --w-consistency $W_CONS --cons-gate-d0 $GATE_D0 \
    --steps $STEPS --seed $SEED \
    --n-ensemble 2 --p-reactive 0.5 --device cuda
# stage the ckpt where configs/diag_v032_sm.yaml expects it (the
# "current arm" slot — only a gate-passing round keeps it)
mkdir -p $HOME/ditto_out/b2d_v032_sm/checkpoints
cp $HOME/ditto_out/$TAG/clp_rx.pt \
   $HOME/ditto_out/b2d_v032_sm/checkpoints/clp_rx.pt
# S1b smoothness eval on the fresh ckpt (the exact A0 protocol)
$PY scripts/v032_smooth_audit.py egosim --tag $TAG \
    --ckpt $HOME/ditto_out/$TAG/clp_rx.pt \
    --cfg $HOME/ditto_out/b2d_v032_sm_cpu.yaml
echo "[$(date '+%Y-%m-%d %H:%M')] V03.2: axis-1 training $SLURM_JOB_ID ($TAG, sigma_yr $SIGMA_YR, $STEPS steps) DONE (d3sm_report.json + a0_egosim_$TAG.json)" \
    >> /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md
echo "V032_SMOOTH_DONE"
