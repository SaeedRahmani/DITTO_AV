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
# v0.3.2 axis-1: D3 rerun with the sigma_yawrate=0.5 reward channel
# (V032_PLAN 1.3.1). Same protocol as v03_d3.sbatch otherwise.

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=1 XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
cd /scratch/$USER/ditto_av/DITTO_AV_v032
$PY scripts/v032_train_smooth.py \
    --data $HOME/ditto_out/v03_w0c/data \
    --w0 $HOME/ditto_out/v03_w0c \
    --champion $HOME/ditto_out/b2d_v02_999s/checkpoints/clp_rl.pt \
    --champion-cfg $HOME/ditto_out/b2d_v02_999s_cpu.yaml \
    --out $HOME/ditto_out/v032_d3sm \
    --sigma-yawrate 0.5 --steps 3500 --n-ensemble 2 --p-reactive 0.5 \
    --device cuda
# stage the ckpt where configs/diag_v032_sm.yaml expects it
mkdir -p $HOME/ditto_out/b2d_v032_sm/checkpoints
cp $HOME/ditto_out/v032_d3sm/clp_rx.pt \
   $HOME/ditto_out/b2d_v032_sm/checkpoints/clp_rx.pt
# S1b smoothness eval on the fresh ckpt (the exact A0 protocol)
$PY scripts/v032_smooth_audit.py egosim --tag v032_sm \
    --ckpt $HOME/ditto_out/b2d_v032_sm/checkpoints/clp_rx.pt \
    --cfg $HOME/ditto_out/b2d_v032_sm_cpu.yaml
echo "[$(date '+%Y-%m-%d %H:%M')] V03.2: axis-1 training $SLURM_JOB_ID DONE (v032_d3sm/d3sm_report.json + v032/a0_egosim_v032_sm.json)" \
    >> /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md
echo "V032_SMOOTH_DONE"
