#!/bin/bash
#SBATCH --job-name=v031-e3-wm
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# E3 step 1: frozen world model for the factored-latent reward.

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=1 XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
cd /scratch/$USER/ditto_av/DITTO_AV
$PY scripts/v031_e3_wm.py \
    --data $HOME/ditto_out/v03_w0c/data \
    --champion-cfg $HOME/ditto_out/b2d_v02_999s_cpu.yaml \
    --out $HOME/ditto_out/v031_e3 \
    --steps 4000 --device cuda
echo "V031_E3_WM_DONE"
