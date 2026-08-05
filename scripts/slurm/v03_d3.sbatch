#!/bin/bash
#SBATCH --job-name=v03-d3
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# D3: fine-tune the v0.2 champion (999s) in the reactive world.

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=1 XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
cd /scratch/$USER/ditto_av/DITTO_AV_v03
$PY scripts/v03_train_reactive.py \
    --data $HOME/ditto_out/v03_w0c/data \
    --w0 $HOME/ditto_out/v03_w0c \
    --champion $HOME/ditto_out/b2d_v02_999s/checkpoints/clp_rl.pt \
    --champion-cfg $HOME/ditto_out/b2d_v02_999s_cpu.yaml \
    --out $HOME/ditto_out/v03_d3 \
    --steps 3500 --n-ensemble 2 --p-reactive 0.5 --device cuda
echo "V03_D3_DONE"
