#!/bin/bash
#SBATCH --job-name=v031-d3
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.3.1 D3 rerun: reactive-world fine-tune WITH the static layout
# penalty (w=0.5). Two arms, one submission each:
#   sbatch v031_d3.sbatch <init_ckpt.pt> <out_dir>
# arm rx: init from v0.3 clp_rx; arm s: init from the 999s champion.

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=1 XDG_CACHE_HOME=$HOME/ditto_out/cache
INIT=${1:?usage: sbatch v031_d3.sbatch <init_ckpt.pt> <out_dir>}
OUT=${2:?usage: sbatch v031_d3.sbatch <init_ckpt.pt> <out_dir>}
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
cd /scratch/$USER/ditto_av/DITTO_AV
$PY scripts/v03_train_reactive.py \
    --data $HOME/ditto_out/v03_w0c/data \
    --w0 $HOME/ditto_out/v03_w0c \
    --champion "$INIT" \
    --champion-cfg $HOME/ditto_out/b2d_v02_999s_cpu.yaml \
    --out "$OUT" \
    --steps 3500 --n-ensemble 2 --p-reactive 0.5 --device cuda \
    --w-layout 0.5
echo "V031_D3_DONE $OUT"
