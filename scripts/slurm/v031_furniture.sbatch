#!/bin/bash
#SBATCH --job-name=v031-furn
#SBATCH --partition=visual
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.3.1-R Stage A: boot CARLA (boot-probe pattern, visual lane needs
# no GRES flag) and dump per-town static-furniture OBBs.

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
PORT=$((20000 + RANDOM % 10000))
nvidia-smi -L | head -2 || true

/scratch/$USER/ditto_av/carla_root/CarlaUE4.sh -RenderOffScreen -nosound \
    -carla-rpc-port=$PORT &
SRV=$!
trap "kill $SRV 2>/dev/null || true" EXIT
sleep 45

$VENVPY /scratch/$USER/ditto_av/DITTO_AV/scripts/v031_dump_furniture.py \
    --port $PORT --out /scratch/$USER/ditto_av/data/layout/furniture
echo "V031_FURNITURE_DONE"
