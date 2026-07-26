#!/bin/bash
#SBATCH --job-name=ditto-test
#SBATCH --partition=compute-p1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# (alternative account: innovation)
# Find it with: sacctmgr show user $USER withassoc format=account%30

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR=/scratch/$USER/ditto_av/cache/pip

cd /scratch/$USER/ditto_av/DITTO_AV
python -m pytest tests/ -q
python scripts/run_pipeline.py --config configs/smoke.yaml --stage all
echo "SMOKE OK"
