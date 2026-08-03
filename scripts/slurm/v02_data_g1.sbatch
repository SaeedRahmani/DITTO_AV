#!/bin/bash
#SBATCH --job-name=v02-data-g1
#SBATCH --partition=compute-p1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.2 M3: build the ##glob1 297-clip npz (data stage) then run the G1
# egosim-as-selector validation (12 banked wp-family models vs dev-10
# truth). Commits results locally (compute nodes have no internet —
# push from a login session). See V02_PLAN §4 G1.

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
cd /scratch/$USER/ditto_av/DITTO_AV

python scripts/run_b2d.py --config configs/b2d_v02.yaml --stage data \
    --manifest manifests/b2d_clips.txt \
    --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted

python scripts/egosim_selector.py --data runs/b2d_v02/data \
    --out runs/egosim_selector --windows 192 --horizon 40

git add runs/egosim_selector
git commit -m "G1 VERDICT: egosim-as-selector on 12 banked wp-family models (192 val windows x 40 steps) — see runs/egosim_selector/selector.md" || true
echo "[$(date '+%Y-%m-%d %H:%M')] SESSION B job $SLURM_JOB_ID: G1 selector DONE — $(grep -o 'G1 VERDICT: [A-Z]*' runs/egosim_selector/selector.md)" \
  >> /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md
echo "V02_DATA_G1_DONE"
