#!/bin/bash
#SBATCH --job-name=b2d-validate
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Extract anno/ from downloaded Bench2Drive tarballs and parse every clip
# with ditto_av/bench2drive.py. Chain it after the download job:
#   sbatch --dependency=afterok:<download_jobid> scripts/slurm/validate_b2d.sbatch

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /scratch/$USER/ditto_av/DITTO_AV
python scripts/validate_b2d.py manifests/b2d_clips.txt \
    --data-dir /scratch/$USER/ditto_av/data/bench2drive
echo "VALIDATION DONE"
