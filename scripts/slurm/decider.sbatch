#!/bin/bash
#SBATCH --job-name=decider
#SBATCH --partition=visual
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Pipeline decider stage (see scripts/pipeline_decider.py). Runs on the
# instant-start visual lane (CPU work only). Chained behind eval jobs
# with --dependency=afterany so the pipeline drives itself on the
# cluster — no interactive session required. User-readable state:
# outputs/PIPELINE_STATUS.md.
set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$VENVPY /scratch/$USER/ditto_av/DITTO_AV/scripts/pipeline_decider.py "$1"
