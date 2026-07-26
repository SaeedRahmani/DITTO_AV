#!/bin/bash
#SBATCH --job-name=ditto-p1
#SBATCH --partition=compute-p1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# One Phase-1 sweep run.
# Usage:
#   sbatch scripts/slurm/phase1.sbatch <config.yaml> [reuse_run_dir]
# With reuse_run_dir: copy that run's collected data + world model and run
# only the policies+eval stages (AC-only ablations share the exact WM).

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CONFIG=$(readlink -f "${1:?usage: phase1.sbatch <config.yaml> [reuse_run_dir]}")
REUSE=${2:-}

cd /scratch/$USER/ditto_av/DITTO_AV
RUN_DIR=$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['run_dir'])" "$CONFIG")

if [ -n "$REUSE" ]; then
    mkdir -p "$RUN_DIR/data" "$RUN_DIR/checkpoints"
    cp "$REUSE"/data/*.npz "$RUN_DIR/data/"
    cp "$REUSE"/checkpoints/world_model.pt "$RUN_DIR/checkpoints/"
    python scripts/run_pipeline.py --config "$CONFIG" --stage policies
    python scripts/run_pipeline.py --config "$CONFIG" --stage eval
else
    python scripts/run_pipeline.py --config "$CONFIG" --stage all
fi
echo "PHASE1 RUN DONE -> $RUN_DIR"
