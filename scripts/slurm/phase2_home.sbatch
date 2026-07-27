#!/bin/bash
#SBATCH --job-name=ditto-p2h
#SBATCH --partition=compute-p1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Phase-2 pipeline, freeze-safe variant: runs from the HOME clone and
# writes every output (run_dir, wandb, slurm log) to $HOME, so it works
# while /scratch is write-frozen (scratch stays read-only: venv, data).
# Usage:
#   sbatch scripts/slurm/phase2_home.sbatch [config] [run_suffix]

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$HOME/ditto_out
export XDG_CACHE_HOME=$HOME/ditto_out/cache

CONFIG=${1:-configs/b2d_v5.yaml}
SUFFIX=${2:-$SLURM_JOB_ID}

cd $HOME/ditto_work/DITTO_AV
RUN_DIR=$HOME/ditto_out/b2d_$SUFFIX
python - "$CONFIG" "$RUN_DIR" <<'EOF'
import sys, yaml
cfg_path, run_dir = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path)) or {}
cfg["run_dir"] = run_dir
yaml.safe_dump(cfg, open(run_dir + "_config.yaml", "w"), sort_keys=False)
EOF
python scripts/run_b2d.py --config ${RUN_DIR}_config.yaml --stage all \
    --manifest manifests/b2d_clips.txt \
    --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted
echo "PHASE2 RUN DONE -> $RUN_DIR"
