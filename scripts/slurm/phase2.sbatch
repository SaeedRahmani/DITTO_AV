#!/bin/bash
#SBATCH --job-name=ditto-p2
#SBATCH --partition=compute-p1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Phase-2 Bench2Drive pipeline (all stages, open-loop eval).
# Usage:
#   sbatch scripts/slurm/phase2.sbatch [config] [run_suffix]

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=/scratch/$USER/ditto_av/outputs

CONFIG=${1:-configs/b2d.yaml}
SUFFIX=${2:-$SLURM_JOB_ID}
BASE=/scratch/$USER/ditto_av

cd $BASE/DITTO_AV
RUN_DIR=$BASE/outputs/b2d_$SUFFIX
python - "$CONFIG" "$RUN_DIR" <<'EOF'
import sys, yaml
cfg_path, run_dir = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path)) or {}
cfg["run_dir"] = run_dir
yaml.safe_dump(cfg, open(run_dir + "_config.yaml", "w"), sort_keys=False)
EOF
python scripts/run_b2d.py --config ${RUN_DIR}_config.yaml --stage all \
    --manifest manifests/b2d_clips.txt \
    --extracted $BASE/data/bench2drive/extracted
echo "PHASE2 RUN DONE -> $RUN_DIR"
