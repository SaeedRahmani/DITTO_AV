#!/bin/bash
#SBATCH --job-name=ditto-train
#SBATCH --partition=compute-p1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
# NOTE: add your account line (see test.sbatch).
# Usage:
#   sbatch scripts/slurm/train.sbatch [config] [run_dir_suffix]
# Defaults: configs/av.yaml, run suffix = SLURM job id.

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=1

CONFIG=${1:-configs/av.yaml}
SUFFIX=${2:-$SLURM_JOB_ID}
BASE=/scratch/$USER/ditto_av

cd $BASE/DITTO_AV
# redirect run outputs to scratch outputs/ by templating the config's run_dir
RUN_DIR=$BASE/outputs/run_$SUFFIX
python - "$CONFIG" "$RUN_DIR" <<'EOF'
import sys, yaml
cfg_path, run_dir = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path)) or {}
cfg["run_dir"] = run_dir
out = run_dir + "_config.yaml"
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(out)
EOF
python scripts/run_pipeline.py --config ${RUN_DIR}_config.yaml --stage all
echo "TRAIN DONE -> $RUN_DIR"
