#!/bin/bash
#SBATCH --job-name=ditto-p2g
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Phase-2 pipeline on GPU (device=cuda), freeze-safe: runs from the HOME
# clone with the self-contained conda env ~/envs/ditto_gpu (torch cu130
# -> A100 partitions only; cu130 wheels dropped V100/sm_70). All outputs
# to $HOME; scratch is read-only (data, extracted clips).
# Defaults suit a smoke run (a100-small MIG, 40 min); for a full run:
#   sbatch -p gpu-a100 -t 06:00:00 scripts/slurm/phase2_gpu.sbatch \
#       configs/b2d_v5.yaml v5_gpu
# Env knobs: MANIFEST=<file> (default full 297), SMOKE=1 (tiny steps).

set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$HOME/ditto_out
export XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python

CONFIG=${1:-configs/b2d_v5.yaml}
SUFFIX=${2:-$SLURM_JOB_ID}
MANIFEST=${MANIFEST:-manifests/b2d_clips.txt}
SMOKE=${SMOKE:-0}

cd $HOME/ditto_work/DITTO_AV
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; \
print('GPU:', torch.cuda.get_device_name(0))"

RUN_DIR=$HOME/ditto_out/b2d_$SUFFIX
$PY - "$CONFIG" "$RUN_DIR" "$SMOKE" <<'EOF'
import sys, yaml
cfg_path, run_dir, smoke = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
cfg = yaml.safe_load(open(cfg_path)) or {}
cfg["run_dir"] = run_dir
cfg["device"] = "cuda"
if smoke:  # tiny steps: exercise every stage on cuda, not the curves
    cfg.setdefault("wm", {})["train_steps"] = 60
    cfg.setdefault("ac", {})["train_steps"] = 40
    cfg.setdefault("bc", {})["train_steps"] = 40
yaml.safe_dump(cfg, open(run_dir + "_config.yaml", "w"), sort_keys=False)
EOF
$PY scripts/run_b2d.py --config ${RUN_DIR}_config.yaml --stage all \
    --manifest $MANIFEST \
    --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted \
    ${TRUST:+--trust-manifest}
echo "PHASE2 GPU RUN DONE -> $RUN_DIR"
