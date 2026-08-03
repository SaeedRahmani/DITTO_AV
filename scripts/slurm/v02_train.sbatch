#!/bin/bash
#SBATCH --job-name=v02-clp
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.2 clp training: data (npz-cache hit) -> clp stage (BC pretrain +
# egosim on-policy RL + G2 verdict). Runs from the HOME clone (pull it
# on a login node BEFORE submitting — compute nodes have no internet).
# Usage: sbatch [--partition=...] scripts/slurm/v02_train.sbatch \
#            [config] [suffix] [manifest]

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline WANDB_DIR=$HOME/ditto_out
export XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
CONFIG=${1:-configs/b2d_v02.yaml}
SUFFIX=${2:-v02}
MANIFEST=${3:-manifests/b2d_clips.txt}
RUN=$HOME/ditto_out/b2d_$SUFFIX
mkdir -p $RUN
cd $HOME/ditto_work/DITTO_AV
$PY - "$CONFIG" "$RUN" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["run_dir"] = sys.argv[2]
cfg["device"] = "cuda"
yaml.safe_dump(cfg, open(sys.argv[2] + "_config.yaml", "w"),
               sort_keys=False)
cfg["device"] = "cpu"  # CARLA eval agents run CPU torch
yaml.safe_dump(cfg, open(sys.argv[2] + "_cpu.yaml", "w"),
               sort_keys=False)
PYEOF
for STAGE in data clp; do
  $PY scripts/run_b2d.py --config ${RUN}_config.yaml \
      --stage $STAGE --manifest "$MANIFEST" --trust-manifest \
      --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted
done
echo "V02_CLP_DONE $SUFFIX"
