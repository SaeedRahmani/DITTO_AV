#!/bin/bash
#SBATCH --job-name=v03-w0
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# V3-M2b: stage ##glob2 npz (cache hit) -> train K=4 traffic seeds ->
# W0 gate report. Runs from the v0.3 WORKTREE.

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=1 XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
RUN=$HOME/ditto_out/v03_w0c
mkdir -p $RUN
cd /scratch/$USER/ditto_av/DITTO_AV_v032
$PY - "$RUN" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/b2d_v02.yaml"))
cfg["run_dir"] = sys.argv[1]
cfg["device"] = "cuda"
yaml.safe_dump(cfg, open(sys.argv[1] + "_config.yaml", "w"),
               sort_keys=False)
PYEOF
$PY scripts/run_b2d.py --config ${RUN}_config.yaml --stage data \
    --manifest manifests/b2d_full999.txt --trust-manifest \
    --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted
$PY scripts/v03_train_traffic.py --data $RUN/data --out $RUN \
    --seeds 4 --steps 25000 --batch 48 --rollout-steps 4000 --rollout-k 15 --device cuda
echo "V03_W0_DONE"
