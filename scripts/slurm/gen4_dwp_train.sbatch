#!/bin/bash
#SBATCH --job-name=g4dwp-train
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# gen3_wph training: BC-only on top of the REUSED gen3_clean world
# model (control-action WM; pins the 82.7%-completion features so the
# wp-head change is isolated — the point of the v2 experiment).
# Stages: data (cache hit a8074aa357d4) -> policies (BC; DITTO heads
# skipped in wp_head mode) -> eval. Submitted 2026-07-31 as job
# 10557266 (via a scratchpad copy of this script).

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline WANDB_DIR=$HOME/ditto_out
export XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
CONFIG=${1:-configs/b2d_gen4_dwp.yaml}
SUFFIX=${2:-gen4_dwp}
RUN=$HOME/ditto_out/b2d_$SUFFIX
mkdir -p $RUN/checkpoints
cp -v $HOME/ditto_out/b2d_gen3_wph/checkpoints/world_model.pt \
      $HOME/ditto_out/b2d_gen3_wph/checkpoints/bc.pt $RUN/checkpoints/
cd $HOME/ditto_work/DITTO_AV
$PY - "$CONFIG" "$RUN" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["run_dir"] = sys.argv[2]
cfg["device"] = "cuda"
yaml.safe_dump(cfg, open(sys.argv[2] + "_config.yaml", "w"),
               sort_keys=False)
PYEOF
for STAGE in data policies eval; do
  $PY scripts/run_b2d.py --config ${RUN}_config.yaml \
      --stage $STAGE --manifest manifests/b2d_full999.txt --trust-manifest \
      --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted
done
echo "GEN3_WPH_TRAIN_DONE"
