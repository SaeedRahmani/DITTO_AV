#!/bin/bash
#SBATCH --job-name=v02-clp
#SBATCH --partition=participation
#SBATCH --gpus-per-task=1
#SBATCH --time=03:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.2 M4: full clp training (seq-BC 8k + egosim A2C 4k) + G2 verdict
# on the 297-clip set. Runs from the HOME clone (freeze-safe pattern),
# npz via the shared scratch cache (job v02_data_g1 must have run).
# Commits clp_g2.json into the home clone (push from a login session).

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline WANDB_DIR=$HOME/ditto_out
export XDG_CACHE_HOME=$HOME/ditto_out/cache
PY=$HOME/envs/ditto_gpu/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
RUN=$HOME/ditto_out/b2d_v02
mkdir -p $RUN
cd $HOME/ditto_work/DITTO_AV
$PY - configs/b2d_v02.yaml $RUN <<'PYIN'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["run_dir"] = sys.argv[2]
cfg["device"] = "cuda"
yaml.safe_dump(cfg, open(sys.argv[2] + "_config.yaml", "w"),
               sort_keys=False)
PYIN
for STAGE in data clp; do
  $PY scripts/run_b2d.py --config ${RUN}_config.yaml --stage $STAGE \
      --manifest manifests/b2d_clips.txt --trust-manifest \
      --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted
done
mkdir -p runs/b2d_v02/results
cp $RUN/results/clp_g2.json runs/b2d_v02/results/
git add runs/b2d_v02/results/clp_g2.json
git commit -m "G2 VERDICT (297 clips): clp_rl vs clp_bc closed-loop in egosim, clean + divergent starts — runs/b2d_v02/results/clp_g2.json" || true
echo "[$(date '+%Y-%m-%d %H:%M')] SESSION B job $SLURM_JOB_ID: v02 clp train+G2 DONE (ckpts $RUN/checkpoints, results committed to HOME clone — push needed)" \
  >> /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md
echo "V02_CLP_DONE"
