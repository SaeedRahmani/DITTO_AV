#!/bin/bash
#SBATCH --job-name=v02-data
#SBATCH --partition=compute-p1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.2 data stage: parse a manifest's EXTRACTED clips into the ##glob1
# npz cache (global egosim arrays). CPU-only; GPU training jobs then hit
# the cache via --trust-manifest. Usage:
#   sbatch scripts/slurm/v02_data.sbatch [config] [manifest]

set -euo pipefail
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
CONFIG=${1:-configs/b2d_v02_297.yaml}
MANIFEST=${2:-manifests/b2d_clips.txt}
cd /scratch/$USER/ditto_av/DITTO_AV
python - "$CONFIG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["device"] = "cpu"
yaml.safe_dump(cfg, open("/tmp/v02_data_config.yaml", "w"),
               sort_keys=False)
PYEOF
python scripts/run_b2d.py --config /tmp/v02_data_config.yaml \
    --stage data --manifest "$MANIFEST" \
    --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted
echo "V02_DATA_DONE $MANIFEST"
