#!/bin/bash
#SBATCH --job-name=b2d-cache
#SBATCH --partition=compute-p1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Gen-2 data stage, inode-safe (DHPC node-local /tmp pattern): extract
# anno/ from ALL manifest tarballs into /tmp on the compute node, run
# run_b2d stage=data there, and copy only the packed npz cache back to
# scratch. Scratch never sees the ~200k extracted files (826k/1M inode
# quota, 2026-07-29). Training jobs then use the SAME manifest with
# --trust-manifest and hit the cache without any extraction.
# Usage: sbatch scripts/slurm/extract_cache_b2d.sbatch [manifest]
#        (default manifests/b2d_full.txt, layout: route conditioning on,
#         lights off — matches the kl01/gen-2 configs)

set -euo pipefail
# self-contained conda env: works on ANY node. The ditto venv is NOT
# usable on compute-p2 (job 10545405: module tree absent, python fell
# through to system python without yaml — same failure mode as GPU
# nodes; venv is safe only on login + compute-p1).
PY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
$PY -c "import yaml, numpy, torch" || { echo "carla_eval env broken"; exit 1; }
export OMP_NUM_THREADS=1
export WANDB_DISABLED=1
export PYTHONUNBUFFERED=1

REPO=/scratch/$USER/ditto_av/DITTO_AV
DATA=/scratch/$USER/ditto_av/data/bench2drive
MANIFEST=$REPO/${1:-manifests/b2d_full.txt}
TMP=/tmp/$SLURM_JOB_ID/bench2drive
trap "rm -rf /tmp/$SLURM_JOB_ID" EXIT
mkdir -p $TMP/extracted

df -h /tmp | tail -1
FREE_GB=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 100 ] || { echo "only ${FREE_GB}G free on /tmp — abort"; exit 1; }

# anno-only extraction, parallel across the job's CPUs
# tarball members are prefixed "./<name>/..." — match with */anno/*
tr -d '\r' < $MANIFEST | sed '/^$/d' | \
  xargs -P 8 -I{} sh -c '
    tar -xzf '"$DATA"'/{} -C '"$TMP"'/extracted --wildcards "*/anno/*" \
      || echo "EXTRACT_FAIL {}"'
echo "extracted: $(ls $TMP/extracted | wc -l) clip dirs"

# temp config: only the data stage runs; run_dir is throwaway on /tmp
$PY - <<EOF
import yaml
cfg = yaml.safe_load(open("$REPO/configs/b2d_v3kl01.yaml"))
cfg["run_dir"] = "/tmp/$SLURM_JOB_ID/datastage"
yaml.safe_dump(cfg, open("/tmp/$SLURM_JOB_ID/data_cfg.yaml", "w"), sort_keys=False)
EOF
cd $REPO
$PY scripts/run_b2d.py --config /tmp/$SLURM_JOB_ID/data_cfg.yaml \
    --stage data --manifest $MANIFEST --extracted $TMP/extracted

# only the packed cache goes to scratch (2 files per key)
mkdir -p $DATA/npz_cache
cp -v $TMP/npz_cache/*.npz $DATA/npz_cache/
ls -lh $DATA/npz_cache/ | tail -5
echo "B2D_CACHE_DONE"
