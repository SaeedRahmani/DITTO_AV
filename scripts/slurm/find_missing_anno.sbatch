#!/bin/bash
#SBATCH --job-name=find-anno
#SBATCH --partition=compute-p2
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# List manifest clips whose tarball contains anno/ -> the "present"
# manifest for --trust-manifest cache-key matching (one 2026-07-30 clip
# ships without anno; key is computed over present clips only).
set -euo pipefail
REPO=/scratch/$USER/ditto_av/DITTO_AV
DATA=/scratch/$USER/ditto_av/data/bench2drive
OUT=$REPO/manifests/b2d_full_present.txt
tr -d '\r' < $REPO/manifests/b2d_full.txt | sed '/^$/d' | \
  xargs -P 16 -I{} sh -c '
    if tar -tzf '"$DATA"'/{} 2>/dev/null | grep -q "/anno/"; then echo {}; fi' \
  | sort > $OUT
echo "present: $(wc -l < $OUT)"
comm -23 <(sort $REPO/manifests/b2d_full.txt) $OUT | sed "s/^/MISSING_ANNO: /"
echo "FIND_ANNO_DONE"
