#!/bin/bash
#SBATCH --job-name=find-anno
#SBATCH --partition=compute-p2
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v2: a clip is "present" only if its tarball root dir MATCHES the
# manifest name AND contains anno/ — extraction keys dirs by manifest
# name, so a root-name mismatch behaves like a missing clip (found
# 2026-07-30: 1000 tarballs have anno/ but only 999 match their name).
set -euo pipefail
REPO=/scratch/$USER/ditto_av/DITTO_AV
DATA=/scratch/$USER/ditto_av/data/bench2drive
OUT=$REPO/manifests/b2d_full_present.txt
tr -d '\r' < $REPO/manifests/b2d_full.txt | sed '/^$/d' | \
  xargs -P 16 -I{} sh -c '
    name="{}"; name=${name%.tar.gz}
    if tar -tzf '"$DATA"'/{} 2>/dev/null | grep -m1 -q "^\./$name/anno/"; then
      echo "{}"
    else
      echo "NOT_PRESENT_AS_NAMED: {}" >&2
    fi' | sort > $OUT
echo "present-as-named: $(wc -l < $OUT)"
touch $OUT.v2done
echo "FIND_ANNO_DONE"
