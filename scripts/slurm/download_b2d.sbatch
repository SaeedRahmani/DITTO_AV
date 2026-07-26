#!/bin/bash
#SBATCH --job-name=b2d-download
#SBATCH --partition=compute-p1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
# NOTE: add your account line (see test.sbatch).
# DO NOT SUBMIT until the user approves the download (~200 GB full base split).
# Usage:
#   sbatch scripts/slurm/download_b2d.sbatch manifests/b2d_clips.txt
# where the manifest is a text file with one clip tarball name per line, e.g.
#   Accident_Town03_Route101_Weather23.tar.gz
# Compute nodes on DelftBlue have outbound internet via proxy; if downloads
# fail on a compute node, run this script's body on a login node with nohup
# instead (transfers are allowed there).

set -euo pipefail
MANIFEST=${1:?usage: sbatch download_b2d.sbatch <manifest.txt>}
DEST=/scratch/$USER/ditto_av/data/bench2drive
mkdir -p "$DEST"
cd "$DEST"

BASE_URL=https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main
while read -r clip; do
    [ -z "$clip" ] && continue
    if [ -f "$clip.done" ]; then echo "skip $clip"; continue; fi
    echo "downloading $clip"
    curl -sSL --retry 5 --retry-delay 10 -o "$clip" "$BASE_URL/$clip"
    tar -tzf "$clip" > /dev/null   # integrity check
    touch "$clip.done"
done < "$MANIFEST"
echo "DOWNLOAD BATCH DONE: $(ls *.done | wc -l) clips verified"
