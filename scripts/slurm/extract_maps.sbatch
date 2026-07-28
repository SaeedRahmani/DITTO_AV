#!/bin/bash
#SBATCH --job-name=maps-extract
#SBATCH --partition=compute-p1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# One-off: extract AdditionalMaps_0.9.15.tar.gz (Town06/07/11/12/13/15,
# 6.9 GB, ~23k files) into an apptainer *directory overlay* so the CARLA
# SIF gains the extra towns without a rebuild. The tarball drops files
# INTO dirs that already exist in the SIF (Maps/, Nav/, OpenDrive/,
# Sublevels/, TM/, Engine/Content) — plain --bind cannot merge that;
# an overlay can (verified on login node: fuse-overlayfs 1.5.2 union
# works unprivileged with --overlay dir:ro).
# Inode budget: ~23k chunk files on scratch — checked against the 1M
# quota (789k used, >=100k headroom rule holds; see DELFTBLUE.md).
# After success: add "--overlay $OV:ro" to carla_root/CarlaUE4.sh.

set -euo pipefail
TARBALL=/scratch/$USER/ditto_av/AdditionalMaps_0.9.15.tar.gz
OV=/scratch/$USER/ditto_av/maps_overlay
[ -s "$TARBALL" ] || { echo "tarball missing: $TARBALL"; exit 1; }

rm -rf $OV
mkdir -p $OV/upper/home/carla $OV/work
tar -xzf $TARBALL -C $OV/upper/home/carla/
echo "extracted:"
ls $OV/upper/home/carla/
echo "new towns in Maps/:"
ls $OV/upper/home/carla/CarlaUE4/Content/Carla/Maps/ | grep -o 'Town[0-9HD]*' | sort -u
echo "OpenDrive files:"
ls $OV/upper/home/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/ 2>/dev/null || true
echo "file count (inode cost):"
find $OV -type f | wc -l
echo "MAPS_EXTRACT_DONE"
