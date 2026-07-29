#!/bin/bash
# Shim: the Bench2Drive evaluator launches $CARLA_ROOT/CarlaUE4.sh itself;
# this forwards to the apptainer image with GPU passthrough.
# AdditionalMaps (Town06/07/11/12/13/15) are merged in via a read-only
# directory overlay (fuse-overlayfs union — plain --bind cannot drop
# files into dirs that already exist in the SIF). Conditional so
# base-town evals survive if scratch cleanup ever removes the overlay;
# rebuild it with scripts/slurm/extract_maps.sbatch.
# CARLA_NULLRHI=1 appends -nullrhi: no rendering backend at all, so the
# server runs on CPU-only nodes (MAP track uses no cameras). GPU probe
# 2026-07-29: research-ceg-tp GPU queues saturated for >14 h,
# innovation assoc rejects all GPU requests (AssocMaxGRESPerJob).
# Mirror of this file lives in DITTO_AV/scripts/patches/CarlaUE4_shim.sh.
OV=/scratch/srahmani/ditto_av/maps_overlay
EXTRA=""
[ -d "$OV/upper" ] && EXTRA="--overlay $OV:ro"
exec apptainer exec --nv $EXTRA /scratch/srahmani/ditto_av/carla_0915.sif \
    /home/carla/CarlaUE4.sh "$@" ${CARLA_NULLRHI:+-nullrhi}
