#!/bin/bash
#SBATCH --job-name=v031-furnb
#SBATCH --partition=visual
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.3.1-R Stage A, banded sweep: Town13 crashed UE4 (SIGBUS, tile
# streaming memory) in one pass — sweep each large map in spatial
# bands with a FRESH server per band, then merge (no server).
# SPEC="Town13:3 Town15:2" (town:nbands), overridable via env.

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
OUT=/scratch/$USER/ditto_av/outputs
FURN=/scratch/$USER/ditto_av/data/layout/furniture
DUMP=/scratch/$USER/ditto_av/DITTO_AV/scripts/v031_dump_furniture.py
SPEC=${SPEC:-"Town13:3 Town15:2"}
nvidia-smi -L | head -2 || true

boot() {  # boot; echoes port; retries 3x
  for attempt in 1 2 3; do
    PORT=$((20000 + RANDOM % 10000))
    /scratch/$USER/ditto_av/carla_root/CarlaUE4.sh -RenderOffScreen \
        -nosound -carla-rpc-port=$PORT \
        > $OUT/carla_server_furnb_$$_$attempt.log 2>&1 &
    SRV=$!
    sleep 40
    for probe in $(seq 1 8); do
      if $VENVPY -c "
import carla
c = carla.Client('localhost', $PORT); c.set_timeout(20.0)
c.get_server_version()" 2>/dev/null; then
        return 0
      fi
      kill -0 $SRV 2>/dev/null || break
      sleep 15
    done
    kill $SRV 2>/dev/null || true
    sleep 10
  done
  return 1
}

for spec in $SPEC; do
  TOWN=${spec%%:*}; NB=${spec##*:}
  for band in $(seq 0 $((NB - 1))); do
    [ -f "$FURN/${TOWN}_furniture_band${band}of${NB}.npz" ] && continue
    [ -f "$FURN/${TOWN}_furniture.npz" ] && continue
    echo "=== $TOWN band $band/$NB ==="
    boot || { echo "BOOT_FAILED $TOWN band $band"; exit 1; }
    $VENVPY $DUMP --port $PORT --out $FURN \
        --towns $TOWN --band $band:$NB
    kill $SRV 2>/dev/null || true
    sleep 5
  done
  $VENVPY $DUMP --out $FURN --towns $TOWN --merge
done
echo "V031_FURNITURE_BANDS_DONE"
