#!/bin/bash
#SBATCH --job-name=v031-furn
#SBATCH --partition=visual
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# v0.3.1-R Stage A: boot CARLA (boot-probe pattern) and dump per-town
# static-furniture OBBs. UE4 boot is flaky (the eval chain tolerates
# per-variant failures) -> up to 3 boot attempts, server log kept.

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
OUT=/scratch/$USER/ditto_av/outputs
nvidia-smi -L | head -2 || true

boot_ok=0
for attempt in 1 2 3; do
  PORT=$((20000 + RANDOM % 10000))
  echo "=== boot attempt $attempt port $PORT ==="
  /scratch/$USER/ditto_av/carla_root/CarlaUE4.sh -RenderOffScreen -nosound \
      -carla-rpc-port=$PORT > $OUT/carla_server_furn_$attempt.log 2>&1 &
  SRV=$!
  trap "kill $SRV 2>/dev/null || true" EXIT
  sleep 40
  for probe in $(seq 1 8); do
    if $VENVPY -c "
import carla, sys
c = carla.Client('localhost', $PORT); c.set_timeout(20.0)
print('server:', c.get_server_version())" 2>/dev/null; then
      boot_ok=1; break
    fi
    kill -0 $SRV 2>/dev/null || { echo "server process died"; break; }
    sleep 15
  done
  [ $boot_ok -eq 1 ] && break
  echo "attempt $attempt failed; server log tail:"
  tail -5 $OUT/carla_server_furn_$attempt.log || true
  kill $SRV 2>/dev/null || true
  sleep 10
done
[ $boot_ok -eq 1 ] || { echo "V031_FURNITURE_BOOT_FAILED"; exit 1; }

$VENVPY /scratch/$USER/ditto_av/DITTO_AV/scripts/v031_dump_furniture.py \
    --port $PORT --out /scratch/$USER/ditto_av/data/layout/furniture
echo "V031_FURNITURE_DONE"
