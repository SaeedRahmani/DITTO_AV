#!/bin/bash
#SBATCH --job-name=carla-boot
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Boot-test the CARLA 0.9.15 server (apptainer shim, offscreen) on a
# candidate GPU lane discovered by the 2026-07-29 probe matrix
# (participation: H100 NVL; visual: Quadro RTX 4000 — both instant-start
# while gpu-a100/gpu-v100 pend >14 h). Verifies: server binds RPC,
# client connects, Town03 loads + ticks + vehicle moves, Town12
# (AdditionalMaps overlay -> also tests fuse-overlayfs on the node).
# Submit: sbatch -p participation --gpus-per-task=1 scripts/slurm/carla_boot_probe.sbatch
#         sbatch -p visual scripts/slurm/carla_boot_probe.sbatch   (GPU unmanaged, no GRES)

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
PORT=$((20000 + RANDOM % 10000))
nvidia-smi -L | head -2 || true

/scratch/$USER/ditto_av/carla_root/CarlaUE4.sh -RenderOffScreen -nosound \
    -carla-rpc-port=$PORT &
SRV=$!
trap "kill $SRV 2>/dev/null || true" EXIT
sleep 45

$VENVPY - "$PORT" <<'EOF'
import sys
import time

import carla

port = int(sys.argv[1])
c = carla.Client("localhost", port)
c.set_timeout(180.0)
print("server version:", c.get_server_version())

for town in ["Town03", "Town12"]:
    t0 = time.time()
    w = c.load_world(town)
    s = w.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    w.apply_settings(s)
    for _ in range(20):
        w.tick()
    bp = w.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    sp = w.get_map().get_spawn_points()[0]
    v = w.try_spawn_actor(bp, sp)
    assert v is not None, f"spawn failed in {town}"
    v.apply_control(carla.VehicleControl(throttle=0.7))
    p0 = v.get_location()
    for _ in range(40):
        w.tick()
    moved = p0.distance(v.get_location())
    lights = w.get_actors().filter("traffic.traffic_light*")
    print(f"{town}: load+60ticks {time.time()-t0:.0f}s, moved {moved:.1f} m, "
          f"{len(lights)} lights")
    assert moved > 1.0, f"vehicle did not move in {town}"
    v.destroy()
print("CARLA_BOOT_OK")
EOF
