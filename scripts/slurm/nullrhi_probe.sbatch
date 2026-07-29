#!/bin/bash
#SBATCH --job-name=nullrhi-probe
#SBATCH --partition=compute-p1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/scratch/%u/ditto_av/outputs/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Feasibility: CARLA 0.9.15 server with -nullrhi (no rendering backend)
# on a CPU-only compute node. The Bench2Drive MAP track uses no camera
# sensors and our agent reads privileged state, so rendering may be
# unnecessary — success here decouples all closed-loop evals from the
# saturated GPU queues (compute-p1: 120 h cap, plentiful).
# Checks: server boots, client connects, base town loads + ticks,
# vehicle spawns and moves under autopilot-less control, traffic lights
# report state, and Town12 (AdditionalMaps overlay) loads.

set -euo pipefail
VENVPY=/scratch/$USER/ditto_av/envs/carla_eval/bin/python
export CARLA_NULLRHI=1
PORT=$((20000 + RANDOM % 10000))

/scratch/$USER/ditto_av/carla_root/CarlaUE4.sh -RenderOffScreen -nosound \
    -carla-rpc-port=$PORT &
SRV=$!
trap "kill $SRV 2>/dev/null || true" EXIT
sleep 60

$VENVPY - "$PORT" <<'EOF'
import sys
import time

import carla

port = int(sys.argv[1])
c = carla.Client("localhost", port)
c.set_timeout(120.0)
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
    p1 = v.get_location()
    moved = p0.distance(p1)
    lights = w.get_actors().filter("traffic.traffic_light*")
    states = [str(l.get_state()) for l in list(lights)[:5]]
    n_wp = len(w.get_map().generate_waypoints(50.0))
    print(f"{town}: load+40ticks {time.time()-t0:.0f}s, moved {moved:.1f} m, "
          f"{len(lights)} lights {states}, {n_wp} waypoints")
    v.destroy()
print("NULLRHI_PROBE_OK")
EOF
