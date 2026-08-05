#!/usr/bin/env python3
"""v0.3.1-R Stage A: dump CARLA static-furniture OBBs per town.

The W3 root cause: every dev-10 "layout" collision was map furniture
(static.fence / static.prop.mesh / static.vegetation) standing on or
beside drivable area — invisible to lane-union geometry. This dump
extracts the actual collidable geometry from a live CARLA server:
world.get_environment_objects per CityObjectLabel, world-frame boxes.

Per town npz: center (N,3), extent (N,3), yaw (N,) [deg], label (N,)
int-coded, plus the label legend. Both the merged environment objects
and the finer get_level_bbs boxes are stored (audit picks per label).

Runs inside the eval CARLA (server already booted by the sbatch):
  v031_dump_furniture.py --port P --out DIR [--towns Town01,...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LABELS = ["Fences", "Walls", "Poles", "Vegetation", "Static",
          "GuardRail", "Buildings", "Other"]
# npz-name -> CARLA load_world name
TOWNS = {f"Town{n:02d}": f"Town{n:02d}" for n in
         (1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15)}
TOWNS["Town10"] = "Town10HD"


def dump_town(client, town_npz: str, town_carla: str, out: Path):
    import carla
    w = client.load_world(town_carla)
    cen, ext, yaw, lab, src = [], [], [], [], []
    for li, lname in enumerate(LABELS):
        clab = getattr(carla.CityObjectLabel, lname)
        for o in w.get_environment_objects(clab):
            b = o.bounding_box
            cen.append([b.location.x, b.location.y, b.location.z])
            ext.append([b.extent.x, b.extent.y, b.extent.z])
            yaw.append(b.rotation.yaw)
            lab.append(li)
            src.append(0)
        for b in w.get_level_bbs(clab):
            cen.append([b.location.x, b.location.y, b.location.z])
            ext.append([b.extent.x, b.extent.y, b.extent.z])
            yaw.append(b.rotation.yaw)
            lab.append(li)
            src.append(1)
    np.savez_compressed(
        out / f"{town_npz}_furniture.npz",
        center=np.array(cen, dtype=np.float32),
        extent=np.array(ext, dtype=np.float32),
        yaw=np.array(yaw, dtype=np.float32),
        label=np.array(lab, dtype=np.int16),
        source=np.array(src, dtype=np.int8),   # 0=env_obj 1=level_bbs
        labels=np.array(LABELS))
    print(f"{town_npz}: {len(cen)} boxes "
          f"({sum(1 for s in src if s == 0)} env / "
          f"{sum(1 for s in src if s == 1)} level)")


def main():
    import carla
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--towns", default=",".join(sorted(TOWNS)))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = carla.Client("localhost", args.port)
    c.set_timeout(300.0)
    print("server:", c.get_server_version())
    for t in args.towns.split(","):
        if (out / f"{t}_furniture.npz").exists():
            print(f"{t}: exists, skip")
            continue
        dump_town(c, t, TOWNS[t], out)
    print("FURNITURE_DUMP_DONE")


if __name__ == "__main__":
    main()
