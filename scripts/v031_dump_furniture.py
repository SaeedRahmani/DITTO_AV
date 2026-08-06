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


# large streamed maps: furniture lives in tiles that only load near an
# actor; sweep the spectator over the road bbox to stream them all
SWEEP_TOWNS = {"Town11", "Town12", "Town13", "Town15"}
SWEEP_STEP = 1200.0     # m between spectator stops
SWEEP_SETTLE = 4.0      # s to let tiles stream in per stop


def _collect(w, seen, cen, ext, yaw, lab, src):
    import carla
    for li, lname in enumerate(LABELS):
        clab = getattr(carla.CityObjectLabel, lname)
        for o in w.get_environment_objects(clab):
            if ("e", o.id) in seen:
                continue
            seen.add(("e", o.id))
            b = o.bounding_box
            cen.append([b.location.x, b.location.y, b.location.z])
            ext.append([b.extent.x, b.extent.y, b.extent.z])
            yaw.append(b.rotation.yaw)
            lab.append(li)
            src.append(0)
        for b in w.get_level_bbs(clab):
            k = ("l", li, round(b.location.x, 1), round(b.location.y, 1),
                 round(b.location.z, 1), round(b.extent.x, 1))
            if k in seen:
                continue
            seen.add(k)
            cen.append([b.location.x, b.location.y, b.location.z])
            ext.append([b.extent.x, b.extent.y, b.extent.z])
            yaw.append(b.rotation.yaw)
            lab.append(li)
            src.append(1)


def dump_town(client, town_npz: str, town_carla: str, out: Path,
              band: str = "0:1"):
    import carla
    import time
    bi, bn = (int(v) for v in band.split(":"))
    dst = out / (f"{town_npz}_furniture.npz" if bn == 1 else
                 f"{town_npz}_furniture_band{bi}of{bn}.npz")
    if dst.exists():
        print(f"{dst.name}: exists, skip")
        return
    w = client.load_world(town_carla)
    cen, ext, yaw, lab, src = [], [], [], [], []
    seen: set = set()
    if town_npz in SWEEP_TOWNS:
        lanes = np.load(Path("/scratch/srahmani/ditto_av/data/layout")
                        / f"{town_npz}_lanes.npz")["lanes"]
        lo = lanes[:, :2].min(axis=0) - 200
        hi = lanes[:, :2].max(axis=0) + 200
        spec = w.get_spectator()
        xs = np.arange(lo[0], hi[0] + SWEEP_STEP, SWEEP_STEP)
        ys = np.arange(lo[1], hi[1] + SWEEP_STEP, SWEEP_STEP)
        xs = np.array_split(xs, bn)[bi]     # band = contiguous x cols
        print(f"{town_npz} band {bi}/{bn}: sweeping "
              f"{len(xs)}x{len(ys)} stops")
        for x in xs:
            for y in ys:
                # skip stops far from any lane (no road, no furniture
                # we care about; keeps the sweep and RAM bounded)
                d = np.abs(lanes[:, :2] - [x, y]).max(axis=1).min()
                if d > SWEEP_STEP:
                    continue
                spec.set_transform(carla.Transform(
                    carla.Location(x=float(x), y=float(y), z=200.0),
                    carla.Rotation(pitch=-90.0)))
                time.sleep(SWEEP_SETTLE)
                _collect(w, seen, cen, ext, yaw, lab, src)
    else:
        _collect(w, seen, cen, ext, yaw, lab, src)
    np.savez_compressed(
        dst,
        center=np.array(cen, dtype=np.float32),
        extent=np.array(ext, dtype=np.float32),
        yaw=np.array(yaw, dtype=np.float32),
        label=np.array(lab, dtype=np.int16),
        source=np.array(src, dtype=np.int8),   # 0=env_obj 1=level_bbs
        labels=np.array(LABELS))
    print(f"{dst.name}: {len(cen)} boxes "
          f"({sum(1 for s in src if s == 0)} env / "
          f"{sum(1 for s in src if s == 1)} level)")


def merge_bands(town: str, out: Path):
    """Merge band npzs into the final per-town file (vectorized
    rounded-key dedup across band overlaps; the per-row loop version
    OOMed on Town13's ~2.5M rows)."""
    parts = sorted(out.glob(f"{town}_furniture_band*.npz"))
    assert parts, f"no band files for {town}"
    ds = [np.load(p) for p in parts]
    cen = np.concatenate([d["center"] for d in ds])
    ext = np.concatenate([d["extent"] for d in ds])
    yaw = np.concatenate([d["yaw"] for d in ds])
    lab = np.concatenate([d["label"] for d in ds])
    src = np.concatenate([d["source"] for d in ds])
    key = np.stack([lab.astype(np.float64), src.astype(np.float64),
                    np.round(cen[:, 0], 1), np.round(cen[:, 1], 1),
                    np.round(cen[:, 2], 1), np.round(ext[:, 0], 1)], 1)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    np.savez_compressed(
        out / f"{town}_furniture.npz",
        center=cen[idx], extent=ext[idx], yaw=yaw[idx],
        label=lab[idx], source=src[idx], labels=ds[0]["labels"])
    print(f"{town}: merged {len(parts)} bands, "
          f"{len(cen)} -> {len(idx)} boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--towns", default=",".join(sorted(TOWNS)))
    ap.add_argument("--band", default="0:1")
    ap.add_argument("--merge", action="store_true",
                    help="merge band files (no server needed)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        for t in args.towns.split(","):
            if (out / f"{t}_furniture.npz").exists():
                print(f"{t}: final exists, skip")
                continue
            merge_bands(t, out)
        return
    import carla
    c = carla.Client("localhost", args.port)
    c.set_timeout(600.0)
    print("server:", c.get_server_version())
    for t in args.towns.split(","):
        if (out / f"{t}_furniture.npz").exists():
            print(f"{t}: exists, skip")
            continue
        dump_town(c, t, TOWNS[t], out, args.band)
    print("FURNITURE_DUMP_DONE")


if __name__ == "__main__":
    main()
