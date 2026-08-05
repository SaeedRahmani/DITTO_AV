#!/usr/bin/env python3
"""v0.3.1 M1: OpenDRIVE -> compact drivable-area geometry per town.

carla.Map parses .xodr WITHOUT a server. We sample lane centers every
1 m with lane half-widths; the layout query is then: distance from the
nearest lane center minus its half-width (+margin) — positive means
off-drivable. Compact (float32 npz per town), fast (KD-tree or grid at
load time), and validated against real expert trajectories (the expert
drove on roads: violations must be ~0 or the REPRESENTATION is wrong).

Usage: v031_extract_layout.py [--xodr-dir ...] [--out ...] [--step 1.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    import carla
    ap = argparse.ArgumentParser()
    ap.add_argument("--xodr-dir",
                    default="/scratch/srahmani/ditto_av/data/layout/xodr")
    ap.add_argument("--out",
                    default="/scratch/srahmani/ditto_av/data/layout")
    ap.add_argument("--step", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    for xodr in sorted(Path(args.xodr_dir).glob("Town*.xodr")):
        town = xodr.stem
        dst = out / f"{town}_lanes.npz"
        if dst.exists():
            print(f"{town}: exists, skip")
            continue
        m = carla.Map(town, xodr.read_text())
        wps = m.generate_waypoints(args.step)
        rows = np.array(
            [[w.transform.location.x, w.transform.location.y,
              max(0.5 * w.lane_width, 0.1)]
             for w in wps
             if w.lane_type == carla.LaneType.Driving],
            dtype=np.float32)
        np.savez_compressed(dst, lanes=rows)
        print(f"{town}: {len(rows)} lane points -> {dst.name}")


if __name__ == "__main__":
    main()
