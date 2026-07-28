#!/usr/bin/env python3
"""Definitive offline replay test of the deployment featurizer frame.

Rebuilds featurize_frame's inputs from a recorded clip's annotations the
same way DittoCarlaAgent.run_step builds them from the live world (ego
box yaw in degrees + yaw_offset, actor boxes, anno command points), then
diffs the result against load_clip's training observations. Whichever
yaw_offset reproduces the training obs is the correct deployment
convention — no CARLA, no rollout stochasticity involved.

Frames where the anno theta is NaN are excluded (load_clip carries the
previous heading forward there; the replay has no such state).

Usage: python scripts/replay_frame_check.py [clip_dir ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ditto_av.bench2drive import _load_frame, load_clip  # noqa: E402
from ditto_av.carla_agent import featurize_frame  # noqa: E402

DEFAULT_CLIPS = [
    "VanillaSignalizedTurnEncounterRedLight_Town03_Route101_Weather23",
    "HighwayCutIn_Town06_Route300_Weather14",
    "LaneChange_Town06_Route307_Weather21",
    "Accident_Town03_Route101_Weather23",
    "ParkingExit_Town12_Route1266_Weather12",
]
EXTRACTED = Path("/scratch/srahmani/ditto_av/data/bench2drive/extracted")


def replay_clip(clip_dir: Path, yaw_offset: float) -> np.ndarray:
    """Run the deployment featurizer over a recorded clip."""
    frames = sorted((clip_dir / "anno").glob("*.json.gz"))
    prev: dict = {}
    rows = []
    for p in frames:
        fr = _load_frame(p)
        ego_xy = np.array([fr["x"], fr["y"]], dtype=np.float64)
        ego_box = next(b for b in fr["bounding_boxes"]
                       if b["class"] == "ego_vehicle")
        raw_yaw = float(np.deg2rad(ego_box["rotation"][2]))
        actors = [(b["id"], np.array(b["location"][:2], dtype=np.float64),
                   float(np.deg2rad(b["rotation"][2])))
                  for b in fr.get("bounding_boxes", [])
                  if b.get("class") in ("vehicle", "walker", "bicycle")]
        route = {"near_xy": None, "near_cmd": 4,
                 "far_xy": None, "far_cmd": 4}
        for tag in ("near", "far"):
            x, y = fr.get(f"x_command_{tag}"), fr.get(f"y_command_{tag}")
            if x is not None and y is not None \
                    and np.isfinite([x, y]).all():
                route[f"{tag}_xy"] = np.array([x, y], dtype=np.float64)
            route[f"{tag}_cmd"] = int(fr.get(f"command_{tag}", 4) or 4)
        obs, prev = featurize_frame(ego_xy, raw_yaw + yaw_offset,
                                    float(fr["speed"]), actors, prev,
                                    route=route)
        rows.append(obs)
    return np.stack(rows)


def main():
    clips = [Path(a) for a in sys.argv[1:]] \
        or [EXTRACTED / c for c in DEFAULT_CLIPS]
    clips = [c for c in clips if (c / "anno").is_dir()]
    print(f"{len(clips)} clips | comparing replayed obs vs load_clip "
          f"(route layout, 65 dims)\n")
    header = (f"{'clip':46s} {'frames':>6s} "
              f"{'offset=pi/2':>22s} {'offset=0':>22s}")
    print(header)
    print(f"{'':46s} {'':>6s} {'mean|max abs diff':>22s} "
          f"{'mean|max abs diff':>22s}")
    agg = {np.pi / 2: [], 0.0: []}
    for clip in clips:
        offline = load_clip(clip, with_route=True)["obs"]
        theta_ok = np.array([np.isfinite(float(_load_frame(p)["theta"]))
                             for p in sorted((clip / "anno")
                                             .glob("*.json.gz"))])
        cols = []
        for off in (np.pi / 2, 0.0):
            replay = replay_clip(clip, off)
            diff = np.abs(replay[theta_ok] - offline[theta_ok])
            agg[off].append(diff.reshape(-1))
            cols.append(f"{diff.mean():.5f}|{diff.max():.4f}")
        print(f"{clip.name[:46]:46s} {int(theta_ok.sum()):6d} "
              f"{cols[0]:>22s} {cols[1]:>22s}")
    print("\nOVERALL")
    for off, name in ((np.pi / 2, "pi/2 (compass fix)"),
                      (0.0, "0    (raw yaw)")):
        d = np.concatenate(agg[off])
        print(f"  yaw_offset {name}: mean {d.mean():.6f}  "
              f"p99 {np.percentile(d, 99):.5f}  max {d.max():.4f}")


if __name__ == "__main__":
    main()
