#!/usr/bin/env python3
"""2D BEV video from a CARLA run's state.jsonl (the same run whose
chase-cam frames make the 3D video — paired views of one scenario).

cv2-only (no matplotlib): draws oriented boxes on an ego-centered
window. cv2's y-down image frame matches CARLA's top-down chirality
directly (verified v0.2, T1 audit) — no flip needed.

Usage: render_bev_video.py state.jsonl out.mp4 [--view 45] [--px 720]
"""
from __future__ import annotations

import argparse
import json
import math

import cv2
import numpy as np

BG = (28, 28, 28)
ACTOR = (150, 150, 150)
EGO = (60, 200, 90)      # BGR green
TRAIL = (60, 200, 90)


def box_pts(x, y, yaw, ex, ey, to_px):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx, dy in ((ex, ey), (ex, -ey), (-ex, -ey), (-ex, ey)):
        wx = x + c * dx - s * dy
        wy = y + s * dx + c * dy
        pts.append(to_px(wx, wy))
    return np.array(pts, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--view", type=float, default=45.0)
    ap.add_argument("--px", type=int, default=720)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    ticks = [json.loads(l) for l in open(args.state) if l.strip()]
    if not ticks:
        raise SystemExit("empty state log")
    W = args.px
    scale = W / (2 * args.view)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, W))
    trail = []
    for k, tk in enumerate(ticks):
        x0, y0, eyaw, espeed = tk["ego"]
        trail.append((x0, y0))

        def to_px(wx, wy, x0=x0, y0=y0):
            return (int((wx - x0) * scale + W / 2),
                    int((wy - y0) * scale + W / 2))

        img = np.full((W, W, 3), BG, dtype=np.uint8)
        for aid, x, y, yaw, bx, by in tk["actors"]:
            if abs(x - x0) > args.view + 5 or abs(y - y0) > args.view + 5:
                continue
            cv2.fillPoly(img, [box_pts(x, y, yaw, bx, by, to_px)], ACTOR)
        if len(trail) > 1:
            pts = np.array([to_px(px_, py_) for px_, py_ in trail[-300:]],
                           dtype=np.int32)
            cv2.polylines(img, [pts], False, TRAIL, 2, cv2.LINE_AA)
        cv2.fillPoly(img, [box_pts(x0, y0, eyaw, 2.45, 1.06, to_px)],
                     EGO)
        cv2.putText(img, f"t={k / args.fps:5.1f}s  v={espeed:4.1f} m/s",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 230, 230), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print(f"wrote {args.out} ({len(ticks)} frames)")


if __name__ == "__main__":
    main()
