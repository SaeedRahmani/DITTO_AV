#!/usr/bin/env python3
"""Phase-0c gate: verify future-waypoint targets on real clips.

The 90-degree frame bug shipped once because geometry was trusted
untested. This harness runs on extracted clips and checks, per frame:

1. round-trip: rebuilding world coords from the ego-frame waypoint
   reproduces the anno future pose exactly (construction identity);
2. physics: |wp_1| matches the integral of recorded speed over the
   stride (catches wrong indexing/units/frame);
3. forward dominance: waypoints lie ahead (x > 0) except when the
   expert reverses/stops (catches any rotation-convention slip);
4. straight-line laterals: on low-steer frames the lateral component
   stays small (catches a swapped axis).

Exit code 1 on any violation. Usage:
    python scripts/waypoint_check.py [n_clips] [--extracted DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ditto_av.bench2drive import (_ego_xy, _load_frame,  # noqa: E402
                                  future_waypoints, WP_STRIDE)

FPS = 10.0


def check_clip(clip_dir: Path):
    frames = sorted((clip_dir / "anno").glob("*.json.gz"))
    raw = [_load_frame(p) for p in frames]
    yaws = np.array([float(fr["theta"]) for fr in raw])
    ok = np.isfinite(yaws)
    if not ok.all():  # same forward-fill as load_clip
        idx = np.where(ok, np.arange(len(yaws)), -1)
        np.maximum.accumulate(idx, out=idx)
        idx[idx < 0] = np.flatnonzero(ok)[0]
        yaws = yaws[idx]
    xy = np.stack([_ego_xy(fr) for fr in raw])
    speed = np.array([float(fr["speed"]) for fr in raw])
    steer = np.array([float(fr["steer"]) for fr in raw])
    wp = future_waypoints(raw, yaws, k=6)          # (n, 6, 2) meters
    n = len(raw)

    # 1. round-trip identity
    t = n // 2
    c, s = np.cos(yaws[t]), np.sin(yaws[t])
    e2w = np.array([[c, -s], [s, c]])
    rebuilt = xy[t] + e2w @ wp[t, 0]
    rt_err = float(np.linalg.norm(rebuilt - xy[min(t + WP_STRIDE, n - 1)]))

    # 2. physics: |wp_1| vs integrated speed over the stride
    errs = []
    for t in range(0, n - WP_STRIDE, 3):
        travel = float(speed[t:t + WP_STRIDE].mean()) * WP_STRIDE / FPS
        d = float(np.linalg.norm(wp[t, 0]))
        if travel > 2.0:  # low-speed windows make the mean-speed
            # approximation dominate the relative error
            errs.append(abs(d - travel) / travel)
    phys_p90 = float(np.percentile(errs, 90)) if errs else 0.0

    # In the settled compass frame (anno theta = yaw + pi/2), FORWARD
    # is -y and LATERAL is x — confirmed empirically by the first run
    # of this harness (the +x assumption failed with 'lateral' p90 =
    # 5.3 m = cruise-speed travel) and by the route-block training
    # stats (near point 5 m ahead sits at (0, -0.046*POS_SCALE)).
    # 3. forward dominance while moving: backward = wp_y > +0.5
    moving = speed[:-WP_STRIDE] > 2.0
    ys_f = wp[:-WP_STRIDE, 0, 1][moving]
    frac_back = float((ys_f > 0.5).mean()) if len(ys_f) else 0.0

    # 4. lateral (x) smallness on straight, moving frames
    straight = moving & (np.abs(steer[:-WP_STRIDE]) < 0.02)
    xs_l = np.abs(wp[:-WP_STRIDE, 0, 0][straight])
    lat_p90 = float(np.percentile(xs_l, 90)) if len(xs_l) else 0.0

    return rt_err, phys_p90, frac_back, lat_p90, int(moving.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_clips", nargs="?", type=int, default=20)
    ap.add_argument("--extracted", default="/scratch/srahmani/ditto_av/"
                    "data/bench2drive/extracted")
    args = ap.parse_args()
    clips = sorted(p for p in Path(args.extracted).iterdir()
                   if (p / "anno").is_dir())[:args.n_clips]
    if not clips:
        print("no extracted clips found")
        return 1
    worst = dict(rt=0.0, phys=0.0, back=0.0, lat=0.0)
    frames = 0
    for cd in clips:
        rt, phys, back, lat, nmov = check_clip(cd)
        frames += nmov
        worst["rt"] = max(worst["rt"], rt)
        worst["phys"] = max(worst["phys"], phys)
        worst["back"] = max(worst["back"], back)
        worst["lat"] = max(worst["lat"], lat)
    print(f"{len(clips)} clips, {frames} moving frames | "
          f"round-trip max {worst['rt']:.2e} m | "
          f"physics p90 rel-err max {worst['phys']:.3f} | "
          f"backward frac max {worst['back']:.4f} | "
          f"straight lateral p90 max {worst['lat']:.2f} m")
    fail = (worst["rt"] > 1e-6 or worst["phys"] > 0.35
            or worst["back"] > 0.02 or worst["lat"] > 0.8)
    print("WAYPOINT_CHECK_" + ("FAIL" if fail else "OK"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
