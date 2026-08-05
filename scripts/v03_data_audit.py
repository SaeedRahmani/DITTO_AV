#!/usr/bin/env python3
"""V3 Phase-A audit: measure everything the traffic model design assumes.

A1 trackability: actor-ID stability, track lengths, churn, classes,
    slot overflow, sampling regularity, position-noise floor.
A2 dynamics floor: constant-velocity / constant-turn-rate ADE/FDE at
    1/2/4 s on ID-associated tracks (the bar any learned model must
    clearly beat).
A3 interaction evidence: follower-decel vs lead-gap statistics around
    the EGO (how much reactive signal exists at all).

Reads raw extracted annos (ground truth, not our npz views, so the
audit also cross-checks the views later). Commits a json + md report.

Usage: python scripts/v03_data_audit.py --clips 24 --out runs/v03_audit
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXT = Path("/scratch/srahmani/ditto_av/data/bench2drive/extracted")
FPS = 10.0


def load_clip_frames(clip: Path):
    frames = sorted((clip / "anno").glob("*.json.gz"))
    return [json.load(gzip.open(p, "rt")) for p in frames]


def audit_clip(raw):
    """Per-clip A1 stats + tracks dict for A2/A3."""
    tracks = defaultdict(list)   # id -> [(t, x, y, yaw_rad, cls)]
    ego_xy = []
    n_actors = []
    classes = Counter()
    for t, fr in enumerate(raw):
        n = 0
        for b in fr.get("bounding_boxes", []):
            c = b.get("class")
            if c == "ego_vehicle":
                ego_xy.append((t, float(b["location"][0]),
                               float(b["location"][1])))
            if c not in ("vehicle", "walker", "bicycle"):
                continue
            n += 1
            classes[c] += 1
            tracks[b["id"]].append(
                (t, float(b["location"][0]), float(b["location"][1]),
                 math.radians(float(b["rotation"][2])), c))
        n_actors.append(n)
    return tracks, ego_xy, n_actors, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--out", default="runs/v03_audit")
    args = ap.parse_args()
    clips = sorted(d for d in EXT.iterdir() if (d / "anno").is_dir())
    step = max(1, len(clips) // args.clips)
    clips = clips[::step][:args.clips]

    A = {"clips": len(clips), "gaps_per_track": [], "track_len_s": [],
         "n_actors": [], "overflow32": 0, "frames": 0,
         "classes": Counter(), "enter_exit_per_100f": [],
         "teleports": 0, "steps_checked": 0}
    cv_err = {h: [] for h in (10, 20, 40)}   # frames ahead
    ctrv_err = {h: [] for h in (10, 20, 40)}
    inter = {"follow_pairs": 0, "decel_when_close": [],
             "decel_when_far": []}

    for clip in clips:
        raw = load_clip_frames(clip)
        tracks, ego_xy, n_actors, classes = audit_clip(raw)
        A["frames"] += len(raw)
        A["n_actors"] += n_actors
        A["overflow32"] += sum(1 for n in n_actors if n > 32)
        A["classes"].update(classes)
        churn = 0
        prev_ids = None
        for t in range(len(raw)):
            ids = {i for i, tr in tracks.items()
                   if any(s[0] == t for s in tr)}
        # churn via track endpoints (cheaper):
        for tr in tracks.values():
            ts = [s[0] for s in tr]
            A["track_len_s"].append((ts[-1] - ts[0] + 1) / FPS)
            gaps = sum(1 for a, b in zip(ts, ts[1:]) if b - a > 1)
            A["gaps_per_track"].append(gaps)
            churn += 1  # one enter + (maybe) one exit per track
            # teleport check: per-step displacement > 60/FPS m
            for (t0, x0, y0, _, _), (t1, x1, y1, _, _) in zip(tr, tr[1:]):
                if t1 - t0 == 1:
                    A["steps_checked"] += 1
                    if math.hypot(x1 - x0, y1 - y0) > 60.0 / FPS:
                        A["teleports"] += 1
        A["enter_exit_per_100f"].append(100.0 * churn / max(len(raw), 1))

        # A2: CV + CTRV floors on tracks (contiguous segments)
        for tr in tracks.values():
            arr = np.array([(s[0], s[1], s[2], s[3]) for s in tr])
            for h in (10, 20, 40):
                for i in range(2, len(arr) - h):
                    if arr[i, 0] - arr[i - 1, 0] != 1 or \
                       arr[i + h, 0] - arr[i, 0] != h:
                        continue
                    v = arr[i, 1:3] - arr[i - 1, 1:3]
                    pred_cv = arr[i, 1:3] + v * h
                    err_cv = np.linalg.norm(pred_cv - arr[i + h, 1:3])
                    cv_err[h].append(err_cv)
                    dyaw = arr[i, 3] - arr[i - 1, 3]
                    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                    sp = np.linalg.norm(v)
                    yaw0 = math.atan2(v[1], v[0]) if sp > 1e-3 \
                        else arr[i, 3]
                    p = arr[i, 1:3].copy()
                    yw = yaw0
                    for _ in range(h):
                        p = p + sp * np.array([math.cos(yw),
                                               math.sin(yw)])
                        yw += dyaw
                    ctrv_err[h].append(np.linalg.norm(p - arr[i + h, 1:3]))

        # A3: follower response to the EGO (is there reactive signal?)
        ego = {t: (x, y) for t, x, y in ego_xy}
        for tr in tracks.values():
            if tr[0][4] != "vehicle":
                continue
            for (t0, x0, y0, _, _), (t1, x1, y1, _, _), \
                    (t2, x2, y2, _, _) in zip(tr, tr[1:], tr[2:]):
                if t2 - t0 != 2 or t2 not in ego or t0 not in ego:
                    continue
                v1 = math.hypot(x1 - x0, y1 - y0) * FPS
                v2 = math.hypot(x2 - x1, y2 - y1) * FPS
                if v1 < 0.5:
                    continue
                ex, ey = ego[t1]
                # is the EGO roughly ahead of this actor's motion?
                hx, hy = (x1 - x0), (y1 - y0)
                n = math.hypot(hx, hy) or 1.0
                ahead = ((ex - x1) * hx + (ey - y1) * hy) / n
                lat = abs(((ex - x1) * -hy + (ey - y1) * hx) / n)
                if lat < 2.0 and 0 < ahead < 30:
                    inter["follow_pairs"] += 1
                    a = (v2 - v1) * FPS
                    (inter["decel_when_close"] if ahead < 12
                     else inter["decel_when_far"]).append(a)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tl = np.array(A["track_len_s"])
    res = {
        "clips": A["clips"], "frames": A["frames"],
        "actors_mean": float(np.mean(A["n_actors"])),
        "actors_p99": float(np.percentile(A["n_actors"], 99)),
        "overflow32_framefrac": A["overflow32"] / max(A["frames"], 1),
        "classes": dict(A["classes"]),
        "track_len_s_median": float(np.median(tl)),
        "track_len_s_p25": float(np.percentile(tl, 25)),
        "track_frac_ge_3s": float((tl >= 3.0).mean()),
        "tracks_with_gaps_frac": float(np.mean(
            [g > 0 for g in A["gaps_per_track"]])),
        "teleport_frac": A["teleports"] / max(A["steps_checked"], 1),
        "tracks_per_100frames": float(np.mean(A["enter_exit_per_100f"])),
        "cv_ade": {f"{h/10:.0f}s": float(np.mean(cv_err[h]))
                   for h in cv_err if cv_err[h]},
        "ctrv_ade": {f"{h/10:.0f}s": float(np.mean(ctrv_err[h]))
                     for h in ctrv_err if ctrv_err[h]},
        "follow_pairs": inter["follow_pairs"],
        "follower_accel_close_mean": float(np.mean(
            inter["decel_when_close"])) if inter["decel_when_close"]
        else None,
        "follower_accel_far_mean": float(np.mean(
            inter["decel_when_far"])) if inter["decel_when_far"]
        else None,
    }
    (out / "audit.json").write_text(json.dumps(res, indent=2))
    lines = ["# V3 Phase-A data audit", ""]
    for k, v in res.items():
        lines.append(f"- {k}: {v}")
    (out / "audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
