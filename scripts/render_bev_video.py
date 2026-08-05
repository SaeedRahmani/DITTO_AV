#!/usr/bin/env python3
"""2D BEV video from a CARLA run's state.jsonl, drawn over the town map.

The same run whose chase-cam frames make the 3D video — paired views of
one scenario. Until now the BEV had no world under it: boxes floating on
flat grey, with no way to see which lane the ego was in or where the road
went. The map comes from the town's OpenDRIVE, parsed by `carla.Map`
WITHOUT a server (the v0.3.1 layout trick), cached per town as a compact
npz of lane samples: centres, headings, half-widths, lane ids. Lane
groups are re-assembled into polylines and painted as road surface plus
lane-edge markings; static furniture (buildings, walls, fences) is
overlaid from the v0.3.1 furniture dump when present.

cv2-only. cv2's y-down image frame matches CARLA's top-down chirality
directly (verified v0.2, T1 audit) — no flip needed.

Town resolution, in order: --town, the `meta` line the agent writes at
the head of state.jsonl, then auto-detection against the cached town
extents (old logs predate the meta line).

Usage:
  render_bev_video.py state.jsonl out.mp4 [--town Town03] [--view 55]
                      [--px 720] [--label "v0.3 clp_rx"]
  render_bev_video.py --build-cache            # (re)build town caches
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

LAYOUT_DIR = Path(os.environ.get(
    "DITTO_LAYOUT_DIR",
    f"/scratch/{os.environ.get('USER', 'srahmani')}/ditto_av/data/layout"))
CACHE_SUFFIX = "_bev.npz"
EXTENTS = "bev_extents.json"

# BGR
BG = (24, 27, 25)          # off-map terrain
C_SIDEWALK = (74, 76, 78)
C_SHOULDER = (48, 48, 50)
C_PARKING = (60, 60, 62)
C_ROAD = (56, 56, 58)
C_JUNCTION = (66, 66, 68)
C_MARK = (175, 175, 175)
C_BUILDING = (46, 42, 40)
C_PROP = (58, 66, 78)
ACTOR = (150, 150, 150)
ACTOR_EDGE = (205, 205, 205)
EGO = (60, 200, 90)
TRAIL = (60, 200, 90)
HUD = (230, 230, 230)

# lane types kept, painted back-to-front in this order
LANE_TYPES = ["Sidewalk", "Median", "Border", "Shoulder", "Parking",
              "Bidirectional", "Driving"]
LANE_COLOR = [C_SIDEWALK, (50, 56, 50), (56, 54, 52), C_SHOULDER,
              C_PARKING, C_ROAD, C_ROAD]
DRIVABLE = LANE_TYPES.index("Bidirectional")   # >= this = carriageway
STEP = 1.0                 # lane sampling, m
GAP = 3.0                  # polyline break when consecutive samples jump
SIDE_LANES = 3             # how far to walk sideways off the driving lane


# --------------------------------------------------------------- cache
def build_cache(town: str, xodr: Path, dst: Path, step: float = STEP):
    """OpenDRIVE -> lane-sample npz. Needs the carla wheel, no server.

    generate_waypoints only yields Driving lanes, so each one also walks
    sideways to pick up the sidewalks/shoulders/parking that give the
    render its kerbs — deduplicated on (road, section, lane, s).
    """
    import carla
    m = carla.Map(town, xodr.read_text())
    keep = {t: i for i, t in enumerate(LANE_TYPES)}
    rows, seen = [], set()

    def add(w):
        t = keep.get(str(w.lane_type))
        if t is None:
            return
        key = (w.road_id, w.section_id, w.lane_id, round(w.s, 1))
        if key in seen:
            return
        seen.add(key)
        tr = w.transform
        rows.append((tr.location.x, tr.location.y,
                     math.radians(tr.rotation.yaw),
                     max(0.5 * w.lane_width, 0.1),
                     w.road_id, w.section_id, w.lane_id, w.s,
                     int(w.is_junction), t, tr.location.z))

    for w in m.generate_waypoints(step):
        add(w)
        for side in ("get_left_lane", "get_right_lane"):
            cur = w
            for _ in range(SIDE_LANES):
                nxt = getattr(cur, side)()
                if nxt is None:
                    break
                add(nxt)
                cur = nxt
    a = np.array(rows, dtype=np.float64)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        xy=a[:, :2].astype(np.float32), yaw=a[:, 2].astype(np.float32),
        hw=a[:, 3].astype(np.float32), road=a[:, 4].astype(np.int32),
        sect=a[:, 5].astype(np.int16), lane=a[:, 6].astype(np.int16),
        s=a[:, 7].astype(np.float32), junc=a[:, 8].astype(np.uint8),
        ltype=a[:, 9].astype(np.uint8), z=a[:, 10].astype(np.float32))
    return len(rows), a[:, :2].min(0), a[:, :2].max(0)


def build_all(layout_dir: Path = LAYOUT_DIR, force: bool = False):
    ext_path = layout_dir / EXTENTS
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else {}
    for xodr in sorted((layout_dir / "xodr").glob("Town*.xodr")):
        town = xodr.stem
        dst = layout_dir / f"{town}{CACHE_SUFFIX}"
        if dst.exists() and not force and town in ext:
            print(f"{town}: cached")
            continue
        n, lo, hi = build_cache(town, xodr, dst)
        ext[town] = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
        print(f"{town}: {n} lane samples -> {dst.name}")
    ext_path.write_text(json.dumps(ext, indent=1, sort_keys=True))


# ----------------------------------------------------------------- map
class TownMap:
    """Lane samples re-assembled into drawable polylines."""

    def __init__(self, npz: Path):
        d = np.load(npz)
        order = np.lexsort((d["s"], d["lane"], d["sect"], d["road"],
                            d["ltype"]))
        self.xy = d["xy"][order]
        self.yaw = d["yaw"][order]
        self.hw = d["hw"][order]
        junc = d["junc"][order]
        ltype = d["ltype"][order]
        key = np.column_stack([d[k][order] for k in
                               ("ltype", "road", "sect", "lane")])
        cut = np.any(key[1:] != key[:-1], axis=1)
        cut |= np.linalg.norm(np.diff(self.xy, axis=0), axis=1) > GAP
        starts = np.concatenate(([0], np.flatnonzero(cut) + 1))
        ends = np.concatenate((starts[1:], [len(self.xy)]))
        keep = (ends - starts) >= 2
        self.starts, self.ends = starts[keep], ends[keep]
        g = np.arange(len(self.starts))
        self.g_ltype = ltype[self.starts]
        self.g_junc = junc[self.starts].astype(bool)
        self.g_hw = np.array([np.median(self.hw[self.starts[i]:self.ends[i]])
                              for i in g], dtype=np.float32)
        z = d["z"][order]
        self.z = z
        self.g_z = np.array([np.median(z[self.starts[i]:self.ends[i]])
                             for i in g], dtype=np.float32)
        self.bmin = np.array([self.xy[self.starts[i]:self.ends[i]].min(0)
                              for i in g], dtype=np.float32)
        self.bmax = np.array([self.xy[self.starts[i]:self.ends[i]].max(0)
                              for i in g], dtype=np.float32)
        # unit lateral of each sample, for lane-edge markings
        self.nrm = np.column_stack((-np.sin(self.yaw), np.cos(self.yaw)))
        self.furniture = None

    def load_furniture(self, path: Path, max_z: float = 12.0):
        """Static collidables (v0.3.1 dump): centre, extent, yaw, label."""
        if not path.exists():
            return
        d = np.load(path)
        c, e, y = d["center"], d["extent"], d["yaw"]
        lab = d["label"]
        # the dump merges two sweeps; drop near-duplicates
        k = np.unique(np.round(np.column_stack((c, e)), 1),
                      axis=0, return_index=True)[1]
        k = np.sort(k)
        self.furniture = (c[k, :2], e[k, :2], y[k], lab[k],
                          [str(s) for s in d["labels"]], c[k, 2])

    def _visible(self, lo, hi):
        return np.flatnonzero((self.bmin[:, 0] <= hi[0]) &
                              (self.bmax[:, 0] >= lo[0]) &
                              (self.bmin[:, 1] <= hi[1]) &
                              (self.bmax[:, 1] >= lo[1]))

    def ground_z(self, x0, y0, radius=12.0):
        """Ego z from the nearest driving-lane sample (state logs are
        2D). Towns 03/11/12/13 stack roads on overpasses — without a
        level the BEV paints every deck at once."""
        near = self._visible((x0 - radius, y0 - radius),
                             (x0 + radius, y0 + radius))
        near = near[self.g_ltype[near] >= DRIVABLE]
        best, best_d = None, radius ** 2
        for i in near:
            p = self.xy[self.starts[i]:self.ends[i]]
            d2 = ((p[:, 0] - x0) ** 2 + (p[:, 1] - y0) ** 2)
            j = int(np.argmin(d2))
            if d2[j] < best_d:
                best_d = float(d2[j])
                best = float(self.z[self.starts[i] + j])
        return best

    def draw(self, img, x0, y0, view, scale, W, z0=None, dz=6.0):
        lo = (x0 - view - 8, y0 - view - 8)
        hi = (x0 + view + 8, y0 + view + 8)
        vis = self._visible(lo, hi)
        if z0 is not None:
            vis = vis[np.abs(self.g_z[vis] - z0) <= dz]

        def px(a):
            return np.column_stack(((a[:, 0] - x0) * scale + W / 2,
                                    (a[:, 1] - y0) * scale + W / 2)
                                   ).astype(np.int32)

        # surfaces, back to front; junctions repaint over lane bands so
        # the intersection reads as one plate instead of crossing strips
        for t in range(len(LANE_TYPES)):
            sel = vis[self.g_ltype[vis] == t]
            for i in sel:
                s, e = self.starts[i], self.ends[i]
                th = max(1, int(round(2 * self.g_hw[i] * scale)))
                col = C_JUNCTION if (self.g_junc[i] and t >= DRIVABLE) \
                    else LANE_COLOR[t]
                cv2.polylines(img, [px(self.xy[s:e])], False, col, th)
        # lane edges (skipped through junctions, as on a real road)
        for i in vis:
            if self.g_ltype[i] < DRIVABLE or self.g_junc[i]:
                continue
            s, e = self.starts[i], self.ends[i]
            c, n, h = self.xy[s:e], self.nrm[s:e], self.hw[s:e, None]
            cv2.polylines(img, [px(c + n * h), px(c - n * h)], False,
                          C_MARK, 1, cv2.LINE_AA)
        if self.furniture is not None:
            self._draw_furniture(img, x0, y0, view, scale, W, z0)

    def _draw_furniture(self, img, x0, y0, view, scale, W, z0=None):
        c, e, yaw, lab, names, cz = self.furniture
        m = ((np.abs(c[:, 0] - x0) < view + 15) &
             (np.abs(c[:, 1] - y0) < view + 15))
        if z0 is not None:
            m &= np.abs(cz - z0) < 10.0
        for ci, ei, yi, li in zip(c[m], e[m], yaw[m], lab[m]):
            name = names[int(li)] if int(li) < len(names) else ""
            col = C_BUILDING if name in ("Buildings", "Walls") else C_PROP
            cv2.fillPoly(img, [box_pts(ci[0], ci[1], float(yi),
                                       max(float(ei[0]), 0.3),
                                       max(float(ei[1]), 0.3),
                                       lambda wx, wy: (
                                           int((wx - x0) * scale + W / 2),
                                           int((wy - y0) * scale + W / 2)))],
                         col)


def resolve_town(name: str) -> str:
    """`Carla/Maps/Town10HD_Opt` -> the cached `Town10` geometry."""
    n = name.strip().split("/")[-1].replace("_Opt", "")
    return "Town10" if n.startswith("Town10") else n


def detect_town(ego: np.ndarray, layout_dir: Path) -> str | None:
    """Old logs carry no town: pick the map whose lanes fit the drive."""
    ext_path = layout_dir / EXTENTS
    if not ext_path.exists():
        return None
    lo, hi = ego.min(0), ego.max(0)
    cands = [t for t, b in json.loads(ext_path.read_text()).items()
             if b[0] - 30 <= lo[0] and b[1] - 30 <= lo[1]
             and b[2] + 30 >= hi[0] and b[3] + 30 >= hi[1]]
    best, best_d = None, 1e9
    probe = ego[np.linspace(0, len(ego) - 1, min(40, len(ego))).astype(int)]
    for t in cands:
        p = layout_dir / f"{t}{CACHE_SUFFIX}"
        if not p.exists():
            continue
        xy = np.load(p)["xy"]
        m = ((np.abs(xy[:, 0] - probe[:, 0].mean()) < 400) &
             (np.abs(xy[:, 1] - probe[:, 1].mean()) < 400))
        if not m.any():
            continue
        sub = xy[m]
        d = np.median([np.min(np.linalg.norm(sub - q, axis=1))
                       for q in probe])
        if d < best_d:
            best, best_d = t, d
    return best if best_d < 4.0 else None


# -------------------------------------------------------------- render
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
    ap.add_argument("state", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--view", type=float, default=55.0)
    ap.add_argument("--px", type=int, default=720)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--town", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--layout-dir", default=str(LAYOUT_DIR))
    ap.add_argument("--furniture", action="store_true",
                    help="overlay the static-collidable dump (busy)")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    layout = Path(args.layout_dir)
    if args.build_cache:
        build_all(layout, force=args.force)
        return
    if not args.state or not args.out:
        raise SystemExit("need state.jsonl and out.mp4")

    ticks, meta = [], {}
    for line in open(args.state):
        if not line.strip():
            continue
        r = json.loads(line)
        (meta.update(r["meta"]) if "meta" in r else ticks.append(r))
    if not ticks:
        raise SystemExit("empty state log")

    ego_xy = np.array([t["ego"][:2] for t in ticks], dtype=np.float32)
    town = args.town or meta.get("town") or detect_town(ego_xy, layout)
    tmap = None
    if town:
        p = layout / f"{resolve_town(town)}{CACHE_SUFFIX}"
        if p.exists():
            tmap = TownMap(p)
            if args.furniture:
                tmap.load_furniture(layout / "furniture" /
                                    f"{resolve_town(town)}_furniture.npz")
        else:
            print(f"no map cache for {town} ({p}) — rendering without map")
    else:
        print("town unresolved — rendering without map")

    W = args.px
    scale = W / (2 * args.view)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, W))
    trail, z_level = [], None
    for k, tk in enumerate(ticks):
        x0, y0, eyaw, espeed = tk["ego"][:4]
        trail.append((x0, y0))

        def to_px(wx, wy, x0=x0, y0=y0):
            return (int((wx - x0) * scale + W / 2),
                    int((wy - y0) * scale + W / 2))

        img = np.full((W, W, 3), BG, dtype=np.uint8)
        if tmap is not None:
            z0 = tk["ego"][4] if len(tk["ego"]) > 4 else \
                tmap.ground_z(x0, y0)
            if z0 is not None:
                z_level = z0
            tmap.draw(img, x0, y0, args.view, scale, W, z0=z_level)
        for aid, x, y, yaw, bx, by in tk["actors"]:
            if abs(x - x0) > args.view + 5 or abs(y - y0) > args.view + 5:
                continue
            box = box_pts(x, y, yaw, bx, by, to_px)
            cv2.fillPoly(img, [box], ACTOR)
            cv2.polylines(img, [box], True, ACTOR_EDGE, 1, cv2.LINE_AA)
        if len(trail) > 1:
            pts = np.array([to_px(px_, py_) for px_, py_ in trail[-300:]],
                           dtype=np.int32)
            cv2.polylines(img, [pts], False, TRAIL, 2, cv2.LINE_AA)
        cv2.fillPoly(img, [box_pts(x0, y0, eyaw, 2.45, 1.06, to_px)], EGO)
        cv2.putText(img, f"t={k / args.fps:5.1f}s  v={espeed:4.1f} m/s",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD, 2,
                    cv2.LINE_AA)
        cap = " | ".join(x for x in (args.label, town or "") if x)
        if cap:
            cv2.putText(img, cap, (12, W - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, HUD, 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print(f"wrote {args.out} ({len(ticks)} frames, town={town})")


if __name__ == "__main__":
    main()
