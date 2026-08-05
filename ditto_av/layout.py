"""v0.3.1: static drivable-area layout (per-town lane geometry).

The v0.3 W3 diagnosis: the training world modeled only vehicles, so
boldness near static obstacles was free in training and punished in
CARLA (layout collisions 7-vs-1). This module supplies the missing
geometry: per-town lane centers + half-widths (extracted from OpenDRIVE
by scripts/v031_extract_layout.py), queried through a uniform grid hash
— dependency-free numpy here, mirrored in torch for the sim.

Query semantics: off(x, y) = dist(nearest lane center) - half_width
- margin; positive => off-drivable. Validated against real expert
trajectories (gate: expert violation < 1% of frames).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LAYOUT_DIR = Path("/scratch/srahmani/ditto_av/data/layout")
CELL = 4.0          # grid cell size, m (lane spacing is 1 m)
MARGIN = 0.5        # m beyond the lane edge before "off-drivable"


class TownLayout:
    def __init__(self, lanes: np.ndarray, cell: float = CELL):
        """lanes (N, 3): x, y, half_width."""
        self.pts = lanes[:, :2].astype(np.float32)
        self.hw = lanes[:, 2].astype(np.float32)
        self.cell = cell
        self.origin = self.pts.min(axis=0) - 2 * cell
        ij = np.floor((self.pts - self.origin) / cell).astype(np.int64)
        self.nx = int(ij[:, 0].max()) + 3
        self.ny = int(ij[:, 1].max()) + 3
        # bucket points by cell: CSR-style index
        flat = ij[:, 0] * self.ny + ij[:, 1]
        order = np.argsort(flat)
        self.sorted_idx = order
        self.cell_of = flat[order]
        # start offset per cell id via searchsorted at query time
        self._flat_sorted = self.cell_of

    def _cell_candidates(self, cx: int, cy: int) -> np.ndarray:
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = (cx + dx) * self.ny + (cy + dy)
                lo = np.searchsorted(self._flat_sorted, c, side="left")
                hi = np.searchsorted(self._flat_sorted, c, side="right")
                if hi > lo:
                    out.append(self.sorted_idx[lo:hi])
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def off_drivable(self, xy: np.ndarray,
                     margin: float = MARGIN) -> np.ndarray:
        """(M, 2) -> (M,) signed clearance beyond the lane edge
        (positive = off-drivable by that many meters). Points with no
        lane candidates within the 3x3 neighborhood (>~8 m from any
        lane) return +inf-ish large values."""
        xy = np.asarray(xy, dtype=np.float32)
        ij = np.floor((xy - self.origin) / self.cell).astype(np.int64)
        ij[:, 0] = np.clip(ij[:, 0], 1, self.nx - 2)
        ij[:, 1] = np.clip(ij[:, 1], 1, self.ny - 2)
        out = np.full(len(xy), 99.0, dtype=np.float32)
        for i, (p, (cx, cy)) in enumerate(zip(xy, ij)):
            cand = self._cell_candidates(int(cx), int(cy))
            if len(cand) == 0:
                continue
            d = np.linalg.norm(self.pts[cand] - p, axis=1)
            j = int(np.argmin(d))
            out[i] = d[j] - self.hw[cand[j]] - margin
        return out


def manifest_towns(manifest: Path, val_every: int = 6
                   ) -> Tuple[List[str], List[str]]:
    """Per-episode town names for the (train, val) npz caches.

    Mirrors run_b2d.split_clips exactly: sorted manifest names, every
    `val_every`-th held out; clips_to_npz concatenates clips in that
    order with one reset per clip, so episode i of each npz IS clip i
    of the corresponding split. Town10HD matches as Town10 (= the
    layout npz name).
    """
    names = sorted(ln.strip().removesuffix(".tar.gz")
                   for ln in Path(manifest).read_text().splitlines()
                   if ln.strip())
    towns = []
    for n in names:
        m = re.search(r"Town\d+", n)
        assert m, f"no TownNN in clip name {n!r}"
        towns.append(m.group(0))
    train = [t for i, t in enumerate(towns)
             if i % val_every != val_every - 1]
    val = [t for i, t in enumerate(towns)
           if i % val_every == val_every - 1]
    return train, val


_cache: Dict[str, TownLayout] = {}


def town_layout(town: str, layout_dir: Path = LAYOUT_DIR) -> TownLayout:
    if town not in _cache:
        z = np.load(layout_dir / f"{town}_lanes.npz")
        _cache[town] = TownLayout(z["lanes"])
    return _cache[town]
