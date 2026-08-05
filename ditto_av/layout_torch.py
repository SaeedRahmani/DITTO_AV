"""v0.3.1 M2: torch mirror of the layout query, for use inside the sim.

TownLayoutTorch reuses the exact grid-hash arrays built by
layout.TownLayout (same origin/cell/CSR ordering), so the two paths are
numerically identical: nearest lane center by distance, then that
point's half-width + margin. The padded 3x3-neighborhood gather is
bounded by the densest cell (<= 27 points across all towns), so a
batched query is one gather + one argmin.

LayoutQuery adds the per-frame dispatch the sim needs: episode -> town
comes from the manifest split (layout.manifest_towns), expanded to a
per-frame town index over a GlobalLog's episode bounds.

layout.py stays torch-free on purpose (it must import in the CARLA eval
env); this module is the training-side counterpart.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .layout import LAYOUT_DIR, MARGIN, TownLayout, town_layout

_NEIGH = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


class TownLayoutTorch:
    def __init__(self, base: TownLayout, device: str = "cpu"):
        self.pts = torch.as_tensor(base.pts, device=device)
        self.hw = torch.as_tensor(base.hw, device=device)
        self.cell = base.cell
        self.origin = torch.as_tensor(base.origin, device=device)
        self.nx, self.ny = base.nx, base.ny
        self.flat_sorted = torch.as_tensor(base._flat_sorted,
                                           device=device)
        self.sorted_idx = torch.as_tensor(base.sorted_idx, device=device)
        # padded-gather width = densest single cell
        self.pmax = int(np.bincount(base.cell_of).max())

    @torch.no_grad()
    def off_drivable(self, xy: Tensor, margin: float = MARGIN) -> Tensor:
        """(M, 2) -> (M,) signed clearance beyond the lane edge; exact
        mirror of TownLayout.off_drivable (99.0 = no lane within the
        3x3 neighborhood)."""
        M = xy.shape[0]
        dev = xy.device
        ij = torch.floor((xy - self.origin) / self.cell).long()
        cx = ij[:, 0].clamp(1, self.nx - 2)
        cy = ij[:, 1].clamp(1, self.ny - 2)
        cells = torch.stack([(cx + dx) * self.ny + (cy + dy)
                             for dx, dy in _NEIGH], dim=1)     # (M, 9)
        lo = torch.searchsorted(self.flat_sorted, cells)
        hi = torch.searchsorted(self.flat_sorted, cells, right=True)
        ar = torch.arange(self.pmax, device=dev)
        slot = lo.unsqueeze(-1) + ar                           # (M, 9, P)
        valid = slot < hi.unsqueeze(-1)
        cand = self.sorted_idx[slot.clamp(max=len(self.pts) - 1)]
        d = (self.pts[cand] - xy[:, None, None, :]).norm(dim=-1)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        d = d.view(M, -1)
        j = d.argmin(dim=1)
        dmin = d.gather(1, j[:, None]).squeeze(1)
        hwj = self.hw[cand.view(M, -1).gather(1, j[:, None]).squeeze(1)]
        out = dmin - hwj - margin
        return torch.where(torch.isfinite(dmin), out,
                           torch.full_like(out, 99.0))


class LayoutQuery:
    """Per-frame town dispatch over a GlobalLog's episodes.

    towns[i] = town name of episode i (order = the npz clip order, i.e.
    the manifest split); episodes = GlobalLog.episodes; n_frames = the
    log length. off(frame, xy) routes each batch row to its town's
    layout and returns signed clearance (positive = off-drivable).
    """

    def __init__(self, towns: Sequence[str],
                 episodes: Sequence[Tuple[int, int]], n_frames: int,
                 device: str = "cpu", layout_dir: Path = LAYOUT_DIR):
        assert len(towns) == len(episodes), \
            f"{len(towns)} towns vs {len(episodes)} episodes — " \
            "manifest split does not match this log"
        uniq = sorted(set(towns))
        self.layouts = [TownLayoutTorch(town_layout(t, layout_dir),
                                        device) for t in uniq]
        tid = {t: k for k, t in enumerate(uniq)}
        ft = torch.zeros(n_frames, dtype=torch.long)
        for t, (s, e) in zip(towns, episodes):
            ft[s:e] = tid[t]
        self.frame_town = ft.to(device)

    @torch.no_grad()
    def off(self, frame: Tensor, xy: Tensor,
            margin: float = MARGIN) -> Tensor:
        t = self.frame_town[frame]
        out = torch.full_like(xy[:, 0], 99.0)
        for k, tl in enumerate(self.layouts):
            m = t == k
            if m.any():
                out[m] = tl.off_drivable(xy[m], margin)
        return out
