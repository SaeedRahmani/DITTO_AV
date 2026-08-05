"""Reactive egosim (v0.3 D1): traffic from the learned model.

ReactiveEgoSim extends EgoSim with a model-evolved traffic buffer:
- reset_reactive() seeds per-agent histories from the log (SceneWindows
  assembly at the start frame);
- step_traffic(ego_state) advances predicted agents via the TrafficModel
  ensemble (member 0 drives the state; the spread across members is the
  W1 pessimism signal) while non-predicted agents keep replaying the
  log (context correctness), with ID-based de-duplication so an actor
  never appears both evolved and replayed;
- build_obs()/collisions() source actors from the buffer instead of
  log.act[frame] — the ONLY semantic change vs the replay world, pinned
  by an equivalence test (buffer==log => identical outputs).

Reward stays the parent's: ego-state matching vs the SAME-scene expert
(from the real log) — the world reacts, the target does not move.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .egosim import EgoSim, GlobalLog, RewardParams, SimParams
from .egosim import _obb_overlap_any, _rot_world_to_ego
from .bench2drive import POS_SCALE, VEL_SCALE
from .models.traffic import SceneWindows


class ReactiveEgoSim(EgoSim):
    def __init__(self, log: GlobalLog, windows: SceneWindows,
                 models: List, act_id: np.ndarray,
                 sim: SimParams = None, rew: RewardParams = None):
        super().__init__(log, sim, rew)
        self.windows = windows
        self.models = models
        self.frame_to_w = {int(f): i for i, f in enumerate(windows.frames)}
        self.act_id = torch.as_tensor(act_id, device=log.obs.device)
        self._buf = None          # (B, A, 8) act-format override
        self._hist = None         # (B, A, H, 7) model state histories
        self._pred = None         # (B, A) bool: model-evolved agents
        self._pred_ids = None     # (B, A) ids of evolved agents (-1)
        self._cls = None
        self.last_disagreement = None

    # ------------------------------------------------------------ init

    def reset_reactive(self, starts: Tensor, perturb=None, rng=None):
        """Seed sim + traffic buffer at log frames (each must have a
        scene window: full-history agents become model-evolved)."""
        xy, th, v = self.reset(starts, perturb, rng)
        dev = starts.device
        widx = [self.frame_to_w.get(int(f)) for f in starts]
        assert all(w is not None for w in widx), \
            "reactive starts must map to scene windows"
        widx = np.asarray(widx)
        self._hist = torch.as_tensor(self.windows.hist[widx], device=dev)
        self._pred = torch.as_tensor(self.windows.pred_mask[widx],
                                     device=dev)
        self._cls = torch.as_tensor(self.windows.cls[widx],
                                    device=dev).long()
        ids = torch.as_tensor(self.windows.ids[widx], device=dev)
        self._pred_ids = torch.where(self._pred, ids,
                                     torch.full_like(ids, -1))
        self._ego_hist = torch.as_tensor(self.windows.ego_hist[widx],
                                         device=dev)
        self._sync_buffer(starts)
        return xy, th, v

    def _sync_buffer(self, frames: Tensor):
        """Compose the act-format buffer: evolved agents from _hist,
        everything else replayed from the log at `frames`, minus
        duplicates of evolved ids."""
        log_act = self.log.act[frames].clone()          # (B, A, 8)
        log_ids = self.act_id[frames]                   # (B, A)
        # drop replay slots whose actor is model-evolved (dedup by id)
        B, A = log_ids.shape
        dup = torch.zeros(B, A, dtype=torch.bool, device=log_ids.device)
        for b in range(B):
            evolved = self._pred_ids[b][self._pred_ids[b] >= 0]
            if len(evolved):
                dup[b] = torch.isin(log_ids[b], evolved)
        log_act[dup] = 0.0
        # write evolved agents from _hist current states
        cur = self._hist[:, :, -1]                      # (B, A, 7)
        rows = torch.stack([
            torch.ones_like(cur[..., 0]), cur[..., 0], cur[..., 1],
            cur[..., 2], cur[..., 3], cur[..., 4],
            cur[..., 5], cur[..., 6]], dim=-1)
        self._buf = torch.where(self._pred[..., None], rows, log_act)

    # ------------------------------------------------------------ step

    @torch.no_grad()
    def step_traffic(self, frames: Tensor, ego_xy: Tensor,
                     ego_theta: Tensor, ego_v: Tensor):
        """Advance evolved agents one tick, conditioned on the SIM ego."""
        ego_state = torch.stack(
            [ego_xy[..., 0], ego_xy[..., 1], ego_theta, ego_v], dim=-1)
        self._ego_hist = torch.cat(
            [self._ego_hist[:, 1:], ego_state.float()[:, None]], dim=1)
        light = self.log.light[frames]
        outs = [m.step(self._hist, self._pred, self._cls,
                       ego_state.float(), light,
                       ego_hist=self._ego_hist) for m in self.models]
        nxt = outs[0]
        if len(outs) > 1:
            stack = torch.stack([o[..., 0:2] for o in outs])
            d = stack.std(dim=0).norm(dim=-1)           # (B, A)
            d = torch.where(self._pred, d, torch.zeros_like(d))
            self.last_disagreement = d.sum(dim=1) / \
                self._pred.sum(dim=1).clamp_min(1)      # (B,)
        self._hist = torch.cat([self._hist[:, :, 1:],
                                nxt[:, :, None]], dim=2)
        self._sync_buffer(frames)

    # ------------------------- overridden traffic sources -------------

    def build_obs(self, frame: Tensor, xy: Tensor, theta: Tensor,
                  speed: Tensor) -> Tensor:
        if self._buf is None:
            return super().build_obs(frame, xy, theta, speed)
        # identical math to the parent; actor source = self._buf
        p = self.p
        B = frame.shape[0]
        R = _rot_world_to_ego(theta)
        act = self._buf
        pres = act[..., 0] > 0.5
        rel = torch.einsum("bij,baj->bai", R,
                           act[..., 1:3] - xy.unsqueeze(1))
        dist = rel.norm(dim=-1)
        ok = pres & (dist <= p.radius) & (dist > 1e-6)
        vel_rel = torch.einsum("bij,baj->bai", R, act[..., 3:5])
        yaw_rel = act[..., 5] - theta.unsqueeze(1)
        rows = torch.stack([
            torch.ones_like(dist),
            rel[..., 0] / POS_SCALE, rel[..., 1] / POS_SCALE,
            vel_rel[..., 0] / VEL_SCALE, vel_rel[..., 1] / VEL_SCALE,
            torch.cos(yaw_rel), torch.sin(yaw_rel)], dim=-1)
        rows = rows * ok.unsqueeze(-1)
        order = torch.where(ok, dist, torch.full_like(dist, float("inf"))
                            ).argsort(dim=1)[:, :p.n_neighbors]
        near = torch.gather(rows, 1,
                            order.unsqueeze(-1).expand(-1, -1, 7))
        ego_row = torch.zeros(B, 1, 7, device=rows.device)
        ego_row[:, 0, 0] = 1.0
        ego_row[:, 0, 3] = speed / VEL_SCALE
        ego_row[:, 0, 5] = 1.0
        parts = [torch.cat([ego_row, near], dim=1).flatten(1)]
        # route/light blocks replay from the log — delegate by building
        # the parent's obs and splicing its tail (cheap, exact)
        base = super().build_obs(frame, xy, theta, speed)
        core = parts[0].shape[1]
        return torch.cat([parts[0].clamp(-2.0, 2.0),
                          base[:, core:]], dim=1)

    def collisions(self, frame: Tensor, xy: Tensor, theta: Tensor,
                   ignore_rear: bool = False) -> Tensor:
        if self._buf is None:
            return super().collisions(frame, xy, theta, ignore_rear)
        act = self._buf
        pres = act[..., 0] > 0.5
        d = (act[..., 1:3] - xy.unsqueeze(1)).norm(dim=-1)
        cand = pres & (d < 10.0)
        if ignore_rear:
            R = _rot_world_to_ego(theta)
            rel = torch.einsum("bij,baj->bai", R,
                               act[..., 1:3] - xy.unsqueeze(1))
            cand = cand & (-rel[..., 1] > -1.0)
        ego_ext = self.log.ego[frame][:, 4:6]
        ego_yaw = theta - np.pi / 2
        return _obb_overlap_any(xy, ego_yaw, ego_ext,
                                act[..., 1:3], act[..., 5],
                                act[..., 6:8], cand)
