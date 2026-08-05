"""v0.3 traffic model: ego-conditioned per-agent next-step prediction.

The one thing replay cannot provide is REACTIONS (V03_PLAN §1). This
model predicts each agent's next-step motion from (a) its own recent
history, (b) the other agents, and (c) the EGO's actual current state —
so a policy that brakes in front of a follower produces a braking
follower in imagination.

Design (v1, deliberately simple; W0 gate decides if it suffices):
- Per-agent features are TRANSLATION/ROTATION INVARIANT locally
  (speed and yaw-rate history in the agent's own frame) plus the pose
  RELATIVE TO EGO (interaction geometry) and class/extent.
- A transformer attends across agent tokens + one ego token + one
  light token; heads emit Gaussian (dx_fwd, dy_left, dyaw) in each
  agent's CURRENT local frame — invariant targets, world-integrated by
  the rollout wrapper.
- Predicted agents = those with a full `hist`-frame history (audited:
  churn is low, median track 12 s); newcomers ride as context via the
  presence mask until they accumulate history.

Rollout contract mirrors EgoSim: step(states, ego_state) -> states'.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal

from ..tracks import STATE_DIM

# track state columns (tracks.STATE_COLS order):
#   0 x, 1 y, 2 vx_w, 3 vy_w, 4 yaw, 5 ext_x, 6 ext_y
DT = 0.1


@dataclass
class SceneWindows:
    """Index of trainable scene samples over a ##glob2 npz dict.

    For each valid frame t: which slots are predictable (full history)
    and per-slot (hist, STATE_DIM) history tensors, plus ego state and
    light context — everything featurize() needs, precomputed once.
    """
    frames: np.ndarray        # (N,) absolute frame index t
    hist: np.ndarray          # (N, A, H, STATE_DIM) float32 (0 padded)
    pred_mask: np.ndarray     # (N, A) bool — full-history agents
    pres_mask: np.ndarray     # (N, A) bool — present at t
    cls: np.ndarray           # (N, A) int8
    ego: np.ndarray           # (N, 4) x, y, theta, speed
    light: np.ndarray         # (N, 4) presence, tv_xy, state
    target: np.ndarray        # (N, A, 3) dx_fwd, dy_left, dyaw @ t+1
    ids: np.ndarray           # (N, A) actor ids (-1 empty): slots
                              # re-sort per frame — cross-window
                              # comparisons MUST match by id
    ego_hist: np.ndarray      # (N, H, 4) ego (x, y, theta, speed)
                              # history — a memoryless per-step model
                              # cannot distinguish a DECELERATING ego
                              # from a slow one (round-3 reactivity
                              # diagnosis); followers need to see the
                              # braking onset


def build_scene_windows(data, hist: int = 10) -> SceneWindows:
    """Assemble per-frame agent histories by actor id (numpy, once)."""
    act, ids = data["act_glob"], data["act_id"]
    cls_arr = data["act_cls"]
    reset = data["reset"]
    ego_g = data["ego_glob"]
    light_g = data["light_glob"]
    T, A = ids.shape
    starts = np.where(reset)[0]
    ep_start = np.zeros(T, dtype=np.int64)
    ep_end = np.zeros(T, dtype=np.int64)
    ends = np.append(starts[1:], T)
    for s, e in zip(starts, ends):
        ep_start[s:e], ep_end[s:e] = s, e

    # id -> slot lookup per frame
    slot_of = [dict() for _ in range(T)]
    for t in range(T):
        for a in range(A):
            if ids[t, a] >= 0:
                slot_of[t][int(ids[t, a])] = a

    frames, H, PM, PR, CL, EG, LI, TG = [], [], [], [], [], [], [], []
    IDS, EH = [], []
    for t in range(T):
        if t - hist + 1 < ep_start[t] or t + 1 >= ep_end[t]:
            continue
        h = np.zeros((A, hist, STATE_DIM), dtype=np.float32)
        pm = np.zeros(A, dtype=bool)
        pr = ids[t] >= 0
        tg = np.zeros((A, 3), dtype=np.float32)
        for a in np.where(pr)[0]:
            aid = int(ids[t, a])
            ok = True
            for k in range(hist):
                s = slot_of[t - hist + 1 + k].get(aid)
                if s is None:
                    ok = False
                    break
                h[a, k] = act[t - hist + 1 + k, s,
                              [1, 2, 3, 4, 5, 6, 7]]
            nxt = slot_of[t + 1].get(aid)
            if ok and nxt is not None:
                pm[a] = True
                x0, y0, yaw0 = h[a, -1, 0], h[a, -1, 1], h[a, -1, 4]
                x1, y1 = act[t + 1, nxt, 1], act[t + 1, nxt, 2]
                yaw1 = act[t + 1, nxt, 5]   # act col 5 = yaw (4 = vy!)
                c, s_ = math.cos(yaw0), math.sin(yaw0)
                dx, dy = x1 - x0, y1 - y0
                dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
                tg[a] = [c * dx + s_ * dy, -s_ * dx + c * dy, dyaw]
        if pm.any():
            frames.append(t)
            H.append(h)
            PM.append(pm)
            PR.append(pr)
            CL.append(cls_arr[t])
            EG.append(ego_g[t, :4])
            LI.append(light_g[t])
            TG.append(tg)
            IDS.append(ids[t].copy())
            EH.append(ego_g[t - hist + 1:t + 1, :4].copy())
    return SceneWindows(np.array(frames), np.stack(H), np.stack(PM),
                        np.stack(PR), np.stack(CL), np.stack(EG),
                        np.stack(LI), np.stack(TG), np.stack(IDS),
                        np.stack(EH))


def featurize(hist: Tensor, pres: Tensor, cls: Tensor, ego: Tensor,
              light: Tensor,
              ego_hist: Tensor = None) -> Tuple[Tensor, Tensor]:
    """Model inputs from raw windows (batched, torch).

    hist (B, A, H, 7); ego (B, 4) [x, y, theta, speed]; light (B, 4).
    Returns agent features (B, A, F) and ego/light context (B, Fc).
    Local features are translation-invariant; ego-relative geometry is
    rotated into the EGO's frame (compass convention, like the obs).
    """
    B, A, H, _ = hist.shape
    xy = hist[..., 0:2]
    yaw = hist[..., 4]
    speeds = hist[..., 2:4].norm(dim=-1)                    # (B, A, H)
    dyaw = (yaw[..., 1:] - yaw[..., :-1] + math.pi) \
        % (2 * math.pi) - math.pi                           # (B, A, H-1)
    cur = hist[:, :, -1]                                    # (B, A, 7)
    # pose relative to ego, ego-frame (world_to_ego on compass theta)
    th = ego[:, 2]
    c, s = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, s], -1),
                     torch.stack([-s, c], -1)], -2)         # (B, 2, 2)
    rel = torch.einsum("bij,baj->bai", R,
                       cur[..., 0:2] - ego[:, None, 0:2])
    vel_rel = torch.einsum("bij,baj->bai", R, cur[..., 2:4])
    # agent yaw relative to ego heading (compass offset cancels in use)
    ryaw = cur[..., 4] - (th[:, None] - math.pi / 2)
    onehot = torch.nn.functional.one_hot(
        cls.clamp(0, 2).long(), 3).float() * (cls >= 0)[..., None]
    feat = torch.cat([
        speeds / 15.0,                                      # H
        dyaw / 0.2,                                         # H-1
        rel / 50.0, vel_rel / 15.0,                         # 4
        torch.cos(ryaw)[..., None], torch.sin(ryaw)[..., None],
        cur[..., 5:7] / 3.0,                                # extents
        onehot,
    ], dim=-1)
    feat = feat * pres[..., None]
    if ego_hist is None:
        espeed = ego[:, 3:4].expand(-1, H) / 15.0
    else:
        espeed = ego_hist[..., 3] / 15.0                    # (B, H)
    ctx = torch.cat([espeed, light], dim=-1)                # (B, H+4)
    return feat, ctx


CTX_DIM = 14  # ego speed HISTORY (H=10) + light (4)


class TrafficModel(nn.Module):
    def __init__(self, hist: int = 10, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 8, n_modes: int = 4):
        super().__init__()
        self.hist = hist
        self.n_modes = n_modes
        feat_dim = hist + (hist - 1) + 4 + 2 + 2 + 3
        self.embed = nn.Linear(feat_dim, d_model)
        self.embed_ctx = nn.Linear(CTX_DIM, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, n_modes * 7))
        # target scales: ~1.4 m/step forward max, lateral small, dyaw
        self.register_buffer("t_scale",
                             torch.tensor([1.0, 0.2, 0.05]))

    def modes(self, hist, pres, cls, ego, light, ego_hist=None):
        """GMM heads: (mu (B,A,M,3), log_std (B,A,M,3),
        logits (B,A,M)). Futures at 4 s genuinely branch (regime
        decomposition 2026-08-04: error mass in cruising/launching =
        future behavior changes); a unimodal mean averages modes."""
        feat, ctx = featurize(hist, pres, cls, ego, light, ego_hist)
        x = self.embed(feat)                                # (B, A, D)
        x = torch.cat([self.embed_ctx(ctx)[:, None], x], dim=1)
        pad = torch.cat([torch.zeros_like(pres[:, :1]), ~pres], dim=1)
        x = self.encoder(x, src_key_padding_mask=pad)
        out = self.head(x[:, 1:])                           # (B, A, M*7)
        B, A, _ = out.shape
        out = out.view(B, A, self.n_modes, 7)
        mu = out[..., 0:3] * self.t_scale
        log_std = out[..., 3:6].clamp(-5.0, 1.0)
        return mu, log_std, out[..., 6]

    def dist(self, hist, pres, cls, ego, light,
             ego_hist=None) -> Independent:
        """Highest-probability mode as a Gaussian (rollout/back-compat
        path; training uses the winner-take-all loss over all modes)."""
        mu, log_std, logits = self.modes(hist, pres, cls, ego, light,
                                         ego_hist)
        b = torch.argmax(logits, dim=-1, keepdim=True)      # (B, A, 1)
        take = lambda t: torch.gather(
            t, 2, b[..., None].expand(-1, -1, 1, 3)).squeeze(2)
        return Independent(Normal(take(mu),
                                  take(log_std).exp() * self.t_scale), 1)

    def loss(self, hist, pres, cls, ego, light, target,
             pred_mask, ego_hist=None) -> Tensor:
        """Winner-take-all GMM loss: NLL of the best mode + mode-choice
        cross-entropy (standard MTP recipe)."""
        mu, log_std, logits = self.modes(hist, pres, cls, ego, light,
                                         ego_hist)
        std = log_std.exp() * self.t_scale
        t = target[:, :, None, :]                           # (B, A, 1, 3)
        nll = 0.5 * (((t - mu) / std) ** 2 + 2 * log_std
                     + math.log(2 * math.pi)).sum(-1)       # (B, A, M)
        best_nll, best = nll.min(dim=-1)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.n_modes), best.reshape(-1),
            reduction="none").view_as(best)
        # interaction weighting: 40% of samples are trivially-static
        # agents (regime ADE 0.24) that dominate the gradient; reweight
        # toward near-ego moving agents and decel/accel events
        cur = hist[:, :, -1]
        d_ego = (cur[..., 0:2] - ego[:, None, 0:2]).norm(dim=-1)
        moving = cur[..., 2:4].norm(dim=-1) > 1.0
        dv = (target[..., 0] - cur[..., 2:4].norm(dim=-1) * DT).abs()
        w = (1.0 + 3.0 * ((d_ego < 15.0) & moving).float()
             + 2.0 * (dv > 0.05).float())
        l = (best_nll + 0.5 * ce) * pred_mask * w
        return l.sum() / (pred_mask * w).sum().clamp_min(1)

    def advance(self, cur: Tensor, delta: Tensor) -> Tensor:
        """Integrate local deltas to world states (differentiable)."""
        yaw0 = cur[..., 4]
        c, s = torch.cos(yaw0), torch.sin(yaw0)
        dx_w = c * delta[..., 0] - s * delta[..., 1]
        dy_w = s * delta[..., 0] + c * delta[..., 1]
        return torch.stack([
            cur[..., 0] + dx_w, cur[..., 1] + dy_w,
            dx_w / DT, dy_w / DT, yaw0 + delta[..., 2],
            cur[..., 5], cur[..., 6]], dim=-1)

    @torch.no_grad()
    def step_mode(self, hist, pres, cls, ego, light,
                  mode: int, ego_hist=None) -> Tensor:
        """Rollout step following a FIXED mode index (minADE eval)."""
        mu, _, _ = self.modes(hist, pres, cls, ego, light, ego_hist)
        return self.advance(hist[:, :, -1], mu[:, :, mode])

    @torch.no_grad()
    def step(self, hist, pres, cls, ego, light,
             sample: bool = False, ego_hist=None) -> Tensor:
        """One rollout step: predict local deltas, integrate to world.

        hist (B, A, H, 7) -> next world states (B, A, 7): the caller
        shifts them into the history window for the next step.
        """
        d = self.dist(hist, pres, cls, ego, light, ego_hist)
        delta = d.sample() if sample else d.base_dist.loc   # (B, A, 3)
        return self.advance(hist[:, :, -1], delta)
