"""Log-replay closed-loop training environment — the v0.2 "world".

The DITTO thesis, driving-native (V02_PLAN section 1): the policy
practices on-policy inside an offline world and is rewarded for
closeness to expert STATES. Here the world is the recorded scene itself
— traffic, route and lights replay from the log (counterfactual,
non-reactive) while an analytic kinematic model moves the ego along the
policy's predicted waypoint plan. Nothing in the loop is learned, so
the dream can neither be wrong about physics nor be exploited; the
reward lives entirely in ego-state space (position/heading/speed vs the
same-scene expert), so its dynamic range is 100% driving signal.

Frame conventions (settled v0.1 facts, do not re-derive):
- World arrays are CARLA map coords; ego theta is the anno compass
  = CARLA yaw + pi/2. world_to_ego = [[c, s], [-s, c]] at theta.
- In the compass ego frame FORWARD = -y, LATERAL(right) = +x; heading
  theta moves the ego along world direction (sin theta, -cos theta).
- Actor yaws in act_glob are raw CARLA yaws; obs features use
  yaw_rel = actor_yaw - ego_theta exactly like the offline adapter.

Everything is batched torch; runs on GPU inside the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .bench2drive import (FPS, LIGHT_DIMS, LIGHT_STATES, N_COMMANDS,
                          POS_SCALE, ROUTE_DIMS, VEL_SCALE, WP_SCALE)

DT = 1.0 / FPS


class GlobalLog:
    """The recorded scenes: global per-frame state + episode bounds.

    Loads the ##glob1 npz arrays (see bench2drive.global_frame_arrays)
    onto a device and provides window sampling for rollouts.
    """

    def __init__(self, npz_paths, device: str = "cpu"):
        parts: Dict[str, List[np.ndarray]] = {
            k: [] for k in ("obs", "wp", "reset", "ego_glob", "act_glob",
                            "route_glob", "light_glob")}
        for p in npz_paths:
            d = np.load(p)
            for k in parts:
                assert k in d, f"{p} lacks '{k}' — rebuild npz with " \
                    "global arrays (env.global_arrays: true)"
                parts[k].append(d[k])
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        self.device = device
        self.obs = torch.as_tensor(cat["obs"], dtype=torch.float32,
                                   device=device)
        self.wp = torch.as_tensor(cat["wp"], dtype=torch.float32,
                                  device=device)
        self.ego = torch.as_tensor(cat["ego_glob"], dtype=torch.float32,
                                   device=device)          # (N, 6)
        self.act = torch.as_tensor(cat["act_glob"], dtype=torch.float32,
                                   device=device)          # (N, A, 8)
        self.route = torch.as_tensor(cat["route_glob"], dtype=torch.float32,
                                     device=device)        # (N, 6)
        self.light = torch.as_tensor(cat["light_glob"], dtype=torch.float32,
                                     device=device)        # (N, 4)
        reset = cat["reset"]
        starts = np.where(reset)[0]
        ends = np.append(starts[1:], len(reset))
        self.episodes: List[Tuple[int, int]] = list(zip(starts.tolist(),
                                                        ends.tolist()))
        ep_end = np.zeros(len(reset), dtype=np.int64)
        ep_start = np.zeros(len(reset), dtype=np.int64)
        for s, e in self.episodes:
            ep_end[s:e] = e
            ep_start[s:e] = s
        self.ep_end = torch.as_tensor(ep_end, device=device)
        self.ep_start = torch.as_tensor(ep_start, device=device)

    def window_starts(self, horizon: int, margin: int) -> Tensor:
        """All frame indices t with [t, t + horizon + margin] in-episode."""
        idx = []
        for s, e in self.episodes:
            hi = e - horizon - margin
            if hi > s:
                idx.extend(range(s, hi))
        return torch.tensor(idx, dtype=torch.long, device=self.device)


@dataclass
class SimParams:
    n_neighbors: int = 6
    radius: float = 60.0
    with_route: bool = True
    with_lights: bool = False
    # The plan's own time parametrization (points 0.5 s apart) IS the
    # speed profile; caps exist only against garbage plans, NOT to
    # second-guess a valid schedule (real-data fidelity: the B2D expert
    # brakes at up to ~8 m/s^2 and cruises to 13.6 m/s).
    v_max: float = 14.0       # hard cap, m/s (expert max ~13.6)
    a_max: float = 8.0        # accel/brake cap on speed changes, m/s^2
    dtheta_max: float = 0.35  # yaw-rate cap per 0.1 s step, rad
    speed_gain: float = 1.0
    stride_s: float = 0.5     # wp spacing in seconds (WP_STRIDE / FPS)


@dataclass
class RewardParams:
    tau: int = 5              # time tolerance, frames each way
    sigma_p: float = 1.0      # m
    sigma_yaw: float = 0.3    # rad
    sigma_v: float = 2.0      # m/s
    # multi-scale position kernel: mix in a broad component so the
    # gradient survives beyond ~3 m off-path (G1 finding: the single
    # 1 m kernel floors out exactly where recovery must be learned)
    sigma_p2: float = 4.0     # m; 0 disables the broad component
    p2_weight: float = 0.3    # weight of the broad component
    collision_penalty: float = 0.0  # 0 = thesis-pure arm
    # only front/side impacts count for the penalty: a slower-than-log
    # ego being rear-ended by replayed (non-reactive) traffic is a
    # ghost artifact, not the policy's fault
    penalty_ignore_rear: bool = True


def _rot_world_to_ego(theta: Tensor) -> Tensor:
    """(B,) -> (B, 2, 2): rel = R @ (pos - ego)."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, s], -1),
                        torch.stack([-s, c], -1)], -2)


def _wrap(a: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class EgoSim:
    """Batched counterfactual rollouts through recorded scenes."""

    def __init__(self, log: GlobalLog, sim: SimParams = None,
                 rew: RewardParams = None):
        self.log = log
        self.p = sim or SimParams()
        self.r = rew or RewardParams()

    # ---------------- observation (mirrors featurize_frame) -----------

    def build_obs(self, frame: Tensor, xy: Tensor, theta: Tensor,
                  speed: Tensor) -> Tensor:
        """(B,) frame idx + sim ego state -> (B, obs_dim) obs."""
        p = self.p
        B = frame.shape[0]
        R = _rot_world_to_ego(theta)                       # (B, 2, 2)
        act = self.log.act[frame]                          # (B, A, 8)
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
            torch.cos(yaw_rel), torch.sin(yaw_rel)], dim=-1)  # (B, A, 7)
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

        if p.with_route:
            rt = self.log.route[frame]                     # (B, 6)
            block = torch.zeros(B, ROUTE_DIMS, device=rows.device)
            for i in range(2):
                pxy = rt[:, 2 * i:2 * i + 2]
                fin = torch.isfinite(pxy).all(dim=-1)
                relp = torch.einsum("bij,bj->bi", R,
                                    torch.nan_to_num(pxy) - xy)
                base = i * (2 + N_COMMANDS)
                block[:, base:base + 2] = torch.where(
                    fin.unsqueeze(-1), relp / POS_SCALE,
                    torch.zeros_like(relp))
                cmd = torch.nan_to_num(rt[:, 4 + i], nan=4.0).long() \
                    .clamp(0, N_COMMANDS)
                oh = F.one_hot(cmd, N_COMMANDS + 1)[:, 1:].float()
                block[:, base + 2:base + 2 + N_COMMANDS] = oh
            parts.append(block)

        if p.with_lights:
            lt = self.log.light[frame]                     # (B, 4)
            lb = torch.zeros(B, LIGHT_DIMS, device=rows.device)
            on = lt[:, 0] > 0.5
            lb[:, 0] = on.float()
            fin = torch.isfinite(lt[:, 1:3]).all(dim=-1) & on
            relt = torch.einsum("bij,bj->bi", R,
                                torch.nan_to_num(lt[:, 1:3]) - xy)
            lb[:, 1:3] = torch.where(fin.unsqueeze(-1), relt / POS_SCALE,
                                     torch.zeros_like(relt))
            state = lt[:, 3].long()
            for s_ in range(LIGHT_STATES):
                lb[:, 3 + s_] = ((state == s_) & on).float()
            parts.append(lb)

        return torch.cat(parts, dim=1).clamp(-2.0, 2.0)

    # ---------------- kinematic ego step ------------------------------

    def step_ego(self, action: Tensor, xy: Tensor, theta: Tensor,
                 speed: Tensor):
        """Advance the ego one tick along the predicted plan.

        action: (B, 2k) compass-ego-frame waypoints / WP_SCALE (the
        champion head's action space). Target speed reuses the
        deployment tracker's own spacing/curvature formulas
        (rotation-invariant, so computed directly on the compass-frame
        plan); motion follows the plan polyline under accel and
        yaw-rate caps. Returns new (xy, theta, speed).
        """
        p = self.p
        B = action.shape[0]
        dev = action.device
        plan = action.view(B, -1, 2) * WP_SCALE            # (B, k, 2)
        k = plan.shape[1]
        pts = torch.cat([torch.zeros(B, 1, 2, device=dev), plan], 1)
        seg = pts[:, 1:] - pts[:, :-1]
        seglen = seg.norm(dim=-1)
        arc = torch.cat([torch.zeros(B, 1, device=dev),
                         seglen.cumsum(dim=1)], dim=1)

        # Schedule speed from the plan's own time parametrization. A
        # stride's chord/stride_s is the AVERAGE speed over it (= the
        # instantaneous speed at its midpoint, +0.25 s), so using it
        # directly runs ~1 m ahead of a launching expert; under
        # constant within-second accel the speed at t=0 is exactly
        # 1.5*v1 - 0.5*v2 (two-stride extrapolation back to now).
        # The deployment tracker's n_sp=2 average / curvature cap are
        # control setpoints for CARLA's inertia to low-pass — using
        # them here drifted 1.7-2.8 m over 4 s on real experts. Caps
        # below only bound garbage plans.
        b = torch.arange(B, device=dev)
        if k >= 2:
            v_plan = (2.0 * arc[:, 1] - 0.5 * arc[:, 2]).clamp_min(0.0) \
                / p.stride_s
        else:
            v_plan = arc[:, 1] / p.stride_s
        v_t = (p.speed_gain * v_plan).clamp(max=p.v_max)

        v_next = speed + (v_t - speed).clamp(-p.a_max * DT, p.a_max * DT)
        v_next = v_next.clamp_min(0.0)
        ds = v_next * DT

        j = (torch.searchsorted(arc, ds.unsqueeze(1), right=True)
             .squeeze(1) - 1).clamp(0, k - 1)
        frac = ((ds - arc[b, j]) / seglen[b, j].clamp_min(1e-6)).clamp(0, 1)
        move = pts[b, j] + frac.unsqueeze(-1) * seg[b, j]   # ego frame
        tang = seg[b, j]
        has_path = arc[:, -1] > 1e-3
        move = torch.where(has_path.unsqueeze(-1), move,
                           torch.zeros_like(move))
        # heading: the containing segment's chord angle alpha
        # (atan2(t_x, -t_y); straight ahead in the compass ego frame is
        # (0, -1)) equals kappa * s_mid on an arc — the chord averages
        # curvature over its span — so the per-step heading change is
        # alpha scaled by the fraction of that span actually traveled:
        # delta = alpha * ds / s_mid (exact on circular arcs; raw alpha
        # over-rotates ~2.5x and spirals off curved paths).
        alpha = torch.atan2(tang[..., 0], -tang[..., 1])
        s_mid = arc[b, j] + 0.5 * seglen[b, j]
        delta = alpha * (ds / s_mid.clamp_min(1e-3)).clamp(max=2.0)
        delta = torch.where(has_path & (seglen[b, j] > 1e-3)
                            & (v_next > 0.1),
                            delta, torch.zeros_like(delta))
        delta = delta.clamp(-p.dtheta_max, p.dtheta_max)

        c, s = torch.cos(theta), torch.sin(theta)
        RT = torch.stack([torch.stack([c, -s], -1),
                          torch.stack([s, c], -1)], -2)
        xy_new = xy + torch.einsum("bij,bj->bi", RT, move)
        theta_new = _wrap(theta + delta)
        return xy_new, theta_new, v_next

    # ---------------- reward ------------------------------------------

    def reward(self, frame: Tensor, xy: Tensor, theta: Tensor,
               speed: Tensor) -> Tensor:
        """Time-tolerant ego-state match against the same-scene expert.

        r = max_{|d| <= tau} exp(-1/2 [ |dxy|^2/s_p^2 + dth^2/s_y^2
                                        + dv^2/s_v^2 ])  in [0, 1].
        Expert replay scores ~1 by construction; off the expert path
        the gradient points back to it — the recovery incentive.
        """
        r = self.r
        offs = torch.arange(-r.tau, r.tau + 1, device=frame.device)
        idx = frame.unsqueeze(1) + offs                    # (B, 2tau+1)
        lo = self.log.ep_start[frame].unsqueeze(1)
        hi = (self.log.ep_end[frame] - 1).unsqueeze(1)
        idx = idx.clamp(lo, hi)
        ex = self.log.ego[idx]                             # (B, W, 6)
        dxy = (ex[..., 0:2] - xy.unsqueeze(1)).norm(dim=-1)
        dth = _wrap(ex[..., 2] - theta.unsqueeze(1)).abs()
        dv = (ex[..., 3] - speed.unsqueeze(1)).abs()
        rest = -0.5 * ((dth / r.sigma_yaw) ** 2
                       + (dv / r.sigma_v) ** 2)
        kern = torch.exp(-0.5 * (dxy / r.sigma_p) ** 2 + rest)
        if r.sigma_p2 > 0.0:
            broad = torch.exp(-0.5 * (dxy / r.sigma_p2) ** 2 + rest)
            kern = (1.0 - r.p2_weight) * kern + r.p2_weight * broad
        return kern.max(dim=1).values

    # ---------------- collisions --------------------------------------

    def collisions(self, frame: Tensor, xy: Tensor, theta: Tensor,
                   ignore_rear: bool = False) -> Tensor:
        """(B,) bool: ego OBB overlaps any replayed actor OBB (SAT).

        ignore_rear drops actors centered behind the ego — a
        slower-than-log ego being rear-ended by non-reactive replay
        traffic is a ghost artifact, not a policy fault."""
        log = self.log
        act = log.act[frame]                               # (B, A, 8)
        pres = act[..., 0] > 0.5
        d = (act[..., 1:3] - xy.unsqueeze(1)).norm(dim=-1)
        cand = pres & (d < 10.0)
        if ignore_rear:
            R = _rot_world_to_ego(theta)
            rel = torch.einsum("bij,baj->bai", R,
                               act[..., 1:3] - xy.unsqueeze(1))
            cand = cand & (-rel[..., 1] > -1.0)  # forward = -y
        ego_ext = log.ego[frame][:, 4:6]                   # (B, 2)
        ego_yaw = theta - np.pi / 2   # CARLA yaw of the sim ego
        return _obb_overlap_any(xy, ego_yaw, ego_ext,
                                act[..., 1:3], act[..., 5], act[..., 6:8],
                                cand)

    # ---------------- rollout starts ----------------------------------

    def reset(self, starts: Tensor, perturb: Optional[dict] = None,
              rng: Optional[torch.Generator] = None):
        """Initial sim state at logged frames `starts` (B,).

        perturb (divergent starts): dict with frac, lat_sigma (m),
        yaw_sigma (rad), v_sigma (m/s) applied to a `frac` share of the
        batch — the reward target stays the same-scene expert, so
        recovery is learnable, not poisoned (the gen-4 lesson).
        """
        ego = self.log.ego[starts]
        xy = ego[:, 0:2].clone()
        theta = ego[:, 2].clone()
        speed = ego[:, 3].clone()
        if perturb:
            B = len(starts)
            dev = xy.device
            def _rand(*shape):
                return torch.randn(*shape, device=dev, generator=rng)
            mask = (torch.rand(B, device=dev, generator=rng)
                    < perturb.get("frac", 0.0)).float()
            lat = _rand(B) * perturb.get("lat_sigma", 0.0) * mask
            c, s = torch.cos(theta), torch.sin(theta)
            RT = torch.stack([torch.stack([c, -s], -1),
                              torch.stack([s, c], -1)], -2)
            off = torch.stack([lat, torch.zeros_like(lat)], -1)
            xy = xy + torch.einsum("bij,bj->bi", RT, off)
            theta = _wrap(theta + _rand(B)
                          * perturb.get("yaw_sigma", 0.0) * mask)
            speed = (speed + _rand(B)
                     * perturb.get("v_sigma", 0.0) * mask).clamp_min(0.0)
        return xy, theta, speed


def _obb_overlap_any(xy_a: Tensor, yaw_a: Tensor, ext_a: Tensor,
                     xy_b: Tensor, yaw_b: Tensor, ext_b: Tensor,
                     mask: Tensor) -> Tensor:
    """Batched rectangle SAT: does box a overlap ANY masked box b?

    xy_a (B,2), yaw_a (B,), ext_a (B,2) vs xy_b (B,A,2), yaw_b (B,A),
    ext_b (B,A,2), mask (B,A) -> (B,) bool. Separating axes = both
    boxes' local axes (4 per pair).
    """
    B, A = mask.shape
    ca, sa = torch.cos(yaw_a), torch.sin(yaw_a)
    ax_a = torch.stack([torch.stack([ca, sa], -1),
                        torch.stack([-sa, ca], -1)], 1)    # (B, 2, 2)
    cb, sb = torch.cos(yaw_b), torch.sin(yaw_b)
    ax_b = torch.stack([torch.stack([cb, sb], -1),
                        torch.stack([-sb, cb], -1)], 2)    # (B, A, 2, 2)
    axes = torch.cat([ax_a.unsqueeze(1).expand(-1, A, -1, -1), ax_b],
                     dim=2)                                # (B, A, 4, 2)
    dc = (xy_b - xy_a.unsqueeze(1)).unsqueeze(2)           # (B, A, 1, 2)
    dist = (dc * axes).sum(-1).abs()                       # (B, A, 4)
    proj_a = ((ax_a.unsqueeze(1).unsqueeze(2)              # (B,1,1,2,2)
               * axes.unsqueeze(3)).sum(-1).abs()
              * ext_a.view(B, 1, 1, 2)).sum(-1)            # (B, A, 4)
    proj_b = ((ax_b.unsqueeze(2)                           # (B,A,1,2,2)
               * axes.unsqueeze(3)).sum(-1).abs()
              * ext_b.unsqueeze(2)).sum(-1)                # (B, A, 4)
    sep = dist > proj_a + proj_b                           # (B, A, 4)
    overlap = ~sep.any(dim=-1) & mask
    return overlap.any(dim=-1)
