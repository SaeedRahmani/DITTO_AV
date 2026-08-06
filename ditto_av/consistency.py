"""Plan-consistency (churn) terms for rollout training — V032_PLAN
1.3.2. Torch mirror of smoothness.plan_churn's single-step core,
equivalence pinned by tests/test_consistency.py.

Churn is the motion-compensated lateral disagreement between the plan
emitted at tick t-1 and the plan emitted at tick t: the part of the
new plan that is not explained by having moved along the old one.
Unlike the sigma_yawrate reward channel this needs NO expert data —
it prices the policy's own indecision, wherever it happens.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .egosim import _wrap


def proximity_gate(dmin: Tensor, d0: float, width: float) -> Tensor:
    """(B,) in (0,1): ~1 when the nearest actor is far (> d0), ~0 when
    near — consistency pressure fades exactly where reactivity lives
    (the CARLA split: commitment fixed layout collisions on empty road
    and doubled vehicle collisions near traffic)."""
    return torch.sigmoid((dmin - d0) / width)


def _prev_into_current(prev_plan: Tensor, prev_xy: Tensor, xy: Tensor,
                       prev_theta: Tensor, theta: Tensor) -> Tensor:
    """Transform the previous tick's plan into the current ego frame."""
    c0, s0 = torch.cos(prev_theta), torch.sin(prev_theta)
    c1, s1 = torch.cos(theta), torch.sin(theta)
    R0T = torch.stack([torch.stack([c0, -s0], -1),
                       torch.stack([s0, c0], -1)], -2)
    R1 = torch.stack([torch.stack([c1, s1], -1),
                      torch.stack([-s1, c1], -1)], -2)
    w = prev_xy.unsqueeze(1) + torch.einsum("bij,bkj->bki", R0T,
                                            prev_plan)
    return torch.einsum("bij,bkj->bki", R1, w - xy.unsqueeze(1))


def plan_shape_churn(prev_plan: Tensor, plan: Tensor, prev_xy: Tensor,
                     xy: Tensor, prev_theta: Tensor, theta: Tensor
                     ) -> Tensor:
    """(B,) mean distance from the previous plan's points to the
    CURRENT plan's PATH (point-to-polyline, arc-overlap-limited).

    v0.3.2 round 6: commitment on the path SHAPE only. Re-scheduling
    speed along the same path — braking, creeping, launching — moves
    points ALONG the polyline and costs ~nothing, so the creep-lock
    mechanism (time-aligned consistency made launch hesitation sticky)
    is removed by construction; changing WHERE the path goes is
    priced. Previous-plan points that project beyond the current
    path's end (hard braking shortened it) are excluded.
    """
    B, k, _ = plan.shape
    q = _prev_into_current(prev_plan, prev_xy, xy, prev_theta, theta)
    pts = torch.cat([torch.zeros(B, 1, 2, device=plan.device,
                                 dtype=plan.dtype), plan], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]                     # (B, k, 2)
    L2 = (seg * seg).sum(-1).clamp_min(1e-8)           # (B, k)
    d0 = q.unsqueeze(2) - pts[:, :-1].unsqueeze(1)     # (B, kq, ks, 2)
    t = ((d0 * seg.unsqueeze(1)).sum(-1)
         / L2.unsqueeze(1)).clamp(0.0, 1.0)            # (B, kq, ks)
    proj = pts[:, :-1].unsqueeze(1) + t.unsqueeze(-1) * seg.unsqueeze(1)
    dist = (q.unsqueeze(2) - proj).norm(dim=-1)        # (B, kq, ks)
    mind, argj = dist.min(dim=2)
    tj = t.gather(2, argj.unsqueeze(-1)).squeeze(-1)
    beyond = (argj == k - 1) & (tj >= 1.0 - 1e-6)
    w = (~beyond).float()
    return (mind * w).sum(1) / w.sum(1).clamp_min(1.0)


def plan_churn_lat(prev_plan: Tensor, plan: Tensor, prev_xy: Tensor,
                   xy: Tensor, prev_theta: Tensor, theta: Tensor
                   ) -> Tensor:
    """(B,) mean |lateral disagreement| in meters.

    prev_plan/plan: (B, k, 2) compass-ego-frame waypoint plans emitted
    at the previous / current tick; poses are world compass states at
    those ticks. Time alignment: with WP_STRIDE=5 ticks per waypoint,
    prev_plan's wp_i sits 0.2/0.8 between the current plan's polyline
    nodes i-1 and i (node 0 = origin) — same interpolation as
    smoothness.plan_churn.
    """
    B, k, _ = plan.shape
    c0, s0 = torch.cos(prev_theta), torch.sin(prev_theta)
    c1, s1 = torch.cos(theta), torch.sin(theta)
    # ego(t-1) -> world (transpose of world->ego at theta_{t-1})
    R0T = torch.stack([torch.stack([c0, -s0], -1),
                       torch.stack([s0, c0], -1)], -2)
    # world -> ego(t)
    R1 = torch.stack([torch.stack([c1, s1], -1),
                      torch.stack([-s1, c1], -1)], -2)
    w = prev_xy.unsqueeze(1) + torch.einsum("bij,bkj->bki", R0T,
                                            prev_plan)
    q = torch.einsum("bij,bkj->bki", R1, w - xy.unsqueeze(1))
    pts1 = torch.cat([torch.zeros(B, 1, 2, device=plan.device,
                                  dtype=plan.dtype), plan], dim=1)
    interp = 0.2 * pts1[:, :-1, :] + 0.8 * pts1[:, 1:, :]
    return (q[..., 0] - interp[..., 0]).abs().mean(dim=1)
