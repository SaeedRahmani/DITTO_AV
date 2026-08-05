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
