"""Batched torch port of the deployment WaypointTracker core (gen-4).

Deployment-consistent imagination: DITTO rollouts on the waypoint head
must step the world model through the SAME plan->control mapping the
vehicle uses, or the policy is trained under different dynamics than it
is deployed with (the gen-3 gap). This ports the WaypointTracker's
pure-pursuit steering, spacing-derived target speed, and curvature cap
— batched over rollouts, torch-only, used inside no-grad dream loops
(the trainer is REINFORCE-style, so no gradients flow through it).
Recovery/creep/gap logic is deployment-side only (imagination has no
wedges). Equivalence with the numpy tracker is pinned by
tests/test_tracker_torch.py on randomized plans.

Curvature deviates from the numpy tracker in ONE documented way: the
heading change is the angle between the first and last valid segments
(single wrap), not a cumulative unwrap — identical for |dh| < pi,
which every 3 s expert plan satisfies.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .bench2drive import FPS, WP_SCALE, WP_STRIDE


def wp_to_vehicle_t(plan: Tensor) -> Tensor:
    """(B, 2k) scaled compass-frame action -> (B, k, 2) vehicle meters.

    Same fixed +90 deg rotation as carla_agent.wp_to_vehicle:
    forward (vehicle +x) = compass -y, lateral (vehicle +y) = compass +x.
    """
    wp = plan.view(plan.shape[0], -1, 2) * WP_SCALE
    return torch.stack([-wp[..., 1], wp[..., 0]], dim=-1)


class TorchWaypointTracker:
    """Batched plan -> (throttle, steer, brake); mirrors WaypointTracker."""

    def __init__(self, v_max: float = 8.0, a_lat: float = 2.5,
                 kp_speed: float = 0.5, kp_steer: float = 1.6,
                 lookahead_min: float = 3.0, lookahead_k: float = 1.0,
                 speed_gain: float = 1.0,
                 stride_s: float = WP_STRIDE / FPS):
        self.v_max, self.a_lat = v_max, a_lat
        self.kp_speed, self.kp_steer = kp_speed, kp_steer
        self.lmin, self.lk = lookahead_min, lookahead_k
        self.speed_gain = speed_gain
        self.stride_s = stride_s

    @torch.no_grad()
    def act(self, wp_vehicle: Tensor, ego_speed: Tensor) -> Tensor:
        """wp_vehicle (B, k, 2) meters; ego_speed (B,) m/s -> (B, 3)."""
        B, k, _ = wp_vehicle.shape
        dev = wp_vehicle.device
        pts = torch.cat([torch.zeros(B, 1, 2, device=dev), wp_vehicle], 1)
        seg = pts[:, 1:] - pts[:, :-1]                    # (B, k, 2)
        seglen = seg.norm(dim=-1)                         # (B, k)
        arc = torch.cat([torch.zeros(B, 1, device=dev),
                         seglen.cumsum(dim=1)], dim=1)    # (B, k+1)

        # target speed from the plan's own first-second spacing
        n_sp = min(2, k)
        v_wp = self.speed_gain * arc[:, n_sp] / (n_sp * self.stride_s)

        # curvature cap: angle between first and last valid segment
        valid = seglen > 0.3
        n_valid = valid.sum(dim=1)
        idx_first = torch.argmax(valid.float(), dim=1)
        idx_last = k - 1 - torch.argmax(valid.flip(1).float(), dim=1)
        b = torch.arange(B, device=dev)
        s0 = seg[b, idx_first]
        s1 = seg[b, idx_last]
        cosang = (s0 * s1).sum(-1) / (s0.norm(dim=-1) * s1.norm(dim=-1)
                                      ).clamp_min(1e-8)
        dh = torch.acos(cosang.clamp(-1.0, 1.0))
        kappa = dh / arc[:, -1].clamp_min(1e-3)
        v_curve = (self.a_lat / kappa.clamp_min(1e-4)).sqrt() \
            .clamp(1.5, self.v_max)
        v_curve = torch.where(n_valid >= 2, v_curve,
                              torch.full_like(v_curve, self.v_max))
        v_t = torch.minimum(torch.minimum(v_wp, v_curve),
                            torch.full_like(v_wp, self.v_max))

        # pure pursuit at interpolated arc-length lookahead
        ld = torch.maximum(torch.full_like(ego_speed, self.lmin),
                           self.lk * ego_speed)
        ld = torch.minimum(ld, arc[:, -1])
        j = (torch.searchsorted(arc, ld.unsqueeze(1), right=True)
             .squeeze(1) - 1).clamp(0, k - 1)
        frac = ((ld - arc[b, j]) / seglen[b, j].clamp_min(1e-6)).clamp(max=1.0)
        tp = pts[b, j] + frac.unsqueeze(-1) * seg[b, j]
        alpha = torch.atan2(tp[:, 1], tp[:, 0].clamp_min(0.3))
        steer = (self.kp_steer * alpha).clamp(-1.0, 1.0)
        steer = torch.where(arc[:, -1] > 0.5, steer,
                            torch.zeros_like(steer))

        err = v_t - ego_speed
        throttle = (self.kp_speed * err).clamp(0.0, 0.75)
        brake = torch.where(err < -1.0, (-0.6 * err).clamp(0.0, 1.0),
                            torch.zeros_like(err))
        throttle = torch.where(err >= 0.0, throttle,
                               torch.zeros_like(throttle))
        stop = (v_t < 0.15) & (ego_speed < 1.0)
        throttle = torch.where(stop, torch.zeros_like(throttle), throttle)
        brake = torch.where(stop, torch.ones_like(brake), brake)
        return torch.stack([throttle, steer, brake], dim=-1)
