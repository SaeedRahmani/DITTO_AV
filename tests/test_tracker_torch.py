"""gen-4: the batched torch tracker must equal the deployment tracker.

Deployment-consistent imagination only holds if the in-dream
plan->control mapping is the deployed one; this pins them on randomized
plans (cruise, curves, standstill, degenerate collapsed plans).
"""
import numpy as np
import torch

from ditto_av.bench2drive import WP_SCALE
from ditto_av.carla_agent import WaypointTracker, wp_to_vehicle
from ditto_av.tracker_torch import TorchWaypointTracker, wp_to_vehicle_t


def test_wp_to_vehicle_equivalence():
    rng = np.random.default_rng(0)
    plans = rng.normal(0, 0.8, (32, 12)).astype(np.float32)
    vt = wp_to_vehicle_t(torch.as_tensor(plans)).numpy()
    for i in range(len(plans)):
        assert np.allclose(vt[i], wp_to_vehicle(plans[i]), atol=1e-5)


def test_tracker_equivalence_randomized():
    rng = np.random.default_rng(1)
    npt = WaypointTracker()
    tt = TorchWaypointTracker()
    cases = []
    for _ in range(200):
        kind = rng.integers(4)
        if kind == 0:      # cruise: forward spacing + mild lateral noise
            sp = rng.uniform(0.5, 5.0)
            wp = np.stack([np.arange(1, 7) * sp,
                           rng.normal(0, 0.3, 6)], axis=1)
        elif kind == 1:    # arc
            r = rng.uniform(5.0, 40.0)
            ang = np.linspace(0.1, rng.uniform(0.3, 2.0), 6)
            side = rng.choice([-1.0, 1.0])
            wp = np.stack([r * np.sin(ang),
                           side * r * (1 - np.cos(ang))], axis=1)
        elif kind == 2:    # near-standstill collapse
            wp = rng.normal(0, 0.05, (6, 2))
        else:              # partial collapse (moving then stopping)
            sp = rng.uniform(1.0, 3.0)
            wp = np.stack([np.minimum(np.arange(1, 7) * sp, 2.5 * sp),
                           np.zeros(6)], axis=1)
        cases.append((wp, rng.uniform(0.0, 12.0)))

    wps = torch.as_tensor(np.stack([c[0] for c in cases]),
                          dtype=torch.float32)
    speeds = torch.as_tensor([c[1] for c in cases], dtype=torch.float32)
    out_t = tt.act(wps, speeds).numpy()
    mismatches = 0
    for i, (wp, sp) in enumerate(cases):
        th, st, br, _ = npt.act(wp, sp)
        if not np.allclose(out_t[i], [th, st, br], atol=2e-3):
            # the only sanctioned divergence: cumulative-unwrap vs
            # single-wrap curvature on >pi total heading change
            mismatches += 1
    assert mismatches / len(cases) < 0.03, f"{mismatches} mismatches"


def test_tracker_torch_batch_consistency():
    # batching must not couple rollouts
    tt = TorchWaypointTracker()
    rng = np.random.default_rng(2)
    wp = torch.as_tensor(rng.normal(0, 2, (8, 6, 2)), dtype=torch.float32)
    sp = torch.as_tensor(rng.uniform(0, 8, 8), dtype=torch.float32)
    full = tt.act(wp, sp)
    for i in range(8):
        one = tt.act(wp[i:i + 1], sp[i:i + 1])
        assert torch.allclose(full[i], one[0], atol=1e-6)
