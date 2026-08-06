"""Torch churn term must agree with the numpy audit metric exactly."""
import numpy as np
import torch

from ditto_av.consistency import plan_churn_lat
from ditto_av.smoothness import plan_churn

from test_smoothness import _circle


def test_matches_numpy_plan_churn():
    plans, xy, th = _circle(30)
    got = []
    for t in range(29):
        c = plan_churn_lat(
            torch.tensor(plans[t:t + 1]), torch.tensor(plans[t + 1:t + 2]),
            torch.tensor(xy[t:t + 1]), torch.tensor(xy[t + 1:t + 2]),
            torch.tensor(th[t:t + 1]), torch.tensor(th[t + 1:t + 2]))
        got.append(float(c[0]))
    ref = plan_churn(plans, xy, th)["lat_mean"]
    assert np.allclose(got, ref, atol=1e-9)


def test_batched_and_prices_wobble():
    plans, xy, th = _circle(12)
    wob = plans.copy()
    wob[:, :, 0] += 0.2 * (-1.0) ** np.arange(12)[:, None]
    prev = torch.tensor(np.stack([plans[5], wob[5]]))
    cur = torch.tensor(np.stack([plans[6], wob[6]]))
    c = plan_churn_lat(prev, cur,
                       torch.tensor(np.stack([xy[5], xy[5]])),
                       torch.tensor(np.stack([xy[6], xy[6]])),
                       torch.tensor(np.stack([th[5], th[5]])),
                       torch.tensor(np.stack([th[6], th[6]])))
    assert c[0].item() < 0.04
    assert c[1].item() > 0.3


def test_gradient_flows_through_both_plans():
    plans, xy, th = _circle(8)
    prev = torch.tensor(plans[3:4], requires_grad=True)
    cur = torch.tensor(plans[4:5], requires_grad=True)
    c = plan_churn_lat(prev, cur, torch.tensor(xy[3:4]),
                       torch.tensor(xy[4:5]), torch.tensor(th[3:4]),
                       torch.tensor(th[4:5]))
    c.sum().backward()
    assert prev.grad is not None and torch.isfinite(prev.grad).all()
    assert cur.grad is not None and torch.isfinite(cur.grad).all()
    assert prev.grad.abs().sum() > 0 and cur.grad.abs().sum() > 0


def test_proximity_gate_direction():
    from ditto_av.consistency import proximity_gate
    g = proximity_gate(torch.tensor([2.0, 12.0, 40.0]), 12.0, 3.0)
    assert g[0] < 0.05 and abs(g[1] - 0.5) < 1e-6 and g[2] > 0.95


def _arc_points(s, r=25.0):
    """Points on a constant-turn path at arc positions s (compass
    frame at the path start: heading -y, curving toward +x)."""
    a = np.asarray(s) / r
    return np.stack([r * (1 - np.cos(a)), -r * np.sin(a)], -1)


def _shape(prev_pts, cur_pts):
    from ditto_av.consistency import plan_shape_churn
    z2 = torch.zeros(1, 2)
    z1 = torch.zeros(1)
    return float(plan_shape_churn(
        torch.tensor(prev_pts[None], dtype=torch.float64),
        torch.tensor(cur_pts[None], dtype=torch.float64),
        z2.double(), z2.double(), z1.double(), z1.double())[0])


def test_shape_churn_free_speed_rescheduling():
    s_fast = 5.0 * 0.5 * np.arange(1, 7)      # cruising plan
    s_slow = 3.5 * 0.5 * np.arange(1, 7)      # same path, slower
    assert _shape(_arc_points(s_fast), _arc_points(s_slow)) < 0.04


def test_shape_churn_free_braking():
    s = 5.0 * 0.5 * np.arange(1, 7)
    s_brake = 0.5 * s                          # hard brake, same path
    assert _shape(_arc_points(s), _arc_points(s_brake)) < 0.04


def test_shape_churn_prices_lateral_shift():
    s = 5.0 * 0.5 * np.arange(1, 7)
    shifted = _arc_points(s) + np.array([0.25, 0.0])
    c = _shape(_arc_points(s), shifted)
    assert 0.15 < c < 0.3


def test_shape_churn_consistent_plans_with_motion():
    from ditto_av.consistency import plan_shape_churn
    plans, xy, th = _circle(20)
    c = plan_shape_churn(
        torch.tensor(plans[5:6]), torch.tensor(plans[6:7]),
        torch.tensor(xy[5:6]), torch.tensor(xy[6:7]),
        torch.tensor(th[5:6]), torch.tensor(th[6:7]))
    assert c[0].item() < 0.05


def test_shape_churn_grad_flows():
    from ditto_av.consistency import plan_shape_churn
    s = 5.0 * 0.5 * np.arange(1, 7)
    prev = torch.tensor(_arc_points(s)[None], requires_grad=True)
    cur = torch.tensor((_arc_points(s) + 0.1)[None], requires_grad=True)
    z2, z1 = torch.zeros(1, 2).double(), torch.zeros(1).double()
    c = plan_shape_churn(prev, cur, z2, z2, z1, z1)
    c.sum().backward()
    for g in (prev.grad, cur.grad):
        assert g is not None and torch.isfinite(g).all()
        assert g.abs().sum() > 0
