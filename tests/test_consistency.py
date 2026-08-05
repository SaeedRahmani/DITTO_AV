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
