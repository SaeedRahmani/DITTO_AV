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
