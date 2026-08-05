import numpy as np
import pytest

from ditto_av.smoothness import (FPS, episode_ids, plan_churn,
                                 sign_flips_per_100, summarize, yaw_rate)

DT = 1.0 / FPS
STRIDE_S = 0.5
K = 6


def test_episode_ids():
    ids = episode_ids(np.array([1, 0, 0, 1, 0], dtype=bool))
    assert ids.tolist() == [0, 0, 0, 1, 1]


def test_yaw_rate_and_boundaries():
    theta = np.array([0.0, 0.01, 0.02, 5.0, 5.01])
    ep = np.array([0, 0, 0, 1, 1])
    r = yaw_rate(theta, ep)
    assert r[0] == pytest.approx(0.1)
    assert np.isnan(r[2])            # episode boundary
    assert r[3] == pytest.approx(0.1)


def test_yaw_rate_wraps():
    r = yaw_rate(np.array([np.pi - 0.01, -np.pi + 0.01]))
    assert r[0] == pytest.approx(0.2, abs=1e-6)


def test_sign_flips_counting():
    db = 0.05
    # plain alternation: 2 flips over 3 ticks
    assert sign_flips_per_100(np.array([0.1, -0.1, 0.1]), db) \
        == pytest.approx(200.0 / 3)
    # deadband-crossing wobble still counts
    assert sign_flips_per_100(np.array([0.1, 0.0, -0.1]), db) \
        == pytest.approx(100.0 / 3)
    # jitter inside the deadband does not
    assert sign_flips_per_100(np.array([0.01, -0.01, 0.01]), db) == 0.0
    # episode boundary resets the sign memory
    ep = np.array([0, 1])
    assert sign_flips_per_100(np.array([0.1, -0.1]), db, ep) == 0.0


def test_summarize():
    s = summarize(np.array([1.0, -3.0, np.nan]))
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(2.0)
    assert s["max"] == pytest.approx(3.0)


def _circle(n_ticks: int, v: float = 5.0, omega: float = 0.2):
    """Analytic constant-turn trajectory in the compass convention
    (theta moves the ego along world (sin th, -cos th)) + the exactly
    consistent plans: future positions in the tick's ego frame."""
    tau = np.arange(n_ticks + K * int(STRIDE_S / DT) + 1) * DT
    theta = omega * tau
    pos = (v / omega) * np.stack([1.0 - np.cos(theta),
                                  -np.sin(theta)], axis=-1)
    xy, th = pos[:n_ticks], theta[:n_ticks]
    plans = np.zeros((n_ticks, K, 2))
    s = int(STRIDE_S / DT)
    for i in range(1, K + 1):
        fut = pos[i * s:i * s + n_ticks] - xy
        c, si = np.cos(th), np.sin(th)
        plans[:, i - 1, 0] = c * fut[:, 0] + si * fut[:, 1]
        plans[:, i - 1, 1] = -si * fut[:, 0] + c * fut[:, 1]
    return plans, xy, th


def test_plan_churn_zero_for_consistent_plans():
    plans, xy, th = _circle(40)
    ch = plan_churn(plans, xy, th)
    # only the polyline-interp error on the arc remains (~2 cm here)
    assert np.nanmax(ch["lat_mean"]) < 0.04
    # straight line: exactly zero
    n = 40
    xy = np.stack([np.zeros(n), -5.0 * DT * np.arange(n)], -1)
    plans = np.zeros((n, K, 2))
    plans[:, :, 1] = -np.arange(1, K + 1) * STRIDE_S * 5.0
    ch = plan_churn(plans, xy, np.zeros(n))
    assert np.nanmax(ch["lat_mean"]) < 1e-9


def test_plan_churn_catches_alternating_wobble():
    plans, xy, th = _circle(40)
    plans[:, :, 0] += 0.2 * (-1.0) ** np.arange(40)[:, None]
    ch = plan_churn(plans, xy, th)
    assert 0.3 < np.nanmean(ch["lat_mean"]) < 0.5
    assert 0.3 < np.nanmean(ch["wp1_proxy"]) < 0.5


def test_plan_churn_episode_boundary_nan():
    plans, xy, th = _circle(10)
    ep = np.array([0] * 5 + [1] * 5)
    ch = plan_churn(plans, xy, th, ep)
    assert np.isnan(ch["lat_mean"][4])
