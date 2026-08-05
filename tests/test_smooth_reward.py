"""v0.3.2 axis-1: the sigma_yawrate reward channel (V032_PLAN 1.3.1).

The channel must be exactly inert when off (sigma 0 OR yaw_rate not
passed) — v0.2/v0.3 numbers depend on that — and must price wobble
when on.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ditto_av.egosim import DT, EgoSim, GlobalLog, RewardParams, _wrap

T = 40


def _fake_log(omega: float = 0.3, v: float = 5.0):
    """Constant-turn expert episode as a minimal reward-only log."""
    th = 0.7 + omega * DT * torch.arange(T)
    xy = torch.zeros(T, 2)
    for t in range(1, T):
        xy[t, 0] = xy[t - 1, 0] + v * DT * torch.sin(th[t - 1])
        xy[t, 1] = xy[t - 1, 1] - v * DT * torch.cos(th[t - 1])
    ego = torch.zeros(T, 6)
    ego[:, 0:2] = xy
    ego[:, 2] = th
    ego[:, 3] = v
    yr = torch.zeros(T)
    yr[1:] = _wrap(th[1:] - th[:-1]) / DT
    return SimpleNamespace(ego=ego, ego_yawrate=yr,
                           ep_start=torch.zeros(T, dtype=torch.long),
                           ep_end=torch.full((T,), T, dtype=torch.long))


def _expert_state(log, t: int):
    frame = torch.tensor([t])
    xy = log.ego[t, 0:2].unsqueeze(0).clone()
    th = log.ego[t, 2].unsqueeze(0).clone()
    v = log.ego[t, 3].unsqueeze(0).clone()
    return frame, xy, th, v


def test_channel_inert_when_off():
    log = _fake_log()
    frame, xy, th, v = _expert_state(log, 20)
    yr = log.ego_yawrate[20].unsqueeze(0) + 2.0   # badly wrong yaw rate
    base = EgoSim(log, rew=RewardParams()).reward(frame, xy, th, v)
    # sigma 0 + yaw_rate passed
    r0 = EgoSim(log, rew=RewardParams(sigma_yawrate=0.0)).reward(
        frame, xy, th, v, yaw_rate=yr)
    # sigma on + yaw_rate not passed
    r1 = EgoSim(log, rew=RewardParams(sigma_yawrate=0.5)).reward(
        frame, xy, th, v)
    assert torch.allclose(r0, base) and torch.allclose(r1, base)


def test_channel_prices_wobble_and_rewards_match():
    log = _fake_log()
    sim = EgoSim(log, rew=RewardParams(sigma_yawrate=0.5))
    frame, xy, th, v = _expert_state(log, 20)
    good = sim.reward(frame, xy, th, v,
                      yaw_rate=log.ego_yawrate[20].unsqueeze(0))
    bad = sim.reward(frame, xy, th, v,
                     yaw_rate=log.ego_yawrate[20].unsqueeze(0) + 1.5)
    assert good.item() > 0.95          # expert-shaped motion ~free
    assert bad.item() < 0.05 * good.item()   # clp_rx-scale wobble priced
    # positive wobble and negative wobble cost alike
    bad2 = sim.reward(frame, xy, th, v,
                      yaw_rate=log.ego_yawrate[20].unsqueeze(0) - 1.5)
    assert bad2.item() == pytest.approx(bad.item(), rel=0.3)


def test_time_tolerance_still_applies():
    # a time-shifted expert state with the matching yaw rate should
    # keep most of its reward (the window max hides the shift)
    log = _fake_log()
    sim = EgoSim(log, rew=RewardParams(sigma_yawrate=0.5))
    frame = torch.tensor([20])
    xy = log.ego[23, 0:2].unsqueeze(0).clone()   # 3 frames late
    th = log.ego[23, 2].unsqueeze(0).clone()
    v = log.ego[23, 3].unsqueeze(0).clone()
    r = sim.reward(frame, xy, th, v,
                   yaw_rate=log.ego_yawrate[23].unsqueeze(0))
    assert r.item() > 0.9


def test_globallog_ego_yawrate(tmp_path: Path):
    n, A = 12, 2
    th = 0.5 + 0.4 * DT * np.arange(n, dtype=np.float32)
    ego = np.zeros((n, 6), dtype=np.float32)
    ego[:, 2] = th
    reset = np.zeros(n, dtype=bool)
    reset[0] = reset[7] = True                   # 2 episodes
    z = {"obs": np.zeros((n, 65), np.float32),
         "wp": np.zeros((n, 12), np.float32),
         "reset": reset, "ego_glob": ego,
         "act_glob": np.zeros((n, A, 8), np.float32),
         "route_glob": np.zeros((n, 6), np.float32),
         "light_glob": np.zeros((n, 4), np.float32)}
    p = tmp_path / "log.npz"
    np.savez(p, **z)
    log = GlobalLog([p])
    yr = log.ego_yawrate.numpy()
    assert yr[0] == 0.0 and yr[7] == 0.0         # episode starts
    assert yr[3] == pytest.approx(0.4, abs=1e-4)
    assert yr[8] == pytest.approx(0.4, abs=1e-4)
