import numpy as np
import pytest

from ditto_av.data import TrajectoryData


def make_npz(tmp_path, n_eps=4, T=12, obs_dim=6, continuous=False, seed=0):
    rng = np.random.default_rng(seed)
    n = n_eps * T
    reset = np.zeros(n, dtype=bool)
    reset[::T] = True
    if continuous:
        action = rng.normal(size=(n, 3)).astype(np.float32)
    else:
        action = rng.integers(0, 5, size=n).astype(np.int64)
    p = tmp_path / "data.npz"
    np.savez(p, obs=rng.normal(size=(n, obs_dim)).astype(np.float32),
             action=action, reset=reset)
    return p


def test_episode_bounds(tmp_path):
    td = TrajectoryData([make_npz(tmp_path)])
    assert len(td.episodes) == 4
    assert all(e - s == 12 for s, e in td.episodes)
    assert td.discrete_actions


def test_wm_batch_layout_discrete(tmp_path):
    td = TrajectoryData([make_npz(tmp_path)])
    rng = np.random.default_rng(0)
    obs, act, reset = td.sample_wm_batch(5, 8, rng, action_dim=5)
    assert obs.shape == (8, 5, 6)
    assert act.shape == (8, 5, 5)
    assert reset.shape == (8, 5)
    # one-hot rows (or zero at reset)
    sums = act.sum(-1)
    assert ((sums == 1) | (sums == 0)).all()
    # wherever reset is set, the prev action must be zeroed
    assert (sums[reset] == 0).all()


def test_wm_batch_layout_continuous(tmp_path):
    td = TrajectoryData([make_npz(tmp_path, continuous=True)])
    assert not td.discrete_actions
    rng = np.random.default_rng(0)
    obs, act, reset = td.sample_wm_batch(5, 8, rng, action_dim=3)
    assert act.shape == (8, 5, 3)
    assert (act[reset].abs().sum(-1) == 0).all()


def test_prev_action_is_shifted(tmp_path):
    """The action fed at step t must be the dataset action at t-1."""
    p = make_npz(tmp_path, n_eps=1, T=12)
    td = TrajectoryData([p])
    d = np.load(p)
    rng = np.random.default_rng(3)
    obs, act, reset = td.sample_wm_batch(4, 12, rng, action_dim=5)
    # full-episode window: t=0 is reset, so act[1] == onehot(action[0])
    a0 = act[1].argmax(-1).numpy()
    assert (a0 == d["action"][0]).all()
