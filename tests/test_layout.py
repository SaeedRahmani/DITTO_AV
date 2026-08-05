"""v0.3.1 gates: torch layout query == numpy, manifest->town mapping,
per-frame dispatch, and the reward penalty wiring."""
from pathlib import Path

import numpy as np
import pytest
import torch

from ditto_av.layout import (LAYOUT_DIR, MARGIN, TownLayout,
                             manifest_towns, town_layout)
from ditto_av.layout_torch import LayoutQuery, TownLayoutTorch


def _lanes(n=400, lo=0.0, hi=100.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(lo, hi, size=(n, 2))
    hw = rng.uniform(1.0, 3.0, size=(n, 1))
    return np.concatenate([xy, hw], axis=1).astype(np.float32)


def test_torch_matches_numpy_synthetic():
    base = TownLayout(_lanes())
    tt = TownLayoutTorch(base)
    rng = np.random.default_rng(1)
    # near, far, and empty-neighborhood queries in one batch
    q = np.concatenate([rng.uniform(0, 100, size=(300, 2)),
                        rng.uniform(-60, 160, size=(100, 2)),
                        np.array([[500.0, 500.0], [-200.0, 40.0]])]
                       ).astype(np.float32)
    ref = base.off_drivable(q)
    got = tt.off_drivable(torch.as_tensor(q)).numpy()
    assert np.allclose(ref, got, atol=1e-4), \
        np.abs(ref - got).max()


@pytest.mark.skipif(not (LAYOUT_DIR / "Town01_lanes.npz").exists(),
                    reason="layout data not on this machine")
def test_torch_matches_numpy_real_town():
    base = town_layout("Town01")
    tt = TownLayoutTorch(base)
    rng = np.random.default_rng(2)
    lo = base.pts.min(axis=0) - 20
    hi = base.pts.max(axis=0) + 20
    q = rng.uniform(lo, hi, size=(2000, 2)).astype(np.float32)
    ref = base.off_drivable(q)
    got = tt.off_drivable(torch.as_tensor(q)).numpy()
    assert np.allclose(ref, got, atol=1e-3)


def test_manifest_towns_split_and_regex(tmp_path):
    names = [f"Foo_Town{t}_Route{i}_Weather{i}.tar.gz"
             for i, t in enumerate(
                 ["12", "01", "10HD", "13", "05", "12",
                  "15", "12", "07", "12", "03", "11"])]
    mf = tmp_path / "m.txt"
    # unsorted on disk; the split must sort first (run_b2d contract)
    mf.write_text("\n".join(reversed(names)) + "\n")
    train, val = manifest_towns(mf, val_every=6)
    assert len(train) == 10 and len(val) == 2
    ordered = sorted(n.removesuffix(".tar.gz") for n in names)
    exp = [ordered[i].split("_")[1].replace("10HD", "10")
           for i in range(12)]
    assert train == [t for i, t in enumerate(exp) if i % 6 != 5]
    assert val == [t for i, t in enumerate(exp) if i % 6 == 5]
    assert "Town10" in exp  # Town10HD normalizes to the npz name


def test_layout_query_dispatch(tmp_path):
    # two fake towns with well-separated geometry
    a = np.array([[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]], dtype=np.float32)
    b = np.array([[1000.0, 0.0, 4.0]], dtype=np.float32)
    np.savez(tmp_path / "Town91_lanes.npz", lanes=a)
    np.savez(tmp_path / "Town92_lanes.npz", lanes=b)
    lq = LayoutQuery(["Town91", "Town92"], [(0, 5), (5, 9)], 9,
                     layout_dir=tmp_path)
    frame = torch.tensor([0, 4, 5, 8])
    xy = torch.tensor([[0.0, 1.0], [0.0, 1.0],
                       [1000.0, 1.0], [1000.0, 1.0]])
    off = lq.off(frame, xy)
    # town A: dist 1 - hw 2 - margin; town B: dist 1 - hw 4 - margin
    assert torch.allclose(off[:2], torch.full((2,), 1 - 2 - MARGIN))
    assert torch.allclose(off[2:], torch.full((2,), 1 - 4 - MARGIN))
    # cross-check: a town-A point queried in a town-B frame is lost
    assert lq.off(torch.tensor([5]), torch.tensor([[0.0, 1.0]]))[0] > 90


def test_layout_penalty_in_rollout(tmp_path):
    from ditto_av.bench2drive import clips_to_npz
    from ditto_av.egosim import EgoSim, GlobalLog, RewardParams, SimParams
    from ditto_av.models.policy_v2 import TokenPolicy
    from ditto_av.trainers.reactive_trainer import _rollout
    import tests.test_egosim as te

    clip = te._make_clip(tmp_path, "straight")
    clips_to_npz([clip], tmp_path / "log.npz", with_route=True,
                 with_waypoints=6, with_global=True)
    log = GlobalLog([tmp_path / "log.npz"], device="cpu")

    class ConstLayout:
        def off(self, frame, xy, margin=MARGIN):
            return torch.full_like(xy[:, 0], 2.0)

    obs_dim = log.obs.shape[1]
    torch.manual_seed(0)
    pol = TokenPolicy(obs_dim, action_dim=12, d_model=32, n_layers=1,
                      n_heads=2, gru_dim=32, head_hidden=32)
    pol.eval()
    starts = log.window_starts(10, 6)[:4]

    def run(w):
        sim = EgoSim(log, SimParams(),
                     RewardParams(layout_penalty=w))
        sim.layout = ConstLayout()
        torch.manual_seed(1)
        return _rollout(pol, sim, log, starts, horizon=10, burn_in=4,
                        reactive=False, w1_thresh=0.25,
                        stochastic=False)

    r0 = run(0.0)
    r1 = run(0.5)
    # deterministic rollouts: identical trajectories, reward shifted by
    # exactly w * clamp(2.0, 0, 3.0) = 1.0 each step
    assert torch.allclose(r0[3] - 1.0, r1[3], atol=1e-6)
    offs = r1[9]
    assert offs is not None and (offs > 0).float().mean() == 1.0
