"""gen-4 DITTO-WP units: wp probe, projected matcher, divergent retrieval."""
import numpy as np
import torch

from ditto_av.data import LatentBank
from ditto_av.rewards import LatentMatcher
from ditto_av.trainers.dwp_trainer import fit_wp_probe


def fake_bank(n=200, d=32, horizon=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, d, generator=g)
    z = torch.randn(n, d, generator=g)
    feat = torch.cat([h, z], dim=-1)
    action = torch.rand(n, 3, generator=g)
    wp = torch.randn(n, 12, generator=g)
    bank = LatentBank(h, z, feat, action, [(0, n)], horizon, wp=wp)
    return bank


def test_fit_wp_probe_recovers_linear_map():
    bank = fake_bank()
    W_true = torch.randn(32, 12)
    bank.wp = bank.h @ W_true  # perfectly linear labels
    W = fit_wp_probe(bank, ridge=1e-6)
    assert W.shape == (32, 12)
    assert torch.allclose(bank.h @ W, bank.wp, atol=1e-3)


def test_matcher_projection_and_shapes():
    bank = fake_bank()
    W = torch.randn(32, 12)
    m = LatentMatcher(bank, mode="multi", k=4, n_negatives=8, proj=W)
    ids = torch.arange(6)
    targets = m.targets(ids)
    assert targets.shape == (6, 4, bank.horizon + 1, 12)  # projected dim
    dreamed = torch.randn(bank.horizon + 1, 6, 32)
    r = m.rewards(m.project(dreamed), targets)
    assert r.shape == (bank.horizon, 6)
    # identity when projection off
    m0 = LatentMatcher(bank, mode="multi", k=4)
    assert m0.project(dreamed) is dreamed
    assert m0.targets(ids).shape == (6, 4, bank.horizon + 1, 32)


def test_retrieve_from_h_self_hit():
    bank = fake_bank()
    m = LatentMatcher(bank, mode="multi", k=3)
    # querying with an exact window-start latent must retrieve that
    # window as the top hit (the divergent-start relabel is exact on
    # the manifold, approximate off it)
    q = bank.windows_h[[7, 42], 0, :]
    t = m.retrieve_from_h(q)
    assert t.shape == (2, 3, bank.horizon + 1, 32)
    assert torch.allclose(t[0, 0], bank.windows_h[7])
    assert torch.allclose(t[1, 0], bank.windows_h[42])


def test_retrieve_projected_targets():
    bank = fake_bank()
    W = torch.randn(32, 12)
    m = LatentMatcher(bank, mode="multi", k=2, proj=W)
    t = m.retrieve_from_h(bank.windows_h[[3], 0, :])
    assert t.shape == (1, 2, bank.horizon + 1, 12)
    assert torch.allclose(t[0, 0], bank.windows_h[3] @ W)
