import numpy as np
import torch

from ditto_av.data import LatentBank
from ditto_av.rewards import LatentMatcher, lambda_return, max_cos


def test_max_cos_identity_and_scale():
    x = torch.randn(8, 16)
    assert torch.allclose(max_cos(x, x), torch.ones(8), atol=1e-5)
    # magnitude mismatch is penalized even with identical direction
    half = max_cos(x, 0.5 * x)
    assert (half < 1.0).all()
    assert torch.allclose(half, 0.5 * torch.ones(8), atol=1e-5)


def make_bank(H=4, D=8, T=30, n_eps=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = T * n_eps
    h = torch.randn(n, D, generator=g)
    z = torch.randn(n, D, generator=g)
    feat = torch.cat([h, z], -1)
    action = torch.randint(0, 5, (n,), generator=g)
    bounds = [(i * T, (i + 1) * T) for i in range(n_eps)]
    return LatentBank(h, z, feat, action, bounds, horizon=H)


def test_bank_windows_stay_in_episode():
    bank = make_bank(H=4, T=10, n_eps=3)
    assert bank.n_windows == 3 * (10 - 4)
    for s in bank.window_starts.tolist():
        ep = s // 10
        assert s + 4 < (ep + 1) * 10


def test_single_mode_targets_are_source_window():
    bank = make_bank()
    matcher = LatentMatcher(bank, mode="single")
    ids = torch.tensor([0, 5, 7])
    t = matcher.targets(ids)
    assert t.shape == (3, 1, 5, 8)
    assert torch.allclose(t[:, 0], bank.windows_h[ids])


def test_multi_mode_retrieves_similar_starts():
    """A window whose start latent is duplicated elsewhere must retrieve the
    duplicate among its top-k modes."""
    bank = make_bank(H=4, T=10, n_eps=3)
    # plant a duplicate start: window 0 start latent copied into window 12's
    w12_flat = bank.window_starts[12]
    bank.h[w12_flat] = bank.h[bank.window_starts[0]].clone()
    bank.windows_h = bank.h[bank.window_idx]
    matcher = LatentMatcher(bank, mode="multi", k=3)
    targets = matcher.targets(torch.tensor([0]))
    assert targets.shape == (1, 3, 5, 8)
    # the duplicate's continuation must appear among the retrieved modes
    found = any(torch.allclose(targets[0, j], bank.windows_h[12])
                for j in range(3))
    assert found


def test_rewards_prefer_matching_rollout():
    bank = make_bank()
    matcher = LatentMatcher(bank, mode="single")
    ids = torch.tensor([2])
    targets = matcher.targets(ids)
    # a rollout that exactly follows the expert window gets reward 1 each step
    perfect = bank.windows_h[ids].permute(1, 0, 2)  # (H+1, 1, D)
    r = matcher.rewards(perfect, targets)
    assert torch.allclose(r, torch.ones_like(r), atol=1e-5)
    # a random rollout gets strictly less
    r2 = matcher.rewards(torch.randn_like(perfect), targets)
    assert (r2 < 0.99).all()


def test_multi_mode_max_over_modes():
    """Multimodal reward must equal the best mode, not the average."""
    bank = make_bank()
    matcher = LatentMatcher(bank, mode="multi", k=4)
    ids = torch.tensor([3])
    targets = matcher.targets(ids)
    # follow mode j=1 exactly
    rollout = targets[0, 1].unsqueeze(1)  # (H+1, 1, D)
    r = matcher.rewards(rollout, targets)
    assert torch.allclose(r, torch.ones_like(r), atol=1e-5)


def test_contrastive_reward_subtracts_negative_baseline():
    """The contrastive reward subtracts the mean similarity to random expert
    windows; the exact-match rollout must still rank above a random rollout,
    and the baseline must actually change the raw value."""
    torch.manual_seed(0)
    bank = make_bank()
    raw = LatentMatcher(bank, mode="single", n_negatives=0)
    con = LatentMatcher(bank, mode="single", n_negatives=8)
    ids = torch.tensor([2])
    targets = raw.targets(ids)
    perfect = bank.windows_h[ids].permute(1, 0, 2)
    r_raw = raw.rewards(perfect, targets)
    r_con = con.rewards(perfect, targets)
    assert not torch.allclose(r_con, r_raw)
    # exact match still beats a random rollout under the contrastive reward
    r_rand = con.rewards(torch.randn_like(perfect), targets)
    assert r_con.mean() > r_rand.mean()


def test_lambda_return_matches_monte_carlo_when_lam_1():
    H, B = 4, 2
    g = torch.Generator().manual_seed(1)
    rewards = torch.randn(H, B, generator=g)
    values = torch.randn(H + 1, B, generator=g)
    gamma = 0.9
    ret = lambda_return(rewards, values, gamma=gamma, lam=1.0)
    # with lam=1: R_t = sum_k gamma^k r_{t+k} + gamma^{H-t} v_H
    for t in range(H):
        expected = values[-1] * gamma ** (H - t)
        for k in range(H - t):
            expected = expected + rewards[t + k] * gamma ** k
        assert torch.allclose(ret[t], expected, atol=1e-5)


def test_lambda_return_bootstraps_when_lam_0():
    H, B = 3, 2
    rewards = torch.ones(H, B)
    values = torch.arange(float((H + 1) * B)).reshape(H + 1, B)
    ret = lambda_return(rewards, values, gamma=0.5, lam=0.0)
    for t in range(H):
        assert torch.allclose(ret[t], rewards[t] + 0.5 * values[t + 1])
