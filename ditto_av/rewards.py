from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import LatentBank


def max_cos(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """DITTO's latent similarity: dot(x, y) / max(|x|, |y|)^2.

    Equals 1 iff x == y; penalizes both direction and magnitude mismatch.
    Computed over the last dim.
    """
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    max_norm = torch.maximum(nx, ny).clamp_min(eps)
    return (x * y).sum(-1) / max_norm.pow(2)


class LatentMatcher:
    """Latent-matching rewards against expert windows.

    mode="single": reward is similarity to the *source* expert window the
        rollout was seeded from — exactly the DITTO reward.
    mode="multi": reward at each step is the max similarity over the K expert
        windows whose start latent is nearest (cosine) to the rollout's start
        latent. From near-identical traffic states, different expert episodes
        continue differently (overtake vs follow); nearest-mode matching
        rewards reproducing *any* expert mode instead of averaging them.
    """

    def __init__(self, bank: LatentBank, mode: str = "multi", k: int = 8,
                 n_negatives: int = 0):
        assert mode in ("single", "multi")
        self.bank = bank
        self.mode = mode
        self.k = k
        self.n_negatives = n_negatives
        starts = bank.windows_h[:, 0, :]  # (N, D)
        self._starts_normed = F.normalize(starts, dim=-1)

    @torch.no_grad()
    def targets(self, window_ids: Tensor) -> Tensor:
        """Expert target windows for rollouts seeded at `window_ids`.

        Returns (B, K, H+1, D); K = 1 in single mode.
        """
        if self.mode == "single":
            return self.bank.windows_h[window_ids].unsqueeze(1)
        q = self._starts_normed[window_ids]                # (B, D)
        sim = q @ self._starts_normed.T                    # (B, N)
        _, top = sim.topk(self.k, dim=-1)                  # (B, K)
        return self.bank.windows_h[top]                    # (B, K, H+1, D)

    @torch.no_grad()
    def rewards(self, dreamed_h: Tensor, targets: Tensor) -> Tensor:
        """Stepwise rewards for a dreamed rollout.

        dreamed_h: (H+1, B, D) latent states, index 0 = shared start.
        targets:   (B, K, H+1, D).
        Returns (H, B): reward at step t compares dreamed state t with the
        expert state t of the best-matching mode (max over K).

        With n_negatives > 0, the mean similarity to random expert windows is
        subtracted. Latent similarity to *any* plausible traffic state is high
        (~0.85-0.92 here), so the raw signal has a tiny dynamic range; the
        contrastive baseline cancels this generic-driving floor and leaves
        only behavior-specific matching.
        """
        d = dreamed_h[1:].permute(1, 0, 2).unsqueeze(1)    # (B, 1, H, D)
        t = targets[:, :, 1:, :]                           # (B, K, H, D)
        sim = max_cos(d, t)                                # (B, K, H)
        reward = sim.max(dim=1).values.permute(1, 0)       # (H, B)
        if self.n_negatives > 0:
            B = targets.shape[0]
            idx = torch.randint(self.bank.n_windows, (B, self.n_negatives),
                                device=dreamed_h.device)
            neg = self.bank.windows_h[idx][:, :, 1:, :]    # (B, M, H, D)
            neg_sim = max_cos(d, neg).mean(dim=1)          # (B, H)
            reward = reward - neg_sim.permute(1, 0)
        return reward


def lambda_return(rewards: Tensor, values: Tensor, gamma: float,
                  lam: float) -> Tensor:
    """Dreamer-style lambda-returns.

    rewards: (H, B) with rewards[t] received on arriving at state t+1.
    values:  (H+1, B) bootstrap values for states 0..H.
    Returns (H, B): R[t] is the return target for state t (0..H-1).
    """
    H = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    nxt = values[-1]
    for t in reversed(range(H)):
        nxt = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * nxt)
        returns[t] = nxt
    return returns
