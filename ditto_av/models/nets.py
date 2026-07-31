from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int,
        act=nn.ELU) -> nn.Sequential:
    dims = [in_dim] + [hidden_dim] * layers
    mods = []
    for i in range(layers):
        mods += [nn.Linear(dims[i], dims[i + 1]), act()]
    mods.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*mods)


class VectorEncoder(nn.Module):
    def __init__(self, obs_dim: int, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, embed_dim, layers=2)

    def forward(self, obs):
        return self.net(obs)


class VectorDecoder(nn.Module):
    def __init__(self, feature_dim: int, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(feature_dim, hidden_dim, obs_dim, layers=2)

    def forward(self, features):
        return self.net(features)

    def loss(self, features, obs):
        """Mean squared reconstruction error, summed over obs dims."""
        pred = self.forward(features)
        return ((pred - obs) ** 2).sum(-1), pred


class ActorCritic(nn.Module):
    """Discrete-action actor and critic over world-model features, with an
    EMA target critic for stable lambda-return bootstrapping."""

    def __init__(self, feature_dim: int, action_dim: int,
                 hidden_dim: int = 256, layers: int = 2):
        super().__init__()
        self.actor = mlp(feature_dim, hidden_dim, action_dim, layers)
        self.critic = mlp(feature_dim, hidden_dim, 1, layers)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    def dist(self, features) -> Categorical:
        return Categorical(logits=self.actor(features))

    def value(self, features):
        return self.critic(features).squeeze(-1)

    def target_value(self, features):
        return self.target_critic(features).squeeze(-1)

    @torch.no_grad()
    def update_target(self, tau: float):
        for p, tp in zip(self.critic.parameters(),
                         self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    @torch.no_grad()
    def act(self, features, stochastic: bool = False) -> int:
        d = self.dist(features)
        a = d.sample() if stochastic else d.logits.argmax(-1)
        return int(a.item())


class GaussianActorCritic(nn.Module):
    """Continuous-action counterpart of ActorCritic: diagonal Gaussian actor
    over world-model features, with the same EMA target critic.

    The actor outputs (mean, log_std) per action dim. Sampled actions are
    clamped to `low`/`high` before being fed to the world model; log-probs
    and KL use the unclamped Gaussian (standard for bounded driving controls
    like throttle/steer/brake).
    """

    # std floor 0.135 keeps Gaussian NLL bounded on near-binary controls
    # (B2D throttle/brake): std collapse on one mode otherwise causes NLL
    # spikes on the other and unstable BC training
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 1.0

    def __init__(self, feature_dim: int, action_dim: int,
                 hidden_dim: int = 256, layers: int = 2,
                 low: Optional[Sequence[float]] = None,
                 high: Optional[Sequence[float]] = None):
        super().__init__()
        self.action_dim = action_dim
        self.actor = mlp(feature_dim, hidden_dim, 2 * action_dim, layers)
        self.critic = mlp(feature_dim, hidden_dim, 1, layers)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        low = torch.tensor(low if low is not None else [-1.0] * action_dim)
        high = torch.tensor(high if high is not None else [1.0] * action_dim)
        self.register_buffer("low", low.float())
        self.register_buffer("high", high.float())

    def dist(self, features) -> Independent:
        mu, log_std = self.actor(features).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Independent(Normal(mu, log_std.exp()), 1)

    def clamp(self, action: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(action, self.high), self.low)

    def value(self, features):
        return self.critic(features).squeeze(-1)

    def target_value(self, features):
        return self.target_critic(features).squeeze(-1)

    @torch.no_grad()
    def update_target(self, tau: float):
        for p, tp in zip(self.critic.parameters(),
                         self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    @torch.no_grad()
    def act(self, features, stochastic: bool = False) -> torch.Tensor:
        d = self.dist(features)
        a = d.sample() if stochastic else d.base_dist.loc
        return self.clamp(a)


# act() clamp for the waypoint head, in /WP_SCALE units: measured over
# 43 clips the scaled forward component peaks at 2.04 (13.6 m/s * 3 s),
# lateral at 1.03; +-3.0 leaves headroom for faster highway clips while
# still bounding a diverged dream rollout. NLL on labels is unclamped.
WP_BOUND = 3.0


def make_actor_critic(continuous: bool, feature_dim: int, action_dim: int,
                      hidden_dim: int, layers: int,
                      action_space: str = "continuous"):
    """B2D control bounds (throttle, steer, brake) or symmetric waypoint
    bounds; highway stays discrete."""
    if continuous:
        if action_space == "waypoints":
            low = [-WP_BOUND] * action_dim
            high = [WP_BOUND] * action_dim
        else:
            low, high = [0.0, -1.0, 0.0], [1.0, 1.0, 1.0]
        return GaussianActorCritic(feature_dim, action_dim, hidden_dim,
                                   layers, low=low, high=high)
    return ActorCritic(feature_dim, action_dim, hidden_dim, layers)
