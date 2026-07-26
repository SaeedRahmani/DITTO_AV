from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical


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
