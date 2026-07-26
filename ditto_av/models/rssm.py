"""RSSM with categorical latents, adapted from the original DITTO codebase
(pydreamer-style) with configurable dimensions for vector observations."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gru import GRUCellStack


class RSSMCore(nn.Module):
    def __init__(self, embed_dim=128, action_dim=5, deter_dim=256, stoch_dim=16,
                 stoch_rank=16, hidden_dim=256, gru_layers=1):
        super().__init__()
        self.cell = RSSMCell(embed_dim, action_dim, deter_dim, stoch_dim,
                             stoch_rank, hidden_dim, gru_layers)

    def forward(self,
                embeds: Tensor,   # (T, B, E)
                actions: Tensor,  # (T, B, A)
                resets: Tensor,   # (T, B)
                in_state: Tuple[Tensor, Tensor],
                ):
        reset_masks = ~resets.unsqueeze(2)
        T = embeds.shape[0]

        (h, z) = in_state
        posts, states_h, samples = [], [], []
        for i in range(T):
            post, (h, z) = self.cell.forward(actions[i], (h, z),
                                             reset_masks[i], embeds[i])
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        posts = torch.stack(posts)
        states_h = torch.stack(states_h)
        samples = torch.stack(samples)
        priors = self.cell.batch_prior(states_h)
        features = self.to_feature(states_h, samples)

        return (
            priors,
            posts,
            samples,
            features,
            (states_h, samples),
            (h.detach(), z.detach()),
        )

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)


class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_rank,
                 hidden_dim, gru_layers):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_rank = stoch_rank
        self.deter_dim = deter_dim

        self.init_h = nn.Parameter(torch.zeros(deter_dim))
        self.init_z = nn.Parameter(torch.zeros(stoch_dim * stoch_rank))

        self.z_mlp = nn.Linear(stoch_dim * stoch_rank, hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)
        self.in_norm = nn.LayerNorm(hidden_dim, eps=1e-3)

        self.gru = GRUCellStack(hidden_dim, deter_dim, gru_layers)

        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = nn.LayerNorm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * stoch_rank)

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = nn.LayerNorm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * stoch_rank)

    def init_state(self, batch_size):
        return (
            torch.tile(self.init_h, (batch_size, 1)),
            torch.tile(self.init_z, (batch_size, 1)),
        )

    def forward(self, action, in_state, reset_mask=None, embed=None):
        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask

        B = action.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)

        if embed is not None:
            x = self.post_mlp_h(h) + self.post_mlp_e(embed)
            norm_layer, mlp = self.post_norm, self.post_mlp
        else:
            x = self.prior_mlp_h(h)
            norm_layer, mlp = self.prior_norm, self.prior_mlp

        x = norm_layer(x)
        x = F.elu(x)
        pp = mlp(x)  # posterior or prior logits
        distr = self.zdistr(pp)
        sample = distr.rsample().reshape(B, -1)

        return pp, (h, sample)

    def batch_prior(self, h: Tensor) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        return self.prior_mlp(x)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_rank))
        distr = D.OneHotCategoricalStraightThrough(logits=logits.float())
        return D.independent.Independent(distr, 1)
