"""
Recurrent State Space Model (RSSM) — adapted from original DITTO for
graph-based driving world model.

Supports both ego-only (Option 1) and multi-agent (Option 2) action inputs.
The RSSM itself is agnostic to the encoder type — it operates on fixed-size
embeddings regardless of whether they come from a CNN or a GNN.
"""

from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graph_ditto.models.gru import GRUCellStack


class RSSMCore(nn.Module):
    """Top-level RSSM that processes sequences via RSSMCell."""

    def __init__(
        self,
        embed_dim: int = 256,
        action_dim: int = 2,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        stoch_rank: int = 32,
        hidden_dim: int = 512,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.cell = RSSMCell(
            embed_dim, action_dim, deter_dim, stoch_dim,
            stoch_rank, hidden_dim, gru_layers,
        )

    def forward(
        self,
        embeds: Tensor,       # (T, B, embed_dim)
        actions: Tensor,      # (T, B, action_dim)
        resets: Tensor,       # (T, B)
        in_state: Tuple[Tensor, Tensor],
    ):
        reset_masks = ~resets.unsqueeze(2)
        T = embeds.shape[0]

        (h, z) = in_state
        posts, states_h, samples = [], [], []
        for i in range(T):
            post, (h, z) = self.cell.forward(
                actions[i], (h, z), reset_masks[i], embeds[i]
            )
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

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)


class RSSMCell(nn.Module):
    """Single-step RSSM transition cell."""

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int,
        stoch_dim: int,
        stoch_rank: int,
        hidden_dim: int,
        gru_layers: int,
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_rank = stoch_rank
        self.deter_dim = deter_dim
        norm = nn.LayerNorm

        self.init_h = nn.Parameter(torch.zeros(self.deter_dim))
        self.init_z = nn.Parameter(torch.zeros(self.stoch_dim * self.stoch_rank))

        self.z_mlp = nn.Linear(stoch_dim * (stoch_rank or 1), hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)
        self.in_norm = norm(hidden_dim, eps=1e-3)

        self.gru = GRUCellStack(hidden_dim, deter_dim, gru_layers)

        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_rank or 2))

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_rank or 2))

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
        pp = mlp(x)
        distr = self.zdistr(pp)
        sample = distr.rsample().reshape(B, -1)

        return pp, (h, sample)

    def batch_prior(self, h: Tensor) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_rank))
        distr = D.OneHotCategoricalStraightThrough(logits=logits.float())
        distr = D.independent.Independent(distr, 1)
        return distr
