"""v0.2 policy: per-actor token transformer + GRU memory + waypoint head.

Replaces the (RSSM features -> flat MLP) stack of v0.1: the capacity
moves into the policy's own scene encoder, whose structured attention
over actor tokens is the scaling knob (d_model/n_layers/gru_dim in
config). No prev-action input by design — the gen3_wp action-channel
feedback drift is a settled v0.1 lesson.

Interface mirrors GaussianActorCritic where it matters (dist/act/clamp,
LOG_STD bounds, WP_BOUND, EMA target critic) so trainer code stays
familiar. The recurrent state is explicit: callers own it (batched
sim rollouts, sequence BC, and the single-stream CARLA driver all step
the same cell).
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal

from .nets import WP_BOUND, mlp

N_FEAT = 7      # per-vehicle-row features in the obs layout
N_ROWS = 7      # ego + 6 neighbors
CORE_DIM = N_ROWS * N_FEAT
ROUTE_TOK = 8   # near/far: rel xy + 6-way one-hot command
LIGHT_TOK = 6


class TokenPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 1.0

    def __init__(self, obs_dim: int, action_dim: int = 12,
                 d_model: int = 192, n_layers: int = 3, n_heads: int = 4,
                 gru_dim: int = 512, head_hidden: int = 256,
                 head_layers: int = 2, with_route: bool = True,
                 with_lights: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.with_route = with_route
        self.with_lights = with_lights
        expected = CORE_DIM + (2 * ROUTE_TOK if with_route else 0) \
            + (LIGHT_TOK if with_lights else 0)
        assert obs_dim == expected, \
            f"obs_dim {obs_dim} != layout {expected}"
        self.n_tokens = N_ROWS + (2 if with_route else 0) \
            + (1 if with_lights else 0)

        self.embed_veh = nn.Linear(N_FEAT, d_model)
        self.embed_route = nn.Linear(ROUTE_TOK, d_model)
        self.embed_light = nn.Linear(LIGHT_TOK, d_model)
        self.type_emb = nn.Parameter(
            torch.zeros(self.n_tokens, d_model))
        nn.init.normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.pool = nn.Linear(2 * d_model, d_model)
        self.gru = nn.GRUCell(d_model, gru_dim)
        self.gru_dim = gru_dim
        feat_dim = gru_dim + d_model
        self.feature_dim = feat_dim
        self.actor = mlp(feat_dim, head_hidden, 2 * action_dim,
                         head_layers)
        self.critic = mlp(feat_dim, head_hidden, 1, head_layers)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        bound = torch.full((action_dim,), float(WP_BOUND))
        self.register_buffer("low", -bound)
        self.register_buffer("high", bound)

    # ---------------- encoding ---------------------------------------

    def encode(self, obs: Tensor) -> Tensor:
        """(B, obs_dim) -> (B, d_model) scene embedding."""
        B = obs.shape[0]
        toks = [self.embed_veh(
            obs[:, :CORE_DIM].view(B, N_ROWS, N_FEAT))]
        off = CORE_DIM
        if self.with_route:
            toks.append(self.embed_route(
                obs[:, off:off + 2 * ROUTE_TOK].view(B, 2, ROUTE_TOK)))
            off += 2 * ROUTE_TOK
        if self.with_lights:
            toks.append(self.embed_light(
                obs[:, off:off + LIGHT_TOK].view(B, 1, LIGHT_TOK)))
        x = torch.cat(toks, dim=1) + self.type_emb
        x = self.encoder(x)
        # ego token + mean pool, fused
        return self.pool(torch.cat([x[:, 0], x.mean(dim=1)], dim=-1))

    def init_state(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(batch_size, self.gru_dim,
                           device=device or self.type_emb.device)

    def step(self, emb: Tensor, h: Optional[Tensor]) -> Tensor:
        """One recurrent step: (B, d), (B, G) -> new (B, G)."""
        if h is None:
            h = self.init_state(emb.shape[0], emb.device)
        return self.gru(emb, h)

    def features(self, emb: Tensor, h: Tensor) -> Tensor:
        return torch.cat([h, emb], dim=-1)

    def unroll(self, obs_seq: Tensor, h: Optional[Tensor] = None
               ) -> Tuple[Tensor, Tensor]:
        """(T, B, obs) -> features (T, B, F), final state (B, G)."""
        T, B = obs_seq.shape[:2]
        embs = self.encode(obs_seq.reshape(T * B, -1)).view(T, B, -1)
        feats = []
        for t in range(T):
            h = self.step(embs[t], h)
            feats.append(self.features(embs[t], h))
        return torch.stack(feats), h

    # ---------------- heads ------------------------------------------

    def dist(self, features: Tensor) -> Independent:
        mu, log_std = self.actor(features).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Independent(Normal(mu, log_std.exp()), 1)

    def clamp(self, action: Tensor) -> Tensor:
        return torch.max(torch.min(action, self.high), self.low)

    def value(self, features: Tensor) -> Tensor:
        return self.critic(features).squeeze(-1)

    def target_value(self, features: Tensor) -> Tensor:
        return self.target_critic(features).squeeze(-1)

    @torch.no_grad()
    def update_target(self, tau: float):
        for p, tp in zip(self.critic.parameters(),
                         self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    @torch.no_grad()
    def act(self, features: Tensor, stochastic: bool = False) -> Tensor:
        d = self.dist(features)
        a = d.sample() if stochastic else d.base_dist.loc
        return self.clamp(a)


def make_token_policy(cfg) -> TokenPolicy:
    """Build from the full Config (env layout + clp dims)."""
    from ..bench2drive import extra_obs_layout
    with_route, with_lights = extra_obs_layout(cfg.env.extra_obs_dims,
                                               cfg.env.light_obs)
    c = cfg.clp
    return TokenPolicy(
        obs_dim=cfg.env.obs_dim, action_dim=2 * cfg.env.wp_k,
        d_model=c.d_model, n_layers=c.n_layers, n_heads=c.n_heads,
        gru_dim=c.gru_dim, head_hidden=c.head_hidden,
        head_layers=c.head_layers,
        with_route=with_route, with_lights=with_lights)
