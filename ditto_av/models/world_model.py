from __future__ import annotations

from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
from torch import Tensor

from ..config import WMConfig
from .nets import VectorDecoder, VectorEncoder
from .rssm import RSSMCore


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.GRUCell):
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)


class VectorWorldModel(nn.Module):
    """RSSM world model over vectorized driving observations."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VectorEncoder(obs_dim, cfg.embed_dim)
        self.rssm_core = RSSMCore(
            embed_dim=cfg.embed_dim, action_dim=action_dim,
            deter_dim=cfg.deter_dim, stoch_dim=cfg.stoch_dim,
            stoch_rank=cfg.stoch_rank, hidden_dim=cfg.hidden_dim,
            gru_layers=cfg.gru_layers)
        self.decoder = VectorDecoder(cfg.feature_dim, obs_dim)
        self.apply(init_weights)

    def init_state(self, batch_size: int):
        return self.rssm_core.init_state(batch_size)

    def observe(self, obs: Tensor, action: Tensor, reset: Tensor,
                in_state: Tuple[Tensor, Tensor]):
        """Filter a (T, B, ...) sequence through the posterior.

        `action` is the *previous* action (a_{t-1}) for each step t.
        Returns features (T, B, F), states (h, z) each (T, B, ...), out_state.
        """
        embed = self.encoder(obs)
        _, _, samples, features, states, out_state = self.rssm_core(
            embed, action, reset, in_state)
        return features, states, out_state

    def dream(self, action: Tensor, in_state: Tuple[Tensor, Tensor]):
        """One imagination step through the prior."""
        _, (h, z) = self.rssm_core.cell.forward(action, in_state)
        return (h, z)

    def training_step(self, obs: Tensor, action: Tensor, reset: Tensor,
                      in_state: Tuple[Tensor, Tensor]):
        embed = self.encoder(obs)
        prior, post, samples, features, _, out_state = self.rssm_core(
            embed, action, reset, in_state)

        loss_reconstr, _ = self.decoder.loss(features, obs)

        d = self.rssm_core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_post = D.kl.kl_divergence(dpost, d(prior.detach()))
        loss_kl_prior = D.kl.kl_divergence(d(post.detach()), dprior)
        loss_kl = (1 - self.cfg.kl_balance) * loss_kl_post + \
            self.cfg.kl_balance * loss_kl_prior
        # free nats: don't optimize KL below the floor
        loss_kl_clipped = torch.clamp(loss_kl, min=self.cfg.kl_free)

        loss = (self.cfg.kl_weight * loss_kl_clipped + loss_reconstr).mean()

        metrics = {
            "loss": loss.detach(),
            "loss_recon": loss_reconstr.mean().detach(),
            "loss_kl": loss_kl.mean().detach(),
            "entropy_prior": dprior.entropy().mean().detach(),
            "entropy_post": dpost.entropy().mean().detach(),
        }
        return loss, metrics, out_state
