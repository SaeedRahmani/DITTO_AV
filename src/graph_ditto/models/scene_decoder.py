"""
Scene decoder for Graph-DITTO world model.

Predicts agent states from the RSSM latent features. This serves the same role
as the CNN decoder in the original DITTO: it forces the latent space to retain
enough information about the scene to reconstruct agent states, acting as a
regularizer on latent quality.

Instead of reconstructing pixel images, this decoder predicts the state
(position, velocity, heading) of all agents in the scene.
"""

import torch
import torch.nn as nn
from torch import Tensor

from graph_ditto.models.mlp import MLP


class SceneDecoder(nn.Module):
    """
    Decodes RSSM latent features → predicted agent states.

    The decoder predicts [x, y, vx, vy, cos_h, sin_h] for each agent slot,
    masked to only penalize predictions for agents that actually exist.
    """

    def __init__(
        self,
        features_dim: int = 1536,
        max_agents: int = 32,
        agent_state_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        reconstruction_weight: float = 1.0,
    ):
        super().__init__()
        self.max_agents = max_agents
        self.agent_state_dim = agent_state_dim
        self.reconstruction_weight = reconstruction_weight

        self.decoder = MLP(
            features_dim,
            max_agents * agent_state_dim,
            hidden_dim,
            num_layers,
        )

    def forward(
        self,
        features: Tensor,          # (T, B, features_dim)
        target_agents: Tensor,     # (T, B, max_agents, agent_feat_dim)
        agent_mask: Tensor,        # (T, B, max_agents) bool
    ):
        """
        Returns:
            loss: scalar reconstruction loss (masked MSE)
            loss_raw: unweighted loss for logging
            pred: (T, B, max_agents, agent_state_dim) predicted agent states
        """
        T, B = features.shape[:2]

        # Predict all agent states from latent
        pred = self.decoder(features)  # (T, B, max_agents * state_dim)
        pred = pred.view(T, B, self.max_agents, self.agent_state_dim)

        # Only compare the decodeable dimensions (first agent_state_dim of target)
        target = target_agents[..., :self.agent_state_dim]

        # Masked MSE loss
        mask = agent_mask.unsqueeze(-1).float()  # (T, B, max_agents, 1)
        sq_error = (pred - target).pow(2) * mask
        n_valid = mask.sum().clamp(min=1)
        loss_raw = sq_error.sum() / n_valid

        loss = self.reconstruction_weight * loss_raw

        return loss, loss_raw, pred
