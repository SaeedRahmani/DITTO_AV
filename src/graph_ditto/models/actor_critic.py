"""
Continuous-action Actor-Critic for Graph-DITTO.

Replaces the discrete categorical actor from the original Atari-based DITTO
with a Gaussian policy for continuous driving actions (steering, acceleration).

Supports:
  - Option 1: ego-only policy — actor takes RSSM latent → ego action
  - Option 2: shared multi-agent policy — actor takes RSSM latent + per-agent
    embedding → per-agent action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Independent, Normal

from graph_ditto.models.mlp import MLP


class EgoActorCritic(nn.Module):
    """
    Option 1: Single ego-agent actor-critic.

    Actor: RSSM latent → Gaussian distribution over [steering, acceleration]
    Critic: RSSM latent → scalar value estimate
    """

    def __init__(
        self,
        obs_dim: int = 1536,
        action_dim: int = 2,
        hidden_dim: int = 256,
        layers: int = 4,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale

        self.actor_mean = MLP(obs_dim, action_dim, hidden_dim, layers)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = MLP(obs_dim, 1, hidden_dim, layers)
        self.critic_target = MLP(obs_dim, 1, hidden_dim, layers)
        self.critic_target.requires_grad_(False)

    def forward(self, x: Tensor):
        """Returns action distribution and state value."""
        mean = torch.tanh(self.actor_mean(x)) * self.action_scale
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)
        value = self.critic(x)
        return dist, value

    def forward_t(self, x: Tensor):
        """Forward with target critic (for lambda-return targets)."""
        dist, value = self.forward(x)
        target_value = self.critic_target(x)
        return dist, value, target_value

    def forward_actor(self, x: Tensor):
        """Actor-only forward (for behavior cloning comparison)."""
        mean = torch.tanh(self.actor_mean(x)) * self.action_scale
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)
        return dist

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)


class MultiAgentActorCritic(nn.Module):
    """
    Option 2: Shared multi-agent actor-critic.

    Actor: (RSSM latent, per-agent embedding) → Gaussian action per agent
    Critic: RSSM latent → scalar scene-level value estimate

    All agents share the same actor weights. Different agents produce different
    actions because they have different per-agent embeddings from the GNN encoder.
    """

    def __init__(
        self,
        latent_dim: int = 1536,
        agent_embed_dim: int = 128,
        action_dim: int = 2,
        hidden_dim: int = 256,
        layers: int = 4,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.agent_embed_dim = agent_embed_dim

        # Actor takes concatenated [latent, agent_embed]
        actor_input_dim = latent_dim + agent_embed_dim
        self.actor_mean = MLP(actor_input_dim, action_dim, hidden_dim, layers)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic sees scene-level latent only
        self.critic = MLP(latent_dim, 1, hidden_dim, layers)
        self.critic_target = MLP(latent_dim, 1, hidden_dim, layers)
        self.critic_target.requires_grad_(False)

    def forward(
        self,
        latent: Tensor,         # (B, latent_dim) — RSSM features
        agent_embeds: Tensor,   # (B, max_agents, agent_embed_dim)
        agent_mask: Tensor,     # (B, max_agents) bool
    ):
        """Returns per-agent action distributions and scene value."""
        B, N, _ = agent_embeds.shape

        # Expand latent to all agent slots
        latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([latent_expanded, agent_embeds], dim=-1)

        # Per-agent action distributions
        mean = torch.tanh(self.actor_mean(combined)) * self.action_scale
        std = self.actor_log_std.exp().unsqueeze(0).unsqueeze(0).expand_as(mean)
        # mean: (B, N, action_dim), std: (B, N, action_dim)

        # Scene-level value
        value = self.critic(latent)  # (B, 1)

        return mean, std, value

    def forward_t(self, latent, agent_embeds, agent_mask):
        """Forward with target critic."""
        mean, std, value = self.forward(latent, agent_embeds, agent_mask)
        target_value = self.critic_target(latent)
        return mean, std, value, target_value

    def sample_actions(self, mean, std, agent_mask):
        """Sample actions for all agents (masked)."""
        dist = Independent(Normal(mean, std), 1)
        # Sample per-agent actions
        actions = dist.rsample()  # (B, N, action_dim)
        log_probs = Normal(mean, std).log_prob(actions).sum(dim=-1)  # (B, N)

        # Zero out actions for non-existent agents
        actions = actions * agent_mask.unsqueeze(-1).float()
        log_probs = log_probs * agent_mask.float()

        # Mean entropy across valid agents
        n_valid = agent_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        entropy = dist.entropy()  # (B, N)
        mean_entropy = (entropy * agent_mask.float()).sum(dim=-1) / n_valid.squeeze(-1)

        return actions, log_probs, mean_entropy.mean()

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
