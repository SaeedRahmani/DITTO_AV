"""
Graph-based World Model for driving scenes.

Replaces the pixel-based WorldModelRSSM from original DITTO.
Architecture: SceneEncoder (GNN) + RSSM + SceneDecoder

Supports both:
  - Option 1 (ego-only): single ego action drives the RSSM
  - Option 2 (multi-agent): aggregated multi-agent actions drive the RSSM,
    with per-agent state updates for dreaming
"""

import torch
import torch.distributions as D
import torch.nn as nn

from graph_ditto.models.mlp import MLP
from graph_ditto.models.rssm import RSSMCore
from graph_ditto.models.scene_decoder import SceneDecoder
from graph_ditto.models.scene_encoder import SceneEncoder


def init_weights(m):
    if isinstance(m, (nn.Linear,)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.GRUCell):
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)


class GraphWorldModel(nn.Module):
    """
    World model with graph-based scene encoder.

    Phase 1 (training): encodes scene graphs → RSSM latent → decodes agent states
    Phase 2 (dreaming): rolls forward the RSSM using prior only (no observations)
    """

    def __init__(self, conf):
        super().__init__()
        self.multi_agent = getattr(conf, "multi_agent", False)
        self.kl_balance = conf.kl_balance
        self.kl_weight = conf.kl_weight

        # Scene graph encoder
        self.encoder = SceneEncoder(**conf.encoder_config)

        # RSSM dynamics model
        rssm_action_dim = conf.rssm_config.action_dim
        if self.multi_agent:
            # For multi-agent, we aggregate per-agent actions into a fixed-size vector
            self.action_aggregator = nn.Sequential(
                nn.Linear(conf.action_dim, conf.aggregated_action_dim),
                nn.ReLU(),
            )
            rssm_action_dim = conf.aggregated_action_dim

        rssm_kwargs = dict(conf.rssm_config)
        rssm_kwargs["action_dim"] = rssm_action_dim
        self.rssm_core = RSSMCore(**rssm_kwargs)

        # Feature dimension = deter_dim + stoch_dim * stoch_rank
        features_dim = conf.rssm_config.deter_dim + (
            conf.rssm_config.stoch_dim * conf.rssm_config.stoch_rank
        )
        self.features_dim = features_dim

        # Scene decoder: reconstruct agent states from latent
        self.decoder = SceneDecoder(
            features_dim=features_dim,
            **conf.decoder_config,
        )

        # For Option 2: per-agent state updater for dream rollouts
        if self.multi_agent:
            agent_embed_dim = self.encoder.hidden_dim
            self.agent_state_updater = AgentStateUpdater(
                agent_embed_dim=agent_embed_dim,
                action_dim=conf.action_dim,
                latent_dim=features_dim,
                hidden_dim=conf.agent_updater_hidden_dim,
            )

        for m in self.modules():
            init_weights(m)

    def init_state(self, batch_size):
        return self.rssm_core.init_state(batch_size)

    def encode_scene(self, obs):
        """Encode a batch of scene graphs into embeddings.
        
        Args:
            obs: dict with keys 'agent_features', 'agent_mask', 
                 'lane_features', 'lane_mask', 'ego_mask'
                 Each has shape (T, B, ...)
        
        Returns:
            ego_embeds: (T, B, out_dim) — for RSSM
            agent_embeds: (T, B, max_agents, hidden_dim) — for Option 2
        """
        T, B = obs["agent_features"].shape[:2]

        # Flatten T and B for batch encoding
        af = obs["agent_features"].reshape(T * B, *obs["agent_features"].shape[2:])
        am = obs["agent_mask"].reshape(T * B, *obs["agent_mask"].shape[2:])
        lf = obs["lane_features"].reshape(T * B, *obs["lane_features"].shape[2:])
        lm = obs["lane_mask"].reshape(T * B, *obs["lane_mask"].shape[2:])
        em = obs["ego_mask"].reshape(T * B, *obs["ego_mask"].shape[2:])

        ego_embed, agent_embeds = self.encoder(af, am, lf, lm, em)

        ego_embed = ego_embed.view(T, B, -1)
        agent_embeds = agent_embeds.view(T, B, *agent_embeds.shape[1:])

        return ego_embed, agent_embeds

    def prepare_actions(self, actions, agent_mask=None):
        """Prepare action input for the RSSM.
        
        Option 1: actions is (T, B, action_dim) — ego action, pass through.
        Option 2 (per-agent): actions is (T, B, max_agents, action_dim) — aggregate.
        Option 2 (WM training): actions is (T, B, action_dim) — embed ego action.
        """
        if not self.multi_agent:
            return actions

        if actions.dim() == 3:
            # Ego-only actions during WM training: (T, B, A) → embed directly
            T, B, A = actions.shape
            flat = actions.reshape(T * B, A)
            embedded = self.action_aggregator(flat)
            return embedded.view(T, B, -1)

        # Per-agent actions during dreaming: (T, B, N, A) → embed + mask + sum-pool
        T, B, N, A = actions.shape
        flat_actions = actions.reshape(T * B * N, A)
        embedded = self.action_aggregator(flat_actions)
        embedded = embedded.view(T, B, N, -1)

        # Mask and sum-pool
        mask = agent_mask.unsqueeze(-1).float()  # (T, B, N, 1)
        pooled = (embedded * mask).sum(dim=2)     # (T, B, agg_dim)
        return pooled

    def forward(self, obs, in_state):
        """Full forward pass: encode scenes + run RSSM.
        
        Used during world model training and feature extraction.
        """
        ego_embeds, agent_embeds = self.encode_scene(obs)
        actions = self.prepare_actions(obs["action"], obs.get("agent_mask"))

        prior, post, post_samples, features, hidden_states, out_states = (
            self.rssm_core.forward(ego_embeds, actions, obs["reset"], in_state)
        )
        return features, out_states, agent_embeds

    def dream(self, action, in_state):
        """Single dream step (no observation, prior only).
        
        Args:
            action: (B, action_dim) for Option 1, or (B, agg_action_dim) for Option 2
            in_state: (h, z) RSSM state
        
        Returns:
            (h, z) next RSSM state
        """
        _, (h, z) = self.rssm_core.cell.forward(action, in_state)
        return (h, z)

    def training_step(self, obs, in_state):
        """World model training step.
        
        Returns metrics, decoded output, out_states, and intermediate tensors.
        """
        ego_embeds, agent_embeds = self.encode_scene(obs)
        actions = self.prepare_actions(obs["action"], obs.get("agent_mask"))

        prior, post, post_samples, features, hidden_states, out_states = (
            self.rssm_core.forward(ego_embeds, actions, obs["reset"], in_state)
        )

        # Decoder reconstructs agent states from latent features
        loss_reconstr, loss_agents, _ = self.decoder(
            features, obs["agent_features"], obs["agent_mask"]
        )

        # KL divergence between prior and posterior
        d = self.rssm_core.zdistr
        dprior = d(prior)
        dpost = d(post)

        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)
        loss_kl_post = D.kl.kl_divergence(dpost, d(prior.detach()))
        loss_kl_prior = D.kl.kl_divergence(d(post.detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior

        loss = self.kl_weight * loss_kl + loss_reconstr

        batch_metrics = {
            "loss": loss,
            "loss_kl": loss_kl,
            "loss_kl_exact": loss_kl_exact,
            "loss_kl_post": loss_kl_post,
            "loss_kl_prior": loss_kl_prior,
            "loss_agents": loss_agents,
        }
        batch_metrics = {k: v.mean() for k, v in batch_metrics.items()}

        return batch_metrics, out_states, (features, agent_embeds)


class AgentStateUpdater(nn.Module):
    """
    Updates per-agent embeddings during dream rollouts (Option 2).

    When dreaming, we don't have new observations to re-encode with the GNN.
    This module updates the per-agent embeddings based on:
      - Their previous embedding
      - The action they took
      - The global RSSM latent state

    Uses a residual connection for stable learning.
    """

    def __init__(
        self,
        agent_embed_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.update_mlp = MLP(
            in_dim=agent_embed_dim + action_dim + latent_dim,
            out_dim=agent_embed_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
        )

    def forward(
        self,
        agent_embeds: torch.Tensor,    # (B, max_agents, agent_embed_dim)
        agent_actions: torch.Tensor,   # (B, max_agents, action_dim)
        latent: torch.Tensor,          # (B, latent_dim) — global RSSM features
        agent_mask: torch.Tensor,      # (B, max_agents) bool
    ) -> torch.Tensor:
        B, N, _ = agent_embeds.shape

        # Expand latent to all agents
        latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: [agent_embed, action, latent]
        combined = torch.cat([agent_embeds, agent_actions, latent_expanded], dim=-1)

        # Residual update
        delta = self.update_mlp(combined)
        updated = agent_embeds + delta

        # Zero out padding agents
        updated = updated * agent_mask.unsqueeze(-1).float()

        return updated
