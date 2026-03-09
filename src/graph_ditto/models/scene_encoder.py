"""
Attention-based scene encoder for driving scenes.

Encodes a heterogeneous scene graph (agents + lanes) into fixed-size embeddings
using Transformer-style self-attention. This is equivalent to a GAT on a
fully-connected graph with masking, and follows the SceneTransformer / Wayformer
design pattern commonly used in autonomous driving.

Replaces the CNN encoder from the original pixel-based DITTO.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SceneEncoder(nn.Module):
    """
    Attention-based scene graph encoder.

    Takes padded agent features + lane features with masks, and produces:
      - ego_embed: (B, out_dim) — scene embedding centered on ego (for RSSM)
      - agent_embeds: (B, max_agents, hidden_dim) — per-agent embeddings (for Option 2)
    """

    def __init__(
        self,
        agent_feat_dim: int = 8,
        lane_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Type-specific input projections
        self.agent_proj = nn.Sequential(
            nn.Linear(agent_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.lane_proj = nn.Sequential(
            nn.Linear(lane_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Learned type embeddings to distinguish agents from lanes
        self.type_embed = nn.Embedding(2, hidden_dim)  # 0=agent, 1=lane

        # Transformer self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection for ego embedding
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        agent_features: Tensor,  # (B, max_agents, agent_feat_dim)
        agent_mask: Tensor,      # (B, max_agents) bool — True where agent exists
        lane_features: Tensor,   # (B, max_lanes, lane_feat_dim)
        lane_mask: Tensor,       # (B, max_lanes) bool — True where lane exists
        ego_mask: Tensor,        # (B, max_agents) bool — True only for ego
    ):
        B = agent_features.shape[0]
        N_a = agent_features.shape[1]
        device = agent_features.device

        # Project to common hidden space + add type embeddings
        agent_type_ids = torch.zeros(B, N_a, dtype=torch.long, device=device)
        lane_type_ids = torch.ones(B, lane_features.shape[1], dtype=torch.long, device=device)

        agent_h = self.agent_proj(agent_features) + self.type_embed(agent_type_ids)
        lane_h = self.lane_proj(lane_features) + self.type_embed(lane_type_ids)

        # Concatenate all nodes
        all_nodes = torch.cat([agent_h, lane_h], dim=1)        # (B, N_a+N_l, H)
        all_mask = torch.cat([agent_mask, lane_mask], dim=1)   # (B, N_a+N_l)

        # Transformer expects key_padding_mask: True = ignore (padding)
        padding_mask = ~all_mask

        # Self-attention across all scene nodes
        out = self.transformer(all_nodes, src_key_padding_mask=padding_mask)

        # Split back to get agent embeddings
        agent_embeds = out[:, :N_a]  # (B, max_agents, hidden_dim)

        # Extract ego embedding via ego_mask
        # ego_mask has exactly one True per batch element
        ego_embed = (agent_embeds * ego_mask.unsqueeze(-1).float()).sum(dim=1)
        ego_embed = self.out_proj(ego_embed)  # (B, out_dim)

        return ego_embed, agent_embeds
