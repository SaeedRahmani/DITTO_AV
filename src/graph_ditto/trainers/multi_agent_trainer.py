"""
Option 2: Multi-Agent Shared-Policy DITTO Trainer.

All agents in the scene share one policy network. During dreaming:
  1. The GNN encoder provides per-agent embeddings at the start.
  2. The shared policy takes (RSSM latent + per-agent embed) → per-agent action.
  3. The AgentStateUpdater maintains per-agent embeddings across dream steps.
  4. Actions are aggregated and fed to the RSSM for forward transition.
  5. DITTO reward = max_cos(h_policy, h_expert) at the scene level.

This produces diverse multi-agent behavior that collectively reproduces
the expert scene evolution.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from graph_ditto.data.common import EpisodeSampler, lambda_return, max_cos_reward
from graph_ditto.data.driving_dataset import (
    DrivingDataset,
    DrivingFeaturizer,
    driving_collate,
)
from graph_ditto.models.actor_critic import MultiAgentActorCritic
from graph_ditto.models.graph_world_model import GraphWorldModel

logger = logging.getLogger(__name__)


class MultiAgentTrainer:
    """Trains a shared multi-agent policy via DITTO (Option 2)."""

    def __init__(self, config, wm_checkpoint_path: str):
        self.config = config
        self.policy_conf = config.policy_config
        self.device = self.policy_conf.train_device

        self.batch_size = self.policy_conf.batch_size
        self.seq_length = self.policy_conf.seq_length
        self.dream_horizon = self.policy_conf.dream_horizon
        self.gamma = self.policy_conf.gamma
        self.lambda_gae = self.policy_conf.lambda_gae
        self.entropy_coef = self.policy_conf.entropy_coef
        self.value_coef = self.policy_conf.value_coef
        self.max_grad_norm = self.policy_conf.max_grad_norm
        self.train_steps = self.policy_conf.train_steps
        self.val_interval = self.policy_conf.val_interval
        self.max_agents = self.policy_conf.max_agents
        self.action_dim = self.policy_conf.action_dim
        self.checkpoint_path = Path(self.policy_conf.checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Load frozen world model (with multi-agent support)
        self.world_model = self._load_world_model(config.wm_config, wm_checkpoint_path)
        self.features_dim = self.world_model.features_dim
        self.deter_dim = config.wm_config.rssm_config.deter_dim

        # Build shared multi-agent policy
        self.policy = MultiAgentActorCritic(
            latent_dim=self.features_dim,
            agent_embed_dim=config.agent_config.agent_embed_dim,
            action_dim=config.agent_config.action_dim,
            hidden_dim=config.agent_config.hidden_dim,
            layers=config.agent_config.layers,
            action_scale=config.agent_config.action_scale,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=float(self.policy_conf.lr))

        # Build dataloaders — for multi-agent, we need per-agent embeddings
        # so we pass raw data through the WM encoder at each dream rollout
        self.train_loader, self.val_loader = self._build_dataloaders(config)

        self.global_step = 0
        self.best_val_reward = float("-inf")

        logger.info(
            "MultiAgentTrainer initialized: %d policy parameters",
            sum(p.numel() for p in self.policy.parameters()),
        )

    def _load_world_model(self, wm_config, checkpoint_path):
        """Load trained world model and freeze it."""
        wm = GraphWorldModel(wm_config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        wm.load_state_dict(ckpt["model_state_dict"])
        wm.eval()
        wm.requires_grad_(False)
        logger.info("World model loaded from %s", checkpoint_path)
        return wm

    def _build_dataloaders(self, config):
        """Build dataloaders that provide both latents and per-agent embeddings."""
        dataset = DrivingDataset(config.data_config)

        # Featurize to get RSSM latents + per-agent embeddings
        featurizer = DrivingFeaturizer(self.world_model, self.device)
        latent_data = featurizer.featurize(dataset, batch_size=64)

        n = latent_data["features"].shape[0]
        split_idx = int(n * 0.9)
        episode_starts = torch.where(latent_data["resets"])[0].numpy()

        train_ep_starts = episode_starts[episode_starts < split_idx]
        val_ep_starts = episode_starts[episode_starts >= split_idx] - split_idx

        train_loader = self._make_loader(
            {k: v[:split_idx] for k, v in latent_data.items()},
            train_ep_starts,
        )
        val_loader = self._make_loader(
            {k: v[split_idx:] for k, v in latent_data.items()},
            val_ep_starts,
        )
        return train_loader, val_loader

    def _make_loader(self, data, episode_starts):
        ds = _MultiAgentLatentDataset(data, self.seq_length)
        sampler = EpisodeSampler(
            len(ds), episode_starts, self.seq_length, self.batch_size
        )
        return DataLoader(ds, batch_sampler=sampler, collate_fn=_ma_collate)

    def _split_latent(self, latent):
        """Split features into (h, z) for RSSM dreaming."""
        stoch_size = self.features_dim - self.deter_dim
        h, z = latent.split([self.deter_dim, stoch_size], dim=-1)
        return h, z

    def unroll_policy(self, expert_latents, expert_agent_embeds, agent_masks):
        """
        Multi-agent dream rollout.

        Args:
            expert_latents: (T, B, features_dim) — expert RSSM latents
            expert_agent_embeds: (T, B, max_agents, agent_embed_dim)
            agent_masks: (T, B, max_agents) — True where agent exists

        Returns:
            ac_buffer, rewards, actions, entropy, target_buffer
        """
        ac_buffer = []
        target_buffer = []
        rewards = []
        actions_list = []
        entropies = []

        state = expert_latents[0]          # Current RSSM latent
        agent_embeds = expert_agent_embeds[0]  # Initial per-agent embeddings
        mask = agent_masks[0]              # Agent mask (assumed stable across dream)

        for k in range(self.dream_horizon):
            # Shared policy: get per-agent actions
            mean, std, value, target_value = self.policy.forward_t(
                state, agent_embeds, mask
            )
            actions, log_probs, mean_entropy = self.policy.sample_actions(mean, std, mask)
            # actions: (B, max_agents, action_dim)
            # log_probs: (B, max_agents) per-agent log_probs

            # Sum log_probs across valid agents for the scene-level policy gradient
            n_valid = mask.float().sum(dim=-1).clamp(min=1)
            scene_log_prob = (log_probs * mask.float()).sum(dim=-1) / n_valid

            # Aggregate actions for RSSM transition
            aggregated_action = self._aggregate_actions(actions, mask)

            # Dream step
            h, z = self._split_latent(state)
            with torch.no_grad():
                (h_next, z_next) = self.world_model.dream(aggregated_action, (h, z))
            next_state = torch.cat((h_next, z_next), dim=-1)

            # Update per-agent embeddings (no observation, use state updater)
            agent_embeds = self.world_model.agent_state_updater(
                agent_embeds, actions, state, mask
            )

            # DITTO reward: scene-level max_cos
            if k < self.dream_horizon - 1:
                reward = max_cos_reward(
                    next_state[..., : self.deter_dim],
                    expert_latents[k + 1][..., : self.deter_dim],
                )
            else:
                reward = torch.zeros(state.shape[0], device=self.device)

            ac_buffer.append((scene_log_prob, value.squeeze(-1)))
            target_buffer.append(target_value.squeeze(-1))
            rewards.append(reward)
            actions_list.append(actions)
            entropies.append(mean_entropy)

            state = next_state

        mean_entropy = torch.stack(entropies).mean()
        return ac_buffer, rewards, actions_list, mean_entropy, target_buffer

    def _aggregate_actions(self, actions, mask):
        """Aggregate per-agent actions for RSSM input via the WM's action aggregator."""
        B, N, A = actions.shape
        flat = actions.reshape(B * N, A)
        embedded = self.world_model.action_aggregator(flat)
        embedded = embedded.view(B, N, -1)
        mask_f = mask.unsqueeze(-1).float()
        return (embedded * mask_f).sum(dim=1)  # (B, aggregated_dim)

    @staticmethod
    def calculate_advantage(target_buffer, returns, norm=True):
        values = torch.stack(target_buffer)
        advantage = returns - values
        if norm:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return advantage

    @staticmethod
    def calculate_losses(ac_buffer, returns, advantage):
        log_probs = torch.stack([lp for lp, _ in ac_buffer])
        values = torch.stack([v for _, v in ac_buffer])

        policy_loss = -(log_probs[:-1] * advantage.detach()[:-1]).mean()
        value_loss = 0.5 * F.mse_loss(values[:-1], returns.detach()[:-1])
        return policy_loss, value_loss

    def train(self):
        """Main training loop."""
        pbar = tqdm(total=self.train_steps, desc="Multi-Agent Policy Training")
        self.policy.train()

        while self.global_step < self.train_steps:
            for batch in self.train_loader:
                if self.global_step >= self.train_steps:
                    break

                features = batch["features"].to(self.device)      # (T, B, feat)
                agent_embeds = batch["agent_embeds"].to(self.device)  # (T, B, N, E)
                agent_masks = batch["agent_masks"].to(self.device)   # (T, B, N)
                resets = batch["resets"].to(self.device)

                # Unroll shared policy
                ac_buffer, latent_rewards, ac_actions, entropy, target_buff = (
                    self.unroll_policy(features, agent_embeds, agent_masks)
                )

                # Returns and advantage
                returns = lambda_return(
                    latent_rewards, target_buff,
                    lambda_=self.lambda_gae, gamma=self.gamma,
                )
                advantage = self.calculate_advantage(target_buff, returns)
                policy_loss, value_loss = self.calculate_losses(
                    ac_buffer, returns, advantage
                )

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.global_step % 50 == 0:
                    metrics = {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": entropy.item(),
                        "mean_reward": torch.stack(latent_rewards).mean().item(),
                        "mean_return": returns.mean().item(),
                    }
                    self._log_metrics(metrics, "train")

                if self.global_step % self.val_interval == 0 and self.global_step > 0:
                    self._validate()
                    self.policy.update_critic_target()

                self.global_step += 1
                pbar.update(1)

        pbar.close()
        logger.info("Multi-agent training complete at step %d", self.global_step)

    @torch.no_grad()
    def _validate(self):
        self.policy.eval()
        total_reward = 0.0
        n_batches = 0

        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            agent_embeds = batch["agent_embeds"].to(self.device)
            agent_masks = batch["agent_masks"].to(self.device)
            resets = batch["resets"].to(self.device)

            _, latent_rewards, _, _, _ = self.unroll_policy(
                features, agent_embeds, agent_masks
            )
            total_reward += torch.stack(latent_rewards).mean().item()
            n_batches += 1

        if n_batches > 0:
            mean_reward = total_reward / n_batches
            logger.info("[Step %d] val | mean_reward=%.4f", self.global_step, mean_reward)

            if mean_reward > self.best_val_reward:
                self.best_val_reward = mean_reward
                path = self.checkpoint_path / "multi_agent_policy_best.pt"
                torch.save(self.policy.state_dict(), path)
                logger.info("New best model saved: %.4f", mean_reward)

        self.policy.train()

    def _log_metrics(self, metrics, prefix):
        msg = f"[Step {self.global_step}] {prefix} |"
        for k, v in metrics.items():
            msg += f" {k}={v:.4f}"
        logger.info(msg)


# ---- Helper classes for multi-agent latent data ----


class _MultiAgentLatentDataset(torch.utils.data.Dataset):
    """Dataset with RSSM features + per-agent embeddings as sequences."""

    def __init__(self, data, seq_length):
        self.features = data["features"]
        self.agent_embeds = data["agent_embeds"]
        self.agent_masks = data["agent_masks"]
        self.resets = data["resets"]
        self.seq_length = seq_length

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.seq_length)
        return {
            "features": self.features[sl],
            "agent_embeds": self.agent_embeds[sl],
            "agent_masks": self.agent_masks[sl],
            "resets": self.resets[sl],
        }


def _ma_collate(batch):
    """Collate dict-based samples into (T, B, ...) tensors."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        out[k] = torch.stack(tensors, dim=1)  # stack along B dimension
    return out
