"""
Option 1: Ego-Only DITTO Trainer.

Trains a single ego-agent policy inside the learned Graph World Model
using DITTO's latent divergence reward (max_cos).

Pipeline:
  1. Load a trained GraphWorldModel (frozen).
  2. Featurize expert episodes → extract RSSM latents.
  3. Train EgoActorCritic by unrolling policy in the WM:
     - At each dream step, compute max_cos(h_policy, h_expert) reward.
     - Update actor-critic via policy gradient with lambda-returns.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from graph_ditto.data.common import EpisodeSampler, lambda_return, max_cos_reward
from graph_ditto.data.driving_dataset import (
    DrivingDataset,
    DrivingFeaturizer,
    driving_collate,
)
from graph_ditto.models.actor_critic import EgoActorCritic
from graph_ditto.models.graph_world_model import GraphWorldModel

logger = logging.getLogger(__name__)


class EgoTrainer:
    """Trains a single ego-agent policy via DITTO (Option 1)."""

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
        self.checkpoint_path = Path(self.policy_conf.checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Load frozen world model
        self.world_model = self._load_world_model(config.wm_config, wm_checkpoint_path)
        self.features_dim = self.world_model.features_dim

        # Build policy
        self.policy = EgoActorCritic(
            obs_dim=self.features_dim,
            action_dim=config.agent_config.action_dim,
            hidden_dim=config.agent_config.hidden_dim,
            layers=config.agent_config.layers,
            action_scale=config.agent_config.action_scale,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=float(self.policy_conf.lr))

        # Build featurized dataloader
        self.train_loader, self.val_loader = self._build_dataloaders(config)

        self.global_step = 0
        self.best_val_reward = float("-inf")

        logger.info(
            "EgoTrainer initialized: %d policy parameters",
            sum(p.numel() for p in self.policy.parameters()),
        )

    def _load_world_model(self, wm_config, checkpoint_path):
        """Load a trained world model and freeze it."""
        wm = GraphWorldModel(wm_config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        wm.load_state_dict(ckpt["model_state_dict"])
        wm.eval()
        wm.requires_grad_(False)
        logger.info("World model loaded from %s", checkpoint_path)
        return wm

    def _build_dataloaders(self, config):
        """Featurize expert data and build dataloaders for policy training."""
        dataset = DrivingDataset(config.data_config)

        # Featurize: run expert data through WM to get RSSM latents
        featurizer = DrivingFeaturizer(self.world_model, self.device)
        latent_data = featurizer.featurize(dataset, batch_size=64)

        # latent_data: dict with 'features' (N, features_dim), 'actions' (N, action_dim),
        # 'resets' (N,), 'agent_masks' (N, max_agents)
        n = latent_data["features"].shape[0]
        split_idx = int(n * 0.9)

        # Find episode boundaries
        episode_starts = torch.where(latent_data["resets"])[0].numpy()

        train_features = latent_data["features"][:split_idx]
        train_actions = latent_data["actions"][:split_idx]
        train_resets = latent_data["resets"][:split_idx]

        val_features = latent_data["features"][split_idx:]
        val_actions = latent_data["actions"][split_idx:]
        val_resets = latent_data["resets"][split_idx:]

        # Episode starts for train/val split
        train_ep_starts = episode_starts[episode_starts < split_idx]
        val_ep_starts = episode_starts[episode_starts >= split_idx] - split_idx

        train_loader = self._make_loader(
            train_features, train_actions, train_resets, train_ep_starts
        )
        val_loader = self._make_loader(
            val_features, val_actions, val_resets, val_ep_starts
        )

        return train_loader, val_loader

    def _make_loader(self, features, actions, resets, episode_starts):
        """Create a DataLoader from featurized tensors."""
        ds = _LatentDataset(features, actions, resets, self.seq_length)
        sampler = EpisodeSampler(
            len(ds), episode_starts, self.seq_length, self.batch_size
        )
        return DataLoader(ds, batch_sampler=sampler, collate_fn=_latent_collate)

    def wm_step(self, latent, action):
        """Single dream step in the world model."""
        with torch.no_grad():
            cell = self.world_model.rssm_core.cell
            h, z = latent.split(
                [cell.deter_dim, cell.stoch_dim * cell.stoch_rank], dim=-1
            )
            (h, z) = self.world_model.dream(action, (h, z))
            return torch.cat((h, z), dim=-1)

    def unroll_policy(self, expert_latents, resets):
        """
        Unroll the policy in the world model and compute DITTO rewards.

        Args:
            expert_latents: (T, B, features_dim) — expert RSSM features
            resets: (T, B) — episode boundary flags

        Returns:
            ac_buffer, rewards, actions, entropy, target_buffer
        """
        ac_buffer = []
        target_buffer = []
        rewards = []
        actions_list = []
        entropies = []

        state = expert_latents[0]  # Start from expert's initial latent

        for k in range(self.dream_horizon):
            # Get action from policy
            dist, value, target_value = self.policy.forward_t(state)
            action = dist.rsample()  # Reparameterized sample
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Dream step
            next_state = self.wm_step(state, action)

            # DITTO reward: max_cos between policy-dreamed h and expert h
            if k < self.dream_horizon - 1:
                deter_dim = self.world_model.rssm_core.cell.deter_dim
                reward = max_cos_reward(
                    next_state[..., :deter_dim],
                    expert_latents[k + 1][..., :deter_dim],
                )
            else:
                reward = torch.zeros(state.shape[0], device=self.device)

            ac_buffer.append((log_prob, value.squeeze(-1)))
            target_buffer.append(target_value.squeeze(-1))
            rewards.append(reward)
            actions_list.append(action)
            entropies.append(entropy)

            state = next_state

        mean_entropy = torch.stack(entropies).mean()
        return ac_buffer, rewards, torch.stack(actions_list), mean_entropy, target_buffer

    @staticmethod
    def calculate_advantage(target_buffer, returns, norm=True):
        """GAE-style advantage estimation."""
        values = torch.stack(target_buffer)
        advantage = returns - values
        if norm:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return advantage

    @staticmethod
    def calculate_losses(ac_buffer, returns, advantage):
        """Policy gradient + value loss."""
        log_probs = torch.stack([lp for lp, _ in ac_buffer])
        values = torch.stack([v for _, v in ac_buffer])

        policy_loss = -(log_probs[:-1] * advantage.detach()[:-1]).mean()
        value_loss = 0.5 * F.mse_loss(values[:-1], returns.detach()[:-1])

        return policy_loss, value_loss

    def train(self):
        """Main policy training loop."""
        pbar = tqdm(total=self.train_steps, desc="Ego Policy Training")
        self.policy.train()

        while self.global_step < self.train_steps:
            for batch in self.train_loader:
                if self.global_step >= self.train_steps:
                    break

                latents, resets, actions = [x.to(self.device) for x in batch]

                # Unroll policy in WM
                ac_buffer, latent_rewards, ac_actions, entropy, target_buff = (
                    self.unroll_policy(latents, resets)
                )

                # Compute returns and advantage
                returns = lambda_return(
                    latent_rewards, target_buff,
                    lambda_=self.lambda_gae, gamma=self.gamma,
                )
                advantage = self.calculate_advantage(target_buff, returns)
                policy_loss, value_loss = self.calculate_losses(
                    ac_buffer, returns, advantage
                )

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Logging
                if self.global_step % 50 == 0:
                    metrics = {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": entropy.item(),
                        "mean_reward": torch.stack(latent_rewards).mean().item(),
                        "mean_return": returns.mean().item(),
                    }
                    self._log_metrics(metrics, "train")

                # Validation
                if self.global_step % self.val_interval == 0 and self.global_step > 0:
                    self._validate()
                    self.policy.update_critic_target()

                self.global_step += 1
                pbar.update(1)

        pbar.close()
        logger.info("Ego policy training complete at step %d", self.global_step)

    @torch.no_grad()
    def _validate(self):
        """Validation pass."""
        self.policy.eval()
        total_reward = 0.0
        n_batches = 0

        for batch in self.val_loader:
            latents, resets, actions = [x.to(self.device) for x in batch]
            _, latent_rewards, _, _, _ = self.unroll_policy(latents, resets)
            total_reward += torch.stack(latent_rewards).mean().item()
            n_batches += 1

        if n_batches > 0:
            mean_reward = total_reward / n_batches
            logger.info("[Step %d] val | mean_reward=%.4f", self.global_step, mean_reward)

            if mean_reward > self.best_val_reward:
                self.best_val_reward = mean_reward
                path = self.checkpoint_path / f"ego_policy_best.pt"
                torch.save(self.policy.state_dict(), path)
                logger.info("New best model saved: %.4f", mean_reward)

        self.policy.train()

    def _log_metrics(self, metrics, prefix):
        msg = f"[Step {self.global_step}] {prefix} |"
        for k, v in metrics.items():
            msg += f" {k}={v:.4f}"
        logger.info(msg)


# ---- Helper classes for featurized data ----


class _LatentDataset(torch.utils.data.Dataset):
    """Dataset wrapping pre-computed RSSM latent features as sequences."""

    def __init__(self, features, actions, resets, seq_length):
        self.features = features
        self.actions = actions
        self.resets = resets
        self.seq_length = seq_length

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        sl = slice(idx, idx + self.seq_length)
        return self.features[sl], self.actions[sl], self.resets[sl]


def _latent_collate(batch):
    """Collate for latent dataset: stack into (T, B, ...) format."""
    features, actions, resets = zip(*batch)
    return (
        torch.stack(features, dim=1),   # (T, B, features_dim)
        torch.stack(resets, dim=1),      # (T, B)
        torch.stack(actions, dim=1),     # (T, B, action_dim)
    )
