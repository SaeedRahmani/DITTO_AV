"""
World model trainer for Graph-DITTO.

Trains the GraphWorldModel (SceneEncoder + RSSM + SceneDecoder) on
driving episode data using the ELBO objective:
  loss = KL_weight * D_KL(posterior || prior) + reconstruction_loss

This trainer is shared by both Option 1 and Option 2 — the world model
learns scene dynamics from expert demonstrations regardless of the
downstream policy architecture.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from graph_ditto.data.common import SequentialSampler
from graph_ditto.data.driving_dataset import DrivingDataset, driving_collate
from graph_ditto.models.graph_world_model import GraphWorldModel

logger = logging.getLogger(__name__)


class WMTrainer:
    """Trainer for the graph-based world model."""

    def __init__(self, config):
        self.config = config
        self.wm_conf = config.wm_train_config
        self.device = self.wm_conf.train_device

        self.batch_size = self.wm_conf.batch_size
        self.seq_length = self.wm_conf.seq_length
        self.max_grad_norm = self.wm_conf.max_grad_norm
        self.train_steps = self.wm_conf.train_steps
        self.checkpoint_interval = self.wm_conf.checkpoint_interval
        self.checkpoint_path = Path(self.wm_conf.checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Build dataset and dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders(config.data_config)

        # Build world model
        self.model = GraphWorldModel(config.wm_config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.wm_conf.lr,
            eps=self.wm_conf.eps,
        )
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        self.global_step = 0
        logger.info(
            "WMTrainer initialized: %d parameters",
            sum(p.numel() for p in self.model.parameters()),
        )

    def _build_dataloaders(self, data_config):
        """Build train/val dataloaders."""
        dataset = DrivingDataset(data_config)
        n = len(dataset)
        split_idx = int(n * 0.9)
        indices = torch.arange(n)
        train_set = Subset(dataset, indices[:split_idx])
        val_set = Subset(dataset, indices[split_idx:])

        train_sampler = SequentialSampler(
            len(train_set), self.seq_length, self.batch_size
        )
        val_sampler = SequentialSampler(
            len(val_set), self.seq_length, self.batch_size, init_idx=0
        )

        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            collate_fn=driving_collate,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_sampler=val_sampler,
            collate_fn=driving_collate,
            pin_memory=True,
        )
        logger.info(
            "Data: %d train, %d val transitions", len(train_set), len(val_set)
        )
        return train_loader, val_loader

    def train(self):
        """Main training loop."""
        pbar = tqdm(total=self.train_steps, desc="WM Training")
        self.model.train()

        while self.global_step < self.train_steps:
            in_state = self.model.init_state(self.batch_size)

            for batch in self.train_loader:
                if self.global_step >= self.train_steps:
                    break

                obs = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()

                with autocast(enabled=torch.cuda.is_available()):
                    metrics, out_state, _ = self.model.training_step(obs, in_state)

                in_state = tuple(s.detach() for s in out_state)

                self.scaler.scale(metrics["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Logging
                if self.global_step % 100 == 0:
                    self._log_metrics(metrics, "train")

                # Validation
                if self.global_step % 1000 == 0 and self.global_step > 0:
                    self._validate()

                # Checkpoint
                if (
                    self.global_step % self.checkpoint_interval == 0
                    and self.global_step > 0
                ):
                    self._save_checkpoint()

                self.global_step += 1
                pbar.update(1)

        self._save_checkpoint()
        pbar.close()
        logger.info("Training complete at step %d", self.global_step)

    @torch.no_grad()
    def _validate(self):
        """Run validation and log metrics."""
        self.model.eval()
        running = {}
        n_batches = 0
        in_state = self.model.init_state(self.batch_size)

        for batch in self.val_loader:
            obs = {k: v.to(self.device) for k, v in batch.items()}
            metrics, out_state, _ = self.model.training_step(obs, in_state)
            in_state = tuple(s.detach() for s in out_state)

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

        if n_batches > 0:
            avg = {k: v / n_batches for k, v in running.items()}
            self._log_metrics(avg, "val")

        self.model.train()

    def _log_metrics(self, metrics, prefix):
        """Log metrics (print for now; can integrate wandb)."""
        msg = f"[Step {self.global_step}] {prefix} |"
        for k, v in metrics.items():
            val = v.item() if torch.is_tensor(v) else v
            msg += f" {k}={val:.4f}"
        logger.info(msg)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        path = self.checkpoint_path / f"wm_step{self.global_step}.pt"
        torch.save(
            {
                "steps": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)
