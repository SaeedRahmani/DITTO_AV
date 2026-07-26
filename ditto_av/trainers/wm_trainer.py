from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data import TrajectoryData
from ..models.world_model import VectorWorldModel


def train_world_model(cfg: Config, data: TrajectoryData,
                      seed: int = 0) -> VectorWorldModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    wm = VectorWorldModel(data.obs_dim, cfg.env.action_dim, cfg.wm).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.wm.lr)

    # episode-level train/val split
    n_val = max(1, int(len(data.episodes) * cfg.wm.val_fraction))
    val_eps = set(range(len(data.episodes) - n_val, len(data.episodes)))
    train_data = _subset(data, [i for i in range(len(data.episodes))
                                if i not in val_eps])
    val_data = _subset(data, sorted(val_eps))

    wm.train()
    for step in range(1, cfg.wm.train_steps + 1):
        obs, act, reset = train_data.sample_wm_batch(
            cfg.wm.batch_size, cfg.wm.seq_len, rng, cfg.env.action_dim, device)
        loss, metrics, _ = wm.training_step(
            obs, act, reset, wm.init_state(cfg.wm.batch_size))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.wm.grad_clip)
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"wm step {step:5d} | loss {metrics['loss']:.3f} "
                  f"| recon {metrics['loss_recon']:.3f} "
                  f"| kl {metrics['loss_kl']:.3f} "
                  f"| H(prior) {metrics['entropy_prior']:.2f}")

    # validation
    wm.eval()
    with torch.no_grad():
        losses = []
        for _ in range(20):
            obs, act, reset = val_data.sample_wm_batch(
                cfg.wm.batch_size, cfg.wm.seq_len, rng, cfg.env.action_dim,
                device)
            _, m, _ = wm.training_step(
                obs, act, reset, wm.init_state(cfg.wm.batch_size))
            losses.append(float(m["loss_recon"]))
        print(f"wm val recon loss: {np.mean(losses):.3f}")

    ckpt = Path(cfg.dirs()["ckpt"]) / "world_model.pt"
    torch.save({"model_state_dict": wm.state_dict(),
                "obs_dim": data.obs_dim}, ckpt)
    print(f"saved {ckpt}")
    return wm


def _subset(data: TrajectoryData, episode_ids) -> TrajectoryData:
    sub = TrajectoryData.__new__(TrajectoryData)
    sub.obs = data.obs
    sub.action = data.action
    sub.reset = data.reset
    sub.discrete_actions = data.discrete_actions
    sub.episodes = [data.episodes[i] for i in episode_ids]
    return sub


def load_world_model(cfg: Config, obs_dim: int) -> VectorWorldModel:
    ckpt = torch.load(Path(cfg.dirs()["ckpt"]) / "world_model.pt",
                      map_location=cfg.device)
    wm = VectorWorldModel(ckpt["obs_dim"], cfg.env.action_dim, cfg.wm)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.to(cfg.device)
    wm.eval()
    wm.requires_grad_(False)
    return wm
