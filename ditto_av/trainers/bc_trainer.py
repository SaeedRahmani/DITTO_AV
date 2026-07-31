from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..data import LatentBank
from ..models.nets import make_actor_critic
from .. import wandb_util


def train_bc(cfg: Config, bank: LatentBank, seed: int = 0):
    """Behavior cloning on world-model posterior features (latent BC).

    Uses the same feature space as the DITTO policies so the comparison
    isolates the effect of the on-policy latent-matching objective.
    Discrete actions: cross-entropy; continuous: Gaussian NLL.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    continuous = cfg.env.continuous
    # wp_head: same features (control-action WM), waypoint labels/output
    labels = bank.wp if cfg.env.wp_head else bank.action
    assert labels is not None, "wp_head needs wp arrays in the npz"
    policy = make_actor_critic(continuous, cfg.wm.feature_dim,
                               cfg.env.policy_action_dim, cfg.bc.hidden_dim,
                               cfg.bc.layers,
                               action_space=("waypoints" if cfg.env.wp_head
                                             else cfg.env.action_space)
                               ).to(device)
    opt = torch.optim.Adam(policy.actor.parameters(), lr=cfg.bc.lr)

    n = bank.feat.shape[0]
    n_val = max(1, n // 10)
    perm = rng.permutation(n)
    train_idx = torch.as_tensor(perm[:-n_val], device=device)
    val_idx = torch.as_tensor(perm[-n_val:], device=device)

    for step in range(1, cfg.bc.train_steps + 1):
        i = train_idx[torch.randint(len(train_idx), (cfg.bc.batch_size,),
                                    device=device)]
        if continuous:
            loss = -policy.dist(bank.feat[i]).log_prob(labels[i]).mean()
        else:
            loss = F.cross_entropy(policy.actor(bank.feat[i]), labels[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            wandb_util.log({"step": step, "loss": float(loss.item())},
                           prefix="bc")
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                if continuous:
                    pred = policy.act(bank.feat[val_idx])
                    metric = (pred - labels[val_idx]).abs().mean()
                    label = "val mae"
                else:
                    val_logits = policy.actor(bank.feat[val_idx])
                    metric = (val_logits.argmax(-1) ==
                              labels[val_idx]).float().mean()
                    label = "val acc"
            print(f"bc step {step:5d} | loss {loss.item():.3f} "
                  f"| {label} {metric:.3f}")
            wandb_util.log({"step": step, label.replace(" ", "_"):
                            float(metric)}, prefix="bc")

    ckpt = Path(cfg.dirs()["ckpt"]) / "bc.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"saved {ckpt}")
    return policy
