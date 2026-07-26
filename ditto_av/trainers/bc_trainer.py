from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..data import LatentBank
from ..models.nets import ActorCritic


def train_bc(cfg: Config, bank: LatentBank, seed: int = 0) -> ActorCritic:
    """Behavior cloning on world-model posterior features (latent BC).

    Uses the same feature space as the DITTO policies so the comparison
    isolates the effect of the on-policy latent-matching objective.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    policy = ActorCritic(cfg.wm.feature_dim, cfg.env.action_dim,
                         cfg.bc.hidden_dim, cfg.bc.layers).to(device)
    opt = torch.optim.Adam(policy.actor.parameters(), lr=cfg.bc.lr)

    n = bank.feat.shape[0]
    n_val = max(1, n // 10)
    perm = rng.permutation(n)
    train_idx = torch.as_tensor(perm[:-n_val], device=device)
    val_idx = torch.as_tensor(perm[-n_val:], device=device)

    for step in range(1, cfg.bc.train_steps + 1):
        i = train_idx[torch.randint(len(train_idx), (cfg.bc.batch_size,),
                                    device=device)]
        logits = policy.actor(bank.feat[i])
        loss = F.cross_entropy(logits, bank.action[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                val_logits = policy.actor(bank.feat[val_idx])
                acc = (val_logits.argmax(-1) ==
                       bank.action[val_idx]).float().mean()
            print(f"bc step {step:5d} | loss {loss.item():.3f} "
                  f"| val acc {acc:.3f}")

    ckpt = Path(cfg.dirs()["ckpt"]) / "bc.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"saved {ckpt}")
    return policy
