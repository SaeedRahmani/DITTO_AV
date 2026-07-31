from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..data import LatentBank
from ..models.nets import make_actor_critic
from ..models.world_model import VectorWorldModel
from ..rewards import LatentMatcher, lambda_return
from .. import wandb_util


def train_latent_policy(cfg: Config, wm: VectorWorldModel, bank: LatentBank,
                        reward_mode: str, seed: int = 0,
                        name: str = None):
    """DITTO policy learning: on-policy RL in imagination, rewarded by
    latent similarity to expert windows (single or multimodal matching)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    acfg = cfg.ac
    continuous = cfg.env.continuous
    name = name or f"ditto_{reward_mode}"

    policy = make_actor_critic(continuous, cfg.wm.feature_dim,
                               cfg.env.action_dim, acfg.hidden_dim,
                               acfg.layers,
                               action_space=cfg.env.action_space).to(device)

    # BC anchor: start from the cloned policy and stay in its trust region
    bc_policy = None
    bc_ckpt = Path(cfg.dirs()["ckpt"]) / "bc.pt"
    if acfg.bc_init and bc_ckpt.exists():
        bc_policy = make_actor_critic(continuous, cfg.wm.feature_dim,
                                      cfg.env.action_dim, cfg.bc.hidden_dim,
                                      cfg.bc.layers,
                                      action_space=cfg.env.action_space
                                      ).to(device)
        bc_policy.load_state_dict(torch.load(bc_ckpt, map_location=device))
        policy.actor.load_state_dict(bc_policy.actor.state_dict())
        bc_policy.requires_grad_(False)
        bc_policy.eval()

    actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=acfg.actor_lr)
    critic_opt = torch.optim.Adam(policy.critic.parameters(),
                                  lr=acfg.critic_lr)
    matcher = LatentMatcher(bank, mode=reward_mode, k=acfg.k_modes,
                            n_negatives=acfg.n_negatives)
    H = acfg.horizon

    for step in range(1, acfg.train_steps + 1):
        window_ids = bank.sample_windows(acfg.batch_size, rng)
        h, z = bank.start_states(window_ids)

        # --- unroll the policy in imagination (no grad through dynamics) ---
        feats, actions = [], []
        with torch.no_grad():
            for _ in range(H):
                feat = torch.cat((h, z), dim=-1)
                a = policy.dist(feat).sample()
                feats.append(feat)
                actions.append(a)
                a_wm = (policy.clamp(a) if continuous
                        else F.one_hot(a, cfg.env.action_dim).float())
                h, z = wm.dream(a_wm, (h, z))
            feats.append(torch.cat((h, z), dim=-1))
        feats = torch.stack(feats)          # (H+1, B, F)
        actions = torch.stack(actions)      # (H, B) or (H, B, A)
        dreamed_h = feats[..., :cfg.wm.deter_dim]

        # --- rewards and return targets ---
        targets = matcher.targets(window_ids)
        rewards = matcher.rewards(dreamed_h, targets)      # (H, B)
        with torch.no_grad():
            values_t = policy.target_value(feats)          # (H+1, B)
        returns = lambda_return(rewards, values_t, acfg.gamma, acfg.lam)

        # --- losses ---
        dist = policy.dist(feats[:-1])
        logp = dist.log_prob(actions)
        advantage = returns - values_t[:-1]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        policy_loss = -(logp * advantage.detach()).mean()
        entropy = dist.entropy().mean()
        actor_loss = policy_loss - acfg.entropy_coef * entropy
        if bc_policy is not None and acfg.bc_kl_coef > 0:
            with torch.no_grad():
                bc_dist = bc_policy.dist(feats[:-1])
            kl_bc = torch.distributions.kl.kl_divergence(dist, bc_dist).mean()
            actor_loss = actor_loss + acfg.bc_kl_coef * kl_bc

        values = policy.value(feats[:-1].detach())
        critic_loss = 0.5 * F.mse_loss(values, returns.detach())

        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),
                                       acfg.grad_clip)
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),
                                       acfg.grad_clip)
        critic_opt.step()
        policy.update_target(acfg.target_tau)

        if step % 100 == 0 or step == 1:
            wandb_util.log({"step": step,
                            "reward": float(rewards.mean()),
                            "return": float(returns.mean()),
                            "entropy": float(entropy),
                            "actor_loss": float(actor_loss),
                            "critic_loss": float(critic_loss)},
                           prefix=name)
        if step % 500 == 0 or step == 1:
            # imitation quality at the seed step against expert actions
            with torch.no_grad():
                expert_a = bank.action[bank.window_starts[window_ids]]
                if continuous:
                    m = (policy.clamp(actions[0]) - expert_a).abs().mean()
                    m_label = "step0 mae"
                else:
                    m = (actions[0] == expert_a).float().mean()
                    m_label = "step0 acc"
            print(f"{name} step {step:5d} | reward {rewards.mean():.3f} "
                  f"| return {returns.mean():.3f} | entropy {entropy:.3f} "
                  f"| {m_label} {m:.3f}")

    ckpt = Path(cfg.dirs()["ckpt"]) / f"{name}.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"saved {ckpt}")
    return policy
