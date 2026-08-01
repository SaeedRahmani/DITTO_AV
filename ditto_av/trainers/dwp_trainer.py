"""DITTO-WP (gen-4): on-policy imagination refinement of the waypoint
head — the DITTO objective made compatible with the abstraction that
actually drives well.

Three evidence-driven components on top of train_latent_policy:
1. DEPLOYMENT-CONSISTENT IMAGINATION: the dream steps the world model
   through the batched tracker port (plan -> control), with ego speed
   decoded from the latent — the rollout dynamics ARE the deployment
   stack (policy + tracker + WM-as-vehicle). gen-3 showed both naive
   alternatives fail: waypoints as the WM action drift (dev-10 16.4 vs
   19.2), and a control-space DITTO head cannot emit plans at all.
2. TASK-PROJECTED MATCHING: rewards are computed after projecting h
   through a frozen ridge probe h -> expert-waypoint labels. Raw latent
   similarity to any plausible traffic state is 0.85-0.92 (exogenous
   traffic dominates; measured); the Phase-2 selector study showed the
   raw metric is gameable. The probe subspace is, by construction, the
   ego-intent content of h. Retrieval keys stay full-h (scene context
   is the right retrieval key). Config: ac.reward_proj = "wp"|"none".
3. RETRIEVAL-RELABELED DIVERGENT STARTS (offline DAgger in
   imagination): a fraction of rollouts first walk ac.divergent_steps
   sampled-policy steps from expert starts, then the nearest expert
   modes are re-retrieved FROM THE REACHED LATENT and become the
   targets. Directly targets the measured failure state (41% of
   deployment ticks off the expert manifold). Config:
   ac.divergent_frac / ac.divergent_steps.

BC anchor (bc_kl trust region, dose 0.1 benchmark-proven) and the
actor-critic machinery are unchanged from ac_trainer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..bench2drive import VEL_SCALE
from ..config import Config
from ..data import LatentBank
from ..models.nets import make_actor_critic
from ..models.world_model import VectorWorldModel
from ..rewards import LatentMatcher, lambda_return
from ..tracker_torch import TorchWaypointTracker, wp_to_vehicle_t
from .. import wandb_util


def fit_wp_probe(bank: LatentBank, ridge: float = 1.0) -> torch.Tensor:
    """Closed-form ridge probe W: h -> wp labels; returns (D, 12)."""
    assert bank.wp is not None, "wp probe needs wp labels in the bank"
    H = bank.h
    Y = bank.wp.to(H.dtype)
    D = H.shape[1]
    G = H.T @ H + ridge * torch.eye(D, device=H.device, dtype=H.dtype)
    W = torch.linalg.solve(G, H.T @ Y)                    # (D, 12)
    with torch.no_grad():
        r2 = 1.0 - ((H @ W - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f"wp probe: R^2 {float(r2):.3f} (ridge {ridge})")
    return W


def train_latent_policy_wp(cfg: Config, wm: VectorWorldModel,
                           bank: LatentBank, seed: int = 0,
                           name: str = "ditto_wp"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    acfg = cfg.ac
    assert cfg.env.wp_head, "train_latent_policy_wp requires wp_head mode"

    policy = make_actor_critic(True, cfg.wm.feature_dim,
                               cfg.env.policy_action_dim, acfg.hidden_dim,
                               acfg.layers, action_space="waypoints"
                               ).to(device)

    bc_policy = None
    bc_ckpt = Path(cfg.dirs()["ckpt"]) / "bc.pt"
    if acfg.bc_init and bc_ckpt.exists():
        bc_policy = make_actor_critic(True, cfg.wm.feature_dim,
                                      cfg.env.policy_action_dim,
                                      cfg.bc.hidden_dim, cfg.bc.layers,
                                      action_space="waypoints").to(device)
        bc_policy.load_state_dict(torch.load(bc_ckpt, map_location=device))
        policy.actor.load_state_dict(bc_policy.actor.state_dict())
        bc_policy.requires_grad_(False)
        bc_policy.eval()

    actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=acfg.actor_lr)
    critic_opt = torch.optim.Adam(policy.critic.parameters(),
                                  lr=acfg.critic_lr)
    proj = fit_wp_probe(bank, acfg.proj_ridge) \
        if acfg.reward_proj == "wp" else None
    matcher = LatentMatcher(bank, mode=acfg.reward_mode, k=acfg.k_modes,
                            n_negatives=acfg.n_negatives, proj=proj)
    tracker = TorchWaypointTracker()
    H = acfg.horizon
    deter = cfg.wm.deter_dim

    def decode_speed(feat):
        obs = wm.decoder(feat)
        return (obs[:, 3] * VEL_SCALE).clamp(0.0, 40.0)

    def dream_control(h, z):
        """One deployment-consistent imagination step; returns sampled
        plan and the next state."""
        feat = torch.cat((h, z), dim=-1)
        a = policy.dist(feat).sample()
        wp_v = wp_to_vehicle_t(policy.clamp(a))
        control = tracker.act(wp_v, decode_speed(feat))
        h, z = wm.dream(control, (h, z))
        return feat, a, h, z

    n_div = int(round(acfg.batch_size * acfg.divergent_frac)) \
        if acfg.reward_mode != "single" else 0

    for step in range(1, acfg.train_steps + 1):
        window_ids = bank.sample_windows(acfg.batch_size, rng)
        h, z = bank.start_states(window_ids)

        with torch.no_grad():
            # divergent tail of the batch: walk off-manifold first,
            # then relabel by retrieval from the reached latent
            if n_div > 0:
                hd, zd = h[-n_div:], z[-n_div:]
                for _ in range(acfg.divergent_steps):
                    _, _, hd, zd = dream_control(hd, zd)
                h = torch.cat([h[:-n_div], hd])
                z = torch.cat([z[:-n_div], zd])
                targets = torch.cat([
                    matcher.targets(window_ids[:-n_div]),
                    matcher.retrieve_from_h(hd)])
            else:
                targets = matcher.targets(window_ids)

            feats, actions = [], []
            for _ in range(H):
                feat, a, h, z = dream_control(h, z)
                feats.append(feat)
                actions.append(a)
            feats.append(torch.cat((h, z), dim=-1))
        feats = torch.stack(feats)          # (H+1, B, F)
        actions = torch.stack(actions)      # (H, B, 12)
        dreamed_h = matcher.project(feats[..., :deter])

        rewards = matcher.rewards(dreamed_h, targets)      # (H, B)
        with torch.no_grad():
            values_t = policy.target_value(feats)
        returns = lambda_return(rewards, values_t, acfg.gamma, acfg.lam)

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
            wandb_util.log({"step": step, "reward": float(rewards.mean()),
                            "return": float(returns.mean()),
                            "entropy": float(entropy),
                            "actor_loss": float(actor_loss),
                            "critic_loss": float(critic_loss)},
                           prefix=name)
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                # diagnostic on the non-divergent head only (the
                # divergent tail's step-0 state is off-manifold by
                # design and has no aligned expert label)
                nb = acfg.batch_size - n_div
                expert_wp = bank.wp[bank.window_starts[window_ids[:nb]]]
                m = (policy.clamp(actions[0][:nb])
                     - expert_wp).abs().mean()
            print(f"{name} step {step:5d} | reward {rewards.mean():.4f} "
                  f"| return {returns.mean():.3f} | entropy {entropy:.2f} "
                  f"| step0 wp-mae {m:.4f}")

    ckpt = Path(cfg.dirs()["ckpt"]) / f"{name}.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"saved {ckpt}")
    return policy
