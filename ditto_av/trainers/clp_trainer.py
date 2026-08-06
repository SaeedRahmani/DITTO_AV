"""v0.2 trainers: BC pretrain + closed-loop RL in the log-replay sim.

train_clp_bc  — sequence behavior cloning of the TokenPolicy on logged
    obs -> wp labels (teacher forced). Doubles as the v0.2 BC baseline.
train_clp_rl  — the DITTO loop, driving-native: on-policy rollouts in
    EgoSim (replayed scenes, kinematic ego), rewarded by ego-state
    closeness to the same-scene expert; A2C with lambda-returns, EMA
    target critic, entropy bonus, and a KL trust region to the frozen
    BC anchor (v0.1 settled lesson: the anchor is load-bearing).
evaluate_in_wm — shared closed-loop metrics (G2 gate + selector).

Rollouts are collected no-grad, then the GRU unroll is recomputed with
grad on the stored obs/action sequences (same recompute pattern as
v0.1 ac_trainer, extended through the recurrence = BPTT on-policy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..egosim import (DT, EgoSim, GlobalLog, RewardParams, SimParams,
                      _wrap)
from ..models.policy_v2 import TokenPolicy, make_token_policy
from ..rewards import lambda_return
from .. import wandb_util


def sim_from_config(cfg: Config, log: GlobalLog) -> EgoSim:
    from ..bench2drive import extra_obs_layout
    with_route, with_lights = extra_obs_layout(cfg.env.extra_obs_dims,
                                               cfg.env.light_obs)
    c = cfg.clp
    return EgoSim(
        log,
        SimParams(with_route=with_route, with_lights=with_lights,
                  v_max=c.sim_v_max, a_max=c.sim_a_max,
                  dtheta_max=c.sim_dtheta_max),
        RewardParams(tau=c.reward_tau, sigma_p=c.sigma_p,
                     sigma_yaw=c.sigma_yaw, sigma_v=c.sigma_v,
                     sigma_p2=c.sigma_p2, p2_weight=c.p2_weight,
                     sigma_yawrate=c.sigma_yawrate,
                     collision_penalty=c.collision_penalty,
                     penalty_ignore_rear=c.penalty_ignore_rear))


def _seq_batch(log: GlobalLog, starts: torch.Tensor, T: int):
    """Gather (T, B) logged obs + wp label windows starting at `starts`."""
    idx = starts.unsqueeze(1) + torch.arange(T, device=starts.device)
    return (log.obs[idx].transpose(0, 1).contiguous(),
            log.wp[idx].transpose(0, 1).contiguous())


def train_clp_bc(cfg: Config, log: GlobalLog, seed: int = 0
                 ) -> TokenPolicy:
    torch.manual_seed(seed)
    c = cfg.clp
    device = cfg.device
    policy = make_token_policy(cfg).to(device)
    n_par = sum(p.numel() for p in policy.parameters())
    print(f"clp_bc: TokenPolicy {n_par / 1e6:.2f}M params")
    opt = torch.optim.Adam(policy.parameters(), lr=c.bc_lr)
    pool = log.window_starts(c.bc_seq, 0)
    n_val = max(1, len(pool) // 10)
    perm = torch.randperm(len(pool), device=pool.device,
                          generator=torch.Generator(pool.device)
                          .manual_seed(seed))
    train_pool, val_pool = pool[perm[:-n_val]], pool[perm[-n_val:]]

    for step in range(1, c.bc_steps + 1):
        i = train_pool[torch.randint(len(train_pool), (c.bc_batch,),
                                     device=device)]
        obs_seq, wp_seq = _seq_batch(log, i, c.bc_seq)
        feats, _ = policy.unroll(obs_seq)
        loss = -policy.dist(feats).log_prob(wp_seq).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), c.grad_clip)
        opt.step()
        if step % 100 == 0 or step == 1:
            wandb_util.log({"step": step, "loss": float(loss)},
                           prefix="clp_bc")
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                j = val_pool[torch.randint(len(val_pool),
                                           (min(256, len(val_pool)),),
                                           device=device)]
                vo, vw = _seq_batch(log, j, c.bc_seq)
                vf, _ = policy.unroll(vo)
                mae = (policy.act(vf) - vw).abs().mean()
            print(f"clp_bc step {step:5d} | nll {loss:.3f} "
                  f"| val mae {mae:.4f}")
            wandb_util.log({"step": step, "val_mae": float(mae)},
                           prefix="clp_bc")
    ckpt = Path(cfg.dirs()["ckpt"]) / "clp_bc.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"saved {ckpt}")
    return policy


def _rollout(policy: TokenPolicy, sim: EgoSim, log: GlobalLog,
             starts: torch.Tensor, horizon: int, burn_in: int,
             perturb: Optional[dict] = None, stochastic: bool = True,
             rng: Optional[torch.Generator] = None):
    """No-grad on-policy rollout. Returns tensors for the grad pass.

    burn_obs (L, B, O) logged prefix; obs_seq (H, B, O) sim obs;
    act_seq (H, B, A) unclamped samples; rewards (H, B);
    final_obs (B, O); cols (H, B) collision flags; plus trajectory
    diagnostics (pos error vs expert, progress).
    """
    device = starts.device
    L = burn_in
    lo = log.ep_start[starts]
    bidx = (starts.unsqueeze(1)
            + torch.arange(-L, 0, device=device)).clamp_min(
                lo.unsqueeze(1))
    burn_obs = log.obs[bidx].transpose(0, 1).contiguous()
    with torch.no_grad():
        _, h = policy.unroll(burn_obs)
        xy, th, v = sim.reset(starts, perturb, rng)
        frame = starts.clone()
        obs_l, act_l, rew_l, col_l, err_l = [], [], [], [], []
        for _ in range(horizon):
            obs = sim.build_obs(frame, xy, th, v)
            emb = policy.encode(obs)
            h = policy.step(emb, h)
            feat = policy.features(emb, h)
            d = policy.dist(feat)
            a = d.sample() if stochastic else d.base_dist.loc
            th_prev = th
            xy, th, v = sim.step_ego(policy.clamp(a), xy, th, v)
            frame = frame + 1
            r = sim.reward(frame, xy, th, v,
                           yaw_rate=_wrap(th - th_prev) / DT)
            col = sim.collisions(frame, xy, th)
            if sim.r.collision_penalty > 0:
                col_pen = sim.collisions(
                    frame, xy, th,
                    ignore_rear=sim.r.penalty_ignore_rear) \
                    if sim.r.penalty_ignore_rear else col
                r = r - sim.r.collision_penalty * col_pen.float()
            err = (log.ego[frame][:, 0:2] - xy).norm(dim=-1)
            obs_l.append(obs)
            act_l.append(a)
            rew_l.append(r)
            col_l.append(col)
            err_l.append(err)
        final_obs = sim.build_obs(frame, xy, th, v)
    return (burn_obs, torch.stack(obs_l), torch.stack(act_l),
            torch.stack(rew_l), final_obs, torch.stack(col_l),
            torch.stack(err_l))


def train_clp_rl(cfg: Config, log: GlobalLog, seed: int = 0
                 ) -> TokenPolicy:
    torch.manual_seed(seed)
    c = cfg.clp
    device = cfg.device
    ckpt_dir = Path(cfg.dirs()["ckpt"])
    policy = make_token_policy(cfg).to(device)
    anchor = make_token_policy(cfg).to(device)
    bc_ckpt = ckpt_dir / "clp_bc.pt"
    assert bc_ckpt.exists(), "clp_rl needs clp_bc.pt (run clp_bc first)"
    state = torch.load(bc_ckpt, map_location=device)
    policy.load_state_dict(state)
    anchor.load_state_dict(state)
    anchor.requires_grad_(False)
    anchor.eval()

    head_names = ("critic.", "target_critic.")
    actor_params = [p for n, p in policy.named_parameters()
                    if not n.startswith(head_names)]
    actor_opt = torch.optim.Adam(actor_params, lr=c.actor_lr)
    critic_opt = torch.optim.Adam(policy.critic.parameters(),
                                  lr=c.critic_lr)
    sim = sim_from_config(cfg, log)
    pool = log.window_starts(c.horizon, max(c.reward_tau, 1))
    perturb = {"frac": c.divergent_frac, "lat_sigma": c.lat_sigma,
               "yaw_sigma": c.yaw_sigma, "v_sigma": c.v_sigma}
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    for step in range(1, c.rl_steps + 1):
        starts = pool[torch.randint(len(pool), (c.rl_batch,),
                                    device=device, generator=gen)]
        (burn_obs, obs_seq, act_seq, rewards, final_obs, cols,
         errs) = _rollout(policy, sim, log, starts, c.horizon,
                          c.burn_in, perturb, rng=gen)

        # grad pass: recompute the unroll on stored obs/actions
        with torch.no_grad():
            _, h0 = policy.unroll(burn_obs)
        feats, h_end = policy.unroll(obs_seq, h0)
        with torch.no_grad():
            emb_f = policy.encode(final_obs)
            h_f = policy.step(emb_f, h_end)
            feat_f = policy.features(emb_f, h_f)
            values_t = policy.target_value(
                torch.cat([feats, feat_f.unsqueeze(0)]))     # (H+1, B)
        returns = lambda_return(rewards, values_t, c.gamma, c.lam)

        dist = policy.dist(feats)
        logp = dist.log_prob(act_seq)
        advantage = returns - values_t[:-1]
        advantage = (advantage - advantage.mean()) \
            / (advantage.std() + 1e-8)
        policy_loss = -(logp * advantage.detach()).mean()
        entropy = dist.entropy().mean()
        actor_loss = policy_loss - c.entropy_coef * entropy
        if c.bc_kl_coef > 0:
            with torch.no_grad():
                _, ah0 = anchor.unroll(burn_obs)
                afeats, _ = anchor.unroll(obs_seq, ah0)
                a_dist = anchor.dist(afeats)
            kl_bc = torch.distributions.kl.kl_divergence(
                dist, a_dist).mean()
            actor_loss = actor_loss + c.bc_kl_coef * kl_bc
        else:
            kl_bc = torch.zeros(())

        values = policy.value(feats.detach())
        critic_loss = 0.5 * F.mse_loss(values, returns.detach())

        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, c.grad_clip)
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),
                                       c.grad_clip)
        critic_opt.step()
        policy.update_target(c.target_tau)

        if step % 50 == 0 or step == 1:
            wandb_util.log(
                {"step": step, "reward": float(rewards.mean()),
                 "return": float(returns.mean()),
                 "entropy": float(entropy), "kl_bc": float(kl_bc),
                 "collision_rate": float(cols.any(0).float().mean()),
                 "pos_err_final": float(errs[-1].mean()),
                 "actor_loss": float(actor_loss),
                 "critic_loss": float(critic_loss)}, prefix="clp_rl")
        if step % 500 == 0 or step == 1:
            print(f"clp_rl step {step:5d} | r {rewards.mean():.3f} "
                  f"| pos_err@H {errs[-1].mean():.2f} m "
                  f"| col {cols.any(0).float().mean():.3f} "
                  f"| ent {entropy:.2f} | kl {kl_bc:.3f}")

    ckpt = ckpt_dir / "clp_rl.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"saved {ckpt}")
    return policy


@torch.no_grad()
def evaluate_in_wm(policy: TokenPolicy, sim: EgoSim, log: GlobalLog,
                    horizon: int, burn_in: int, batch: int = 256,
                    n_batches: int = 4, perturb: Optional[dict] = None,
                    seed: int = 0) -> dict:
    """Deterministic closed-loop metrics on sampled windows (G2 gate)."""
    device = log.obs.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pool = log.window_starts(horizon, max(sim.r.tau, 1))
    agg = {"reward": [], "pos_err_final": [], "pos_err_mean": [],
           "collision": [], "progress": []}
    for _ in range(n_batches):
        starts = pool[torch.randint(len(pool), (batch,), device=device,
                                    generator=gen)]
        (_, _, _, rewards, _, cols, errs) = _rollout(
            policy, sim, log, starts, horizon, burn_in, perturb,
            stochastic=False, rng=gen)
        ex0 = log.ego[starts][:, 0:2]
        exH = log.ego[starts + horizon][:, 0:2]
        expert_dist = (exH - ex0).norm(dim=-1).clamp_min(1e-3)
        # progress proxy: expert displacement minus final error, ratio
        prog = ((expert_dist - errs[-1]) / expert_dist).clamp(0, 1)
        agg["reward"].append(rewards.mean().item())
        agg["pos_err_final"].append(errs[-1].mean().item())
        agg["pos_err_mean"].append(errs.mean().item())
        agg["collision"].append(cols.any(0).float().mean().item())
        agg["progress"].append(prog.mean().item())
    return {k: float(np.mean(v)) for k, v in agg.items()}
