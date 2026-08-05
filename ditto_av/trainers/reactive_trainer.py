"""D3 (V03_PLAN §5): fine-tune the v0.2 champion in the REACTIVE world.

The recipe is v0.2's proven A2C skeleton with three changes:
- a fraction of rollouts (annealed 0 -> p_max) run in ReactiveEgoSim:
  traffic evolves via the W0-passing ensemble, conditioned on the SIM
  ego — closing v0.2's ghost-traffic gap;
- W1 pessimism: once a rollout's ensemble disagreement crosses the
  calibrated threshold, its remaining rewards are zeroed — the policy
  cannot farm reward where the world model does not know;
- KL anchor to the (frozen) replay-trained champion, so the reactive
  refinement stays in the trust region of what already works.

Reward is UNCHANGED: ego-state matching vs the same-scene logged
expert. The world reacts; the target does not move.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..egosim import DT, EgoSim, GlobalLog, _wrap
from ..models.policy_v2 import TokenPolicy, make_token_policy
from ..reactive import ReactiveEgoSim
from ..rewards import lambda_return
from ..trainers.clp_trainer import sim_from_config
from .. import wandb_util


@torch.no_grad()
def _rollout(policy, sim, log, starts, horizon, burn_in,
             reactive: bool, w1_thresh: float, perturb=None, rng=None,
             stochastic=True, w_churn: float = 0.0):
    device = starts.device
    lo = log.ep_start[starts]
    bidx = (starts.unsqueeze(1)
            + torch.arange(-burn_in, 0, device=device)).clamp_min(
                lo.unsqueeze(1))
    burn_obs = log.obs[bidx].transpose(0, 1).contiguous()
    _, h = policy.unroll(burn_obs)
    if reactive:
        xy, th, v = sim.reset_reactive(starts, perturb, rng)
    else:
        xy, th, v = sim.reset(starts, perturb, rng)
    frame = starts.clone()
    dead = torch.zeros(len(starts), dtype=torch.bool, device=device)
    obs_l, act_l, rew_l, col_l, err_l, dis_l = [], [], [], [], [], []
    prev_plan = prev_pxy = prev_pth = None
    pxy_l, pth_l, gate_l = [], [], []
    for _ in range(horizon):
        obs = sim.build_obs(frame, xy, th, v)
        # emission pose of this tick's plan (pre-step) — constants for
        # the mean-plan consistency loss's motion compensation
        pxy_l.append(xy)
        pth_l.append(th)
        # nearest-actor distance for the proximity-gated consistency
        # loss — live reactive actors when the sim has them, else log
        act = getattr(sim, "_buf", None)
        if act is None:
            act = log.act[frame]
        adist = (act[..., 1:3] - xy.unsqueeze(1)).norm(dim=-1)
        adist = torch.where(act[..., 0] > 0.5, adist,
                            torch.full_like(adist, 1e3))
        gate_l.append(adist.min(dim=1).values)
        emb = policy.encode(obs)
        h = policy.step(emb, h)
        d = policy.dist(policy.features(emb, h))
        a = d.sample() if stochastic else d.base_dist.loc
        churn = None
        if w_churn > 0.0:
            from ..bench2drive import WP_SCALE
            from ..consistency import plan_churn_lat
            plan_m = policy.clamp(a).view(len(starts), -1, 2) * WP_SCALE
            if prev_plan is not None:
                churn = plan_churn_lat(prev_plan, plan_m, prev_pxy, xy,
                                       prev_pth, th)
            prev_plan, prev_pxy, prev_pth = plan_m, xy, th
        th_prev = th
        xy, th, v = sim.step_ego(policy.clamp(a), xy, th, v)
        if reactive:
            sim.step_traffic(frame, xy, th, v)
            if sim.last_disagreement is not None:
                dead = dead | (sim.last_disagreement > w1_thresh)
                dis_l.append(sim.last_disagreement)
        frame = frame + 1
        r = sim.reward(frame, xy, th, v,
                       yaw_rate=_wrap(th - th_prev) / DT)
        if churn is not None:
            r = r - w_churn * churn
        r = torch.where(dead, torch.zeros_like(r), r)   # W1 pessimism
        obs_l.append(obs)
        act_l.append(a)
        rew_l.append(r)
        col_l.append(sim.collisions(frame, xy, th))
        err_l.append((log.ego[frame][:, 0:2] - xy).norm(dim=-1))
    final_obs = sim.build_obs(frame, xy, th, v)
    return (burn_obs, torch.stack(obs_l), torch.stack(act_l),
            torch.stack(rew_l), final_obs, torch.stack(col_l),
            torch.stack(err_l), dead,
            torch.stack(dis_l) if dis_l else None,
            torch.stack(pxy_l), torch.stack(pth_l),
            torch.stack(gate_l))


def train_clp_reactive(cfg: Config, log: GlobalLog,
                       rsim: ReactiveEgoSim, champion_ckpt: Path,
                       out_dir: Path, seed: int = 0,
                       p_reactive_max: float = 0.5,
                       w1_thresh: float = 0.25) -> TokenPolicy:
    torch.manual_seed(seed)
    c = cfg.clp
    device = cfg.device
    policy = make_token_policy(cfg).to(device)
    anchor = make_token_policy(cfg).to(device)
    state = torch.load(champion_ckpt, map_location=device)
    policy.load_state_dict(state)
    anchor.load_state_dict(state)
    anchor.requires_grad_(False)
    anchor.eval()

    head = ("critic.", "target_critic.")
    actor_params = [p for n, p in policy.named_parameters()
                    if not n.startswith(head)]
    a_opt = torch.optim.Adam(actor_params, lr=c.actor_lr)
    c_opt = torch.optim.Adam(policy.critic.parameters(), lr=c.critic_lr)
    replay_sim = sim_from_config(cfg, log)

    # reactive rollouts must start at frames with scene windows
    pool_all = log.window_starts(c.horizon, max(c.reward_tau, 1))
    w_ok = torch.tensor(
        [int(f) in rsim.frame_to_w for f in pool_all.cpu()],
        dtype=torch.bool, device=pool_all.device)
    pool_r = pool_all[w_ok]
    print(f"pools: replay {len(pool_all)}, reactive {len(pool_r)}")
    perturb = {"frac": c.divergent_frac, "lat_sigma": c.lat_sigma,
               "yaw_sigma": c.yaw_sigma, "v_sigma": c.v_sigma}
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    for step in range(1, c.rl_steps + 1):
        p_r = p_reactive_max * min(1.0, step / (0.3 * c.rl_steps))
        reactive = (step % 100) < int(100 * p_r)
        pool = pool_r if reactive else pool_all
        sim = rsim if reactive else replay_sim
        i = torch.randint(len(pool), (c.rl_batch,), device=device,
                          generator=gen)
        starts = pool[i]
        (burn_obs, obs_seq, act_seq, rewards, final_obs, cols, errs,
         dead, dis, pxy, pth, dmin) = _rollout(
             policy, sim, log, starts, c.horizon,
             c.burn_in, reactive, w1_thresh, perturb, gen,
             w_churn=getattr(c, "w_churn", 0.0))

        with torch.no_grad():
            _, h0 = policy.unroll(burn_obs)
        feats, h_end = policy.unroll(obs_seq, h0)
        with torch.no_grad():
            emb_f = policy.encode(final_obs)
            feat_f = policy.features(emb_f, policy.step(emb_f, h_end))
            values_t = policy.target_value(
                torch.cat([feats, feat_f.unsqueeze(0)]))
        returns = lambda_return(rewards, values_t, c.gamma, c.lam)
        dist = policy.dist(feats)
        logp = dist.log_prob(act_seq)
        adv = returns - values_t[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        entropy = dist.entropy().mean()
        actor_loss = -(logp * adv.detach()).mean() \
            - c.entropy_coef * entropy
        with torch.no_grad():
            _, ah0 = anchor.unroll(burn_obs)
            afeats, _ = anchor.unroll(obs_seq, ah0)
            a_dist = anchor.dist(afeats)
        kl = torch.distributions.kl.kl_divergence(dist, a_dist).mean()
        actor_loss = actor_loss + c.bc_kl_coef * kl
        w_cons = getattr(c, "w_consistency", 0.0)
        if w_cons > 0.0:
            # v0.3.2 axis-3: differentiable consistency of the MEAN
            # plans along the visited states — sampled-churn pricing
            # (axis 2) drowned in exploration noise (~2.4 m sampled vs
            # 0.2 m mean churn); this term sees the mean directly.
            from ..bench2drive import WP_SCALE
            from ..consistency import plan_churn_lat
            T, B = pxy.shape[0], pxy.shape[1]
            mu = dist.base_dist.loc.view(T, B, -1, 2) * WP_SCALE
            ch = plan_churn_lat(
                mu[:-1].reshape((T - 1) * B, -1, 2),
                mu[1:].reshape((T - 1) * B, -1, 2),
                pxy[:-1].reshape(-1, 2), pxy[1:].reshape(-1, 2),
                pth[:-1].reshape(-1), pth[1:].reshape(-1))
            ch = ch.view(T - 1, B)
            d0 = getattr(c, "cons_gate_d0", 0.0)
            if d0 > 0.0:
                from ..consistency import proximity_gate
                g = proximity_gate(dmin[1:], d0,
                                   getattr(c, "cons_gate_w", 3.0))
                ch = ch * g
            cons = ch.mean()
            actor_loss = actor_loss + w_cons * cons

        values = policy.value(feats.detach())
        critic_loss = 0.5 * F.mse_loss(values, returns.detach())

        a_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, c.grad_clip)
        a_opt.step()
        c_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),
                                       c.grad_clip)
        c_opt.step()
        policy.update_target(c.target_tau)

        if step % 100 == 0 or step == 1:
            wandb_util.log(
                {"step": step, "reward": float(rewards.mean()),
                 "reactive": int(reactive), "p_r": p_r,
                 "kl": float(kl), "entropy": float(entropy),
                 "dead_frac": float(dead.float().mean()),
                 "col": float(cols.any(0).float().mean()),
                 "perr": float(errs[-1].mean())}, prefix="clp_rx")
        if step % 500 == 0 or step == 1:
            tag = "RX" if reactive else "RP"
            print(f"clp_rx {step:5d} [{tag} p={p_r:.2f}] "
                  f"r {rewards.mean():.3f} | perr {errs[-1].mean():.2f}"
                  f" | col {cols.any(0).float().mean():.3f} "
                  f"| dead {dead.float().mean():.2f} | kl {kl:.3f}")

    ckpt = out_dir / "clp_rx.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"saved {ckpt}")
    return policy


@torch.no_grad()
def eval_both_worlds(cfg, policy, log, rsim, w1_thresh=0.25,
                     batch=192, seed=0):
    """G2-style verdict in BOTH worlds (their divergence is itself a
    diagnostic: big gaps = the reactive world changes the task)."""
    c = cfg.clp
    device = cfg.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    replay_sim = sim_from_config(cfg, log)
    pool_all = log.window_starts(c.horizon, max(c.reward_tau, 1))
    w_ok = torch.tensor(
        [int(f) in rsim.frame_to_w for f in pool_all.cpu()],
        dtype=torch.bool, device=pool_all.device)
    pool_r = pool_all[w_ok]
    out = {}
    for name, sim, reactive, pool in (
            ("replay", replay_sim, False, pool_all),
            ("reactive", rsim, True, pool_r)):
        starts = pool[torch.randint(len(pool), (batch,), device=device,
                                    generator=gen)]
        (_, _, _, rewards, _, cols, errs, dead, _, _, _,
         _) = _rollout(
            policy, sim, log, starts, c.horizon, c.burn_in, reactive,
            w1_thresh, stochastic=False)
        out[name] = {"reward": float(rewards.mean()),
                     "collision": float(cols.any(0).float().mean()),
                     "perr_final": float(errs[-1].mean()),
                     "dead_frac": float(dead.float().mean())}
    return out
