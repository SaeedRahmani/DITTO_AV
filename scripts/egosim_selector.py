#!/usr/bin/env python3
"""G1 gate: does the egosim reward rank real closed-loop drivers?

Drives the banked v0.1 wp-output models (known test-10 truth, range
3.46-30.49) through EgoSim with their full deployment semantics — RSSM
posterior filter on the rebuilt obs, waypoint head, executed-control
feedback through the torch tracker port — and rank-correlates the
egosim score against the banked test-10 numbers.

v0.1's on-policy latent metric scored Spearman -0.60 on this exact
question (runs/phase2_selector). The v0.2 reward must score clearly
POSITIVE or V02_PLAN says STOP — do not scale training on a reward
that cannot rank known drivers.

Usage (needs the ##glob1 297-clip npz built by run_b2d --stage data
with configs/b2d_v02.yaml):
    python scripts/egosim_selector.py --data runs/b2d_v02/data \
        --out runs/egosim_selector
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from ditto_av.config import load_config  # noqa: E402
from ditto_av.egosim import EgoSim, GlobalLog  # noqa: E402
from ditto_av.models.nets import make_actor_critic  # noqa: E402
from ditto_av.tracker_torch import (TorchWaypointTracker,  # noqa: E402
                                    wp_to_vehicle_t)
from ditto_av.trainers.clp_trainer import sim_from_config  # noqa: E402
from ditto_av.trainers.wm_trainer import load_world_model  # noqa: E402

HOME_OUT = Path.home() / "ditto_out"

# label, train config, ckpt dir, policy file, test-10 (score, completion)
# test-10 numbers are the banked champion-config (rec) runs; see
# saeed/ver0.1 NEXT_STEPS + runs/carla_smoke/{gen3_wph_era,gen4_dwp}
REGISTRY = [
    ("wph_bc_s0",   "configs/b2d_gen3_wph.yaml",
     HOME_OUT / "b2d_gen3_wph", "bc", 30.49, 83.2),
    ("wph_bc_s1",   "configs/b2d_gen3_wph_s1.yaml",
     HOME_OUT / "b2d_gen3_wph_s1", "bc", 25.86, 73.0),
    ("wph_bc_s2",   "configs/b2d_gen3_wph_s2.yaml",
     HOME_OUT / "b2d_gen3_wph_s2", "bc", 28.45, 70.6),
    ("wph_cap",     "configs/b2d_gen3_wph_cap.yaml",
     HOME_OUT / "b2d_gen3_wph_cap", "bc", 20.40, 68.7),
    ("wph_lw",      "configs/b2d_gen3_wph_lw.yaml",
     HOME_OUT / "b2d_gen3_wph_lw", "bc", 19.72, 78.4),
    ("wph_lw2",     "configs/b2d_gen3_wph_lw2.yaml",
     HOME_OUT / "b2d_gen3_wph_lw2", "bc", 15.64, 75.2),
    ("dwp_v1",      "configs/b2d_gen4_dwp.yaml",
     HOME_OUT / "b2d_gen4_dwp", "ditto_wp", 3.46, 50.4),
    ("dwp_k03",     "configs/b2d_gen4_dwp_k03.yaml",
     HOME_OUT / "b2d_gen4_dwp_k03", "ditto_wp", 19.49, 70.8),
    ("dwp_k03nd",   "configs/b2d_gen4_dwp_k03nd.yaml",
     HOME_OUT / "b2d_gen4_dwp_k03nd", "ditto_wp", 24.07, 80.8),
    ("dwp_es3k",    "configs/b2d_gen4_dwp_k03nd3k.yaml",
     HOME_OUT / "b2d_gen4_dwp_k03nd3k", "ditto_wp", 13.31, 71.1),
    ("dwp_k10nd",   "configs/b2d_gen4_dwp_k10nd.yaml",
     HOME_OUT / "b2d_gen4_dwp_k10nd", "ditto_wp", 18.08, 60.5),
    ("wp_action_bc", "configs/b2d_gen3_wp.yaml",
     HOME_OUT / "b2d_gen3_wp", "bc", 16.37, 69.0),
]

BURN_IN = 16


def load_model(cfg_path: str, ckpt_dir: Path, policy_name: str,
               device: str):
    cfg = load_config(cfg_path)
    cfg.run_dir = str(ckpt_dir)
    cfg.device = device
    wm = load_world_model(cfg, cfg.env.obs_dim)
    wm.eval()
    hid, lay = ((cfg.bc.hidden_dim, cfg.bc.layers) if policy_name == "bc"
                else (cfg.ac.hidden_dim, cfg.ac.layers))
    policy = make_actor_critic(
        cfg.env.continuous, cfg.wm.feature_dim, cfg.env.policy_action_dim,
        hid, lay, action_space=("waypoints" if cfg.env.wp_out
                                else cfg.env.action_space)).to(device)
    policy.load_state_dict(torch.load(ckpt_dir / "checkpoints"
                                      / f"{policy_name}.pt",
                                      map_location=device))
    policy.eval()
    return cfg, wm, policy


def launch_starts(log: GlobalLog, pool: torch.Tensor,
                  n: int) -> torch.Tensor:
    """Start frames where the expert is (nearly) stopped but moves off
    within 2 s — the launch states where test-10 actually differentiated
    the banked models (41% of champion ticks were plan-GO-static)."""
    speed = log.ego[:, 3]
    ahead = torch.stack([speed[(pool + k).clamp(
        max=len(speed) - 1)] for k in range(1, 21)])
    mask = (speed[pool] < 1.5) & (ahead.max(dim=0).values > 2.0)
    sel = pool[mask]
    if len(sel) == 0:
        return pool[:n]
    step = max(1, len(sel) // n)
    return sel[::step][:n]


@torch.no_grad()
def rollout_model(label, cfg, wm, policy, sim: EgoSim, log: GlobalLog,
                  starts: torch.Tensor, horizon: int, device: str,
                  perturb=None, rng=None):
    """Deployment-semantics rollout: WM filter + wp head + tracker port.

    wp_head models: the WM action channel is the EXECUTED control
    (tracker output) — training-consistent feedback, as deployed.
    waypoints-action models (gen3_wp): the raw plan feeds back.
    """
    B = len(starts)
    wp_head = cfg.env.wp_head
    action_dim = cfg.env.action_dim
    tracker = TorchWaypointTracker()
    state = wm.init_state(B)

    # burn-in on logged frames with logged executed actions
    lo = log.ep_start[starts]
    bidx = (starts.unsqueeze(1)
            + torch.arange(-BURN_IN, 0, device=device)).clamp_min(
                lo.unsqueeze(1))
    burn_obs = log.obs[bidx].transpose(0, 1)               # (L, B, O)
    # executed WM actions along the log: control actions for wp_head
    # (the expert's executed controls), raw wp labels for wp-action WMs
    src = log._action_t if wp_head else log.wp
    acts = torch.zeros(BURN_IN, B, action_dim, device=device)
    acts[1:] = src[bidx.T[:-1]]        # prev action = action at f_{t-1}
    reset = torch.zeros(BURN_IN, B, dtype=torch.bool, device=device)
    reset[0] = True
    _, _, state = wm.observe(burn_obs, acts, reset, state)

    xy, th, v = sim.reset(starts, perturb, rng)
    frame = starts.clone()
    # first sim step's prev action = executed at the last burn-in frame
    prev_action = src[bidx[:, -1]]
    rews, cols, errs = [], [], []
    for t in range(horizon):
        obs = sim.build_obs(frame, xy, th, v).unsqueeze(0)  # (1, B, O)
        act_in = prev_action.unsqueeze(0)
        rst = torch.zeros(1, B, dtype=torch.bool, device=device)
        feat, _, state = wm.observe(obs, act_in, rst, state)
        plan = policy.act(feat[0], stochastic=False)        # (B, 12)
        if wp_head:
            control = tracker.act(wp_to_vehicle_t(plan), v)
            prev_action = control
        else:
            prev_action = plan
        xy, th, v = sim.step_ego(plan, xy, th, v)
        frame = frame + 1
        rews.append(sim.reward(frame, xy, th, v))
        cols.append(sim.collisions(frame, xy, th))
        errs.append((log.ego[frame][:, 0:2] - xy).norm(dim=-1))
    rews = torch.stack(rews)
    cols = torch.stack(cols)
    errs = torch.stack(errs)
    return {"label": label,
            "sim_reward": float(rews.mean()),
            "sim_reward_late": float(rews[horizon // 2:].mean()),
            "pos_err_final": float(errs[-1].mean()),
            "collision_rate": float(cols.any(0).float().mean())}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum()
                 / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/b2d_v02/data")
    ap.add_argument("--out", default="runs/egosim_selector")
    ap.add_argument("--windows", type=int, default=192)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default=None,
                    help="comma-separated label filter (default: all)")
    args = ap.parse_args()
    device = args.device

    log = GlobalLog([Path(args.data) / "b2d_val.npz"], device=device)
    # expert executed controls for wp_head burn-in
    d = np.load(Path(args.data) / "b2d_val.npz")
    log._action_t = torch.as_tensor(d["action"], dtype=torch.float32,
                                    device=device)
    base = load_config("configs/b2d_v02.yaml")
    base.device = device
    sim = sim_from_config(base, log)

    pool = log.window_starts(args.horizon, max(sim.r.tau, 1))
    gen = torch.Generator()
    gen.manual_seed(0)
    starts = pool[torch.randint(len(pool), (args.windows,),
                                generator=gen)].to(device)
    # three batteries mirror where test-10 differentiates: clean cruise,
    # launches (stopped expert about to move), recovery from
    # perturbed poses (same-scene targets)
    batteries = {
        "clean": (starts, None),
        "launch": (launch_starts(log, pool, args.windows // 2), None),
        "divergent": (starts, {"frac": 1.0, "lat_sigma": 0.5,
                               "yaw_sigma": 0.1, "v_sigma": 1.0}),
    }
    print({k: len(s) for k, (s, _) in batteries.items()})

    keep = set(args.models.split(",")) if args.models else None
    rows = []
    for (label, cfg_path, ckpt_dir, pol, d10s, d10c) in REGISTRY:
        if keep is not None and label not in keep:
            continue
        if not (ckpt_dir / "checkpoints" / f"{pol}.pt").exists():
            print(f"SKIP {label}: no checkpoint")
            continue
        cfg, wm, policy = load_model(cfg_path, ckpt_dir, pol, device)
        m = {"label": label}
        for bat, (bstarts, perturb) in batteries.items():
            rng = torch.Generator(device=device)
            rng.manual_seed(1)
            bm = rollout_model(label, cfg, wm, policy, sim, log,
                               bstarts, args.horizon, device,
                               perturb=perturb, rng=rng)
            for k, v in bm.items():
                if k != "label":
                    m[f"{bat}_{k}"] = v
        # primary: battery-mean of the late-half reward (early steps
        # are on-manifold for everyone — uninformative)
        m["sim_score"] = float(np.mean(
            [m[f"{b}_sim_reward_late"] for b in batteries]))
        m["sim_reward"] = float(np.mean(
            [m[f"{b}_sim_reward"] for b in batteries]))
        m["pos_err_final"] = float(np.mean(
            [m[f"{b}_pos_err_final"] for b in batteries]))
        m["collision_rate"] = float(np.mean(
            [m[f"{b}_collision_rate"] for b in batteries]))
        m["d10_score"], m["d10_completion"] = d10s, d10c
        rows.append(m)
        print(f"{label:14s} score {m['sim_score']:.3f} "
              f"(cl {m['clean_sim_reward_late']:.3f} "
              f"la {m['launch_sim_reward_late']:.3f} "
              f"dv {m['divergent_sim_reward_late']:.3f}) "
              f"| err@H {m['pos_err_final']:.2f} m "
              f"| col {m['collision_rate']:.3f} "
              f"| dev10 {d10s:.2f}/{d10c:.1f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"windows": args.windows, "horizon": args.horizon,
           "burn_in": BURN_IN, "models": rows}
    for metric in ("sim_score", "sim_reward", "pos_err_final",
                   "collision_rate"):
        x = np.array([r[metric] for r in rows])
        sgn = -1.0 if metric in ("pos_err_final", "collision_rate") else 1.0
        res[f"spearman_{metric}_vs_d10_score"] = spearman(
            sgn * x, np.array([r["d10_score"] for r in rows]))
        res[f"spearman_{metric}_vs_d10_completion"] = spearman(
            sgn * x, np.array([r["d10_completion"] for r in rows]))
    (out / "selector.json").write_text(json.dumps(res, indent=2))

    lines = ["# G1: egosim-as-selector validation", "",
             f"{len(rows)} banked wp-family models, {args.windows} val "
             f"windows x {args.horizon} steps, burn-in {BURN_IN}; "
             "batteries: clean / launch / divergent (score = "
             "battery-mean late-half reward)", "",
             "| model | sim score | pos err@H | col rate | test-10 |",
             "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["sim_score"]):
        lines.append(f"| {r['label']} | {r['sim_score']:.3f} "
                     f"| {r['pos_err_final']:.2f} "
                     f"| {r['collision_rate']:.3f} "
                     f"| {r['d10_score']:.2f}/{r['d10_completion']:.1f} |")
    lines += ["", "Spearman (higher-is-better orientation):"]
    for k, v in res.items():
        if k.startswith("spearman"):
            lines.append(f"- {k}: {v:+.3f}")
    verdict = ("PASS" if res["spearman_sim_score_vs_d10_score"] >= 0.4
               else "FAIL")
    lines += ["", f"**G1 VERDICT: {verdict}** "
              "(gate: sim_score vs test-10 score >= +0.4; v0.1 latent "
              "metric was -0.60 on the same question)"]
    (out / "selector.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
