#!/usr/bin/env python3
"""G1: does the egosim reward rank banked v0.1 models like real CARLA?

Drives the wp-head model family (champion + seeds + probes + all gen-4
DWP doses; dev-10 DS spread 3.46-30.49) through EgoSim with their full
deployment semantics — WM posterior filter with the executed-control
feedback (TorchWaypointTracker port), deterministic policy — on held-out
val-split windows, and correlates the egosim metrics with the banked
dev-10 driving scores.

v0.1's on-policy latent metric scored Spearman -0.60 on this question
(runs/phase2_selector). V02_PLAN gate: clearly positive or DO NOT TRAIN.

Usage:
  python scripts/egosim_selector.py --npz /tmp/claude-601880/g1_val48.npz \
      [--windows 192] [--horizon 40] [--device cpu] [--out runs/egosim_g1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from ditto_av.config import load_config  # noqa: E402
from ditto_av.egosim import (EgoSim, GlobalLog, RewardParams,  # noqa: E402
                             SimParams)
from ditto_av.models.nets import make_actor_critic  # noqa: E402
from ditto_av.models.world_model import VectorWorldModel  # noqa: E402
from ditto_av.tracker_torch import (TorchWaypointTracker,  # noqa: E402
                                    wp_to_vehicle_t)

CKPT_BASE = Path.home() / "ditto_out"

# label, run dir under ~/ditto_out, policy ckpt name, banked dev-10 DS
# (sources: NEXT_STEPS.md @ saeed/ver0.1 + runs/carla_smoke ledgers)
REGISTRY = [
    ("wph champion s0", "b2d_gen3_wph",        "bc",       30.49),
    ("wph s1",          "b2d_gen3_wph_s1",     "bc",       25.86),
    ("wph s2",          "b2d_gen3_wph_s2",     "bc",       28.45),
    ("wph cap512x3",    "b2d_gen3_wph_cap",    "bc",       20.40),
    ("wph lw@4/2",      "b2d_gen3_wph_lw",     "bc",       19.72),
    ("wph lw2",         "b2d_gen3_wph_lw2",    "bc",       15.64),
    ("dwp v1 kl.1+div", "b2d_gen4_dwp",        "ditto_wp",  3.46),
    ("dwp kl.3+div",    "b2d_gen4_dwp_k03",    "ditto_wp", 19.49),
    ("dwp kl.3",        "b2d_gen4_dwp_k03nd",  "ditto_wp", 24.07),
    ("dwp kl.3 es3k",   "b2d_gen4_dwp_k03nd3k", "ditto_wp", 13.31),
    ("dwp kl1.0",       "b2d_gen4_dwp_k10nd",  "ditto_wp", 18.08),
    ("wp-action BC",    "b2d_gen3_wp",         "bc",       19.27),
]


class BatchedWMWpDriver:
    """Deployment ContinuousWMDriver semantics, batched for egosim.

    wp_head runs (gen3_wph/gen4): the WM's action channel is the
    EXECUTED control — the TorchWaypointTracker's reading of the plan
    (training-consistent, same as DittoCarlaAgent.set_executed).
    wp-as-action runs (gen3_wp): the raw 12-dim plan feeds back.
    Recovery/creep/gap are deployment-side levers, absent here (the
    imagination-has-no-wedges precedent) — noted in the verdict.
    """

    def __init__(self, wm, policy, wp_head: bool, action_dim: int,
                 device: str):
        self.wm, self.policy = wm, policy
        self.wp_head = wp_head
        self.action_dim = action_dim
        self.device = device
        self.tracker = TorchWaypointTracker()

    def reset(self, batch: int):
        self.state = self.wm.init_state(batch)
        self.prev = torch.zeros(batch, self.action_dim,
                                device=self.device)
        self.first = True

    @torch.no_grad()
    def act(self, obs: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        reset_t = torch.full((1, B), self.first, dtype=torch.bool,
                             device=self.device)
        feat, _, self.state = self.wm.observe(
            obs.view(1, B, -1), self.prev.view(1, B, -1), reset_t,
            self.state)
        self.first = False
        plan = self.policy.act(feat[0], stochastic=False)   # (B, 12)
        if self.wp_head:
            self.prev = self.tracker.act(wp_to_vehicle_t(plan), speed)
        else:
            self.prev = plan
        return plan


def load_model(run: str, policy_name: str, device: str):
    cfg = load_config(str(CKPT_BASE / run / "config.yaml"))
    ckpt_dir = CKPT_BASE / run / "checkpoints"
    wm = VectorWorldModel(cfg.env.obs_dim, cfg.env.action_dim,
                          cfg.wm).to(device)
    wm.load_state_dict(torch.load(ckpt_dir / "world_model.pt",
                                  map_location=device))
    wm.eval()
    hid, lay = ((cfg.bc.hidden_dim, cfg.bc.layers) if policy_name == "bc"
                else (cfg.ac.hidden_dim, cfg.ac.layers))
    policy = make_actor_critic(
        True, cfg.wm.feature_dim, cfg.env.policy_action_dim, hid, lay,
        action_space=("waypoints" if cfg.env.wp_out
                      else cfg.env.action_space)).to(device)
    policy.load_state_dict(torch.load(ckpt_dir / f"{policy_name}.pt",
                                      map_location=device))
    policy.eval()
    assert cfg.env.wp_out, f"{run} is not a wp-output model"
    return cfg, wm, policy


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum()
                 / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--windows", type=int, default=192)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/egosim_g1")
    args = ap.parse_args()
    device = args.device

    log = GlobalLog([args.npz], device=device)
    sim = EgoSim(log, SimParams(), RewardParams())
    pool = log.window_starts(args.horizon, sim.r.tau + 1)
    stride = max(1, len(pool) // args.windows)
    starts = pool[::stride][:args.windows]
    print(f"{log.obs.shape[0]} frames / {len(log.episodes)} clips; "
          f"{len(starts)} windows x H={args.horizon}")

    rows = []
    for label, run, pol_name, dev10 in REGISTRY:
        try:
            cfg, wm, policy = load_model(run, pol_name, device)
        except (FileNotFoundError, AssertionError) as e:
            print(f"SKIP {label}: {e}")
            continue
        driver = BatchedWMWpDriver(wm, policy, cfg.env.wp_head,
                                   cfg.env.action_dim, device)
        B = len(starts)
        driver.reset(B)
        xy, th, v = sim.reset(starts)
        frames = starts.clone()
        rew, col, err = [], [], []
        for _ in range(args.horizon):
            obs = sim.build_obs(frames, xy, th, v)
            plan = driver.act(obs, v)
            xy, th, v = sim.step_ego(plan, xy, th, v)
            frames = frames + 1
            rew.append(sim.reward(frames, xy, th, v))
            col.append(sim.collisions(frames, xy, th))
            err.append((log.ego[frames][:, 0:2] - xy).norm(dim=-1))
        rew = torch.stack(rew)
        col_rate = float(torch.stack(col).any(0).float().mean())
        row = {"label": label, "run": run, "policy": pol_name,
               "dev10_ds": dev10,
               "sim_reward": float(rew.mean()),
               "sim_collision": col_rate,
               "sim_pos_err_final": float(torch.stack(err)[-1].mean())}
        rows.append(row)
        print(f"{label:18s} dev10 {dev10:5.2f} | r {row['sim_reward']:.4f}"
              f" | col {col_rate:.3f} | perr {row['sim_pos_err_final']:.2f}")

    ds = np.array([r["dev10_ds"] for r in rows])
    verdict = {
        "n_models": len(rows),
        "windows": len(starts), "horizon": args.horizon,
        "npz": str(args.npz),
        "spearman_reward_vs_dev10": spearman(
            np.array([r["sim_reward"] for r in rows]), ds),
        "spearman_negcol_vs_dev10": spearman(
            -np.array([r["sim_collision"] for r in rows]), ds),
        "spearman_negperr_vs_dev10": spearman(
            -np.array([r["sim_pos_err_final"] for r in rows]), ds),
        "note": "deployment recovery/gap levers absent in sim; dev-10 "
                "ground truth uses each run's championed deployment",
        "rows": rows,
    }
    print("\nG1 VERDICT: Spearman(sim reward, dev-10 DS) = "
          f"{verdict['spearman_reward_vs_dev10']:+.3f} "
          f"(neg-collision {verdict['spearman_negcol_vs_dev10']:+.3f}, "
          f"neg-poserr {verdict['spearman_negperr_vs_dev10']:+.3f}) "
          f"over {len(rows)} models  [v0.1 latent metric: -0.60]")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"saved {out / 'g1_verdict.json'}")


if __name__ == "__main__":
    main()
