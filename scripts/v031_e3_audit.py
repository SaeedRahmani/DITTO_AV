#!/usr/bin/env python3
"""E3-A0: can the factored latent see ego deviation? (gates the RL variant)

The v0.1 failure mode, made testable: latent similarity must respond
to EGO deviation when traffic is shared. We rebuild sim-path obs at
laterally offset ego poses (same frames, same traffic) and measure
max_cos(h_offset, h_logged) through the frozen E3 world model.

PRE-REGISTERED PASS CRITERIA (committed before the wm checkpoint
existed; on FAIL fix the model or kill E3 — never these numbers):
  (a) control: obs rebuilt at the LOGGED pose scores >= 0.95 matched
      reward (the sim obs path and the log agree through the wm);
  (b) mean reward strictly decreases over offsets 0.5/1/2/4 m;
  (c) reward at 2 m <= 0.75 x reward at 0 m (dynamic range exists
      where the state kernel discriminates hard: exp(-2) = 0.135).

Usage: v031_e3_audit.py --data <dir> --wm-dir <v031_e3 run dir>
           --champion-cfg <yaml> [--n 512] [--burn 20] [--win 8]

--probe (the ONE pre-registered refinement, ledger 2026-08-05 17:50):
similarity is computed in the frozen wp-probe's projection h @ W
(checkpoints/wp_probe.pt from v031_e3_probe.py) instead of raw h;
same criteria, report -> results/a0_audit_probe.json. The raw-space
FAIL record in a0_audit.json stands either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ditto_av.config import load_config  # noqa: E402
from ditto_av.egosim import GlobalLog  # noqa: E402
from ditto_av.rewards import max_cos  # noqa: E402
from ditto_av.trainers.clp_trainer import sim_from_config  # noqa: E402
from ditto_av.trainers.wm_trainer import load_world_model  # noqa: E402

OFFSETS = [0.0, 0.5, 1.0, 2.0, 4.0]


def observe_windows(wm, obs_seq, wp, starts, burn, win):
    """Filter (B,) windows through the posterior; returns h (T, B, D).

    obs_seq: (T, B, O) — burn frames logged, then the variant's window.
    prev-action stream and reset mirror build_latent_bank exactly.
    """
    T, B, _ = obs_seq.shape
    prev = wp[(starts.unsqueeze(0)
               + torch.arange(T).unsqueeze(1)) - 1]        # (T, B, A)
    reset = torch.zeros(T, B, dtype=torch.bool)
    reset[0] = True
    prev = prev * (~reset).unsqueeze(-1)
    with torch.no_grad():
        _, (h, _), _ = wm.observe(obs_seq, prev, reset,
                                  wm.init_state(B))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wm-dir", required=True)
    ap.add_argument("--champion-cfg", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--burn", type=int, default=20)
    ap.add_argument("--win", type=int, default=8)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.champion_cfg)
    cfg.run_dir = args.wm_dir
    cfg.device = "cpu"
    cfg.env.action_space = "waypoints"
    cfg.env.wp_head = False

    log = GlobalLog([Path(args.data) / "b2d_val.npz"], device="cpu")
    wm = load_world_model(cfg, log.obs.shape[1])
    sim = sim_from_config(cfg, log)
    W = None
    if args.probe:
        W = torch.load(Path(args.wm_dir) / "checkpoints"
                       / "wp_probe.pt")["W"]

    T = args.burn + args.win
    pool = log.window_starts(T, 1)
    g = torch.Generator().manual_seed(0)
    starts = pool[torch.randperm(len(pool), generator=g)[:args.n]]
    B = len(starts)

    # logged-obs pass = the expert bank for these windows
    idx = starts.unsqueeze(0) + torch.arange(T).unsqueeze(1)  # (T, B)
    obs_log = log.obs[idx]
    h_bank = observe_windows(wm, obs_log, log.wp, starts,
                             args.burn, args.win)

    report = {"n": B, "burn": args.burn, "win": args.win,
              "space": "probe" if args.probe else "raw", "variants": {}}
    means = []
    for off in OFFSETS:
        obs_arm = obs_log.clone()
        for k in range(args.win):
            fr = starts + args.burn + k
            ego = log.ego[fr]
            xy, th, v = ego[:, 0:2], ego[:, 2], ego[:, 3]
            lat = torch.stack([torch.cos(th), torch.sin(th)], -1)
            obs_arm[args.burn + k] = sim.build_obs(
                fr, xy + off * lat, th, v)
        h_arm = observe_windows(wm, obs_arm, log.wp, starts,
                                args.burn, args.win)
        ha, hb = h_arm[args.burn:], h_bank[args.burn:]
        if W is not None:
            ha, hb = ha @ W, hb @ W
        r = max_cos(ha, hb)
        m = float(r.mean())
        means.append(m)
        report["variants"][str(off)] = m
        kern = float(np.exp(-0.5 * off ** 2))
        print(f"offset {off:3.1f} m: latent reward {m:.4f} "
              f"(state kernel ref {kern:.3f})")

    ok_a = means[0] >= 0.95
    ok_b = all(means[i] > means[i + 1] for i in range(len(means) - 1))
    ok_c = means[3] <= 0.75 * means[0]
    report["gates"] = {"control>=0.95": ok_a, "monotone": ok_b,
                       "2m<=0.75x": ok_c}
    verdict = ok_a and ok_b and ok_c
    report["verdict"] = "PASS" if verdict else "FAIL"
    name = "a0_audit_probe.json" if args.probe else "a0_audit.json"
    out = Path(args.wm_dir) / "results" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"E3-A0 {report['verdict']} "
          f"(a={ok_a} b={ok_b} c={ok_c}) -> {out}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
