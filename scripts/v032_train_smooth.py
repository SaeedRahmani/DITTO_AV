#!/usr/bin/env python3
"""v0.3.2 axis-1 driver: D3 rerun + sigma_yawrate (V032_PLAN 1.3.1).

Identical to scripts/v03_train_reactive.py (champion fine-tune in the
reactive world, 3500 steps, p_r 0.5, W1 0.25) with ONE knob: the
training sims price motion quality (clp.sigma_yawrate > 0). The
both-world G2 verdict is computed with sigma_yawrate = 0 so the
yardstick is the SAME one that graded clp_rx — the arm must win on
the old currency while trained on the new one.

Usage:
  python scripts/v032_train_smooth.py \
      --data <dir with b2d_train/val.npz> --w0 <v03_w0c dir> \
      --champion <clp_rl.pt> --champion-cfg <b2d_v02_999s_cpu.yaml> \
      --out <run dir> [--sigma-yawrate 0.5] [--steps 3500]
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

from ditto_av.config import load_config  # noqa: E402
from ditto_av.egosim import GlobalLog  # noqa: E402
from ditto_av.models.traffic import SceneWindows, TrafficModel  # noqa: E402
from ditto_av.reactive import ReactiveEgoSim  # noqa: E402
from ditto_av.trainers.clp_trainer import sim_from_config  # noqa: E402
from ditto_av.trainers.reactive_trainer import (eval_both_worlds,  # noqa: E402
                                                train_clp_reactive)

W1_THRESH = 0.25   # calibrated: W0 round-4 disagreement p99 = 0.223


def load_sw(path: Path) -> SceneWindows:
    z = np.load(path)
    return SceneWindows(**{k: z[k] for k in z.files})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--w0", required=True)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--champion-cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-yawrate", type=float, default=0.5)
    ap.add_argument("--w-churn", type=float, default=0.0)
    ap.add_argument("--w-consistency", type=float, default=0.0)
    ap.add_argument("--cons-gate-d0", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=-1,
                    help="override cfg.seed (training seed)")
    ap.add_argument("--n-ensemble", type=int, default=2)
    ap.add_argument("--p-reactive", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    cfg = load_config(args.champion_cfg)
    cfg.device = device
    cfg.clp.rl_steps = args.steps
    cfg.clp.sigma_yawrate = args.sigma_yawrate   # TRAIN sims only
    cfg.clp.w_churn = args.w_churn               # rollout-only term
    cfg.clp.w_consistency = args.w_consistency   # actor aux loss
    cfg.clp.cons_gate_d0 = args.cons_gate_d0
    if args.seed >= 0:
        cfg.seed = args.seed

    log = GlobalLog([Path(args.data) / "b2d_train.npz"], device=device)
    raw = np.load(Path(args.data) / "b2d_train.npz")
    sw = load_sw(Path(args.w0) / "windows3_train.npz")
    models = []
    for k in range(args.n_ensemble):
        m = TrafficModel(hist=10).to(device)
        m.load_state_dict(torch.load(
            Path(args.w0) / "checkpoints" / f"traffic_s{k}_rf.pt",
            map_location=device))
        m.eval()
        models.append(m)
    rsim_p = sim_from_config(cfg, log)
    rsim = ReactiveEgoSim(log, sw, models, raw["act_id"],
                          rsim_p.p, rsim_p.r)

    policy = train_clp_reactive(cfg, log, rsim, Path(args.champion),
                                out, seed=cfg.seed,
                                p_reactive_max=args.p_reactive,
                                w1_thresh=W1_THRESH)

    # both-world verdict on held-out clips — OLD yardstick (sigma 0)
    cfg.clp.sigma_yawrate = 0.0
    vlog = GlobalLog([Path(args.data) / "b2d_val.npz"], device=device)
    vraw = np.load(Path(args.data) / "b2d_val.npz")
    vsw = load_sw(Path(args.w0) / "windows3_val.npz")
    vsim_p = sim_from_config(cfg, vlog)
    vrsim = ReactiveEgoSim(vlog, vsw, models, vraw["act_id"],
                           vsim_p.p, vsim_p.r)
    report = {"sigma_yawrate": args.sigma_yawrate,
              "w_churn": args.w_churn,
              "w_consistency": args.w_consistency,
              "clp_sm": eval_both_worlds(cfg, policy, vlog, vrsim,
                                         W1_THRESH)}
    # clp_rx baseline row on the same held-out sims
    from ditto_av.models.policy_v2 import make_token_policy
    rx_ckpt = Path.home() / "ditto_out/v03_d3/clp_rx.pt"
    if rx_ckpt.exists():
        rx = make_token_policy(cfg).to(device)
        rx.load_state_dict(torch.load(rx_ckpt, map_location=device))
        rx.eval()
        report["clp_rx"] = eval_both_worlds(cfg, rx, vlog, vrsim,
                                            W1_THRESH)
    (out / "d3sm_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
