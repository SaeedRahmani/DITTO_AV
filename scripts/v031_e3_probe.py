#!/usr/bin/env python3
"""E3 refinement (gen-4 precedent): frozen wp-probe projection.

Fits a closed-form ridge probe W (D_h -> 12) from the frozen wm's
posterior h to the expert wp labels on the TRAIN split; the audit
then re-tests deviation sensitivity in probe space (h @ W) on VAL.
The wm stays frozen; only the reward's similarity space changes —
the one committed refinement, judged by the same audit criteria.

Usage: v031_e3_probe.py --data <dir> --wm-dir <v031_e3>
           --champion-cfg <yaml> [--ridge 1e-3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ditto_av.config import load_config  # noqa: E402
from ditto_av.egosim import GlobalLog  # noqa: E402
from ditto_av.trainers.wm_trainer import load_world_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wm-dir", required=True)
    ap.add_argument("--champion-cfg", required=True)
    ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = load_config(args.champion_cfg)
    cfg.run_dir = args.wm_dir
    cfg.device = "cpu"
    cfg.env.action_space = "waypoints"
    cfg.env.wp_head = False

    log = GlobalLog([Path(args.data) / "b2d_train.npz"], device="cpu")
    wm = load_world_model(cfg, log.obs.shape[1])

    hs, ys = [], []
    with torch.no_grad():
        for s, e in log.episodes:
            T = e - s
            obs = log.obs[s:e].unsqueeze(1)
            prev = torch.zeros(T, 1, log.wp.shape[1])
            prev[1:, 0] = log.wp[s:e - 1]
            reset = torch.zeros(T, 1, dtype=torch.bool)
            reset[0, 0] = True
            _, (h, _), _ = wm.observe(obs, prev, reset,
                                      wm.init_state(1))
            hs.append(h.squeeze(1))
            ys.append(log.wp[s:e])
    H = torch.cat(hs).double()
    Y = torch.cat(ys).double()
    print(f"probe fit: {H.shape[0]} frames, h {H.shape[1]} -> "
          f"wp {Y.shape[1]}")
    # ridge: W = (H^T H + lam I)^-1 H^T Y
    D = H.shape[1]
    A = H.T @ H + args.ridge * H.shape[0] * torch.eye(D,
                                                      dtype=torch.double)
    W = torch.linalg.solve(A, H.T @ Y)
    pred = H @ W
    r2 = 1.0 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f"probe train R^2: {float(r2):.4f}")
    out = Path(args.wm_dir) / "checkpoints" / "wp_probe.pt"
    torch.save({"W": W.float(), "r2": float(r2)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
