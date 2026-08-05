#!/usr/bin/env python3
"""E3 step 1: world model for the factored-latent reward.

Trains VectorWorldModel on the SAME 999-split training log the
champion saw, with the wp plans as the action stream (action_space
"waypoints" -> action_dim 12): in sim rollouts the executed action IS
the policy's clamped plan, so the posterior can be advanced with
exactly the inputs the reward will see. Frozen afterwards; the E3-A0
audit (deviation sensitivity) gates any RL use.

Usage:
  python scripts/v031_e3_wm.py --data <dir with b2d_train.npz> \
      --champion-cfg <b2d_v02_999s_cpu.yaml> --out <run dir> \
      [--steps 4000] [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ditto_av.config import load_config  # noqa: E402
from ditto_av.data import TrajectoryData  # noqa: E402
from ditto_av.trainers.wm_trainer import train_world_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--champion-cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.champion_cfg)
    cfg.run_dir = args.out
    cfg.device = args.device
    # the wm's action stream is the wp plan (what rollouts execute),
    # not the control triple the wp_head config would report
    cfg.env.action_space = "waypoints"
    cfg.env.wp_head = False
    cfg.wm.train_steps = args.steps

    data = TrajectoryData([Path(args.data) / "b2d_train.npz"],
                          action_key="wp")
    print(f"wm data: {len(data.obs)} frames, {len(data.episodes)} "
          f"episodes, obs {data.obs_dim}, act {data.action.shape[1]}")
    assert data.action.shape[1] == cfg.env.action_dim
    train_world_model(cfg, data, seed=cfg.seed)


if __name__ == "__main__":
    main()
