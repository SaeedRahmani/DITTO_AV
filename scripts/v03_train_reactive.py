#!/usr/bin/env python3
"""D3 driver: champion fine-tune in the reactive world + both-world G2.

Usage:
  python scripts/v03_train_reactive.py \
      --data <dir with b2d_train/val.npz> --w0 <v03_w0c dir> \
      --champion <clp_rl.pt> --champion-cfg <b2d_v02_999s_cpu.yaml> \
      --out <run dir> [--steps 4000] [--device cuda]
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
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-ensemble", type=int, default=2)
    ap.add_argument("--p-reactive", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    # v0.3.1: static layout in the training world. --manifest ties
    # npz episode order to towns (must be the manifest the caches were
    # built from); --w-layout 0 = metric-only, >0 = penalty arm.
    ap.add_argument("--manifest",
                    default=str(REPO / "manifests" / "b2d_full999.txt"))
    ap.add_argument("--layout-dir", default=None,
                    help="Town*_lanes.npz dir; empty string disables")
    ap.add_argument("--w-layout", type=float, default=0.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    cfg = load_config(args.champion_cfg)
    cfg.device = device
    cfg.clp.rl_steps = args.steps

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

    lay = lay_val = None
    if args.layout_dir != "":
        from ditto_av.layout import LAYOUT_DIR, manifest_towns
        from ditto_av.layout_torch import LayoutQuery
        ldir = Path(args.layout_dir) if args.layout_dir else LAYOUT_DIR
        tr_towns, va_towns = manifest_towns(Path(args.manifest))
        lay = LayoutQuery(tr_towns, log.episodes, len(log.obs),
                          device, ldir)
        lay_val = (va_towns, ldir)
        print(f"layout: {len(set(tr_towns))} towns, "
              f"w_layout={args.w_layout}")

    policy = train_clp_reactive(cfg, log, rsim, Path(args.champion),
                                out, seed=cfg.seed,
                                p_reactive_max=args.p_reactive,
                                w1_thresh=W1_THRESH,
                                layout=lay, w_layout=args.w_layout)

    # both-world verdict on held-out clips
    vlog = GlobalLog([Path(args.data) / "b2d_val.npz"], device=device)
    vraw = np.load(Path(args.data) / "b2d_val.npz")
    vsw = load_sw(Path(args.w0) / "windows3_val.npz")
    vsim_p = sim_from_config(cfg, vlog)
    vrsim = ReactiveEgoSim(vlog, vsw, models, vraw["act_id"],
                           vsim_p.p, vsim_p.r)
    vlay = None
    if lay_val is not None:
        va_towns, ldir = lay_val
        vlay = LayoutQuery(va_towns, vlog.episodes, len(vlog.obs),
                           device, ldir)
    report = {"clp_rx": eval_both_worlds(cfg, policy, vlog, vrsim,
                                         W1_THRESH, layout=vlay)}
    # champion baseline rows
    from ditto_av.models.policy_v2 import make_token_policy
    champ = make_token_policy(cfg).to(device)
    champ.load_state_dict(torch.load(args.champion,
                                     map_location=device))
    champ.eval()
    report["champion"] = eval_both_worlds(cfg, champ, vlog, vrsim,
                                          W1_THRESH, layout=vlay)
    (out / "d3_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
