#!/usr/bin/env python3
"""DITTO-AV pipeline: collect -> world model -> policies -> closed-loop eval.

Usage:
    python scripts/run_pipeline.py --config configs/av.yaml --stage all
    python scripts/run_pipeline.py --stage collect|wm|policies|eval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# tiny tensors + many sequential RSSM ops: multi-threaded BLAS is a 15-30x
# slowdown from thread oversubscription on this workload
torch.set_num_threads(1)

from ditto_av.collect import run_collection  # noqa: E402
from ditto_av.config import load_config, save_config  # noqa: E402
from ditto_av.data import TrajectoryData, build_latent_bank  # noqa: E402
from ditto_av.evaluate import (ExpertDriver, RandomDriver,  # noqa: E402
                               WMPolicyDriver, evaluate_suite)
from ditto_av.models.nets import ActorCritic  # noqa: E402
from ditto_av.trainers.ac_trainer import train_latent_policy  # noqa: E402
from ditto_av.trainers.bc_trainer import train_bc  # noqa: E402
from ditto_av.trainers.wm_trainer import (load_world_model,  # noqa: E402
                                          train_world_model)


def expert_paths(cfg):
    d = cfg.dirs()
    return [d["data"] / "expert.npz"]


def all_paths(cfg):
    d = cfg.dirs()
    paths = [d["data"] / "expert.npz"]
    noisy = d["data"] / "noisy.npz"
    if noisy.exists():
        paths.append(noisy)
    return paths


def stage_collect(cfg):
    run_collection(cfg)


def stage_wm(cfg):
    data = TrajectoryData(all_paths(cfg))
    print(f"world-model data: {len(data.obs)} steps, "
          f"{len(data.episodes)} episodes")
    train_world_model(cfg, data, seed=cfg.seed)


def stage_policies(cfg):
    wm = load_world_model(cfg, cfg.env.obs_dim)
    expert_data = TrajectoryData(expert_paths(cfg))
    bank = build_latent_bank(wm, expert_data, cfg.env.action_dim,
                             cfg.ac.horizon, cfg.device)
    print(f"latent bank: {bank.feat.shape[0]} steps, "
          f"{bank.n_windows} windows")
    train_bc(cfg, bank, seed=cfg.seed)
    train_latent_policy(cfg, wm, bank, reward_mode="single", seed=cfg.seed)
    train_latent_policy(cfg, wm, bank, reward_mode="multi", seed=cfg.seed)


def stage_eval(cfg):
    device = cfg.device
    wm = load_world_model(cfg, cfg.env.obs_dim)
    d = cfg.dirs()

    def load_policy(name, hidden, layers):
        p = ActorCritic(cfg.wm.feature_dim, cfg.env.action_dim, hidden,
                        layers).to(device)
        p.load_state_dict(torch.load(d["ckpt"] / f"{name}.pt",
                                     map_location=device))
        p.eval()
        return p

    drivers = {
        "expert": ExpertDriver(cfg.collect.aggressive_prob, seed=1),
        "random": RandomDriver(seed=1),
        "bc": WMPolicyDriver(wm, load_policy("bc", cfg.bc.hidden_dim,
                                             cfg.bc.layers),
                             cfg.env.action_dim, device,
                             cfg.eval.stochastic),
        "ditto_single": WMPolicyDriver(wm, load_policy("ditto_single",
                                                       cfg.ac.hidden_dim,
                                                       cfg.ac.layers),
                                       cfg.env.action_dim, device,
                                       cfg.eval.stochastic),
        "ditto_multi": WMPolicyDriver(wm, load_policy("ditto_multi",
                                                      cfg.ac.hidden_dim,
                                                      cfg.ac.layers),
                                      cfg.env.action_dim, device,
                                      cfg.eval.stochastic),
    }
    evaluate_suite(cfg, drivers, out_path=d["results"] / "results.json")


STAGES = {
    "collect": stage_collect,
    "wm": stage_wm,
    "policies": stage_policies,
    "eval": stage_eval,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default="all",
                        choices=list(STAGES) + ["all"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_config(cfg, cfg.dirs()["run"] / "config.yaml")

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        print(f"\n===== stage: {s} =====")
        STAGES[s](cfg)
        print(f"===== stage {s} done in {time.time() - t0:.0f}s =====")


if __name__ == "__main__":
    main()
