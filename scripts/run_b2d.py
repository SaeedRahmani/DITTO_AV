#!/usr/bin/env python3
"""Phase-2 pipeline: Bench2Drive clips -> world model -> policies -> eval.

Fully offline. Evaluation is open-loop on held-out clips (closed-loop CARLA
is a separate task): world-model reconstruction / H-step prediction error,
policy action NLL/MAE on the posterior, and latent imitation score of
imagined policy rollouts against the held-out latent trajectories.

Usage:
    python scripts/run_b2d.py --config configs/b2d.yaml --stage all \
        --manifest manifests/b2d_clips.txt \
        --extracted /scratch/$USER/ditto_av/data/bench2drive/extracted
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from ditto_av.bench2drive import clips_to_npz  # noqa: E402
from ditto_av.config import load_config, save_config  # noqa: E402
from ditto_av.data import TrajectoryData, build_latent_bank  # noqa: E402
from ditto_av.models.nets import make_actor_critic  # noqa: E402
from ditto_av.rewards import max_cos  # noqa: E402
from ditto_av.trainers.ac_trainer import train_latent_policy  # noqa: E402
from ditto_av.trainers.bc_trainer import train_bc  # noqa: E402
from ditto_av.trainers.wm_trainer import (load_world_model,  # noqa: E402
                                          train_world_model)

VAL_EVERY = 6  # every 6th manifest clip is held out (~17%)


def split_clips(manifest: Path, extracted: Path):
    names = sorted(ln.strip().removesuffix(".tar.gz")
                   for ln in manifest.read_text().splitlines() if ln.strip())
    dirs = [extracted / n for n in names]
    have = [d for d in dirs if (d / "anno").is_dir()]
    missing = len(dirs) - len(have)
    train = [d for i, d in enumerate(have) if i % VAL_EVERY != VAL_EVERY - 1]
    val = [d for i, d in enumerate(have) if i % VAL_EVERY == VAL_EVERY - 1]
    print(f"clips: {len(have)} extracted ({missing} missing), "
          f"{len(train)} train / {len(val)} val")
    return train, val


def stage_data(cfg, args):
    d = cfg.dirs()
    train, val = split_clips(Path(args.manifest), Path(args.extracted))
    tr = clips_to_npz(train, d["data"] / "b2d_train.npz")
    va = clips_to_npz(val, d["data"] / "b2d_val.npz")
    print(f"train: {tr['obs'].shape[0]} frames / {int(tr['reset'].sum())} "
          f"clips | val: {va['obs'].shape[0]} frames / "
          f"{int(va['reset'].sum())} clips")


def stage_wm(cfg, args):
    data = TrajectoryData([cfg.dirs()["data"] / "b2d_train.npz"])
    print(f"world-model data: {len(data.obs)} steps, "
          f"{len(data.episodes)} episodes, "
          f"discrete={data.discrete_actions}")
    train_world_model(cfg, data, seed=cfg.seed)


def stage_policies(cfg, args):
    wm = load_world_model(cfg, cfg.env.obs_dim)
    data = TrajectoryData([cfg.dirs()["data"] / "b2d_train.npz"])
    bank = build_latent_bank(wm, data, cfg.env.action_dim, cfg.ac.horizon,
                             cfg.device)
    print(f"latent bank: {bank.feat.shape[0]} steps, {bank.n_windows} windows")
    train_bc(cfg, bank, seed=cfg.seed)
    train_latent_policy(cfg, wm, bank, reward_mode="single", seed=cfg.seed)
    train_latent_policy(cfg, wm, bank, reward_mode="multi", seed=cfg.seed)


@torch.no_grad()
def _dream_rollout(wm, bank, obs, start_flat, policy=None, actions=None,
                   horizon=15, chunk=512):
    """Imagined rollouts from bank start states.

    Either `policy` acts, or recorded `actions` (N, H, A) are replayed.
    Returns (latent_match, obs_mse): mean max_cos of dreamed h against the
    recorded latent window, and decoded-obs MSE per dim.
    """
    n = len(start_flat)
    sims, mses = [], []
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        flat = start_flat[sl]
        h = bank.h[flat].clone()
        z = bank.z[flat].clone()
        dreamed_h, dreamed_feat = [], []
        for t in range(horizon):
            feat = torch.cat((h, z), dim=-1)
            if policy is not None:
                a = policy.act(feat, stochastic=False)
            else:
                a = actions[sl][:, t]
            h, z = wm.dream(a, (h, z))
            dreamed_h.append(h)
            dreamed_feat.append(torch.cat((h, z), dim=-1))
        dh = torch.stack(dreamed_h, dim=1)            # (B, H, D)
        target_h = bank.h[bank.window_idx[sl][:, 1:]]  # (B, H, D)
        sims.append(max_cos(dh, target_h).mean(dim=1))
        pred = wm.decoder(torch.stack(dreamed_feat, dim=1))
        target_obs = obs[bank.window_idx[sl][:, 1:]]
        mses.append(((pred - target_obs) ** 2).mean(dim=(1, 2)))
    return (float(torch.cat(sims).mean()), float(torch.cat(mses).mean()))


@torch.no_grad()
def stage_eval(cfg, args):
    device = cfg.device
    d = cfg.dirs()
    wm = load_world_model(cfg, cfg.env.obs_dim)
    val = TrajectoryData([d["data"] / "b2d_val.npz"])
    bank = build_latent_bank(wm, val, cfg.env.action_dim, cfg.ac.horizon,
                             cfg.device)
    obs = torch.as_tensor(val.obs, device=device)
    act = torch.as_tensor(val.action, device=device)
    H = cfg.ac.horizon

    results = {"n_val_frames": len(val.obs), "n_val_clips": len(val.episodes),
               "horizon": H}

    # world model: teacher-forced reconstruction (per-dim MSE)
    recon = wm.decoder(bank.feat)
    results["wm_recon_mse"] = float(((recon - obs) ** 2).mean())

    # world model ceiling: H-step dream replaying the expert's actions
    start = bank.window_starts
    replay = act[bank.window_idx[:, :-1]]  # action at window steps 0..H-1
    sim, mse = _dream_rollout(wm, bank, obs, start, actions=replay, horizon=H)
    results["expert_replay"] = {"latent_match": sim, "hstep_obs_mse": mse}

    # policies
    results["policies"] = {}
    for name in ("bc", "ditto_single", "ditto_multi"):
        ckpt = d["ckpt"] / f"{name}.pt"
        if not ckpt.exists():
            continue
        hid, lay = ((cfg.bc.hidden_dim, cfg.bc.layers) if name == "bc"
                    else (cfg.ac.hidden_dim, cfg.ac.layers))
        policy = make_actor_critic(cfg.env.continuous, cfg.wm.feature_dim,
                                   cfg.env.action_dim, hid, lay).to(device)
        policy.load_state_dict(torch.load(ckpt, map_location=device))
        policy.eval()
        dist = policy.dist(bank.feat)
        nll = float(-dist.log_prob(bank.action).mean())
        pred = policy.act(bank.feat, stochastic=False)
        if cfg.env.continuous:
            mae = float((pred - bank.action).abs().mean())
        else:
            mae = float((pred != bank.action).float().mean())
        sim, mse = _dream_rollout(wm, bank, obs, start, policy=policy,
                                  horizon=H)
        results["policies"][name] = {
            "action_nll": nll, "action_mae": mae,
            "latent_match": sim, "hstep_obs_mse": mse}
        print(f"{name:14s} nll {nll:7.3f} | mae {mae:.4f} "
              f"| latent match {sim:.4f} | H-step obs mse {mse:.5f}")

    out = d["results"] / "results.json"
    out.write_text(json.dumps(results, indent=2))
    lines = ["# Bench2Drive open-loop results", "",
             f"val: {results['n_val_clips']} clips, "
             f"{results['n_val_frames']} frames | horizon {H}", "",
             f"WM teacher-forced recon MSE: {results['wm_recon_mse']:.5f} | "
             f"expert-replay latent match "
             f"{results['expert_replay']['latent_match']:.4f} "
             f"(dynamics ceiling)", "",
             "| policy | action NLL | action MAE | latent match "
             "| H-step obs MSE |", "|---|---|---|---|---|"]
    for name, m in results["policies"].items():
        lines.append(f"| {name} | {m['action_nll']:.3f} "
                     f"| {m['action_mae']:.4f} | {m['latent_match']:.4f} "
                     f"| {m['hstep_obs_mse']:.5f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"saved {out} and {out.with_suffix('.md')}")


STAGES = {"data": stage_data, "wm": stage_wm, "policies": stage_policies,
          "eval": stage_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/b2d.yaml")
    ap.add_argument("--stage", default="all",
                    choices=list(STAGES) + ["all"])
    ap.add_argument("--manifest", default="manifests/b2d_clips.txt")
    ap.add_argument("--extracted",
                    default="/scratch/srahmani/ditto_av/data/bench2drive/"
                            "extracted")
    args = ap.parse_args()
    cfg = load_config(args.config)
    save_config(cfg, cfg.dirs()["run"] / "config.yaml")
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        print(f"\n===== stage: {s} =====")
        STAGES[s](cfg, args)
        print(f"===== stage {s} done in {time.time() - t0:.0f}s =====")


if __name__ == "__main__":
    main()
