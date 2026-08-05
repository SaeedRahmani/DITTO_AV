#!/usr/bin/env python3
"""A0 smoothness audit (V032_PLAN 1.2): where is the jiggle born?

Populations:
  expert    — logged expert states + wp labels + expert steer (npz)
  openloop  — a policy's plans teacher-forced on logged obs (npz)
  egosim    — a policy closed-loop in the plain log-replay EgoSim
  carla     — deployed per-tick logs (agent_ticks_*.jsonl)

Each subcommand prints a compact summary and writes
outputs/v032/a0_<pop>[_<tag>].json. All CPU, val split by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ditto_av.smoothness import (FPS, episode_ids, plan_churn,  # noqa: E402
                                 sign_flips_per_100, summarize, yaw_rate)

OUT = Path("/scratch/srahmani/ditto_av/outputs/v032")
STEER_DB = 0.02   # steer command deadband (fraction of full lock)
YR_DB = 0.02      # yaw-rate deadband, rad/s


def _save(name: str, payload: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"a0_{name}.json"
    p.write_text(json.dumps(payload, indent=1))
    print(f"-> {p}")


def _state_metrics(theta, ep):
    yr = yaw_rate(theta, ep)
    ya = np.diff(yr) * FPS          # NaNs propagate over boundaries
    return {"yaw_rate": summarize(yr),
            "yaw_accel": summarize(ya),
            "yr_flips_per_100": sign_flips_per_100(
                np.nan_to_num(yr), YR_DB, ep[:-1])}


def _churn_metrics(plans, xy, theta, ep):
    ch = plan_churn(plans, xy, theta, ep)
    return {k: summarize(v) for k, v in ch.items()}


def run_expert(args):
    z = np.load(args.data)
    ep = episode_ids(z["reset"])
    ego = z["ego_glob"]
    from ditto_av.bench2drive import WP_SCALE
    plans = z["wp"].reshape(len(ego), -1, 2) * WP_SCALE
    out = _state_metrics(ego[:, 2], ep)
    out["steer_flips_per_100"] = sign_flips_per_100(
        z["action"][:, 1], STEER_DB, ep)
    out["steer_dstep"] = summarize(np.where(
        np.diff(ep) == 0, np.diff(z["action"][:, 1]), np.nan))
    out["label_churn"] = _churn_metrics(plans, ego[:, 0:2], ego[:, 2], ep)
    # position-noise floor: along-track accel of the logged positions
    v = ego[:, 3]
    acc = np.where(np.diff(ep) == 0, np.diff(v) * FPS, np.nan)
    out["accel"] = summarize(acc)
    out["n_frames"] = int(len(ego))
    print(json.dumps(out, indent=1))
    _save("expert", out)


def _load_policy(cfg_path: str, ckpt: str):
    import torch
    from ditto_av.config import load_config
    from ditto_av.models.policy_v2 import make_token_policy
    cfg = load_config(cfg_path)
    cfg.device = "cpu"
    pol = make_token_policy(cfg)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return cfg, pol


def run_openloop(args):
    import torch
    from ditto_av.bench2drive import WP_SCALE
    cfg, pol = _load_policy(args.cfg, args.ckpt)
    z = np.load(args.data)
    ep = episode_ids(z["reset"])
    ego = z["ego_glob"]
    obs = torch.as_tensor(z["obs"], dtype=torch.float32)
    eps = []
    start = 0
    for i in range(1, len(ep) + 1):
        if i == len(ep) or ep[i] != ep[start]:
            eps.append((start, i))
            start = i
    if args.limit:
        eps = eps[:args.limit]
    lat_mean, lat_wp1, proxy = [], [], []
    with torch.no_grad():
        for s, e in eps:
            feats, _ = pol.unroll(obs[s:e].unsqueeze(1))
            a = pol.clamp(pol.dist(feats).base_dist.loc.squeeze(1))
            plans = a.view(e - s, -1, 2).numpy() * WP_SCALE
            ch = plan_churn(plans, ego[s:e, 0:2], ego[s:e, 2])
            lat_mean.append(ch["lat_mean"])
            lat_wp1.append(ch["lat_wp1"])
            proxy.append(ch["wp1_proxy"])
    out = {"episodes": len(eps),
           "lat_mean": summarize(np.concatenate(lat_mean)),
           "lat_wp1": summarize(np.concatenate(lat_wp1)),
           "wp1_proxy": summarize(np.concatenate(proxy))}
    print(json.dumps(out, indent=1))
    _save(f"openloop_{args.tag}", out)


def run_egosim(args):
    import torch
    from ditto_av.bench2drive import WP_SCALE
    from ditto_av.egosim import GlobalLog
    from ditto_av.trainers.clp_trainer import sim_from_config
    if args.labels:
        from ditto_av.config import load_config
        cfg, pol = load_config(args.cfg), None
        cfg.device = "cpu"
    else:
        cfg, pol = _load_policy(args.cfg, args.ckpt)
    log = GlobalLog([Path(args.data)], device="cpu")
    sim = sim_from_config(cfg, log)
    c = cfg.clp
    H = args.horizon or c.horizon
    pool = log.window_starts(H + c.reward_tau + 1, 0)
    idx = torch.linspace(0, len(pool) - 1, args.batch).long()
    starts = pool[idx]
    B = len(starts)
    lo = log.ep_start[starts]
    bidx = (starts.unsqueeze(1)
            + torch.arange(-c.burn_in, 0)).clamp_min(lo.unsqueeze(1))
    with torch.no_grad():
        if pol is not None:
            _, h = pol.unroll(log.obs[bidx].transpose(0, 1).contiguous())
        xy, th, v = sim.reset(starts)
        frame = starts.clone()
        ths, xys, plans_l, rews, cols = [th], [xy], [], [], []
        for _ in range(H):
            if pol is None:
                a = log.wp[frame]          # expert labels, replayed
            else:
                o = sim.build_obs(frame, xy, th, v)
                emb = pol.encode(o)
                h = pol.step(emb, h)
                a = pol.clamp(pol.dist(pol.features(emb, h)).base_dist.loc)
            plans_l.append(a.view(B, -1, 2) * WP_SCALE)
            xy, th, v = sim.step_ego(a, xy, th, v)
            frame = frame + 1
            rews.append(sim.reward(frame, xy, th, v))
            cols.append(sim.collisions(frame, xy, th))
            ths.append(th)
            xys.append(xy)
    theta = torch.stack(ths).numpy()                 # (H+1, B)
    xyn = torch.stack(xys).numpy()                   # (H+1, B, 2)
    plans = torch.stack(plans_l).numpy()             # (H, B, k, 2)
    ep = np.repeat(np.arange(B)[None, :], H + 1, 0)
    out = _state_metrics(theta.T.ravel(), ep.T.ravel())
    lat_mean, lat_wp1, proxy = [], [], []
    for b in range(B):
        ch = plan_churn(plans[:, b], xyn[:-1, b], theta[:-1, b])
        lat_mean.append(ch["lat_mean"])
        lat_wp1.append(ch["lat_wp1"])
        proxy.append(ch["wp1_proxy"])
    out["plan_churn"] = {
        "lat_mean": summarize(np.concatenate(lat_mean)),
        "lat_wp1": summarize(np.concatenate(lat_wp1)),
        "wp1_proxy": summarize(np.concatenate(proxy))}
    out["reward_mean"] = float(torch.stack(rews).mean())
    out["collision_rate"] = float(torch.stack(cols).any(0).float().mean())
    out["rollouts"] = B
    out["horizon"] = H
    print(json.dumps(out, indent=1))
    _save(f"egosim_{args.tag}", out)


def run_carla(args):
    segs = []          # lists of (steer, wp1lat, alpha, speed) rows
    n_rec = 0
    for f in args.ticks:
        cur = []
        last_step = None
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            new_ep = last_step is not None and d["step"] <= last_step
            last_step = d["step"]
            if d.get("rec", 0):
                n_rec += 1
                if cur:
                    segs.append(cur)
                cur = []
                continue
            if new_ep and cur:
                segs.append(cur)
                cur = []
            wp1 = d.get("wp1") or [np.nan, np.nan]
            trk = d.get("trk") or {}
            cur.append((d["steer"], wp1[1], trk.get("alpha", np.nan),
                        d.get("speed", np.nan)))
        if cur:
            segs.append(cur)
    steer = np.concatenate([[r[0] for r in s] for s in segs])
    ep = np.concatenate([[i] * len(s) for i, s in enumerate(segs)])
    wp1lat = np.concatenate([[r[1] for r in s] for s in segs])
    alpha = np.concatenate([[r[2] for r in s] for s in segs])
    speed = np.concatenate([[r[3] for r in s] for s in segs])
    same = np.diff(ep) == 0
    mov = same & (speed[:-1] > 1.0) & (speed[1:] > 1.0)
    out = {
        "ticks": int(len(steer)), "segments": len(segs),
        "rec_ticks_dropped": n_rec,
        "steer_flips_per_100": sign_flips_per_100(steer, STEER_DB, ep),
        "steer_flips_per_100_moving": sign_flips_per_100(
            np.where(speed > 1.0, steer, 0.0), STEER_DB, ep),
        "steer_dstep": summarize(np.where(same, np.diff(steer), np.nan)),
        "steer_abs": summarize(steer),
        "alpha_flips_per_100": sign_flips_per_100(
            np.nan_to_num(alpha), 0.01, ep),
        "wp1_proxy": summarize(np.where(mov, np.abs(np.diff(wp1lat)),
                                        np.nan)),
    }
    print(json.dumps(out, indent=1))
    _save(f"carla_{args.tag}", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="pop", required=True)
    d = "/home/srahmani/ditto_out/v03_w0c/data/b2d_val.npz"
    p = sub.add_parser("expert")
    p.add_argument("--data", default=d)
    p.set_defaults(fn=run_expert)
    p = sub.add_parser("openloop")
    p.add_argument("--data", default=d)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=run_openloop)
    p = sub.add_parser("egosim")
    p.add_argument("--data", default=d)
    p.add_argument("--ckpt", default="")
    p.add_argument("--cfg", required=True)
    p.add_argument("--labels", action="store_true",
                   help="replay expert wp labels through step_ego "
                        "(sim execution-noise floor; no policy)")
    p.add_argument("--tag", required=True)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--horizon", type=int, default=0)
    p.set_defaults(fn=run_egosim)
    p = sub.add_parser("carla")
    p.add_argument("--ticks", nargs="+", required=True)
    p.add_argument("--tag", required=True)
    p.set_defaults(fn=run_carla)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
