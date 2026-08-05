#!/usr/bin/env python3
"""V3-M2b: train the K-seed traffic-model ensemble and run the W0 gate.

Stages:
  1. build (or load cached) scene windows from the ##glob2 npz pair;
  2. train K TrafficModel seeds (teacher-forced NLL);
  3. W0 gate on held-out clips (pre-registered, V03_PLAN §4):
     - rollout ADE @4 s <= 3.80 m (>=20% under the CTRV floor 4.75);
     - agent-agent proximity-event rate within 2x of the log's;
     - reactivity probe: braking ego -> followers slow down (effect
       size reported; CV ghosts = 0 by construction);
     - ensemble disagreement percentiles (W1 calibration).
Writes <out>/w0_report.{json,md} with PASS/FAIL per criterion.

Usage:
  python scripts/v03_train_traffic.py --data runs/v03_w0/data \
      --out runs/v03_w0 [--seeds 4] [--steps 25000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ditto_av.models.traffic import (TrafficModel, build_scene_windows,  # noqa: E402
                                     featurize)
from ditto_av import wandb_util  # noqa: E402

CV_FLOOR_4S = 5.02   # audited (runs/v03_audit)
CTRV_FLOOR_4S = 4.75
W0_ADE_GATE = 3.80   # pre-registered: >=20% under CTRV
HIST = 10
ROLL = 40            # 4 s


def load_windows(npz_path: Path, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        from ditto_av.models.traffic import SceneWindows
        return SceneWindows(**{k: z[k] for k in z.files})
    data = dict(np.load(npz_path))
    t0 = time.time()
    sw = build_scene_windows(data, hist=HIST)
    print(f"windows({npz_path.name}): {len(sw.frames)} scenes "
          f"in {time.time()-t0:.0f}s")
    np.savez_compressed(cache, frames=sw.frames, hist=sw.hist,
                        pred_mask=sw.pred_mask, pres_mask=sw.pres_mask,
                        cls=sw.cls, ego=sw.ego, light=sw.light,
                        target=sw.target, ids=sw.ids)
    return sw


def batch_tensors(sw, idx, device):
    return (torch.as_tensor(sw.hist[idx], device=device),
            torch.as_tensor(sw.pres_mask[idx], device=device),
            torch.as_tensor(sw.cls[idx], device=device).long(),
            torch.as_tensor(sw.ego[idx], device=device),
            torch.as_tensor(sw.light[idx], device=device),
            torch.as_tensor(sw.target[idx], device=device),
            torch.as_tensor(sw.pred_mask[idx], device=device))


def train_seed(sw, seed, steps, batch, device, lr=3e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TrafficModel(hist=HIST).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(sw.frames)
    print(f"seed {seed}: TrafficModel {n_par/1e6:.2f}M params, "
          f"{n} train scenes")
    for step in range(1, steps + 1):
        idx = rng.integers(n, size=batch)
        h, pr, cl, eg, li, tg, pm = batch_tensors(sw, idx, device)
        loss = model.loss(h, pr, cl, eg, li, tg, pm.float())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            wandb_util.log({"step": step, "nll": float(loss)},
                           prefix=f"traffic_s{seed}")
        if step % 5000 == 0 or step == 1:
            print(f"seed {seed} step {step:6d} | nll {float(loss):.3f}")
    return model


@torch.no_grad()
def fast_rollout_ade(models, val, device, n_scenes=256,
                     ego_override=None):
    """ADE via track-consistent windows: roll the model from window i0
    and compare with the LOGGED hist of window i0+k (same slots hold
    the same actors only per-window, so restrict to agents that keep
    their slot: verified by act_id equality)."""
    frames = val.frames
    frame_to_i = {int(f): i for i, f in enumerate(frames)}
    ok = [i for i, f in enumerate(frames)
          if all(int(f) + k in frame_to_i for k in (10, 20, ROLL))]
    pick = ok[:: max(1, len(ok) // n_scenes)][:n_scenes]
    ades = {10: [], 20: [], ROLL: []}
    prox_m = prox_l = n_prox = 0
    disagree = []
    follower_dv = []
    for i0 in pick:
        h, pr, cl, eg, li, tg, pm = batch_tensors(val, [i0], device)
        hist = h.clone()
        base_speed = None
        for k in range(1, ROLL + 1):
            f_k = int(frames[i0]) + k
            i_k = frame_to_i.get(f_k)
            eg_k = torch.as_tensor(val.ego[[i_k]], device=device) \
                if i_k is not None else eg
            li_k = torch.as_tensor(val.light[[i_k]], device=device) \
                if i_k is not None else li
            if ego_override is not None:
                eg_k = ego_override(k, eg_k.clone())
            outs = [m.step(hist, pr, cl, eg_k, li_k) for m in models]
            nxt = outs[0]
            if len(outs) > 1:
                stack = torch.stack([o[..., 0:2] for o in outs])
                disagree.append(float(
                    stack.std(dim=0).norm(dim=-1)[0, pm[0]].mean()))
            hist = torch.cat([hist[:, :, 1:], nxt[:, :, None]], dim=2)
            if k in ades and i_k is not None:
                # ID-MATCHED comparison: slot layouts re-sort per
                # frame; matching by slot compares DIFFERENT actors
                # (the first W0 run's 91 m ADE artifact).
                ids0 = val.ids[i0]
                slot_k = {int(v): b
                          for b, v in enumerate(val.ids[i_k]) if v >= 0}
                logged = val.hist[i_k][:, -1, 0:2]
                nxt_np = nxt[0, :, 0:2].cpu().numpy()
                for a in np.where(pm[0].cpu().numpy())[0]:
                    b = slot_k.get(int(ids0[a]))
                    if b is not None:
                        ades[k].append(float(np.linalg.norm(
                            nxt_np[a] - logged[b])))
        # proximity events at final step (model) vs log
        i_R = frame_to_i.get(int(frames[i0]) + ROLL)
        if i_R is not None:
            pos_m = hist[0, :, -1, 0:2][pm[0]]
            pos_l = torch.as_tensor(val.hist[[i_R]][0, :, -1, 0:2],
                                    device=device)[
                torch.as_tensor(val.pres_mask[[i_R]][0],
                                device=device)]
            def prox(p):
                if len(p) < 2:
                    return 0
                d = torch.cdist(p, p)
                d.fill_diagonal_(99.0)
                return int((d < 3.0).any(dim=1).sum())
            prox_m += prox(pos_m)
            prox_l += prox(pos_l)
            n_prox += max(len(pos_m), 1)
        if ego_override is not None:
            # follower speed change probe handled by caller via ades
            pass
    out = {f"ade_{k//10}s": float(np.mean(v)) for k, v in ades.items()
           if v}
    out["prox_rate_model"] = prox_m / max(n_prox, 1)
    out["prox_rate_log"] = prox_l / max(n_prox, 1)
    if disagree:
        out["disagree_p50"] = float(np.percentile(disagree, 50))
        out["disagree_p95"] = float(np.percentile(disagree, 95))
        out["disagree_p99"] = float(np.percentile(disagree, 99))
    return out


@torch.no_grad()
def reactivity_probe(models, val, device, n_scenes=128):
    """Follower speed at +2 s: logged ego vs braking ego."""
    frames = val.frames
    frame_to_i = {int(f): i for i, f in enumerate(frames)}
    picks = []
    for i in range(len(frames)):
        if not all(int(frames[i]) + k in frame_to_i
                   for k in range(1, 21)):
            continue
        # follower: agent within 14 m behind-ish of ego, moving
        eg = val.ego[i]
        cur = val.hist[i, :, -1]
        rel = cur[:, 0:2] - eg[0:2]
        d = np.linalg.norm(rel, axis=1)
        sp = np.linalg.norm(cur[:, 2:4], axis=1)
        cand = (val.pred_mask[i]) & (d < 14) & (sp > 2.0)
        if cand.any():
            picks.append((i, int(np.where(cand)[0][0])))
        if len(picks) >= n_scenes:
            break
    print(f"reactivity probe: {len(picks)} scenes")

    def run(i0, a, override):
        h, pr, cl, eg, li, tg, pm = batch_tensors(val, [i0], device)
        hist = h.clone()
        for k in range(1, 21):
            i_k = frame_to_i.get(int(frames[i0]) + k)
            eg_k = torch.as_tensor(val.ego[[i_k]], device=device)
            li_k = torch.as_tensor(val.light[[i_k]], device=device)
            if override:
                # ego brakes to a stop at its t0 position
                eg0 = torch.as_tensor(val.ego[[i0]], device=device)
                eg_k = eg0.clone()
                eg_k[0, 3] = max(0.0, float(eg0[0, 3]) * (1 - k / 10))
            hist = torch.cat(
                [hist[:, :, 1:],
                 models[0].step(hist, pr, cl, eg_k, li_k)[:, :, None]],
                dim=2)
        v = hist[0, a, -1, 2:4].norm()
        return float(v)

    dvs = []
    for i0, a in picks:
        v_log = run(i0, a, False)
        v_brk = run(i0, a, True)
        dvs.append(v_log - v_brk)
    return {"n": len(picks),
            "follower_slowdown_mean": float(np.mean(dvs)) if dvs else 0,
            "follower_slowdown_p75": float(np.percentile(dvs, 75))
            if dvs else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = args.device
    

    tr = load_windows(Path(args.data) / "b2d_train.npz",
                      out / "windows2_train.npz")
    va = load_windows(Path(args.data) / "b2d_val.npz",
                      out / "windows2_val.npz")

    models = []
    for s in range(args.seeds):
        ck = out / "checkpoints" / f"traffic_s{s}.pt"
        m = TrafficModel(hist=HIST).to(device)
        if ck.exists():
            m.load_state_dict(torch.load(ck, map_location=device))
            print(f"seed {s}: loaded existing ckpt")
        else:
            m = train_seed(tr, s, args.steps, args.batch, device)
            torch.save(m.state_dict(), ck)
        m.eval()
        models.append(m)

    print("=== W0 evaluation ===")
    res = fast_rollout_ade(models, va, device)
    res["reactivity"] = reactivity_probe(models, va, device)
    res["floors"] = {"cv_4s": CV_FLOOR_4S, "ctrv_4s": CTRV_FLOOR_4S,
                     "gate_ade_4s": W0_ADE_GATE}
    ade4 = res.get("ade_4s", float("inf"))
    res["pass_ade"] = bool(ade4 <= W0_ADE_GATE)
    ratio = res["prox_rate_model"] / max(res["prox_rate_log"], 1e-6)
    res["pass_prox"] = bool(ratio <= 2.0)
    res["pass_react"] = bool(
        res["reactivity"]["follower_slowdown_mean"] > 0.3)
    res["W0_PASS"] = bool(res["pass_ade"] and res["pass_prox"]
                          and res["pass_react"])
    (out / "w0_report.json").write_text(json.dumps(res, indent=2))
    lines = ["# W0 gate report", ""] + \
        [f"- {k}: {v}" for k, v in res.items()]
    (out / "w0_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
