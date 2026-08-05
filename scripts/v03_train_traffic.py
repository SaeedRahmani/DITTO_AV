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
                        target=sw.target, ids=sw.ids,
                        ego_hist=sw.ego_hist)
    return sw


def batch_tensors(sw, idx, device):
    return (torch.as_tensor(sw.hist[idx], device=device),
            torch.as_tensor(sw.pres_mask[idx], device=device),
            torch.as_tensor(sw.cls[idx], device=device).long(),
            torch.as_tensor(sw.ego[idx], device=device),
            torch.as_tensor(sw.light[idx], device=device),
            torch.as_tensor(sw.target[idx], device=device),
            torch.as_tensor(sw.pred_mask[idx], device=device),
            torch.as_tensor(sw.ego_hist[idx], device=device))


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
        h, pr, cl, eg, li, tg, pm, eh = batch_tensors(sw, idx, device)
        loss = model.loss(h, pr, cl, eg, li, tg, pm.float(), eh)
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

def build_chains(sw, K=15, max_chains=30000, stride=2):
    """Rollout-training chains: for scenes with K consecutive windows,
    ID-MATCHED logged positions per step (+ ego/light sequences).
    Returns dict of arrays over M chains."""
    frames = sw.frames
    frame_to_i = {int(f): i for i, f in enumerate(frames)}
    A = sw.ids.shape[1]
    idx0, TGT, VAL, EGO, LIG = [], [], [], [], []
    for i0 in range(0, len(frames), stride):
        if not all(int(frames[i0]) + k in frame_to_i
                   for k in range(1, K + 1)):
            continue
        tgt = np.full((K, A, 2), np.nan, dtype=np.float32)
        val = np.zeros((K, A), dtype=bool)
        eg = np.zeros((K, 4), dtype=np.float32)
        li = np.zeros((K, 4), dtype=np.float32)
        ids0 = sw.ids[i0]
        for k in range(1, K + 1):
            i_k = frame_to_i[int(frames[i0]) + k]
            eg[k - 1] = sw.ego[i_k]
            li[k - 1] = sw.light[i_k]
            slot_k = {int(v): b for b, v in enumerate(sw.ids[i_k])
                      if v >= 0}
            for a in np.where(sw.pred_mask[i0])[0]:
                b = slot_k.get(int(ids0[a]))
                if b is not None:
                    tgt[k - 1, a] = sw.hist[i_k][b, -1, 0:2]
                    val[k - 1, a] = True
        idx0.append(i0)
        TGT.append(tgt)
        VAL.append(val)
        EGO.append(eg)
        LIG.append(li)
        if len(idx0) >= max_chains:
            break
    print(f"chains: {len(idx0)} (K={K})")
    return {"i0": np.array(idx0), "tgt": np.stack(TGT),
            "val": np.stack(VAL), "ego": np.stack(EGO),
            "lig": np.stack(LIG)}


def rollout_finetune(model, sw, ch, steps, batch, device, K=15,
                     lr=1e-4, seed=0):
    """Closed-loop fine-tuning: differentiable K-step unroll, Huber on
    ID-matched logged positions — attacks the teacher-forced/rollout
    gap that failed W0 v1 (ADE@4s 7.62 vs floor 4.75)."""
    rng = np.random.default_rng(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    M = len(ch["i0"])
    for step in range(1, steps + 1):
        pick = rng.integers(M, size=batch)
        i0 = ch["i0"][pick]
        h, pr, cl, eg, li, tg, pm, eh = batch_tensors(sw, i0, device)
        hist = h
        tgt = torch.as_tensor(ch["tgt"][pick], device=device)
        val = torch.as_tensor(ch["val"][pick], device=device)
        egs = torch.as_tensor(ch["ego"][pick], device=device)
        lis = torch.as_tensor(ch["lig"][pick], device=device)
        loss = 0.0
        n = 0
        ehb = eh.clone()
        for k in range(K):
            ehb = torch.cat([ehb[:, 1:], egs[:, k][:, None]], dim=1)
            d = model.dist(hist, pr, cl, egs[:, k], lis[:, k],
                           ego_hist=ehb)
            nxt = model.advance(hist[:, :, -1], d.base_dist.loc)
            hist = torch.cat([hist[:, :, 1:], nxt[:, :, None]], dim=2)
            m = val[:, k] & pm
            if m.any():
                err = nxt[..., 0:2] - tgt[:, k]
                loss = loss + torch.nn.functional.huber_loss(
                    err[m], torch.zeros_like(err[m]), delta=2.0)
                n += 1
        loss = loss / max(n, 1)
        # JOINT objective: keep one-step teacher-forced NLL in the mix —
        # pure rollout fine-tuning collapsed ego-reactivity in round-2
        # (0.21 -> 0.04 m/s): with the logged ego always paired with
        # logged futures, ignoring the ego minimizes rollout loss.
        ti = rng.integers(len(sw.frames), size=batch)
        (th_, tpr, tcl, teg, tli, ttg, tpm,
         teh) = batch_tensors(sw, ti, device)
        loss = loss + model.loss(th_, tpr, tcl, teg, tli, ttg,
                                 tpm.float(), teh)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"  rf seed {seed} step {step:5d} | loss "
                  f"{float(loss):.4f}")
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
    min_ades = []   # per-agent min over fixed-mode rollouts @4s
    # error decomposition @4s by log-behavior regime (lever picker):
    #   stopped (<0.5 m/s), launching (stopped->moving), cruising
    #   (straight), turning (|yaw rate| > 0.05 rad/frame)
    regime_err = {"stopped": [], "launching": [], "cruising": [],
                  "turning": []}
    prox_m = prox_l = n_prox = 0
    disagree = []
    follower_dv = []
    for i0 in pick:
        h, pr, cl, eg, li, tg, pm, eh = batch_tensors(val, [i0], device)
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
            eh_k = torch.as_tensor(val.ego_hist[[i_k]], device=device) \
                if i_k is not None else eh
            outs = [m.step(hist, pr, cl, eg_k, li_k, ego_hist=eh_k)
                    for m in models]
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
                        e = float(np.linalg.norm(nxt_np[a] - logged[b]))
                        ades[k].append(e)
                        if k == ROLL:
                            h0 = val.hist[i0][a]
                            v0 = float(np.linalg.norm(h0[-1, 2:4]))
                            vk = float(np.linalg.norm(
                                val.hist[i_k][b, -1, 2:4]))
                            dy = abs(float(h0[-1, 4] - h0[0, 4]))
                            if v0 < 0.5 and vk < 0.5:
                                regime_err["stopped"].append(e)
                            elif v0 < 0.5 <= vk:
                                regime_err["launching"].append(e)
                            elif dy > 0.5:
                                regime_err["turning"].append(e)
                            else:
                                regime_err["cruising"].append(e)
        # minADE over fixed-mode rollouts (pre-registered refinement,
        # justified by the regime decomposition: futures branch)
        n_modes = getattr(models[0], "n_modes", 1)
        i_R = frame_to_i.get(int(frames[i0]) + ROLL)
        if n_modes > 1 and i_R is not None:
            per_mode = []
            for mm in range(n_modes):
                hh = h.clone()
                for k in range(1, ROLL + 1):
                    i_k2 = frame_to_i.get(int(frames[i0]) + k)
                    eg_k2 = torch.as_tensor(val.ego[[i_k2]],
                                            device=device) \
                        if i_k2 is not None else eg
                    li_k2 = torch.as_tensor(val.light[[i_k2]],
                                            device=device) \
                        if i_k2 is not None else li
                    eh_k2 = torch.as_tensor(
                        val.ego_hist[[i_k2]], device=device) \
                        if i_k2 is not None else eh
                    nn_ = models[0].step_mode(hh, pr, cl, eg_k2, li_k2,
                                              mm, ego_hist=eh_k2)
                    hh = torch.cat([hh[:, :, 1:], nn_[:, :, None]],
                                   dim=2)
                ids0m = val.ids[i0]
                slot_R = {int(v): b for b, v in
                          enumerate(val.ids[i_R]) if v >= 0}
                fin = hh[0, :, -1, 0:2].cpu().numpy()
                errs_m = {}
                for a in np.where(pm[0].cpu().numpy())[0]:
                    b = slot_R.get(int(ids0m[a]))
                    if b is not None:
                        errs_m[a] = float(np.linalg.norm(
                            fin[a] - val.hist[i_R][b, -1, 0:2]))
                per_mode.append(errs_m)
            agents = set().union(*[set(d) for d in per_mode]) \
                if per_mode else set()
            for a in agents:
                vals = [d[a] for d in per_mode if a in d]
                if vals:
                    min_ades.append(min(vals))

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
    if min_ades:
        out["min_ade_4s_modes"] = float(np.mean(min_ades))
    out["regimes_4s"] = {
        r: {"n": len(v), "ade": float(np.mean(v)) if v else None}
        for r, v in regime_err.items()}
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
        h, pr, cl, eg, li, tg, pm, eh = batch_tensors(val, [i0], device)
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
    ap.add_argument("--rollout-steps", type=int, default=4000)
    ap.add_argument("--rollout-k", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = args.device
    

    tr = load_windows(Path(args.data) / "b2d_train.npz",
                      out / "windows3_train.npz")
    va = load_windows(Path(args.data) / "b2d_val.npz",
                      out / "windows3_val.npz")

    chains = None
    models = []
    for s in range(args.seeds):
        ck = out / "checkpoints" / f"traffic_s{s}.pt"
        ck_rf = out / "checkpoints" / f"traffic_s{s}_rf.pt"
        m = TrafficModel(hist=HIST).to(device)
        if ck_rf.exists():
            m.load_state_dict(torch.load(ck_rf, map_location=device))
            print(f"seed {s}: loaded fine-tuned ckpt")
        else:
            if ck.exists():
                m.load_state_dict(torch.load(ck, map_location=device))
                print(f"seed {s}: loaded teacher-forced ckpt")
            else:
                m = train_seed(tr, s, args.steps, args.batch, device)
                torch.save(m.state_dict(), ck)
            if args.rollout_steps > 0:
                if chains is None:
                    chains = build_chains(tr, K=args.rollout_k)
                m.train()
                m = rollout_finetune(m, tr, chains, args.rollout_steps,
                                     max(8, args.batch // 3), device,
                                     K=args.rollout_k, seed=s)
                torch.save(m.state_dict(), ck_rf)
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
