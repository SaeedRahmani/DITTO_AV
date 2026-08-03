#!/usr/bin/env python3
"""BEV visualizations of v0.2 policies driving recorded scenes (egosim).

For each selected val scene, rolls the given TokenPolicy checkpoints
closed-loop in the log-replay world and renders:
  - a trajectory plate (PNG): expert path + each policy's path, traffic
    snapshots, collision markers;
  - an animation (GIF): replayed traffic boxes + policy ego boxes with
    trails, expert ghost;
  - a divergent-start recovery demo (GIF): perturbed pose, does the
    policy return to the expert path?

Scenes are auto-picked from the val log: dense+moving, launch, turning.
Pure matplotlib/Pillow; runs on a login node in minutes.

Usage:
  python scripts/visualize_egosim.py --out runs/viz_v02 \
      --npz ~/ditto_out/b2d_v02_999t/data/b2d_val.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

torch.set_num_threads(4)

from ditto_av.config import load_config  # noqa: E402
from ditto_av.egosim import EgoSim, GlobalLog, RewardParams, SimParams  # noqa: E402
from ditto_av.models.policy_v2 import make_token_policy  # noqa: E402

HOME_OUT = Path.home() / "ditto_out"

# label, run-config (cpu), checkpoint, color — the current leaders
MODELS = [
    ("BC @297 (dev10 74.1)", HOME_OUT / "b2d_v02_cpu.yaml",
     HOME_OUT / "b2d_v02/checkpoints/clp_bc.pt", "#d62728"),
    ("shaped-RL @297 (68.4)", HOME_OUT / "b2d_v02_shaped_cpu.yaml",
     HOME_OUT / "b2d_v02_shaped/checkpoints/clp_rl.pt", "#1f77b4"),
    ("tight-RL @999 (d10 pending)", HOME_OUT / "b2d_v02_999t_cpu.yaml",
     HOME_OUT / "b2d_v02_999t/checkpoints/clp_rl.pt", "#2ca02c"),
]
EXPERT_C = "black"
H = 80          # 8 s rollouts
BURN = 8


def load_models(device="cpu"):
    out = []
    for label, cfgp, ckpt, color in MODELS:
        if not Path(ckpt).exists():
            print(f"skip {label}: no ckpt")
            continue
        cfg = load_config(str(cfgp))
        pol = make_token_policy(cfg)
        pol.load_state_dict(torch.load(ckpt, map_location=device))
        pol.eval()
        out.append((label, pol, color))
    return out


@torch.no_grad()
def rollout(sim: EgoSim, log: GlobalLog, pol, start: int,
            perturb=None, seed=0):
    """Single-scene rollout; returns dict of per-step arrays."""
    starts = torch.tensor([start])
    h = None
    for i in range(BURN, 0, -1):
        emb = pol.encode(log.obs[starts - i])
        h = pol.step(emb, h)
    rng = torch.Generator().manual_seed(seed) if perturb else None
    xy, th, v = sim.reset(starts, perturb, rng)
    frame = starts.clone()
    traj = {"xy": [xy[0].numpy().copy()], "th": [float(th[0])],
            "v": [float(v[0])], "col": [False]}
    for _ in range(H):
        obs = sim.build_obs(frame, xy, th, v)
        emb = pol.encode(obs)
        h = pol.step(emb, h)
        a = pol.act(pol.features(emb, h))
        xy, th, v = sim.step_ego(a, xy, th, v)
        frame = frame + 1
        traj["xy"].append(xy[0].numpy().copy())
        traj["th"].append(float(th[0]))
        traj["v"].append(float(v[0]))
        traj["col"].append(bool(sim.collisions(frame, xy, th)[0]))
    traj["xy"] = np.stack(traj["xy"])
    return traj


def pick_scenes(log: GlobalLog, n=4):
    """Windows ranked by interest: traffic density x expert motion x
    heading change; at most one per episode."""
    scored = []
    for s, e in log.episodes:
        if e - s < H + BURN + 10:
            continue
        for t0 in range(s + BURN, e - H - 6, 20):
            ego = log.ego[t0:t0 + H]
            dens = float((log.act[t0, :, 0] > 0.5).sum())
            move = float((ego[-1, :2] - ego[0, :2]).norm())
            turn = float((ego[:, 2].diff().abs().sum()))
            launch = 2.0 if ego[0, 3] < 1.0 and move > 5 else 1.0
            if move < 3:
                continue
            scored.append((dens * min(turn + 0.2, 2.0) * launch,
                           t0, s))
    scored.sort(reverse=True)
    picked, eps = [], set()
    for _, t0, ep in scored:
        if ep in eps:
            continue
        eps.add(ep)
        picked.append(t0)
        if len(picked) >= n:
            break
    return picked


def _boxes_at(log, f):
    act = log.act[f]
    m = act[:, 0] > 0.5
    return act[m, 1:3].numpy(), act[m, 5].numpy(), act[m, 6:8].numpy()


def _draw_box(ax, x, y, yaw, ex, ey, color, alpha=1.0, lw=1.2,
              fill=False):
    r = Rectangle((-ex, -ey), 2 * ex, 2 * ey, edgecolor=color,
                  facecolor=color if fill else "none", alpha=alpha,
                  lw=lw, zorder=5)
    r.set_transform(Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    ax.add_patch(r)


def _frame_view(ax, log, start, trajs):
    ego = log.ego[start:start + H + 1, :2].numpy()
    allxy = [ego] + [t["xy"] for _, t, _ in trajs]
    a = np.concatenate(allxy)
    cx, cy = a.mean(0)
    r = max(28.0, float(np.abs(a - [cx, cy]).max()) + 12)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy + r, cy - r)   # CARLA y-down -> flip for intuitive view
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plate(log, sim, models, start, out_png, title):
    trajs = [(lab, rollout(sim, log, pol, start), c)
             for lab, pol, c in models]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _frame_view(ax, log, start, trajs)
    # traffic snapshots (start, mid, end) fading in
    for k, alpha in ((0, 0.15), (H // 2, 0.3), (H, 0.5)):
        pos, yaw, ext = _boxes_at(log, start + k)
        for p, yw, ex in zip(pos, yaw, ext):
            _draw_box(ax, p[0], p[1], yw, ex[0], ex[1], "gray",
                      alpha=alpha, fill=True)
    exp = log.ego[start:start + H + 1, :2].numpy()
    ax.plot(exp[:, 0], exp[:, 1], "--", color=EXPERT_C, lw=2,
            label="expert (log)", zorder=6)
    for lab, t, c in trajs:
        ax.plot(t["xy"][:, 0], t["xy"][:, 1], "-", color=c, lw=2,
                label=lab, zorder=7)
        cols = np.where(t["col"])[0]
        if len(cols):
            ax.plot(t["xy"][cols, 0], t["xy"][cols, 1], "x", color=c,
                    ms=9, mew=2.5, zorder=8)
    ax.plot(exp[0, 0], exp[0, 1], "o", color=EXPERT_C, ms=8, zorder=8)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return trajs


def animate(log, sim, models, start, out_gif, title, perturb=None):
    trajs = [(lab, rollout(sim, log, pol, start, perturb=perturb), c)
             for lab, pol, c in models]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    def draw(k):
        ax.clear()
        _frame_view(ax, log, start, trajs)
        pos, yaw, ext = _boxes_at(log, start + k)
        for p, yw, ex in zip(pos, yaw, ext):
            _draw_box(ax, p[0], p[1], yw, ex[0], ex[1], "dimgray",
                      alpha=0.6, fill=True)
        exp = log.ego[start:start + H + 1]
        ax.plot(exp[:, 0], exp[:, 1], "--", color=EXPERT_C, lw=1.5,
                alpha=0.7)
        e = exp[k]
        _draw_box(ax, float(e[0]), float(e[1]),
                  float(e[2]) - np.pi / 2, float(e[4]), float(e[5]),
                  EXPERT_C, alpha=0.9, lw=1.8)
        for lab, t, c in trajs:
            ax.plot(t["xy"][:k + 1, 0], t["xy"][:k + 1, 1], "-",
                    color=c, lw=1.8, alpha=0.9)
            _draw_box(ax, t["xy"][k, 0], t["xy"][k, 1],
                      t["th"][k] - np.pi / 2, 2.45, 1.06, c, lw=2.0)
            if t["col"][k]:
                ax.plot(t["xy"][k, 0], t["xy"][k, 1], "x", color="red",
                        ms=14, mew=3, zorder=9)
        hands = [plt.Line2D([], [], color=EXPERT_C, ls="--",
                            label="expert (log)")]
        hands += [plt.Line2D([], [], color=c, label=lab)
                  for lab, _, c in trajs]
        ax.legend(handles=hands, loc="upper left", fontsize=8,
                  framealpha=0.9)
        ax.set_title(f"{title}   t={k / 10:.1f}s", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=range(0, H + 1, 2))
    anim.save(out_gif, writer=PillowWriter(fps=6))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(
        HOME_OUT / "b2d_v02_999t/data/b2d_val.npz"))
    ap.add_argument("--out", default="runs/viz_v02")
    ap.add_argument("--scenes", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log = GlobalLog([args.npz], device="cpu")
    sim = EgoSim(log, SimParams(), RewardParams())
    models = load_models()
    starts = pick_scenes(log, n=args.scenes)
    print(f"scenes: {starts}")

    for i, t0 in enumerate(starts):
        print(f"scene {i} (frame {t0}): plate + gif ...")
        plate(log, sim, models, t0, out / f"scene{i}_plate.png",
              f"held-out scene {i}: 8 s closed-loop in replayed traffic")
        animate(log, sim, models, t0, out / f"scene{i}_anim.gif",
                f"scene {i}")

    # divergent-start recovery demo on the most dynamic scene
    print("recovery demo ...")
    perturb = {"frac": 1.0, "lat_sigma": 1.5, "yaw_sigma": 0.15,
               "v_sigma": 0.0}
    animate(log, sim, models, starts[0], out / "recovery_demo.gif",
            "divergent start (+lateral/yaw perturbation): recovery",
            perturb=perturb)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
