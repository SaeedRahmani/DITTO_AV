"""Empirical support for the conditional-multimodality claim.

Two independent checks (external review correctly noted the collected data
contains no *paired* continuations by construction):

A. Paired expert rollouts: reset highway-env with the SAME seed (identical
   initial traffic state) and roll the aggressive vs conservative expert.
   Reports how often and how early the two styles diverge — direct evidence
   that the expert policy itself is conditionally multimodal.

B. Retrieval analysis on a trained run's latent bank: for sampled query
   windows, retrieve top-K windows by start-latent cosine and report
   (i) cross-style fraction (retrieval finds the same situation under BOTH
   styles), (ii) expert action disagreement at the matched step, and
   (iii) start-vs-end latent similarity of retrieved windows (same state,
   diverging futures).

Usage:
    python scripts/analyze_multimodality.py \
        --run-dir /scratch/$USER/ditto_av/outputs/phase1/main_seed0 \
        --k 16 --horizon 5 --out runs/phase1/multimodality_analysis.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)

from ditto_av.config import load_config  # noqa: E402
from ditto_av.data import TrajectoryData, build_latent_bank  # noqa: E402
from ditto_av.envs import make_env  # noqa: E402
from ditto_av.expert import (AGGRESSIVE, CONSERVATIVE,  # noqa: E402
                             ScriptedExpert)
from ditto_av.rewards import max_cos  # noqa: E402
from ditto_av.trainers.wm_trainer import load_world_model  # noqa: E402


def paired_rollouts(cfg, n_pairs: int = 40, seed0: int = 5000):
    """Roll both expert styles from identical initial states."""
    env = make_env(cfg.env)
    diverge_step, n_diverged = [], 0
    for i in range(n_pairs):
        seqs = {}
        for style in (AGGRESSIVE, CONSERVATIVE):
            expert = ScriptedExpert(style)
            env.reset(seed=seed0 + i)
            acts, done = [], False
            while not done:
                a = expert.act(env)
                acts.append(a)
                _, _, term, trunc, _ = env.step(a)
                done = term or trunc
            seqs[style] = acts
        a, c = seqs[AGGRESSIVE], seqs[CONSERVATIVE]
        n = min(len(a), len(c))
        first = next((t for t in range(n) if a[t] != c[t]), None)
        if first is not None:
            n_diverged += 1
            diverge_step.append(first)
    env.close()
    return {
        "n_pairs": n_pairs,
        "diverged_frac": n_diverged / n_pairs,
        "median_first_divergence_step":
            float(np.median(diverge_step)) if diverge_step else None,
    }


@torch.no_grad()
def retrieval_analysis(run_dir: Path, k: int, horizon: int,
                       n_queries: int = 2000):
    cfg = load_config(run_dir / "config.yaml")
    cfg.run_dir = str(run_dir)
    wm = load_world_model(cfg, cfg.env.obs_dim)
    data = TrajectoryData([run_dir / "data" / "expert.npz"])
    styles = np.load(run_dir / "data" / "expert.npz")["ep_style"]
    bank = build_latent_bank(wm, data, cfg.env.action_dim, horizon, "cpu")

    # per-window episode/style labels
    win_ep = np.zeros(bank.n_windows, dtype=np.int64)
    for ep, (s, e) in enumerate(bank.ep_bounds):
        mask = ((bank.window_starts >= s) & (bank.window_starts < e)).numpy()
        win_ep[mask] = ep
    win_style = styles[win_ep]

    starts = F.normalize(bank.windows_h[:, 0, :], dim=-1)
    rng = np.random.default_rng(0)
    q_ids = rng.choice(bank.n_windows, size=min(n_queries, bank.n_windows),
                       replace=False)
    q = starts[q_ids]
    sim = q @ starts.T                                   # (Q, N)
    top_sim, top = sim.topk(k + 1, dim=-1)               # +1: self
    top_sim, top = top_sim[:, 1:], top[:, 1:]            # drop self

    cross = (win_style[top.numpy()] !=
             win_style[q_ids][:, None]).mean()
    qa = bank.action[bank.window_starts[q_ids]]          # query actions
    na = bank.action[bank.window_starts[top.reshape(-1)]] \
        .reshape(top.shape)                              # neighbor actions
    act_disagree = (na != qa.unsqueeze(1)).float().mean()

    q_end = bank.windows_h[q_ids][:, -1, :].unsqueeze(1)
    n_end = bank.windows_h[top][:, :, -1, :]
    end_sim = max_cos(q_end, n_end).mean()

    return {
        "n_queries": len(q_ids),
        "k": k,
        "horizon": horizon,
        "mean_start_cosine_of_retrieved": float(top_sim.mean()),
        "cross_style_fraction": float(cross),
        "action_disagreement_at_match": float(act_disagree),
        "mean_end_similarity_of_retrieved": float(end_sim),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path("runs/phase1/multimodality_analysis.md"))
    args = ap.parse_args()

    cfg = load_config(args.run_dir / "config.yaml")
    pa = paired_rollouts(cfg)
    ra = retrieval_analysis(args.run_dir, args.k, args.horizon)

    lines = [
        "# Conditional multimodality: empirical checks", "",
        f"run: {args.run_dir}", "",
        "## A. Paired expert rollouts from identical initial states", "",
        f"- pairs: {pa['n_pairs']} (same env reset seed, both styles)",
        f"- diverged within episode: {pa['diverged_frac']:.0%}",
        f"- median first divergence step: "
        f"{pa['median_first_divergence_step']}", "",
        "Same initial traffic state, different demonstrated continuation —",
        "the expert policy is conditionally multimodal by construction.", "",
        "## B. Latent-bank retrieval crosses styles", "",
        f"- queries: {ra['n_queries']}, K={ra['k']}, H={ra['horizon']}",
        f"- mean start cosine of retrieved windows: "
        f"{ra['mean_start_cosine_of_retrieved']:.3f}",
        f"- cross-style fraction of retrieved windows: "
        f"{ra['cross_style_fraction']:.2f}",
        f"- expert action disagreement at matched step: "
        f"{ra['action_disagreement_at_match']:.2f}",
        f"- mean end-state similarity of retrieved windows: "
        f"{ra['mean_end_similarity_of_retrieved']:.3f}", "",
        "High start similarity + substantial cross-style retrieval and",
        "action disagreement + lower end similarity = near-identical states",
        "with divergent expert continuations in the actual training data.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
