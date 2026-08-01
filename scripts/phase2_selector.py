#!/usr/bin/env python3
"""Phase-2: can in-model (offline) metrics select closed-loop drivers?

DITTO's claim: on-policy latent divergence in the world model predicts
true closed-loop return where action-MAE does not. Every training run
already banked its in-model metrics (results/results.json: on-policy
imagined-rollout latent match vs held-out expert windows, the
expert-replay ceiling, action MAE/NLL, H-step obs MSE) — computed at
train time with that run's era-correct data. This script JOINS those
with the banked closed-loop results (3-route 3x3 and dev-10) and
reports Spearman rank correlations per metric.

Scope: the CONTROL-action family (one action/label space, so MAE is
comparable across rows). The wp-action model is reported as a separate
row; wp-head models have no in-model rollout metric by construction
(a 12-dim head cannot act in the 3-dim dream) and are excluded.

Usage: python scripts/phase2_selector.py  (login node, seconds)
Writes runs/phase2_selector/{table.md,summary.json}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = [REPO / "runs/carla_smoke", REPO / "runs/carla_smoke/gen3_wph_era",
            Path("/scratch") / os.environ.get("USER", "srahmani")
            / "ditto_av/outputs"]
CKPT_BASE = Path.home() / "ditto_out"

# label, run dir, policy, closed-loop 3x3 file candidates,
# dev-10 file pair (or None), family
REGISTRY = [
    ("kl01 multi",      "b2d_v3kl01",      "ditto_multi",
     ["kl01_3x3.json"], ["d10_kl01_A.json", "d10_kl01_B.json"], "control"),
    ("kl015 multi",     "b2d_kl015",       "ditto_multi",
     ["kl015_3x3.json"], None, "control"),
    ("kl02 multi",      "b2d_kl02",        "ditto_multi",
     ["kl02_3x3.json"], None, "control"),
    ("v5kl01 multi",    "b2d_v5kl01",      "ditto_multi",
     ["v5kl01_3x3.json"], None, "control"),
    ("kl01k16 multi",   "b2d_kl01k16",     "ditto_multi",
     ["kl01k16_3x3.json"], None, "control"),
    ("kl01_5x multi",   "b2d_kl01_5x",     "ditto_multi",
     ["kl01_5x_3x3.json"],
     ["carla_results_d10g2_kl01_5x_A.json",
      "carla_results_d10g2_kl01_5x_B.json"], "control"),
    ("kl01_20x multi",  "b2d_kl01_20x",    "ditto_multi",
     ["kl01_20x_3x3.json"], None, "control"),
    ("gen2 multi",      "b2d_gen2",        "ditto_multi",
     ["gen2_3x3.json"], None, "control"),
    ("gen2_20x multi",  "b2d_gen2_20x",    "ditto_multi",
     ["gen2_20x_3x3.json"], None, "control"),
    ("gen2_10x multi",  "b2d_gen2_10x",    "ditto_multi",
     ["carla_results_gen2_10x_3x3.json"],
     ["carla_results_d10g2_gen2_10x_A.json",
      "carla_results_d10g2_gen2_10x_B.json"], "control"),
    ("gen2_10x single", "b2d_gen2_10x",    "ditto_single",
     ["carla_results_gen2_10x_single_3x3.json"], None, "control"),
    ("gen2_10x BC",     "b2d_gen2_10x",    "bc",
     ["carla_results_gen2_10x_bc_3x3.json"],
     ["carla_results_d10_bc_A.json", "carla_results_d10_bc_B.json"],
     "control"),
    ("gen2_10x_s1 BC",  "b2d_gen2_10x_s1", "bc",
     ["carla_results_gen2_10x_s1_bc_3x3.json"], None, "control"),
    ("gen2_10x_s2 BC",  "b2d_gen2_10x_s2", "bc",
     ["carla_results_gen2_10x_s2_bc_3x3.json"], None, "control"),
    ("gen2_10x_s1 multi", "b2d_gen2_10x_s1", "ditto_multi",
     ["carla_results_gen2_10x_s1_3x3.json"], None, "control"),
    ("gen2_10x_s2 multi", "b2d_gen2_10x_s2", "ditto_multi",
     ["carla_results_gen2_10x_s2_3x3.json"], None, "control"),
    ("gen3_clean BC",   "b2d_gen3_clean",  "bc",
     ["carla_results_gen3_clean_bc_3x3.json"],
     ["carla_results_d10_g3c_A.json", "carla_results_d10_g3c_B.json"],
     "control"),
    ("gen3_wp BC",      "b2d_gen3_wp",     "bc",
     ["carla_results_gen3_wp_3x3.json"],
     ["carla_results_d10_g3wp_A.json", "carla_results_d10_g3wp_B.json"],
     "wp-action"),
    ("gen3_wp multi",   "b2d_gen3_wp",     "ditto_multi",
     ["carla_results_g3wp_multi_3x3.json"], None, "wp-action"),
]


def find(fname):
    for d in OUT_DIRS:
        p = d / fname
        if p.is_file():
            return p
    return None


def closed_loop(files):
    recs = []
    for f in files:
        p = find(f)
        if p is None:
            return None
        d = json.load(open(p))
        recs += d["_checkpoint"]["records"]
    if not recs:
        return None
    return {
        "completion": float(np.mean([r["scores"]["score_route"]
                                     for r in recs])),
        "score": float(np.mean([r["scores"]["score_composed"]
                                for r in recs])),
        "n": len(recs),
    }


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    r = np.corrcoef(rx, ry)[0, 1]
    return float(r), len(x)


def main():
    rows = []
    for label, run, policy, cl3, cl10, family in REGISTRY:
        res_f = CKPT_BASE / run / "results" / "results.json"
        if not res_f.is_file():
            print(f"skip {label}: no {res_f}")
            continue
        res = json.load(open(res_f))
        pol = res.get("policies", {}).get(policy)
        if pol is None:
            print(f"skip {label}: no policy entry {policy}")
            continue
        ceiling = res.get("expert_replay", {}).get("latent_match")
        cl = closed_loop(cl3)
        if cl is None:
            print(f"skip {label}: closed-loop files missing {cl3}")
            continue
        d10 = closed_loop(cl10) if cl10 else None
        rows.append({
            "label": label, "family": family, "policy": policy,
            "latent_match": pol["latent_match"],
            "divergence": (None if ceiling is None
                           or not np.isfinite(pol["latent_match"])
                           else ceiling - pol["latent_match"]),
            "hstep_obs_mse": pol["hstep_obs_mse"],
            "action_mae": pol["action_mae"],
            "action_nll": pol["action_nll"],
            "cl3_completion": cl["completion"], "cl3_score": cl["score"],
            "cl3_n": cl["n"],
            "d10_completion": None if d10 is None else d10["completion"],
            "d10_score": None if d10 is None else d10["score"],
        })

    ctl = [r for r in rows if r["family"] == "control"]
    metrics = ["latent_match", "divergence", "hstep_obs_mse",
               "action_mae", "action_nll"]
    corr = {}
    for target in ("cl3_completion", "cl3_score", "d10_completion"):
        y = [r[target] if r[target] is not None else np.nan for r in ctl]
        corr[target] = {}
        for m in metrics:
            x = [r[m] if r[m] is not None else np.nan for r in ctl]
            rho, n = spearman(x, y)
            corr[target][m] = {"rho": rho, "n": n}
    # within-objective slices: rules out "the metric merely re-discovers
    # BC>multi" — the inversion must hold among same-objective models to
    # be a property of the metric itself
    within = {}
    for fam, sel in (("multi_only",
                      [r for r in ctl if r["policy"] == "ditto_multi"]),
                     ("bc_only", [r for r in ctl if r["policy"] == "bc"])):
        y = [r["cl3_completion"] for r in sel]
        within[fam] = {}
        for m in metrics:
            rho, n = spearman([r[m] for r in sel], y)
            within[fam][m] = {"rho": rho, "n": n}

    out = REPO / "runs/phase2_selector"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "spearman": corr, "within_objective": within},
              open(out / "summary.json", "w"), indent=1)

    lines = ["# Phase-2: in-model metrics vs closed-loop (selector study)",
             "",
             f"{len(ctl)} control-family (run, policy) pairs; ground truth "
             "= banked 3-route 3x3 (all) and dev-10 (subset).", "",
             "| model | latent match | divergence | H-step MSE | MAE | NLL "
             "| 3x3 compl | 3x3 score | dev-10 compl |", "|" + "---|" * 9]
    for r in sorted(rows, key=lambda r: -r["cl3_completion"]):
        f = lambda v, p=4: "-" if v is None or not np.isfinite(v) \
            else f"{v:.{p}f}"
        lines.append(
            f"| {r['label']} ({r['family']}) | {f(r['latent_match'])} "
            f"| {f(r['divergence'])} | {f(r['hstep_obs_mse'], 5)} "
            f"| {f(r['action_mae'])} | {f(r['action_nll'], 2)} "
            f"| {r['cl3_completion']:.1f} | {r['cl3_score']:.2f} "
            f"| {f(r['d10_completion'], 1)} |")
    lines += ["", "## Spearman rank correlations (control family)", "",
              "| metric | vs 3x3 completion | vs 3x3 score "
              "| vs dev-10 completion |", "|---|---|---|---|"]
    for m in metrics:
        cells = []
        for t in ("cl3_completion", "cl3_score", "d10_completion"):
            c = corr[t][m]
            cells.append("-" if not np.isfinite(c["rho"])
                         else f"{c['rho']:+.2f} (n={c['n']})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Within-objective slices (vs 3x3 completion)", "",
              "| metric | multi-only | bc-only |", "|---|---|---|"]
    for m in metrics:
        cells = []
        for fam in ("multi_only", "bc_only"):
            c = within[fam][m]
            cells.append("-" if not np.isfinite(c["rho"])
                         else f"{c['rho']:+.2f} (n={c['n']})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/table.md and summary.json")


if __name__ == "__main__":
    main()
