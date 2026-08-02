#!/usr/bin/env python3
"""Generate the four paper figures from banked artifacts (paper v0.1).

Reads only committed evidence (runs/); writes paper/figures/*.{pdf,png}.
Rerunnable end to end — no hand-entered numbers except the champion
dev-10 reference (30.49, runs/carla_smoke/gen3_wph_era d10_wphr_*),
which is also recomputed here from its jsons.

Design: colorblind-safe fixed categorical assignment (multi=blue,
single=orange, BC=aqua, references=gray dashed), thin marks, one axis
per panel, recessive grid, direct labels where they earn their place.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_MULTI = "#2a78d6"   # slot 1 blue  — DITTO-multi (ours)
C_SINGLE = "#eb6834"  # slot 2 orange — DITTO-single
C_BC = "#1baf7a"      # slot 3 aqua  — latent BC
C_REF = "#52514e"     # reference lines / expert
C_MUTED = "#9a9891"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf/.png")


def seed_stats(block, cond, policy, field="return_mean"):
    vals = [block[s][cond][policy][field] for s in block]
    return float(np.mean(vals)), float(np.std(vals))


# ---------- Figure 1: highway main result (k16h5, 3 seeds) ----------
def fig1(d):
    blk = d["k16h5"]
    conds = [("in_distribution", "in-distribution"), ("shifted", "shifted")]
    pols = [("bc", "BC", C_BC), ("ditto_single", "DITTO-single", C_SINGLE),
            ("ditto_multi", "DITTO-multi", C_MULTI)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, field, label in ((axes[0], "return_mean", "closed-loop return"),
                             (axes[1], "collision_rate", "collision rate")):
        x = np.arange(len(conds))
        w = 0.24
        for i, (p, pl, c) in enumerate(pols):
            m = [seed_stats(blk, cn, p, field)[0] for cn, _ in conds]
            s = [seed_stats(blk, cn, p, field)[1] for cn, _ in conds]
            ax.bar(x + (i - 1) * w, m, w * 0.92, yerr=s, capsize=2,
                   color=c, label=pl, error_kw={"lw": 0.8})
        for j, (cn, _) in enumerate(conds):
            e = seed_stats(blk, cn, "expert", field)[0]
            ax.hlines(e, j - 0.42, j + 0.42, color=C_REF, ls="--", lw=1.0)
            if j == 0:
                va = "bottom" if field == "collision_rate" else "center"
                ax.annotate("expert", (j + 0.44, e), fontsize=7.5,
                            color=C_REF, va=va)
        ax.set_xticks(x, [c for _, c in conds])
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Highway (two-style expert): mean ± std over 3 seeds",
                 fontsize=9, y=1.02)
    savefig(fig, "fig1_highway")


# ---------- Figure 2: the gap appears only under multimodality ----------
def fig2(d):
    def gaps(block):
        return [block[s]["shifted"]["ditto_multi"]["return_mean"]
                - block[s]["shifted"]["ditto_single"]["return_mean"]
                for s in block]
    g5050 = gaps(d["main"])
    g2575 = [d["ablations"]["style25"]["shifted"]["ditto_multi"]
             ["return_mean"]
             - d["ablations"]["style25"]["shifted"]["ditto_single"]
             ["return_mean"]]
    guni = gaps(d["uni"])
    rows = [("50 / 50\n(3 seeds)", g5050), ("25 / 75\n(1 seed)", g2575),
            ("100 / 0\nunimodal (3 seeds)", guni)]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(rows))
    m = [np.mean(g) for _, g in rows]
    s = [np.std(g) for _, g in rows]
    ax.bar(x, m, 0.5, yerr=s, capsize=3, color=C_MULTI,
           error_kw={"lw": 0.8})
    ax.axhline(0, color=C_REF, lw=1.0)
    ax.set_xticks(x, [r for r, _ in rows], fontsize=8)
    ax.set_xlabel("expert style ratio (aggressive / conservative)")
    ax.set_ylabel("multi − single shifted return")
    ax.set_title("Multimodal advantage requires multimodal data",
                 fontsize=9)
    savefig(fig, "fig2_multimodality")


# ---------- Figure 3: selector inversion ----------
def fig3():
    d = json.load(open(REPO / "runs/phase2_selector/summary.json"))
    fam = {"ditto_multi": (C_MULTI, "o", "DITTO-multi"),
           "ditto_single": (C_SINGLE, "s", "DITTO-single"),
           "bc": (C_BC, "^", "BC")}
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    seen = set()
    for r in d["rows"]:
        if r["family"] != "control" or r["latent_match"] is None:
            continue
        c, mk, lbl = fam[r["policy"]]
        ax.scatter(r["latent_match"], r["cl3_completion"], s=26, color=c,
                   marker=mk, label=lbl if lbl not in seen else None,
                   zorder=3)
        seen.add(lbl)
    rho = d["spearman"]["cl3_completion"]["latent_match"]["rho"]
    ax.set_xlabel("in-model on-policy latent match")
    ax.set_ylabel("closed-loop route completion (%)")
    ax.set_title(f"Higher latent match, worse driving "
                 f"(Spearman {rho:+.2f})", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    savefig(fig, "fig3_selector_inversion")


# ---------- Figure 4: gen-4 dose-response ----------
def fig4():
    def dev10(tag_a, tag_b, base=REPO / "runs/carla_smoke/gen4_dwp"):
        recs = []
        for t in (tag_a, tag_b):
            recs += json.load(open(base / f"carla_results_{t}.json"))[
                "_checkpoint"]["records"]
        return float(np.mean([r["scores"]["score_composed"]
                              for r in recs]))
    champ_base = REPO / "runs/carla_smoke/gen3_wph_era"
    champ = dev10("d10_wphr_A", "d10_wphr_B", champ_base)
    rows = [
        ("anchor 0.1 + divergent starts", dev10("d10_dwp_A", "d10_dwp_B")),
        ("anchor 0.3 + divergent starts", dev10("d10_k03_A", "d10_k03_B")),
        ("anchor 0.3", dev10("d10_k03nd_A", "d10_k03nd_B")),
        ("anchor 0.3, early stop", dev10("d10_es_A", "d10_es_B")),
        ("anchor 1.0", dev10("d10_k10_A", "d10_k10_B")),
    ]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [v for _, v in rows], 0.55, color=C_MULTI)
    for yi, (_, v) in zip(y, rows):
        ax.annotate(f"{v:.1f}", (v + 0.4, yi), va="center", fontsize=7.5,
                    color="#0b0b0b")
    ax.axvline(champ, color=C_REF, ls="--", lw=1.2)
    ax.annotate(f"BC (no imagination): {champ:.1f}",
                (champ - 0.4, len(rows) - 0.55), fontsize=7.5,
                color=C_REF, ha="right")
    ax.set_yticks(y, [r for r, _ in rows], fontsize=8)
    ax.set_xlabel("dev-10 driving score")
    ax.set_title("Imagination refinement of the waypoint head: "
                 "every dose loses to BC", fontsize=9)
    ax.grid(axis="y", visible=False)
    savefig(fig, "fig4_dose_response")


def main():
    d = json.load(open(REPO / "runs/phase1/phase1_results.json"))
    fig1(d)
    fig2(d)
    fig3()
    fig4()


if __name__ == "__main__":
    main()
