# BC-anchor (bc_kl_coef) sweep — closed-loop verdict (final, 2026-07-29)

Setting: honest stack (theta fix + RouteCursor route semantics),
3 base-town routes (25381/Town05, 25378/Town03, 27494/Town04),
deterministic, no recovery. Training: v3 config (K=8, H=15) with only
bc_kl_coef varied, except v5kl01 = v5 (lights) + kl 0.1 and kl01k16 =
kl 0.1 + K=16. Baseline bc_kl_coef is 0.3. All at 3x3 (2026-07-29
confirmations ran on the participation/H100 + visual/Quadro lanes).

| config | score | completion | penalty | 100% routes | raw json |
|---|---|---|---|---|---|
| v3 baseline (kl 0.3) | 5.91 | 22.5% | 0.387 | 0 | routefix2_3x3.json |
| kl003 (3x1 only) | 7.05 | 26.4% | 0.387 | 0 | kl003_3x1.json |
| **kl01** | **6.38** | **64.6%** | 0.110 | 1 | kl01_3x3.json |
| kl015 | 6.93 | 54.2% | 0.176 | 2 | kl015_3x3.json |
| kl02 | 4.45 | 46.9% | 0.134 | 0 | kl02_3x3.json |
| kl01k16 (K=16) | 3.81 | 39.4% | 0.124 | 0 | kl01k16_3x3.json |
| v5kl01 (lights + kl 0.1) | 9.19 | 27.5% | 0.404 | 0 | v5kl01_3x3.json |

**Anchor dose-response (mean completion, 3-route set):**
0.03 -> 26.4% | **0.1 -> 64.6%** | 0.15 -> 54.2% | 0.2 -> 46.9% |
0.3 -> 22.5%. Clean single peak at 0.1; tripling over baseline.

**K=16 does NOT transfer** (39.4% vs K=8's 64.6% at the same anchor):
Phase-1's strong K=16 preference is highway-env-specific — retrieval
breadth, like the anchor, must be tuned closed-loop (5th instance of
offline/proxy orderings failing to predict closed-loop driving).

## Findings

- **The BC anchor is the freeze.** Relaxing bc_kl_coef 0.3 -> 0.1
  triples completion (64.6% vs 22.5%) and produced the first 100%
  route (27494_rep1). kl015/kl02 at 3x1 sit at ~52% — the effect is a
  broad plateau around 0.1-0.2, not a knife-edge; 3x3 confirmations
  running to rank within the plateau (kl01's own 3x1 was 47.7%, so
  3x1 rankings inside the plateau are noise).
- **Commitment-vs-precision tradeoff is real and monotone**: penalty
  degrades as completion rises (0.387 baseline -> 0.110 at kl01).
  Freed steering wanders; composed score barely moves while
  completion triples. Report completion AND penalty; composed score
  alone is misleading at these route lengths.
- **v5kl01 is the opposite corner**: lights + weak anchor drives
  cleanest (penalty 0.480, best composed 10.99) but wedges earliest
  (24.7%). Lights training data teaches stopping; combined with the
  freed anchor it re-freezes. v5-vs-v3 interaction with the anchor is
  NOT additive — the winner must be picked closed-loop, per config.
- **Open-loop anti-prediction, third instance**: open-loop val
  (runs/b2d_kl015, b2d_kl02, b2d_v5kl01 results.json) is flat across
  all KL variants (MAE 0.209-0.217, latent-match within 0.0004) while
  closed-loop completion varies 3x. Open-loop metrics cannot select
  among these models — closed-loop eval is the only gate.

## Provenance

- Training runs: jobs 10527449/10527450 (kl003/kl01),
  10527520/10527521/10527522 (kl015/kl02/v5kl01), GPU A100,
  outputs ~/ditto_out/b2d_*; open-loop results committed to
  runs/b2d_kl015, runs/b2d_kl02, runs/b2d_v5kl01.
- Closed-loop: kl 3x1 sweep job 10527486 (kldiag), kl01 3x3 job
  10527501 (kl01x3), frontier 3x1 job 10527523 (kl015/kl02/v5kl01).
