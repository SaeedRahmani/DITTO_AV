# Round-2 closed-loop results (2026-07-28, post theta-fix)

All numbers: 3 base-town routes (25381/Town05, 25378/Town03,
27494/Town04) x 3 reps, compass frame (yaw_offset pi/2 — settled by
scripts/replay_frame_check.py), deterministic, no stuck recovery,
gain 1.0 unless stated. Round-1 numbers are NOT comparable (90-deg
rotated obs, see NEXT_STEPS).

| agent / lever | score | completion | dominant failure |
|---|---|---|---|
| **v5 (v3 + traffic-light obs)** | **8.07 +- 4.96** | **33.2%** | blocked x8, tick-cap x1 |
| v3 baseline | 4.54 +- 2.63 | 15.5% | blocked x9 |
| v3 + stochastic sampling | 3.42 +- 2.42 | 8.8% | blocked/tick-cap/dev |
| v3 + steer x2, throttle x1.5 | 2.80 | ~4.8% | early wall contact |
| v3 + steer x3, throttle x1.5 | 1.81 | lower | worse still |

Raw records: v5_3x3_postfix.json, v3_3x3_postfix.json (v3 arm =
ab_thetaonly), gain/stoch arms in ~/ditto_out (freeze).

## Takeaways

- v5 doubles completion (33.2% vs 15.5%) and nearly doubles score. The
  only difference is the 6-dim traffic-light block — a training-side
  change. Every deployment-side lever tried on v3 made things worse
  (see ACTION_SHRINKAGE.md): the deployment knob space is exhausted.
- Light block verified in vivo (agent_ticks_10527224.jsonl): present on
  20.6% of 8070 model ticks, red-dominant while queued (R 1210 / Y 90 /
  G 364) — matches the offline anno distribution.
- "Agent got blocked" is still the terminal state (8/9): obstacle
  bypass remains THE missing behavior. Best single run: 54% completion
  (25381_rep0) — score crushed by infractions (penalty collapse), so
  collision avoidance scales in importance as routes get longer.
- n=9 with std ~5: score difference v5-v3 is ~1.9 sigma — treat the
  ranking as strong-but-provisional; the completion doubling is the
  robust signal. Any future claim: >= 3 reps, this route set.

## Next (training-side, in order)

1. Fix action shrinkage at the source (turn-frame |steer| mu is ~28% of
   expert): actor-head scale calibration / non-Gaussian head; re-check
   the offline probe before burning GPU on closed loop.
2. Obstacle-bypass behavior: the blocked terminal state is data-limited
   (the expert rarely wedges); consider recovery-state augmentation or
   closed-loop-aware fine-tuning.
3. Then the 10-route dev eval (needs AdditionalMaps re-extract, post
   freeze) with v5 at gain 1.0.
