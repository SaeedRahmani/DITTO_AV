# Gen-2 (1000 clips) — steps-per-datum sweep (2026-07-30)

3 base routes x 3 reps, honest stack. Reference: kl01_5x = 297 clips,
5x steps (score 17.11 / 56.3% / pen .426, 3 full routes).

| steps | score | completion | penalty | 100% routes | open-loop MAE |
|---|---|---|---|---|---|
| 5x  | 14.72 | 46.3% | 0.326 | 0 | .1822 |
| 10x | **16.80** | 48.1% | **0.382** | **3** | .1735 |
| 20x | 6.32 | **57.6%** | 0.123 | 1 | .1703 |

- Steps must scale with data: at fixed 5x steps, 4x data HURT
  closed-loop (14.72 vs 17.11); 10x restores parity. 20x flips to the
  far-but-sloppy profile (like weak-anchor gen-1 configs).
- Open-loop MAE improves monotonically with steps while closed-loop
  peaks at 10x — instance #7 of open!=closed.
- Both 25381 full completions of the project are in this sweep (10x
  rep1; 20x rep0) — the data scale DOES unlock the hardest route,
  just not reliably (route-level variance is huge; hence 3 seeds).
- H100 training verified (10x in 39:38 on participation) — use it for
  the seed runs.
- Frontrunners for the final 220: gen2_10x vs kl01_5x -> dev-10 decides.
