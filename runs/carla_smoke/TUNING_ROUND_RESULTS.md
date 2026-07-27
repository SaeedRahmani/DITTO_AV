# Closed-loop tuning round (2026-07-27, during scratch freeze)
3 base-town routes (25381/25378/27494) x 3 repetitions, home outputs.
- v3 (K=8, H=15, route-cond): score 12.6 +- 11.3 | completion 43% +- 34
- v4 (K=16, H=5, route-cond):  score  6.7 +-  3.9 | completion 17% +- 11
Conclusions:
1. v3 confirmed better closed-loop across 9 runs each -> keep as driver.
2. Phase-1-optimal K16/H5 does NOT transfer to closed-loop B2D.
3. Run-to-run variance is huge (same route 4.7..24.2) -> paper eval
   protocol MUST use repetitions; single-route claims invalid.
4. Universal failure mode: agent blocked (wall-wedging, no recovery).
TODO when scratch unfreezes: commit runs/carla_smoke/* + this file to
repo; re-extract AdditionalMaps (partial!); traffic-light obs + retrain;
stuck-recovery behavior; then full dev10 x reps.
