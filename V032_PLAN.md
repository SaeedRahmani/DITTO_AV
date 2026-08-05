# DITTO_AV v0.3.2 — plan

Branch `saeed/v0.3.2`, cut from frozen `saeed/v0.3` on 2026-08-05.
Workspace: `/scratch/$USER/ditto_av/DITTO_AV_v032` (git worktree).
Runs in parallel with v0.3.1, which continues on `main` in the
DITTO_AV checkout — never edit or build on main from here.

## 1. Mission — kill the jiggle, keep the reactivity dividend

v0.2/v0.3 drive with visible steering wobble (3x3 probe: 8-13 steer
sign flips/100 ticks on bad runs). Mission: make the ego drive smoothly
WITHOUT giving back v0.3's reactivity dividend (vehicle collisions 6 =
all-time best), and do it thesis-loyally: motion quality must be
priced inside the training world (richer expert-STATE matching), not
patched by test-time filters. Rules carry over from V03_PLAN §3: one
axis per iteration, pre-registered gates, every result -> ledger with
job id.

### 1.1 Diagnosis (mechanism, to be confirmed by A0)

Wobble is free in training and expensive only in CARLA:
- EgoSim executes each fresh 10 Hz plan perfectly (slide along the
  polyline, no tracking dynamics; dtheta_max 0.35 rad/0.1 s permits
  3.5 rad/s yaw rate) — successive plans that disagree laterally cost
  ~nothing.
- The reward matches (pos, yaw, speed) under Gaussian tolerances;
  oscillation lives in the DERIVATIVES (yaw rate, lat jerk), which no
  channel prices.
- Deployment tracks those churny plans with P-only pure pursuit
  (kp_steer 1.6, 10 Hz replan, no damping) on a car with inertia ->
  limit-cycle steering.
- Possible co-factor: expert wp labels are finite differences of
  logged positions (position-noise floor, V03_PLAN A1), so BC may have
  learned label jitter.

### 1.2 A0 — smoothness audit (measure before building)

Metrics (10 Hz model ticks): yaw rate + yaw accel distributions; sign
flips/100 ticks above a deadband; steer |delta| mean/p95 (CARLA);
plan churn = motion-compensated lateral disagreement of consecutive
plans (exact compensation from ego_glob on npz populations; wp1-lat
proxy on CARLA tick logs). Populations:
  (a) expert logs (target band + label noise floor),
  (b) BC/champion/clp_rx plans open-loop on logged obs,
  (c) clp_rx closed-loop in EgoSim,
  (d) clp_rx dev-10 CARLA tick logs (existing
      agent_ticks_v03rx_d10A/B.jsonl — no new CARLA runs needed).
Decision tree: churn already in (b) -> labels/policy axis; born in
(c) -> reward/entropy axis; only in (d) -> tracker-interaction axis.

### 1.3 Axes (ranked; ONE per iteration)

1. Match how the expert MOVES: add a yaw-rate channel to the EgoSim
   reward kernel (sigma_yawrate, same pattern as sigma_p2; expert yaw
   rate from ego_glob finite differences read above the noise floor).
   Retrain = D3-style champion fine-tune, everything else frozen.
2. Price churn in the dream: penalize motion-compensated disagreement
   of consecutive executed plans during rollouts.
3. Policy-side temporal-consistency aux loss (BC and/or RL).
4. Refit expert tracks with a smoother if A0 shows the label noise
   floor dominates.
5. Control arm (attribution only, NOT the contribution): eval-only
   tracker damping on the unchanged clp_rx — plan EMA 0.5 (~0.2 s
   lag). Bounds how much jiggle is tracker-side vs model-side.

### 1.4 Pre-registered gates

Driving gate (pinned NOW, judges every arm; same dev-10 A/B chain
protocol as W3): DS mean(A,B) >= 82.5 (clp_rx tie-or-beat), vehicle
collisions <= 8 over the 30 runs (dividend kept; clp_rx = 6),
completion 30/30. An arm that smooths but breaks this gate FAILS.
Smoothness gate S1 (numbers pinned at A0 close, 2026-08-05, BEFORE
any arm's training/eval runs; multiples committed once, per metric):
- S1a CARLA: steer sign flips/100 ticks <= 14.8
  (1.5x expert-log steer flips 9.84; clp_rx today: 18.1).
- S1b EgoSim closed-loop (deterministic, val, B=192 H=40, the A0
  protocol): yr sign flips/100 <= 7.1 (1.5x label-replay floor 4.7)
  AND yaw-accel p95 <= 5.7 rad/s^2 (2x floor 2.83 — 2x not 1.5x
  because a p95 of a second difference is the noisiest of the three
  estimators; clp_rx today: 39.2 / 24.6).
The EMA control arm (1.3.5) is attribution-only and is NOT judged by
S1/driving gates. 3x3 stays a canary, never a selector.

## 9. Ledger

- 2026-08-05 AXIS-1 ROUND 1: job 10582600 (a100-small, 54 min),
  sigma_yr 0.5, 3500 steps. Wobble HALVED but gate missed: S1b
  yr_flips 24.2 (gate 7.1), ya_p95 12.24 (gate 5.7), churn 0.38 ->
  0.215; old-currency G2 REGRESSED vs clp_rx (reactive 0.654 vs
  0.685, perr 1.71 vs 1.38) -> does NOT go to CARLA (in-sim no-go).
  Training telemetry: reward plateaus 0.22-0.28 from step 500 (init
  0.092) — at sigma 0.5 the channel is a CLIFF: residual wobble
  0.9-1.5 rad/s scores e^-1.6..e^-4.5 ~ 0, so no gradient through the
  wobble region while the KL anchor (kl ~0.9) pins the wobbly init.
- 2026-08-05 AXIS-1 ROUND 2 pre-registered BEFORE the run (same
  axis, one refinement + capacity): sigma_yr 0.5 -> 1.0 (restores
  slope: 1.5 rad/s costs 0.32, expert-band 0.5 costs 0.88) and steps
  3500 -> 6000 (plateau-at-500 says optimization, given gradient,
  needs room; D3's 3500 was tuned for the old objective).
  Go/no-go to CARLA unchanged: old-currency reactive reward >= 0.68
  AND the FULL S1b gate (yr_flips <= 7.1, ya_p95 <= 5.7). On a second
  miss: stop tuning sigma, take the finding to Axis 2/3 (churn is the
  quantity to price, not its yaw-rate shadow).
- 2026-08-05 AXIS-5 (EMA attribution) CLOSED — jobs 10582712 (d10A) /
  10582713 (d10B), visual; earlier 10582601/02 burned by the sbatch
  --export comma trap (documented in v032_carla_chain.sbatch).
  UNCHANGED clp_rx + tracker ema 0.5: dev-10 A 89.67 (IDENTICAL to
  clp_rx A, same 3 veh + 1 lay collisions) / B 62.60 (vs 75.40) ->
  mean 76.14 vs 82.53. Vehicle collisions 12 vs 6 — the 0.2 s plan
  lag DOUBLES vehicle collisions, all on the interaction-heavy B
  routes, while completion stays 30/30. Steering DID smooth: flips
  18.1 -> 13.2/100 (S1a-passing), |dsteer| p95 0.36 -> 0.23.
  ATTRIBUTION: ~27% of steer flips are tracker-removable, but
  test-time filtering spends the ENTIRE reactivity dividend (+6.4 DS)
  to get them. Smoothness must come from consistent plans (Axis 1),
  not filtering — the thesis-loyal route is now also the
  empirically-forced one.
- 2026-08-05 A0 CLOSED (login-CPU, no jobs; outputs/v032/a0_*.json).
  Exact lat churn of consecutive plans at wp1 (m, val): expert labels
  0.0126 / clp_bc 0.037 / clp_rl 0.166 / clp_rx 0.208 open-loop;
  clp_rx closed-loop in sim 0.38. Sim-state wobble (closed-loop):
  clp_rx yr-flips 39.2/100, yaw-accel p95 24.6 rad/s^2, yaw rate
  RIDES the dtheta_max clamp (max 3.5 rad/s) — vs label-replay floor
  4.7 / 2.83 (execution adds ~nothing; sim kinematics innocent) and
  expert log 7.0 / 3.0. clp_bc closed-loop is smooth (6.2 flips) but
  low-reward 0.35 (drifts); clp_rx collects reward 0.770 > label
  replay 0.724 while wobbling 8x harder -> the reward PAYS MORE for
  churny driving than for near-perfect label execution. CARLA
  (existing v03rx dev-10 ticks): steer flips 18.1/100 moving 16.8,
  |dsteer| p95 0.36, wp1 churn proxy 0.22 m (expert same-proxy
  0.038). VERDICT: jiggle is born in the RL stage (BC 0.037 -> RL
  0.166), amplified closed-loop, invisible to the reward. Labels are
  clean -> Axis 4 dead. Sim execution clean -> Axis 2 not the lever.
  PROCEED: Axis 1 (sigma_yawrate reward channel), sigma_yr = 0.5
  rad/s pre-registered (expert yr p95 0.497 costs exp(-0.5); clp_rx
  wobble 1.5 rad/s costs exp(-4.5)). Axis 5 EMA arm = attribution.

## 2. Ops deltas vs V03_PLAN §7 (everything else inherits)

- This worktree replaces the removed `DITTO_AV_v03` one; ignore stale
  references to that path in older docs on this branch.
- `v03_d3` / `v03_w0` / `v03_data_cache` sbatch scripts are repointed
  to `DITTO_AV_v032`. Older v0.1/v0.2-era sbatch scripts still target
  main's checkout (`ditto_av/DITTO_AV`) — repoint before reusing any
  of them from here, or the job will run v0.3.1 code from main.
- Shared with main, coordinate instead of colliding: `../outputs`
  (CLAIM stages in `outputs/PIPELINE_STATUS.md` with tag "V03.2:",
  read the tail first), `../data`, `../envs`, CARLA sif + overlay.
  Never cancel jobs you didn't submit; main's v0.3.1 runs have right
  of way on lanes.
- Name v0.3.2 run outputs distinctly (`v032_*`) so collectors that
  glob `outputs/` don't mix versions.
- Worktree lifecycle (same as DITTO_AV_v03): when v0.3.2 merges or is
  abandoned, remove the worktree, keep the branch.
- Commits as Saeed Rahmani, no AI attribution. Scratch has a 1M-inode
  quota; extractions go to node-local /tmp.
