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
Smoothness gate S1: numbers pinned from A0's expert band BEFORE any
arm's training/eval runs (form: CARLA steer sign flips/100 ticks and
yaw-accel p95 within a fixed multiple of expert; multiple chosen and
committed at A0 close). 3x3 stays a canary, never a selector.

## 9. Ledger

- (open) A0 audit: submitted; results land here with job ids.

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
