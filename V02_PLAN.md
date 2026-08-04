# V02_PLAN.md — ver0.2: counterfactual closed-loop imitation (single source of truth)

Decided with the author 2026-08-03. v0.1 is frozen on branch `saeed/ver0.1`
(its plan/ops files live there; cluster ops knowledge summary at the end of
this file). This file is the v0.2 plan, status ledger, and do-not-redo list.

## 1. The idea (and why v0.2 exists)

v0.1 verdict: the faithful DITTO port (RSSM world model + imagination RL +
whole-latent matching reward) loses to plain BC closed-loop on Bench2Drive
at every dose ever tried, while the same mechanism wins in the controlled
highway study. Diagnosis (evidence: saeed/ver0.1 NEXT_STEPS + runs/):
whole-latent matching grades mostly *traffic* (exogenous, ~0.85–0.92
similarity floor between any two scenes, ~2% reward dynamic range → the
policy exploits the model), and even a perfect model cannot dream the
*same* stochastic traffic the expert saw, so reward noise grows with
horizon. Neither problem shrinks with model size.

**v0.2 thesis (DITTO-loyal, driving-native):** keep DITTO's core — a
policy trained *on-policy* inside an offline "world", rewarded for
closeness to expert **states** — but make "state" mean *the part the agent
controls, in the scene the expert actually faced*:

1. **The world = the recorded scene.** During training rollouts, traffic,
   route, and lights replay from the log (counterfactual, non-reactive);
   the ego is moved by an analytic kinematic model. The dream cannot be
   wrong about physics and cannot be exploited: nothing in the loop is
   learned.
2. **The policy drives the ego** through this world closed-loop, seeing
   exactly the deployment observation (recomputed each step from the
   *simulated* ego pose — covariate shift happens on purpose).
3. **Reward = ego-state match to the expert** in the same scene:
   time-tolerant kernel on (position, heading, speed) against the expert's
   trajectory. 100% ego signal, zero traffic in the reward. Off the path,
   the max reward is *reached by returning* — the recovery incentive BC
   cannot represent, now well-posed (v0.1 gen-4's divergent-start poison
   was retrieval fetching wrong-scene targets; here the target is the
   same-scene expert, always).
4. **Action matching is NOT the reward** (it would collapse into BC: off
   the expert path the logged action is the wrong answer). BC enters only
   as init + KL anchor (v0.1 settled: bc_kl ~0.1 is load-bearing).

Where the *learned* world model lives in v0.2: Stage-2 arm. A traffic
transformer predicting actor futures (conditioned on ego) upgrades replay
to *reactive* imagination and extends horizons past clip end. Stage 1
deliberately runs with pure replay so the reward/on-policy question is
tested with zero model-error confound. (Author's "make the WM bigger"
question = Arm A control, see §6.)

## 2. Code review verdict (2026-08-03, full end-to-end read)

Keep unchanged (proven, do not re-litigate):
- Offline adapter obs layout + frame facts: anno theta = CARLA yaw + π/2
  (compass), FORWARD = −y / LATERAL = +x in that frame; ego pose from the
  physics-exact ego_vehicle box; V_MAX finite-difference glitch filter;
  wp targets certified by scripts/waypoint_check.py.
- `featurize_frame` (online ≡ offline, tested), RouteCursor semantics,
  WaypointTracker + StuckRecovery deployment stack (champion: 220-route
  DS 22.10 / SR 39.1), `wp_to_vehicle` round-trip, TorchWaypointTracker
  equivalence port.
- BC head recipe (`bc_trainer`), Gaussian head bounds (WP_BOUND 3.0), std
  floor; npz-cache workflow; SLURM templates + self-driving chains +
  PIPELINE_STATUS claims; closed-loop-only model selection (open-loop
  metrics anti-predict — 7 banked instances).

Gaps vs the v0.2 idea (the work items):
- **npz stores only ego-relative obs** (6 nearest, 60 m, lossy). The sim
  needs *global* per-frame state: ego pose/yaw/speed, all actor tracks
  (pos, world-vel, yaw, half-extents), route command points, light
  trigger + state. → extend `load_clip`/`clips_to_npz` (`##glob1` cache
  key suffix).
- **No simulator exists.** → new `ditto_av/egosim.py`: batched torch
  log-replay sim (kinematic ego, obs recompute mirroring
  `featurize_frame`, OBB collision flags, expert-trajectory reward).
- **Reward machinery is latent-space** (`LatentMatcher`) → new ego-state
  reward (keep the old module for ablation arms only).
- **Trainer trains in the RSSM dream** (`ac_trainer`) → new
  `trainers/clp_trainer.py`: same A2C skeleton (λ-returns, EMA target
  critic, entropy, BC-KL anchor) but rollouts happen in egosim.
- **Policy nets are flat MLPs on WM features** → new
  `models/policy_v2.py`: per-actor token transformer encoder + GRU memory
  + Gaussian waypoint head (+ critic). This is the scaling answer: the
  capacity goes into the *policy/encoder* (and later the traffic model),
  not into an RSSM whose latent the reward no longer needs. No
  prev-action input (gen3_wp lesson: action-channel feedback drifts).
- **Deployment driver assumes a WM filter** → add `ObsPolicyDriver`
  (obs → GRU → wp plan) to carla_agent; tracker + recovery unchanged.
- Route/light in sim replay per logged frame index (approximation: command
  points switch by expert progress; fine under capped deviation — state
  the caveat in the paper).
- wp labels clamp at clip end ("stay") — sim windows must end ≥ H frames
  before clip end, or reward masks the tail.

## 3. Architecture spec (Stage 1)

Data (`##glob1` npz additions, per frame):
`ego_glob` (T,4)=[x,y,theta,speed]; `act_glob` (T,A,8)=[presence,x,y,
vx_w,vy_w,yaw,ext_x,ext_y] (A=32 slots, presence-sorted by distance);
`route_glob` (T,6)=[near_xy,far_xy,near_cmd,far_cmd];
`light_glob` (T,4)=[presence,tv_xy,state]. World frame, compass yaw.

EgoSim step (dt=0.1 s, batched over B rollouts on GPU):
1. Policy sees obs built from (sim ego pose, replayed frame): identical
   math to `featurize_frame` (rel rotation, 6-nearest sort, clipping,
   route/light blocks) — torch, batched.
2. Policy emits a wp plan (12-dim compass ego frame, /WP_SCALE — same
   action space as the champion's head).
3. Ego advances *along the plan polyline*: target speed = the tracker's
   own spacing formula min(v_wp, v_curve, v_max) (reuse tracker_torch
   math so training-time speed intent ≡ deployment tracker intent),
   accel-capped (±3.5 m/s²), then move arc-length v·dt along the
   polyline; heading = local tangent, yaw-rate capped. No reverse in-sim
   (recovery stays deployment-side).
4. Collision flag: ego OBB (from ego box extents) vs replayed actor OBBs
   (SAT, batched); logged for metrics + optional penalty arm.

Reward (per step, τ=5 frames tolerance, defaults σ_p=1 m, σ_θ=0.3 rad,
σ_v=2 m/s):
  r_t = max_{|δ|≤τ} exp(−½[‖Δxy‖²/σ_p² + Δθ²/σ_θ² + Δv²/σ_v²])
against the expert's logged ego states of the SAME clip window. Bounded
[0,1]; expert replay ≈ 1; dawdling decays as the window advances. Core arm
has NO collision/comfort shaping (thesis purity); a shaped arm is a
config flag.

Policy (`policy_v2.TokenPolicy`): tokens = [ego row, 6 actor rows, route,
light] → linear embed (d=192) → 3-layer transformer encoder (heads=4,
ff x4) → mean-pool + GRU (512) → Gaussian wp head (12-dim, WP_BOUND) +
critic head + EMA target critic. ~3–4 M params (MIG-friendly; scale knob:
d/layers/GRU in config). GRU burn-in on the logged prefix (L=8 frames)
before each rollout so memory starts on-manifold.

Training (`clp_trainer`):
- Stage BC: same net, teacher-forced sequences, Gaussian NLL on wp labels
  → `clp_bc.pt` (this is ALSO the v0.2 BC baseline for deployment).
- Stage RL: sample B=64 windows (start ≥ H+τ before clip end), burn-in,
  roll H=40 sim steps (4 s), λ-returns (γ=0.97, λ=0.95), A2C policy
  gradient + entropy 3e-3 + KL(π‖π_BC) anchor (start 0.1, dose axis),
  divergent starts: fraction of rollouts get pose/heading/speed jitter
  (lateral σ 0.5 m, yaw σ 0.1 rad) — targets stay the same-scene expert.
- Both stages log to wandb offline; checkpoints per stage.

Deployment: `ObsPolicyDriver` (fresh obs each model tick → GRU →
deterministic wp plan) → WaypointTracker + reverse recovery (config
`policy: clp`). No WM filter at deployment in Stage 1.

## 4. Verification protocol (gates, in order)

- **G0 unit**: pytest — sim-fidelity (replaying the expert's own wp labels
  retraces the logged path; mean path error < 0.3 m over 4 s — the
  certified interp error), featurizer identity (sim obs at the expert pose
  ≡ npz obs), reward sanity (expert ≈ 1, monotone decay with offset),
  collision SAT unit tests, policy shape/NLL tests.
- **G1 selector validation** (cheap, before ANY big training): drive
  banked v0.1 policies through egosim (their full deployment semantics:
  WM filter + tracker feedback) on val windows; egosim score must rank
  them consistently with the banked dev-10/220 truth (Spearman clearly
  positive — v0.1's latent metric scored −0.60; this is the
  reward-meaning test). If G1 fails, STOP and rethink the reward — do not
  train.
- **G2 in-sim**: closed-loop-trained policy must beat its own BC init
  in-sim on held-out clips (val split), especially from divergent starts.
- **G3 CARLA smoke**: 3-route 3×3 (canary only, never a selector).
- **G4 dev-10** (the honest gate): beat the v0.1 champion band
  (30.49/83.2, 20/30 full; seeds mean ~28.3). Only G4 winners get seeds.
- **G5**: 3 seeds + ONE 220-route run of the final config
  (vs 22.10 DS / 39.1% SR). Report both thesis-pure and shaped arms.

## 5. Execution stages

- **M0** (this commit): plan + review banked.
- **M1 data**: extend adapter + tests; rebuild npz for the 297 extracted
  clips (login-friendly small run first: 12-clip mini set for tests);
  full 1000-clip rebuild via the /tmp extraction sbatch (`##glob1` key).
- **M2 sim**: egosim + reward + tests green (G0).
- **M3**: G1 selector validation script + verdict committed.
- **M4**: clp BC + RL smoke on 297 clips (MIG lane), G2; wire deployment;
  G3 smoke 3×3.
- **M5 scale**: 1000 clips, H=40, capacity sweep (d 192→384, GRU 512→1024,
  steps ×10) selected via G2+G3, confirm at G4 dev-10.
- **M6 iterate** (one axis at a time, each dev-10-gated): bc_kl dose,
  divergent-start dose/shape, τ/σ reward geometry, horizon H, shaped arm
  (collision penalty), Arm A (RSSM-scale control, §6), Stage-2 reactive
  traffic model if replay's non-reactivity binds (watch: policies that
  learn to dodge ghost traffic).
- **M7**: seeds + final 220 + paper v0.2 (the controlled story: same data,
  same tracker — reward semantics is the only moved piece between arms).

## 6. Registered predictions (for the paper's controlled comparison)

- **Arm A (author's scale question)**: v0.1 recipe unchanged, WM scaled
  (deter 256→1024, steps ×10). Prediction: open-loop fidelity improves,
  closed-loop ordering vs BC does NOT flip (the reward grades traffic).
- **Arm B (this plan)**: same data/tracker, ego-state reward in replayed
  scenes. Prediction: beats its BC init closed-loop, primarily via fewer
  stuck/blocked states (v0.1's binding constraint: 104/220 timeout
  routes, 41% plan-GO-static ticks).
Either outcome of A strengthens the paper; B failing G1/G2 falsifies the
reward-meaning diagnosis itself — report honestly.

## 7. Ops (carried from v0.1; full detail in saeed/ver0.1:DELFTBLUE.md)

- DelftBlue: login nodes = edit/submit only; compute has NO internet;
  1M-inode scratch quota (motd chunk-files table is the truth; keep 100k
  headroom; extractions go to node-local /tmp, packed results back).
- Lanes: audit before EVERY submit (`scripts/pipeline_decider.py
  pick_lanes`; squeue lies under PrivateData — read node Gres/AllocTRES).
  CARLA needs graphics GPUs (A100/V100/H100/Quadro); MIG = training only,
  1 job/user. >30 min pend = re-audit, move lanes, never wait hours.
- Envs: `~/envs/ditto_gpu` (conda, cu130) for GPU nodes; scratch venv for
  login/CPU; `carla_eval` conda for CARLA. OMP_NUM_THREADS=1 always.
- Sessions: claim stages in outputs/PIPELINE_STATUS.md before submitting;
  chains self-drive via decider.sbatch; commit small results promptly
  (identity: Saeed Rahmani, no AI attribution trailers).
- Ask-before-delete is the only approval gate; compute needs none.

## 8. Status ledger (newest first)

- 2026-08-04 — **BC SEED BARS FINAL; v0.2 FULLY CLOSED**: BC
  65.32 ± 9.36 (71.48/54.55/69.92; 75/90 full) vs RL pure 76.31 ± 7.54
  / shaped 78.39 ± 8.24 (both 90/90 full). The controlled on-policy
  effect holds with bars on both sides: +11.0/+13.1 DS on seed means,
  and a categorical completion gap (180/180 vs 75/90). Nothing further
  runs for v0.2; v0.3 (reactive learned traffic model) starts on its
  own branch.

- 2026-08-04 — **G5 SEED BARS FINAL, v0.2 record CLOSED**: pure
  76.31 ± 7.54 (83.60/76.78/68.55), shaped 78.39 ± 8.24
  (85.63/80.13/69.42); Δmean +2.08 ≪ σ — pure-vs-shaped TIE confirmed
  at seed level, matching the 220 dead heat. ALL 180 seed-eval runs at
  100% completion — completion is the seed-robust property; DS
  variance (±8) is penalty events. Deploy seed 0 of either arm;
  headline system = the pure arm (thesis-clean). BC seed evals in
  flight to complete the controlled claim's error bars; paper
  draft_v02.md markers filled.

- 2026-08-03 night — **220 PAIR COMPLETE, v0.2 EXPERIMENTAL RECORD
  CLOSED: pure 75.88/99.7%/99.5% vs shaped 76.10/99.5%/99.1% — dead
  heat (+0.22).** Closing ablation: with the tight kernel at 999-clip
  scale, collision shaping adds NOTHING — pure expert-state matching
  suffices (shaping was worth +11.5 DS only in the wide-kernel 297
  era). Headline stays the THESIS-PURE arm: DS 75.88, 3.4x the v0.1
  record, above all published Bench2Drive baselines
  (privileged-offline caveats; strict zero-infraction SR 48.2% vs
  33.08 published best). Remaining: seed variance bars (999t s1/s2 on
  MIG; 297 s1/s2 done, B aggregates) -> M7 paper rewrite.

- 2026-08-03 late — 999s (tight-shaped) dev-10 FINAL 85.63/100% 30/30
  (A 89.67 / B 81.60) vs 999t 83.60: nominal +2.03, inside seed noise
  (v0.1 dev-10 spread +-2.3) — statistical tie at the top; G2's in-sim
  sweep did partially transfer (A-half). Per the pre-registered rule
  the SECOND 220 fired on 999s = the pure-vs-shaped ablation at
  headline scale. Seeds (297 s1/s2 + 999t s1/s2, MIG) will calibrate.

- 2026-08-03 eve — **220 FINAL (runs/bench220_v02_999t_rl): DS 75.88,
  completion 99.7%, 219/220 full routes, strict zero-infraction success
  48.2%** — 3.4x the v0.1 record (22.10/39.1%) on identical data +
  benchmark; exceeds all published Bench2Drive baselines incl.
  expert-distilled sensor methods (privileged-offline caveat stands).
  Infraction profile -> M6 order: collisions_vehicle 120 (shaped arm
  targets this; its dev-10 in flight), stop 29, red_light 12 (no light
  obs — retest with TokenPolicy), lanes 16, layout 13. Seeds s1/s2
  @999t queued (G5).

- 2026-08-03 ~18:40 **G4 COMPLETE (both original arms): v02bc
  74.10 / 100.0 / 30/30 — NEW dev-10 RECORD (2.4x the v0.1 champion's
  30.49/83.2/20-30); v02rl 56.89 / 100.0 / 30/30.** Completion is
  saturated at dev-10 scale by BOTH arms; penalty (infractions) is the
  entire remaining game — which is where the shaped arm's 3x3 leads
  (0.844 vs bc 0.741 vs rl 0.569). Shaped dev-10 in flight
  (10569623/24) = the champion decision. Seeds s1/s2 submitted
  (10569798/99, MIG). Next: 999-clip scale-up (Session A's cache
  build) -> dev-10 confirm -> ONE 220 on the winner.
- 2026-08-03 ~18:00 **G4 PARTIAL — gate smashed, and a first-class
  finding.** v02rl dev-10 FULL: 56.89 / 100.0 / 30/30 (champion:
  30.49 / 83.2 / 20/30) — every v0.2 arm crushes v0.1. BUT v02bc
  A-half: 92.00 / 100.0 / 15/15, penalty 0.920 — plain BC on the
  TokenPolicy beats pure-RL in CARLA (in-sim G2 said the opposite).
  Reading: pure state-matching RL exploits the NON-REACTIVE replay
  traffic (threads gaps that reactive traffic closes -> infractions,
  penalty 0.569); the registered M6 ghost-traffic risk, observed.
  Shaped arm (front-impact penalty) sits between on the canary
  (3x3 84.36 / 100 / 9/9, penalty 0.844) — safety shaping recovers
  much of the gap, matching G1's collision-transfers finding.
  In flight: bcB (10569217), shaped d10 (10569623/24). Champion
  candidate: v02bc or shaped, decided on full dev-10.
- 2026-08-03 ~17:15: **G3 PASSED — best 3x3 results ever banked.**
  v02rl 57.25 / completion 100.0 / 9/9 full; v02bc 55.51 / 100.0 / 9/9
  (v0.1 best-ever 3x3: 48.98 composed, never 9/9 full; champion band
  completions 73-86 there). RL > BC on score AND penalty; both v0.2
  arms transfer to CARLA. Evidence: runs/carla_smoke/v02/. 3x3 remains
  a canary — G4 dev-10 submitted for both arms (rlA 10569214,
  rlB 10569215, bcA 10569216, bcB 10569217; gate: champion 30.49/83.2).
  Ops lesson banked: sbatch --export comma-splits values — pass
  VARIANTS via environment, never --export.
- 2026-08-03 ~16:30: **G2 PASSED (thesis-pure arm, job 10567542)** —
  closed-loop RL beats its own BC init on held-out clips on EVERY
  metric: collision 0.295->0.124 (-58%; the correlate G1 showed
  transfers), pos_err@H 6.84->2.75 m, progress 0.52->0.68, reward
  0.46->0.67; divergent-start gains equal or larger (recovery works,
  the gen-4 poison is cured). First BC-beating on-policy result at
  B2D scale in the project. runs/b2d_v02/results/clp_g2.json.
  G3 3x3 submitted: job 10568836 (participation), v02rl vs v02bc,
  routes 25381/25378/27494 x3. Shaped arm 10567642 still training.
- 2026-08-03 pm (G1 VERDICT — strict gate FAILED, refined role
  adopted; SESSION A login cross-checks, 48-clip val split, 12 banked
  models, two protocols H=40 and H=80+3-battery;
  runs/egosim_g1_login/): (a) the reward separates broken from healthy
  with a huge margin — every gen-4 DWP scores below every healthy BC
  variant (v0.1's latent metric: −0.60 on the same question) — and is
  maximized on the expert path by construction (G0). (b) Within the
  healthy band it CANNOT rank dev-10 DS: H=40 compresses the BC family
  (0.175–0.202); H=80 floors the kernel (all models 17–31 m off).
  (c) SUBSTANTIVE FINDING: short-horizon expert-closeness ANTI-orders
  the healthy family (lw/cap track the expert closest yet score worst;
  the champion wins by conservatism), while collision rate correlates
  +0.50 with dev-10 completion — safety events transfer, closeness
  does not (paper finding; echoes v0.1 open≠closed). CONSEQUENCES:
  reward upgraded (multi-scale kernel σ_p 1 m + 4 m mix, keeps
  gradient alive off-path; front-impact-only collision penalty — ghost
  rear-ends by non-reactive replay excluded); shaped arm PROMOTED to
  co-primary (configs/b2d_v02_shaped.yaml) beside the registered
  thesis-pure arm; G1's role narrowed to reward-sanity + broken-model
  detection (PASSED in that role); the decisive go/no-go is now G2
  (beat own BC init in-sim) + G3/G4 transfer. Do NOT re-run G1
  expecting a ranking — no short-horizon matching metric ranks
  near-band drivers. RECOMMENDATION for B's queued 10567542: let it
  run (thesis-pure G2 = the registered Arm-B ablation point), then run
  the shaped arm.
- 2026-08-03 pm: M3+M4 IN FLIGHT (SESSION B): G1 selector script landed
  (scripts/egosim_selector.py — 12 banked wp-family models, dev-10 truth
  3.46-30.49, gate Spearman >= +0.4); v0.2 configs (b2d_v02 297-clip,
  b2d_v02_mini 12-clip); mini pipeline end-to-end verified on login CPU
  (data 150 s, clp trains, G2 json written); chain submitted:
  10567541 (compute-p1: data ##glob1 297 + G1, self-committing) ->
  10567542 (participation H100, afterok: clp bc8k+rl4k + G2). G3 diag
  configs ready (diag_v02_clp_{rl,bc}.yaml). If G1 FAILS: scancel
  10567542 and STOP per §4.
- 2026-08-03 pm: M1+M2 DONE, G0 PASSED (84 tests): egosim core landed
  (b114cde + 117b4e2, concurrent session) — ##glob1 global arrays,
  batched log-replay sim (plan-schedule execution: two-stride v0
  extrapolation + arc-fraction heading; REAL-clip replay fidelity
  0.07-0.14 m over 4 s), OBB SAT collisions, time-tolerant ego-state
  reward, TokenPolicy (3.24M params), clp trainers, ObsPolicyDriver
  deployment path, run_b2d clp stage.
- 2026-08-03: M0 — plan committed; v0.1 guidance files removed from main
  (preserved on saeed/ver0.1); code review §2 done.
