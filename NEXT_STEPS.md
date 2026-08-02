# NEXT_STEPS.md — live status + plan (single source of truth)

Read DELFTBLUE.md for cluster rules (lane audit protocol, storage,
conventions). PAPER_PLAN.md is the paper-side plan and evidence log.
This file: where the project stands, the plan, what not to redo.
History: git log of this file — old session narratives were compressed
out on 2026-07-29 (C2 cleanup); nothing scientific was lost, the
verdicts live in runs/ and the settled-facts list below.

## >>> HANDOFF 2026-08-02 — read this first <<<

**GEN-4 DITTO-WP CLOSED (2026-08-02): the DITTO-loyal imagination
refinement of the wp head LOSES to plain BC at EVERY dose — 5-point
curve, all dev-10 (champion BC = 30.49/83.2, 20/30):**
| variant | score/compl | full |
|---|---|---|
| v1: kl 0.1 + divergent starts | 3.46/50.4 | 4/30 |
| kl 0.3 + divergent | 19.49/70.8 | 11/28 |
| kl 0.3, no divergent | 24.07/80.8 | 15/30 |
| kl 0.3, no div, 3k steps (early stop) | 13.31/71.1 | 16/30 |
| kl 1.0, no divergent, 30k | 18.08/60.5 | 12/30 |
The architecture was the most careful loyal instantiation possible:
deployment-consistent imagination (dream through the equivalence-
pinned tracker port), task-projected nearest-mode rewards (wp-probe
subspace, R2 0.82), retrieval-relabeled divergent starts, bc_kl
anchor. Attributions: divergent starts are actively harmful
(off-manifold retrieval injects bad targets); the residual harm is the
imagination pressure itself; and in the es/k10 runs the DETERMINISTIC
mean wp-MAE improved BELOW BC's (0.045-0.051 vs 0.062) while driving
degraded — the metric-inversion finding in its sharpest form. Evidence
in runs/carla_smoke/gen4_dwp/. DO NOT re-run imagination-matching
variants on B2D without a fundamentally new reward idea; the dose axes
(anchor, steps, start distribution) are exhausted.
CHAMPION UNCHANGED: gen3_wph BC + rec (220: DS 22.10 / SR 39.1%).

## (previous handoff, 2026-08-01, still-valid context below)

State: GEN-3 Phases 0 AND 1 are CLOSED. **Champion = gen3_wph rec**
(waypoint BC head on the reused gen3_clean control-action WM,
WaypointTracker + reverse recovery; configs/diag_gen3_wph_bc_rec.yaml;
ckpts ~/ditto_out/b2d_gen3_wph): **220-route FINAL DS 22.10 /
completion 68.7% / SR 39.1%** (runs/bench220_gen3wph) — project
records; SR beats all published baselines (DriveAdapter 33.08).
Phase-5 seed bars exist (dev-10 seeds 0/1/2: 30.49/25.86/28.45,
mean ~28.3 +- 2.3 — full-pipeline retrains s1/s2).
The COMPLETE post-champion probe ledger (7 verdicts, all dev-10-gated,
evidence in runs/carla_smoke/gen3_wph_era/) is in the sections below —
read it before proposing any deployment tweak or BC-sampling change:
everything cheap has been measured; upweighting family DEAD (both
doses collapse penalty), 3-route 3x3 now actively MISLEADS vs dev-10
(lw2: 45.40 3x3 vs 15.64 dev-10 — even short-route closed-loop is not
a selector; dev-10 is the minimum honest signal).
Binding constraint for DS 40+: the 104/220 in-game-budget timeouts
(plan slows/stalls in obstructed+dense states, state-level OOD).
PROBE LEDGER FINAL (2026-08-01): SEVEN consecutive post-220 probes
failed the dev-10 gate — v12/v14g (speed), gap1/gap2 (corridor),
lw/lw2 (upweight), cap 512x3 (20.40/68.7, 10/30: capacity HURT —
sharper copy-prior, timeouts 14). The champion config is a genuine
local optimum on every cheap axis; do NOT burn more compute on
single-axis config probes. Remaining levers are STRUCTURAL:
(1) imagination-DAgger / state-OOD robustness (Phase-4 idea — the
evidence points exactly here: plan quality in states the expert never
visits); (2) WM+steps scaling done jointly at final scale (Phase 3
proper, not the BC head alone); (3) Phase-2 divergence selector —
DONE 2026-08-01, VERDICT INVERTED: on-policy latent match ANTI-predicts
closed-loop (Spearman -0.60 all pairs / -0.47 within multi-only;
divergence +0.56) — the metric measures reward exploitation, not
driving; a first-class negative finding (runs/phase2_selector/,
scripts/phase2_selector.py — 17-model registry, reproducible join);
(4) paper drafting — STARTED: **paper/draft.md v0.1** is a complete
working draft (abstract -> limitations, four contributions, all real
numbers with src pointers, figure specs). Open decision for the
author: keep the multimodality title or re-scope to the full
what-transfers arc. Check `outputs/PIPELINE_STATUS.md` +
`squeue -u $USER` at session start (self-driving chains may be in
flight).

Benchmark rows banked (runs/bench220*, all 220 routes each):
| model | DS | completion | success |
|---|---|---|---|
| gen-1 kl01 multi (297 clips) | 11.47 | 53.5% | 18.2% |
| gen-2 10x multi | 21.49 | 58.9% | 23.6% |
| gen-2 10x **BC** | 20.56 | 69.1% | 34.1% |
| **gen-3 wph BC+rec** (wp head) | **22.10** | 68.7% | **39.1%** |
39.1% SR (gen-3 wph, 2026-07-31 220-run, runs/bench220_gen3wph)
exceeds all published Bench2Drive baselines by 6 points — VERIFIED
2026-07-31 vs official benchmark_v3 (DriveAdapter 33.08, ThinkTwice
31.23, TCP-traj 30.00; AD-MLP SR is 0.00, a prior note misread it);
DS trails sensor SOTA (~60, all with expert-feature distillation;
TCP-traj w/o distillation: 49.30/20.45).

Phase-0 verdicts (2026-07-30/31, all committed):
- 0a: **BC beats DITTO-multi closed-loop at scale** (3 seeds + dev-10:
  27.78/70.1% vs 22.51/64.2%) — BC is the deployed policy; the
  when-does-imagination-matching-help question is a paper contribution
  (Phase-1 highway says multimodal data; B2D at scale says BC).
- 0b: agent stopped 84% of ticks vs expert 32% (cruise speed fine) —
  stop-frequency is the DS bottleneck; brake-threshold probes NEGATIVE
  (policy-intent, not flicker). runs/carla_smoke/SPEED_AUDIT.md.
- 0c: **DATA LANDMINE FIXED** — anno top-level x/y is GNSS-noisy
  (0.6-2.1 m/frame); ego pose now from the physics-exact ego_vehicle
  box everywhere (obs + waypoints), npz cache key ##egobox1.
  scripts/waypoint_check.py certifies waypoint targets
  (WAYPOINT_CHECK_OK: phys .069, backward 0, lateral .33 m).
  Frame fact: in the compass frame FORWARD = -y, LATERAL = x.
- 0d: **route-PID oracle scores 100.00 smoke / 94.00 dev-10 (30/30
  routes, incl the wedge route)** — waypoint-tracking ceiling ==
  benchmark max; RoutePIDDriver in ditto_av/carla_agent.py
  (configs/diag_route_pid.yaml), also the paper's privileged
  rule-based reference row. GO for Phase 1.
- gen3_clean (cleaned-obs BC retrain, runs/b2d_gen3_clean): smoke
  25.36/82.7% (up from 22.12/65.8%); dev-10 ~19.2/75.3%, 16/30 full —
  completion up, penalty mixed (single-seed noise). Cleanup's main
  role: certified poses for Phase-1 targets.

### Phase 1 implementation — LANDED 2026-07-31 (this commit)
All four items built and gated (59 tests green, WAYPOINT_CHECK_OK,
mini end-to-end train + deployment-tail run on 12 clips):
1. `env.action_space: waypoints` (action_dim 12 = wp_k 6 x 2); `wp`
   plumbed as the action via TrajectoryData(action_key="wp"); npz
   cache key extended ##wp6; Gaussian head bounds +-3.0 (nets.WP_BOUND,
   measured range: fwd max 2.04 scaled).
2. WM/BC/DITTO train on wp actions unchanged (configs/b2d_gen3_wp.yaml
   = gen3_clean recipe, only the action swapped).
3. Deployment: carla_agent.wp_to_vehicle (compass->vehicle = fixed
   +90deg rot; fwd=-y settled) + WaypointTracker (pure pursuit on
   predicted points, target speed from their spacing, curvature cap;
   0d-proven gains; config-gated creep, default off) wired into
   DittoCarlaAgent (configs/diag_gen3_wp_bc.yaml).
4. tests/test_waypoints.py incl the offline-wp -> wp_to_vehicle ->
   world round-trip identity on a curved trajectory.
**v1 VERDICT (2026-07-31, jobs 10556343/4/5): wp-BC LOSES.**
3x3 11.99/58.4%/0.301 (vs gen3_clean control-BC 25.36/82.7%/0.297);
dev-10 ~19.3/60.3% (vs ~19.2/75.3%). Same penalty, far less completion.
Diagnosis (runs/carla_smoke tick logs, agent_ticks_gen3_wp_3x3.jsonl):
- NOT inert (the plan-copy fear): plans command v_wp~3.4 mean, agent
  commits to lateral maneuvers. Failure is the opposite: plan DRIFT
  (wp6 lateral -4.7 -> -10 m over 40 ticks; action-channel covariate
  shift — the 12-dim prev-action feeds back) -> off-lane ->
  collisions_layout -> terminal POWER-WEDGE (throttle 0.75 at
  standstill for 100s of ticks; wp mode had NO recovery). Plus steer
  oscillation 8-13 sign-flips/100 ticks (fresh per-tick plans jitter).
- Steer-sign/frame chain PROVEN clean (round-trip test + route-PID
  geometry identity) — do not re-litigate the frame.
Probe round VERDICT (2026-07-31, jobs 10556814/5; 3x3 each):
| variant | score | compl | pen | full |
|---|---|---|---|---|
| wp v1 | 11.99 | 58.4 | 0.301 | 2/9 |
| wp + reverse recovery | **27.93** | 61.9 | 0.401 | 4/9 |
| wp + smooth tracker | 18.88 | 54.8 | 0.374 | 2/9 |
| wp + smooth + rec | 13.39 | 59.7 | 0.293 | 3/9 |
| wp ditto_multi | 7.34 | 25.1 | 0.412 | 0/9 |
- rec posts the best composed 3x3 banked in the project (beats
  gen3_clean 25.36) — the power-wedge diagnosis was the right target.
  Note recovery was config-dead in wp mode v1; levers are per-mode.
- smooth+rec < rec alone: interactions non-additive AGAIN (settled
  lesson holds); smoothing alone mildly positive but not stacking.
- multi wp head collapses (7.34/25.1): BC>multi extends to waypoint
  actions, stronger than for control actions.
- completion still 62 vs control-BC 82.7 — plan drift remains the
  binding constraint; if dev-10 confirms, the next training-side lever
  list: shorter wp horizon (k=4), prev-action noise augmentation.
dev-10 on rec VERDICT (jobs 10557053/4): **GATE FAILED — no 220.**
wp+rec 16.37/69.0%/0.251 (12/30 full) vs v1 19.27/60.3 (6/30) vs
gen3_clean control-BC ~19.2/75.3 (16/30) vs gen2_10x BC 27.78/70.1.
Recovery is a real, consistent lever (completion +8.7, penalty -0.06,
full routes x2 — keep stuck_recovery ON for wp-family deployments; the
3x3's 27.93 composed was short-route noise). But wp-as-WM-action loses
to control actions, period. Root cause stands: action-channel drift.

### v2 in progress — gen3_wph: waypoint HEAD, control-action WM
(the TCP-traj architecture in our stack; started 2026-07-31)
WM + latent features stay exactly gen3_clean (control prev-actions —
REUSE its world_model.pt to pin the 82.7%-completion features); only
the BC head changes: regresses wp labels (12-dim, bounds +-3) from the
same features; deployment tracks predicted wp with WaypointTracker +
reverse recovery, and feeds the EXECUTED control (tracker output) back
into the WM (training-consistent). DITTO heads skipped in this mode
(12-dim head cannot drive a 3-dim-action dream; BC>multi settled for
both action types anyway). env config: action_space continuous +
wp_head true; same npz cache as wp mode (a8074aa357d4 — has both
action and wp). Gates: pytest + mini e2e, 3x3 (plain + rec), dev-10,
220 only if it beats gen2_10x BC dev-10.
**3x3 VERDICT (2026-07-31, train 10557266, eval 10557267):
BREAKTHROUGH.** plain 48.98/73.5%/pen 0.542 (6/9 full) — nearly 2x
the best composed 3x3 ever banked; rec 33.16/86.1%/0.341 (7/9, best
completion banked). The output-parameterization hypothesis (TCP-traj
lesson) confirmed once the action-channel drift was removed; penalty
0.542 = far cleaner driving. plain-vs-rec trades penalty vs
completion; dev-10 on BOTH = jobs 10557439/40 (A/B splits). 220 gate:
beat gen2_10x BC dev-10 27.78/70.1.
**dev-10 VERDICT: GATE PASSED.** wph+rec 30.49/83.2%/0.335 (20/30
full - record) beats the champion 27.78/70.1; plain 29.69/67.6 (13/30,
29 runs - one rep lost, winner unaffected). Reverse recovery earns its
keep at dev-10 scale in wph too (completion +15.6 over plain).
**FINAL 220 LAUNCHED on gen3_wph_bc_rec** (2026-07-31 ~17:30): 19
chunks wph220_00..18 = jobs 10557555-10557573 (participation/visual
alternating), collector 10557574 -> runs/bench220_gen3wph (self-
committing). Compare against gen2_10x rows (21.49 multi / 20.56 BC,
SR 34.1%) and the verified benchmark_v3 baselines.
**220 FINAL (landed 2026-07-31 21:02, commit b343997): DS 22.10,
completion 68.7%, penalty 0.308, SR 39.1% (86/220) — NEW RECORDS on
DS and SR (previous SR record 34.1; DriveAdapter 33.08).**
Failure profile SHIFTED (2026-08-01 analysis): wedging nearly
eliminated (9 blocked vs 136 in gen-1 era); 104/220 routes now die at
the ~200 s in-game budget ("Failed - TickRuntime", mean completion
47.6% at cutoff; completed routes take 65 s) with min_speed 2750.
SPEED is the binding constraint for completion AND penalty.
DEPLOYMENT-LEVER LEDGER (2026-08-01, all dev-10, gate = rec
30.49/83.2/0.335 20-30 full; verdicts committed):
- v_max 12 / 14+gain: NEGATIVE (21.55/75.3; cap engaged 16% of ticks
  but mean speed unchanged 1.4 m/s — standstill time dominates).
- Tick audit: 41% of ALL ticks = plan-says-GO-but-static (queuing only
  2.2%); recovery active 12.5%; ~21 obstruction stretches/route. The
  car grinds into leads/obstacles (no gap logic) and sits in (obs,
  action) states the expert never produced — where the BC plan is OOD.
- lead_gap (RoutePID corridor cap): collisions halved 44->19, penalty
  0.449, timeouts 5 — but blocked 0->10 (brake-hold starved the
  throttle>0 recovery trigger); completion 65.4. GATE FAILED.
- gap2 (+recovery sees plan intent during hold): 31.46/64.9/0.470
  (13/30) — DS edges the champion, SR collapses (blocked 12: the plan
  never commits the bypass from standstill even with room). NOT a
  champion replacement; KEEP as the paper's controller-safety ablation.
CONCLUSION (echoes the control-era settled lesson): deployment levers
are exhausted; the binding constraint is PLAN QUALITY IN
STOPPED/OBSTRUCTED (OOD) STATES — training-side.
TRAINING-SIDE ROUND (2026-08-01, all dev-10):
- lw@4/2 (launch 4x + maneuver 2x upweight): 19.72/78.4/0.258 — fixed
  launches (blocked 1/30) but penalty collapsed. GATE FAILED.
- lw2 (launch 2x only): 15.64/75.2/0.185 — WORSE penalty at the lower
  dose; family DEAD. (Its 3x3 said 45.40 — 3x3 is no longer a
  selector; another open!=closed instance for the paper.)
- Phase-5 seeds of the champion (full-pipeline): s0 30.49/83.2 (20/30)
  s1 25.86/73.0 (16/30) s2 28.45/70.6 (12/30) — mean ~28.3 +- 2.3,
  all >= gen2 champion band; deploy seed 0.
Champion for the paper remains wph rec (220: 22.10/39.1%).

## Goal (do not lose sight of this)

A strong paper for a top ML venue (CoRL/ICRA or NeurIPS/ICLR):
multimodal latent-matching imitation, fully offline, no simulator in
the training loop. Two evidence pillars:
1. Phase-1 controlled study — DONE, paper-grade (PAPER_PLAN.md).
2. Bench2Drive closed-loop — make the numbers as strong as possible
   (no compromises), positioned honestly: privileged-input planner
   trained offline; SOTA target is the fully-offline/no-simulator
   class, NOT sensor-based UniAD/VAD parity. The methodological
   findings (anchor dose-response, open-loop != closed-loop x5,
   K-transfer failure) are first-class contributions.

## Plan

### GEN-3: the SOTA program (adopted 2026-07-30; supersedes gen-2 plan — gen-2 is DONE, DS 21.49)

Target: driving score 40+ (solid), 60+ (= published SOTA class:
TCP-traj 59.9, ThinkTwice 62.4, DriveAdapter 64.2 — all imitate the
same Think2Drive expert, through cameras; we use privileged state, so
frame claims honestly). Ceiling argument: the expert scores ~90%
success; our gap is imitation quality, not a ceiling.

**Phase 0 — correctness & triage review (gates everything; IN PROGRESS)**
- 0a Policy-objective triage: BC beat multi closed-loop at gen-2 scale
  (22.12/65.8% vs 16.80/48.1% smoke)! Confirm: BC on seeds s1/s2 +
  BC dev-10 vs multi dev-10. If BC holds, the deployed policy AND the
  anchor story need rethink (paper finding either way: when does
  imagination matching help? Phase-1 says multimodal data; B2D may be
  effectively unimodal per-state — measure with the multimodality
  probe from Phase 1 on B2D latents).
- 0b Speed audit: 2309 min-speed infractions on the 220 — is the
  expert also slow (data property) or is it policy shrinkage? Compare
  expert clip speed profiles vs agent tick logs on same route types.
- 0c Waypoint-target harness BEFORE training on them: extend
  replay_frame_check-style exact test to future-pose extraction
  (frame convention pi/2 again — never trust, always replay-test).
- 0d Control-ceiling test: PID-tracking GROUND-TRUTH future waypoints
  (privileged oracle) on the 3 smoke routes + dev-10. This bounds the
  waypoint abstraction: expect near-expert scores; go/no-go for
  Phase 1. Also calibrates the PID gains offline of any learning.
- 0e Protocol audit: published baselines run the SENSORS track; we run
  MAP (privileged). Document precisely; never claim parity without
  the caveat. Verify weather/scenario settings match the official
  eval (leaderboard defaults).
- 0f Standing regressions: pytest + replay_frame_check after ANY
  featurizer/controller change; 3-route 3x3 as canary; >=3 reps
  always; multi-session claims via PIPELINE_STATUS.

**Phase 1 — waypoint action abstraction + PID tracker (the big lever)**
[entry: 0d passes]
Redefine action = future ego-frame waypoints (from anno ego poses; the
data already contains them). WM learns dynamics under that abstraction;
BC/DITTO objectives unchanged (they live in latent space). Deployment:
PID tracks predicted waypoints (port a Bench2Drive team-code PID);
creep-when-blocked in the controller (standard practice, config-gated).
Directly attacks all three top failure modes: min-speed (2309 events),
collisions (~356), wedge (136 blocked routes). This is the TCP-traj
lesson (+~10 DS from output parameterization alone in their ablations).

**Phase 2 — in-model on-policy divergence as the model selector**
From the original DITTO paper: on-policy latent divergence in the WM
predicts true return where action-MAE does not. Validate on our ~12
configs with known closed-loop numbers; if it ranks them correctly,
use it to search hyperparameters 10x cheaper (then confirm top-k with
3x3s). Paper-positive either way (extends the open!=closed story).

**Phase 3 — scale (entry: Phases 1-2 stable)**
Capacity sweep on H100 (nets are tiny; trainings are 39 min);
ability-weighted sampling (upweight the scenario families of the 136
blocked routes); lights re-integration; anchor/objective re-tune at
final scale — all selected via Phase-2 metric + 3x3 confirmation.

**Phase 4 — wedge-directed (if blocked-rate still dominates)**
Controller creep tuning; retrieval-conditioned commitment;
imagination-DAgger (perturbed-start WM rollouts relabeled via
retrieval) — one axis at a time, closed-loop selected.

**Phase 5 — final protocol (no cherry-picking)**
3 seeds of the final config; dev-10 select; ONE honest 220 run
(+2 more reps if budget allows for variance); per-ability error
analysis; full comparison table with track caveats.

### Paper (start in parallel, now)
9. Method + Phase-1 sections are fully evidenced — draft them.
   Closed-loop section: honest privileged-offline framing; the
   dose-response figure; open!=closed as a finding. Theory sketch in
   PAPER_PLAN. The paper/ dir holds the original DITTO paper (TMLR) as
   reference only — our draft is greenfield.

## Settled facts — do NOT redo or re-litigate
- **Frame**: anno theta = CARLA yaw + pi/2 (compass). Deployment MUST
  use yaw_offset pi/2. Proven by offline replay (scripts/
  replay_frame_check.py, diff ~1e-4). Road A/Bs can NEVER rank frame
  conventions. All pre-fix closed-loop numbers are void.
- **Route conditioning**: RouteCursor ports the collector's exact
  near/far semantics (dense-plan pop ~4 m / command-node pop ~7.5 m).
- **Deployment levers are dead**: stochastic sampling, action gains,
  aggressive AND conservative stuck-recovery all measured worse or
  neutral. Progress comes from training-side changes only.
- **Anchor**: bc_kl 0.1 is the peak (64.6% vs 22.5% baseline). K=16
  hurts closed-loop despite Phase-1 (39.4%). v5 lights + weak anchor
  re-freezes (27.5%) — interactions are non-additive; select closed-loop.
- **Open-loop != closed-loop, 5 instances** — never select a model on
  open-loop metrics; 3-route 3x3 smoke is the cheapest honest signal;
  >=3 reps always (run variance is huge).
- **Brake binarized** at deployment (Gaussian mean rides the brake).
- **AdditionalMaps**: apptainer dir-overlay (maps_overlay/), conditional
  in the CarlaUE4.sh shim (mirror: scripts/patches/CarlaUE4_shim.sh);
  rebuild via scripts/slurm/extract_maps.sbatch. Bind CANNOT merge into
  existing SIF dirs.
- **npz cache** keys on clip split + obs layout — extend the key if the
  obs change again.
- Bench2Drive harness quirks (patched clone on scratch; archive in
  scripts/patches/): py3.10 getchildren fix; '+save_name' appended to
  agent-config (agent strips it); cwd must be Bench2Drive/;
  routes-subset takes route IDs.

## Session-start checklist
1. `cat /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md`
2. `squeue -u $USER`; DELFTBLUE motd inode check.
3. `git -C /scratch/$USER/ditto_av/DITTO_AV status` — push anything the
   deciders committed; pull the home clone if training will run there.
4. Download log tail; pilot results in ~/ditto_out/b2d_kl01_{5,20}x.
5. wandb sync loop: restart after login-node reboots if dashboards
   wanted (scripts/wandb_sync.sh).
