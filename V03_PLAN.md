# V03_PLAN.md — v0.3: a learned, reactive world model (branch saeed/v0.3)

Decided with the author 2026-08-04. v0.2 continues on `main` (other
sessions); v0.1 frozen on `saeed/ver0.1`. Work ONLY in the worktree
`/scratch/$USER/ditto_av/DITTO_AV_v03` — NEVER check saeed/v0.3 out in
the main clone or the home clone (live v0.2 jobs read those trees).

## 0. Prime directive: measure, don't assume

Every design fact below is either (a) verified by a committed audit, or
(b) carries an explicit gate that falsifies it. v0.2's record shows the
method: two "obvious" integrator assumptions failed only on REAL data;
all four generations of v0.1 died from one unmeasured assumption about
what a latent encodes.

## 1. Goal

v0.2 proved the mechanism (on-policy imitation in a replayed world,
pure ego-state-matching reward; full 220-route DS ~76; DITTO-AV v0.2 > same-net BC
+12 test-10) but its world is NON-REACTIVE: replayed traffic ignores the
ego. v0.3 learns the ONE thing replay cannot provide — reactions:

- Keep the factorization. Ego = analytic kinematics (never learn what
  you know). Learn a TRAFFIC MODEL: per-agent token transformer
  predicting other agents' motion, conditioned on agent histories AND
  the ego's actual current state, rolled out autoregressively to make
  the egosim REACTIVE.
- Policy recipe stays v0.2's (TokenPolicy, seq-BC init, A2C, tight
  ego-state-matching kernel — shaping proven redundant at 220 scale).
  v0.3's policy training = curriculum fine-tune of the v0.2 champion.

## 2. Phase A — ANALYSIS (all findings committed to runs/v03_audit/)

- **A1 trackability audit** (the assumption-killer): on real extracted
  clips measure — actor-ID stability across frames (the anno `id`
  field; v0.1's velocity finite-differencing relies on it but nobody
  quantified it), track-length distribution, churn (enter/exit rates
  of the annotation radius), per-class counts (vehicle/walker/bicycle),
  slot-overflow rate (>32 actors), sampling-rate regularity, and
  position-noise floor (per-track jerk). GATE: ids stable within
  clips and median track length >= 3 s, else the model design changes
  (association layer needed).
- **A2 dynamics floor**: constant-velocity and constant-turn-rate
  predictors on held-out clips — ADE/FDE at 1/2/4 s. These floors set
  the W0 thresholds (no arbitrary numbers): the learned model must
  beat CV at 4 s by a pre-registered margin (>=20% rollout ADE) or it
  adds nothing over replay+extrapolation.
- **A3 interaction evidence**: quantify ego-dependence of traffic —
  lead-gap response statistics (follower decel vs gap/closing-speed),
  yielding events near the ego path. Sets an upper bound on what
  "reactive" can even learn from this data; if interactions are rare,
  the reactivity dividend is small and we should know BEFORE building.
- **A4 light context inventory**: what per-agent signal exists for
  OTHER agents' stops (annos store all traffic_light boxes + states;
  our arrays keep only the ego-affecting one). Decide evidence-based:
  scene-level light tokens vs per-light tokens vs history-only.
  Documented decision, not a silent assumption.

## 3. Phase B — DATA (V3-M1)

- B1: additive adapter extension: `act_id` (stable per-slot actor id)
  + `act_cls` (class) arrays; new cache key suffix `##glob2`. Shared
  files touched additively only (merge safety with main).
- B2: track-view utilities: slots -> ID-associated tracks; unit tests
  against synthetic clips with known churn/AND the A1 audit numbers.
- B3: caches: 12-clip mini (login) for tests; 999-clip rebuild via the
  /tmp extraction job. Scratch inode rules as always.

## 4. Phase C — TRAFFIC MODEL (V3-M2)

- C1 architecture: tokens = agents (history-encoded, class-embedded,
  agent-centric state encoding) + ego token (input-only) + light/route
  context per A4; transformer trunk (~10-30M, config-scaled); heads
  predict per-agent next-step delta distributions (Gaussian first, GMM
  if A2 shows multimodality matters). Autoregressive rollout API
  mirroring EgoSim's step contract.
- C2 training: teacher-forced NLL, then rollout fine-tuning (closed-
  loop training against compounding error — measure both variants; the
  open!=closed lesson applies to the WORLD MODEL too).
- C3 ensemble: K=4 (seed/bootstrap); calibrate a disagreement metric
  on held-out data (percentile thresholds, committed).
- C4 **W0 fidelity gate** (pre-registered): held-out rollout ADE @4s
  >=20% better than CV; agent-agent collision rate within 2x of the
  log's; speed/accel distribution overlap (W1-distance bands);
  **reactivity probe**: inject a braking ego in front of followers —
  modeled followers must decelerate (CV ghosts do not); report effect
  size. FAIL any -> fix the model, never lower the gate.
- C5 lever ladder on W0 FAIL (ordered, one at a time; amended
  2026-08-04 BEFORE round-2 results, with round-1 evidence only):
  (1) rollout fine-tuning [round-2, running]; (2) error decomposition
  by regime (stopped/launching vs cruising vs turning) to pick between
  (3a) multimodal heads (GMM) — 4 s futures at junctions genuinely
  branch and a Gaussian mean averages modes (the SAME point-metric
  lesson as v0.1); (3b) per-light scene context (A4) — other agents'
  stops are unexplained by current inputs; (4) capacity/history.
  PRE-REGISTERED metric note: mean point-ADE punishes correct-but-
  different modes; if collision realism + reactivity + calibration
  pass while mean-ADE fails AND the error decomposition shows
  branching dominates, the gate criterion may be refined to
  minADE-over-modes (standard sim-agents practice) — refinement must
  be justified by the decomposition, never by the score alone.

## 5. Phase D — REACTIVE EGOSIM + POLICY (V3-M3)

Schedule amendment 2026-08-04: D1/D2 are pure code + tests — build
them IN PARALLEL with the W0 iterations (only D3 training requires a
W0-passing model). Serializing them wastes wall-clock.

- D1: EgoSim reactive mode (additive code path): traffic from model
  rollout (initialized from logged history) instead of log indexing;
  obs builder unchanged; reward unchanged (ego-state matching vs the
  same-scene expert — still valid under moderate divergence);
  **W1 pessimism**: terminate/penalize rollouts where ensemble
  disagreement crosses the calibrated threshold — the structural
  anti-exploitation mechanism v0.1 lacked.
- D2: sim-level fidelity: ego replaying expert actions in the REACTIVE
  sim must reproduce logged trajectories statistically (the G0
  analogue, on both synthetic and real clips; regression-tested).
- D3: **W2 curriculum**: init from the v0.2 champion; anneal
  replay->reactive batch ratio; KL anchor to the replay-trained
  policy; in-WM eval in BOTH worlds every N steps (divergence between
  the two eval verdicts is itself a diagnostic).

## 6. Phase E — EXTERNAL GATES + SCIENCE (V3-M4)

- E1 **W3, the only verdicts that count**: 3x3 canary -> test-10 vs the
  v0.2 champion band (83.60-85.63; seed bars from main's G5) -> ONE
  220 only for a clear test-10 winner. The learned world NEVER grades
  itself (most-replicated lesson of the whole project).
- E2 reactivity-dividend analysis: per-scenario-family deltas
  (interaction-heavy families should move most; divergent-start
  recovery should improve; ghost-gap collisions should drop).
- E3 full-circle experiment: factored-latent DITTO retest — latent
  matching inside the (ego | learned-traffic) factorization; closes
  the v0.1 question scientifically regardless of outcome.
- E4 paper: v0.1 (why latent matching fails) -> v0.2 (replay world +
  state matching suffices) -> v0.3 (when and why reactions pay).

## 7. REVIEW protocol (continuous, not a phase)

- Tests-first per milestone (unit + fidelity regression; the synthetic
  clips get churn/spawn cases); pre-registered gate numbers committed
  BEFORE the runs they judge; every result lands in the ledger with
  the job id; code-review + simplification pass at the end of C and D
  before anything merges toward main; PIPELINE_STATUS claims tagged
  "V03:"; all run/result names prefixed v03_; MIG budget shared with
  main (1 job/user TOTAL).

## 8. Honest novelty position (searched 2026-08-04)

Closed-loop replay training exists (Urban Driver CoRL'21; BC-SAC'22;
Waymax/nuPlan/GPUDrive infra); learned sim-agents exist (WOMD Sim
Agents line). Ours that holds: pure state-matching reward sufficiency
(shaping-redundancy ablation), the measured latent-failure diagnosis,
Bench2Drive-220 evidence; v0.3 adds gated reactive imagination with
ensemble pessimism + the factored-latent DITTO retest. READ BEFORE
PRIORITY CLAIMS: arXiv 2512.18662 (Dec 2025).

## 9. Status ledger (newest first)

- 2026-08-05 — **W3 test-10 verdict (clp_rx): DS 82.53/100% 30/30 —
  gate NOT cleared (band 83.60-85.63, sigma 2.3: statistical tie with
  pure, nominally below shaped). THE REACTIVITY DIVIDEND IS CONFIRMED:
  vehicle collisions 6 = best of any model ever (shaped 9, pure 12,
  -40..50%) — exactly the interaction-failure class reactive training
  targets. The offset: collisions_layout 7 (vs 1/0) concentrated on
  the urban B-half (75.40 vs 81.60) — STRUCTURAL cause: the training
  world models/replays only VEHICLES; static layout (walls, poles)
  does not exist in the state, so boldness near obstacles is free in
  training and punished in CARLA. E2 delivered. NEXT DECISION: (a)
  bank v0.3 as-is (dividend confirmed + methodology + the layout-gap
  finding = complete honest chapter) or (b) v0.3.1: add static
  layout/map geometry to the world state (new data lane: OpenDRIVE
  extraction) — the same no-map limitation that has surfaced at every
  version now binds the TRAINING WORLD itself.

- 2026-08-05 — **W0 PASSED (pre-registered refined criterion), round
  4 of 4**: minADE-over-modes@4s 2.79 <= 3.80, proximity 1.36x <= 2x,
  reactivity 0.567 > 0.3 (from 0.06 — the round-4 EGO-HISTORY fix:
  a memoryless-per-step model cannot distinguish braking from slow,
  so reactions were unlearnable; + interaction-weighted loss). Mean
  point-ADE 4.55 (best of 4 rounds, below the CV floor 5.02) remains
  above 3.80 solely in branching regimes (stopped 0.23 vs
  launch/cruise/turn 7-10) — the refinement's justification, per the
  rule committed before round-2. W1 calibration: disagreement p95
  0.021 / p99 0.223 (on-distribution)  -> termination threshold 0.25.
  Campaign findings banked for the paper: (1) naive rollout FT of a
  conditional WM destroys its conditioning; (2) ego-history context
  is necessary for learnable reactions; (3) regime decomposition as
  lever-picker; (4) mode heads + minADE for branching futures.
  NEXT: D3 — champion fine-tune in ReactiveEgoSim (W1 threshold 0.25,
  replay->reactive curriculum), then W3 CARLA gates.

- 2026-08-04 night — W0 round-2 verdict + round-3 design: rollout
  fine-tuning MOVED ADE 7.62 -> 5.00 @4s (1-2 s now at CV parity) but
  (a) plateaus at extrapolation level and (b) COLLAPSED ego-reactivity
  0.21 -> 0.04 m/s — finding: naive rollout fine-tuning of a
  conditional world model destroys the conditioning (logged-ego
  pairing makes ego-ignoring optimal). Regime decomposition (the
  lever picker): stopped 1.48 (n=493) EXCELLENT vs launching 6.97 /
  cruising 7.85 / turning 9.41 — error mass = FUTURE BEHAVIOR CHANGES
  = multimodal, as pre-registered. ROUND-3 (running): GMM mode heads
  (WTA + mode-CE loss, argmax-mode rollouts) + JOINT objective
  (teacher NLL term restores conditioning) + minADE-over-modes
  reported per the pre-registered refinement rule (decomposition
  justifies it). Primary mean-ADE gate UNCHANGED.

- 2026-08-04 eve: W0 round-1 verdicts: (a) first FAIL was an EVAL BUG
  (slot-vs-id matching produced a 91 m ADE artifact; fixed, windows
  cache v2); (b) corrected verdict = HONEST FAIL: teacher-forced model
  rolls out at ADE 1.35/3.58/7.62 m @1/2/4 s — WORSE than the CV floor
  (0.84/2.12/5.02): the open!=closed lesson hit the world model, as
  pre-registered. Proximity realism PASSES (1.2x log); reactivity
  positive but under gate (0.21 m/s mean, skewed). LEVER APPLIED:
  differentiable K-step rollout fine-tuning vs ID-matched logged
  positions (smoke: toy ADE@4s 3.75 -> 3.20 with 60 steps). Round-2
  job fine-tunes all 4 seeds + re-gates. Gate numbers UNCHANGED.

- 2026-08-04: V3-M2 core landed: TrafficModel (ego-conditioned agent
  transformer, invariant local features + ego-relative geometry,
  Gaussian local-delta heads, rollout step API), build_scene_windows
  (id-associated histories; caught+fixed a yaw/vy column bug via the
  synthetic gates), 7 v0.3 tests green (tracks churn + featurize
  invariance + overfit sanity). NEXT: W0 harness + ensemble training
  on the 999 ##glob2 cache (job 10576942), gate ADE@4s <= 3.80 m.
- 2026-08-04: V3-M1 DONE: act_id/act_cls (##glob2, deterministic
  actor keys — string-id fixture assumption caught by tests),
  tracks.py re-association + windows; 88 tests green; real-clip
  validation consistent with the audit.

- 2026-08-04: V3-A1/A2/A3 AUDIT DONE (runs/v03_audit, 24 clips, 6.4k
  frames, 83k actor-observations): IDs STABLE (1.8% gapped tracks,
  teleports 0.02%), median track 12 s, 84% >= 3 s -> A1 GATE PASSED,
  no association layer needed. Slot layout holds (p99 = 30 actors,
  zero overflow of 32). Dynamics floors: CV ADE 0.84/2.12/5.02 m @
  1/2/4 s, CTRV 4.75 @4s -> W0 gate pre-registered: learned rollout
  ADE@4s <= 3.80 m (>=20% under CTRV). Interaction signal STRONG:
  3,445 follower-behind-ego pairs, decel -2.40 m/s^2 when close vs
  -0.13 far -> reactions are in the data; the reactivity dividend is
  learnable. CAVEAT for B1: this 24-clip sample contains class
  "vehicle" ONLY — verify walker/bicycle class strings on
  pedestrian-scenario families before freezing act_cls.

- 2026-08-04: V3-M0 — branch + worktree + this plan; Phase A audit
  launched.
