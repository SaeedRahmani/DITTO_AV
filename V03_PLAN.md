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
pure ego-state-matching reward; 220-route DS ~76; RL > same-net BC
+12 dev-10) but its world is NON-REACTIVE: replayed traffic ignores the
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

## 5. Phase D — REACTIVE EGOSIM + POLICY (V3-M3)

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
  policy; in-sim eval in BOTH worlds every N steps (divergence between
  the two eval verdicts is itself a diagnostic).

## 6. Phase E — EXTERNAL GATES + SCIENCE (V3-M4)

- E1 **W3, the only verdicts that count**: 3x3 canary -> dev-10 vs the
  v0.2 champion band (83.60-85.63; seed bars from main's G5) -> ONE
  220 only for a clear dev-10 winner. The learned world NEVER grades
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
