# One Expert Is Not Enough: Multimodal Latent Matching for Offline World-Model Imitation in Autonomous Driving

*Version 0.1 — COMPLETE first draft (2026-08-02). Every number is banked
in the repo (runs/) with inline `[src: ...]` pointers; all four figures
are generated reproducibly by `scripts/paper_figures.py` into
paper/figures/ (PDF + PNG). Venue target: CoRL / NeurIPS.
OPEN ITEMS FOR v0.2 (author decisions): (1) title framing — keep the
multimodality thesis or re-scope to the full what-transfers arc; (2)
LaTeX conversion once the venue template is chosen; (3) related-work
citations need full bibliography entries; (4) optional: a joint
WM+data scaling run could strengthen the headline row.*

## Abstract

Learning driving policies entirely offline — no simulator interaction
during training — is attractive wherever online rollouts are expensive or
unsafe. We study DITTO-style on-policy imitation inside a learned world
model, where the policy is trained in imagination to match expert latent
trajectories, and extend it to driving along three axes. (1) We identify
expert **multimodality** as a failure mode of single-window latent
matching and fix it with retrieval-based nearest-mode matching; in a
controlled highway study the multimodal objective is near-expert
in-distribution and degrades least under distribution shift, and the gain
appears when and only when the demonstrations are multimodal. (2) On the
closed-loop Bench2Drive benchmark (CARLA leaderboard 2.0, 220 routes) we
scale a fully offline, privileged-input planner to a **39.1% success
rate — above every published Bench2Drive baseline** (DriveAdapter 33.08)
— driven by a waypoint **output parameterization**: predicting short
ego-frame trajectories tracked by a classical controller. Crucially, the
same abstraction *fails* when waypoints are also used as the world
model's action channel: the policy's own plan feeds back, drifts, and
degrades driving — evidence that *where* a trajectory abstraction enters
the architecture matters. (3) Across 17 (model, policy) pairs with known
closed-loop scores, the in-model on-policy latent-matching metric
**anti-predicts** closed-loop driving (Spearman −0.60 overall, −0.47
within same-objective models): in a domain where exogenous traffic
dominates the latent, the metric measures reward exploitation rather
than driving quality. Together these results map where imagination-based
imitation helps, where behavior cloning suffices, and why offline
selection metrics — including in-model closed-loop proxies — cannot
replace closed-loop evaluation.

## 1 Introduction

Offline imitation with a learned world model promises the best of two
worlds: the corrective feedback of on-policy training (the policy visits
its own states, in imagination) without a simulator in the loop. DITTO
[DeMoss et al., 2023] instantiates this idea in Atari: roll the policy
inside a recurrent state-space model from expert start states and reward
latent similarity to the expert's continuation. We ask what happens when
this recipe meets driving.

Driving stresses the recipe in specific ways. First, expert data is
**multimodal**: from near-identical traffic states the dataset contains
both "overtake" and "yield" continuations, and matching a single
time-aligned expert window punishes the other valid mode. Second, the
latent space is dominated by **exogenous traffic**: any plausible
rollout is latently similar to any other (cosine 0.85–0.92 in our world
models), so the raw matching reward has ~2% dynamic range and is easily
exploited [src: PAPER_PLAN method; measured policies beat expert-replay
raw reward while crashing 76–82% closed-loop]. Third, closed-loop
driving quality is notoriously decoupled from offline metrics; we find
this extends even to *in-model closed-loop* metrics.

Contributions:

1. **Nearest-mode latent matching.** Reward = max latent similarity over
   K retrieved expert windows with similar start latents, with
   contrastive negatives; K=1 with no negatives recovers exact DITTO. In
   a controlled two-style highway environment (3 full-pipeline seeds),
   DITTO-multi reaches near-expert return in-distribution (20.1 ± 0.9 vs
   expert 21.6) and degrades least under density shift (18.3 ± 1.5 vs
   BC 10.5 ± 2.2), with collision rate 0.10 ± 0.06 vs BC 0.49 ± 0.05. A
   unimodal control collapses the multi-vs-single gap to noise — the
   advantage appears when and only when the data is multimodal
   [src: runs/phase1/phase1_results.md, k16h5_seed*, unimodal_seed*].
2. **A fully offline, privileged-input planner at state-of-the-art
   success rate on Bench2Drive.** Trained purely from the released
   1000-clip corpus, evaluated on the official 220-route closed-loop
   protocol: driving score 22.10, route completion 68.7%, success rate
   **39.1%** — above all published baselines on SR (DriveAdapter 33.08,
   ThinkTwice 31.23, TCP-traj 30.00), with the privileged-input caveat
   stated plainly [src: runs/bench220_gen3wph; official benchmark_v3].
3. **Where the trajectory abstraction belongs.** Predicting future
   ego-frame waypoints tracked by a pure-pursuit PID (+ reverse
   recovery) is the single largest lever (+5.0 SR over direct control
   output). Using the *same* waypoints as the world model's action
   channel instead *degrades* driving (dev-10 16.4/69.0 vs 19.2/75.3
   for control actions): the 12-dim plan fed back as prev-action drifts
   — measured lateral drift −4.7 m → −10 m over 40 ticks — and carries
   the policy off-lane. Output head: yes; dynamics action: no.
4. **Selection metrics invert.** Across 17 (model, policy) pairs with
   banked closed-loop scores, on-policy latent match correlates
   *negatively* with closed-loop completion (−0.60; −0.47 within
   same-objective models), and its complement (divergence from the
   expert-replay ceiling) correlates *positively* (+0.56). The in-model
   metric measures how hard the policy optimized the matching reward —
   which, at benchmark scale, anti-predicts driving. Action MAE is flat
   within-objective (−0.20). Model selection must be closed-loop; even
   short-route closed-loop smoke tests misrank (we document a 45.4 vs
   15.6 composed-score disagreement between a 3-route smoke and the
   10-route dev set for the same checkpoint)
   [src: runs/phase2_selector/table.md].

## 2 Related work

| Work | What it does | Delta from us |
| --- | --- | --- |
| DITTO (DeMoss et al. 2023) | single-mode latent matching, Atari | multimodality, driving, vector WM, selection-metric analysis |
| NVIDIA covariate-shift WM (Popov et al., ICRA 2025) | latent WM + align-to-demo for AV, CARLA | single-demo alignment; simulator in training loop; ours retrieval-multimodal + fully offline |
| CoIRL-AD (ICML 2026) | IL+RL dual policy in latent WM, nuScenes | competition mechanism, open-loop-centric |
| WorldRFT (AAAI 2026) | WM planning + RL fine-tuning, NAVSIM | hand-designed metric reward, not expert latent matching |
| MILE (2022) | BC in latent WM space, CARLA | no on-policy imagination |
| Think2Drive (ECCV 2024) | WM RL with env reward, CARLA | online RL; also the expert that generated our training data |
| TCP (2022) | trajectory-vs-control output heads | our waypoint-head finding replicates their ablation offline, and localizes *where* the abstraction helps |

Positioning: our niche is **offline + on-policy-in-imagination +
multimodal expert matching**, no hand-designed reward, no simulator in
the training loop; on Bench2Drive we are a **privileged planner**
(ground-truth actor states + route, MAP track) and compare against
sensor-based baselines only with that caveat explicit (AD-MLP is the
closest input class; all starred baselines additionally distill expert
features — TCP-traj *without* distillation scores 49.30/20.45).

## 3 Method

### 3.1 World model and latent bank

RSSM with categorical latents over vectorized scene observations:
rows [ego, 6 nearest actors] × [presence, x, y, vx, vy, cos h, sin h]
in the ego frame, plus a 16-dim route-conditioning block (near/far
command points + one-hot commands), 65 dims total. Deterministic state
256, stochastic 16×16, trained on 999 Bench2Drive clips (833 train /
166 val, 208k frames at 10 Hz). Expert episodes are filtered through
the posterior to build a **latent bank** of (h, z) states and all
length-(H+1) windows (H=15, i.e. 1.5 s).

### 3.2 Policies

All policies act on posterior features (h, z). **Latent BC**: Gaussian
NLL on expert actions. **DITTO-single**: actor-critic in imagination,
reward = cosine similarity to the time-aligned expert window.
**DITTO-multi (ours)**: reward = max over K windows retrieved by
start-latent similarity, minus the mean over M random negative windows
(the contrastive term restores dynamic range against the exogenous
traffic background; M=0 collapses, M=16 plateaus). A trust region to
the BC policy on imagined states (bc_kl coefficient) anchors the RL;
its dose is the single most sensitive hyperparameter closed-loop
(Section 6.2). A trajectory-consistent variant (commit to one retrieved
window per rollout) performs identically to per-step max (highway,
3 seeds: ID return 20.67 ± 0.24 vs 20.67 ± 0.30) and is the cleaner
method to present.

### 3.3 Driving-specific output parameterization (gen-3)

The deployed policy head predicts **six future ego-frame waypoints**
(0.5 s stride, 3 s horizon) regressed from the same posterior features;
a classical tracker turns the plan into control: pure pursuit on the
predicted polyline, target speed from the plan's own point spacing,
lateral-acceleration curvature cap, and a reverse recovery behavior
when forward progress stalls. The world model's action channel remains
the 3-dim executed control (throttle/steer/brake) — at deployment the
tracker's output is fed back as prev-action, keeping the (obs, action)
stream in-distribution. Frame conventions are certified by round-trip
tests (offline waypoint construction → deployment conversion → world
pose identity) and a physics check against integrated speed
[src: scripts/waypoint_check.py, tests/test_waypoints.py].

## 4 Controlled study: multimodality (highway-env)

Setup: two-style scripted expert (aggressive overtakes / conservative
yields, 50/50), 300 expert + 100 noisy episodes; in-distribution and
density-shift evaluation; 3 full-pipeline seeds; identical WM,
features, and networks across objectives — the comparison isolates the
policy objective. Main table (improved config K=16, H=5):

| policy | ID return | ID collisions | shifted return | shifted collisions |
| --- | --- | --- | --- | --- |
| expert (oracle) | 21.6 | 0.00 | 20.5 | 0.04 |
| BC (latent) | 13.9 ± 0.7 | 0.49 ± 0.05 | 10.5 ± 2.2 | 0.63 ± 0.10 |
| DITTO-single | 18.0 ± 1.5 | 0.21 ± 0.10 | 13.6 ± 1.6 | 0.48 ± 0.07 |
| **DITTO-multi** | **20.1 ± 0.9** | **0.10 ± 0.06** | **18.3 ± 1.5** | **0.21 ± 0.11** |

Key ablations: contrastive negatives essential (raw reward collapses to
0.70/0.88 collisions); retrieval breadth K=16 best; short imagination
horizon (H=5) helps — model error compounding dominates beyond ~1 s;
style imbalance 25/75 *widens* the multi-vs-single gap (retrieval
substitutes the right mode exactly where the time-aligned window is the
minority); **unimodal control collapses the gap to noise** (−0.05 ±
0.84 shifted return difference) — the causal link between data
multimodality and the multi objective's advantage. Conditional
multimodality of the data is established directly: 57% of paired expert
rollouts from identical resets diverge; in the trained bank the top-16
retrieved windows (start cosine 0.979) come 36% from the opposite style
with 33% action disagreement [src: runs/phase1/multimodality_analysis.md].

*Figure 1: `paper/figures/fig1_highway.pdf` — return and collision
bars, ID vs shifted, three policies + expert reference (3 seeds).*
*Figure 2: `paper/figures/fig2_multimodality.pdf` — multi−single
shifted-return gap at style ratios 50/50, 25/75, 100/0; the gap
vanishes exactly when the data is unimodal.*

## 5 Bench2Drive: fully offline closed-loop driving at scale

Setup: official 220-route closed-loop protocol (CARLA leaderboard 2.0
metrics: Driving Score = route completion × infraction penalty; Success
Rate = routes fully completed without disqualifying infraction), MAP
track. Training data: the released 1000-clip base corpus (999 usable),
collected by Think2Drive; **no simulator interaction during training**.
Our observation uses privileged annotations (ground-truth actor boxes,
ego pose, route): we are a privileged planner and mark sensor-based
baselines (§) accordingly.

| model | DS | completion | SR |
|---|---|---|---|
| AD-MLP (privileged state input) | 18.05 | – | 0.00 |
| UniAD-Base § | 45.81 | – | 16.36 |
| VAD § | 42.35 | – | 15.00 |
| TCP-traj §* | 59.90 | – | 30.00 |
| TCP-traj w/o distillation § | 49.30 | – | 20.45 |
| ThinkTwice §* | 62.44 | – | 31.23 |
| DriveAdapter §* | 64.22 | – | 33.08 |
| ours gen-1: DITTO-multi, 297 clips | 11.47 | 53.5 | 18.2 |
| ours gen-2: DITTO-multi, 999 clips, 10× steps | 21.49 | 58.9 | 23.6 |
| ours gen-2: latent BC (same WM) | 20.56 | 69.1 | 34.1 |
| **ours gen-3: waypoint head + PID (BC)** | **22.10** | 68.7 | **39.1** |

§ sensor-based (camera) input; * expert-feature distillation.
[src: runs/bench220*, official benchmark_v3 table verified 2026-07-31.]

Seed robustness (Phase-5 protocol, 10-route dev set × 3 reps, 3
full-pipeline seeds of the final config): composed score 30.49 / 25.86 /
28.45 (mean 28.3 ± 2.3) [src: runs/carla_smoke/gen3_wph_era/].
A privileged rule-based reference (route-following PID on the
ground-truth plan with lead/light gating) scores 94–100 on the same
dev routes — waypoint *tracking* is not the bottleneck; plan *decision*
quality is [src: Phase-0d, configs/diag_route_pid.yaml].

## 6 Findings

### 6.1 Where the trajectory abstraction belongs

Two placements of the same 6-waypoint abstraction:

- **As the policy output** (world model keeps control actions): 3-route
  smoke completion 73.5→86.1% with penalty 0.34–0.54; dev-10 30.49/83.2;
  220-route SR +5.0 points over control output. The plan's point
  spacing doubles as a learned speed profile for the tracker.
- **As the world model's action channel**: closed-loop *degrades*
  (dev-10 16.37/69.0 vs 19.2/75.3 for the control-action twin on the
  same data). Mechanism, measured from deployment tick logs: the
  policy's previous plan is its own prev-action input; small execution
  deviations produce (obs, prev-plan) pairs unseen in training; the
  plan drifts laterally (−4.7 m → −10 m over 40 ticks) and carries the
  car off-lane into terminal wedges. An imagination-matching (multi)
  head on the waypoint action space collapses entirely (3-route
  completion 25.1%).

This localizes the TCP trajectory-head lesson: the benefit is the
output parameterization + classical tracking, and it can be *undone* by
letting the trajectory feed the dynamics model's action channel.

### 6.2 Closed-loop selection cannot be replaced

- **BC-anchor dose-response** (bc_kl coefficient, all else fixed):
  completion 26.4% (0.03) → **64.6% (0.1)** → 54.2 (0.15) → 46.9 (0.2)
  → 22.5% (0.3). Single clean peak; the most sensitive knob in the
  system, and invisible to every offline metric we measured.
- **Open-loop ≠ closed-loop, 7+ instances** (NLL ordering
  anti-predicts; MAE flat across a 3× completion spread; K=16 transfers
  from highway but fails closed-loop driving; lights×anchor
  non-additive; ...).
- **In-model on-policy metrics invert** (17 pairs): latent match −0.60
  vs closed-loop completion (−0.47 within same-objective models);
  divergence from the expert-replay ceiling +0.56. The metric ranks by
  reward exploitation, not driving. Action MAE only re-discovers
  "BC beats multi" across objectives (−0.90 on the dev-10 subset) and
  is flat within-objective (−0.20)
  [src: runs/phase2_selector/table.md].
  Notably, the original DITTO paper already documented this decoupling
  *in its own favor*: their expert-prediction-accuracy figure shows
  DITTO matching expert actions worst among baselines while achieving
  the best returns — action agreement is not performance. Our results
  are the same decoupling with the sign flipped by the domain: under
  an effectively unimodal expert with exogenous-dominated latents,
  the proxy improves (our gen-4 policies beat their own BC anchor on
  waypoint MAE, 0.045 vs 0.062) while behavior degrades. The
  practitioner's rule is symmetric: imitation proxies can under- or
  over-state control quality, and only closed-loop evaluation
  distinguishes the two regimes.
- **Even short-route closed-loop misranks**: a 3-route smoke and the
  10-route dev set disagree by 3× on composed score for the same
  checkpoint (45.4 vs 15.6). Our protocol: 10-route × 3-rep dev set as
  the minimum honest selection signal; the 220 only for finals.

*Figure 3: `paper/figures/fig3_selector_inversion.pdf` — latent match
vs closed-loop completion, 17 control-family points by objective.*

### 6.3 When does imagination matching help?

Highway (controlled, genuinely multimodal data): multi > single > BC on
every metric, causally tied to multimodality (§4). Bench2Drive at
scale: latent BC ≥ DITTO-multi closed-loop (dev-10 27.78/70.1 vs
22.51/64.2; 220 SR 34.1 vs 23.6), and the gap *widens* on the waypoint
action space.

We then built the strongest imagination-refinement variant our
evidence allowed (DITTO-WP): (i) *deployment-consistent imagination* —
dream rollouts step the world model through a batched port of the
deployment tracker (equivalence-pinned by randomized tests), with ego
speed decoded from the latent, so the imagined dynamics are exactly
the deployed stack; (ii) *task-projected matching* — rewards computed
after projecting h through a frozen ridge probe onto expert-waypoint
labels (R² 0.82), attacking the measured exogenous-latent dominance;
(iii) *retrieval-relabeled divergent starts* — offline DAgger in
imagination, with nearest-mode retrieval as the relabeler; (iv) the
benchmark-proven BC trust-region anchor. The result is a clean
five-point dose-response, every point below plain BC (10-route dev
set, BC = 30.49/83.2):

| variant | score / completion |
|---|---|
| anchor 0.1 + divergent starts | 3.46 / 50.4 |
| anchor 0.3 + divergent starts | 19.49 / 70.8 |
| anchor 0.3, no divergent | 24.07 / 80.8 |
| anchor 0.3, no divergent, early stop (3k) | 13.31 / 71.1 |
| anchor 1.0, no divergent | 18.08 / 60.5 |

Attribution: divergent starts are actively harmful (retrieval from
off-manifold latents fetches behaviorally wrong targets); the residual
deficit is the imagination pressure itself. The sharpest coda: in the
early-stop and strong-anchor runs the policy's *deterministic-mean*
waypoint error improved *below* the BC anchor's (0.045–0.051 vs
0.062) while closed-loop driving degraded — the objective moves the
policy in MAE-invisible directions that damage behavior, the
metric-inversion of §6.2 reproduced inside the training loop itself.

Conclusion: imagination matching is a tool for multimodal
demonstration corpora; under an effectively unimodal expert at
benchmark scale it is dose-invariantly harmful even with
deployment-consistent dynamics, task-projected rewards, and anchoring
— a boundary we could only locate with closed-loop evaluation.

*Figure 4: `paper/figures/fig4_dose_response.pdf` — the five-point
dose-response vs the BC reference line.*

### 6.4 Failure-mode evolution and the remaining gap

Deployment-side reverse recovery nearly eliminates terminal wedging
(9/220 blocked vs 136 blocked routes in gen-1). The dominant remaining
failure is **in-game time budget exhaustion** (104/220 routes at the
~200 s budget with 2750 min-speed infractions): the planner is too
conservative in dense/obstructed states — precisely the states pushed
off-distribution by its own imperfect execution. We measured 41% of all
deployment ticks in a "plan commands motion, vehicle static" condition.
Seven controlled interventions (speed caps, privileged gap gating ×2,
hard-frame upweighted sampling ×2, head capacity) all failed a
10-route×3-rep gate against the champion — the residual bottleneck is
state-level OOD robustness of the plan, which motivates
imagination-DAgger-style training (roll perturbed starts in the world
model, relabel via retrieval) as future work.

## 7 Limitations

- **Privileged input**: ground-truth actor states and route (MAP
  track); claims are restricted to the privileged-planner class, and
  sensor-based DS SOTA (~60) remains far ahead.
- **Expert realism**: Bench2Drive's Think2Drive expert is itself a
  policy; real-log corpora (nuPlan/NAVSIM via the same vector
  featurization) are the natural next benchmark.
- **Retrieval cost**: O(N) per batch (fine to ~1M windows; ANN after).
- **Mode splicing**: the per-step max can reward hybrid trajectories;
  the trajectory-consistent variant performs identically and closes
  this hole.
- **Selector study scope**: 17 pairs from one lab's sweep; correlations
  are rank-based and the dev-10 subset is n=5. The within-objective
  inversion (n=12) is the load-bearing result.

## 8 Conclusion

Offline world-model imitation transfers to driving, but not as a single
recipe: nearest-mode matching earns its keep exactly where
demonstrations are multimodal; at benchmark scale, behavior cloning on
posterior features with a trajectory output head and classical tracking
is the stronger driver, reaching state-of-the-art Bench2Drive success
rate fully offline. The consistent negative space — trajectory
abstractions poisoning the dynamics action channel, in-model selection
metrics inverting, short-route smoke tests misranking — is, we argue,
as load-bearing for practitioners as the positive results.

---

## Appendix pointers (to be expanded)

- A. Frame conventions + certification harnesses (compass = CARLA yaw
  + π/2; forward = −y; round-trip tests; GNSS-noise data landmine and
  the ego-box fix).
- B. Full probe ledger with per-probe diagnosis (7 dev-10-gated
  interventions), tick-log methodology.
- C. Reproducibility: configs for every row; single-command chains
  (cache → train → eval) on Slurm; all evidence JSONs in runs/.
- D. Highway ablation tables (K, H, M, style ratio, data scale).
