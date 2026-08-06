# DITTO-AV: Making Offline World-Model Imitation Work for Driving by Putting the Reward in the Right State Space

*Working draft v0.2 — 2026-08-04. All numbers are real and carry source
pointers (runs/). Seed error bars: [SEEDS-TBD] markers filled when the
G5 evals land. Author decision points marked [DECIDE].*

**Title alternatives** [DECIDE]:
1. "One Reward Is Enough: Counterfactual Closed-Loop Imitation for
   Autonomous Driving"
2. "From Latent Matching to Ego Matching: What It Takes to Make
   Offline World-Model Imitation Drive"

## Abstract

DITTO-style offline imitation — learn a world model from expert
demonstrations, then train a policy by reinforcement learning inside it,
rewarded for staying close to expert states — promises on-policy
imitation without a simulator. We show that a faithful instantiation
fails on real closed-loop driving (Bench2Drive) for a measurable reason:
the latent state is dominated by exogenous traffic the policy cannot
control, collapsing the reward's dynamic range (~0.85–0.92 cosine floor
between any two scenes) and inviting model exploitation; the policy
achieves higher matching reward than the expert replay while crashing.
No world-model scale fixes this. We keep DITTO's thesis and change what
"state" means: rollouts happen in the *recorded scene itself* — traffic,
route and lights replay from the log; an analytic kinematic model moves
the ego — and the reward is a time-tolerant kernel on the *ego's*
pose/speed against the same-scene expert. Nothing in the loop is
learned, so the reward cannot be exploited: it is maximized on the
expert's real path by construction. On Bench2Drive's official full 220-route
closed-loop benchmark our policy scores **75.9 driving score with 99.7%
route completion** (privileged-input, fully offline training) — 3.4×
the best latent-DITTO system on identical data and above all published
baselines including expert-distilled sensor methods. In a controlled
same-architecture comparison, the on-policy stage adds **+12.1 driving
score over behavior cloning** (83.6 vs 71.5 test-10), and a collision
penalty is *redundant*: pure state matching ties the shaped variant at
scale (75.88 vs 76.10). We further contribute a negative-results ledger
on metric transfer: short-horizon expert-closeness *anti-orders*
healthy policies while collision rate transfers, and teacher-forced
probes misread on-policy-trained recurrent policies.

## 1. Introduction

Behavior cloning degrades in closed loop because deployment states
drift from the expert's (covariate shift). DITTO [DeMoss et al., 2023]
offers an elegant offline fix: learn a world model from demonstrations,
roll the policy inside it, and reward closeness to expert *states* —
recovery becomes a trained skill, no simulator needed. The thesis
transfers poorly to driving, and this paper is about why, and what the
working driving-native instantiation looks like.

Contributions:
1. **A measured failure analysis** of latent state-matching imitation
   in driving (Sec. 3): exogenous latent dominance → reward floor →
   exploitation; dose–response evidence across five imagination
   variants; a selector study where the latent metric *anti-predicts*
   closed-loop score (Spearman −0.60).
2. **A driving-native instantiation of the DITTO thesis** (Sec. 4):
   counterfactual rollouts in replayed scenes with an analytic ego and
   a pure, time-tolerant ego-state-matching reward. Verified world:
   observation rebuild ≡ deployment featurizer (3e-7); expert replay
   retraces real logs to 0.07–0.14 m over 4 s.
3. **State of the art on Bench2Drive** in the fully-offline,
   privileged-input class (Sec. 5): full 220 routes, DS 75.9–76.1,
   completion 99.5–99.7%, above all published baselines [caveat
   discussion]; controlled same-net evidence that the on-policy stage
   is the cause (+12.1 DS over BC); shaping redundancy (pure ≈ shaped).
4. **A metric-transfer ledger** (Sec. 6) extending the open-loop ≠
   closed-loop literature: which offline/sim metrics rank real drivers
   (collision rate: yes, +0.50; expert-closeness: no — it anti-orders
   the healthy band), and a probe-validity result for recurrent
   on-policy policies.

## 2. Related work (honest positioning)

**Closed-loop training in data-driven/log-replay simulators.** Urban
Driver [Scheel et al., CoRL 2021] trains a policy closed-loop in a
differentiable replay simulator with a distance-to-expert objective —
the closest prior to our loop; we differ in the RL formulation (reward
kernel + A2C rather than BPTT through the sim), the *pure* reward (no
auxiliary costs; we show shaping is redundant), the time-tolerant
matching, and the evaluation venue (official Bench2Drive closed-loop
protocol vs in-house replay evaluation — the latter partially grades in
the training distribution). BC-SAC [Lu et al., 2022] combines BC with
closed-loop RL on replayed logs but requires hand-engineered safety
rewards; our result is that with the right state space, imitation alone
suffices. Log-replay simulation infrastructure is standard (nuPlan,
Waymax, L5Kit, GPUDrive); our contribution is not the simulator but
the reward semantics and the controlled evidence. KING, CW-ERM and
perturbation-based lines attack covariate shift by data augmentation
rather than on-policy training.

**Offline world-model imitation.** DITTO (Atari, latent matching),
V-MAIL, EfficIL (conservative world models), the NVIDIA latent-WM
covariate-shift system [Popov et al., 2024] and CoIRL-AD (latent-WM
IL+RL on nuScenes, open-loop-centric) learn latent dynamics and match
in latent space. We bridge these to the replay-training line: Sec. 3
measures why whole-latent matching fails in dense traffic, and Sec. 4
shows the thesis survives when matching moves to the controllable
state, with the world model's irreplaceable remainder (reactivity)
deferred and delimited.

**CARLA / Bench2Drive.** Think2Drive (online RL expert; data source),
sensor-track students with expert-feature distillation (TCP, ThinkTwice,
DriveAdapter), privileged baselines (AD-MLP). We evaluate on the
official full 220-route protocol; our planner consumes privileged object
states (no cameras) and trains fully offline — comparisons carry that
caveat explicitly, and the closest published comparables are the
privileged/no-distillation rows. [Check before submission: the Dec-2025
"pseudo-expert regularized offline RL" line and any 2026 successors on
this protocol.]

## 3. Why latent-space matching fails in driving (v0.1 evidence)

Setup summary (full detail: branch saeed/ver0.1): RSSM world model on
vector scene observations (ego + 6 nearest agents + route), DITTO
reward = max-cosine to expert latent windows, dose-controlled variants.
Findings, each with banked runs:

- **Latent saturation**: cosine similarity between *any* two plausible
  traffic latents is 0.85–0.92; the DITTO reward's usable range is ~2%.
  Contrastive baselines restore range but not meaning.
- **Reward exploitation**: policies exceed the *expert replay's own
  reward* while crashing 76–82% closed-loop — the dream is gameable.
- **Dose–response**: five imagination-refinement variants (KL anchors
  0.1–1.0, divergent starts, early stopping) all lose to plain BC
  closed-loop; deterministic open-loop MAE *improves below BC's* while
  driving degrades (metric inversion at its sharpest).
- **Selector inversion**: on-policy latent match anti-predicts
  closed-loop score across 17 banked models (Spearman −0.60).
- The same machinery *wins* in a controlled low-exogeneity toy
  (highway-env, multimodal experts) — the failure is driving-specific
  state semantics, not the algorithm class. Scale does not rescue it:
  a 3.4× data / 10× steps scale-up left BC ahead.

## 4. Method: counterfactual closed-loop imitation

**World.** For each training window, the recorded scene replays:
actor tracks, route command points, and lights come from the log
(non-reactive); the ego is moved by an analytic kinematic model that
follows the policy's predicted waypoint plan on the plan's own time
parametrization (two-stride instantaneous-speed extrapolation,
arc-fraction heading update; caps only bound degenerate plans).
Verification, not assumption: (i) the sim's observation function
matches the offline adapter/deployment featurizer to 3e-7 over real
frames; (ii) feeding the expert's own plans retraces real logs to
0.07–0.14 m mean over 4 s (gate: <0.3 m). Nothing in the loop is
learned → the reward cannot be exploited by construction; expert
replay scores ≈1.0.

**Reward.** r_t = max_{|δ|≤τ} exp(−½[‖Δxy‖²/σ_p² + Δθ²/σ_θ² +
Δv²/σ_v²]) against the same-scene expert states; τ=0.5 s tolerance
absorbs timing slack; σ_p=1 m (the *tight* kernel — a wider mixture
kernel degrades lane discipline; Sec. 5 ablation). No collision,
progress, or comfort terms in the headline variant.

**Policy.** Per-actor token transformer (ego/agents/route tokens,
d=192, 3 layers) + GRU memory + Gaussian waypoint head (12-dim plan,
identical action space and downstream tracker as the strongest v0.1
deployment stack; ~3.2 M parameters). No prev-action input
(action-feedback drift, v0.1 lesson).

**Training.** Stage 1: sequence BC (also the controlled baseline).
Stage 2: A2C on sim rollouts (H=4 s, burn-in 0.8 s on logged prefix,
λ-returns, EMA target critic, entropy bonus) with a KL trust region to
the frozen BC snapshot; 25% of rollouts start from perturbed poses
(divergent starts) whose targets remain the same-scene expert —
recovery is well-posed, unlike retrieval-based relabeling in latent
space (Sec. 3).

**Deployment.** Fresh observations per tick → GRU → deterministic
plan → the same pure-pursuit tracker + reverse recovery as all v0.1
baselines. The only moved piece between compared systems is the
training objective.

## 5. Results (all closed-loop CARLA, official protocols)

**Controlled same-net comparison, test-10 (10 routes × 3 reps):**

| policy (identical net/data/tracker), 3 full-pipeline seeds | DS (mean ± sd; seeds) | full routes |
|---|---|---|
| sequence BC | 65.32 ± 9.36 (71.48/54.55/69.92) | 75/90 |
| + on-policy stage, pure reward | **76.31 ± 7.54** (83.60/76.78/68.55) | **90/90** |
| + on-policy stage, +collision penalty | **78.39 ± 8.24** (85.63/80.13/69.42) | **90/90** |

The on-policy stage adds **+11.0 (pure) / +13.1 (shaped) DS on seed
means** — consistent with the matched-seed-0 gap of +12.1 — and the
effect is uniform in kind: *every* RL seed completes *every* route
(180/180 runs at 100% completion) while every BC seed leaves routes
unfinished or blocked (75/90). Seed variance in DS is large at this
band (±7.5–9.4, penalty events, not completion), so the per-seed DS
ordering fluctuates but the RL-over-BC and completion effects do not.
Pure vs shaped is a statistical tie across seeds (Δmean +2.08 ≪ σ≈8),
consistent with the 220-scale dead heat (75.88 vs 76.10) — the shaping
redundancy claim survives seeding.

**Official full 220-route benchmark:**

| system | DS | completion | success |
|---|---|---|---|
| v0.1 best (latent-WM + BC head, same data) | 22.10 | 68.7% | 39.1% |
| published best (DriveAdapter*, sensors+distill) | 64.22 | — | 33.08% |
| **DITTO-AV pure** | **75.88** | 99.7% | 99.5% full / 48.2% strict |
| **DITTO-AV shaped** | **76.10** | 99.5% | 99.1% full / 49.5% strict |

*Privileged-input, fully-offline caveat applies throughout; starred
baselines are sensor-track with expert distillation — included for
scale, not parity.* "Strict" counts zero-infraction completions.
Shaping is redundant at scale: the pure state-matching variant ties the
shaped variant — the paper's cleanest single claim.

**Ablations** (each test-10-gated): kernel width (a 4 m mixture
component costs 21-vs-0 lane infractions at 999 scale — matching
geometry is load-bearing); data scale (297→999 flips RL past BC);
architecture (the same recipe's BC at 297 already reaches 74.10 —
2.4× the v0.1 champion — separating representation gains from
objective gains) [DECIDE: promote to its own subsection].

## 6. Findings: which metrics transfer?

1. **Expert-closeness does not rank working drivers.** Across 12
   banked v0.1 models with known test-10 scores, sim-closeness
   *anti-orders* the healthy band (conservative champions sit far from
   the expert; aggressive trackers sit close and crash); Spearman
   +0.26 overall — but broken models separate from working ones by a
   wide margin (the metric's valid role: sanity, not selection).
2. **Collision rate transfers** (+0.50 vs completion) — and is the
   axis where the shaped variant helps at small scale (3×3/test-10) before
   becoming redundant at 999.
3. **Teacher-forced probes misread on-policy-trained recurrent
   policies**: the champion's plans anti-correlate with expert steer
   under teacher forcing (−0.28) while driving 30/30 routes with zero
   lane infractions — recurrent state distribution matters; only
   closed-loop evaluation is admissible.
4. **Executor semantics leak**: the sim's polyline-follower permits
   backward micro-steps real trackers cannot execute (4% of steps for
   the champion, 0% for the expert) — cosmetic here, but a caution for
   replay-sim training generally.

## 7. Limitations

Privileged object-state inputs (no perception); non-reactive replay
(ghost traffic: measured artifacts, front-impact-only penalties, capped
divergence — reactivity is exactly what a learned world model must add
[v0.3]); no traffic-light input (12 red-light infractions/220) and no
map/lane input (lane discipline learned implicitly); single benchmark
family (CARLA/Bench2Drive) and a single expert (Think2Drive); strict
zero-infraction success is 48–50% — collisions with replayed-vs-live
traffic mismatches dominate the residual; test-10 seed variance at this performance band is
large (±7.5–8.2 DS over 3 full-pipeline seeds — penalty events, not
completion), so single-seed test-10 comparisons within ~8 DS are not
individually conclusive; the 220-scale results and the
completion/blocked contrast carry the load-bearing claims.

## 8. Conclusion

DITTO's thesis — offline on-policy imitation with a state-matching
reward — survives contact with driving once "state" is the part of the
world the policy controls, evaluated in the scene the expert actually
faced. The learned world model was never the load-bearing component in
this setting; the reward semantics were. What a learned model *is*
irreplaceably for — reacting to counterfactual egos — is the next
question (v0.3).

---
*Source pointers: test-10/220 records runs/bench220_v02_999{t,s}_rl,
runs/carla_results_v999*; gates V02_PLAN.md §8; v0.1 evidence branch
saeed/ver0.1; visualizations runs/viz_v02_final; probe scripts
scripts/egosim_selector.py, /tmp diag (reverse_probe).*
