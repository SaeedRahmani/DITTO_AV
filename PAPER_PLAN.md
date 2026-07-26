# Paper plan: Multimodal Latent-Matching Imitation for Driving, Fully Offline

Working title: **"One Expert Is Not Enough: Multimodal Latent Matching for
Offline World-Model Imitation in Autonomous Driving"**

## Positioning

Offline, simulator-free planner learning: the world model *is* the simulator.
We extend DITTO's on-policy latent-matching imitation to driving and identify
its failure mode there — **expert multimodality** — and fix it with
retrieval-based nearest-mode matching.

Claim structure:

1. BC on driving logs degrades closed-loop (covariate shift) — known, we
   reproduce it.
2. DITTO-style single-trajectory latent matching corrects covariate shift but
   *averages over expert modes*: from near-identical traffic states the data
   contains both "overtake" and "yield" continuations, and matching a single
   time-aligned window punishes the other valid mode.
3. Nearest-mode matching (reward = max latent similarity over K retrieved
   expert windows with similar start latents) recovers both modes, improving
   closed-loop return/collision rate, especially under distribution shift.

## Relation to prior work (checked July 2026)

| Work | What it does | Delta from us |
| --- | --- | --- |
| DITTO (DeMoss et al. 2023) | single-mode latent matching, Atari | multimodality, driving, vector WM |
| NVIDIA covariate-shift WM (Popov et al., ICRA 2025, arXiv:2409.16663) | latent WM + align-to-demo for AV, CARLA | single-demo alignment; needs CARLA for training loop; ours is retrieval-multimodal + fully offline |
| CoIRL-AD (ICML 2026, arXiv:2510.12560) | IL+RL dual policy in latent WM, nuScenes | competition mechanism, not multimodal expert matching; open-loop-centric |
| WorldRFT (AAAI 2026, arXiv:2512.19133) | WM planning + RL fine-tuning, NAVSIM | reward from hand metrics, not expert latent matching |
| MILE (2022) | BC in latent WM space, CARLA | no on-policy imagination reward |
| Think2Drive (ECCV 2024) | WM RL with env reward in CARLA | online RL, needs simulator reward |

Our niche: **offline + on-policy + multimodal expert matching**, no
hand-designed reward, no simulator in the training loop.

## Method summary

- RSSM world model (categorical latents) over vectorized scene features
  (ego + N nearest agents, relative kinematics).
- Latent bank: expert episodes filtered through the posterior; windows of
  length H+1.
- Policy learned by actor-critic in imagination from expert start states.
- Reward `r_t = max_{k<=K} maxcos(h_t, h^{(k)}_t) - mean_j maxcos(h_t, h^(neg_j)_t)`
  where the K windows are retrieved by cosine similarity of start latents
  (K=1, source window, no negatives = exact DITTO). The contrastive term is
  essential in driving: raw latent similarity to *any* plausible traffic
  state is ~0.85-0.92 (exogenous traffic dominates the latent), so the raw
  DITTO reward has a ~2% dynamic range and RL exploits the model instead
  (measured: trained policies reached higher raw reward than expert replay
  while crashing 76-82% closed-loop).
- Actor initialized from the BC policy with a KL trust region to it on
  imagined states (coef 0.3): imagination RL refines BC instead of
  rediscovering driving from scratch.
- Everything offline; closed-loop evaluation only at test time.

## First results (single seed, `runs/av1`, July 2026)

| policy | in-dist return | in-dist collisions | shifted return | shifted collisions |
| --- | --- | --- | --- | --- |
| expert (oracle) | 21.6 | 0.00 | 20.5 | 0.04 |
| random | 7.3 | 0.82 | 6.0 | 0.92 |
| BC (latent) | 15.4 | 0.40 | 10.9 | 0.64 |
| DITTO-single | 14.8 | 0.42 | 12.1 | 0.60 |
| **DITTO-multi (ours)** | **17.4** | **0.30** | **12.1** | **0.56** |

The predicted ordering holds: multi > single ≈ BC in-distribution
(multimodality is the dominant error source there), and both latent-matching
policies > BC under density shift (covariate-shift correction), with multi
best overall. Next: 3 seeds, K/H/negatives ablations, longer training.

## Experiment matrix

### Phase 1 — highway-env (done in this repo; `configs/av.yaml`)

- Env: highway-fast-v0, 3 lanes, density 1.3, 30 steps, 5 meta-actions.
- Expert: scripted, two styles (aggressive overtakes / conservative yields),
  50/50 per episode, ~0-3% crash rate. 300 expert + 100 noisy episodes.
- Policies: BC (latent), DITTO-single, DITTO-multi (K=8). Same WM, same
  feature space, same nets — isolates the objective.
- Metrics: closed-loop return, collision rate, mean speed, episode length;
  in-distribution and density-shift (1.3 -> 1.6, 20 -> 30 vehicles).
- Ablations to run next: K ∈ {1(no retrieval), 2, 4, 8, 16}; horizon H;
  retrieval on h vs full features; expert style ratio (25/75, 50/50);
  data scale (75/150/300 episodes); seed x3.

### Phase 2 — Bench2Drive (paper benchmark)

- Data: Bench2Drive base split (1000 clips, Apache-2.0, HF). Adapter
  (`ditto_av/bench2drive.py`) already parses clips into the vector format
  (validated on a real clip).
- Changes needed: continuous-action actor (Gaussian head on
  throttle/steer/brake) — WM side already supports continuous actions;
  scene features could add route/command conditioning (command_near /
  command_far fields are in the annotations).
- Evaluation: official Bench2Drive closed-loop protocol (CARLA leaderboard
  v2 routes, multi-ability splits) against published baselines (AD-MLP, UniAD,
  VAD, TCP); needs a CUDA Linux box with CARLA 0.9.15.
- The pitch there: no CARLA in the training loop; trained purely from the
  released clips; evaluated closed-loop.

### Phase 3 (optional, strengthens "real logs" claim)

- nuPlan/NAVSIM via the same vector featurization (requires nuPlan
  registration), or Argoverse 2 + ScenarioNet/MetaDrive replay for
  closed-loop eval on real data with no license gate.

## Theory section sketch

DITTO's regret bound builds on matching latent state distributions. Extend to
mixture experts: if the expert policy is a mixture over modes, single-window
matching bounds divergence to the *mean* trajectory (which can be infeasible);
nearest-mode matching bounds divergence to the *closest mixture component*.
Formalize as divergence to the mixture support vs the mixture mean.

## Venue targets

CoRL / ICRA / IROS (robot learning framing) or NeurIPS/ICLR (algorithmic
framing with the theory extension). Bench2Drive results are the gate for the
first tier.

## Honest limitations to state

- Latent bank retrieval is O(N) per batch (fine to ~1M windows; ANN index
  afterwards).
- Nearest-mode matching assumes retrieved neighbors are behaviorally
  compatible with the rollout's start; degenerate retrieval (K too large,
  sparse data) can reward mode-switching mid-rollout. Ablate K.
- The highway-env expert is scripted; Bench2Drive's Think2Drive expert and
  real logs (Phase 3) address "real expert" concerns.
