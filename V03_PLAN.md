# V03_PLAN.md — v0.3: a learned, reactive world model (branch saeed/v0.3)

Decided with the author 2026-08-04. v0.2 continues on `main` (another
session); v0.1 is frozen on `saeed/ver0.1`. This branch works ONLY in
the worktree `/scratch/$USER/ditto_av/DITTO_AV_v03` — NEVER check
saeed/v0.3 out in the main clone (`ditto_av/DITTO_AV`): running v0.2
jobs read scripts/configs from that tree at runtime.

## 1. Goal

v0.2 proved the mechanism (on-policy imitation in a replayed world, pure
ego-state-matching reward: 220-route DS ~76, RL > same-net BC +12 dev-10)
but its world is NON-REACTIVE: replayed traffic ignores the ego's
deviations. v0.3 learns the one thing replay cannot do — REACTIONS —
and nothing else:

- Keep the factorization: ego = analytic kinematics (never learn what
  you know). Learn a TRAFFIC MODEL: token-transformer over per-agent
  history predicting other agents' next states, conditioned on the
  ego's ACTUAL current pose — so traffic yields/brakes when our policy
  deviates. This is the mature "sim agents" task (WOMD Sim Agents);
  recipes exist. Data: the same ##glob1 arrays (act_glob world tracks),
  999 clips, train/val split unchanged.
- The policy recipe stays v0.2's (TokenPolicy, sequence-BC init,
  A2C + ego-state-matching reward, tight kernel — shaping proven
  redundant at 220 scale).

## 2. Trust ladder (how the learned world earns its place)

- **W0 fidelity gate** (analogue of v0.2's G0): with the RECORDED ego
  replayed, the traffic model must reproduce held-out logs within
  tolerance (rollout ADE/FDE, collision-rate realism vs log). Never
  train a policy in a world that fails W0.
- **W1 non-exploitability**: ensemble of K traffic models; during
  policy training, penalize/terminate rollouts where ensemble
  disagreement spikes (MOPO-style pessimism). Structural answer to
  v0.1's reward hacking: the policy cannot farm reward where the model
  doesn't know.
- **W2 curriculum**: start from the v0.2 champion policy; anneal
  replay->reactive rollout ratio; KL-anchor to the replay-trained
  policy. Reward stays ego-state matching to the same-scene expert.
- **W3 external**: dev-10/220 CARLA gates as always — the learned
  world NEVER grades itself (v0.1/v0.2's most-replicated lesson:
  internal metrics do not rank closed-loop drivers).

## 3. What v0.3 buys (the paper deltas)

1. Removes ghost-traffic bias — negotiation/yielding become learnable.
2. Divergent-start recovery becomes fully coherent (traffic responds).
3. Horizons beyond clip end; scenario generation (traffic variations).
4. Full-circle experiment: re-test DITTO's ORIGINAL latent reward
   inside the factored (ego|traffic) latent — does latent matching work
   once the state is factored? Closes the v0.1 question scientifically.
5. M6 carry-over lever: tracker-port execution in egosim (quantified
   4% backward-microstep executor gap in v0.2).

## 4. Honest novelty position (searched 2026-08-04)

Closed-loop training in log-replay sims is established (Urban Driver
CoRL'21 = closest prior: differentiable replay sim + distance-to-expert
loss; BC-SAC'22 = closed-loop RL+BC with hand-engineered safety reward;
Waymax/nuPlan/GPUDrive = infra). Ours that holds: (a) pure state-
matching reward suffices closed-loop (shaping-redundancy ablation);
(b) the measured diagnosis of WHY latent-space DITTO fails in driving +
the controlled fix; (c) Bench2Drive-220 evidence. v0.3 adds: gated
reactive imagination with pessimism + the factored-latent DITTO retest.
READ BEFORE CLAIMING PRIORITY: arXiv 2512.18662 (offline RL,
photorealistic closed-loop envs, Dec 2025).

## 5. Coexistence rules (two branches, one cluster)

- Work ONLY in `DITTO_AV_v03/`; the main clone belongs to the v0.2
  session. Same for `~/ditto_work/DITTO_AV` (home clone = main's).
- SLURM lanes are shared: CLAIM stages in outputs/PIPELINE_STATUS.md
  (tag entries "V03:"), read the last ~10 lines before every submit;
  MIG = 1 job/user TOTAL (both branches count); >30 min pend =
  re-audit lanes.
- Run dirs / results / wandb names: prefix everything `v03_`
  (~/ditto_out/b2d_v03_*, carla_results_v03*). Never reuse a v0.2 name.
- npz caches are content-keyed — new layouts get new keys, no
  clobbering; extractions still go to node-local /tmp (1M-inode rule).
- Merging back: v0.3 mostly ADDS files (models/traffic.py, trainers,
  configs). Keep edits to shared files (config.py, egosim.py) additive
  (new fields/classes, no renames) so the eventual merge into main is
  trivial.

## 6. First milestones

- V3-M0 (this commit): branch + worktree + plan.
- V3-M1: traffic-model dataset views over ##glob1 (per-agent history
  windows, ego-conditioned targets) + TrafficModel (token transformer,
  ~10-30M) + W0 fidelity harness on held-out clips.
- V3-M2: ensemble + disagreement stats; reactive EgoSim mode
  (traffic states from model rollout instead of log index).
- V3-M3: W2 curriculum fine-tune of the v0.2 champion; G-gates.
- V3-M4: factored-latent DITTO retest; paper integration.

## 7. Status ledger (newest first)

- 2026-08-04: V3-M0 — branch `saeed/v0.3` + worktree
  `DITTO_AV_v03/` created from main @ a8a84b1; plan committed.
