# NEXT_STEPS.md — live status + plan (single source of truth)

Read DELFTBLUE.md for cluster rules (lane audit protocol, storage,
conventions). PAPER_PLAN.md is the paper-side plan and evidence log.
This file: where the project stands, the plan, what not to redo.
History: git log of this file — old session narratives were compressed
out on 2026-07-29 (C2 cleanup); nothing scientific was lost, the
verdicts live in runs/ and the settled-facts list below.

## >>> HANDOFF 2026-07-29 — what is true right now <<<

**The pipeline drives itself on the cluster** (survives SSH/session
death): eval chains -> decider jobs chained with `--dependency`
(scripts/pipeline_decider.py) -> dev-10 -> full 220-route benchmark,
each stage aggregating results, committing to the repo (push manually
from a login session), appending to `outputs/PIPELINE_STATUS.md`
(**check that file + `squeue -u $USER` first, every session**), and
submitting the next stage on lanes chosen by a LIVE free-GPU audit
(pick_lanes(); never assume yesterday's fastest lane).

In flight at handoff:
- **Anchor/K grid DONE** (runs/carla_smoke/KL_ANCHOR_RESULTS.md):
  kl01 (bc_kl 0.1, K=8, H=15) wins at **64.6%** completion, clean
  dose-response peak, K=16 negative transfer, lights non-additive.
- **Town12-via-overlay proven in the leaderboard** (53% on route 2091).
- **dev-10 (kl01 vs v3, 10 routes x 3 reps)** auto-launched by the
  confirm3x3 decider; then the **220-route benchmark auto-fires** on
  the dev-10 winner (user-authorized). This gen-1 220 run doubles as
  benchmark-pipeline dress rehearsal AND the small-data ablation row —
  the FINAL 220 number must come from the scaled gen-2 winner.
- **Full-data download running** (login nohup, `outputs/
  b2d_download_full.log`): 703 new clips -> 1000 total (~335 GB).
  Tarballs only; extraction happens per the node-local /tmp pattern.
- **Training-length pilots** on gpu-a100-small (serialize, 1-job QOS):
  kl01 at 5x and 20x steps (`runs/b2d_kl01_{5,20}x` when done) — all
  runs so far were smoke-scale (wm 6000 steps, ~5 min on MIG).

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

### Running now (automated — just verify, don't relaunch)
1. dev-10: kl01 vs v3 -> `runs/carla_smoke/dev10_results.json`.
2. 220-route gen-1 benchmark (auto after dev-10) -> `runs/bench220/`.
3. 1000-clip download; training-length pilots 5x/20x.

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
