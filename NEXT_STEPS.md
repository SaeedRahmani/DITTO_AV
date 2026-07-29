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

### Gen-2: scale for performance (start when download lands)
4. Extract new clips via node-local /tmp -> validate (validate_b2d
   pattern; watch the inode budget, DELFTBLUE storage rules).
5. Retrain kl01-config on 1000 clips; steps per pilot verdict; if 5x/
   20x helps, fold in. GPU lanes per live audit (MIG fine for training).
6. Closed-loop selection at gen-2: 3-route 3x3 smoke for {anchor 0.1
   +-0.05 re-check, lights on/off at the new scale}; then dev-10 on the
   winner; **3 seeds** on the final config (paper needs error bars).
7. Attack the residual wedge if it survives scale (obstacle-blocked is
   still the terminal state everywhere): recovery-data augmentation or
   steer-authority pairing — closed-loop-selected, one axis at a time.
8. FINAL 220-route benchmark on the gen-2 winner (+ the gen-1 row as
   the data-scale ablation).

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
