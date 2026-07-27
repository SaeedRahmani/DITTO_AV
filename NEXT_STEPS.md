# NEXT_STEPS.md — state + plan (updated 2026-07-28, Round-2 code session)

Read DELFTBLUE.md first (cluster rules). This file: where the project
stands, what we learned, and the prioritized plan.

## 2026-07-28 session: Round-2 code done + CRITICAL bug found

Scratch write-freeze returned (1-byte group quotas on the pool backing
/scratch; home writable). Workaround in place: writable clone at
`~/ditto_work/DITTO_AV` (origin = github), scratch stays readable
(venv, data, Bench2Drive clone all usable read-only).

**CRITICAL FIX — closed-loop obs were rotated 90°.** Bench2Drive's anno
`theta` is the IMU compass = CARLA yaw + pi/2 (data_collect.py converts
back with `rad2deg(compass) - 90`; verified exact on 233 frames / 20
clips, max dev 1e-4 rad). Training obs therefore live in the compass
frame, but DittoCarlaAgent featurized with the raw yaw — every relative
feature (neighbors, velocities, headings, route points) was rotated 90°
vs training in ALL closed-loop rounds so far. Fixed deployment-side
(`ego_yaw = deg2rad(yaw) + pi/2` in run_step), which keeps every
trained checkpoint valid. Consequences:
- All prior closed-loop numbers (v3 12.6±11.3, v4 6.7±3.9; wedging;
  huge variance) were measured WITH the bug → the v3-vs-v4 ranking and
  the tuning conclusions must be re-established after the fix.
- Re-run the cheap 3-route × 3-rep smoke FIRST (expect a large jump);
  only then the 10-route evals.

**Round-2 code (41 tests green):**
1. Traffic-light obs block: offline `_light_block` (presence + ego-frame
   trigger volume + red/yellow/green one-hot, 6 dims after the route
   block; `env.light_obs`, extra_obs_dims 22, `configs/b2d_v5.yaml`).
   Online: exact port of data_collect's most_affect_light (dense-plan
   override of set_global_plan — the leaderboard downsamples to ~50 m,
   too sparse for the ~4x2 m trigger boxes; waypoint walk cached per
   light). Verified on real clips: expert brake-rate 46% on red frames
   vs 5% on green.
2. StuckRecovery (config-gated, enabled in configs/carla_agent.yaml):
   >40 model ticks under 0.3 m/s at commanded throttle → 15 ticks
   reverse with mirrored steer; braking at red never counts as stuck.
   Logged per tick (`rec` field; also `light`).
3. npz cache: key extended for lights; cache writes best-effort
   (survive scratch freezes).

v5 training SUBMITTED freeze-safe (job 10527119,
scripts/slurm/phase2_home.sbatch -> ~/ditto_out/b2d_v5; commit results
to runs/b2d_v5 once done). Still blocked by the freeze: closed-loop
re-eval of v3 post-fix, v3-vs-v5 comparison. When the freeze lifts:
`git -C /scratch/$USER/ditto_av/DITTO_AV pull` FIRST (the scratch clone
predates the theta fix — closed-loop jobs run from it), re-extract
AdditionalMaps, re-baseline v3 (plan item 0), restart wandb sync.
Stop signs deliberately NOT in the obs (6-dim light block per spec; the
anno has `traffic_sign`/`traffic.stop` with affects_ego if wanted
later).

Everything below is the pre-session state (through `1634abf`) with the
plan; read it with the bug fix in mind.

## Where we stand

**Phase 1 (highway-env) — DONE, paper-grade.** 24 sweep runs in
`runs/phase1/`. Headline (K=16,H=5, 3 seeds): DITTO-multi 20.1±0.9 return
/ 0.10±0.06 collisions ID; 18.3±1.5 / 0.21±0.11 shifted — near-expert,
large gap over single/BC. Evidence trio against reviewer objections:
paired same-seed rollouts (57% diverge), trajectory-consistent multi_traj
(== multi), unimodal control (gap vanishes). All in PAPER_PLAN.md.

**Phase 2 (Bench2Drive offline) — DONE.** 297 clips (94 GB) validated;
route-conditioned obs (49+16 dims); v3 = K8/H15 (best driver), v4 =
K16/H5. Open-loop results `runs/b2d_v*`. Key: open-loop metrics did NOT
predict closed-loop ranking.

**Closed-loop CARLA — WORKING, needs driving-quality work.**
Full pipeline proven (apptainer CARLA 0.9.15 + Bench2Drive leaderboard +
DittoCarlaAgent). Tuning round (3 base-town routes × 3 reps,
`runs/carla_smoke/`): v3 12.6±11.3 score / 43%±34 completion; v4 6.7±3.9
/ 17%±11 → **v3 stays deployed** (configs/carla_agent.yaml). Run-to-run
variance is huge → any claim needs ≥3 reps. Universal failure: agent
wedges against obstacles/walls and never recovers. Second gap: no
traffic-light/stop-sign in obs (privileged planner framing in
PAPER_PLAN applies).

## Hard-won infrastructure facts (do not rediscover)

- GPU nodes have NO python modules and don't mount the venv's spack tree
  → only self-contained conda envs run there: `carla_eval` (torch is
  CPU-only — eval driving) and `~/envs/ditto_gpu` (torch cu130 — GPU
  training; A100 partitions only, cu130 dropped V100/sm_70). CPU jobs
  keep the ditto venv. See the env table in DELFTBLUE.md.
- Queue: gpu-a100/gpu-v100 allow 48 h walltime (a100-small 4 h) — long
  GPU jobs are fine. ≤59-min requests backfill in minutes, a useful
  tactic for smokes when the queue is busy, NOT a limit.
  `--gpus-per-task` is mandatory syntax. innovation account blocks GPU
  jobs in practice; use research-ceg-tp.
- CARLA: SIF at /scratch/$USER/ditto_av/carla_0915.sif; evaluator
  launches the server itself via $CARLA_ROOT shim (carla_root/). Town11-15
  need AdditionalMaps (tarball on scratch; extraction to
  additional_maps_extract/ — verify complete, was re-launched 2026-07-27;
  then bind dirs into the shim's apptainer call with --bind).
- Bench2Drive harness quirks (patched clone on scratch; patch archived in
  scripts/patches/): py3.10 getchildren fix; evaluator appends
  '+save_name' to agent-config (agent strips it); cwd must be
  Bench2Drive/ (relative weather.xml); routes-subset takes route IDs.
- Brake must be binarized at deployment (Gaussian mean rides the brake);
  done in carla_agent (brake_threshold).
- run_b2d data stage uses an npz cache (npz_cache/) — key = clip split +
  route flag; extend the key if obs layout changes again.
- Scratch write-freezes (1-byte group quotas; hit 2026-07-27 AND
  2026-07-28) are admin-side incidents. Full workflow for working
  through them: DELFTBLUE.md "Scratch write-freezes: the dual-clone
  workflow". Diagnose: `beegfs-ctl --getquota --gid <each group>`.

## Plan (in order)

### Round 2: driving quality (code DONE 2026-07-28; runs remain)
0. **Re-baseline v3 with the theta fix** (do FIRST when jobs can run):
   3 base-town routes × 3 reps, ROUTES_SUBSET=25381,25378,27494,
   carla_smoke.sbatch. Prior numbers are invalid (90° obs rotation).
1. **Traffic-light/stop observation**: anno `bounding_boxes` contain
   traffic_light/stop entries (check `state` field names on a real
   frame). Append compact block (nearest relevant light: presence, rel
   x/y, red/yellow/green one-hot ≈ 6 dims → extra_obs_dims 22). Extend
   `_route_block`-style code + featurize_frame + tests (mirror the route
   block pattern exactly, keep off by default). New cache key follows
   automatically. Retrain v5 (= v3 config + lights); phase2.sbatch.
2. **Stuck recovery** in DittoCarlaAgent: if speed < 0.3 m/s for > 40
   model ticks while commanding throttle → brief reverse+steer sequence,
   then resume. Deployment-side, config-gated, log events.
3. **Evaluate properly**: v3 vs v5 on all 10 dev routes × 3 reps (needs
   AdditionalMaps bound into shim). ~6-8 GPU-h total on gpu-v100 with
   59-min-per-chunk jobs (split subsets). Keep results in runs/.
4. Optional cheap wins to test in the same eval: stochastic=true;
   action_repeat=1; brake_threshold sweep {0.3,0.5,0.7}.

### Round 3: scale + benchmark (needs user sign-off on budget)
5. Full Bench2Drive base split (703 more clips, ~240 GB, login-node
   nohup) + retrain on ~1000 clips (GPU training now worthwhile —
   wm.device cuda path untested; verify).
6. **Full official 220-route eval** ×(≥1 rep): ~36+ GPU-h — get explicit
   user OK (DELFTBLUE rule). Position vs privileged baselines only.

### Round 4: paper
7. Write: method + Phase-1 evidence are complete; closed-loop section
   honest framing (privileged planner; open-loop ≠ closed-loop finding
   is a contribution). Theory sketch in PAPER_PLAN. Venue: CoRL/ICRA.

## Live loose ends (check on session start)
- AdditionalMaps extraction complete? (`ls additional_maps_extract/CarlaUE4/Content/Carla/Maps` then wire --bind into carla_root/CarlaUE4.sh shim and test route 2091/Town12.)
- wandb sync loop running? (restart via scripts/wandb_sync.sh nohup.)
- W&B project `ditto-av` has all training curves.
