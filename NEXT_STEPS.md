# NEXT_STEPS.md — state + plan

## >>> HANDOFF 2026-07-28 late — READ THIS FIRST, it corrects everything below <<<

1. **Storage saga RESOLVED — root cause was the 1M chunk-file (inode)
   quota on /scratch, NOT group quotas and NOT an admin freeze.** The
   "1-byte group quota" theory below and in older notes is WRONG (red
   herring in beegfs-ctl output). Confirmed by DHPC: limit is fixed;
   check `bash /etc/profile.d/ZZ_motd-info.sh` (chunk files column) at
   session start; keep >=100k headroom. Full rules + adopted DHPC
   node-local-/tmp I/O pattern: DELFTBLUE.md storage section. Writes
   work now (789k/1M). The dual-clone workflow remains as an emergency
   pattern only.
2. **Frame question SETTLED**: anno theta = CARLA yaw + pi/2 (compass);
   box rotations = true headings. Deployment MUST use yaw_offset pi/2
   (configs/carla_agent.yaml, default). PROOF: scripts/
   replay_frame_check.py reproduces training obs exactly (diff ~1e-4);
   road-test A/Bs are too noisy to rank frame conventions — never
   revert geometry from road tests. ALL closed-loop scores measured
   before this fix (v3 12.6, v4 6.7, the 24.2 single run) are void.
3. **All three clones synced at this commit** (scratch /scratch/$USER/
   ditto_av/DITTO_AV = primary again; home ~/ditto_work/DITTO_AV =
   backup; GitHub = truth). Stuck-recovery exists but is OFF (its road
   A/B regression was real); brake binarization ON; lights code ready.
4. **Where the work actually is — read the session log below for the
   live thread.** Already DONE (do not redo): post-fix re-baselines,
   stochastic sweep (dead), recovery tuning (dead), route-semantics fix
   (RouteCursor, second deployment bug), KL-anchor sweep — bc_kl_coef
   0.1 doubles completion (47.7% vs 22.5%), kl01 3x3 confirmation was
   running at handoff. **Next (in order):** (a) collect kl01
   confirmation + v5 GPU outputs (job 10527146, ~/ditto_out/) and
   commit results; (b) anchor/k_modes grid + v3-vs-v5 on the winner;
   (c) AdditionalMaps via node-local /tmp -> all 10 dev routes x 3
   reps; (d) 220-route eval decision (user sign-off ~36 GPU-h), data
   scale-up, paper (PAPER_PLAN.md).
5. Cleanup approvals PENDING from user (do not touch without OK):
   test/PufferDrive_hetero (71k files), pufferdrive archives (~400k) —
   these are the user's other projects.
6. wandb sync loop is stopped; restart after next training if live
   dashboards wanted.


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

**Re-baseline VERDICT (2026-07-28, runs/carla_smoke/ diag files).**
The 3×3 re-baseline with fix+recovery scored LOWER than the buggy
baseline (1.72 / 17.3% vs 12.6 / 42.8%) — but the A/B on route 25381
(recovery OFF both arms) shows why, and it vindicates the fix:
- fix: 16.5% completion, ONE layout collision, penalty 0.65, ends
  "blocked" behind a scenario obstacle — clean driving, then wedges.
- old frame (control): 100% completion but NINE layout collisions +
  1 vehicle, penalty 0.12, score 0.39 — the rotated obs act as a
  90°-rotated feedback loop that RICOCHETS along the route corridor.
  Yesterday's 43% completion was this artifact. Training-npz stats
  independently confirm the compass frame (near point median (0,-.046),
  neighbor heading sin=-0.999).
Implications: (1) keep the fix — per-meter driving quality is far
better; report completion AND penalty, composed score alone is
misleading here. (2) The completion blocker is the policy wedging
behind obstacles (lane-change commitment) — a policy/data question
(try stochastic=true; multimodal capture is our method's whole point).
(3) First recovery tuning (reverse+mirrored steer, 4 s) made things
WORSE with the fix (collision cascades, route deviations, zero
"blocked" endings but 5-8 collisions/run) → retuned conservative:
straight gentle reverse, 3-strike give-up (StuckRecovery).

**stochastic=true is a dead lever** (3x3, runs job 10527197):
3.42 / 8.8% — worse than deterministic (4.5); sampling adds collisions
without unlocking the obstacle wedge.

**SECOND deployment bug found+fixed: route conditioning semantics**
(commit 1b2ed7c). The tick log showed deployed near points at median
rel (+0.24, -0.02), |x| p90 0.48, some BEHIND the ego — training has
them hovering ~5 m ahead at (0.000, -0.046). plan_to_command_points'
change-point heuristics over the 50 m-downsampled plan were invented
semantics. Measured the collector's real ones from 2.6k anno frames:
near = dense-plan node (1-2 m spacing) popped at ~4 m; far =
downsampled command node popped at ~7.5 m. RouteCursor now ports this
exactly. Route conditioning was off-distribution in EVERY closed-loop
run to date — including everything above; the routefix 3x3
(job 10527378) is the first honest closed-loop measurement.

**BC anchor = the freeze (closed-loop KL sweep, 3x1 each).**
With the honest stack, bc_kl_coef 0.1 (default 0.3) breaks the obstacle
wedge no deployment lever touched: 54% completion on 25381 (v3: never
past ~10%, 5 runs), mean completion 47.7% vs 22.5% baseline; kl003
intermediate (26.4%). Cost: freed steering is sloppy (7 layout
collisions on 25381) — commitment-vs-precision tradeoff. Open-loop
NLL/MAE ordering (v3 < kl01 < kl003) ANTI-predicts closed-loop
completion — second clean instance of the open-loop != closed-loop
finding. Deployment levers all measured dead: stochastic (worse),
aggressive recovery (collision cascades), conservative recovery
(score-neutral, completion down; deterministic re-approach re-wedges).
kl01 3x3 confirmation running (job 10527512-ish); next: anchor/k_modes
grid + precision recovery (maybe pair with lower steer authority), and
v3-vs-v5 on the winner.

**Frame question CLOSED by the offline replay test**
(scripts/replay_frame_check.py): rebuilding featurize_frame inputs from
recorded clips' boxes and diffing against load_clip obs gives, over 460
frames / 3 towns: yaw_offset pi/2 -> max abs diff 1e-4 (EXACT);
yaw_offset 0 -> mean 0.24, max sqrt(2) (a 90-deg rotation). The
raw-yaw composed-score "win" (12.6 vs 4.5) is the ricochet artifact,
not perception quality — unusable for any paper claim. Deployment
default restored to pi/2; completion has to come from the policy.
Next lever queued: stochastic=true 3x3 (breaks the deterministic
wedge-freeze; plan item 4).

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
- GPU access (verified via sinfo/sacctmgr 2026-07-28 — the user HAS
  strong A100 access; never claim otherwise):
  * partition walltime caps: gpu-a100 & gpu-v100 **48 h**, gpu-a100-small
    4 h, compute-p1 120 h.
  * account `research-ceg-tp`: up to **8 GPUs/job**, 64 concurrent jobs
    → default choice for GPU work (up to 48 h A100).
  * account `innovation`: up to 2 GPUs/job, **24 h**, 1 job at a time —
    a valid fallback when research-ceg-tp queues badly.
  * `--gpus-per-task=N` is the required syntax (`--gres` is rejected).
  * ≤59-min requests backfill within minutes on busy days — an OPTIONAL
    queue tactic for short smokes only, NOT any kind of limit.
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
   AdditionalMaps bound into shim). ~6-8 GPU-h total on gpu-v100 (single 48 h-capable job is fine; split into
   59-min chunks only if the queue is congested). Keep results in runs/.
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
