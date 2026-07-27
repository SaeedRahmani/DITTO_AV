# NEXT_STEPS.md — state + plan (written 2026-07-27, after closed-loop round 1)

Read DELFTBLUE.md first (cluster rules). This file: where the project
stands, what we learned, and the prioritized plan. Everything below is
committed through `1634abf`.

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
  → use `/scratch/$USER/ditto_av/envs/carla_eval/bin/python` (conda,
  self-contained) for anything on gpu-*. CPU jobs keep the ditto venv.
- Queue: request ≤59 min on gpu partitions → backfill starts in minutes;
  2 h requests can wait 12 h. `--gpus-per-task` is mandatory syntax.
  innovation account blocks GPU jobs in practice; use research-ceg-tp.
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
- 2026-07-27 scratch freeze (1-byte group quotas) was an admin-side
  incident, resolved same day. If writes fail again: check
  `beegfs-ctl --getquota --gid <each group>`, outputs can go to $HOME.

## Plan (in order)

### Round 2: driving quality (next session, ~1 day, CPU + few GPU-h)
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
