# V05_PLAN.md — v0.5: camera-input DITTO-AV (SENSORS track)

Written 2026-08-06 at mission start. Read this first in any v0.5
session. Ops digest: V03_PLAN §7; project honesty rules: V031_PLAN §3
(inherited verbatim, additions in §6 here). This line lives in the
worktree /scratch/$USER/ditto_av/DITTO_AV_v05 on branch saeed/v0.5 —
NEVER work on v0.5 from the main clone or any other worktree.

## 0. Mission in one paragraph

Every DITTO-AV number so far carries the privileged-input caveat: the
policy is handed ground-truth actor states at train AND eval time,
so it competes fairly only against AD-MLP (18.05), not against the
camera-based systems (UniAD 45.81, VAD 42.35, TCP-traj 59.90,
ThinkTwice 62.44, DriveAdapter 64.22 — all DS @220). v0.5 removes the
caveat: drive Bench2Drive's SENSORS track from cameras + GNSS + IMU +
speedometer only. HEADLINE TARGET (pre-registered): beat 64.22 @220 —
the strongest published sensor baseline in our table — with the
factored offline-RL planner. STRETCH: close to the privileged
reference (v0.3.2 s1gate 79.57 @220, v0.2 999s 76.10). The paper
claim this buys: the offline factored-RL planner survives the
privileged->sensor transition and beats end-to-end camera systems,
i.e. the training-world thesis, not the privileged inputs, was the
load-bearing part.

## 1. Verified facts (all measured 2026-08-06, this repo/cluster)

Data (inspected AccidentTwoWays_Town12_Route1102_Weather10):
- Each clip carries a FULL sensor suite, frame-aligned with anno/:
  camera/{rgb,depth,semantic,instance}_{front,front_left,front_right,
  back,back_left,back_right}/ + rgb_top_down/ (322 frames in the
  sample clip), lidar/ (322), radar/ (322), expert_assessment/.
- rgb images are 1600x900 jpg, ~43 KB each.
- anno[i]["sensors"] has per-frame calibration for every camera:
  intrinsic, cam2ego, world2cam, fov, image sizes. Nothing external
  is needed for geometry.
- INODE WALL: ~8051 camera files per clip x 999 clips ≈ 8M files vs
  the 1M scratch inode quota. Camera data must NEVER be extracted
  loose on /scratch — repack to one shard file per clip (M0).
- Storage: 6 rgb cams ≈ 6 x 43KB x ~280 avg frames x 999 ≈ 72 GB;
  3 front cams ≈ 36 GB; halved again if resized 800x450 at repack.

Planner side (what perception must produce):
- The champion policies use light_obs: false — NO traffic-light
  features. The obs vector = ego row (speed) + 6 nearest actor rows
  (presence, rel pos, rel vel, cos/sin rel-yaw; extents enter only
  via collision checks, not obs) + 16-dim route block. So perception
  = BEV actor detection + velocities. Route + ego speed come from
  leaderboard-provided route / speedometer, NOT from cameras.
- Privileged references: dev-10 85.63 (v0.2 999s), @220 79.57
  (v0.3.2 s1gate, RC 99.2, 2026-08-06) and 76.10 (v0.2 999s).
- Eval infra: carla_eval_chain.sbatch boots CARLA per variant on
  participation/visual; current agent runs --track=MAP. SENSORS
  track needs a new agent config + the sensor stack (M3).

## 2. Architecture decision (and what was rejected, with evidence)

CHOSEN: factored perception, two stages, mirroring the project
thesis ("never learn what is already known; learn only what replay
cannot provide"):

  cameras -> [phi: BEV actor detector + tracker] -> the SAME state
  vector the planner already eats -> [planner: frozen champion, then
  noise-hardened retrain] -> waypoints -> PID tracker (unchanged).

- The DITTO training world stays STATE-based. Reason (structural,
  not preference): on-policy rollouts visit counterfactual ego poses;
  camera frames exist only along the expert path, and nothing on this
  cluster can re-render views from displaced poses. So cameras can
  never enter the closed training loop directly — they enter through
  (a) supervised perception on (image, anno) pairs, which needs no
  counterfactuals, and (b) the ERROR MODEL: replaying phi's measured
  error distribution as noise injection on the privileged obs inside
  EgoSim, which is exactly representable at counterfactual poses.
  The privileged->sensor bridge IS the error model. This is the
  central design idea of v0.5.
- Localization for the route frame: GNSS + IMU + speedometer fusion
  (leaderboard SENSORS allowance). Route is given by the leaderboard
  in world coords; ego pose from the fusion localizes it.

REJECTED (all with prior measured evidence, do not relitigate
without new data):
- Camera latent world model (Dreamer/DITTO-faithful): v0.1 lost to
  BC (22.10) because whole-scene latents grade traffic, and E3
  measured that latent matching cannot read ego deviation even with
  pure attribution (V031_PLAN "E3: CLOSED"). Pixels make both worse.
- Neural re-rendering (NeRF/3DGS) for counterfactual views: compute
  and engineering far beyond the lanes available; no internet on
  compute nodes for model zoo pulls.
- End-to-end BC from pixels: covariate shift is the exact failure
  this project exists to fix; v0.2 measured RL > BC by +11-13 DS.
- Lidar/radar as primary input: would keep a privileged-ish flavor
  (geometry given, not inferred) and matches no baseline row we
  compare against; cameras are the claim. Lidar MAY be used as aux
  supervision at TRAIN time only (depth targets are free anyway).

## 3. Perception design (M1 details)

Input: 3 front cameras first (front, front_left, front_right cover
the ~120 deg the planner's decisions live in; rear actors matter
mainly via rear-collision rejection which the reward already
excludes). 6-cam is a measured upgrade later if M3 shows rear misses
cost score. Resize to 800x450 at repack (jpeg bytes kept in shards).

Model: single-stage BEV center detector (CenterPoint-style, small):
- Backbone: ResNet-34-class (torchvision weights are NOT downloadable
  on compute nodes — check the login-node cache first; training from
  scratch is acceptable at this data size: ~280k frames).
- View transform: lift-splat with depth distribution over bins
  (depth cameras give a FREE dense depth supervision signal at train
  time — use it; this is a large advantage over nuScenes setups).
- BEV grid: 0.5 m cells, front 48 m, sides ±24 m, rear 8 m.
- Heads: center heatmap (vehicle/walker/bicycle), offset, yaw
  (sin/cos), extent, per-center velocity (2 frames of images as
  input, stride 0.1 s) OR tracker-derived velocity (decide in M1 by
  measuring both; tracker_torch.py exists and is proven).
- Size budget: must run at eval on the visual lane's Quadro RTX 4000
  (8 GB) alongside CARLA — target < 30 ms/frame fp16 at 3x800x450 on
  the H100 and functional on the Quadro; measure both in M1.

Supervision: annos give exact BEV boxes per frame (the same source
the privileged obs is built from — so perception is trained to
predict exactly what the planner consumes; no ontology mismatch).
Depth/semantic images are free aux losses. Occlusion labels can be
derived from instance masks if needed for the error model.

## 4. Milestones and gates (numbers pre-registered at each kickoff)

V5-M0 DATA LANE. scripts/v05_repack_shards.py: clip tar (streamed on
  node-local /tmp) -> ONE hdf5 per clip holding jpeg bytes for the
  3 front cams (resized 800x450, quality ~85), per-frame calib, and
  the frame index aligned to anno/. Output:
  /scratch/$USER/ditto_av/data/b2d_shards/<clip>.h5 (999 files).
  GATES: (a) alignment — for a 20-clip sample every shard frame k
  matches anno frame k timestamps/counts exactly; (b) size — total
  ≤ 45 GB; (c) throughput — a DataLoader on one H100 node sustains
  ≥ 1500 decoded frames/s at batch 64 (else pre-decode smaller).
V5-M1 PERCEPTION MVP + calibrated gates. First MEASURE the anno
  noise floor and actor-distance distribution on the 999-split val;
  THEN pre-register detection gates in the operating region (front
  sector ≤ 40 m — where the 6-nearest slots live 95% of the time;
  measure that number too). Provisional shape of the gates (exact
  numbers fixed at kickoff, BEFORE training): recall ≥ 0.85 within
  20 m / ≥ 0.7 at 20-40 m; position RMSE ≤ 0.5 m within 20 m; yaw
  RMSE ≤ 10 deg; velocity RMSE ≤ 0.8 m/s after tracking; runtime
  fits the Quadro. Val split only; dev-10 NEVER selects checkpoints.
V5-M2 ERROR MODEL. From M1's val predictions: miss rate, false
  positives, and error covariances conditioned on (range bin, class,
  occlusion). Deliverable: ditto_av/percep_noise.py — a sampler that
  corrupts a privileged obs batch exactly per this model (drop,
  jitter, ghost). GATE: two-sample tests — noise-injected privileged
  obs statistically match perceived obs on held-out frames (KS on
  per-slot position error marginals, miss-rate curves within CI).
V5-M3 MODULAR CLOSED-LOOP BASELINE (the honest gap measurement).
  scripts/carla_sensor_agent.py: SENSORS track config (3 cams, GNSS,
  IMU, speedometer), GNSS+IMU localization filter, phi + tracker ->
  obs -> FROZEN 999s champion -> PID. PRIVILEGE FIREWALL: the agent
  module imports no carla world-query paths; a test asserts the
  sensor config lists only allowed sensors and the agent touches
  only sensor payloads. dev-10 x3. No gate — this MEASURES the
  privileged->sensor gap (expect a big drop; that is the point).
  Also run the same agent with ground-truth obs substituted (MAP
  track) as the paired control.
V5-M4 NOISE-HARDENED PLANNER. Retrain the v0.2 recipe (or fine-tune
  the champion — run BOTH if lanes allow, one axis each) in EgoSim
  with percep_noise applied to build_obs outputs (curriculum: clean
  -> full error model). GATE: recover ≥ 50% of the M3 gap on dev-10
  under the SAME sensor agent, and lose ≤ 1 DS on the privileged
  dev-10 (the hardening must not break the clean policy).
V5-M5 UNCERTAINTY CHANNEL (optional axis, only if M4 passes with
  margin): add per-slot confidence as an obs feature; retrain; keep
  only on a dev-10 win > seed sigma.
V5-M6 DEV-10 GATE for the headline candidate. Pre-registered NOW:
  sensor-track dev-10 DS ≥ 74 with 30/30 completion, vehicle
  collisions ≤ 12, before any 220 spend. (74 ≈ DriveAdapter's 64.22
  plus the dev10-vs-220 offset our privileged pairs show: 85.63->
  76.10 = -9.5; 74 - 9.5 ≈ 64.5. Refine the offset estimate at M6
  with the v0.3.2 pair 82.80->79.57 and state the final number in
  the ledger BEFORE the 220.)
V5-M7 ONE 220 SENSORS RUN -> the headline row vs the published
  sensor baselines (re-verify their current numbers from the local
  Bench2Drive repo tables before claiming; our table is from
  paper/draft.md, checked 2026-07-31).
V5-M8 PAPER v3: "closing the privileged gap" — the factored planner
  + error-model bridge as the contribution; v0.1-v0.3.2 results as
  the foundation; the M3-vs-M4 delta is the ablation that shows the
  bridge (not just the detector) carries the transfer.

Sequencing note: M0 -> M1 -> {M2, M3} -> M4 -> M6 -> M7. M3 needs
only M1 (frozen champion, no retraining), so the first closed-loop
sensor number arrives early — keep it that way; it de-risks
everything downstream.

## 5. Ops (v0.5-specific; base ops = V03_PLAN §7)

- Worktree /scratch/$USER/ditto_av/DITTO_AV_v05, branch saeed/v0.5,
  pushed to origin. All v05 sbatch scripts cd HERE. Do not touch
  main or other worktrees; other lines are active in parallel.
- Outputs: runs -> ~/ditto_out/v05_*; shared results ->
  /scratch/$USER/ditto_av/outputs/ with v05_ prefix; ledger claims
  in outputs/PIPELINE_STATUS.md tagged "V05:". Read the tail first.
- Shards: /scratch/$USER/ditto_av/data/b2d_shards/ (999 h5 files).
  Repack jobs extract tars on node-local /tmp ONLY (inode quota).
- Lanes: perception training on participation H100 (4 h cap ->
  checkpoint-resume chained jobs from day one); MIG a100-small for
  ablations (1 job/user TOTAL); CARLA evals on participation/visual
  as today. Audit lanes before submitting; > 30 min pend = move.
- Envs: ~/envs/ditto_gpu for training (check torchvision/h5py
  presence BEFORE the first job — no internet on compute nodes;
  install from the login node into the venv if missing).
  carla_eval env runs the sensor agent: same check for the
  perception model's deps (torch is present; cv2 present; NO scipy
  anywhere — project rule).
- Job naming: v05-m0-repack, v05-m1-percep, v05-m3-d10, ... so
  squeue reads cleanly next to v0.3.2's jobs.

## 6. Honesty rules (inherited from V031_PLAN §3, plus)

- Pre-register every gate BEFORE the run it judges; on FAIL fix the
  model, never the gate; refinements only with committed
  justification.
- The training world never grades itself; CARLA dev-10/220 are the
  only verdicts. In-sim and perception-val numbers are canaries.
- Perception checkpoints are selected on val-split metrics ONLY;
  dev-10 is a gate, never a selector (30 runs cannot rank
  detectors without overfitting the benchmark).
- PRIVILEGE FIREWALL: the sensor agent may consume only the
  leaderboard sensor payloads + provided route. A committed test
  enforces the sensor list and grep-audits the agent's imports.
  Every v0.5 CARLA number states its track (SENSORS vs MAP) in the
  ledger line. No mixed-track comparisons without labels.
- Measure before building, at every stage: anno noise floor before
  detection gates; error model before noise training; gap before
  hardening targets.

## 7. Risks -> mitigations

- Inode quota blowup: shards only, /tmp extraction, 999 files total.
- BeeGFS small-read IO starving training: jpeg bytes inside h5 read
  sequentially per clip + in-memory decode; measure in M0 gate (c).
- 4 h H100 cap vs perception training time: resumable trainer +
  chained submissions from the first run (pattern exists in v0.2
  bench220 chunking).
- 8 GB Quadro at eval: model sized for it from the start (M1 gate);
  fp16; fallback = participation-only evals (slower lane budget).
- Velocity estimation too noisy -> following/merging degrades: two
  paths in M1 (2-frame head vs tracker differencing), pick by
  measurement; M2 injects the residual noise so the planner trains
  against it either way.
- GNSS/IMU localization drift corrupts the route frame: measure the
  filter error against ground truth on logged runs in M3 BEFORE
  blaming perception; the route block tolerates ~1 m (measure).
- Torchvision weights unavailable on compute nodes: verify login
  cache first; from-scratch training is the fallback (dataset is
  ~280k frames, enough for a small backbone with aux depth/semantic
  losses).
- Scope creep toward 6 cams / lidar / lights: 3 front cams, no
  lights (champion uses none), until a MEASURED miss pattern in M3
  demands more. One axis per iteration.
- Schedule realism: M0-M1 are multi-session; do not promise the 220
  before M6 passes. v0.4 (traffic LC fidelity) stays reserved and
  untouched; v0.3.2's 220 characterization continues in parallel.

## 8. Ledger

(append results with job ids here; claims go to PIPELINE_STATUS.md
tagged "V05:")

- 2026-08-06 mission start: branch saeed/v0.5 + worktree created;
  data audit facts in §1 measured (sensor suite, calib-in-anno,
  inode math, champion needs no lights); plan committed.
