# V031_PLAN.md — v0.3.1 handoff (READ THIS FIRST in a new session)

Written 2026-08-05 as the session handoff. This file + `V03_PLAN.md`
(ops digest in its §7; findings in its §9 ledger) are the guides for
continuing on `main`. Historical records: `saeed/v0.1` (v0.1 +
DELFTBLUE.md/NEXT_STEPS.md), `saeed/v0.2` (v0.2), `saeed/v0.3` (v0.3).

## 0. Project in one paragraph

DITTO-AV: offline imitation for driving where the recorded logs ARE the
training world. v0.1 (faithful latent-space DITTO) lost to BC — the
latent graded traffic, not driving. v0.2 rebuilt the thesis with an
ego-state-matching reward in a log-replay world: RL beat same-net BC by
+11–13 DS (seeds), 220-route DS 75.88/76.10 (~3.4x v0.1, above all
published Bench2Drive baselines; privileged-offline caveat). v0.3 added
a LEARNED reactive traffic model (4-round W0 fidelity campaign, all
gates eventually passed) and confirmed the reactivity dividend: vehicle
collisions 6 = all-time best (champions: 9–12), but net dev-10 82.53
missed the 83.60–85.63 band because the training world had NO STATIC
LAYOUT (7 layout collisions vs 1). v0.3.1 = fix exactly that.

## 1. Where everything stands (2026-08-05)

| line | branch | state |
|---|---|---|
| v0.1 | saeed/v0.1 | frozen; DS 22.10 @220 |
| v0.2 | saeed/v0.2 | frozen; champions 999t/999s (dev-10 83.60/85.63, 220 75.88/76.10, seeds ±7.5-9.4); paper draft in paper/ |
| v0.3 | saeed/v0.3 | frozen; W0 campaign PASSED (minADE 2.79, react 0.567), clp_rx dev-10 82.53, dividend + layout-gap banked |
| v0.3.1 | **main** (HEAD) | ONGOING — M1 done (below) |

Key checkpoints (all under ~/ditto_out/):
- v0.2 champions: b2d_v02_999s/checkpoints/clp_rl.pt (85.63),
  b2d_v02_999t/checkpoints/clp_rl.pt (83.60); configs b2d_v02_999*_cpu.yaml
- v0.3 traffic ensemble (W0-passing): v03_w0c/checkpoints/traffic_s{0..3}_rf.pt
  (+ windows3_{train,val}.npz caches, data/ npzs = 999-split ##glob2)
- v0.3 policy: b2d_v03_rx/checkpoints/clp_rx.pt (dev-10 82.53)
- Reports: v03_w0c/w0_report.json, v03_d3/d3_report.json
- Layout geometry (v0.3.1 M1): /scratch/$USER/ditto_av/data/layout/
  (Town*_lanes.npz + xodr/); query module ditto_av/layout.py
- Videos: runs/videos20 (v0.2, untracked), runs/videos20_v03 (v0.3,
  untracked); backups in ~/ditto_out/videos20*/

## 2. v0.3.1 mission

Put static layout into the training world so boldness near walls costs
what CARLA charges for it. Target: close the 7-vs-1 layout-collision
gap and beat dev-10 85.63; then ONE 220 for the headline; then paper.

### DONE (M1, commit 438d48d)
- 12 towns' OpenDRIVE extracted (scripts/v031_extract_layout.py;
  Town10=Town10HD from base SIF, Town11/12/13/15 from
  /scratch/$USER/ditto_av/maps_overlay/.../Maps/TownXX/OpenDrive/).
- Per-town lane-center+half-width npz + dependency-free grid-hash
  off-drivable query (ditto_av/layout.py; MARGIN 0.5 m, CELL 4 m).
- FIDELITY GATE PASSED: real expert trajectories 0.53% off-drivable
  (<1% pre-registered), 8/11 towns exactly 0.000% (residual =
  Town06/12/13 parking-style frames).

### DONE (M2 steps 1-3, commit 20e6873, 2026-08-05)
1. Torch layout query: ditto_av/layout_torch.py (TownLayoutTorch ==
   numpy exactly, tests pin it) + LayoutQuery per-frame town dispatch
   from layout.manifest_towns (sorted manifest, VAL_EVERY=6; Town10HD
   regexes to Town10; Town11 train-only — confirmed in the gate).
2. Sim signal: EgoSim.layout handle + layout_off; RewardParams
   layout_penalty/layout_clip (clip 3 m bounds the 99 m "no lane"
   sentinel); reactive _rollout logs layout_viol and applies the
   penalty AFTER W1 zeroing (geometry stays exact where the traffic
   model is distrusted); eval_both_worlds reports layout_viol[_any]
   metric-only. Gate through the FULL sim path PASSED:
   train 0.438% / val 0.532% expert off-drivable (scripts/
   v031_layout_gate.py; M1 raw-clip number was 0.53%).
3. D3 rerun SUBMITTED (w_layout 0.5, participation): 10581924 init
   clp_rx -> ~/ditto_out/v031_d3_rx; 10581925 init 999s champion ->
   ~/ditto_out/v031_d3_s (slurm/v031_d3.sbatch <init> <out>).

### W3 re-gate: FAIL (2026-08-05) — v0.3.1 CLOSED as a negative result

Dev-10 A/B x3 (jobs 10582571-74): arm rx 66.01 (layout 7 / veh 21),
arm s 74.89 (layout 6 / veh 24) vs clp_rx 82.53 (7/6) and 999s 85.63.
Neither arm approaches the gate. Root cause MEASURED, three links:
1. ALL dev-10 "layout" collisions are map furniture ON drivable area
   (static.fence at the 2091 junction corner, prop.mesh at the 3514
   ParkingExit, vegetation at 27494): our query reads them 0.3-1.5 m
   INSIDE the lane+margin. The training penalty was satisfied
   (lviol -> 0) while CARLA's layout collisions stayed unchanged —
   the signal cannot express the failure mode it was built for.
2. These objects exist in NO available data source: annos carry only
   vehicle/walker/bicycle (+signs), OpenDRIVE Driving lanes exclude
   fences/vegetation; nothing to replay into the world.
3. The refinement candidate (heading-conditioned drivability: only
   lanes aligned within 45 deg count) FAILS the expert pre-gate:
   2.795% val violations (>1%; Town06 10.2%, Town12 3.8%) — a turning
   expert's heading legitimately excludes both roads' corridors
   mid-junction. Wider cones re-admit the corner blindness.
Suggestive but NOT conclusive (1 training seed, ~1 sigma_train=8):
the w=0.5 penalty coincided with veh collisions 6 -> 21/24. Zero
benefit + possible harm => penalty default stays 0. KEEP the
layout_viol metric (free diagnostics); keep layout_torch/LayoutQuery
infra. Both-world in-sim reward canary IMPROVED in both arms while
CARLA dropped — logged as another "sim never grades itself" instance.

### NEXT
- E3: factored-latent DITTO retest — latent matching inside the
  (ego | learned-traffic) factorization; closes the v0.1 question.
- Paper v0.3/v0.3.1 (findings ledger in V03_PLAN §9): the v0.3
  reactivity dividend + the v0.3.1 negative result (what static
  geometry can and cannot buy in a log-replay world) are both
  publishable findings. Headline stays v0.2 220 76.10 / v0.3 dev-10
  82.53 with the collision decomposition story.

## 3. Rules that keep this project honest (do not relax)

- Pre-register gate numbers BEFORE the runs they judge; on FAIL fix
  the model, never the gate (refinements only with committed
  justification — see W0 minADE precedent in V03_PLAN §9).
- The world model / sim NEVER grades itself: CARLA dev-10/220 are the
  only verdicts that count. 3x3 is a canary, never a selector.
- Measure before building (the audit pattern); synthetic tests must
  include the failure modes (churn, curvature, braking experts).
- One axis per iteration; every result -> ledger with job id.

## 4. Ops quickstart (details: V03_PLAN §7; history: saeed/v0.1)

- Work in /scratch/$USER/ditto_av/DITTO_AV (main). The DITTO_AV_v03/
  worktree was REMOVED 2026-08-05 (its content is fully merged into
  main and preserved on branch saeed/v0.3); ignore any older notes
  that mention it.
- Login nodes: edit/submit only; compute nodes have NO internet;
  scratch has a 1M-inode quota (extractions -> node-local /tmp).
- Envs: ~/envs/ditto_gpu (GPU nodes), scratch venv ditto (login/CPU),
  carla_eval conda (CARLA + cv2; NO scipy anywhere — layout.py is
  dependency-free on purpose). OMP_NUM_THREADS=1 in jobs.
- Lanes: participation (H100, 4 h cap) + visual (Quadro) for
  CARLA/GPU; gpu-a100-small MIG = training only, 1 job/user TOTAL;
  audit lanes before submitting; >30 min pend = move.
- Multi-session protocol: CLAIM stages in outputs/PIPELINE_STATUS.md
  (tag "V03.1:"), read the tail first, never duplicate a claimed
  stage, never cancel jobs you didn't submit.
- Commits as Saeed Rahmani, NO AI attribution. Ask before deleting
  anything; compute needs no approval.
