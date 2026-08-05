# V031_PLAN.md — v0.3.1 handoff (READ THIS FIRST in a new session)

Written 2026-08-05 as the session handoff. This file + `V03_PLAN.md`
(ops digest in its §7; findings in its §9 ledger) are the guides for
continuing on `main`. Historical records: `saeed/ver0.1` (v0.1 +
DELFTBLUE.md/NEXT_STEPS.md), `saeed/v0_2` (v0.2), `saeed/v0.3` (v0.3).

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
| v0.1 | saeed/ver0.1 | frozen; DS 22.10 @220 |
| v0.2 | saeed/v0_2 | frozen; champions 999t/999s (dev-10 83.60/85.63, 220 75.88/76.10, seeds ±7.5-9.4); paper draft in paper/ |
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

### NEXT (M2 — the actual fix; est. one work session)
1. Torch layout query: port TownLayout.off_drivable to torch (the
   numpy grid-hash structure is designed for this) and give EgoSim an
   optional per-episode layout handle. Episode->town mapping: derive
   clip names from the manifest split (sorted names, VAL_EVERY=6 —
   see the validation snippet pattern in git history of this work) and
   regex Town\d+; NOTE Town11 has clips in train only.
2. Sim signal: per-step ego off-drivable clearance ->
   (a) metric logged in rollouts (layout_violation rate), and
   (b) penalty arm: reward -= w_layout * relu(off) (start w=0.5,
   dose axis) — mirror how collision_penalty is wired in
   egosim.RewardParams (penalty flags live there).
3. Rerun D3 (scripts/v03_train_reactive.py + slurm/v03_d3.sbatch from
   the v0.3 worktree — port them to main or run from the worktree
   after merging main->saeed/v0.3? NO: v0.3 stays frozen; copy the two
   files into main if not already merged — CHECK: they ARE on main via
   the 73c0e0f merge) with the layout signal on, init from clp_rx or
   from the 999s champion (run BOTH arms if lanes allow: layout-on
   fine-tune of each; ~2 h/job on participation H100).
4. W3 re-gate: dev-10 A/B (carla_eval_chain.sbatch, routes
   A=3514,3255,26405,25381,25378 B=25424,2091,27494,17569,28198, x3;
   diag config pattern: configs/diag_v03_rx.yaml). GATE: beat 85.63
   nominal / clear the band by > seed sigma (2.3) for a claim. Watch
   the layout-vs-vehicle collision decomposition — the whole point.
5. If cleared: seeds (s1/s2) + ONE 220 (v02_bench220_submit.sh
   pattern; collector self-commits) vs 76.10 headline.
6. Then E3 (factored-latent DITTO retest — latent matching inside the
   (ego | learned-traffic) factorization; closes the v0.1 question)
   and paper v0.3/v0.3.1 (findings ledger in V03_PLAN §9).

## 3. Rules that keep this project honest (do not relax)

- Pre-register gate numbers BEFORE the runs they judge; on FAIL fix
  the model, never the gate (refinements only with committed
  justification — see W0 minADE precedent in V03_PLAN §9).
- The world model / sim NEVER grades itself: CARLA dev-10/220 are the
  only verdicts that count. 3x3 is a canary, never a selector.
- Measure before building (the audit pattern); synthetic tests must
  include the failure modes (churn, curvature, braking experts).
- One axis per iteration; every result -> ledger with job id.

## 4. Ops quickstart (details: V03_PLAN §7; history: saeed/ver0.1)

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
