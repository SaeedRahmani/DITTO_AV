# V031_PLAN.md — v0.3.1 handoff (READ THIS FIRST in a new session)

Written 2026-08-05 as the session handoff. This file + `V03_PLAN.md`
(ops digest in its §7; findings in its §9 ledger) are the guides for
continuing on `main`. Historical records: `saeed/v0.1` (v0.1 +
DELFTBLUE.md/NEXT_STEPS.md), `saeed/v0.2` (v0.2), `saeed/v0.3` (v0.3).

## 0. Project in one paragraph

DITTO-AV: offline imitation for driving where the recorded logs ARE the
training world. v0.1 (faithful latent-space DITTO) lost to BC — the
latent graded traffic, not driving. v0.2 rebuilt the thesis with an
ego-state-matching reward in a log-replay world: DITTO-AV v0.2 beat same-net BC by
+11–13 DS (seeds), full 220-route DS 75.88/76.10 (~3.4x v0.1, above all
published Bench2Drive baselines; privileged-offline caveat). v0.3 added
a LEARNED reactive traffic model (4-round W0 fidelity campaign, all
gates eventually passed) and confirmed the reactivity dividend: vehicle
collisions 6 = all-time best (champions: 9–12), but net test-10 82.53
missed the 83.60–85.63 band because the training world had NO STATIC
LAYOUT (7 layout collisions vs 1). v0.3.1 = fix exactly that.

## 1. Where everything stands (2026-08-05)

| line | branch | state |
|---|---|---|
| v0.1 | saeed/v0.1 | frozen; DS 22.10 @220 |
| v0.2 | saeed/v0.2 | frozen; champions 999t/999s (test-10 83.60/85.63, 220 75.88/76.10, seeds ±7.5-9.4); paper draft in paper/ |
| v0.3 | saeed/v0.3 | frozen; W0 campaign PASSED (minADE 2.79, react 0.567), clp_rx test-10 82.53, dividend + layout-gap banked |
| v0.3.1 | **main** (HEAD) | ONGOING — M1 done (below) |

Key checkpoints (all under ~/ditto_out/):
- v0.2 champions: b2d_v02_999s/checkpoints/clp_rl.pt (85.63),
  b2d_v02_999t/checkpoints/clp_rl.pt (83.60); configs b2d_v02_999*_cpu.yaml
- v0.3 traffic ensemble (W0-passing): v03_w0c/checkpoints/traffic_s{0..3}_rf.pt
  (+ windows3_{train,val}.npz caches, data/ npzs = 999-split ##glob2)
- v0.3 policy: b2d_v03_rx/checkpoints/clp_rx.pt (test-10 82.53)
- Reports: v03_w0c/w0_report.json, v03_d3/d3_report.json
- Layout geometry (v0.3.1 M1): /scratch/$USER/ditto_av/data/layout/
  (Town*_lanes.npz + xodr/); query module ditto_av/layout.py
- Videos (2026-08-06, one folder per line's best model, each 10 test-10
  routes x {2d BEV over the town map, 3d chase cam, state.jsonl}):
  /scratch/$USER/ditto_av/outputs/videos/{v0.1_gen3wph_bc,
  v0.2_999s_shaped, v0.3_reactive_rx, v0.3.1_layout_s,
  v0.3.2_smooth_cons}/ — MODEL.md in each names the checkpoint and
  its per-route scores. Producer scripts/slurm/videos_best.sbatch
  (REPO/CONF/VID/LABEL/ROUTES); the 2D can be re-rendered offline from
  the state logs (scripts/render_bev_video.py, town geometry cached as
  data/layout/Town*_bev.npz). Superseded: runs/videos20{,_v03} (v0.2 /
  v0.3, no map under the BEV).

## 2. v0.3.1 mission

Put static layout into the training world so boldness near walls costs
what CARLA charges for it. Target: close the 7-vs-1 layout-collision
gap and beat test-10 85.63; then ONE 220 for the headline; then paper.

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

Test-10 A/B x3 (jobs 10582571-74): variant rx 66.01 (layout 7 / veh 21),
variant s 74.89 (layout 6 / veh 24) vs clp_rx 82.53 (7/6) and 999s 85.63.
Neither variant approaches the gate. Root cause MEASURED, three links:
1. ALL test-10 "layout" collisions are map furniture ON drivable area
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
infra. Both-world in-WM reward canary IMPROVED in both variants while
CARLA dropped — logged as another "sim never grades itself" instance.

### v0.3.1-R: REOPENED 2026-08-06 (user request) — furniture axis

Review verdict on the W3 FAIL — three distinct errors, not one:
1. WRONG GEOMETRY: the premise (missing static world) was right but
   lane-union off-drivable is not what CARLA charges for — the events
   are map-furniture clips (fence/prop/vegetation) ON drivable area.
   Independently confirmed by v0.3.2: pure smoothness got layout
   7 -> 0 with zero map data (wobble corner-clipping mechanism), at
   the cost of veh 6 -> 12 (global prior taxes reactivity).
2. WRONG DELIVERY: reward-side penalty through normalized A2C
   advantages on stochastic rollouts — v0.3.2 axis-2 measured this
   channel pricing exploration noise (10x mean churn); the veh-col
   explosion in BOTH w=0.5 variants (6 -> 21/24, two seeds agreeing) is
   the same channel failing. Working channel (v0.3.2 axis-3):
   differentiable auxiliary loss on the mean plan.
3. PREMATURELY CLOSED DATA QUESTION: "no data source has these
   objects" covered only annos+xodr. CARLA exposes the actual
   collidable furniture offline: world.get_environment_objects /
   get_level_bbs per CityObjectLabel (verified in the eval venv) —
   one server job dumps world-frame OBBs per town.

FIX (synthesis): differentiable PLAN-CLEARANCE aux loss against real
furniture boxes — mean predicted waypoints -> world frame; penalize
relu(margin - dist to nearest box). Local (active only near actual
objects), targeted (prices exactly the charged events), not through
advantages, no global smoothness tax. Complementary to v0.3.2's
proximity-gated consistency (their lane, do not duplicate).

Stage A OUTCOME (2026-08-06): AXIS CLOSED AT THE AUDIT.
- A1 (ego-center semantics, justified from scenario_runner source):
  3/3 map-furniture collision points covered — fences 1.21/1.44 m
  center-to-box (touching), vegetation inside a crown box; the 3514
  prop is a SCENARIO-SPAWNED actor (id=4948 != 0), out of map scope
  by identity (addressable events were 6/7).
- A2 FAILED at every margin and every allowed refinement: 21.1% of
  expert frames body-OVERLAP a furniture box (canopy z-filter:
  21.3 -> 19.6% only; moving vs stopped flat 23.9/22.4%; thin-box
  subset still 2.9% overlap). CARLA's exposed geometry conflates
  ground and overhang footprints in single boxes, includes objects
  whose collision CARLA disables ("overlaps a driving lane" server
  warnings), and is not a collision predicate at any subset. The
  dumps (data/layout/furniture/) and audits are kept as the record.
- Cost discipline held: the axis died offline; no training run.

CONCLUSION of the v0.3.1 review (three representations, three
measured deaths at three layers): lane-union geometry (wrong events),
heading-conditioned drivability (expert-infidelity), furniture boxes
(data not collision-faithful). The static-world CONTENT axis is
exhausted with the data CARLA exposes. The layout gap remains real
and remains FIXABLE — by the v0.3.2 smoothness mechanism (proximity-
gated consistency; already measured layout 7 -> 0), whose Pareto
tuning is the v0.3.2 session's claimed lane. For the paper: this
triple negative + the smoothness reframe is the "what static geometry
cannot buy" section.

### v0.3.1-R FINAL (2026-08-06): closed with the trade-off measured

Stage C-lite: the untested w_cons 0.25 variant (v0.3.2 axis-3 Pareto
middle, never CARLA-evaluated there) under the pre-registered gate
(DS>85.63, layout<=2, veh<=8): DS 82.12 (30/30), layout 3, veh 12 —
FAIL on all three. The consistency dose curve is now measured at
three points:

  w_cons   DS      layout  veh
  0.0      82.53   7       6      (clp_rx)
  0.25     82.12   3       12
  0.5      82.80   0       12     (v032 s2 variant)

The vehicle tax SATURATES at the first dose step while the layout
benefit accrues gradually — no passing region exists; the mechanism
trades layout for vehicle ~1:1 in DS and the reactive fine-tune
family is pinned at ~82-83 against 999s' 85.63.

v0.3.1(-R) closes having fully characterized the layout gap:
- CAUSE: steering wobble corner-clipping map furniture (not
  off-drivable driving).
- WHAT CANNOT PRICE IT (measured): lane-union geometry, heading-
  conditioned drivability, CARLA furniture boxes (21.1% expert
  body-overlap; not collision-faithful), reward-side penalties
  (price exploration noise; veh 6->21/24).
- WHAT FIXES IT AND ITS COST: differentiable plan-consistency
  (layout 7->0) at a saturating vehicle-collision tax (6->12) that
  cancels the gain.
- Champion unchanged: v0.2 999s test-10 85.63 / 220 76.10.

Candidate v0.4 mission (NOT claimed): traffic-model lane-change
fidelity. Route 17569 (SequentialLaneChange, T12) scores 100 for
clp_rx but 36/36/21 for EVERY variant fine-tuned further in the reactive
world — the learned traffic's lane-change behavior is the likely
divergence and W0 never gated it. Fixing W0-LC then re-running D3
is a coherent next campaign if the 85.63 target stays.

Next on main: the PAPER (v0.2 record + v0.3 dividend + the complete
v0.3.1/E3 negative-result chain above).

### E3: CLOSED as a measured negative (2026-08-05, no RL run needed)

A0 audit chain (wm 10584612, frozen; scripts/v031_e3_audit.py):
- RAW latent space: FAIL on dynamic range — 2 m ego offset moves
  max_cos only 0.952->0.929 (0.976x, gate <=0.75x) where the state
  kernel gives 0.135x. Monotone + control PASS (results/a0_audit.json).
- Pre-registered refinement (ledger 17:50, the ONE allowed): ridge
  wp-probe h->wp on train fits R^2 0.9597 — the latent LINEARLY
  CONTAINS the expert plan — yet the audit in probe space on val
  FAILS ALL THREE gates (control 0.7525<0.95: the sim obs path
  destabilizes the wp-relevant subspace even at zero offset;
  non-monotone at 4 m; 2 m drop 3% vs required 25%)
  (results/a0_audit_probe.json).
CONCLUSION (closes the v0.1 question, measured): even inside the
(ego | shared-traffic) factorization that grants pure ego
attribution, latent MATCHING has ~zero usable dynamic range for ego
deviation — the information is present but the similarity channel
cannot read it. v0.1's failure was not just the dreamed world: latent
matching as a reward is dead on its own axis. Per pre-registration,
both factorization axes dead -> E3 closes; ego-STATE matching stays
the load-bearing reward. Next on main: the paper.

### E3 design record (for the paper; audits above overrule step 4-5)

The v0.1 question, isolated: v0.1 changed two things at once vs what
works — the world (dreamed) and the reward (latent matching). v0.2/3
fixed both. E3 keeps the v0.3 world EXACTLY and swaps ONLY the reward
back to latent matching (one axis). The factorization is inherent to
this world: at matched frames sim and expert share the SAME traffic
(replayed or same evolved buffer), so a latent difference is
ego-attributable by construction — the property v0.1's dreamed
full-scene latent lacked (gen-4 finding: similarity to any plausible
traffic 0.85-0.92).

Recipe (reuses v0.1 machinery, all still on main):
1. Train VectorWorldModel on the 999-split b2d_train obs/wp stream
   (wm_trainer.train_world_model; frozen after).
2. Latent lane: teacher-force the log through the posterior
   (build_latent_bank pattern) -> per-frame expert h_t. In rollouts,
   run the frozen wm alongside the policy on the SIM obs/action
   stream; reward_t = max_cos(h_sim_t, h_exp_{t+-d}) max over
   |d|<=tau (the same time tolerance as the state kernel), replacing
   sim.reward. No negatives to start (shared traffic should remove
   the generic-driving floor — verify in the audit).
3. E3-A0 offline audit BEFORE the RL run (measure-before-building):
   (a) expert replay scores ~1; (b) ego-deviation sensitivity — 
   rebuild obs at laterally offset ego (0.5/1/2/4 m) at the same
   frames: latent reward must fall monotonically with offset and
   the matched-vs-mismatched window gap must be large where the
   state kernel's is (if the latent cannot see a 1 m ego offset
   through shared traffic, E3 dies here and that IS the v0.1 answer,
   measured).
4. E3-L variant: D3 protocol unchanged (init 999s champion, reactive
   p_r 0.5, W1 pessimism, KL anchor, 3500 steps) with the latent
   reward. Direct counterpart: clp_rx (state reward, same protocol,
   test-10 82.53).
5. GATE (pre-registered): test-10 A/B x3. |E3L - 82.53| <= 2.3 (seed
   sigma) => latent matching ~ state matching in a factored world
   (v0.1's failure was the world/latent content, not latent matching
   per se); < 80.23 => explicit state matching is load-bearing;
   > 85.63 => upgrade candidate (then seeds + 220 per M2 step 5).

Also NEXT: paper v0.3/v0.3.1 (findings ledger in V03_PLAN §9): the
reactivity dividend + the v0.3.1 negative result (what static
geometry can and cannot buy in a log-replay world) are both
publishable. Headline stays v0.2 220 76.10 / v0.3 test-10 82.53 with
the collision decomposition story.

## 3. Rules that keep this project honest (do not relax)

- Pre-register gate numbers BEFORE the runs they judge; on FAIL fix
  the model, never the gate (refinements only with committed
  justification — see W0 minADE precedent in V03_PLAN §9).
- The world model / sim NEVER grades itself: CARLA test-10/220 are the
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
