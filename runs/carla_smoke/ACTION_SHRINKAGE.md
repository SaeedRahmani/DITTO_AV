# Offline diagnosis: action shrinkage explains the closed-loop gap (2026-07-28)

Follow-up to the frame question (settled by `scripts/replay_frame_check.py`:
yaw_offset pi/2 reproduces training obs exactly; raw yaw = 90-deg rotation).
Question here: with the CORRECT frame, why does closed-loop driving still
fail (4.5 +- 2.8 on the 3-route smoke)?

## Method

Teacher-forced posterior replay on real clips (no CARLA): featurize each
anno frame exactly as deployment does (`replay_clip`, compass frame), run
the v3 world model + ditto_multi policy, compare the policy's action to
the expert's recorded action at every frame. 460 frames over 3 clips
(HighwayCutIn/Town06, LaneChange/Town06, Accident/Town03), 185 turn
frames (|expert steer| > 0.1).

## Results

Frame comparison (mean over clips):

| convention | act MAE | NLL   | steer MAE |
|---|---|---|---|
| compass (pi/2) | 0.2025 | 0.805 | 0.1649 |
| raw yaw (0)    | 0.2106 | 1.356 | 0.1785 |

Compass wins every metric (on-manifold, as the replay test requires) —
but the margin is small: a 90-deg rotation of ALL geometry barely moves
the policy's actions. The policy's geometric coupling is weak.

Where the signal actually is (compass frame):

- corr(pred, expert): throttle +0.35, steer +0.50 (+0.56 on turn
  frames), brake +0.26 — direction is RIGHT, magnitude is not:
- turn frames: policy |steer| mu = 0.098 vs expert 0.355 (~28%);
  pred std vs expert std: throttle 0.13/0.33, steer 0.10/0.25,
  brake 0.11/0.33.
- policy sigma on turn-frame steer: 0.156 around mu 0.098 =>
  P(sample commits |steer| > 0.2 toward the turn) ~ 0.26 per tick, and
  10 Hz zero-mean noise mostly averages out in the vehicle dynamics —
  stochastic sampling alone leaves the systematic shortfall in place.

## Interpretation

Classic Gaussian mean-action shrinkage (BC anchor + NLL on multimodal
continuous controls). It explains all three road-test arms:

- correct frame: right-direction but ~28%-strength steering — cannot
  make junctions or bypass obstacles; drives cleanly (penalty 0.65) but
  wedges or drifts off-route;
- rotated frame (old bug): policy effectively geometry-blind =>
  straight-driving prior that ricochets along the route farming
  completion at penalty 0.12 (the 12.6 was a metric pathology, see
  configs/carla_agent.yaml note);
- recovery arm: unwedging re-exposes the same weak steering repeatedly
  => collision cascades (1.7).

The brake dim already gets deployment-side binarization for exactly this
pathology; steering/throttle need the analogous calibration.

## Experiment

`steer_gain`/`throttle_gain` knobs in DittoCarlaAgent (default 1.0).
Arms (3 base-town routes x 3 reps, compass, deterministic, no recovery):

- diag_gain2_norec: steer x2.0, throttle x1.5
- diag_gain3_norec: steer x3.0 (matches expert magnitude), throttle x1.5

Compare against diag_fix_norec (gain 1.0 = 4.5 +- 2.8) and
diag_stoch_norec (sampling arm). Prediction: gain arms raise completion
via made junctions; if gain3 oversteers on straights (amplified noise),
gain2 or a state-dependent gain is next.

Training-side fix for v5+ (if gains confirm): address shrinkage at the
source — e.g. temperature/scale calibration on the actor head or a
non-Gaussian action head; deployment gain is the honest short-term knob.
