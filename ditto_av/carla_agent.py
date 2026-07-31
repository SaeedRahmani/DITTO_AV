"""Online featurizer + closed-loop driver for CARLA (privileged planner).

Builds, live from the CARLA world, the same observation the offline
adapter (`bench2drive.py`) builds from annotations — verified equal by
`tests/test_carla_agent.py` — and drives with a trained world model +
continuous policy, mirroring `evaluate.WMPolicyDriver`.

The `carla` package is imported only inside `DittoCarlaAgent`; the
featurizer and driver are plain numpy/torch and fully testable offline.
Positioning: privileged planner (ground-truth actors + route), NOT a
sensor-based agent — compare against privileged baselines only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .bench2drive import (FPS, LIGHT_DIMS, LIGHT_STATES, N_COMMANDS,
                          POS_SCALE, ROUTE_DIMS, V_MAX, VEL_SCALE,
                          WP_SCALE, WP_STRIDE, extra_obs_layout)


def featurize_frame(ego_xy: np.ndarray, ego_yaw: float, ego_speed: float,
                    actors: Sequence[Tuple[object, np.ndarray, float]],
                    prev_xy: Dict[object, np.ndarray],
                    n_neighbors: int = 6, radius: float = 60.0,
                    route: Optional[dict] = None,
                    light: Optional[dict] = None
                    ) -> Tuple[np.ndarray, Dict[object, np.ndarray]]:
    """One observation from world state; mirrors bench2drive.load_clip.

    ego_yaw must be in the annotation convention: Bench2Drive's theta,
        which is the IMU compass = CARLA yaw + pi/2 (see run_step).
    actors: (actor_id, world_xy(2,), yaw_rad) for vehicles/walkers/bikes,
        excluding the ego. prev_xy: last tick's positions for velocity
        finite-differencing (pass the returned dict back next tick).
    route: {"near_xy", "far_xy", "near_cmd", "far_cmd"} or None.
    light: {"xy": trigger-volume world xy or None, "state": int} to append
        the 6-dim light block (None omits the block entirely; "xy" None
        means block present but no relevant light -> zeros).
    Returns (obs vector, current positions dict).
    """
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    world_to_ego = np.array([[c, s], [-s, c]])
    n_feat = 7
    rows = np.zeros((1 + n_neighbors, n_feat), dtype=np.float32)
    rows[0] = [1.0, 0.0, 0.0, float(ego_speed) / VEL_SCALE, 0.0, 1.0, 0.0]

    cur_xy: Dict[object, np.ndarray] = {}
    neighbors: List[Tuple[float, list]] = []
    for actor_id, pos, yaw in actors:
        pos = np.asarray(pos, dtype=np.float64)
        cur_xy[actor_id] = pos
        rel = world_to_ego @ (pos - ego_xy)
        dist = float(np.linalg.norm(rel))
        if dist > radius or dist < 1e-6:
            continue
        prev = prev_xy.get(actor_id)
        vel_w = (pos - prev) * FPS if prev is not None else np.zeros(2)
        if np.linalg.norm(vel_w) > V_MAX:
            vel_w = np.zeros(2)
        vel_rel = world_to_ego @ vel_w
        yaw_rel = float(yaw) - ego_yaw
        neighbors.append((dist, [
            1.0, rel[0] / POS_SCALE, rel[1] / POS_SCALE,
            vel_rel[0] / VEL_SCALE, vel_rel[1] / VEL_SCALE,
            float(np.cos(yaw_rel)), float(np.sin(yaw_rel))]))
    neighbors.sort(key=lambda x: x[0])
    for i, (_, row) in enumerate(neighbors[:n_neighbors]):
        rows[1 + i] = row

    parts = [np.clip(rows, -2.0, 2.0).reshape(-1)]
    if route is not None:
        block = np.zeros(ROUTE_DIMS, dtype=np.float32)
        for i, tag in enumerate(("near", "far")):
            base = i * (2 + N_COMMANDS)
            xy = route.get(f"{tag}_xy")
            if xy is not None and np.isfinite(xy).all():
                rel = world_to_ego @ (np.asarray(xy, dtype=np.float64)
                                      - ego_xy)
                block[base:base + 2] = rel / POS_SCALE
            cmd = int(route.get(f"{tag}_cmd", 4) or 4)
            if 1 <= cmd <= N_COMMANDS:
                block[base + 2 + cmd - 1] = 1.0
        parts.append(np.clip(block, -2.0, 2.0))
    if light is not None:
        lb = np.zeros(LIGHT_DIMS, dtype=np.float32)
        xy = light.get("xy")
        if xy is not None and np.isfinite(xy).all():
            lb[0] = 1.0
            rel = world_to_ego @ (np.asarray(xy, dtype=np.float64) - ego_xy)
            lb[1:3] = rel / POS_SCALE
            state = int(light.get("state", -1))
            if 0 <= state < LIGHT_STATES:
                lb[3 + state] = 1.0
        parts.append(np.clip(lb, -2.0, 2.0))
    if len(parts) == 1:
        return parts[0], cur_xy
    return np.concatenate(parts), cur_xy


class ContinuousWMDriver:
    """Closed-loop continuous-control driver: posterior filtering + policy.

    The continuous counterpart of evaluate.WMPolicyDriver; consumes obs
    vectors from `featurize_frame` and returns (throttle, steer, brake).
    """

    def __init__(self, wm, policy, action_dim: int = 3,
                 device: str = "cpu", stochastic: bool = False,
                 external_feedback: bool = False):
        self.wm, self.policy = wm, policy
        self.action_dim = action_dim
        self.device = device
        self.stochastic = stochastic
        # wp_head mode: the policy's output (waypoints) is NOT the WM's
        # action; the agent sets set_executed() with the control that
        # actually drove the vehicle (training-consistent feedback)
        self.external_feedback = external_feedback
        self.reset()

    def reset(self):
        self.state = None
        self.prev_action: Optional[torch.Tensor] = None

    def set_executed(self, control: Sequence[float]):
        self.prev_action = torch.as_tensor(control, dtype=torch.float32,
                                           device=self.device)

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32,
                                device=self.device).view(1, 1, -1)
        first = self.state is None
        if first:
            self.state = self.wm.init_state(1)
            act_t = torch.zeros((1, 1, self.action_dim), device=self.device)
        else:
            act_t = self.prev_action.view(1, 1, -1)
        reset_t = torch.tensor([[first]], dtype=torch.bool,
                               device=self.device)
        feat, _, self.state = self.wm.observe(obs_t, act_t, reset_t,
                                              self.state)
        a = self.policy.act(feat[0, 0], stochastic=self.stochastic)
        if not self.external_feedback:
            self.prev_action = a
        return a.cpu().numpy()


class RouteCursor:
    """Stateful pop-radius route follower for the near/far conditioning.

    Reproduces the data collector's two command planners, whose
    semantics were measured from 2.6k annotation frames (2026-07-28):
    near = dense-plan node (spacing 1-2 m) popped at ~4 m, hovering
    4-6 m ahead; far = downsampled command-plan node (spacing 20-50 m)
    popped at ~7.5 m. The previous `plan_to_command_points`
    (change-point + arc-length fallbacks over the sparse plan) produced
    near points tens of meters lateral/behind the ego — nothing like
    the training distribution. The index only moves forward, so
    self-crossing routes cannot teleport the target backwards.
    """

    def __init__(self, xy, cmds: Sequence[int], pop_radius: float):
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        self.cmds = [int(c) for c in cmds]
        self.pop_radius = float(pop_radius)
        self.i = 0

    def step(self, ego_xy: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """Advance past reached nodes; return (node_xy, command)."""
        if not len(self.xy):
            return None, 4
        while self.i < len(self.xy) - 1:
            node = self.xy[self.i]
            if np.linalg.norm(node - ego_xy) < self.pop_radius:
                self.i += 1
                continue
            # the expert always passes within pop_radius of every node,
            # but our policy can run laterally offset (routefix 3x3: the
            # cursor stalled and near pointed 14 m BACKWARDS). Also pop
            # nodes the ego has passed along the local route direction;
            # on-plan this never fires — the radius pop wins first.
            seg = self.xy[self.i + 1] - node
            if seg @ (ego_xy - node) > 0:
                self.i += 1
                continue
            break
        return self.xy[self.i], self.cmds[self.i]


def route_hits_box(points_xy: np.ndarray, center_xy: np.ndarray,
                   extent_xy: np.ndarray, multiplier: float = 1.5) -> bool:
    """Do any route points fall inside an axis-aligned trigger box?

    Mirrors Bench2Drive data_collect._point_inside_boundingbox — which
    ignores the box rotation and uses strict inequalities — so online
    light relevance reproduces the annotation's `affects_ego` exactly.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if not len(pts):
        return False
    c = np.asarray(center_xy, dtype=np.float64)
    e = np.asarray(extent_xy, dtype=np.float64) * multiplier
    return bool((np.abs(pts - c) < e).all(axis=1).any())


class StuckRecovery:
    """Deployment-side unwedging, counted in model ticks (config-gated).

    The universal closed-loop failure (tuning round, runs/carla_smoke) is
    wedging against an obstacle with the policy commanding full throttle
    forever. If ego speed stays below `speed` for more than `stuck_ticks`
    ticks while throttle (not brake) is commanded, take over for
    `recover_ticks` of straight, gentle reverse, then hand control back.
    Braking at a red light never counts as stuck.

    Conservative by design (the 2026-07-28 re-baseline showed aggressive
    reverse+steer loops rack up layout collisions and route deviations):
    reverse straight rather than steering blind, and after
    `max_consecutive` recoveries with no real movement in between
    (never exceeding `free_speed`), give up and let the leaderboard's
    blocked-detection end the route cleanly.
    """

    def __init__(self, speed: float = 0.3, stuck_ticks: int = 40,
                 recover_ticks: int = 10, throttle: float = 0.3,
                 steer: float = 0.0, free_speed: float = 1.0,
                 max_consecutive: int = 3):
        self.speed, self.stuck_ticks = speed, stuck_ticks
        self.recover_ticks = recover_ticks
        self.throttle, self.steer = throttle, steer
        self.free_speed = free_speed
        self.max_consecutive = max_consecutive
        self.low_ticks = 0
        self.left = 0
        self.events = 0
        self.consecutive = 0
        self._moved_free = True
        self._steer_sign = 1.0

    def update(self, ego_speed: float, throttle: float, brake: float,
               steer: float) -> Optional[Tuple[float, float, bool]]:
        """One model tick: returns None to keep the policy's control, or
        a (throttle, steer, reverse) override while recovering."""
        if ego_speed > self.free_speed:
            self._moved_free = True
            self.consecutive = 0
        if self.left == 0:
            if ego_speed < self.speed and throttle > 0.0 and brake == 0.0:
                self.low_ticks += 1
            else:
                self.low_ticks = 0
            if self.low_ticks > self.stuck_ticks:
                self.low_ticks = 0
                self.consecutive = 1 if self._moved_free \
                    else self.consecutive + 1
                if self.consecutive > self.max_consecutive:
                    return None  # give up: let blocked-detection end it
                self.left = self.recover_ticks
                self._steer_sign = 1.0 if steer >= 0.0 else -1.0
                self._moved_free = False
                self.events += 1
        if self.left > 0:
            self.left -= 1
            return (self.throttle, -self._steer_sign * self.steer, True)
        return None


def wp_to_vehicle(wp_flat: np.ndarray) -> np.ndarray:
    """Predicted waypoint action -> vehicle-frame points in meters.

    The policy emits waypoints in the training (compass) frame, scaled
    by 1/WP_SCALE. The compass frame is CARLA yaw + pi/2, so vehicle
    coords are a fixed +90 deg rotation of compass coords: FORWARD
    (vehicle +x) = compass -y, LATERAL (vehicle +y, right) = compass +x
    — the settled Phase-0c frame fact. Round-trip against the offline
    future_waypoints construction is enforced by tests/test_waypoints.py.
    """
    wp = np.asarray(wp_flat, dtype=np.float64).reshape(-1, 2) * WP_SCALE
    return np.stack([-wp[:, 1], wp[:, 0]], axis=1)


class WaypointTracker:
    """Tracks the policy's predicted waypoints (Phase-1 deployment).

    Stateless per tick except the config-gated creep counter: every
    model tick gets a fresh 3 s plan in the vehicle frame, so unlike
    RoutePIDDriver there is no route index to advance. Steering is pure
    pursuit at a speed-scaled lookahead along the predicted polyline;
    the target speed comes from the plan's own spacing (the expert's
    intended travel over the first second — the TCP-style speed source),
    capped by curvature. Gains default to the RoutePIDDriver values that
    scored 100.00 smoke / 94.00 dev-10 in Phase-0d. No privileged
    gating: stopping for actors/lights must come from the learned
    waypoints (the obs already contains both). Pure numpy — unit-tested
    without CARLA.
    """

    def __init__(self, v_max: float = 8.0, a_lat: float = 2.5,
                 kp_speed: float = 0.5, kp_steer: float = 1.6,
                 lookahead_min: float = 3.0, lookahead_k: float = 1.0,
                 speed_gain: float = 1.0, stride_s: float = WP_STRIDE / FPS,
                 creep_after: int = 0, creep_throttle: float = 0.4,
                 ema: float = 0.0):
        self.v_max, self.a_lat = v_max, a_lat
        self.kp_speed, self.kp_steer = kp_speed, kp_steer
        self.lmin, self.lk = lookahead_min, lookahead_k
        self.speed_gain = speed_gain
        self.stride_s = stride_s
        # ema > 0 low-pass filters the plan across ticks (weight on the
        # PREVIOUS smoothed plan): fresh per-tick predictions jitter and
        # the 3x3 probe showed steer oscillation (8-13 sign flips/100
        # ticks on the bad runs); at 0.1 s/model-tick, 0.5 ~ 0.2 s lag
        self.ema = float(ema)
        self._plan: Optional[np.ndarray] = None
        # creep_after > 0: after that many consecutive commanded-stop
        # ticks at standstill, apply creep_throttle (the standard
        # Bench2Drive team-code unblock heuristic). Off by default —
        # measure the pure tracker first.
        self.creep_after = creep_after
        self.creep_throttle = creep_throttle
        self._still = 0

    def act(self, wp_vehicle: np.ndarray, ego_speed: float):
        """-> (throttle, steer, brake, dbg). wp_vehicle: (k, 2) meters."""
        wp_vehicle = np.asarray(wp_vehicle, dtype=float)
        if self.ema > 0.0:
            if self._plan is not None:
                # plain EMA in the vehicle frame: unbiased in cruise
                # (the plan rolls with the car, so it is stationary in
                # this frame; motion-compensating would shift it back
                # ema/(1-ema) ticks = a ~1 m standing bias at 6 m/s).
                # Toward a FIXED world point (stop line) it lags by at
                # most one tick of travel, vanishing at standstill.
                wp_vehicle = self.ema * self._plan \
                    + (1.0 - self.ema) * wp_vehicle
            self._plan = wp_vehicle
        pts = np.vstack([np.zeros(2), wp_vehicle])
        seg = np.diff(pts, axis=0)
        seglen = np.linalg.norm(seg, axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seglen)])

        # target speed = planned travel over the first two strides (1 s);
        # a stopping expert collapses the spacing -> v_t -> 0
        n_sp = min(2, len(seglen))
        v_wp = self.speed_gain * arc[n_sp] / (n_sp * self.stride_s)

        # curvature cap from heading change along the usable segments
        # (repeated end points at standstill carry no heading)
        keep = seglen > 0.3
        segk = seg[keep]
        v_curve = self.v_max
        if len(segk) >= 2:
            h = np.unwrap(np.arctan2(segk[:, 1], segk[:, 0]))
            kappa = abs(h[-1] - h[0]) / max(arc[-1], 1e-3)
            v_curve = float(np.clip(np.sqrt(self.a_lat / max(kappa, 1e-4)),
                                    1.5, self.v_max))
        v_t = min(v_wp, v_curve, self.v_max)

        # pure pursuit at arc-length lookahead, interpolated on the
        # polyline (predicted points are up to ~5 m apart at speed)
        steer = 0.0
        alpha = 0.0
        if arc[-1] > 0.5:
            ld = min(max(self.lmin, self.lk * ego_speed), arc[-1])
            j = int(np.searchsorted(arc, ld, side="right") - 1)
            j = min(j, len(seglen) - 1)
            frac = (ld - arc[j]) / max(seglen[j], 1e-6)
            tp = pts[j] + min(frac, 1.0) * seg[j]
            alpha = float(np.arctan2(tp[1], max(tp[0], 0.3)))
            steer = float(np.clip(self.kp_steer * alpha, -1.0, 1.0))

        err = v_t - ego_speed
        if v_t < 0.15 and ego_speed < 1.0:
            throttle, brake = 0.0, 1.0
        elif err >= 0.0:
            throttle = float(np.clip(self.kp_speed * err, 0.0, 0.75))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-0.6 * err, 0.0, 1.0)) if err < -1.0 \
                else 0.0

        if self.creep_after > 0:
            if ego_speed < 0.3 and v_t < 0.15:
                self._still += 1
            else:
                self._still = 0
            if self._still > self.creep_after:
                throttle, brake = self.creep_throttle, 0.0
        return throttle, steer, brake, {"v_t": round(v_t, 2),
                                        "v_wp": round(float(v_wp), 2),
                                        "alpha": round(alpha, 3)}


class RoutePIDDriver:
    """Privileged route-following reference controller (Phase-0d).

    Bounds what waypoint-tracking control achieves on the benchmark,
    independent of any learned policy: pure-pursuit steering on the
    dense route plan, curvature-limited target speed, and privileged
    gating on lead actors and red lights. Expected to drive the
    non-obstacle scenario families near-perfectly and to wedge (by
    design — it never leaves the route lane) where the route is
    blocked; that contrast isolates control quality from decision
    quality. Also the PID gains calibrated here carry into the Phase-1
    waypoint tracker. Pure numpy — unit-testable without CARLA.

    All geometry uses raw CARLA yaw (right-handed steer sign); the
    training-frame compass offset is irrelevant here.
    """

    def __init__(self, plan_xy, v_max: float = 6.5, a_lat: float = 2.5,
                 kp_speed: float = 0.5, kp_steer: float = 1.6,
                 lookahead_min: float = 3.0, lookahead_k: float = 1.0,
                 stop_wall: float = 6.0, corridor_half: float = 1.7):
        self.plan = np.asarray(plan_xy, dtype=float)
        seg = np.diff(self.plan, axis=0)
        self._arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        self.v_max, self.a_lat = v_max, a_lat
        self.kp_speed, self.kp_steer = kp_speed, kp_steer
        self.lmin, self.lk = lookahead_min, lookahead_k
        self.stop_wall = stop_wall
        self.corridor_half = corridor_half
        self._idx = 0

    def _advance(self, ego_xy):
        w = self.plan[self._idx:self._idx + 60]
        self._idx += int(np.argmin(np.linalg.norm(w - ego_xy, axis=1)))
        return self._idx

    def _lookahead_point(self, i, dist):
        target_arc = self._arc[i] + dist
        j = int(np.searchsorted(self._arc, target_arc))
        return self.plan[min(j, len(self.plan) - 1)]

    def _curvature_speed(self, i):
        """Speed cap from heading change over the next ~12 m of plan."""
        j = int(np.searchsorted(self._arc, self._arc[i] + 12.0))
        j = min(j, len(self.plan) - 1)
        if j <= i + 1:
            return self.v_max
        seg = np.diff(self.plan[i:j + 1], axis=0)
        keep = np.linalg.norm(seg, axis=1) > 1e-3
        seg = seg[keep]
        if len(seg) < 2:
            return self.v_max
        h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        arc = max(self._arc[j] - self._arc[i], 1e-3)
        kappa = abs(h[-1] - h[0]) / arc
        return float(np.clip(np.sqrt(self.a_lat / max(kappa, 1e-4)),
                             1.5, self.v_max))

    def act(self, ego_xy, ego_yaw, ego_speed, actors,
            light_dist=None, light_state=None):
        """-> (throttle, steer, brake, dbg). actors: [(id, xy, yaw)];
        light_dist/state: from the relevant-light probe (0=Red 1=Yellow
        2=Green)."""
        i = self._advance(np.asarray(ego_xy, dtype=float))
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        rot = np.array([[c, -s], [s, c]])

        # lateral: pure pursuit on a speed-scaled lookahead
        ld = max(self.lmin, self.lk * ego_speed)
        tp = rot @ (self._lookahead_point(i, ld) - ego_xy)
        alpha = float(np.arctan2(tp[1], max(tp[0], 0.3)))
        steer = float(np.clip(self.kp_steer * alpha, -1.0, 1.0))

        # longitudinal target: curvature cap, lead-actor gap, red light
        v_t = self._curvature_speed(i)
        gap = None
        horizon = max(self.stop_wall, ego_speed * 2.2)
        for _, a_xy, _a_yaw in actors:
            rel = rot @ (np.asarray(a_xy, dtype=float) - ego_xy)
            if -1.0 < rel[0] < horizon and abs(rel[1]) < self.corridor_half:
                gap = rel[0] if gap is None else min(gap, rel[0])
        if gap is not None:
            # linear ramp: full stop at stop_wall behind the lead
            v_t = min(v_t, max(0.0, 0.6 * (gap - self.stop_wall)))
        if (light_state in (0, 1) and light_dist is not None
                and light_dist < max(12.0, ego_speed * ego_speed / 4.0)):
            v_t = 0.0 if light_dist < 18.0 else min(v_t, 2.0)

        err = v_t - ego_speed
        if v_t < 0.15 and ego_speed < 1.0:
            throttle, brake = 0.0, 1.0
        elif err >= 0.0:
            throttle, brake = float(np.clip(self.kp_speed * err, 0.0, 0.75)), 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-0.6 * err, 0.0, 1.0)) if err < -1.0 else 0.0
        return throttle, steer, brake, {"v_t": round(v_t, 2),
                                        "gap": None if gap is None
                                        else round(float(gap), 1),
                                        "alpha": round(alpha, 3)}


try:  # pragma: no cover - requires the carla package + leaderboard on path
    import carla
    from leaderboard.autoagents.autonomous_agent import (AutonomousAgent,
                                                         Track)
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    def get_entry_point():
        return "DittoCarlaAgent"

    class DittoCarlaAgent(AutonomousAgent):
        """Bench2Drive closed-loop agent (privileged planner track).

        Reads ground-truth actors and the route plan (privileged — see
        PAPER_PLAN positioning), builds the training-time observation via
        featurize_frame, and drives with ContinuousWMDriver. The world
        model runs at 10 Hz (annotation rate); with the 20 Hz leaderboard
        tick we repeat each control `action_repeat` frames.
        """

        def setup(self, path_to_conf_file):
            import yaml

            from .config import load_config
            from .models.nets import make_actor_critic
            from .trainers.wm_trainer import load_world_model

            self.track = Track.MAP
            # Bench2Drive's evaluator appends '+<save_name>' to the config
            # path before calling setup(); strip it
            conf_path = str(path_to_conf_file).split("+")[0]
            conf = yaml.safe_load(open(conf_path))
            # Phase-0d reference mode: privileged route-PID controller,
            # no world model / policy involved (RoutePIDDriver docstring)
            self._route_pid_conf = (conf.get("route_pid") or None) \
                if isinstance(conf.get("route_pid"), dict) \
                else ({} if conf.get("route_pid") else None)
            self._route_pid = None  # built on set_global_plan
            if self._route_pid_conf is not None:
                self._dense_plan_xy = getattr(self, "_dense_plan_xy", None)
                self._near_cursor = getattr(self, "_near_cursor", None)
                self._far_cursor = getattr(self, "_far_cursor", None)
                self._light_boxes = {}
                self._repeat = int(conf.get("action_repeat", 2))
                self._prev_xy = {}
                self._step = -1
                self._last = carla.VehicleControl()
                import os as _os
                self._log_path = _os.environ.get("DITTO_AGENT_LOG")
                if self._dense_plan_xy is not None:
                    self._route_pid = RoutePIDDriver(
                        self._dense_plan_xy, **self._route_pid_conf)
                return
            run_cfg = load_config(conf["run_config"])
            self._cfg = run_cfg
            wm = load_world_model(run_cfg, run_cfg.env.obs_dim)
            policy = make_actor_critic(
                True, run_cfg.wm.feature_dim, run_cfg.env.policy_action_dim,
                run_cfg.ac.hidden_dim, run_cfg.ac.layers,
                action_space=("waypoints" if run_cfg.env.wp_out
                              else run_cfg.env.action_space))
            # Phase-1 waypoint abstraction: the policy predicts future
            # ego-frame waypoints; a PID tracker turns them into control.
            # wp_head additionally keeps the WM on control actions and
            # feeds back the EXECUTED control (set_executed below).
            self._wp_mode = run_cfg.env.wp_out
            self._wp_head = run_cfg.env.wp_head
            self._tracker = WaypointTracker(**(conf.get("tracker") or {})) \
                if self._wp_mode else None
            import torch as _torch
            ckpt = (run_cfg.dirs()["ckpt"]
                    / f"{conf.get('policy', 'ditto_multi')}.pt")
            policy.load_state_dict(_torch.load(ckpt, map_location="cpu"))
            policy.eval()
            self._driver = ContinuousWMDriver(
                wm, policy, run_cfg.env.action_dim,
                stochastic=bool(conf.get("stochastic", False)),
                external_feedback=self._wp_head)
            self._with_route, self._with_lights = extra_obs_layout(
                run_cfg.env.extra_obs_dims, run_cfg.env.light_obs)
            # set_global_plan can run before setup(); keep its state
            self._dense_plan_xy = getattr(self, "_dense_plan_xy", None)
            self._near_cursor = getattr(self, "_near_cursor", None)
            self._far_cursor = getattr(self, "_far_cursor", None)
            self._light_boxes: Dict[int, Optional[tuple]] = {}
            self._recovery = None
            if bool(conf.get("stuck_recovery", False)):
                self._recovery = StuckRecovery(
                    stuck_ticks=int(conf.get("stuck_ticks", 40)),
                    recover_ticks=int(conf.get("recover_ticks", 15)))
            # heading offset added to CARLA yaw when featurizing; the
            # annotation frame is the IMU compass = yaw + pi/2, so pi/2
            # matches training. Overridable for frame-convention A/B
            # diagnostics only.
            self._yaw_off = float(conf.get("yaw_offset", np.pi / 2))
            self._repeat = int(conf.get("action_repeat", 2))
            # expert brake is near-binary (0 almost always, 1 sometimes);
            # the Gaussian mean outputs a constant ~0.15 which in CARLA
            # drags enough brake torque to hold the car against full
            # throttle (verified: run 10522236 never exceeded 0.24 m/s at
            # 0.8 throttle / 0.15 brake). Binarize at deployment.
            self._brake_threshold = float(conf.get("brake_threshold", 0.5))
            # deployment-side action calibration: the Gaussian mean is
            # shrunken vs the expert (offline probe 2026-07-28: on turn
            # frames policy |steer| mu 0.098 vs expert 0.355, corr +0.56;
            # throttle std 0.13 vs 0.33) — too timid to make junctions in
            # closed loop. Gains rescale the commanded control (the raw
            # action still feeds the world model, same precedent as the
            # brake binarization above). 1.0 = off.
            self._steer_gain = float(conf.get("steer_gain", 1.0))
            self._throttle_gain = float(conf.get("throttle_gain", 1.0))
            self._prev_xy: Dict[object, np.ndarray] = {}
            self._step = -1
            self._last = carla.VehicleControl()
            # per-model-tick behavior log (jsonl) for debugging closed-loop
            import os as _os
            self._log_path = _os.environ.get("DITTO_AGENT_LOG")

        def sensors(self):
            return [{"type": "sensor.speedometer", "id": "speed",
                     "reading_frequency": 20}]

        def set_global_plan(self, global_plan_gps, global_plan_world_coord):
            # the base class downsamples the stored plan to ~50 m spacing;
            # light relevance needs the dense (~1 m) input plan — trigger
            # boxes are only ~4x2 m, sparse points would miss them
            super().set_global_plan(global_plan_gps,
                                    global_plan_world_coord)
            self._dense_plan_xy = np.asarray(
                [[t.location.x, t.location.y]
                 for t, _ in global_plan_world_coord])
            # near/far command cursors: pop radii measured from the
            # annotations (4.0 m dense / 7.5 m sparse)
            self._near_cursor = RouteCursor(
                self._dense_plan_xy,
                [int(opt.value) for _, opt in global_plan_world_coord],
                4.0)
            self._far_cursor = RouteCursor(
                [[t.location.x, t.location.y]
                 for t, _ in self._global_plan_world_coord],
                [int(opt.value) for _, opt in self._global_plan_world_coord],
                7.5)

        def _trigger_box(self, cmap, lt):
            """Trigger volume walked forward to its junction.

            Port of the waypoint walk in Bench2Drive data_collect
            get_actor_filter_traffic_light: step 0.5 m along the lane
            from the trigger volume, stopping at the last waypoint before
            the junction. Returns (tv_world_xy, box_center_xy,
            box_extent_xy), or None for dead-end lanes (skipped there
            too). Static per light — cached by actor id.
            """
            tvw = lt.get_transform().transform(lt.trigger_volume.location)
            wp = cmap.get_waypoint(tvw)
            for _ in range(400):  # the reference walk is unbounded; ~200 m
                if wp.is_junction:
                    break
                nxt = wp.next(0.5)
                if not nxt:
                    return None
                if nxt[0].is_junction:
                    break
                wp = nxt[0]
            else:
                return None
            ext = lt.trigger_volume.extent
            return (np.array([tvw.x, tvw.y]),
                    np.array([wp.transform.location.x,
                              wp.transform.location.y]),
                    np.array([ext.x, ext.y]))

        def _relevant_light(self, world, cmap, ego, ego_xy):
            """most_affect_light port == the annotation's `affects_ego`.

            A light is relevant when the next ~50 m of the dense route
            passes through its walked trigger box; among those, ahead of
            the ego and nearest by trigger-volume distance (within 100 m,
            data_collect's DIS_LIGHT_SAVE).
            """
            if self._dense_plan_xy is None or not len(self._dense_plan_xy):
                return None
            cur = int(np.argmin(np.linalg.norm(
                self._dense_plan_xy - ego_xy, axis=1)))
            ahead = self._dense_plan_xy[cur:cur + 50]
            fwd = ego.get_transform().get_forward_vector()
            best, best_d = None, 101.0
            for lt in world.get_actors().filter("*traffic_light*"):
                loc = lt.get_location()
                if np.hypot(loc.x - ego_xy[0], loc.y - ego_xy[1]) > 100.0:
                    continue
                if lt.id not in self._light_boxes:
                    self._light_boxes[lt.id] = self._trigger_box(cmap, lt)
                box = self._light_boxes[lt.id]
                if box is None:
                    continue
                tv_xy, center, extent = box
                if not route_hits_box(ahead, center, extent):
                    continue
                ray = tv_xy - ego_xy
                if fwd.x * ray[0] + fwd.y * ray[1] < 0:
                    continue
                d = float(np.hypot(*ray))
                if d < best_d:
                    state = {carla.TrafficLightState.Red: 0,
                             carla.TrafficLightState.Yellow: 1,
                             carla.TrafficLightState.Green: 2}.get(
                                 lt.get_state(), -1)
                    best, best_d = {"xy": tv_xy, "state": state}, d
            return best

        def run_step(self, input_data, timestamp):
            self._step += 1
            if self._step % self._repeat:
                return self._last
            if getattr(self, "_route_pid_conf", None) is not None:
                return self._run_step_route_pid()
            ego = CarlaDataProvider.get_hero_actor()
            tr = ego.get_transform()
            ego_xy = np.array([tr.location.x, tr.location.y])
            # Bench2Drive's anno `theta` is the IMU compass = CARLA yaw
            # + pi/2 (data_collect.py builds rotations as
            # rad2deg(compass) - 90; verified exact on 20 clips). All
            # training obs are expressed in that rotated frame, so the
            # online featurizer must reproduce it (yaw_offset, pi/2 by
            # default): without it every relative feature (neighbors,
            # route, lights) is rotated 90 deg against what the policy
            # was trained on — this bug shipped in the first
            # closed-loop rounds.
            ego_yaw = float(np.deg2rad(tr.rotation.yaw)) + self._yaw_off
            v = ego.get_velocity()
            ego_speed = float(np.linalg.norm([v.x, v.y, v.z]))

            actors = []
            world = CarlaDataProvider.get_world()
            for pattern in ("vehicle.*", "walker.*"):
                for a in world.get_actors().filter(pattern):
                    if a.id == ego.id:
                        continue
                    loc = a.get_transform()
                    actors.append((a.id,
                                   np.array([loc.location.x,
                                             loc.location.y]),
                                   float(np.deg2rad(loc.rotation.yaw))))

            route = None
            if self._with_route:
                if self._near_cursor is not None:
                    near_xy, near_cmd = self._near_cursor.step(ego_xy)
                    far_xy, far_cmd = self._far_cursor.step(ego_xy)
                else:  # set_global_plan never ran (defensive)
                    near_xy = far_xy = None
                    near_cmd = far_cmd = 4
                route = {"near_xy": near_xy, "near_cmd": near_cmd,
                         "far_xy": far_xy, "far_cmd": far_cmd}
            light = None
            if self._with_lights:
                light = (self._relevant_light(
                    world, CarlaDataProvider.get_map(), ego, ego_xy)
                    or {"xy": None, "state": -1})

            obs, self._prev_xy = featurize_frame(
                ego_xy, ego_yaw, ego_speed, actors, self._prev_xy,
                route=route, light=light)
            a = self._driver.act(obs)
            if self._wp_mode:
                return self._control_from_waypoints(a, ego_speed, obs,
                                                    route, light, actors)
            brake = float(np.clip(a[2], 0.0, 1.0))
            if brake >= self._brake_threshold:
                throttle, brake = 0.0, 1.0
            else:
                throttle = float(np.clip(a[0] * self._throttle_gain,
                                          0.0, 1.0))
                brake = 0.0
            self._last = carla.VehicleControl(
                throttle=throttle,
                steer=float(np.clip(a[1] * self._steer_gain, -1.0, 1.0)),
                brake=brake)
            rec = None
            if self._recovery is not None:
                rec = self._recovery.update(ego_speed, throttle, brake,
                                            self._last.steer)
                if rec is not None:
                    self._last = carla.VehicleControl(
                        throttle=rec[0], steer=rec[1], brake=0.0,
                        reverse=rec[2])
            if self._log_path:
                import json as _json
                with open(self._log_path, "a") as f:
                    f.write(_json.dumps({
                        "step": self._step, "speed": round(ego_speed, 3),
                        "throttle": round(self._last.throttle, 3),
                        "steer": round(self._last.steer, 3),
                        "brake": round(self._last.brake, 3),
                        "n_actors": len(actors),
                        "near_cmd": (route or {}).get("near_cmd"),
                        # ego-frame route points as the policy sees them
                        # (obs dims 49:51 / 57:59) — for comparing the
                        # deployed distribution against training stats
                        "near": ([round(float(obs[49]), 4),
                                  round(float(obs[50]), 4)]
                                 if self._with_route else None),
                        "far": ([round(float(obs[57]), 4),
                                 round(float(obs[58]), 4)]
                                if self._with_route else None),
                        "light": (None if light is None
                                  or light["xy"] is None
                                  else light["state"]),
                        "rec": int(rec is not None),
                    }) + "\n")
            return self._last

        def _control_from_waypoints(self, a, ego_speed, obs, route, light,
                                    actors):
            """Phase-1 deployment tail: predicted wp -> tracker -> control.

            The raw 12-dim action already fed the world model inside
            ContinuousWMDriver (same precedent as the control-mode
            gains: the WM sees what the policy said, the vehicle gets
            the tracker's interpretation of it).
            """
            wp_v = wp_to_vehicle(np.asarray(a))
            throttle, steer, brake, dbg = self._tracker.act(wp_v, ego_speed)
            if self._wp_head:
                # the WM's action channel is the executed control (the
                # tracker's raw output, pre-recovery — same precedent as
                # control mode feeding the raw policy action)
                self._driver.set_executed([throttle, steer, brake])
            self._last = carla.VehicleControl(
                throttle=throttle, steer=steer, brake=brake)
            # gen3_wp 3x3 diagnosis: dominant failure is a POWER-WEDGE
            # (throttle 0.75 at standstill against layout for hundreds
            # of ticks, plan saying forward) — reverse recovery applies
            rec = None
            if self._recovery is not None:
                rec = self._recovery.update(ego_speed, throttle, brake,
                                            steer)
                if rec is not None:
                    self._last = carla.VehicleControl(
                        throttle=rec[0], steer=rec[1], brake=0.0,
                        reverse=rec[2])
            if self._log_path:
                import json as _json
                with open(self._log_path, "a") as f:
                    f.write(_json.dumps({
                        "step": self._step, "speed": round(ego_speed, 3),
                        "throttle": round(throttle, 3),
                        "steer": round(steer, 3), "brake": round(brake, 3),
                        "n_actors": len(actors),
                        "near_cmd": (route or {}).get("near_cmd"),
                        "wp1": [round(float(wp_v[0, 0]), 2),
                                round(float(wp_v[0, 1]), 2)],
                        "wp6": [round(float(wp_v[-1, 0]), 2),
                                round(float(wp_v[-1, 1]), 2)],
                        "light": (None if light is None
                                  or light["xy"] is None
                                  else light["state"]),
                        "trk": dbg,
                        "rec": int(rec is not None),
                    }) + "\n")
            return self._last

        def _run_step_route_pid(self):
            ego = CarlaDataProvider.get_hero_actor()
            tr = ego.get_transform()
            ego_xy = np.array([tr.location.x, tr.location.y])
            ego_yaw = float(np.deg2rad(tr.rotation.yaw))  # raw CARLA yaw
            v = ego.get_velocity()
            ego_speed = float(np.linalg.norm([v.x, v.y, v.z]))
            if self._route_pid is None:
                if self._dense_plan_xy is None:
                    return carla.VehicleControl()  # plan not set yet
                self._route_pid = RoutePIDDriver(
                    self._dense_plan_xy, **self._route_pid_conf)
            world = CarlaDataProvider.get_world()
            actors = []
            for pattern in ("vehicle.*", "walker.*"):
                for a in world.get_actors().filter(pattern):
                    if a.id == ego.id:
                        continue
                    loc = a.get_transform()
                    actors.append((a.id,
                                   np.array([loc.location.x,
                                             loc.location.y]),
                                   float(np.deg2rad(loc.rotation.yaw))))
            light = self._relevant_light(
                world, CarlaDataProvider.get_map(), ego, ego_xy)
            ld = ls = None
            if light is not None and light["xy"] is not None:
                ld = float(np.hypot(*(light["xy"] - ego_xy)))
                ls = light["state"]
            throttle, steer, brake, dbg = self._route_pid.act(
                ego_xy, ego_yaw, ego_speed, actors,
                light_dist=ld, light_state=ls)
            self._last = carla.VehicleControl(
                throttle=throttle, steer=steer, brake=brake)
            if self._log_path:
                import json as _json
                with open(self._log_path, "a") as f:
                    f.write(_json.dumps({
                        "step": self._step, "speed": round(ego_speed, 3),
                        "throttle": round(throttle, 3),
                        "steer": round(steer, 3), "brake": round(brake, 3),
                        "n_actors": len(actors), "light": ls,
                        "pid": dbg}) + "\n")
            return self._last

        def destroy(self):
            if hasattr(self, "_driver"):
                self._driver.reset()

except ImportError:  # carla/leaderboard not on path (e.g. login nodes)
    DittoCarlaAgent = None
