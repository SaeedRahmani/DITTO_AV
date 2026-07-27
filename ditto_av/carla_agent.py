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

from .bench2drive import (FPS, N_COMMANDS, POS_SCALE, ROUTE_DIMS, V_MAX,
                          VEL_SCALE)


def featurize_frame(ego_xy: np.ndarray, ego_yaw: float, ego_speed: float,
                    actors: Sequence[Tuple[object, np.ndarray, float]],
                    prev_xy: Dict[object, np.ndarray],
                    n_neighbors: int = 6, radius: float = 60.0,
                    route: Optional[dict] = None
                    ) -> Tuple[np.ndarray, Dict[object, np.ndarray]]:
    """One observation from world state; mirrors bench2drive.load_clip.

    actors: (actor_id, world_xy(2,), yaw_rad) for vehicles/walkers/bikes,
        excluding the ego. prev_xy: last tick's positions for velocity
        finite-differencing (pass the returned dict back next tick).
    route: {"near_xy", "far_xy", "near_cmd", "far_cmd"} or None.
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

    core = np.clip(rows, -2.0, 2.0).reshape(-1)
    if route is None:
        return core, cur_xy

    block = np.zeros(ROUTE_DIMS, dtype=np.float32)
    for i, tag in enumerate(("near", "far")):
        base = i * (2 + N_COMMANDS)
        xy = route.get(f"{tag}_xy")
        if xy is not None and np.isfinite(xy).all():
            rel = world_to_ego @ (np.asarray(xy, dtype=np.float64) - ego_xy)
            block[base:base + 2] = rel / POS_SCALE
        cmd = int(route.get(f"{tag}_cmd", 4) or 4)
        if 1 <= cmd <= N_COMMANDS:
            block[base + 2 + cmd - 1] = 1.0
    return np.concatenate([core, np.clip(block, -2.0, 2.0)]), cur_xy


class ContinuousWMDriver:
    """Closed-loop continuous-control driver: posterior filtering + policy.

    The continuous counterpart of evaluate.WMPolicyDriver; consumes obs
    vectors from `featurize_frame` and returns (throttle, steer, brake).
    """

    def __init__(self, wm, policy, action_dim: int = 3,
                 device: str = "cpu", stochastic: bool = False):
        self.wm, self.policy = wm, policy
        self.action_dim = action_dim
        self.device = device
        self.stochastic = stochastic
        self.reset()

    def reset(self):
        self.state = None
        self.prev_action: Optional[torch.Tensor] = None

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
        self.prev_action = a
        return a.cpu().numpy()


def plan_to_command_points(plan: Sequence[Tuple[float, float, int]],
                           ego_xy: np.ndarray,
                           min_ahead: float = 5.0,
                           near_fallback: float = 20.0,
                           far_fallback: float = 50.0) -> dict:
    """Compress a route plan into the near/far command signal.

    plan: [(x, y, command_int)] in world coords, ordered along the route
        (leaderboard `_global_plan_world_coord` with RoadOption -> int).
    Command points are where the command changes (junction decisions,
    matching Bench2Drive's sparse command annotations). near = first
    change point at least `min_ahead` beyond the closest plan index;
    far = the next one. Fallback: lane-follow points ~20/50 m ahead.
    Returns the `route` dict for featurize_frame.
    """
    if len(plan) == 0:
        return {"near_xy": None, "near_cmd": 4,
                "far_xy": None, "far_cmd": 4}
    xy = np.asarray([(p[0], p[1]) for p in plan], dtype=np.float64)
    cmds = [int(p[2]) for p in plan]
    dist_to_ego = np.linalg.norm(xy - ego_xy, axis=1)
    cur = int(np.argmin(dist_to_ego))

    # arc-length ahead of the current index
    seg = np.linalg.norm(np.diff(xy[cur:], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # s[j] = dist to plan[cur+j]

    changes = [cur + j for j in range(1, len(s))
               if cmds[cur + j] != cmds[cur + j - 1] and s[j] >= min_ahead]

    def fallback(dist):
        j = int(np.searchsorted(s, dist))
        j = min(j, len(s) - 1)
        return {"xy": xy[cur + j], "cmd": cmds[cur + j]}

    if len(changes) >= 1:
        near = {"xy": xy[changes[0]], "cmd": cmds[changes[0]]}
        far = ({"xy": xy[changes[1]], "cmd": cmds[changes[1]]}
               if len(changes) >= 2 else fallback(far_fallback))
    else:
        near, far = fallback(near_fallback), fallback(far_fallback)
    return {"near_xy": near["xy"], "near_cmd": near["cmd"],
            "far_xy": far["xy"], "far_cmd": far["cmd"]}


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
            run_cfg = load_config(conf["run_config"])
            self._cfg = run_cfg
            wm = load_world_model(run_cfg, run_cfg.env.obs_dim)
            policy = make_actor_critic(
                True, run_cfg.wm.feature_dim, run_cfg.env.action_dim,
                run_cfg.ac.hidden_dim, run_cfg.ac.layers)
            import torch as _torch
            ckpt = (run_cfg.dirs()["ckpt"]
                    / f"{conf.get('policy', 'ditto_multi')}.pt")
            policy.load_state_dict(_torch.load(ckpt, map_location="cpu"))
            policy.eval()
            self._driver = ContinuousWMDriver(
                wm, policy, run_cfg.env.action_dim,
                stochastic=bool(conf.get("stochastic", False)))
            self._with_route = run_cfg.env.extra_obs_dims > 0
            self._repeat = int(conf.get("action_repeat", 2))
            # expert brake is near-binary (0 almost always, 1 sometimes);
            # the Gaussian mean outputs a constant ~0.15 which in CARLA
            # drags enough brake torque to hold the car against full
            # throttle (verified: run 10522236 never exceeded 0.24 m/s at
            # 0.8 throttle / 0.15 brake). Binarize at deployment.
            self._brake_threshold = float(conf.get("brake_threshold", 0.5))
            self._prev_xy: Dict[object, np.ndarray] = {}
            self._step = -1
            self._last = carla.VehicleControl()
            # per-model-tick behavior log (jsonl) for debugging closed-loop
            import os as _os
            self._log_path = _os.environ.get("DITTO_AGENT_LOG")

        def sensors(self):
            return [{"type": "sensor.speedometer", "id": "speed",
                     "reading_frequency": 20}]

        def run_step(self, input_data, timestamp):
            self._step += 1
            if self._step % self._repeat:
                return self._last
            ego = CarlaDataProvider.get_hero_actor()
            tr = ego.get_transform()
            ego_xy = np.array([tr.location.x, tr.location.y])
            ego_yaw = float(np.deg2rad(tr.rotation.yaw))
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
                plan = [(t.location.x, t.location.y, int(opt.value))
                        for t, opt in self._global_plan_world_coord]
                route = plan_to_command_points(plan, ego_xy)

            obs, self._prev_xy = featurize_frame(
                ego_xy, ego_yaw, ego_speed, actors, self._prev_xy,
                route=route)
            a = self._driver.act(obs)
            brake = float(np.clip(a[2], 0.0, 1.0))
            if brake >= self._brake_threshold:
                throttle, brake = 0.0, 1.0
            else:
                throttle, brake = float(np.clip(a[0], 0.0, 1.0)), 0.0
            self._last = carla.VehicleControl(
                throttle=throttle,
                steer=float(np.clip(a[1], -1.0, 1.0)),
                brake=brake)
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
                    }) + "\n")
            return self._last

        def destroy(self):
            if hasattr(self, "_driver"):
                self._driver.reset()

except ImportError:  # carla/leaderboard not on path (e.g. login nodes)
    DittoCarlaAgent = None
