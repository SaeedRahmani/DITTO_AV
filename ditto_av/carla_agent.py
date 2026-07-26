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


try:  # pragma: no cover - requires the carla package + leaderboard
    import carla  # noqa: F401
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent

    class DittoCarlaAgent(AutonomousAgent):
        """Bench2Drive/leaderboard entry point (privileged track).

        TODO(closed-loop): wire in setup() to load config/checkpoints,
        track the route plan for near/far commands (leaderboard provides
        _global_plan in GPS/world coords), read actors from
        CarlaDataProvider, and map ContinuousWMDriver output to
        carla.VehicleControl at 10 Hz. See PAPER_PLAN Phase-2 notes.
        """

except ImportError:  # carla not installed (e.g. on DelftBlue login nodes)
    DittoCarlaAgent = None
