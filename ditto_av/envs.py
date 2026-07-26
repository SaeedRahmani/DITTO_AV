from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  (registers envs)

from .config import EnvConfig


def make_env(cfg: EnvConfig,
             vehicles_density: Optional[float] = None,
             vehicles_count: Optional[int] = None) -> gym.Env:
    """Create a highway env with the vectorized kinematics observation."""
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": cfg.observed_vehicles,
            "features": list(cfg.obs_features),
            "absolute": False,
            "normalize": True,
        },
        # default target_speeds [20,25,30] cannot brake below 20 m/s and
        # guarantees rear-end crashes behind slow traffic — widen the range
        "action": {"type": "DiscreteMetaAction",
                   "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0]},
        "lanes_count": cfg.lanes_count,
        "vehicles_count": vehicles_count if vehicles_count is not None else cfg.vehicles_count,
        "vehicles_density": vehicles_density if vehicles_density is not None else cfg.vehicles_density,
        "duration": cfg.duration,
    }
    return gym.make(cfg.env_id, config=env_config)


def featurize(obs: np.ndarray) -> np.ndarray:
    """Flatten the (V, F) kinematics observation into a bounded vector.

    The ego row's absolute longitudinal position grows without bound on the
    highway and carries no decision-relevant information, so it is zeroed.
    Everything else is clipped to a safe range for the world model.
    """
    obs = np.array(obs, dtype=np.float32, copy=True)
    obs[0, 1] = 0.0  # ego absolute x
    obs = np.clip(obs, -2.0, 2.0)
    return obs.reshape(-1)
