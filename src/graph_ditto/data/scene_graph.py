"""
Scene graph construction utilities.

Converts raw driving scenario data (agent states, lane geometry) into
padded tensor representations suitable for the SceneEncoder.
"""

import numpy as np
import torch
from torch import Tensor


class SceneGraphBuilder:
    """
    Builds padded tensor representations of driving scenes.

    All scenes are padded to (max_agents, max_lanes) for uniform batching.
    Masks indicate which entries are real vs. padding.
    """

    def __init__(
        self,
        max_agents: int = 32,
        max_lanes: int = 128,
        agent_feat_dim: int = 8,
        lane_feat_dim: int = 6,
    ):
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.agent_feat_dim = agent_feat_dim
        self.lane_feat_dim = lane_feat_dim

    def build_graph(
        self,
        agent_states: np.ndarray,   # (N_agents, agent_feat_dim)
        lane_points: np.ndarray,    # (N_lanes, lane_feat_dim)
        ego_idx: int = 0,
    ) -> dict:
        """
        Build a padded scene graph from raw data.

        Agent features: [x, y, vx, vy, cos_heading, sin_heading, length, width]
        Lane features:  [x, y, cos_tangent, sin_tangent, lane_type, speed_limit]

        Returns dict of numpy arrays (padded to max sizes):
            agent_features: (max_agents, agent_feat_dim)
            agent_mask: (max_agents,) bool
            lane_features: (max_lanes, lane_feat_dim)
            lane_mask: (max_lanes,) bool
            ego_mask: (max_agents,) bool
        """
        n_agents = min(agent_states.shape[0], self.max_agents)
        n_lanes = min(lane_points.shape[0], self.max_lanes)

        # Pad agent features
        agent_features = np.zeros(
            (self.max_agents, self.agent_feat_dim), dtype=np.float32
        )
        agent_features[:n_agents] = agent_states[:n_agents, :self.agent_feat_dim]

        agent_mask = np.zeros(self.max_agents, dtype=bool)
        agent_mask[:n_agents] = True

        # Pad lane features
        lane_features = np.zeros(
            (self.max_lanes, self.lane_feat_dim), dtype=np.float32
        )
        lane_features[:n_lanes] = lane_points[:n_lanes, :self.lane_feat_dim]

        lane_mask = np.zeros(self.max_lanes, dtype=bool)
        lane_mask[:n_lanes] = True

        # Ego mask
        ego_mask = np.zeros(self.max_agents, dtype=bool)
        ego_idx_clamped = min(ego_idx, n_agents - 1)
        ego_mask[ego_idx_clamped] = True

        return {
            "agent_features": agent_features,
            "agent_mask": agent_mask,
            "lane_features": lane_features,
            "lane_mask": lane_mask,
            "ego_mask": ego_mask,
        }

    def normalize_to_ego(
        self,
        agent_states: np.ndarray,
        lane_points: np.ndarray,
        ego_idx: int = 0,
    ):
        """
        Transform coordinates to ego-centric frame.

        Centers positions on ego vehicle and rotates so ego heading = 0.
        """
        ego = agent_states[ego_idx]
        ego_x, ego_y = ego[0], ego[1]
        ego_cos, ego_sin = ego[4], ego[5]

        # Rotation matrix (rotate by -ego_heading)
        rot = np.array([[ego_cos, ego_sin], [-ego_sin, ego_cos]], dtype=np.float32)

        # Transform agent positions
        agent_pos = agent_states[:, :2] - np.array([ego_x, ego_y])
        agent_pos = agent_pos @ rot.T
        agent_states = agent_states.copy()
        agent_states[:, :2] = agent_pos

        # Transform agent velocities
        agent_vel = agent_states[:, 2:4] @ rot.T
        agent_states[:, 2:4] = agent_vel

        # Transform agent headings relative to ego
        cos_h, sin_h = agent_states[:, 4], agent_states[:, 5]
        new_cos = cos_h * ego_cos + sin_h * ego_sin
        new_sin = -cos_h * ego_sin + sin_h * ego_cos
        agent_states[:, 4] = new_cos
        agent_states[:, 5] = new_sin

        # Transform lane positions
        lane_pos = lane_points[:, :2] - np.array([ego_x, ego_y])
        lane_pos = lane_pos @ rot.T
        lane_points = lane_points.copy()
        lane_points[:, :2] = lane_pos

        # Transform lane tangents
        lane_tan = lane_points[:, 2:4] @ rot.T
        lane_points[:, 2:4] = lane_tan

        return agent_states, lane_points
