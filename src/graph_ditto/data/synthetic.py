"""
Synthetic driving data generator for testing Graph-DITTO.

Generates simple highway scenarios with multiple vehicles and straight lanes,
providing a test bed for verifying the full pipeline without real driving data.

Can be replaced with real dataset loaders (nuScenes, Waymo, etc.) for actual
experiments.
"""

import numpy as np


class SyntheticDrivingGenerator:
    """
    Generates simple multi-lane highway scenarios.

    Scenario:
      - 3-lane straight highway, 200m long
      - 5-15 vehicles with constant-velocity + noise
      - Ego vehicle is always index 0
      - Lane center points sampled every 5m

    Agent state: [x, y, vx, vy, cos_heading, sin_heading, length, width]
    Lane point:  [x, y, cos_tangent, sin_tangent, lane_type, speed_limit]
    Ego action:  [steering, acceleration]
    """

    def __init__(
        self,
        n_episodes: int = 100,
        episode_length: int = 100,
        dt: float = 0.1,
        n_lanes: int = 3,
        lane_width: float = 3.7,
        road_length: float = 200.0,
        lane_point_spacing: float = 5.0,
        min_vehicles: int = 5,
        max_vehicles: int = 15,
        seed: int = 42,
    ):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.dt = dt
        self.n_lanes = n_lanes
        self.lane_width = lane_width
        self.road_length = road_length
        self.lane_point_spacing = lane_point_spacing
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles
        self.rng = np.random.RandomState(seed)

    def generate_lane_points(self) -> np.ndarray:
        """Generate lane center points for a straight multi-lane road."""
        points = []
        n_points_per_lane = int(self.road_length / self.lane_point_spacing) + 1
        for lane_idx in range(self.n_lanes):
            lane_y = (lane_idx - self.n_lanes // 2) * self.lane_width
            for i in range(n_points_per_lane):
                x = i * self.lane_point_spacing
                y = lane_y
                cos_t = 1.0   # tangent along x-axis (straight road)
                sin_t = 0.0
                lane_type = float(lane_idx)
                speed_limit = 30.0  # m/s (~108 km/h)
                points.append([x, y, cos_t, sin_t, lane_type, speed_limit])
        return np.array(points, dtype=np.float32)

    def generate_initial_vehicles(self, n_vehicles: int) -> np.ndarray:
        """Generate initial states for n vehicles on the highway."""
        states = []
        lanes_y = [
            (i - self.n_lanes // 2) * self.lane_width for i in range(self.n_lanes)
        ]

        for i in range(n_vehicles):
            lane = self.rng.randint(0, self.n_lanes)
            x = self.rng.uniform(10, self.road_length - 10)
            y = lanes_y[lane] + self.rng.normal(0, 0.2)  # slight lateral noise
            speed = self.rng.uniform(20, 35)  # m/s
            heading_noise = self.rng.normal(0, 0.02)
            cos_h = np.cos(heading_noise)
            sin_h = np.sin(heading_noise)
            length = self.rng.uniform(4.0, 5.5)
            width = self.rng.uniform(1.7, 2.1)
            states.append([x, y, speed, 0.0, cos_h, sin_h, length, width])

        # Make ego vehicle (index 0) well-placed
        states[0][0] = 50.0   # x
        states[0][1] = 0.0    # y (center lane)
        states[0][2] = 25.0   # vx
        states[0][4] = 1.0    # cos(0)
        states[0][5] = 0.0    # sin(0)

        return np.array(states, dtype=np.float32)

    def step_vehicles(
        self, states: np.ndarray, ego_action: np.ndarray
    ) -> tuple:
        """Advance all vehicles one timestep.

        Args:
            states: (N, 8) current vehicle states
            ego_action: (2,) [steering, acceleration] for ego

        Returns:
            next_states: (N, 8) updated states
            ego_action: (2,) the applied ego action
        """
        next_states = states.copy()

        for i in range(len(states)):
            if i == 0:
                # Ego: apply action
                steer, accel = ego_action
                vx = states[i, 2] + accel * self.dt
                vx = np.clip(vx, 5.0, 40.0)
                heading = np.arctan2(states[i, 5], states[i, 4])
                heading += steer * self.dt
                next_states[i, 0] += vx * np.cos(heading) * self.dt
                next_states[i, 1] += vx * np.sin(heading) * self.dt
                next_states[i, 2] = vx * np.cos(heading)
                next_states[i, 3] = vx * np.sin(heading)
                next_states[i, 4] = np.cos(heading)
                next_states[i, 5] = np.sin(heading)
            else:
                # Other vehicles: constant velocity + small random perturbations
                noise_accel = self.rng.normal(0, 0.5)
                noise_steer = self.rng.normal(0, 0.01)
                vx = states[i, 2] + noise_accel * self.dt
                vx = np.clip(vx, 10.0, 40.0)
                heading = np.arctan2(states[i, 5], states[i, 4])
                heading += noise_steer * self.dt

                next_states[i, 0] += vx * np.cos(heading) * self.dt
                next_states[i, 1] += vx * np.sin(heading) * self.dt
                next_states[i, 2] = vx * np.cos(heading)
                next_states[i, 3] = vx * np.sin(heading)
                next_states[i, 4] = np.cos(heading)
                next_states[i, 5] = np.sin(heading)

        return next_states, ego_action

    def generate_episode(self):
        """Generate one full episode of driving data.

        Returns:
            agent_states_seq: (T, max_vehicles, 8) — padded to max_vehicles
            agent_masks_seq: (T, max_vehicles) bool
            lane_points: (N_lanes, 6) — static across episode
            ego_actions_seq: (T, 2) — ego actions
            resets: (T,) bool — True at episode start
        """
        n_vehicles = self.rng.randint(self.min_vehicles, self.max_vehicles + 1)
        states = self.generate_initial_vehicles(n_vehicles)
        lane_points = self.generate_lane_points()

        agent_states_seq = []
        ego_actions_seq = []
        resets = []

        for t in range(self.episode_length):
            # Expert ego action: gently follow the road with small corrections
            ego_y = states[0, 1]
            steer = -0.1 * ego_y + self.rng.normal(0, 0.05)
            accel = self.rng.normal(0, 0.3)
            ego_action = np.array([steer, accel], dtype=np.float32)

            agent_states_seq.append(states.copy())
            ego_actions_seq.append(ego_action)
            resets.append(t == 0)

            states, _ = self.step_vehicles(states, ego_action)

        agent_states_seq = np.array(agent_states_seq, dtype=np.float32)
        ego_actions_seq = np.array(ego_actions_seq, dtype=np.float32)
        resets = np.array(resets, dtype=bool)

        return {
            "agent_states": agent_states_seq,       # (T, N, 8)
            "n_agents": n_vehicles,
            "lane_points": lane_points,             # (N_l, 6)
            "ego_actions": ego_actions_seq,         # (T, 2)
            "resets": resets,                        # (T,)
        }

    def generate_dataset(self, save_path: str = None):
        """Generate multiple episodes and optionally save to disk.

        Returns:
            episodes: list of episode dicts
        """
        episodes = []
        for _ in range(self.n_episodes):
            ep = self.generate_episode()
            episodes.append(ep)

        if save_path:
            np.savez_compressed(
                save_path,
                episodes=episodes,
                n_episodes=self.n_episodes,
            )
            print(f"Saved {self.n_episodes} episodes to {save_path}")

        return episodes
