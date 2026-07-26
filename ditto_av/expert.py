from __future__ import annotations

import numpy as np

# DiscreteMetaAction indices
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4

AGGRESSIVE, CONSERVATIVE = "aggressive", "conservative"


class ScriptedExpert:
    """Rule-based highway driver with two behavior styles.

    Both styles are competent (safety-checked, time-to-collision aware), but
    they respond differently to the same situation: when blocked by a slower
    leader, the aggressive style overtakes while the conservative style slows
    down and follows. This makes the demonstration distribution genuinely
    multimodal, which is the failure mode that single-trajectory latent
    matching cannot handle.
    """

    GAP_CRITICAL = 12.0  # emergency braking distance
    GAP_TIGHT = 25.0     # leader closer than this: must react
    GAP_OPEN = 45.0      # leader further than this: free road
    TTC_CRITICAL = 2.0   # seconds to collision: emergency
    TTC_REACT = 5.0      # seconds to collision: react
    SAFE_FRONT = 16.0    # required front gap in target lane
    SAFE_REAR = 12.0     # required rear gap in target lane
    CRUISE_SPEED = 28.0

    def __init__(self, style: str = AGGRESSIVE):
        assert style in (AGGRESSIVE, CONSERVATIVE)
        self.style = style

    def act(self, env) -> int:
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road

        front, _ = road.neighbour_vehicles(ego, ego.lane_index)
        if front is None:
            gap, ttc = np.inf, np.inf
        else:
            gap = ego.lane_distance_to(front)
            closing = ego.speed - front.speed
            ttc = gap / closing if closing > 1e-3 else np.inf

        if gap < self.GAP_CRITICAL or ttc < self.TTC_CRITICAL:
            return SLOWER
        if gap < self.GAP_TIGHT or ttc < self.TTC_REACT:
            if self.style == AGGRESSIVE:
                lane_change = self._safe_lane_change(env, ego, road)
                if lane_change is not None:
                    return lane_change
            return SLOWER
        if gap > self.GAP_OPEN and ttc > 2 * self.TTC_REACT:
            return FASTER if ego.speed < self.CRUISE_SPEED else IDLE
        # buffer zone: follow the leader's speed
        if front is not None and ego.speed > front.speed + 3.0:
            return SLOWER
        if front is not None and ego.speed < front.speed - 3.0 \
                and ego.speed < self.CRUISE_SPEED:
            return FASTER
        return IDLE

    def _safe_lane_change(self, env, ego, road):
        """Return LANE_LEFT/LANE_RIGHT if an adjacent lane is safe, else None.

        A lane is safe when front/rear gaps are sufficient, the rear vehicle
        is not closing fast, and the new leader is not slower and imminent.
        Prefers the left lane (overtaking side).
        """
        candidates = []
        for lane in road.network.side_lanes(ego.lane_index):
            front, rear = road.neighbour_vehicles(ego, lane)
            front_gap = ego.lane_distance_to(front) if front is not None else np.inf
            rear_gap = -ego.lane_distance_to(rear) if rear is not None else np.inf
            rear_closing = (rear.speed - ego.speed) if rear is not None else 0.0
            front_closing = (ego.speed - front.speed) if front is not None else 0.0
            front_ttc = front_gap / front_closing if front_closing > 1e-3 else np.inf
            if (front_gap > self.SAFE_FRONT
                    and front_ttc > self.TTC_REACT
                    and rear_gap > self.SAFE_REAR + 2.0 * max(0.0, rear_closing)):
                action = LANE_LEFT if lane[2] < ego.lane_index[2] else LANE_RIGHT
                candidates.append((lane[2], action))
        if not candidates:
            return None
        # leftmost candidate first
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]


class NoisyExpert:
    """Expert with epsilon-random actions, for world-model data coverage."""

    def __init__(self, style: str, eps: float, rng: np.random.Generator):
        self.expert = ScriptedExpert(style)
        self.eps = eps
        self.rng = rng

    def act(self, env) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, 5))
        return self.expert.act(env)
