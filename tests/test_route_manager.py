"""Route-plan -> near/far command signal (pure logic, no carla needed)."""
import numpy as np

from ditto_av.carla_agent import plan_to_command_points


def _straight_plan(n=100, cmd=4):
    return [(float(i), 0.0, cmd) for i in range(n)]


def test_fallback_on_pure_lanefollow():
    r = plan_to_command_points(_straight_plan(), np.array([0.0, 0.0]))
    assert r["near_cmd"] == 4 and r["far_cmd"] == 4
    # ~20 m and ~50 m ahead along the route
    assert abs(r["near_xy"][0] - 20.0) <= 1.0
    assert abs(r["far_xy"][0] - 50.0) <= 1.0


def test_command_change_points():
    # lane-follow, then a LEFT turn command at x=30..40, then lane-follow
    plan = ([(float(i), 0.0, 4) for i in range(30)]
            + [(float(i), 0.0, 1) for i in range(30, 40)]
            + [(float(i), 0.0, 4) for i in range(40, 80)])
    r = plan_to_command_points(plan, np.array([0.0, 0.0]))
    assert r["near_cmd"] == 1 and abs(r["near_xy"][0] - 30.0) < 1e-6
    assert r["far_cmd"] == 4 and abs(r["far_xy"][0] - 40.0) < 1e-6


def test_change_behind_ego_is_ignored():
    plan = ([(float(i), 0.0, 1) for i in range(10)]
            + [(float(i), 0.0, 4) for i in range(10, 100)])
    # ego already past the turn at x=50: only lane-follow remains ahead
    r = plan_to_command_points(plan, np.array([50.0, 0.0]))
    assert r["near_cmd"] == 4 and r["far_cmd"] == 4
    assert r["near_xy"][0] > 50.0


def test_min_ahead_skips_imminent_change():
    # command changes 2 m ahead of ego: too close to act on, use the next
    plan = ([(float(i), 0.0, 4) for i in range(52)]
            + [(float(i), 0.0, 2) for i in range(52, 60)]
            + [(float(i), 0.0, 4) for i in range(60, 120)])
    r = plan_to_command_points(plan, np.array([50.0, 0.0]), min_ahead=5.0)
    assert r["near_cmd"] == 4 and abs(r["near_xy"][0] - 60.0) < 1e-6


def test_empty_plan():
    r = plan_to_command_points([], np.array([0.0, 0.0]))
    assert r["near_xy"] is None and r["near_cmd"] == 4
