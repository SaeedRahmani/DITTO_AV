"""Phase-1 waypoint action abstraction: data plumbing, frame round-trip,
and the deployment tracker.

The round-trip test is the deployment-side gate demanded by Phase-0c/0f:
waypoints built by the offline convention (anno theta = CARLA yaw + pi/2,
FORWARD = -y) must, after the deployment conversion `wp_to_vehicle`,
rebuild the exact future world poses with raw CARLA yaw. Any frame slip
anywhere in the chain breaks the identity.
"""
import gzip
import json

import numpy as np
import pytest
import torch

from ditto_av.bench2drive import (WP_SCALE, WP_STRIDE, clips_to_npz,
                                  future_waypoints, load_clip)
from ditto_av.carla_agent import WaypointTracker, wp_to_vehicle
from ditto_av.config import EnvConfig
from ditto_av.data import TrajectoryData
from ditto_av.models.nets import WP_BOUND, make_actor_critic


def curved_trajectory(n=40, speed=6.0, yaw_rate=0.04, dt=0.1):
    """World path with heading consistent with motion (CARLA yaw)."""
    xy = np.zeros((n, 2))
    yaw = np.zeros(n)
    for t in range(1, n):
        yaw[t] = yaw[t - 1] + yaw_rate
        step = speed * dt
        xy[t] = xy[t - 1] + step * np.array([np.cos(yaw[t]), np.sin(yaw[t])])
    return xy, yaw


def fake_frames(xy, yaw, speed=6.0):
    """Minimal anno dicts: theta = CARLA yaw + pi/2 (the settled fact)."""
    frames = []
    for t in range(len(xy)):
        frames.append({
            "x": float(xy[t, 0]), "y": float(xy[t, 1]),
            "theta": float(yaw[t] + np.pi / 2), "speed": speed,
            "throttle": 0.5, "steer": 0.0, "brake": 0.0,
            "bounding_boxes": [
                {"class": "ego_vehicle", "id": "ego",
                 "location": [float(xy[t, 0]), float(xy[t, 1]), 0.0],
                 "rotation": [0, 0, float(np.rad2deg(yaw[t]))]}],
        })
    return frames


def test_wp_deployment_round_trip():
    """offline wp (compass frame) -> wp_to_vehicle -> world == truth."""
    xy, yaw = curved_trajectory()
    frames = fake_frames(xy, yaw)
    thetas = np.array([f["theta"] for f in frames])
    wp = future_waypoints(frames, thetas, k=6)          # (n, 6, 2) m
    scaled = (wp / WP_SCALE).reshape(len(xy), -1)       # the action layout
    for t in (0, 7, 20):
        wp_v = wp_to_vehicle(scaled[t])                  # (6, 2) m
        c, s = np.cos(yaw[t]), np.sin(yaw[t])            # raw CARLA yaw
        veh_to_world = np.array([[c, -s], [s, c]])
        for j in range(6):
            rebuilt = xy[t] + veh_to_world @ wp_v[j]
            truth = xy[min(t + (j + 1) * WP_STRIDE, len(xy) - 1)]
            assert np.allclose(rebuilt, truth, atol=1e-9)


def test_wp_vehicle_frame_semantics():
    # straight drive: future points dead ahead -> +x forward, no lateral
    xy, yaw = curved_trajectory(yaw_rate=0.0)
    frames = fake_frames(xy, yaw)
    thetas = np.array([f["theta"] for f in frames])
    scaled = (future_waypoints(frames, thetas, k=6)
              / WP_SCALE).reshape(len(xy), -1)
    wp_v = wp_to_vehicle(scaled[0])
    assert (wp_v[:, 0] > 0).all()
    assert np.allclose(wp_v[:, 1], 0.0, atol=1e-9)
    assert np.isclose(wp_v[0, 0], 6.0 * WP_STRIDE / 10.0, atol=1e-6)
    # curving toward +yaw (CARLA right): lateral goes positive
    xy, yaw = curved_trajectory(yaw_rate=0.05)
    frames = fake_frames(xy, yaw)
    thetas = np.array([f["theta"] for f in frames])
    scaled = (future_waypoints(frames, thetas, k=6)
              / WP_SCALE).reshape(len(xy), -1)
    assert wp_to_vehicle(scaled[0])[-1, 1] > 0.5


def test_env_config_waypoints():
    cfg = EnvConfig(action_space="waypoints", extra_obs_dims=16)
    assert cfg.continuous and cfg.waypoints
    assert cfg.action_dim == 12
    assert EnvConfig(action_space="continuous").action_dim == 3
    assert not EnvConfig(action_space="continuous").waypoints


def test_make_actor_critic_wp_bounds():
    p = make_actor_critic(True, 16, 12, 32, 1, action_space="waypoints")
    assert torch.allclose(p.low, torch.full((12,), -WP_BOUND))
    assert torch.allclose(p.high, torch.full((12,), WP_BOUND))
    a = p.act(torch.zeros(16))
    assert a.shape == (12,)
    # control mode keeps the asymmetric throttle/steer/brake bounds
    q = make_actor_critic(True, 16, 3, 32, 1)
    assert torch.allclose(q.low, torch.tensor([0.0, -1.0, 0.0]))


def test_npz_wp_as_action(tmp_path):
    xy, yaw = curved_trajectory(n=30)
    frames = fake_frames(xy, yaw)
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    for i, fr in enumerate(frames):
        with gzip.open(anno / f"{i:05d}.json.gz", "wt") as f:
            json.dump(fr, f)
    out = tmp_path / "b2d.npz"
    clips_to_npz([tmp_path / "clip"], out, with_waypoints=6)
    d = np.load(out)
    assert d["wp"].shape == (30, 12)
    thetas = np.array([f["theta"] for f in frames])
    expect = (future_waypoints(frames, thetas, k=6)
              / WP_SCALE).reshape(30, -1)
    assert np.allclose(d["wp"], expect, atol=1e-6)
    # TrajectoryData swaps the action role; controls stay available
    td = TrajectoryData([out], action_key="wp")
    assert td.action.shape == (30, 12)
    assert not td.discrete_actions
    rng = np.random.default_rng(0)
    obs, act, reset = td.sample_wm_batch(2, 8, rng, action_dim=12)
    assert act.shape == (8, 2, 12)
    assert TrajectoryData([out]).action.shape == (30, 3)


def test_load_clip_without_waypoints_unchanged(tmp_path):
    xy, yaw = curved_trajectory(n=12)
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    for i, fr in enumerate(fake_frames(xy, yaw)):
        with gzip.open(anno / f"{i:05d}.json.gz", "wt") as f:
            json.dump(fr, f)
    d = load_clip(tmp_path / "clip")
    assert "wp" not in d


def straight_wp(spacing=3.0, k=6):
    return np.stack([np.arange(1, k + 1) * spacing, np.zeros(k)], axis=1)


def test_tracker_straight_cruise():
    trk = WaypointTracker()
    # plan says 6 m/s (3 m per 0.5 s stride); ego already there
    throttle, steer, brake, dbg = trk.act(straight_wp(3.0), ego_speed=6.0)
    assert abs(steer) < 1e-6
    assert brake == 0.0 and throttle < 0.1
    assert abs(dbg["v_t"] - 6.0) < 0.05
    # ego below plan speed -> throttle
    throttle, steer, brake, _ = trk.act(straight_wp(3.0), ego_speed=3.0)
    assert throttle > 0.5 and brake == 0.0
    # ego far above plan speed -> brake
    throttle, _, brake, _ = trk.act(straight_wp(1.0), ego_speed=8.0)
    assert throttle == 0.0 and brake > 0.5


def test_tracker_stop_plan():
    trk = WaypointTracker()
    throttle, steer, brake, _ = trk.act(np.zeros((6, 2)), ego_speed=0.1)
    assert throttle == 0.0 and brake == 1.0 and steer == 0.0
    # still moving with a collapsed plan: brake, not full-stop hold
    throttle, _, brake, _ = trk.act(np.zeros((6, 2)), ego_speed=5.0)
    assert throttle == 0.0 and brake > 0.5


def test_tracker_steer_sign_and_curvature():
    trk = WaypointTracker()
    # radius-8 arc, ~4 m point spacing: spacing says ~8 m/s but the
    # lateral-acceleration cap at kappa=1/8 allows only sqrt(2.5*8)
    ang = np.linspace(0.5, 3.0, 6)
    right = 8.0 * np.stack([np.sin(ang), 1.0 - np.cos(ang)], axis=1)
    throttle, steer, brake, dbg = trk.act(right, ego_speed=5.0)
    assert steer > 0.1
    left = right * np.array([1.0, -1.0])
    _, steer_l, _, _ = trk.act(left, ego_speed=5.0)
    assert np.isclose(steer_l, -steer)
    # the sharp arc must cap v_t below the raw spacing speed
    assert dbg["v_t"] < dbg["v_wp"]


def test_tracker_ema_smoothing():
    # jittering plan: EMA damps the lookahead target swing -> less steer
    noisy = WaypointTracker(ema=0.0)
    smooth = WaypointTracker(ema=0.6)
    rng = np.random.default_rng(3)
    steers_n, steers_s = [], []
    for _ in range(40):
        wp = straight_wp(3.0) + rng.normal(0, 0.8, (6, 2))
        _, sn, _, _ = noisy.act(wp, ego_speed=6.0)
        _, ss, _, _ = smooth.act(wp, ego_speed=6.0)
        steers_n.append(sn)
        steers_s.append(ss)
    assert np.std(steers_s) < 0.65 * np.std(steers_n)
    # cruise steady state: a plan constant in the vehicle frame (it
    # rolls with the car) must pass through the filter unbiased
    trk = WaypointTracker(ema=0.6)
    for _ in range(30):
        _, _, _, dbg = trk.act(straight_wp(3.0), ego_speed=6.0)
    assert abs(dbg["v_t"] - 6.0) < 0.05


def test_tracker_creep_gate():
    trk = WaypointTracker(creep_after=3, creep_throttle=0.33)
    for _ in range(3):
        throttle, _, brake, _ = trk.act(np.zeros((6, 2)), ego_speed=0.0)
        assert throttle == 0.0 and brake == 1.0
    throttle, _, brake, _ = trk.act(np.zeros((6, 2)), ego_speed=0.0)
    assert throttle == pytest.approx(0.33) and brake == 0.0
    # default tracker never creeps
    trk2 = WaypointTracker()
    for _ in range(10):
        throttle, _, brake, _ = trk2.act(np.zeros((6, 2)), ego_speed=0.0)
    assert throttle == 0.0 and brake == 1.0
