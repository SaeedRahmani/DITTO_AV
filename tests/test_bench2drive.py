import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ditto_av.bench2drive import (clips_to_npz, extra_obs_layout, load_clip)
from ditto_av.data import TrajectoryData

# real-clip test: set B2D_CLIP_DIR to an extracted clip directory, or rely on
# the DelftBlue default layout (see scripts/validate_b2d.py); skips if absent
SCRATCH_CLIP = Path(os.environ.get(
    "B2D_CLIP_DIR",
    f"/scratch/{os.environ.get('USER', '')}/ditto_av/data/bench2drive/"
    "extracted/Accident_Town03_Route101_Weather23"))


def write_frame(path: Path, t: float, theta: float = 0.0,
                leader_dx: float = 20.0):
    """Synthetic Bench2Drive-style annotation frame."""
    ego_x, ego_y = 100.0 + 5.0 * t, 50.0
    frame = {
        "x": ego_x, "y": ego_y, "theta": theta, "speed": 5.0,
        "throttle": 0.6, "steer": 0.05, "brake": 0.0,
        "bounding_boxes": [
            {"class": "ego_vehicle", "id": "ego",
             "location": [ego_x, ego_y, 0.0], "rotation": [0, 0, 0.0]},
            # leader ahead, driving at the same speed
            {"class": "vehicle", "id": "v1",
             "location": [ego_x + leader_dx, ego_y, 0.0],
             "rotation": [0, 0, 0.0]},
            # oncoming vehicle in the other lane
            {"class": "vehicle", "id": "v2",
             "location": [ego_x + 40.0, ego_y - 4.0, 0.0],
             "rotation": [0, 0, 180.0]},
            # static sign should be ignored
            {"class": "traffic_sign", "id": "s1",
             "location": [ego_x + 10.0, ego_y + 3.0, 0.0],
             "rotation": [0, 0, 0.0]},
        ],
    }
    with gzip.open(path, "wt") as f:
        json.dump(frame, f)


@pytest.fixture
def synthetic_clip(tmp_path):
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    for i in range(10):
        write_frame(anno / f"{i:05d}.json.gz", t=i * 0.1)
    return tmp_path / "clip"


def test_load_clip_layout(synthetic_clip):
    d = load_clip(synthetic_clip, n_neighbors=6)
    assert d["obs"].shape == (10, 49)
    assert d["action"].shape == (10, 3)
    assert d["reset"][0] and not d["reset"][1:].any()
    rows = d["obs"].reshape(10, 7, 7)
    # ego row: presence, ego-frame speed
    assert (rows[:, 0, 0] == 1).all()
    assert np.allclose(rows[:, 0, 3], 5.0 / 40.0)
    # two vehicles tracked, sign ignored
    assert (rows[:, 1:, 0].sum(1) == 2).all()
    # leader is the nearest neighbor: 20 m ahead => x_rel 0.2
    assert np.allclose(rows[:, 1, 1], 0.2, atol=1e-5)


def test_leader_velocity_from_finite_difference(synthetic_clip):
    d = load_clip(synthetic_clip, n_neighbors=6)
    rows = d["obs"].reshape(10, 7, 7)
    # leader moves with the ego at 5 m/s (dt=0.1 between frames)
    assert np.allclose(rows[1:, 1, 3], 5.0 / 40.0, atol=1e-4)
    # frame 0 has no history: velocity defaults to 0
    assert np.allclose(rows[0, 1, 3], 0.0)


def test_nan_theta_frame_uses_last_finite_heading(tmp_path):
    # real Bench2Drive clips occasionally record theta = NaN for one frame
    # (e.g. BlockedIntersection_Town03_Route136_Weather6 frame 22)
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    for i in range(10):
        theta = float("nan") if i == 4 else 0.0
        write_frame(anno / f"{i:05d}.json.gz", t=i * 0.1, theta=theta)
    d = load_clip(tmp_path / "clip", n_neighbors=6)
    assert np.isfinite(d["obs"]).all()
    rows = d["obs"].reshape(10, 7, 7)
    # heading carried forward => frame 4 identical to its neighbors
    assert np.allclose(rows[4], rows[5])


def test_route_conditioning_block(tmp_path):
    # route block: ego-frame command points + one-hot commands, appended
    # after the 49 vehicle dims; tolerant of frames missing the fields
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    for i in range(6):
        write_frame(anno / f"{i:05d}.json.gz", t=i * 0.1)
        # inject command fields into every second frame
        if i % 2 == 0:
            frame = json.load(gzip.open(anno / f"{i:05d}.json.gz", "rt"))
            frame.update({"x_command_near": frame["x"] + 30.0,
                          "y_command_near": frame["y"],
                          "command_near": 3,      # STRAIGHT
                          "x_command_far": frame["x"] + 80.0,
                          "y_command_far": frame["y"] + 10.0,
                          "command_far": 1})      # LEFT
            with gzip.open(anno / f"{i:05d}.json.gz", "wt") as f:
                json.dump(frame, f)
    d = load_clip(tmp_path / "clip", n_neighbors=6, with_route=True)
    assert d["obs"].shape == (6, 49 + 16)
    route = d["obs"][:, 49:]
    # frame 0: near point 30 m ahead => 0.3; STRAIGHT one-hot at index 2
    assert np.isclose(route[0, 0], 0.3, atol=1e-5)
    assert route[0, 2 + 2] == 1.0 and route[0].sum() > 0
    # far block: LEFT one-hot at its index 0
    assert route[0, 8 + 2 + 0] == 1.0
    # frames without fields: zero offsets, LANEFOLLOW default (index 3)
    assert np.allclose(route[1, :2], 0.0)
    assert route[1, 2 + 3] == 1.0
    # default path unchanged
    d49 = load_clip(tmp_path / "clip", n_neighbors=6)
    assert d49["obs"].shape == (6, 49)


def add_light(path: Path, dx: float = 15.0, dy: float = 2.0,
              state: int = 0, affects: bool = True):
    """Append a traffic_light bounding box to a written frame."""
    frame = json.load(gzip.open(path, "rt"))
    frame["bounding_boxes"].append({
        "class": "traffic_light", "id": f"tl{int(affects)}",
        "location": [frame["x"] + dx + 12.0, frame["y"] - 6.0, 3.0],
        "rotation": [0, 0, 90.0],
        "distance": float(np.hypot(dx + 12.0, 6.0)),
        "state": state, "affects_ego": affects,
        "trigger_volume_location": [frame["x"] + dx, frame["y"] + dy, 0.0],
    })
    with gzip.open(path, "wt") as f:
        json.dump(frame, f)


def test_light_block(tmp_path):
    # light block: presence + ego-frame trigger volume + red/yellow/green
    # one-hot, appended after the route block; non-affecting lights and
    # Off/Unknown states must not leak in
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    states = [0, 1, 2, 3, 0, 0]
    for i in range(6):
        p = anno / f"{i:05d}.json.gz"
        write_frame(p, t=i * 0.1)
        add_light(p, dx=40.0, dy=-3.0, state=2, affects=False)  # decoy
        if i < 4:
            add_light(p, dx=15.0, dy=2.0, state=states[i], affects=True)
    d = load_clip(tmp_path / "clip", n_neighbors=6, with_route=True,
                  with_lights=True)
    assert d["obs"].shape == (6, 49 + 16 + 6)
    light = d["obs"][:, 65:]
    # frames 0-2: presence, trigger volume 15 m ahead / 2 m left, one-hot
    for i, st in enumerate(states[:3]):
        assert light[i, 0] == 1.0
        assert np.allclose(light[i, 1:3], [0.15, 0.02], atol=1e-5)
        expect = np.zeros(3)
        expect[st] = 1.0
        assert np.allclose(light[i, 3:], expect)
    # frame 3: light present but Off (state 3) -> presence only
    assert light[3, 0] == 1.0 and np.allclose(light[3, 3:], 0.0)
    # frames 4-5: no affecting light (decoy ignored) -> all zero
    assert np.allclose(light[4:], 0.0)
    # route-only and default layouts unchanged
    assert load_clip(tmp_path / "clip", with_route=True)["obs"].shape[1] == 65
    assert load_clip(tmp_path / "clip")["obs"].shape[1] == 49


def test_extra_obs_layout():
    assert extra_obs_layout(0, False) == (False, False)
    assert extra_obs_layout(16, False) == (True, False)
    assert extra_obs_layout(22, True) == (True, True)
    for dims, lights in ((22, False), (16, True), (6, True), (5, False)):
        with pytest.raises(ValueError):
            extra_obs_layout(dims, lights)


def test_teleport_velocity_rejected(tmp_path):
    # a tracked actor jumping 30 m between 10 Hz frames implies 300 m/s;
    # that finite-difference is a respawn artifact and must be zeroed,
    # not clipped into the observation
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    for i in range(10):
        dx = 50.0 if i >= 5 else 20.0
        write_frame(anno / f"{i:05d}.json.gz", t=i * 0.1, leader_dx=dx)
    d = load_clip(tmp_path / "clip", n_neighbors=6)
    rows = d["obs"].reshape(10, 7, 7)
    # after the jump the leader (50 m) sorts behind v2 (~40 m): row 2
    assert np.allclose(rows[5, 2, 3], 0.0)          # spike rejected
    assert np.allclose(rows[6:, 2, 3], 5.0 / 40.0,  # tracking resumes
                       atol=1e-4)


def test_clips_to_npz_roundtrip(synthetic_clip, tmp_path):
    out = tmp_path / "b2d.npz"
    clips_to_npz([synthetic_clip, synthetic_clip], out)
    td = TrajectoryData([out])
    assert len(td.episodes) == 2
    assert not td.discrete_actions
    rng = np.random.default_rng(0)
    obs, act, reset = td.sample_wm_batch(3, 6, rng, action_dim=3)
    assert obs.shape == (6, 3, 49)
    assert act.shape == (6, 3, 3)


@pytest.mark.skipif(not SCRATCH_CLIP.exists(),
                    reason="real Bench2Drive clip not downloaded")
def test_real_clip():
    d = load_clip(SCRATCH_CLIP)
    assert d["obs"].shape[1] == 49
    assert len(d["obs"]) > 100
    assert np.isfinite(d["obs"]).all()
    assert d["obs"].min() >= -2.0 and d["obs"].max() <= 2.0
