import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ditto_av.bench2drive import clips_to_npz, load_clip
from ditto_av.data import TrajectoryData

# real-clip test: set B2D_CLIP_DIR to an extracted clip directory, or rely on
# the DelftBlue default layout (see scripts/validate_b2d.py); skips if absent
SCRATCH_CLIP = Path(os.environ.get(
    "B2D_CLIP_DIR",
    f"/scratch/{os.environ.get('USER', '')}/ditto_av/data/bench2drive/"
    "extracted/Accident_Town03_Route101_Weather23"))


def write_frame(path: Path, t: float, theta: float = 0.0):
    """Synthetic Bench2Drive-style annotation frame."""
    ego_x, ego_y = 100.0 + 5.0 * t, 50.0
    frame = {
        "x": ego_x, "y": ego_y, "theta": theta, "speed": 5.0,
        "throttle": 0.6, "steer": 0.05, "brake": 0.0,
        "bounding_boxes": [
            {"class": "ego_vehicle", "id": "ego",
             "location": [ego_x, ego_y, 0.0], "rotation": [0, 0, 0.0]},
            # leader 20 m ahead, driving at the same speed
            {"class": "vehicle", "id": "v1",
             "location": [ego_x + 20.0, ego_y, 0.0], "rotation": [0, 0, 0.0]},
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
