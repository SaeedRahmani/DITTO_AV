"""v0.3 Phase-B gates: act_id/act_cls arrays + track re-association.

The synthetic clip has deliberate CHURN — one actor leaves mid-clip,
another appears late, one walker crosses — the cases slot-sorted
arrays scramble and ID association must untangle.
"""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ditto_av.bench2drive import clips_to_npz
from ditto_av.tracks import STATE_DIM, build_tracks, track_windows


def _make_churn_clip(root: Path, n=60):
    anno = root / "clip" / "anno"
    anno.mkdir(parents=True)
    for t in range(n):
        x, y = 100.0 + 5.0 * 0.1 * t, 50.0
        boxes = [{"class": "ego_vehicle", "id": 1,
                  "location": [x, y, 0.0], "rotation": [0, 0, -90.0],
                  "extent": [2.45, 1.06, 0.75]}]
        # A: constant companion, 10 m ahead (present all frames)
        boxes.append({"class": "vehicle", "id": 100,
                      "location": [x + 10.0, y, 0.0],
                      "rotation": [0, 0, 0.0], "extent": [2.3, 1.0, 0.7]})
        # B: leaves after frame 29 (exits annotation radius)
        if t < 30:
            boxes.append({"class": "vehicle", "id": 200,
                          "location": [x + 20.0, y + 4.0, 0.0],
                          "rotation": [0, 0, 0.0],
                          "extent": [2.3, 1.0, 0.7]})
        # C: appears from frame 40
        if t >= 40:
            boxes.append({"class": "vehicle", "id": 300,
                          "location": [x - 15.0, y - 4.0, 0.0],
                          "rotation": [0, 0, 180.0],
                          "extent": [2.3, 1.0, 0.7]})
        # W: walker crossing, present all frames, moving in +y
        boxes.append({"class": "walker", "id": 400,
                      "location": [x + 5.0, y - 10.0 + 0.15 * t, 0.0],
                      "rotation": [0, 0, 90.0],
                      "extent": [0.3, 0.3, 0.9]})
        frame = {"x": x, "y": y, "theta": 0.0, "speed": 5.0,
                 "throttle": 0.5, "steer": 0.0, "brake": 0.0,
                 "bounding_boxes": boxes}
        with gzip.open(anno / f"{t:05d}.json.gz", "wt") as f:
            json.dump(frame, f)
    return root / "clip"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("tracks")
    clip = _make_churn_clip(root)
    return clips_to_npz([clip], root / "log.npz", with_route=True,
                        with_waypoints=6, with_global=True)


def test_glob2_arrays_present_and_typed(data):
    assert data["act_id"].shape == data["act_cls"].shape
    assert data["act_id"].dtype == np.int64
    assert data["act_cls"].dtype == np.int8
    # empty slots are -1 in both
    empty = data["act_glob"][..., 0] < 0.5
    assert (data["act_id"][empty] == -1).all()
    assert (data["act_cls"][empty] == -1).all()


def test_ids_follow_actors_across_slot_reshuffles(data):
    """Slot orders change with distance; ids must stay with actors."""
    ids = data["act_id"]
    act = data["act_glob"]
    for t in (0, 20, 50):
        for slot in np.where(ids[t] >= 0)[0]:
            aid = ids[t, slot]
            # actor 400 is the walker: class code 1
            if aid == 400:
                assert data["act_cls"][t, slot] == 1
            else:
                assert data["act_cls"][t, slot] == 0
    # actor 100 exists at every frame exactly once
    assert ((ids == 100).sum(axis=1) == 1).all()


def test_build_tracks_churn(data):
    tracks = build_tracks(data)
    by_id = {}
    for tr in tracks:
        by_id.setdefault(tr.actor_id, []).append(tr)
    assert set(by_id) == {100, 200, 300, 400}
    assert len(by_id[100]) == 1 and len(by_id[100][0].states) == 60
    assert len(by_id[200]) == 1 and len(by_id[200][0].states) == 30
    assert by_id[300][0].t0 == 40 and len(by_id[300][0].states) == 20
    assert by_id[400][0].cls == 1
    # states carry world positions: actor 100 tracks the ego +10 m
    s = by_id[100][0].states
    assert s.shape[1] == STATE_DIM
    assert abs(s[0, 0] - 110.0) < 1e-4
    assert abs(s[-1, 0] - (100.0 + 5.0 * 0.1 * 59 + 10.0)) < 1e-3


def test_track_windows_contiguity(data):
    tracks = build_tracks(data)
    w = track_windows(tracks, hist=5, fut=10, stride=3)
    assert w is not None
    assert w["hist"].shape[1:] == (5, STATE_DIM)
    assert w["fut"].shape[1:] == (10, STATE_DIM)
    # future must continue history: walker moves +0.15/frame in y
    walker = w["cls"] == 1
    assert walker.any()
    dy = w["fut"][walker][:, 0, 1] - w["hist"][walker][:, -1, 1]
    assert np.allclose(dy, 0.15, atol=1e-4)
