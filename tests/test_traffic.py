"""v0.3 Phase-C gates: scene windows, invariant features, model API,
and a tiny overfit sanity (can the model learn a constant-velocity
world at all — the floor below the floor)."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ditto_av.bench2drive import clips_to_npz
from ditto_av.models.traffic import (TrafficModel, build_scene_windows,
                                     featurize)


def _make_clip(root: Path, n=60):
    anno = root / "clip" / "anno"
    anno.mkdir(parents=True)
    for t in range(n):
        x, y = 100.0 + 0.5 * t, 50.0
        boxes = [{"class": "ego_vehicle", "id": 1,
                  "location": [x, y, 0.0], "rotation": [0, 0, -90.0],
                  "extent": [2.45, 1.06, 0.75]},
                 {"class": "vehicle", "id": 100,
                  "location": [x + 10.0, y, 0.0],
                  "rotation": [0, 0, 0.0], "extent": [2.3, 1.0, 0.7]},
                 {"class": "vehicle", "id": 200,
                  "location": [150.0, y - 8.0 + 0.3 * t, 0.0],
                  "rotation": [0, 0, 90.0], "extent": [2.3, 1.0, 0.7]},
                 {"class": "walker", "id": 400,
                  "location": [x + 5.0, y - 10.0 + 0.15 * t, 0.0],
                  "rotation": [0, 0, 90.0],
                  "extent": [0.3, 0.3, 0.9]}]
        frame = {"x": x, "y": y, "theta": 0.0, "speed": 5.0,
                 "throttle": 0.5, "steer": 0.0, "brake": 0.0,
                 "bounding_boxes": boxes}
        with gzip.open(anno / f"{t:05d}.json.gz", "wt") as f:
            json.dump(frame, f)
    return root / "clip"


@pytest.fixture(scope="module")
def sw(tmp_path_factory):
    root = tmp_path_factory.mktemp("traffic")
    clip = _make_clip(root)
    data = clips_to_npz([clip], root / "log.npz", with_route=True,
                        with_waypoints=6, with_global=True)
    return build_scene_windows(data, hist=10)


def _tensors(sw, idx):
    return (torch.as_tensor(sw.hist[idx]),
            torch.as_tensor(sw.pres_mask[idx]),
            torch.as_tensor(sw.cls[idx]).long(),
            torch.as_tensor(sw.ego[idx]),
            torch.as_tensor(sw.light[idx]),
            torch.as_tensor(sw.target[idx]),
            torch.as_tensor(sw.pred_mask[idx]))


def test_scene_windows_shapes_and_targets(sw):
    assert len(sw.frames) > 20
    assert (sw.pred_mask <= sw.pres_mask).all()
    assert np.isfinite(sw.target[sw.pred_mask]).all()
    # constant-velocity vehicle id=100 moves 0.5 m/frame along +x with
    # yaw 0 -> local forward delta 0.5, lateral ~0, dyaw ~0
    for i in range(0, len(sw.frames), 7):
        for a in np.where(sw.pred_mask[i])[0]:
            if sw.cls[i, a] == 0 and abs(sw.target[i, a, 0] - 0.5) < 0.2:
                assert abs(sw.target[i, a, 1]) < 1e-3
                assert abs(sw.target[i, a, 2]) < 1e-3
    # walker id=400: 0.15 m/frame forward in its own frame
    walk = (sw.cls == 1) & sw.pred_mask
    assert walk.any()
    tw = sw.target[walk]
    assert np.allclose(tw[:, 0], 0.15, atol=1e-3)


def test_featurize_translation_invariance(sw):
    h, pres, cls, ego, light, _, _ = _tensors(sw, slice(0, 8))
    f1, c1 = featurize(h, pres, cls, ego, light)
    h2 = h.clone()
    h2[..., 0:2] += torch.tensor([1000.0, -500.0])
    ego2 = ego.clone()
    ego2[:, 0:2] += torch.tensor([1000.0, -500.0])
    f2, c2 = featurize(h2, pres, cls, ego2, light)
    assert torch.allclose(f1, f2, atol=1e-4)
    assert torch.allclose(c1, c2)


def test_model_shapes_and_overfit(sw):
    torch.manual_seed(0)
    model = TrafficModel(hist=10, d_model=64, n_layers=2, n_heads=4)
    h, pres, cls, ego, light, tgt, pm = _tensors(sw, slice(0, 32))
    d = model.dist(h, pres, cls, ego, light)
    assert d.base_dist.loc.shape == (h.shape[0], h.shape[1], 3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    first = None
    for i in range(300):
        loss = model.loss(h, pres, cls, ego, light, tgt, pm.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < first - 1.0, (first, float(loss))
    # rollout step: predicted next positions close to truth after overfit
    nxt = model.step(h, pres, cls, ego, light)
    assert torch.isfinite(nxt[pres]).all()
    cur = h[:, :, -1]
    moved = (nxt[..., 0:2] - cur[..., 0:2]).norm(dim=-1)
    err = []
    for b in range(h.shape[0]):
        for a in torch.where(pm[b])[0]:
            yaw = cur[b, a, 4]
            c, s = torch.cos(yaw), torch.sin(yaw)
            true_next = cur[b, a, 0:2] + torch.stack(
                [c * tgt[b, a, 0] - s * tgt[b, a, 1],
                 s * tgt[b, a, 0] + c * tgt[b, a, 1]])
            err.append(float((nxt[b, a, 0:2] - true_next).norm()))
    assert np.mean(err) < 0.15, np.mean(err)
    assert moved[pm].mean() > 0.05
