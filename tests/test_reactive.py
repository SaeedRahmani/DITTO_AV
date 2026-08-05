"""D1 gates: reactive world equivalence + dedup + stepping."""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ditto_av.bench2drive import clips_to_npz
from ditto_av.egosim import EgoSim, GlobalLog, RewardParams, SimParams
from ditto_av.models.traffic import TrafficModel, build_scene_windows
from ditto_av.reactive import ReactiveEgoSim
from tests.test_traffic import _make_clip


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("reactive")
    clip = _make_clip(root)
    data = clips_to_npz([clip], root / "log.npz", with_route=True,
                        with_waypoints=6, with_global=True)
    log = GlobalLog([root / "log.npz"], device="cpu")
    sw = build_scene_windows(data, hist=10)
    models = [TrafficModel(hist=10, d_model=32, n_layers=1, n_heads=2)
              for _ in range(2)]
    for m in models:
        m.eval()
    sim = ReactiveEgoSim(log, sw, models, data["act_id"],
                         SimParams(), RewardParams())
    return sim, log, sw


def _expert_state(log, frames):
    ego = log.ego[frames]
    return ego[:, 0:2].clone(), ego[:, 2].clone(), ego[:, 3].clone()


def test_equivalence_with_replay_before_stepping(world):
    """Right after reset_reactive (buffer seeded from the log), obs and
    collisions must EQUAL the pure-replay parent's."""
    sim, log, sw = world
    starts = torch.tensor([int(sw.frames[5])])
    xy, th, v = sim.reset_reactive(starts)
    base = EgoSim(log, sim.p, sim.r)
    xb, tb, vb = base.reset(starts)
    obs_r = sim.build_obs(starts, xy, th, v)
    obs_b = base.build_obs(starts, xb, tb, vb)
    assert torch.allclose(obs_r, obs_b, atol=1e-4), \
        float((obs_r - obs_b).abs().max())
    assert torch.equal(sim.collisions(starts, xy, th),
                       base.collisions(starts, xy, th))


def test_no_duplicate_actors(world):
    sim, log, sw = world
    starts = torch.tensor([int(sw.frames[5])])
    sim.reset_reactive(starts)
    pres = sim._buf[0, :, 0] > 0.5
    pos = sim._buf[0, pres, 1:3]
    if len(pos) > 1:
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(99.0)
        assert float(d.min()) > 0.5, "duplicated actor in buffer"


def test_reactive_stepping_runs(world):
    sim, log, sw = world
    starts = torch.tensor([int(sw.frames[5])])
    xy, th, v = sim.reset_reactive(starts)
    frames = starts.clone()
    for _ in range(8):
        sim.step_traffic(frames, xy, th, v)
        obs = sim.build_obs(frames, xy, th, v)
        assert torch.isfinite(obs).all()
        plan = log.wp[frames]
        xy, th, v = sim.step_ego(plan, xy, th, v)
        frames = frames + 1
    assert sim.last_disagreement is not None
    assert torch.isfinite(sim.last_disagreement).all()
    # reward still targets the real-log expert
    r = sim.reward(frames, xy, th, v)
    assert torch.isfinite(r).all()
