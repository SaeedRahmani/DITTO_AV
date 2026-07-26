"""Online featurizer must match the offline adapter exactly."""
import gzip
import json

import numpy as np
import torch

from ditto_av.bench2drive import load_clip
from ditto_av.carla_agent import ContinuousWMDriver, featurize_frame
from ditto_av.config import load_config
from ditto_av.models.nets import make_actor_critic
from ditto_av.models.world_model import VectorWorldModel

from test_bench2drive import write_frame


def _scene(t):
    """The synthetic scene from write_frame, as world-state tuples."""
    ego_x, ego_y = 100.0 + 5.0 * t, 50.0
    actors = [("v1", np.array([ego_x + 20.0, ego_y]), 0.0),
              ("v2", np.array([ego_x + 40.0, ego_y - 4.0]), np.pi)]
    return np.array([ego_x, ego_y]), actors


def test_online_featurizer_matches_offline_adapter(tmp_path):
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    for i in range(6):
        write_frame(anno / f"{i:05d}.json.gz", t=i * 0.1)
        frame = json.load(gzip.open(anno / f"{i:05d}.json.gz", "rt"))
        frame.update({"x_command_near": frame["x"] + 25.0,
                      "y_command_near": frame["y"] - 2.0,
                      "command_near": 2,
                      "x_command_far": frame["x"] + 90.0,
                      "y_command_far": frame["y"] + 5.0,
                      "command_far": 5})
        with gzip.open(anno / f"{i:05d}.json.gz", "wt") as f:
            json.dump(frame, f)

    for with_route in (False, True):
        offline = load_clip(tmp_path / "clip", n_neighbors=6,
                            with_route=with_route)["obs"]
        prev = {}
        for i in range(6):
            ego_xy, actors = _scene(i * 0.1)
            route = None
            if with_route:
                route = {"near_xy": ego_xy + [25.0, -2.0], "near_cmd": 2,
                         "far_xy": ego_xy + [90.0, 5.0], "far_cmd": 5}
            obs, prev = featurize_frame(ego_xy, 0.0, 5.0, actors, prev,
                                        route=route)
            np.testing.assert_allclose(obs, offline[i], atol=1e-6,
                                       err_msg=f"frame {i} route={with_route}")


def test_continuous_wm_driver_smoke():
    cfg = load_config(None)
    cfg.env.action_space = "continuous"
    cfg.wm.embed_dim, cfg.wm.deter_dim = 16, 32
    cfg.wm.stoch_dim = cfg.wm.stoch_rank = 4
    cfg.wm.hidden_dim = 32
    wm = VectorWorldModel(49, 3, cfg.wm)
    wm.eval()
    policy = make_actor_critic(True, cfg.wm.feature_dim, 3, 32, 1)
    driver = ContinuousWMDriver(wm, policy)
    for _ in range(3):
        a = driver.act(np.zeros(49, dtype=np.float32))
        assert a.shape == (3,)
        assert 0.0 <= a[0] <= 1.0 and -1.0 <= a[1] <= 1.0 and 0.0 <= a[2] <= 1.0
    driver.reset()
    assert driver.state is None
