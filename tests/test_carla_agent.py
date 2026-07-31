"""Online featurizer must match the offline adapter exactly."""
import gzip
import json

import numpy as np
import torch

from ditto_av.bench2drive import load_clip
from ditto_av.carla_agent import (ContinuousWMDriver, StuckRecovery,
                                  featurize_frame, route_hits_box)
from ditto_av.config import load_config
from ditto_av.models.nets import make_actor_critic
from ditto_av.models.world_model import VectorWorldModel

from test_bench2drive import add_light, write_frame


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


def test_online_light_block_matches_offline_adapter(tmp_path):
    anno = tmp_path / "clip" / "anno"
    anno.mkdir(parents=True)
    states = [0, 1, 2]
    for i in range(6):
        p = anno / f"{i:05d}.json.gz"
        write_frame(p, t=i * 0.1)
        if i < 3:
            add_light(p, dx=18.0, dy=-1.5, state=states[i], affects=True)
    offline = load_clip(tmp_path / "clip", n_neighbors=6,
                        with_lights=True)["obs"]
    assert offline.shape[1] == 49 + 6
    prev = {}
    for i in range(6):
        ego_xy, actors = _scene(i * 0.1)
        light = ({"xy": ego_xy + [18.0, -1.5], "state": states[i]}
                 if i < 3 else {"xy": None, "state": -1})
        obs, prev = featurize_frame(ego_xy, 0.0, 5.0, actors, prev,
                                    light=light)
        np.testing.assert_allclose(obs, offline[i], atol=1e-6,
                                   err_msg=f"frame {i}")


def test_route_hits_box():
    # 1 m-spaced route through the box center: hit
    pts = np.stack([np.arange(0.0, 30.0), np.zeros(30)], axis=1)
    assert route_hits_box(pts, [10.0, 0.0], [1.5, 0.7])
    # laterally offset beyond 1.5x extent: miss (strict inequality)
    assert not route_hits_box(pts, [10.0, 1.05], [1.5, 0.7])
    assert not route_hits_box(pts[:0], [10.0, 0.0], [1.5, 0.7])


def test_stuck_recovery_gives_up_without_progress():
    # after max_consecutive recoveries with no free movement in between,
    # recovery stops overriding (leaderboard blocked-detection takes over)
    r = StuckRecovery(stuck_ticks=2, recover_ticks=2, max_consecutive=2)
    outs = [r.update(0.05, 0.8, 0.0, 0.0) for _ in range(40)]
    assert r.events == 2
    assert all(o is None for o in outs[-20:])
    # real movement resets the cap
    assert r.update(2.0, 0.6, 0.0, 0.0) is None
    for _ in range(2):
        assert r.update(0.05, 0.8, 0.0, 0.0) is None
    assert r.update(0.05, 0.8, 0.0, 0.0) is not None
    assert r.events == 3


def test_stuck_recovery():
    r = StuckRecovery(stuck_ticks=5, recover_ticks=3, steer=0.5)
    # driving normally: never triggers
    for _ in range(20):
        assert r.update(2.0, 0.6, 0.0, 0.1) is None
    # braking at a red light at standstill: never triggers
    for _ in range(20):
        assert r.update(0.0, 0.0, 1.0, 0.0) is None
    # wedged at commanded throttle: triggers on tick stuck_ticks+1
    for _ in range(5):
        assert r.update(0.05, 0.8, 0.0, 0.2) is None
    outs = [r.update(0.05, 0.8, 0.0, 0.2) for _ in range(4)]
    # exactly recover_ticks reverse overrides, steer mirrored, then done
    assert [o is not None for o in outs] == [True, True, True, False]
    for o in outs[:3]:
        throttle, steer, reverse = o
        assert reverse and throttle > 0 and steer < 0
    assert r.events == 1
    # counter restarts cleanly after recovery: reset by movement, then a
    # full stuck_ticks streak is needed again
    assert r.update(2.0, 0.6, 0.0, -0.2) is None
    for _ in range(5):
        assert r.update(0.05, 0.8, 0.0, -0.2) is None
    out = r.update(0.05, 0.8, 0.0, -0.2)
    assert out is not None and out[1] > 0  # mirrors the negative steer
    assert r.events == 2


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


# ---------------- RoutePIDDriver (Phase-0d reference controller) -------------

def _straight_plan(n=200, spacing=1.0):
    import numpy as np
    return np.stack([np.arange(n) * spacing, np.zeros(n)], axis=1)


def test_route_pid_straight_drives_forward():
    import numpy as np
    from ditto_av.carla_agent import RoutePIDDriver
    d = RoutePIDDriver(_straight_plan())
    throttle, steer, brake, dbg = d.act(
        np.array([5.0, 0.0]), 0.0, 3.0, actors=[])
    assert throttle > 0.0 and brake == 0.0
    assert abs(steer) < 0.05
    assert dbg["v_t"] > 5.0  # straight: near v_max


def test_route_pid_steers_toward_offset_plan():
    import numpy as np
    from ditto_av.carla_agent import RoutePIDDriver
    # ego displaced to the left of the lane (negative y in CARLA frame):
    # the lookahead sits to the ego's right -> positive steer
    throttle, steer, brake, dbg = RoutePIDDriver(_straight_plan()).act(
        np.array([5.0, -2.0]), 0.0, 3.0, actors=[])
    assert steer > 0.1


def test_route_pid_stops_for_lead_vehicle():
    import numpy as np
    from ditto_av.carla_agent import RoutePIDDriver
    d = RoutePIDDriver(_straight_plan())
    lead = [(1, np.array([9.0, 0.0]), 0.0)]  # 4 m ahead of ego at x=5
    throttle, steer, brake, dbg = d.act(
        np.array([5.0, 0.0]), 0.0, 2.0, actors=lead)
    assert dbg["v_t"] == 0.0
    assert throttle == 0.0


def test_route_pid_stops_at_red_light():
    import numpy as np
    from ditto_av.carla_agent import RoutePIDDriver
    d = RoutePIDDriver(_straight_plan())
    throttle, steer, brake, dbg = d.act(
        np.array([5.0, 0.0]), 0.0, 5.0, actors=[],
        light_dist=10.0, light_state=0)
    assert dbg["v_t"] == 0.0


def test_route_pid_slows_for_curve():
    import numpy as np
    from ditto_av.carla_agent import RoutePIDDriver
    t = np.linspace(0, np.pi / 2, 120)
    r = 12.0
    plan = np.stack([r * np.sin(t), r * (1 - np.cos(t))], axis=1)
    d = RoutePIDDriver(plan)
    throttle, steer, brake, dbg = d.act(
        np.array([1.0, 0.0]), 0.0, 6.0, actors=[])
    assert dbg["v_t"] < 6.0  # curvature cap engaged
    assert steer > 0.0  # curve bends toward +y
