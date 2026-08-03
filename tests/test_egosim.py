"""G0 gates for the v0.2 egosim (V02_PLAN §4).

The synthetic clip integrates a physically CONSISTENT ego trajectory in
the compass convention (heading theta moves the ego along world
(sin theta, -cos theta)), unlike test_bench2drive's layout-only fixture
— fidelity/reward claims need motion and heading to agree.
"""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ditto_av.bench2drive import clips_to_npz
from ditto_av.egosim import (EgoSim, GlobalLog, RewardParams, SimParams,
                             _obb_overlap_any)
from ditto_av.models.policy_v2 import TokenPolicy


def _traj(n, v=5.0, yaw_rate=0.0, x0=100.0, y0=50.0, th0=0.7):
    xs, ys, ths = [x0], [y0], [th0]
    for _ in range(n - 1):
        x, y, th = xs[-1], ys[-1], ths[-1]
        xs.append(x + v * 0.1 * math.sin(th))
        ys.append(y - v * 0.1 * math.cos(th))
        ths.append(th + yaw_rate * 0.1)
    return xs, ys, ths


def _make_clip(root: Path, name: str, n=90, v=5.0, yaw_rate=0.0):
    anno = root / name / "anno"
    anno.mkdir(parents=True)
    xs, ys, ths = _traj(n + 25, v=v, yaw_rate=yaw_rate)
    for i in range(n):
        x, y, th = xs[i], ys[i], ths[i]
        # neighbor rides the same path 20 frames (~10 m) ahead
        nx, ny, nth = xs[i + 20], ys[i + 20], ths[i + 20]
        cx = x + 20.0 * math.sin(th)
        cy = y - 20.0 * math.cos(th)
        frame = {
            "x": x, "y": y, "theta": th, "speed": v,
            "throttle": 0.5, "steer": 0.0, "brake": 0.0,
            "x_command_near": cx, "y_command_near": cy,
            "command_near": 4,
            "x_command_far": x + 45.0 * math.sin(th),
            "y_command_far": y - 45.0 * math.cos(th),
            "command_far": 4,
            "bounding_boxes": [
                {"class": "ego_vehicle", "id": "ego",
                 "location": [x, y, 0.0],
                 "rotation": [0, 0, math.degrees(th - math.pi / 2)],
                 "extent": [2.45, 1.06, 0.75]},
                {"class": "vehicle", "id": "v1",
                 "location": [nx, ny, 0.0],
                 "rotation": [0, 0, math.degrees(nth - math.pi / 2)],
                 "extent": [2.3, 1.0, 0.75]},
            ],
        }
        with gzip.open(anno / f"{i:05d}.json.gz", "wt") as f:
            json.dump(frame, f)
    return root / name


@pytest.fixture(scope="module")
def glog(tmp_path_factory):
    root = tmp_path_factory.mktemp("egosim")
    c1 = _make_clip(root, "straight", yaw_rate=0.0)
    c2 = _make_clip(root, "curved", yaw_rate=0.15)
    clips_to_npz([c1, c2], root / "log.npz", with_route=True,
                 with_waypoints=6, with_global=True)
    return GlobalLog([root / "log.npz"], device="cpu")


def _sim(glog, **rew):
    return EgoSim(glog, SimParams(), RewardParams(**rew))


def _expert_state(glog, frames):
    ego = glog.ego[frames]
    return ego[:, 0:2].clone(), ego[:, 2].clone(), ego[:, 3].clone()


# ------------------------------------------------------------ G0: obs


def test_obs_identity_with_offline_adapter(glog):
    """Sim obs at the logged ego pose == the offline adapter's obs."""
    sim = _sim(glog)
    frames = torch.arange(glog.obs.shape[0])
    xy, th, v = _expert_state(glog, frames)
    obs = sim.build_obs(frames, xy, th, v)
    err = (obs - glog.obs).abs()
    assert float(err.max()) < 1e-4, f"max obs err {float(err.max())}"


# ------------------------------------------- G0: expert replay fidelity


@pytest.mark.parametrize("ep", [0, 1])
def test_expert_replay_retraces_the_log(glog, ep):
    """Feeding the logged wp labels through step_ego retraces the
    expert path (the analytic-ego fidelity gate; < 0.3 m mean)."""
    sim = _sim(glog)
    s, e = glog.episodes[ep]
    H = 40
    frames = torch.tensor([s])
    xy, th, v = _expert_state(glog, frames)
    errs = []
    for t in range(H):
        plan = glog.wp[frames]
        xy, th, v = sim.step_ego(plan, xy, th, v)
        frames = frames + 1
        errs.append(float((glog.ego[frames][:, 0:2] - xy).norm()))
    assert np.mean(errs) < 0.3, f"mean path err {np.mean(errs):.3f}"
    assert errs[-1] < 0.6, f"final path err {errs[-1]:.3f}"


# ---------------------------------------------------- G0: reward sanity


def test_reward_expert_is_near_one(glog):
    sim = _sim(glog)
    frames = torch.arange(5, 60)
    xy, th, v = _expert_state(glog, frames)
    r = sim.reward(frames, xy, th, v)
    assert float(r.min()) > 0.98


def test_reward_monotone_in_lateral_offset(glog):
    sim = _sim(glog)
    frames = torch.arange(10, 50)
    xy, th, v = _expert_state(glog, frames)
    c, s = torch.cos(th), torch.sin(th)
    lat_world = torch.stack([c, s], -1)  # compass ego +x in world coords
    prev = None
    for d in (0.0, 0.5, 1.0, 2.0, 4.0):
        r = float(sim.reward(frames, xy + d * lat_world, th, v).mean())
        if prev is not None:
            assert r < prev - 1e-4, f"not decreasing at offset {d}"
        prev = r


def test_reward_speed_and_heading_terms(glog):
    sim = _sim(glog)
    frames = torch.arange(10, 50)
    xy, th, v = _expert_state(glog, frames)
    base = float(sim.reward(frames, xy, th, v).mean())
    assert float(sim.reward(frames, xy, th, v + 3.0).mean()) < base - 0.2
    assert float(sim.reward(frames, xy, th + 0.5, v).mean()) < base - 0.2


def test_reward_time_tolerance(glog):
    """Running a few frames behind the expert on its own path stays
    near-max (tau window) but a large lag decays."""
    sim = _sim(glog, tau=5)
    frames = torch.arange(20, 50)
    xy, th, v = _expert_state(glog, frames)
    r_lag3 = float(sim.reward(frames + 3, xy, th, v).mean())
    r_lag15 = float(sim.reward(frames + 15, xy, th, v).mean())
    assert r_lag3 > 0.95
    assert r_lag15 < 0.5


# ------------------------------------------------------- G0: collisions


def test_collision_flags(glog):
    sim = _sim(glog)
    frames = torch.arange(10, 30)
    xy, th, v = _expert_state(glog, frames)
    assert not sim.collisions(frames, xy, th).any()  # 10 m gap
    nb = glog.act[frames][:, 0, 1:3]                 # neighbor position
    assert sim.collisions(frames, nb, th).all()      # on top of it


def test_obb_sat_rotated_cases():
    xy_a = torch.zeros(1, 2)
    yaw_a = torch.zeros(1)
    ext_a = torch.tensor([[2.0, 1.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)

    def overlap(bx, byaw, bex=(2.0, 1.0)):
        return bool(_obb_overlap_any(
            xy_a, yaw_a, ext_a,
            torch.tensor([[[bx, 0.0]]]), torch.tensor([[byaw]]),
            torch.tensor([[list(bex)]]), mask)[0])

    assert overlap(3.9, 0.0)            # 4.0 sum extent, axis-aligned
    assert not overlap(4.1, 0.0)
    # 45 deg box: projection on x = (2+1)/sqrt(2) ~ 2.12
    assert overlap(4.0, math.pi / 4)
    assert not overlap(4.3, math.pi / 4)


# ------------------------------------------------- G0: kinematic step


def test_step_ego_straight_plan(glog):
    sim = _sim(glog)
    th0 = 0.7
    xy = torch.tensor([[0.0, 0.0]])
    th = torch.tensor([th0])
    v = torch.tensor([5.0])
    # straight-ahead plan: compass ego frame forward = -y, 2.5 m spacing
    plan = torch.tensor([[0.0, -2.5 * (j + 1)] for j in range(6)]) \
        .flatten().unsqueeze(0) / 20.0
    for _ in range(10):
        xy, th, v = sim.step_ego(plan, xy, th, v)
    assert abs(float(th[0]) - th0) < 1e-5
    assert abs(float(v[0]) - 5.0) < 1e-5
    # 10 steps * 0.5 m along heading (sin th, -cos th)
    expect = 5.0 * torch.tensor([math.sin(th0), -math.cos(th0)])
    assert torch.allclose(xy[0], expect, atol=1e-4)


def test_divergent_reset_perturbs_only_masked(glog):
    sim = _sim(glog)
    starts = torch.arange(10, 26)
    rng = torch.Generator().manual_seed(0)
    xy0, th0, v0 = _expert_state(glog, starts)
    xy, th, v = sim.reset(starts, {"frac": 1.0, "lat_sigma": 0.5,
                                   "yaw_sigma": 0.1, "v_sigma": 1.0}, rng)
    assert (xy - xy0).norm(dim=-1).mean() > 0.1
    xy, th, v = sim.reset(starts, {"frac": 0.0, "lat_sigma": 0.5}, rng)
    assert torch.equal(xy, xy0)


# ------------------------------------------------- real-clip fidelity


def test_real_clip_replay_fidelity(tmp_path):
    """Same gate on a real Bench2Drive clip (skips off-cluster). Real
    experts accelerate/brake — this is what caught the schedule-speed
    bias the synthetic constant-speed clips cannot see."""
    import os
    clip = Path(os.environ.get(
        "B2D_CLIP_DIR",
        f"/scratch/{os.environ.get('USER', '')}/ditto_av/data/"
        "bench2drive/extracted/Accident_Town03_Route101_Weather23"))
    if not (clip / "anno").is_dir():
        pytest.skip("no extracted Bench2Drive clip available")
    clips_to_npz([clip], tmp_path / "real.npz", with_route=True,
                 with_waypoints=6, with_global=True)
    log = GlobalLog([tmp_path / "real.npz"], device="cpu")
    sim = EgoSim(log, SimParams(), RewardParams())
    s, e = log.episodes[0]
    assert e - s >= 60, "clip too short for the fidelity window"
    frames = torch.tensor([s + 5])
    xy, th, v = _expert_state(log, frames)
    errs = []
    for _ in range(40):
        xy, th, v = sim.step_ego(log.wp[frames], xy, th, v)
        frames = frames + 1
        errs.append(float((log.ego[frames][:, 0:2] - xy).norm()))
    assert np.mean(errs) < 0.3, f"real-clip mean err {np.mean(errs):.3f}"


# ------------------------------------------------------ policy_v2 API


def test_token_policy_shapes():
    obs_dim = 7 * 7 + 16
    pol = TokenPolicy(obs_dim, action_dim=12, d_model=32, n_layers=1,
                      n_heads=2, gru_dim=48, head_hidden=32)
    B, T = 4, 5
    obs = torch.randn(B, obs_dim).clamp(-2, 2)
    emb = pol.encode(obs)
    assert emb.shape == (B, 32)
    h = pol.step(emb, None)
    assert h.shape == (B, 48)
    feat = pol.features(emb, h)
    d = pol.dist(feat)
    a = d.sample()
    assert a.shape == (B, 12)
    assert torch.isfinite(d.log_prob(a)).all()
    assert pol.act(feat).abs().max() <= 3.0
    seq = torch.randn(T, B, obs_dim).clamp(-2, 2)
    feats, h_end = pol.unroll(seq)
    assert feats.shape == (T, B, pol.feature_dim)
    assert h_end.shape == (B, 48)
    assert pol.value(feats).shape == (T, B)


# ------------------------------------------------ trainer smoke (CPU)


def test_clp_end_to_end_smoke(glog, tmp_path):
    from ditto_av.config import Config
    from ditto_av.trainers.clp_trainer import (evaluate_in_sim,
                                               sim_from_config,
                                               train_clp_bc, train_clp_rl)
    cfg = Config()
    cfg.run_dir = str(tmp_path / "run")
    cfg.device = "cpu"
    cfg.env.action_space = "continuous"
    cfg.env.wp_head = True
    cfg.env.extra_obs_dims = 16
    cfg.env.global_arrays = True
    c = cfg.clp
    c.d_model, c.n_layers, c.n_heads = 32, 1, 2
    c.gru_dim, c.head_hidden = 32, 32
    c.bc_steps, c.bc_batch, c.bc_seq = 30, 8, 8
    c.rl_steps, c.horizon, c.rl_batch, c.burn_in = 5, 10, 8, 4

    train_clp_bc(cfg, glog, seed=0)
    train_clp_rl(cfg, glog, seed=0)
    assert (Path(cfg.run_dir) / "checkpoints" / "clp_bc.pt").exists()
    assert (Path(cfg.run_dir) / "checkpoints" / "clp_rl.pt").exists()

    from ditto_av.models.policy_v2 import make_token_policy
    pol = make_token_policy(cfg)
    pol.load_state_dict(torch.load(
        Path(cfg.run_dir) / "checkpoints" / "clp_rl.pt"))
    sim = sim_from_config(cfg, glog)
    m = evaluate_in_sim(pol, sim, glog, c.horizon, c.burn_in,
                        batch=8, n_batches=1)
    assert all(np.isfinite(v) for v in m.values())
