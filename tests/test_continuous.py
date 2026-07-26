"""Continuous-action (Bench2Drive) path: Gaussian actor, trainers, pipeline."""
import numpy as np
import pytest
import torch

from ditto_av.config import load_config
from ditto_av.data import TrajectoryData, build_latent_bank
from ditto_av.models.nets import GaussianActorCritic, make_actor_critic
from ditto_av.models.world_model import VectorWorldModel
from ditto_av.trainers.ac_trainer import train_latent_policy
from ditto_av.trainers.bc_trainer import train_bc
from ditto_av.trainers.wm_trainer import train_world_model


@pytest.fixture
def cfg(tmp_path):
    c = load_config(None)
    c.run_dir = str(tmp_path / "run")
    c.env.action_space = "continuous"
    c.wm.embed_dim, c.wm.deter_dim = 16, 32
    c.wm.stoch_dim = c.wm.stoch_rank = 4
    c.wm.hidden_dim = 32
    c.wm.train_steps, c.wm.batch_size, c.wm.seq_len = 30, 8, 8
    c.ac.train_steps, c.ac.batch_size, c.ac.horizon = 5, 8, 4
    c.ac.k_modes, c.ac.n_negatives = 2, 2
    c.ac.hidden_dim = c.bc.hidden_dim = 32
    c.bc.train_steps = 30
    return c


@pytest.fixture
def cont_data(tmp_path):
    """Synthetic continuous-action trajectories in the b2d layout."""
    rng = np.random.default_rng(0)
    T, n_ep = 240, 4
    obs = rng.normal(0, 0.3, (T, 49)).astype(np.float32)
    action = rng.uniform([0, -1, 0], [1, 1, 1], (T, 3)).astype(np.float32)
    reset = np.zeros(T, dtype=bool)
    reset[:: T // n_ep] = True
    path = tmp_path / "cont.npz"
    np.savez_compressed(path, obs=obs, action=action, reset=reset)
    return TrajectoryData([path])


def test_gaussian_actor_critic_basics():
    ac = GaussianActorCritic(20, 3, hidden_dim=16, layers=1,
                             low=[0, -1, 0], high=[1, 1, 1])
    feat = torch.randn(7, 20)
    d = ac.dist(feat)
    lp = d.log_prob(torch.zeros(7, 3))
    assert lp.shape == (7,) and torch.isfinite(lp).all()
    assert d.entropy().shape == (7,)
    a = ac.act(feat, stochastic=True)
    assert a.shape == (7, 3)
    assert (a >= ac.low - 1e-6).all() and (a <= ac.high + 1e-6).all()
    # analytic KL between two Gaussian policies works (BC trust region)
    ac2 = GaussianActorCritic(20, 3, hidden_dim=16, layers=1)
    kl = torch.distributions.kl.kl_divergence(ac.dist(feat), ac2.dist(feat))
    assert kl.shape == (7,) and torch.isfinite(kl).all()


def test_make_actor_critic_dispatch():
    assert isinstance(make_actor_critic(True, 10, 3, 16, 1),
                      GaussianActorCritic)
    assert not isinstance(make_actor_critic(False, 10, 5, 16, 1),
                          GaussianActorCritic)


def test_continuous_wm_and_bank(cfg, cont_data):
    assert not cont_data.discrete_actions
    wm = train_world_model(cfg, cont_data, seed=0)
    bank = build_latent_bank(wm, cont_data, cfg.env.action_dim,
                             cfg.ac.horizon, cfg.device)
    assert bank.action.dtype == torch.float32
    assert bank.action.shape[1] == 3
    assert bank.n_windows > 0


def test_continuous_bc_and_ditto_end_to_end(cfg, cont_data):
    wm = train_world_model(cfg, cont_data, seed=0)
    wm.requires_grad_(False)
    bank = build_latent_bank(wm, cont_data, cfg.env.action_dim,
                             cfg.ac.horizon, cfg.device)
    bc = train_bc(cfg, bank, seed=0)
    assert isinstance(bc, GaussianActorCritic)
    pol = train_latent_policy(cfg, wm, bank, reward_mode="multi", seed=0)
    assert isinstance(pol, GaussianActorCritic)
    # bc_init path ran (bc.pt existed) and produced a usable policy
    feat = bank.feat[:5]
    a = pol.act(feat)
    assert a.shape == (5, 3) and torch.isfinite(a).all()
    assert (cfg.dirs()["ckpt"] / "ditto_multi.pt").exists()
