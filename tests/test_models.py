import torch

from ditto_av.config import WMConfig
from ditto_av.models.nets import ActorCritic
from ditto_av.models.world_model import VectorWorldModel


def small_cfg():
    return WMConfig(embed_dim=32, deter_dim=64, stoch_dim=8, stoch_rank=8,
                    hidden_dim=64)


def test_world_model_shapes():
    cfg = small_cfg()
    wm = VectorWorldModel(obs_dim=49, action_dim=5, cfg=cfg)
    T, B = 6, 4
    obs = torch.randn(T, B, 49)
    act = torch.zeros(T, B, 5)
    reset = torch.zeros(T, B, dtype=torch.bool)
    reset[0] = True

    feat, (h, z), out_state = wm.observe(obs, act, reset, wm.init_state(B))
    assert feat.shape == (T, B, cfg.feature_dim)
    assert h.shape == (T, B, cfg.deter_dim)
    assert z.shape == (T, B, cfg.stoch_flat)

    loss, metrics, _ = wm.training_step(obs, act, reset, wm.init_state(B))
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_dream_step_shapes():
    cfg = small_cfg()
    wm = VectorWorldModel(obs_dim=49, action_dim=5, cfg=cfg)
    B = 3
    h, z = wm.init_state(B)
    a = torch.zeros(B, 5)
    a[:, 1] = 1
    h2, z2 = wm.dream(a, (h, z))
    assert h2.shape == (B, cfg.deter_dim)
    assert z2.shape == (B, cfg.stoch_flat)


def test_reset_masking_clears_state():
    """A reset flag must zero the recurrent state before the update."""
    cfg = small_cfg()
    wm = VectorWorldModel(obs_dim=49, action_dim=5, cfg=cfg)
    B = 2
    torch.manual_seed(0)
    obs = torch.randn(1, B, 49)
    act = torch.randn(1, B, 5)
    reset = torch.ones(1, B, dtype=torch.bool)

    # a scaled input state must not change the output when reset is set
    h0 = torch.ones(B, cfg.deter_dim) * 5.0
    z0 = torch.ones(B, cfg.stoch_flat) * -3.0
    torch.manual_seed(1)
    feat_a, _, _ = wm.observe(obs, act, reset, (h0, z0))
    torch.manual_seed(1)
    feat_b, _, _ = wm.observe(obs, act, reset, (h0 * 2, z0 * 2))
    assert torch.allclose(feat_a, feat_b)


def test_actor_critic_target_update():
    ac = ActorCritic(feature_dim=32, action_dim=5, hidden_dim=32, layers=1)
    with torch.no_grad():
        for p in ac.critic.parameters():
            p.add_(1.0)
    before = [tp.clone() for tp in ac.target_critic.parameters()]
    ac.update_target(tau=0.5)
    for tp, b, p in zip(ac.target_critic.parameters(), before,
                        ac.critic.parameters()):
        assert torch.allclose(tp, b + 0.5 * (p - b))
