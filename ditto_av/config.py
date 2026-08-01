from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EnvConfig:
    env_id: str = "highway-fast-v0"
    lanes_count: int = 3
    vehicles_count: int = 20
    vehicles_density: float = 1.3
    duration: int = 30
    observed_vehicles: int = 7
    obs_features: tuple = ("presence", "x", "y", "vx", "vy", "cos_h", "sin_h")
    # "discrete_meta": highway-env 5-way meta-actions;
    # "continuous": e.g. Bench2Drive (throttle, steer, brake);
    # "waypoints": Bench2Drive future ego-frame waypoints (wp_k points,
    #   compass frame, /WP_SCALE) tracked by a PID at deployment
    action_space: str = "discrete_meta"
    continuous_dims: int = 3
    wp_k: int = 6  # waypoints per action (0.5 s stride -> 3 s horizon)
    # wp_head: the WM keeps CONTROL actions (the proven feature regime)
    # and only the BC head outputs waypoints, tracked by a PID at
    # deployment with the executed control fed back to the WM. The v2
    # answer to gen3_wp's action-channel drift (wp-as-WM-action lost
    # dev-10 to control actions). Requires action_space: continuous.
    wp_head: bool = False
    # appended observation dims beyond the vehicle rows (e.g. Bench2Drive
    # route conditioning: 16 = near/far command point + one-hot command)
    extra_obs_dims: int = 0
    # also append the nearest relevant traffic light (presence, ego-frame
    # trigger volume, red/yellow/green one-hot): +6 dims -> extra_obs_dims 22
    light_obs: bool = False

    @property
    def obs_dim(self) -> int:
        return self.observed_vehicles * len(self.obs_features) \
            + self.extra_obs_dims

    @property
    def continuous(self) -> bool:
        return self.action_space in ("continuous", "waypoints")

    @property
    def waypoints(self) -> bool:
        return self.action_space == "waypoints"

    @property
    def action_dim(self) -> int:
        """The WORLD MODEL's action dim (wp_head keeps control actions)."""
        if self.waypoints:
            return 2 * self.wp_k
        return self.continuous_dims if self.continuous else 5

    @property
    def policy_action_dim(self) -> int:
        """The policy head's output dim (wp under waypoints OR wp_head)."""
        if self.wp_head:
            if not self.continuous or self.waypoints:
                raise ValueError("wp_head requires action_space: continuous")
            return 2 * self.wp_k
        return self.action_dim

    @property
    def wp_out(self) -> bool:
        """Does the deployed policy emit waypoints (either wp mode)?"""
        return self.waypoints or self.wp_head


@dataclass
class CollectConfig:
    n_expert_episodes: int = 300
    n_noisy_episodes: int = 100
    noise_eps: float = 0.3
    aggressive_prob: float = 0.5
    seed: int = 0


@dataclass
class WMConfig:
    embed_dim: int = 128
    deter_dim: int = 256
    stoch_dim: int = 16
    stoch_rank: int = 16
    hidden_dim: int = 256
    gru_layers: int = 1
    kl_weight: float = 0.3
    kl_balance: float = 0.8
    kl_free: float = 1.0
    lr: float = 3e-4
    batch_size: int = 32
    seq_len: int = 16
    train_steps: int = 4000
    grad_clip: float = 100.0
    val_fraction: float = 0.1

    @property
    def stoch_flat(self) -> int:
        return self.stoch_dim * self.stoch_rank

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_flat


@dataclass
class ACConfig:
    horizon: int = 15
    batch_size: int = 64
    train_steps: int = 3000
    gamma: float = 0.95
    lam: float = 0.95
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 3e-3
    target_tau: float = 0.01
    hidden_dim: int = 256
    layers: int = 2
    # latent matching
    reward_mode: str = "multi"  # "single" (DITTO) | "multi" (ours)
    k_modes: int = 8
    n_negatives: int = 16   # contrastive baseline windows (0 = raw DITTO)
    # also train the trajectory-consistent variant (commits to one retrieved
    # mode per rollout instead of a per-step max that can splice modes)
    train_multi_traj: bool = False
    # BC anchor: init actor from the BC checkpoint and keep a KL trust region
    # to it on imagined states (prevents drift into world-model exploits)
    bc_init: bool = True
    bc_kl_coef: float = 0.3
    grad_clip: float = 10.0


@dataclass
class BCConfig:
    train_steps: int = 3000
    batch_size: int = 512
    lr: float = 1e-3
    hidden_dim: int = 256
    layers: int = 2
    # hard-frame upweighting (gen-3 wph: the deployed plan is worst in
    # stopped/obstructed states — launches and bypass maneuvers are
    # rare per-frame in the data). Additive sampling weight multipliers;
    # 0 = uniform sampling (default, all pre-gen3 runs unaffected).
    upweight_launch: float = 0.0    # standstill frames whose plan moves
    upweight_maneuver: float = 0.0  # frames with large lateral wp


@dataclass
class EvalConfig:
    n_episodes: int = 50
    seed: int = 10_000
    shifted_density: float = 1.6
    shifted_vehicles: int = 30
    stochastic: bool = False


@dataclass
class Config:
    run_dir: str = "runs/av1"
    device: str = "cpu"
    seed: int = 0  # training seed for wm/bc/ac stages (collect has its own)
    env: EnvConfig = field(default_factory=EnvConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    wm: WMConfig = field(default_factory=WMConfig)
    ac: ACConfig = field(default_factory=ACConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def dirs(self):
        run = Path(self.run_dir)
        d = {
            "run": run,
            "data": run / "data",
            "ckpt": run / "checkpoints",
            "results": run / "results",
        }
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: Optional[str] = None) -> Config:
    """Build the default config, optionally overriding from a yaml file.

    The yaml mirrors the dataclass structure, e.g. `wm: {train_steps: 100}`.
    """
    cfg = Config()
    if path is None:
        return cfg
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for section, values in raw.items():
        if not hasattr(cfg, section):
            raise KeyError(f"Unknown config section: {section}")
        target = getattr(cfg, section)
        if isinstance(values, dict):
            for k, v in values.items():
                if not hasattr(target, k):
                    raise KeyError(f"Unknown config key: {section}.{k}")
                setattr(target, k, v)
        else:
            setattr(cfg, section, values)
    return cfg


def save_config(cfg: Config, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
