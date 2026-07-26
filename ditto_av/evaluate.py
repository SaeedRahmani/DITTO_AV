from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .envs import featurize, make_env
from .expert import AGGRESSIVE, CONSERVATIVE, ScriptedExpert
from .models.nets import ActorCritic
from .models.world_model import VectorWorldModel


class WMPolicyDriver:
    """Closed-loop driver: filters observations through the world-model
    posterior and acts from the latent features."""

    def __init__(self, wm: VectorWorldModel, policy: ActorCritic,
                 action_dim: int, device: str, stochastic: bool = False):
        self.wm, self.policy = wm, policy
        self.action_dim = action_dim
        self.device = device
        self.stochastic = stochastic
        self.reset()

    def reset(self):
        self.state = None
        self.prev_action: Optional[int] = None

    @torch.no_grad()
    def act(self, env, obs) -> int:
        obs_t = torch.as_tensor(featurize(obs), device=self.device)
        obs_t = obs_t.view(1, 1, -1)
        first = self.state is None
        if first:
            self.state = self.wm.init_state(1)
            act_t = torch.zeros((1, 1, self.action_dim), device=self.device)
        else:
            act_t = F.one_hot(
                torch.tensor([[self.prev_action]], device=self.device),
                self.action_dim).float()
        reset_t = torch.tensor([[first]], dtype=torch.bool,
                               device=self.device)
        feat, _, self.state = self.wm.observe(obs_t, act_t, reset_t,
                                              self.state)
        a = self.policy.act(feat[0, 0], stochastic=self.stochastic)
        self.prev_action = a
        return a


class ExpertDriver:
    """Scripted expert with per-episode style sampling (reference)."""

    def __init__(self, aggressive_prob: float = 0.5, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.p = aggressive_prob
        self.reset()

    def reset(self):
        style = AGGRESSIVE if self.rng.random() < self.p else CONSERVATIVE
        self.expert = ScriptedExpert(style)

    def act(self, env, obs) -> int:
        return self.expert.act(env)


class RandomDriver:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, env, obs) -> int:
        return int(self.rng.integers(0, 5))


def evaluate_driver(cfg: Config, driver, n_episodes: int, seed: int,
                    vehicles_density: Optional[float] = None,
                    vehicles_count: Optional[int] = None) -> Dict:
    env = make_env(cfg.env, vehicles_density=vehicles_density,
                   vehicles_count=vehicles_count)
    returns, lengths, speeds, crashes = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        driver.reset()
        done, ep_ret, ep_len, crashed = False, 0.0, 0, False
        ep_speeds = []
        while not done:
            a = driver.act(env, obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ep_ret += float(r)
            ep_len += 1
            ep_speeds.append(float(info.get("speed", np.nan)))
            crashed = crashed or bool(info.get("crashed", False))
        returns.append(ep_ret)
        lengths.append(ep_len)
        speeds.append(float(np.nanmean(ep_speeds)))
        crashes.append(crashed)
    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "collision_rate": float(np.mean(crashes)),
        "mean_speed": float(np.mean(speeds)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": n_episodes,
    }


def evaluate_suite(cfg: Config, drivers: Dict[str, object],
                   out_path: Optional[Path] = None) -> Dict:
    """Evaluate all drivers in-distribution and under a density shift."""
    conditions = {
        "in_distribution": {},
        "shifted": {"vehicles_density": cfg.eval.shifted_density,
                    "vehicles_count": cfg.eval.shifted_vehicles},
    }
    results: Dict[str, Dict] = {}
    for cond_name, kwargs in conditions.items():
        results[cond_name] = {}
        for name, driver in drivers.items():
            m = evaluate_driver(cfg, driver, cfg.eval.n_episodes,
                                cfg.eval.seed, **kwargs)
            results[cond_name][name] = m
            print(f"[{cond_name}] {name:14s} return "
                  f"{m['return_mean']:6.2f} ± {m['return_std']:5.2f} | "
                  f"collisions {m['collision_rate']:.2f} | "
                  f"speed {m['mean_speed']:5.2f} | len {m['mean_length']:5.1f}")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.write_text(json.dumps(results, indent=2))
        md = results_markdown(results)
        out_path.with_suffix(".md").write_text(md)
        print(f"saved {out_path} and {out_path.with_suffix('.md')}")
    return results


def results_markdown(results: Dict) -> str:
    lines = []
    for cond, table in results.items():
        lines.append(f"## {cond}\n")
        lines.append("| policy | return | collision rate | mean speed (m/s) "
                     "| episode length |")
        lines.append("|---|---|---|---|---|")
        for name, m in table.items():
            lines.append(
                f"| {name} | {m['return_mean']:.2f} ± {m['return_std']:.2f} "
                f"| {m['collision_rate']:.2f} | {m['mean_speed']:.2f} "
                f"| {m['mean_length']:.1f} |")
        lines.append("")
    return "\n".join(lines)
