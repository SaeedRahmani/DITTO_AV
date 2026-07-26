from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .envs import featurize, make_env
from .expert import AGGRESSIVE, CONSERVATIVE, NoisyExpert, ScriptedExpert


def collect_episodes(cfg: Config, n_episodes: int, seed: int,
                     noise_eps: float = 0.0, verbose: bool = True) -> dict:
    """Roll out the scripted expert and record (obs, action, reset) sequences.

    Per episode, the expert style is sampled (aggressive vs conservative),
    which creates a multimodal demonstration distribution.
    """
    rng = np.random.default_rng(seed)
    env = make_env(cfg.env)

    obs_list, act_list, reset_list = [], [], []
    ep_styles, ep_returns, ep_lengths, ep_crashed = [], [], [], []

    for ep in range(n_episodes):
        style = AGGRESSIVE if rng.random() < cfg.collect.aggressive_prob else CONSERVATIVE
        driver = (NoisyExpert(style, noise_eps, rng) if noise_eps > 0
                  else ScriptedExpert(style))
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret, ep_len, crashed = False, 0.0, 0, False
        first = True
        while not done:
            action = driver.act(env)
            obs_list.append(featurize(obs))
            act_list.append(action)
            reset_list.append(first)
            first = False
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            ep_len += 1
            crashed = crashed or bool(info.get("crashed", False))
        ep_styles.append(style)
        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
        ep_crashed.append(crashed)
    env.close()

    data = {
        "obs": np.stack(obs_list).astype(np.float32),
        "action": np.array(act_list, dtype=np.int64),
        "reset": np.array(reset_list, dtype=bool),
        "ep_style": np.array([0 if s == AGGRESSIVE else 1 for s in ep_styles], dtype=np.int64),
        "ep_return": np.array(ep_returns, dtype=np.float32),
        "ep_length": np.array(ep_lengths, dtype=np.int64),
        "ep_crashed": np.array(ep_crashed, dtype=bool),
    }
    if verbose:
        n = len(ep_returns)
        print(f"collected {n} episodes | return {np.mean(ep_returns):.2f} "
              f"± {np.std(ep_returns):.2f} | crash rate {np.mean(ep_crashed):.3f} "
              f"| len {np.mean(ep_lengths):.1f} | aggressive frac "
              f"{np.mean(data['ep_style'] == 0):.2f}")
        hist = np.bincount(data["action"], minlength=5) / len(data["action"])
        print("action distribution [LEFT, IDLE, RIGHT, FASTER, SLOWER]:",
              np.round(hist, 3))
    return data


def run_collection(cfg: Config, out_dir: Optional[Path] = None) -> None:
    """Collect the expert set (for imitation) and a noisy set (for WM coverage)."""
    d = cfg.dirs()
    out_dir = out_dir or d["data"]
    print("== expert episodes ==")
    expert = collect_episodes(cfg, cfg.collect.n_expert_episodes, cfg.collect.seed)
    np.savez_compressed(out_dir / "expert.npz", **expert)
    if cfg.collect.n_noisy_episodes > 0:
        print("== noisy episodes (world-model coverage) ==")
        noisy = collect_episodes(cfg, cfg.collect.n_noisy_episodes,
                                 cfg.collect.seed + 100_000,
                                 noise_eps=cfg.collect.noise_eps)
        np.savez_compressed(out_dir / "noisy.npz", **noisy)
    print(f"saved to {out_dir}")
