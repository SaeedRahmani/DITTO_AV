from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class TrajectoryData:
    """Flat (obs, action, reset) arrays with episode boundary bookkeeping.

    `action[t]` is the action taken *at* step t (after seeing obs[t]).
    `action_key` selects which npz array plays the action role — "wp"
    swaps in the future-waypoint targets (the Phase-1 abstraction); the
    control array stays in the npz untouched.
    """

    def __init__(self, npz_paths: Sequence[Path], action_key: str = "action"):
        obs, action, reset = [], [], []
        for p in npz_paths:
            d = np.load(p)
            obs.append(d["obs"])
            action.append(d[action_key])
            reset.append(d["reset"])
        self.obs = np.concatenate(obs).astype(np.float32)
        self.action = np.concatenate(action)
        # discrete envs store int action ids; Bench2Drive stores continuous
        # (throttle, steer, brake) vectors
        self.discrete_actions = self.action.ndim == 1
        self.action = self.action.astype(
            np.int64 if self.discrete_actions else np.float32)
        self.reset = np.concatenate(reset)
        starts = np.where(self.reset)[0]
        ends = np.append(starts[1:], len(self.obs))
        self.episodes: List[Tuple[int, int]] = list(zip(starts.tolist(),
                                                        ends.tolist()))

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[1]

    def sample_wm_batch(self, batch_size: int, seq_len: int,
                        rng: np.random.Generator, action_dim: int,
                        device: str = "cpu"):
        """Sample (T, B) windows lying inside episodes, Dreamer-layout.

        Returns obs (T,B,O), prev_action one-hot (T,B,A), reset (T,B).
        """
        valid = [(s, e) for s, e in self.episodes if e - s >= seq_len]
        obs_b, act_b, reset_b = [], [], []
        for _ in range(batch_size):
            s, e = valid[rng.integers(len(valid))]
            t0 = int(rng.integers(s, e - seq_len + 1))
            idx = np.arange(t0, t0 + seq_len)
            obs_b.append(self.obs[idx])
            # prev-action: a_{t-1}; zero (with reset flag) at episode start
            prev = np.zeros((seq_len,) + self.action.shape[1:],
                            dtype=self.action.dtype)
            r = np.zeros((seq_len,), dtype=bool)
            if t0 == s:
                prev[1:] = self.action[idx[:-1]]
                r[0] = True
            else:
                prev[0] = self.action[t0 - 1]
                prev[1:] = self.action[idx[:-1]]
            act_b.append(prev)
            reset_b.append(r)
        obs_t = torch.as_tensor(np.stack(obs_b, axis=1), device=device)
        reset_t = torch.as_tensor(np.stack(reset_b, axis=1), device=device)
        if self.discrete_actions:
            act_idx = torch.as_tensor(np.stack(act_b, axis=1), device=device)
            act_t = F.one_hot(act_idx, action_dim).float()
        else:
            act_t = torch.as_tensor(np.stack(act_b, axis=1), device=device)
        # zero the action where reset (no previous action exists)
        act_t = act_t * (~reset_t).unsqueeze(-1)
        return obs_t, act_t, reset_t


class LatentBank:
    """Expert episodes filtered through the frozen world model posterior.

    Holds per-step full latent states (h, z), features, and action labels,
    plus a window index over all length-(H+1) expert segments used both for
    seeding imagination and as latent-matching targets.
    """

    def __init__(self, h: torch.Tensor, z: torch.Tensor, feat: torch.Tensor,
                 action: torch.Tensor, ep_bounds: List[Tuple[int, int]],
                 horizon: int):
        self.h, self.z, self.feat, self.action = h, z, feat, action
        self.ep_bounds = ep_bounds
        self.horizon = horizon
        # window index: flat positions t such that [t, t+H] stays in-episode
        idx = []
        for s, e in ep_bounds:
            idx.extend(range(s, e - horizon))
        self.window_starts = torch.tensor(idx, dtype=torch.long,
                                          device=h.device)
        offsets = torch.arange(horizon + 1, device=h.device)
        self.window_idx = self.window_starts.unsqueeze(1) + offsets  # (N, H+1)
        self.windows_h = h[self.window_idx]  # (N, H+1, D)

    @property
    def n_windows(self) -> int:
        return len(self.window_starts)

    def sample_windows(self, batch_size: int,
                       rng: np.random.Generator) -> torch.Tensor:
        i = rng.integers(self.n_windows, size=batch_size)
        return torch.as_tensor(i, dtype=torch.long, device=self.h.device)

    def start_states(self, window_ids: torch.Tensor):
        flat = self.window_starts[window_ids]
        return self.h[flat].clone(), self.z[flat].clone()


@torch.no_grad()
def build_latent_bank(wm, data: TrajectoryData, action_dim: int, horizon: int,
                      device: str = "cpu") -> LatentBank:
    """Teacher-force every expert episode through the posterior."""
    wm.eval()
    hs, zs, feats, acts = [], [], [], []
    bounds, cursor = [], 0
    for s, e in data.episodes:
        T = e - s
        obs = torch.as_tensor(data.obs[s:e], device=device).unsqueeze(1)
        if data.discrete_actions:
            prev = np.zeros((T,), dtype=np.int64)
            prev[1:] = data.action[s:e - 1]
            act = F.one_hot(torch.as_tensor(prev, device=device),
                            action_dim).float().unsqueeze(1)
        else:
            prev = np.zeros((T, data.action.shape[1]), dtype=np.float32)
            prev[1:] = data.action[s:e - 1]
            act = torch.as_tensor(prev, device=device).unsqueeze(1)
        reset = torch.zeros((T, 1), dtype=torch.bool, device=device)
        reset[0, 0] = True
        act = act * (~reset).unsqueeze(-1)
        feat, (h_seq, z_seq), _ = wm.observe(obs, act, reset,
                                             wm.init_state(1))
        hs.append(h_seq.squeeze(1))
        zs.append(z_seq.squeeze(1))
        feats.append(feat.squeeze(1))
        acts.append(torch.as_tensor(data.action[s:e], device=device))
        bounds.append((cursor, cursor + T))
        cursor += T
    return LatentBank(torch.cat(hs), torch.cat(zs), torch.cat(feats),
                      torch.cat(acts), bounds, horizon)
