"""
Data utilities for Graph-DITTO.

Includes samplers, return computation functions, and other shared data
processing tools adapted from the original DITTO codebase.
"""

import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler


class SequentialSampler(Sampler):
    """
    Equidistant batch sampler for sequential data.

    Yields batch_size equidistant indices, advancing by seq_length each step.
    Used for both world model training (seq_length > 1) and feature extraction
    (seq_length = 1).
    """

    def __init__(self, data_size: int, seq_length: int, batch_size: int, init_idx: int = None):
        self.data_size = data_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.init_idx = init_idx
        self.chunk_size = math.ceil(self.data_size / self.batch_size)
        self.n_steps = math.ceil(self.chunk_size / self.seq_length)

    def __iter__(self):
        init_idx = self.init_idx if self.init_idx is not None else np.random.randint(self.data_size)
        for i in range(self.n_steps):
            batch_indices = []
            for j in range(self.batch_size):
                start_idx = (init_idx + i * self.seq_length + j * self.chunk_size) % self.data_size
                batch_indices.append(start_idx)
            yield batch_indices

    def __len__(self):
        return self.n_steps


class EpisodeSampler(Sampler):
    """
    Sampler for policy training that respects episode boundaries.

    For any given episode, yields a start index between 0 and
    (episode_length - seq_length), ensuring sequences don't cross episodes.
    """

    def __init__(self, n_transitions: int, episode_starts: np.ndarray, seq_length: int, batch_size: int):
        self.indices = self._build_indices(n_transitions, episode_starts, seq_length)
        self.data_size = len(self.indices)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.chunk_size = math.ceil(self.data_size / self.batch_size)
        self.n_steps = math.ceil(self.chunk_size / self.seq_length)

    @staticmethod
    def _build_indices(n_transitions, episode_starts, seq_length):
        indices = []
        last_start = episode_starts[0]
        for start_idx in episode_starts[1:]:
            indices.extend(np.arange(last_start, start_idx - seq_length))
            last_start = start_idx
        indices.extend(np.arange(last_start, n_transitions - seq_length))
        return np.array(indices)

    def __iter__(self):
        init_idx = np.random.randint(self.data_size)
        for i in range(self.n_steps):
            batch_indices = []
            for j in range(self.batch_size):
                start_idx = (init_idx + i * self.seq_length + j * self.chunk_size) % self.data_size
                idx = self.indices[start_idx]
                batch_indices.append(idx)
            yield batch_indices

    def __len__(self):
        return self.n_steps


def lambda_return(
    rewards: list,
    values: list,
    lambda_: float = 0.95,
    gamma: float = 0.99,
) -> Tensor:
    """
    Compute lambda-returns (TD(lambda)).

    V^lambda_t = r_t + gamma * ((1-lambda) * v_{t+1} + lambda * V^lambda_{t+1})
    V^lambda_H = v_H  (bootstrap from value estimate at horizon)
    """
    R = values[-1]
    returns = [R]
    rewards_less_last = rewards[:-1]
    values_less_first = values[1:]
    for r_t, v_tplus1 in zip(reversed(rewards_less_last), reversed(values_less_first)):
        R = r_t + gamma * ((1 - lambda_) * v_tplus1 + lambda_ * R)
        returns.insert(0, R)
    return torch.stack(returns)


def mc_return(
    rewards: list,
    bootstrap: Tensor,
    gamma: float = 0.99,
) -> Tensor:
    """Compute discounted Monte Carlo returns with bootstrap."""
    rewards[-1] = rewards[-1] + bootstrap
    R = torch.zeros_like(rewards[0])
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.stack(returns)


def max_cos_reward(state: Tensor, target: Tensor) -> Tensor:
    """
    DITTO's intrinsic reward: modified cosine similarity.

    r_t = (z_E · z_pi) / max(||z_E||, ||z_pi||)^2

    This is the core distance function from the DITTO paper (Equation 8).
    Domain-agnostic — works the same regardless of whether the latent was
    produced by a CNN or a GNN encoder.
    """
    n_state = torch.norm(state, dim=-1, keepdim=True)
    n_target = torch.norm(target, dim=-1, keepdim=True)
    norms = torch.cat((n_state, n_target), dim=-1)
    max_norm = torch.max(norms, dim=-1)[0]
    dot_prod = (state * target).sum(dim=-1)
    reward = dot_prod / torch.square(max_norm).clamp(min=1e-8)
    return reward
