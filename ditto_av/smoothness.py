"""Smoothness / comfort metrics for the v0.3.2 audit (V032_PLAN 1.2).

All series are 10 Hz model ticks. Pure numpy on purpose (login-node
friendly, no scipy — same rule as layout.py). Frame conventions are
the settled v0.1 facts (egosim docstring): world_to_ego at compass
theta is R = [[c, s], [-s, c]]; in the compass ego frame FORWARD = -y
and LATERAL(right) = +x; plans/labels are compass-ego-frame waypoints
WP_STRIDE frames apart.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

FPS = 10.0


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def episode_ids(reset: np.ndarray) -> np.ndarray:
    """(N,) bool reset flags -> (N,) int episode id per frame."""
    reset = np.asarray(reset).astype(bool)
    ids = np.cumsum(reset.astype(np.int64))
    return ids - 1 if reset[0] else ids


def yaw_rate(theta: np.ndarray, ep: Optional[np.ndarray] = None
             ) -> np.ndarray:
    """Within-episode yaw rate, rad/s. Returns (N-1,) with NaN at
    episode boundaries (callers drop NaNs)."""
    theta = np.asarray(theta, dtype=np.float64)
    r = _wrap(np.diff(theta)) * FPS
    if ep is not None:
        r[np.diff(np.asarray(ep)) != 0] = np.nan
    return r


def sign_flips_per_100(x: np.ndarray, deadband: float,
                       ep: Optional[np.ndarray] = None) -> float:
    """Sign alternations of x per 100 ticks.

    A flip is counted at tick t when |x[t]| > deadband and the most
    recent above-deadband sample in the SAME episode had the opposite
    sign (deadband-crossing wobble still counts; jitter around zero
    inside the deadband does not). Rate is per 100 ticks of the whole
    series — comparable across populations of different length.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return 0.0
    ep = np.zeros(n, dtype=np.int64) if ep is None else np.asarray(ep)
    flips = 0
    last_sign = 0
    last_ep = None
    for i in range(n):
        if last_ep is None or ep[i] != last_ep:
            last_ep = ep[i]
            last_sign = 0
        if abs(x[i]) > deadband:
            s = 1 if x[i] > 0 else -1
            if last_sign != 0 and s != last_sign:
                flips += 1
            last_sign = s
    return 100.0 * flips / n


def summarize(x: np.ndarray) -> Dict[str, float]:
    """mean/std/p50/p95/p99/max/n of |finite entries|."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = np.abs(x[np.isfinite(x)])
    if len(x) == 0:
        return {"n": 0}
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "max": float(x.max()), "n": int(len(x))}


def _rot_w2e(theta: np.ndarray) -> np.ndarray:
    """(...,) theta -> (..., 2, 2) world->compass-ego rotation."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], -1),
                     np.stack([-s, c], -1)], -2)


def plan_churn(plans: np.ndarray, xy: np.ndarray, theta: np.ndarray,
               ep: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Motion-compensated disagreement of consecutive plans.

    plans (N, k, 2) compass-ego meters at each tick; xy (N, 2) world;
    theta (N,) compass. Consecutive plans target time grids offset by
    one tick, so plan t's wp_i (frame t + S*i, S = stride ticks) is
    compared against plan t+1 linearly interpolated at (S*i - 1)/S
    strides — with S = 5 that is 0.2/0.8 between polyline nodes i-1
    and i (node 0 = the origin). Both are expressed in the tick-t+1
    compass ego frame using the logged/simulated ego motion, which is
    EXACT compensation (unlike the wp1 proxy below).

    Returns dict of (N-1,) series with NaN at episode boundaries:
      lat_mean  — mean over i of |lateral disagreement| (m)
      lat_wp1   — |lateral disagreement| at wp1 (m)
      wp1_proxy — |d lateral of wp1| with NO compensation (the only
                  churn observable in CARLA tick logs; computed here
                  identically for like-for-like comparison)
    """
    plans = np.asarray(plans, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    N, k, _ = plans.shape
    R0T = _rot_w2e(theta[:-1]).swapaxes(-1, -2)      # ego(t) -> world
    R1 = _rot_w2e(theta[1:])                         # world -> ego(t+1)
    # plan t into world, then into frame t+1
    w = xy[:-1, None, :] + np.einsum("nij,nkj->nki", R0T, plans[:-1])
    q = np.einsum("nij,nkj->nki", R1, w - xy[1:, None, :])
    # plan t+1 polyline interpolated one tick back in time
    pts1 = np.concatenate([np.zeros((N - 1, 1, 2)), plans[1:]], axis=1)
    interp = 0.2 * pts1[:, :-1, :] + 0.8 * pts1[:, 1:, :]  # (N-1, k, 2)
    dlat = np.abs(q[..., 0] - interp[..., 0])         # lateral = +x
    lat_mean = dlat.mean(axis=1)
    lat_wp1 = dlat[:, 0]
    wp1_proxy = np.abs(np.diff(plans[:, 0, 0]))       # no compensation
    if ep is not None:
        bad = np.diff(np.asarray(ep)) != 0
        lat_mean[bad] = np.nan
        lat_wp1[bad] = np.nan
        wp1_proxy[bad] = np.nan
    return {"lat_mean": lat_mean, "lat_wp1": lat_wp1,
            "wp1_proxy": wp1_proxy}
