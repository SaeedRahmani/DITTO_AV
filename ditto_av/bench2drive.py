"""Bench2Drive -> DITTO-AV adapter.

Converts a Bench2Drive clip (CARLA expert demonstrations, Apache-2.0,
https://huggingface.co/datasets/rethinklab/Bench2Drive) into the same
vectorized observation layout used by the highway-env pipeline:

    rows [ego, N nearest vehicles] x features
    [presence, x, y, vx, vy, cos_h, sin_h]

positions/velocities are expressed in the ego frame and normalized. Actions
are the expert's continuous controls (throttle, steer, brake). The world
model consumes these directly (continuous action vector); discrete
meta-action policies are a highway-env-only construct.

Actor velocities are estimated by finite-differencing tracked actor IDs
across consecutive frames (annotations are sampled at 10 Hz).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FPS = 10.0
POS_SCALE = 100.0    # meters, matches highway-env Kinematics feature range
VEL_SCALE = 40.0     # m/s
V_MAX = 60.0         # m/s; faster finite-difference = tracking glitch/teleport


def _load_frame(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _yaw_rad(rotation: List[float]) -> float:
    # rotation = [pitch, roll, yaw] in degrees (CARLA convention)
    return float(np.deg2rad(rotation[2]))


N_COMMANDS = 6   # CARLA: LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELEFT/RIGHT
ROUTE_DIMS = 2 * (2 + N_COMMANDS)  # near+far: rel position + one-hot command

LIGHT_STATES = 3   # CARLA TrafficLightState ints: 0 red, 1 yellow, 2 green
LIGHT_DIMS = 1 + 2 + LIGHT_STATES  # presence, rel trigger volume, one-hot


def extra_obs_layout(extra_obs_dims: int, light_obs: bool):
    """Map config to (with_route, with_lights), validating the dim count.

    Appended block order is fixed: route (16), then lights (6). Lights
    require the route block (the privileged planner always has a route).
    """
    with_route = extra_obs_dims > 0
    expected = ((ROUTE_DIMS if with_route else 0)
                + (LIGHT_DIMS if light_obs else 0))
    if extra_obs_dims != expected:
        raise ValueError(
            f"env.extra_obs_dims={extra_obs_dims} inconsistent with "
            f"light_obs={light_obs}: expected {expected} "
            f"(route {ROUTE_DIMS} + lights {LIGHT_DIMS})")
    return with_route, light_obs


def _route_block(fr: dict, world_to_ego: np.ndarray,
                 ego_xy: np.ndarray) -> np.ndarray:
    """Ego-frame route conditioning from the annotation's command points."""
    out = np.zeros(ROUTE_DIMS, dtype=np.float32)
    for i, tag in enumerate(("near", "far")):
        base = i * (2 + N_COMMANDS)
        x, y = fr.get(f"x_command_{tag}"), fr.get(f"y_command_{tag}")
        if x is not None and y is not None and np.isfinite([x, y]).all():
            rel = world_to_ego @ (np.array([x, y], dtype=np.float64) - ego_xy)
            out[base:base + 2] = rel / POS_SCALE
        cmd = int(fr.get(f"command_{tag}", 4) or 4)  # default LANEFOLLOW
        if 1 <= cmd <= N_COMMANDS:
            out[base + 2 + cmd - 1] = 1.0
    return out


def _light_block(fr: dict, world_to_ego: np.ndarray,
                 ego_xy: np.ndarray) -> np.ndarray:
    """Nearest traffic light affecting the ego, from the annotation.

    `affects_ego` marks at most one light per frame (Bench2Drive's
    most_affect_light: the light whose trigger volume the next ~50 m of
    the ego route passes through, ahead of the ego, nearest first). The
    position is the trigger volume — the stop line, not the light pole
    across the junction. Off/Unknown states leave the one-hot zero.
    """
    out = np.zeros(LIGHT_DIMS, dtype=np.float32)
    lights = [b for b in fr.get("bounding_boxes", [])
              if b.get("class") == "traffic_light" and b.get("affects_ego")]
    if not lights:
        return out
    b = min(lights, key=lambda b: b.get("distance", 0.0))
    out[0] = 1.0
    tv = b.get("trigger_volume_location")
    if tv is not None and np.isfinite(tv[:2]).all():
        rel = world_to_ego @ (np.asarray(tv[:2], dtype=np.float64) - ego_xy)
        out[1:3] = rel / POS_SCALE
    state = int(b.get("state", -1))
    if 0 <= state < LIGHT_STATES:
        out[3 + state] = 1.0
    return out


def load_clip(clip_dir: Path, n_neighbors: int = 6,
              radius: float = 60.0,
              with_route: bool = False,
              with_lights: bool = False) -> Dict[str, np.ndarray]:
    """Parse one clip directory (containing `anno/*.json.gz`)."""
    clip_dir = Path(clip_dir)
    frames = sorted((clip_dir / "anno").glob("*.json.gz"))
    if not frames:
        raise FileNotFoundError(f"no annotations under {clip_dir}/anno")

    raw = [_load_frame(p) for p in frames]

    # Bench2Drive annotations occasionally record theta = NaN for isolated
    # frames; carry the last finite heading forward (backfill leading NaNs).
    yaws = np.array([float(fr["theta"]) for fr in raw])
    finite = np.isfinite(yaws)
    if not finite.any():
        raise ValueError(f"no finite ego heading in {clip_dir}")
    if not finite.all():
        idx = np.where(finite, np.arange(len(yaws)), -1)
        np.maximum.accumulate(idx, out=idx)
        idx[idx < 0] = np.flatnonzero(finite)[0]
        yaws = yaws[idx]

    # actor world positions per frame, keyed by id, for velocity estimation
    tracks: List[Dict[str, np.ndarray]] = []
    for fr in raw:
        d = {}
        for b in fr.get("bounding_boxes", []):
            if b.get("class") in ("vehicle", "walker", "bicycle"):
                d[b["id"]] = np.array(b["location"][:2], dtype=np.float64)
        tracks.append(d)

    n_feat = 7
    core_dim = (1 + n_neighbors) * n_feat
    obs_dim = core_dim + (ROUTE_DIMS if with_route else 0) \
        + (LIGHT_DIMS if with_lights else 0)
    obs = np.zeros((len(raw), obs_dim), dtype=np.float32)
    action = np.zeros((len(raw), 3), dtype=np.float32)
    reset = np.zeros((len(raw),), dtype=bool)
    reset[0] = True

    for t, fr in enumerate(raw):
        ego_xy = np.array([fr["x"], fr["y"]], dtype=np.float64)
        ego_yaw = float(yaws[t])
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        world_to_ego = np.array([[c, s], [-s, c]])

        rows = np.zeros((1 + n_neighbors, n_feat), dtype=np.float32)
        # ego row: ego-frame kinematics (heading 0 by construction)
        rows[0] = [1.0, 0.0, 0.0, float(fr["speed"]) / VEL_SCALE, 0.0,
                   1.0, 0.0]

        neighbors = []
        for b in fr.get("bounding_boxes", []):
            if b.get("class") not in ("vehicle", "walker", "bicycle"):
                continue
            pos = np.array(b["location"][:2], dtype=np.float64)
            rel = world_to_ego @ (pos - ego_xy)
            dist = float(np.linalg.norm(rel))
            if dist > radius or dist < 1e-6:
                continue
            # finite-difference world velocity from the previous frame;
            # implausible speeds mean the actor teleported/respawned between
            # frames — treat like a first appearance instead of feeding the
            # world model a spurious ~100 m/s spike
            prev = tracks[t - 1].get(b["id"]) if t > 0 else None
            vel_w = (pos - prev) * FPS if prev is not None else np.zeros(2)
            if np.linalg.norm(vel_w) > V_MAX:
                vel_w = np.zeros(2)
            vel_rel = world_to_ego @ vel_w
            yaw_rel = _yaw_rad(b["rotation"]) - ego_yaw
            neighbors.append((dist, [
                1.0, rel[0] / POS_SCALE, rel[1] / POS_SCALE,
                vel_rel[0] / VEL_SCALE, vel_rel[1] / VEL_SCALE,
                float(np.cos(yaw_rel)), float(np.sin(yaw_rel))]))
        neighbors.sort(key=lambda x: x[0])
        for i, (_, row) in enumerate(neighbors[:n_neighbors]):
            rows[1 + i] = row

        obs[t, :core_dim] = np.clip(rows, -2.0, 2.0).reshape(-1)
        off = core_dim
        if with_route:
            obs[t, off:off + ROUTE_DIMS] = np.clip(
                _route_block(fr, world_to_ego, ego_xy), -2.0, 2.0)
            off += ROUTE_DIMS
        if with_lights:
            obs[t, off:off + LIGHT_DIMS] = np.clip(
                _light_block(fr, world_to_ego, ego_xy), -2.0, 2.0)
        action[t] = [float(fr["throttle"]), float(fr["steer"]),
                     float(fr["brake"])]

    return {"obs": obs, "action": action, "reset": reset}


def clips_to_npz(clip_dirs: List[Path], out_path: Path,
                 n_neighbors: int = 6,
                 with_route: bool = False,
                 with_lights: bool = False) -> Dict[str, np.ndarray]:
    """Convert clips into one npz compatible with TrajectoryData."""
    parts = [load_clip(d, n_neighbors=n_neighbors, with_route=with_route,
                       with_lights=with_lights)
             for d in clip_dirs]
    data = {
        "obs": np.concatenate([p["obs"] for p in parts]),
        "action": np.concatenate([p["action"] for p in parts]),
        "reset": np.concatenate([p["reset"] for p in parts]),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return data
