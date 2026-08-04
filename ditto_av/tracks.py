"""ID-associated track views over the ##glob2 arrays (v0.3 Phase B).

The egosim arrays store actors as per-frame nearest-sorted SLOTS —
slot k at frame t and t+1 can be different actors. The traffic model
needs per-ACTOR histories, so this module re-associates slots into
tracks via `act_id` (audited stable: 1.8% gapped tracks, 0.02%
teleports; gapped tracks are split into contiguous segments rather
than interpolated — segments are honest observations, interpolation
would invent data).

Pure numpy; consumers wrap into torch datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# state columns extracted per step: x, y, vx_w, vy_w, yaw, ext_x, ext_y
STATE_COLS = list((1, 2, 3, 4, 5, 6, 7))
STATE_DIM = len(STATE_COLS)


@dataclass
class Track:
    episode: int
    actor_id: int
    cls: int
    t0: int                 # absolute frame index of states[0]
    states: np.ndarray      # (L, STATE_DIM), contiguous frames


def _episodes(reset: np.ndarray) -> List[Tuple[int, int]]:
    starts = np.where(reset)[0]
    ends = np.append(starts[1:], len(reset))
    return list(zip(starts.tolist(), ends.tolist()))


def build_tracks(data, min_len: int = 2) -> List[Track]:
    """npz dict/arrays -> contiguous per-actor track segments."""
    act, ids, cls = data["act_glob"], data["act_id"], data["act_cls"]
    reset = data["reset"]
    out: List[Track] = []
    for ep, (s, e) in enumerate(_episodes(reset)):
        segs = {}  # aid -> {t0, rows, cls, last}

        def close(aid):
            g = segs.pop(aid)
            if len(g["rows"]) >= min_len:
                out.append(Track(ep, aid, g["cls"], g["t0"],
                                 np.stack(g["rows"])))

        for t in range(s, e):
            for slot in np.where(ids[t] >= 0)[0]:
                aid = int(ids[t, slot])
                g = segs.get(aid)
                if g is not None and t - g["last"] > 1:
                    close(aid)
                    g = None
                if g is None:
                    g = segs[aid] = {"t0": t, "rows": [],
                                     "cls": int(cls[t, slot]), "last": t}
                g["rows"].append(
                    act[t, slot][STATE_COLS].astype(np.float32))
                g["last"] = t
        for aid in list(segs):
            close(aid)
    return out


def track_windows(tracks: List[Track], hist: int = 10, fut: int = 40,
                  stride: int = 5):
    """Training windows: (history, future) state pairs per actor.

    Returns dict of arrays: hist (N, hist, D), fut (N, fut, D),
    cls (N,), episode (N,), t0 (N,) — t0 = absolute frame index of the
    PRESENT step (last history row). None if no window fits.
    """
    H, F, C, E, T0 = [], [], [], [], []
    for tr in tracks:
        L = len(tr.states)
        for i in range(hist - 1, L - fut, stride):
            H.append(tr.states[i - hist + 1:i + 1])
            F.append(tr.states[i + 1:i + 1 + fut])
            C.append(tr.cls)
            E.append(tr.episode)
            T0.append(tr.t0 + i)
    if not H:
        return None
    return {"hist": np.stack(H), "fut": np.stack(F),
            "cls": np.array(C, dtype=np.int64),
            "episode": np.array(E, dtype=np.int64),
            "t0": np.array(T0, dtype=np.int64)}
