#!/usr/bin/env python3
"""v0.3.1 M2 gate: expert fidelity through the SIM's layout path.

M1 validated the geometry per-town on raw clips. This gate revalidates
through everything the sim will actually use: manifest_towns split ->
LayoutQuery per-frame dispatch -> TownLayoutTorch query, over the
cached npz ego trajectories. A wrong episode->town mapping cannot pass
(querying a trajectory against the wrong town's lanes reads ~99 m off).

Gate (pre-registered in M1): off-drivable < 1% of frames per split.

Usage: v031_layout_gate.py --data ~/ditto_out/v03_w0c/data \
           [--manifest manifests/b2d_full999.txt] [--chunk 16384]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ditto_av.layout import manifest_towns  # noqa: E402
from ditto_av.layout_torch import LayoutQuery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest",
                    default=str(REPO / "manifests" / "b2d_full999.txt"))
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()

    tr_towns, va_towns = manifest_towns(Path(args.manifest))
    ok = True
    for split, towns in (("train", tr_towns), ("val", va_towns)):
        z = np.load(Path(args.data) / f"b2d_{split}.npz")
        reset = z["reset"]
        starts = np.where(reset)[0]
        ends = np.append(starts[1:], len(reset))
        episodes = list(zip(starts.tolist(), ends.tolist()))
        assert len(towns) == len(episodes), \
            f"{split}: {len(towns)} clips vs {len(episodes)} episodes"
        lq = LayoutQuery(towns, episodes, len(reset))
        xy = torch.as_tensor(z["ego_glob"][:, 0:2], dtype=torch.float32)
        frames = torch.arange(len(reset))
        offs = torch.cat([lq.off(frames[i:i + args.chunk],
                                 xy[i:i + args.chunk])
                          for i in range(0, len(reset), args.chunk)])
        viol = (offs > 0).float()
        rate = float(viol.mean())
        per_town = defaultdict(list)
        town_of = lq.frame_town
        uniq = sorted(set(towns))
        for k, t in enumerate(uniq):
            m = town_of == k
            per_town[t] = float(viol[m].mean())
        verdict = "PASS" if rate < 0.01 else "FAIL"
        ok &= rate < 0.01
        print(f"{split}: {rate * 100:.3f}% off-drivable "
              f"({int(viol.sum())}/{len(viol)} frames) [{verdict}]")
        for t in uniq:
            print(f"  {t}: {per_town[t] * 100:.3f}%")
    print("GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
