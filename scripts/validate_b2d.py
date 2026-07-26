"""Validate downloaded Bench2Drive clips against the DITTO-AV adapter.

For every tarball listed in a manifest: extract only its `anno/` members
(the adapter needs nothing else), parse it with `ditto_av.bench2drive
.load_clip`, and report per-clip stats. Finally build one combined npz via
`clips_to_npz` and print aggregate shapes.

Usage:
    python scripts/validate_b2d.py manifests/b2d_clips.txt \
        --data-dir /scratch/$USER/ditto_av/data/bench2drive
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ditto_av.bench2drive import clips_to_npz, load_clip  # noqa: E402


def extract_anno(tar_path: Path, out_root: Path) -> Path:
    """Extract only anno/*.json.gz; returns the extracted clip directory."""
    clip_name = tar_path.name.removesuffix(".tar.gz")
    clip_dir = out_root / clip_name
    if (clip_dir / "anno").is_dir() and any((clip_dir / "anno").iterdir()):
        return clip_dir
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if "/anno/" in m.name and m.name.endswith(".json.gz")]
        if not members:
            raise RuntimeError(f"no anno/*.json.gz members in {tar_path.name}")
        # strip any leading directories so we land at clip_dir/anno/<frame>
        for m in members:
            parts = m.name.split("/")
            m.name = "/".join(parts[parts.index("anno"):])
        tf.extractall(clip_dir, members=members)
    return clip_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="directory holding the downloaded .tar.gz clips")
    ap.add_argument("--out-npz", type=Path, default=None,
                    help="combined npz path (default <data-dir>/b2d_valid.npz)")
    args = ap.parse_args()

    names = [ln.strip() for ln in args.manifest.read_text().splitlines()
             if ln.strip()]
    out_root = args.data_dir / "extracted"
    out_root.mkdir(parents=True, exist_ok=True)

    ok_dirs, failures = [], []
    for name in names:
        tar_path = args.data_dir / name
        if not tar_path.exists():
            failures.append((name, "tarball missing"))
            continue
        try:
            clip_dir = extract_anno(tar_path, out_root)
            d = load_clip(clip_dir)
            obs, act = d["obs"], d["action"]
            assert np.isfinite(obs).all() and np.isfinite(act).all()
            assert obs.shape[1] == 49 and act.shape[1] == 3
            n_neigh = (obs.reshape(len(obs), 7, 7)[:, 1:, 0] > 0).sum(1)
            print(f"OK   {clip_dir.name}: frames={len(obs)} "
                  f"obs[{obs.min():+.2f},{obs.max():+.2f}] "
                  f"neighbors(mean)={n_neigh.mean():.1f} "
                  f"throttle(mean)={act[:, 0].mean():.2f} "
                  f"|steer|(mean)={np.abs(act[:, 1]).mean():.3f}")
            ok_dirs.append(clip_dir)
        except Exception as e:  # noqa: BLE001 - report and keep going
            failures.append((name, repr(e)))
            print(f"FAIL {name}: {e!r}")

    print(f"\n{len(ok_dirs)}/{len(names)} clips parsed")
    if ok_dirs:
        out_npz = args.out_npz or args.data_dir / "b2d_valid.npz"
        data = clips_to_npz(ok_dirs, out_npz)
        print(f"combined npz: {out_npz} obs={data['obs'].shape} "
              f"action={data['action'].shape} "
              f"episodes={int(data['reset'].sum())}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
