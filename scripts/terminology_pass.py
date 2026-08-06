#!/usr/bin/env python3
"""Repo-wide terminology pass (2026-08-06), applied to every branch.

Renames, with the reasoning that justifies each:

1. dev-10 -> test-10        the 10-route set is a TEST gate, not a dev
                            scratchpad; "dev" implied it was tuned on.
2. 220 routes -> full 220 routes   distinguishes the complete benchmark
                            from any subset at a glance.
3. arm -> variant           "arm" is clinical-trial / bandit vocabulary,
                            not AV-benchmark vocabulary; the eval
                            harness already says VARIANTS.
4. RL (when it means OUR method) -> DITTO-AV v0.X   naming the method
                            instead of the algorithm family. Generic
                            uses (reinforcement learning, A2C, rl_steps)
                            stay: they are field-standard. Applied BY
                            HAND, not by this script — it is a semantic
                            call, not a string swap.
5. in-sim -> in-WM          the counterfactual world IS the world model;
                            "sim" collides with CARLA-the-simulator,
                            which is the exact distinction the term
                            exists to draw. evaluate_in_sim ->
                            evaluate_in_wm (internal symbol, no on-disk
                            artifact depends on it).

WHAT IS DELIBERATELY NOT RENAMED (renaming these would falsify or
break records):
- drivetransformer_bench2drive_dev10.xml — Bench2Drive's own file.
- recorded result artifacts and their keys: runs/**.json, d10_score,
  carla_results_*_d10A.json, tags like v031r_c025_d10A. They are tied
  to ledger entries by name; renaming rewrites measured history.
- checkpoint / config identifiers on disk: clp_rl.pt, rl_steps,
  b2d_v02_999s, bench220.
- outputs/PIPELINE_STATUS.md — an append-only timestamped ledger.
  New entries use the new vocabulary; old ones stand as written.

Usage: terminology_pass.py [--apply] [--root .]
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

EXTS = {".md", ".py", ".yaml", ".yml", ".sbatch", ".sh", ".txt"}
SKIP_DIRS = {".git", "docs/media", "runs"}
PROTECT = [
    ("drivetransformer_bench2drive_dev10.xml", "\x00XML\x00"),
    ("dev10_winner", "\x00WIN\x00"),          # produced artifact name
]

RULES = [
    # 1. test-10 (prose form only; bare d10/dev10 identifiers stay)
    (r"\bdev-10\b", "test-10"),
    (r"\bDev-10\b", "Test-10"),
    (r"\bdev 10\b", "test-10"),
    # 2. full 220 routes — context-aware (see _full220): must not
    # double up an existing "full" (even across a line break) and must
    # not fire inside expressions like "{n}/220 routes"
    (r"\b220 routes\b", "_full220"),
    (r"\b220-route\b", "_full220"),
    # 3. variant
    (r"\barm\b", "variant"),
    (r"\barms\b", "variants"),
    (r"\bArm\b", "Variant"),
    (r"\bArms\b", "Variants"),
    # 5. in-WM
    (r"\bin-sim\b", "in-WM"),
    (r"\bIn-sim\b", "In-WM"),
    (r"\bevaluate_in_sim\b", "evaluate_in_wm"),
]


def tracked_files(root: Path):
    out = subprocess.run(["git", "ls-files"], cwd=root,
                         capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        p = Path(line)
        if p.suffix not in EXTS:
            continue
        if any(str(p).startswith(d) for d in SKIP_DIRS):
            continue
        yield root / p


def _full220(m: "re.Match") -> str:
    """Prefix 'full' only where it reads correctly: not after an
    existing 'full' (possibly across a newline), and not glued to a
    number/slash as in '{n}/220 routes' or '216/220 routes'."""
    before = m.string[:m.start()]
    if before.rstrip().lower().endswith("full"):
        return m.group(0)
    if before and before[-1] in "/-0123456789":
        return m.group(0)
    return "full " + m.group(0)


def convert(text: str):
    for lit, ph in PROTECT:
        text = text.replace(lit, ph)
    n = 0
    for pat, rep in RULES:
        text, k = re.subn(pat, _full220 if rep == "_full220" else rep,
                          text)
        n += k
    for lit, ph in PROTECT:
        text = text.replace(ph, lit)
    return text, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    total = 0
    for f in tracked_files(root):
        old = f.read_text()
        new, n = convert(old)
        if n:
            total += n
            print(f"{n:4d}  {f.relative_to(root)}")
            if args.apply:
                f.write_text(new)
    print(f"{'APPLIED' if args.apply else 'DRY RUN'}: {total} replacements")


if __name__ == "__main__":
    main()
