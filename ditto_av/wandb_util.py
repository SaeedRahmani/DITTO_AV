"""Optional Weights & Biases logging — a no-op when wandb is absent,
disabled, or no run was initialized.

DelftBlue compute nodes have no internet: jobs must set WANDB_MODE=offline
(the SLURM templates do) and `scripts/wandb_sync.sh` on a login node syncs
the offline runs to wandb.ai every couple of minutes.
"""
from __future__ import annotations

import os
from dataclasses import asdict

try:
    import wandb
    if not hasattr(wandb, "init"):
        # a bare `wandb/` DIRECTORY on sys.path (e.g. the repo root's
        # offline-run logs, when the env has no real wandb installed)
        # imports as an empty namespace package — treat as absent
        wandb = None
except ImportError:  # keep the pipeline runnable without wandb
    wandb = None


def init(cfg, name: str):
    """Start a W&B run (project `ditto-av` unless WANDB_PROJECT is set)."""
    if wandb is None or os.environ.get("WANDB_DISABLED"):
        return None
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ditto-av"),
        name=name,
        config=asdict(cfg),
        dir=os.environ.get("WANDB_DIR"),
    )


def log(metrics: dict, prefix: str = ""):
    """Log metrics if a run is active. Include your own step counter in
    `metrics` (e.g. {"step": i}) — sections use it as their x-axis."""
    if wandb is None or wandb.run is None:
        return
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb.log(metrics)


def finish():
    if wandb is not None and wandb.run is not None:
        wandb.finish()
