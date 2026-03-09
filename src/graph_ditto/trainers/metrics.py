"""
Metrics utilities for Graph-DITTO training.
"""

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Simple metrics accumulator for training/validation."""

    def __init__(self):
        self._running: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            val = v.item() if torch.is_tensor(v) else v
            self._running[k] = self._running.get(k, 0.0) + val
            self._counts[k] = self._counts.get(k, 0) + 1

    def compute(self) -> Dict[str, float]:
        return {
            k: self._running[k] / max(self._counts[k], 1)
            for k in self._running
        }

    def reset(self):
        self._running.clear()
        self._counts.clear()

    def log(self, prefix: str, step: int):
        avgs = self.compute()
        msg = f"[Step {step}] {prefix} |"
        for k, v in avgs.items():
            msg += f" {k}={v:.4f}"
        logger.info(msg)
        return avgs
