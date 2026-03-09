#!/usr/bin/env python3
"""
Graph-DITTO Option 1: Ego-Only Training Pipeline.

Usage:
    python -m graph_ditto.train_option1 [--config path/to/option1_config.yaml]

Stage 1: Train the graph-based world model on expert driving data.
Stage 2: Train ego-only policy via DITTO reward in the learned world model.
"""

import argparse
import logging
import os
import sys

import torch

from graph_ditto.config.config import build_config
from graph_ditto.trainers.ego_trainer import EgoTrainer
from graph_ditto.trainers.wm_trainer import WMTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Graph-DITTO Option 1: Ego-Only")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "option1_config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--wm-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained world model checkpoint (skip WM training)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["wm", "policy", "both"],
        default="both",
        help="Which stage to run",
    )
    args = parser.parse_args()

    config = build_config(args.config)
    logger.info("Config loaded: mode=%s", config.mode)
    logger.info("Device: %s", config.wm_train_config.train_device)

    wm_checkpoint = args.wm_checkpoint

    # --- Stage 1: World Model Training ---
    if args.stage in ("wm", "both"):
        logger.info("=" * 60)
        logger.info("STAGE 1: World Model Training")
        logger.info("=" * 60)

        wm_trainer = WMTrainer(config)
        wm_trainer.train()

        # Use the final checkpoint for policy training
        wm_checkpoint = str(
            wm_trainer.checkpoint_path / f"wm_step{wm_trainer.global_step}.pt"
        )
        logger.info("World model training complete. Checkpoint: %s", wm_checkpoint)

    # --- Stage 2: Ego Policy Training ---
    if args.stage in ("policy", "both"):
        if wm_checkpoint is None:
            logger.error("No world model checkpoint provided. Run with --stage=both or provide --wm-checkpoint.")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("STAGE 2: Ego Policy Training (DITTO)")
        logger.info("=" * 60)

        ego_trainer = EgoTrainer(config, wm_checkpoint)
        ego_trainer.train()

        logger.info("Ego policy training complete.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
