"""Leaderboard team-agent entry point.

The leaderboard loads the agent file by path (imp.load_source), which
breaks relative imports inside packages — so this thin wrapper puts the
repo on sys.path and re-exports DittoCarlaAgent as a proper package
import. Point --agent at THIS file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ditto_av.carla_agent import DittoCarlaAgent  # noqa: E402,F401

assert DittoCarlaAgent is not None, \
    "carla/leaderboard not importable — check PYTHONPATH"


def get_entry_point():
    return "DittoCarlaAgent"
