"""Shared argparse flags for paper panel training resume."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CHECKPOINT_INTERVAL = 50


def add_training_resume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Load checkpoint and continue training from saved episode index",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help="Episode index to resume from (default: infer from checkpoint)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N episodes during train (default {DEFAULT_CHECKPOINT_INTERVAL})",
    )
