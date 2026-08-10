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
    add_push_kb_arg(parser)
    add_update_report3_arg(parser)


def add_push_kb_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--push-kb",
        action="store_true",
        help="After promote, copy replication PNGs to the knowledge-base vault",
    )


def add_update_report3_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="After promote, refresh Report 3 gallery image links in the knowledge-base",
    )


def configure_push_kb(args: argparse.Namespace, promote_module: object) -> None:
    setter = getattr(promote_module, "set_push_kb_images", None)
    if setter is None:
        return
    setter(bool(getattr(args, "push_kb", False)))


def configure_update_report3(args: argparse.Namespace, promote_module: object) -> None:
    setter = getattr(promote_module, "set_update_report3", None)
    if setter is None:
        return
    setter(bool(getattr(args, "update_report", False)))


def configure_promote_publish(args: argparse.Namespace, promote_module: object) -> None:
    """Apply ``--push-kb`` and ``--update-report`` to the promote module."""
    configure_push_kb(args, promote_module)
    configure_update_report3(args, promote_module)
