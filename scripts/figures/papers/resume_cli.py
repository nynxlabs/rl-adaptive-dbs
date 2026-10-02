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
    add_export_notes_arg(parser)
    add_update_report3_arg(parser)


def add_export_notes_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export-notes",
        action="store_true",
        help="After promote, copy replication PNGs into the notes folder (RL_DBS_NOTES_DIR / export.notes_dir)",
    )
    # Former name, kept so existing commands keep working.
    parser.add_argument("--push-kb", dest="export_notes", action="store_true", help=argparse.SUPPRESS)


def add_update_report3_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="After promote, refresh Report 3 gallery image links in the notes folder",
    )


def configure_export_notes(args: argparse.Namespace, promote_module: object) -> None:
    setter = getattr(promote_module, "set_export_notes_images", None)
    if setter is None:
        return
    setter(bool(getattr(args, "export_notes", False)))


def configure_update_report3(args: argparse.Namespace, promote_module: object) -> None:
    setter = getattr(promote_module, "set_update_report3", None)
    if setter is None:
        return
    setter(bool(getattr(args, "update_report", False)))


def configure_promote_publish(args: argparse.Namespace, promote_module: object) -> None:
    """Apply ``--export-notes`` and ``--update-report`` to the promote module."""
    configure_export_notes(args, promote_module)
    configure_update_report3(args, promote_module)
