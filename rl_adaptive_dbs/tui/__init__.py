"""``rl-dbs-tui`` entry point and TUI subpackage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-dbs-tui", description="Browse benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Benchmark output root (default: results/)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Training artifacts root (reserved for future tabs)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Force monochrome (sets NO_COLOR)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Print ASCII table and exit (no TUI)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    if args.ascii or not sys.stdout.isatty():
        from rl_adaptive_dbs.tui.data import ascii_fallback

        print(ascii_fallback(args.results_dir))
        return 0

    from rl_adaptive_dbs.tui.app import RlDbsTuiApp

    app = RlDbsTuiApp(args.results_dir, refresh_s=args.refresh)
    app.run()
    return 0


__all__ = ["main"]
