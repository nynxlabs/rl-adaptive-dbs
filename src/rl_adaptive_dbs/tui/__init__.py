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
        help="Training artifacts root (default: artifacts/)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Manual tmux / probe log root (default: logs/)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Restart automatically when rl_adaptive_dbs/tui/*.py changes",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Enable color (default: monochrome)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Force monochrome (default; sets NO_COLOR)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Print ASCII table and exit (no TUI)",
    )
    return parser


def configure_color(*, enabled: bool) -> None:
    """Apply Textual color mode via NO_COLOR (read at App construction)."""
    if enabled:
        os.environ.pop("NO_COLOR", None)
    else:
        os.environ["NO_COLOR"] = "1"


def _run_interactive(
    results_dir: Path,
    *,
    artifacts_dir: Path,
    logs_dir: Path,
    refresh_s: float,
    color_enabled: bool,
) -> int:
    from rl_adaptive_dbs.tui.reload import RESTART_EXIT_CODE, reload_tui_modules
    from rl_adaptive_dbs.tui.settings_data import (
        TuiSettings,
        load_settings,
        settings_file,
    )

    defaults = TuiSettings(refresh_s=refresh_s, color_enabled=color_enabled)
    settings = load_settings(settings_file(artifacts_dir), defaults=defaults)

    while True:
        reload_tui_modules()
        settings = load_settings(settings_file(artifacts_dir), defaults=defaults)
        from rl_adaptive_dbs.tui.app import RlDbsTuiApp

        app = RlDbsTuiApp(
            results_dir,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            settings=settings,
        )
        exit_code = app.run()
        if exit_code != RESTART_EXIT_CODE:
            return 0 if exit_code is None else exit_code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_color(enabled=args.color and not args.no_color)

    if args.ascii or not sys.stdout.isatty():
        from rl_adaptive_dbs.tui.data import ascii_fallback

        print(ascii_fallback(args.results_dir))
        return 0

    if args.dev:
        from rl_adaptive_dbs.tui.reload import run_dev_watcher

        return run_dev_watcher(argv)

    return _run_interactive(
        args.results_dir,
        artifacts_dir=args.artifacts_dir,
        logs_dir=args.logs_dir,
        refresh_s=args.refresh,
        color_enabled=args.color and not args.no_color,
    )


__all__ = ["configure_color", "main"]
