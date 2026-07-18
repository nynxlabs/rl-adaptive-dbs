"""Run a repo script with thread limits applied before the script imports NumPy.

Usage::

    uv run python -m rl_adaptive_dbs.run scripts/probes/foo.py --other-args
    uv run python -m rl_adaptive_dbs.run --max-threads 8 scripts/foo.py  # override default
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from rl_adaptive_dbs.thread_limits import (
    DEFAULT_PLANT_HEAVY_MAX_THREADS,
    MAX_THREADS_HELP,
    apply_max_threads,
    bootstrap_thread_limits,
    format_thread_limit_banner,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_adaptive_dbs.run",
        description="Run a script after applying --max-threads / RL_DBS_MAX_THREADS.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help=MAX_THREADS_HELP,
    )
    parser.add_argument(
        "script",
        type=Path,
        help="Script path (e.g. scripts/probes/run_task177_continuous_freq_probe.py)",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the script",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.script.is_file():
        parser.error(f"script not found: {args.script}")

    if args.max_threads is not None:
        if args.max_threads < 1:
            parser.error("--max-threads must be >= 1")
        apply_max_threads(args.max_threads)
        cap = args.max_threads
    else:
        cap = bootstrap_thread_limits([], default=DEFAULT_PLANT_HEAVY_MAX_THREADS)

    banner = format_thread_limit_banner(cap)
    if banner:
        print(banner, flush=True)

    # Forward argv so downstream argparse in the script sees a normal invocation.
    forwarded = [str(args.script.resolve())] + list(args.script_args)
    if forwarded and forwarded[-1] == "--":
        forwarded = forwarded[:-1]
    sys.argv = forwarded

    runpy.run_path(forwarded[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
