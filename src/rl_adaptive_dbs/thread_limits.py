"""Cap Numba/OpenBLAS thread pools for plant-heavy runs.

Set limits **before** NumPy/NumPy-dependent imports when possible:

- ``uv run python -m rl_adaptive_dbs.run scripts/probes/alphabet_diversity/run_plant_continuity_probe.py`` (default cap: 1 thread)
- ``uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/4a/plot.py``
- ``rl-dbs train ...`` (default cap: 1 thread for train/eval/benchmark)
- ``RL_DBS_MAX_THREADS=1`` in the environment (override fallback)

``taskset`` pins logical CPUs; ``--max-threads`` caps in-process thread pools
(one worker per pool — typically one logical CPU worth of parallel math per process).
Use both for a hard resource budget.
"""

from __future__ import annotations

import argparse
import os
import sys

THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

ENV_VAR = "RL_DBS_MAX_THREADS"

# Default in-process thread cap for plant-heavy standalone scripts and CLI runs.
# Override with ``--max-threads N`` or ``RL_DBS_MAX_THREADS``.
DEFAULT_PLANT_HEAVY_MAX_THREADS = 1
DEFAULT_PROBE_MAX_THREADS = DEFAULT_PLANT_HEAVY_MAX_THREADS
DEFAULT_TRAIN_MAX_THREADS = DEFAULT_PLANT_HEAVY_MAX_THREADS

PLANT_HEAVY_CLI_SUBCOMMANDS = frozenset({"train", "eval", "benchmark"})

_GLOBAL_FLAGS_WITH_VALUE = frozenset(
    {"--config", "--seed", "--max-threads", "--results-dir"},
)
_GLOBAL_FLAGS_BOOL = frozenset({"--verbose", "--quiet", "-v", "-q"})

MAX_THREADS_HELP = (
    "Cap Numba/OpenBLAS thread pools (sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, "
    "NUMBA_NUM_THREADS, etc.). Apply before NumPy import via rl-dbs entry, "
    "rl_adaptive_dbs.run, or RL_DBS_MAX_THREADS. "
    f"Plant-heavy entry points default to {DEFAULT_PLANT_HEAVY_MAX_THREADS} when unset."
)


def apply_max_threads(n: int) -> None:
    """Set standard thread-pool env vars to *n* (must be >= 1)."""
    if n < 1:
        msg = f"max threads must be >= 1, got {n}"
        raise ValueError(msg)
    value = str(int(n))
    for key in THREAD_ENV_KEYS:
        os.environ[key] = value
    os.environ[ENV_VAR] = value


def _read_env_max_threads() -> int | None:
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"{ENV_VAR} must be an integer, got {raw!r}"
        raise ValueError(msg) from exc


def parse_max_threads_from_argv(argv: list[str] | None = None) -> int | None:
    """Return ``--max-threads N`` from *argv* (default ``sys.argv[1:]``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--max-threads", type=int, default=None)
    parsed, _ = pre.parse_known_args(args)
    return parsed.max_threads


def first_cli_subcommand(argv: list[str] | None = None) -> str | None:
    """Return the first ``rl-dbs`` positional subcommand, skipping known global flags."""
    args = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in _GLOBAL_FLAGS_BOOL:
            index += 1
            continue
        if arg in _GLOBAL_FLAGS_WITH_VALUE:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg
    return None


def default_max_threads_for_cli(argv: list[str] | None = None) -> int | None:
    """Default thread cap for plant-heavy ``rl-dbs`` subcommands."""
    subcommand = first_cli_subcommand(argv)
    if subcommand in PLANT_HEAVY_CLI_SUBCOMMANDS:
        return DEFAULT_TRAIN_MAX_THREADS
    return None


def bootstrap_thread_limits(
    argv: list[str] | None = None,
    *,
    default: int | None = None,
) -> int | None:
    """Apply ``--max-threads``, ``RL_DBS_MAX_THREADS``, or *default*; return cap or None."""
    flag = parse_max_threads_from_argv(argv)
    if flag is not None:
        if flag < 1:
            raise SystemExit("--max-threads must be >= 1")
        apply_max_threads(flag)
        return flag
    env_n = _read_env_max_threads()
    if env_n is not None:
        if env_n < 1:
            raise SystemExit(f"{ENV_VAR} must be >= 1")
        apply_max_threads(env_n)
        return env_n
    if default is not None:
        if default < 1:
            msg = f"default max threads must be >= 1, got {default}"
            raise ValueError(msg)
        apply_max_threads(default)
        return default
    return None


def add_max_threads_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--max-threads`` on an ``ArgumentParser`` (for help text)."""
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        metavar="N",
        help=MAX_THREADS_HELP,
    )


def format_thread_limit_banner(n: int | None) -> str | None:
    if n is None:
        return None
    keys = ", ".join(THREAD_ENV_KEYS[:3])
    return f"thread pools capped at {n} ({keys}, ...)"
