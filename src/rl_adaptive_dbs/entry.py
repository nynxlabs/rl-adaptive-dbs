"""Console entry points that bootstrap thread limits before heavy imports."""

from __future__ import annotations

import sys


def cli_main(argv: list[str] | None = None) -> int:
    from rl_adaptive_dbs.thread_limits import bootstrap_thread_limits, default_max_threads_for_cli

    bootstrap_thread_limits(argv, default=default_max_threads_for_cli(argv))
    from rl_adaptive_dbs.cli import main

    return main(argv)


def tui_main(argv: list[str] | None = None) -> int:
    from rl_adaptive_dbs.thread_limits import bootstrap_thread_limits

    bootstrap_thread_limits(argv)
    from rl_adaptive_dbs.tui import main

    return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli_main())
