"""Crash diagnostics and default thread limits for plant-heavy probe scripts."""

from __future__ import annotations

from typing import Callable

from rl_adaptive_dbs.thread_limits import DEFAULT_PROBE_MAX_THREADS, bootstrap_thread_limits

from scripts.lib.train_runtime_guard import install_crash_hooks


def run_main(main: Callable[[], int], *, label: str = "probe") -> int:
    """Run ``main()`` with probe default thread cap and crash hooks.

    Prefer ``uv run python -m rl_adaptive_dbs.run scripts/probes/...`` when the
    script imports NumPy at module load; this helper is for scripts that defer
    heavy imports until ``main()``.
    """
    bootstrap_thread_limits(default=DEFAULT_PROBE_MAX_THREADS)
    install_crash_hooks(label=label)
    try:
        code = int(main())
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else 0
        print(f"=== {label}: SystemExit code={code} ===", flush=True)
        return code
    except BaseException:
        print(f"=== {label}: aborting with exception (see traceback above) ===", flush=True)
        raise
    else:
        print(f"=== {label}: finished normally exit={code} ===", flush=True)
        return code
