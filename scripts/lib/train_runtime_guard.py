"""Crash diagnostics and exit logging for long-running training scripts (TASK-97)."""

from __future__ import annotations

import atexit
import sys
import traceback
from typing import Callable


def install_crash_hooks(*, label: str = "training") -> None:
    """Install excepthook + atexit handlers so silent exits leave a traceback trail."""

    _orig_excepthook = sys.excepthook

    def _excepthook(exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        print(
            f"\n=== FATAL {label}: unhandled exception ===",
            flush=True,
        )
        traceback.print_exception(exc_type, exc, tb)
        print(f"=== exit due to unhandled {exc_type.__name__} ===", flush=True)
        _orig_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _on_exit() -> None:
        print(f"=== {label}: process exiting (atexit) ===", flush=True)

    atexit.register(_on_exit)


def run_main(main: Callable[[], int], *, label: str = "training") -> int:
    """Run ``main()`` with crash hooks and a final exit-status line."""
    from rl_adaptive_dbs.thread_limits import DEFAULT_TRAIN_MAX_THREADS, bootstrap_thread_limits

    bootstrap_thread_limits(default=DEFAULT_TRAIN_MAX_THREADS)
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
