"""Process relaunch helpers for TUI development."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

RESTART_EXIT_CODE = 42
_DEV_POLL_S = 0.35


def tui_package_root() -> Path:
    """Directory containing ``rl_adaptive_dbs.tui`` modules."""
    return Path(__file__).resolve().parent


def snapshot_py_files(root: Path) -> dict[Path, float]:
    """Map ``.py`` files under ``root`` to modification times."""
    snapshot: dict[Path, float] = {}
    for path in root.rglob("*.py"):
        if path.is_file():
            snapshot[path] = path.stat().st_mtime
    return snapshot


def py_files_changed(before: dict[Path, float], root: Path) -> bool:
    """Return True when any watched ``.py`` file was added, removed, or touched."""
    return snapshot_py_files(root) != before


def _entrypoint_script() -> Path | None:
    """``rl-dbs-tui`` console script beside the active interpreter, if installed."""
    candidate = Path(sys.executable).resolve().parent / "rl-dbs-tui"
    return candidate if candidate.is_file() else None


def child_argv(argv: list[str]) -> list[str]:
    """Argv for a child TUI process (drops ``--dev``)."""
    filtered = [arg for arg in argv if arg != "--dev"]
    if filtered and Path(filtered[0]).name == "rl-dbs-tui":
        return filtered
    script = _entrypoint_script()
    if script is not None:
        return [str(script), *filtered]
    return [sys.executable, "-m", "rl_adaptive_dbs.tui", *filtered]


def run_dev_watcher(argv: list[str]) -> int:
    """Run the TUI in a subprocess and restart when TUI Python sources change."""
    command = child_argv(argv)
    watch_root = tui_package_root()
    proc: subprocess.Popen[int] | None = None

    try:
        while True:
            snapshot = snapshot_py_files(watch_root)
            proc = subprocess.Popen(command)
            while proc.poll() is None:
                time.sleep(_DEV_POLL_S)
                if py_files_changed(snapshot, watch_root):
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break
            else:
                code = proc.returncode
                if code == RESTART_EXIT_CODE:
                    continue
                return 0 if code is None else code
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return 130


def reload_tui_modules() -> None:
    """Reload TUI modules so in-process restarts pick up code edits."""
    import importlib

    import rl_adaptive_dbs.tui.app as app_module
    import rl_adaptive_dbs.tui.data as data_module
    import rl_adaptive_dbs.tui.training_data as training_data_module

    importlib.reload(data_module)
    importlib.reload(training_data_module)
    importlib.reload(app_module)


__all__ = [
    "RESTART_EXIT_CODE",
    "child_argv",
    "py_files_changed",
    "reload_tui_modules",
    "run_dev_watcher",
    "snapshot_py_files",
    "tui_package_root",
]
