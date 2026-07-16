"""Detached command launcher for the TUI Run tab."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rl_adaptive_dbs.run_log_meta import RunLogMeta, write_run_log_exit, write_run_log_header
from rl_adaptive_dbs.tui.logs_data import add_bookmark, bookmarks_file


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of starting a detached run."""

    pid: int
    log_path: Path
    command_text: str
    recipe_id: str


def detect_tmux_session() -> str | None:
    """Return the active tmux session name when inside tmux."""
    if not os.environ.get("TMUX"):
        return None
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#S"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip()
    return name or None


def tail_log_in_terminal(
    log_path: Path,
    *,
    tail_lines: int = 200,
    tmux_session: str | None = None,
) -> bool:
    """Open ``tail -f`` in a new tmux split pane below the current pane."""
    if os.name == "nt" or not shutil.which("tail") or not shutil.which("tmux"):
        return False
    session = tmux_session if tmux_session is not None else detect_tmux_session()
    if not session:
        return False
    quoted = shlex.quote(str(log_path.resolve()))
    cmd = f"tail -n {max(1, int(tail_lines))} -f {quoted}"
    try:
        result = subprocess.run(
            ["tmux", "split-window", "-v", "-p", "30", cmd],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def tail_log_command(log_path: Path, *, tail_lines: int = 200) -> str:
    """Shell command the user can run to follow a launch log manually."""
    quoted = shlex.quote(str(log_path.resolve()))
    return f"tail -n {max(1, int(tail_lines))} -f {quoted}"


def command_text(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def default_log_path(artifacts_dir: Path, recipe_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    slug = recipe_id.replace("/", "_").replace(" ", "-")[:56]
    log_dir = artifacts_dir / "tui-runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{slug}-{stamp}.log"


def _worker_argv(
    *,
    log_path: Path,
    repo_root: Path,
    pid_file: Path,
    command_argv: list[str],
    tmux_session: str | None,
    cpus: str | None,
) -> list[str]:
    base = [
        sys.executable,
        "-m",
        "rl_adaptive_dbs.tui.run_launch",
        "worker",
        "--log",
        str(log_path),
        "--repo-root",
        str(repo_root),
        "--pid-file",
        str(pid_file),
        "--command-text",
        command_text(command_argv),
    ]
    if tmux_session:
        base.extend(["--tmux-session", tmux_session])
    if cpus:
        base.extend(["--cpus", cpus])
    base.append("--")
    base.extend(command_argv)
    return base


def launch_detached(
    command_argv: list[str],
    *,
    repo_root: Path,
    artifacts_dir: Path,
    recipe_id: str,
    log_path: Path | None = None,
    tmux_session: str | None = None,
    cpus: str | None = None,
    bookmark: bool = True,
) -> LaunchResult:
    """Start ``command_argv`` in a new session; stdout/stderr go to ``log_path``."""
    if not command_argv:
        raise ValueError("command_argv must not be empty")

    log_path = (log_path or default_log_path(artifacts_dir, recipe_id)).resolve()
    pid_file = log_path.with_suffix(log_path.suffix + ".pid")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.write_text("", encoding="utf-8")

    session = tmux_session if tmux_session is not None else detect_tmux_session()
    worker = _worker_argv(
        log_path=log_path,
        repo_root=repo_root.resolve(),
        pid_file=pid_file,
        command_argv=command_argv,
        tmux_session=session,
        cpus=cpus,
    )

    popen_kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "cwd": repo_root.resolve(),
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(worker, **popen_kwargs)
    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

    if bookmark:
        add_bookmark(bookmarks_file(artifacts_dir), log_path)

    text = command_text(command_argv)
    return LaunchResult(pid=proc.pid, log_path=log_path, command_text=text, recipe_id=recipe_id)


def _run_command(
    command_argv: list[str],
    *,
    repo_root: Path,
    log_path: Path,
    cpus: str | None,
) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    run_argv = list(command_argv)
    if cpus and os.name != "nt":
        run_argv = ["taskset", "-c", cpus, *run_argv]
    with log_path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(
            run_argv,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return int(result.returncode)


def worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Detached run worker (internal)")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--command-text", required=True)
    parser.add_argument("--tmux-session", default=None)
    parser.add_argument("--cpus", default=None)
    parser.add_argument("command_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command_argv = args.command_argv
    if command_argv and command_argv[0] == "--":
        command_argv = command_argv[1:]
    if not command_argv:
        return 2

    log_path = args.log.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    pid_file = args.pid_file.expanduser().resolve()

    meta = RunLogMeta(
        pid=os.getpid(),
        command=args.command_text,
        started_at=datetime.now(UTC).isoformat(),
        tmux_session=args.tmux_session,
        pid_file=str(pid_file),
        log_file=str(log_path),
        repo_root=str(repo_root),
        cpus=args.cpus,
    )
    write_run_log_header(log_path, meta)

    exit_code = _run_command(command_argv, repo_root=repo_root, log_path=log_path, cpus=args.cpus)
    write_run_log_exit(log_path, exit_code=exit_code)
    pid_file.unlink(missing_ok=True)
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detached TUI run launcher")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("worker", help="Internal worker entry (do not call manually)")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        return worker_main(argv[1:])
    _build_parser().parse_args(argv)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
