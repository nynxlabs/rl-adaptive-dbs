"""Run metadata headers embedded at the top of plain-text log files.

Launch scripts write a machine-readable header when a job starts and an exit
footer when it finishes. The TUI Logs tab parses these markers to show pid,
tmux session, and live vs finished state while tailing ``.log`` files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_META_PREFIX = "# rl-dbs-run-meta: "
RUN_EXIT_PREFIX = "# rl-dbs-run-exit: "
_HEADER_READ_BYTES = 16_384
_EXIT_READ_BYTES = 8_192


@dataclass(frozen=True)
class RunLogMeta:
    """Metadata written at the start of a detached run log."""

    pid: int
    command: str
    started_at: str
    tmux_session: str | None = None
    pid_file: str | None = None
    log_file: str | None = None
    repo_root: str | None = None
    cpus: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", None) or {}
        if extra:
            payload.update(extra)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunLogMeta:
        known = {
            "pid",
            "command",
            "started_at",
            "tmux_session",
            "pid_file",
            "log_file",
            "repo_root",
            "cpus",
        }
        core = {key: payload[key] for key in known if key in payload}
        extra = {key: value for key, value in payload.items() if key not in known}
        return cls(**core, extra=extra)


@dataclass(frozen=True)
class RunLogExit:
    exit_code: int
    finished_at: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunLogExit:
        return cls(exit_code=int(payload["exit_code"]), finished_at=str(payload["finished_at"]))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def format_run_log_header(meta: RunLogMeta) -> str:
    return f"{RUN_META_PREFIX}{json.dumps(meta.to_json(), separators=(',', ':'))}\n"


def format_run_log_exit(exit_info: RunLogExit) -> str:
    return f"{RUN_EXIT_PREFIX}{json.dumps(exit_info.to_json(), separators=(',', ':'))}\n"


def write_run_log_header(path: Path, meta: RunLogMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_run_log_header(meta))
        handle.flush()


def write_run_log_exit(path: Path, *, exit_code: int) -> None:
    exit_info = RunLogExit(exit_code=exit_code, finished_at=utc_now_iso())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_run_log_exit(exit_info))
        handle.flush()


def _parse_marker_line(line: str, prefix: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(prefix):
        return None
    try:
        payload = json.loads(stripped[len(prefix) :])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_run_log_header_text(text: str) -> RunLogMeta | None:
    for line in text.splitlines()[:32]:
        payload = _parse_marker_line(line, RUN_META_PREFIX)
        if payload is not None:
            try:
                return RunLogMeta.from_json(payload)
            except (KeyError, TypeError, ValueError):
                return None
    return None


def parse_run_log_exit_text(text: str) -> RunLogExit | None:
    for line in reversed(text.splitlines()[-32:]):
        payload = _parse_marker_line(line, RUN_EXIT_PREFIX)
        if payload is not None:
            try:
                return RunLogExit.from_json(payload)
            except (KeyError, TypeError, ValueError):
                return None
    return None


def read_run_log_markers(path: Path) -> tuple[RunLogMeta | None, RunLogExit | None]:
    if not path.is_file():
        return None, None
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size <= _HEADER_READ_BYTES:
                text = handle.read().decode("utf-8", errors="replace")
                return parse_run_log_header_text(text), parse_run_log_exit_text(text)
            head = handle.read(_HEADER_READ_BYTES).decode("utf-8", errors="replace")
            handle.seek(max(0, size - _EXIT_READ_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None, None
    return parse_run_log_header_text(head), parse_run_log_exit_text(tail)


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def resolve_run_pid(meta: RunLogMeta | None, log_path: Path) -> int | None:
    if meta is not None and meta.pid > 0 and is_pid_alive(meta.pid):
        return meta.pid
    if meta is not None and meta.pid_file:
        pid_path = Path(meta.pid_file).expanduser()
        if not pid_path.is_absolute() and meta.repo_root:
            pid_path = Path(meta.repo_root).expanduser() / pid_path
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return meta.pid if meta.pid > 0 else None
        return pid if is_pid_alive(pid) else None
    if meta is not None and meta.pid > 0:
        return meta.pid
    return None


def run_state_for_log(
    log_path: Path,
    *,
    meta: RunLogMeta | None = None,
    exit_info: RunLogExit | None = None,
) -> str | None:
    if meta is None and exit_info is None:
        meta, exit_info = read_run_log_markers(log_path)
    if meta is None and exit_info is None:
        return None
    if exit_info is not None:
        return "finished" if exit_info.exit_code == 0 else "failed"
    if meta is None:
        return None
    pid = resolve_run_pid(meta, log_path)
    if pid is not None and is_pid_alive(pid):
        return "running"
    return "finished"


def run_status_label(
    log_path: Path,
    *,
    meta: RunLogMeta | None = None,
    exit_info: RunLogExit | None = None,
    state: str | None = None,
) -> str:
    if meta is None and exit_info is None:
        meta, exit_info = read_run_log_markers(log_path)
    if state is None:
        state = run_state_for_log(log_path, meta=meta, exit_info=exit_info)
    if state is None:
        return ""
    if state == "running":
        pid = resolve_run_pid(meta, log_path) if meta is not None else None
        parts = ["running"]
        if pid is not None:
            parts.append(f"pid {pid}")
        if meta is not None and meta.tmux_session:
            parts.append(f"tmux:{meta.tmux_session}")
        return " ".join(parts)
    if state == "failed":
        code = exit_info.exit_code if exit_info is not None else "?"
        return f"failed (exit {code})"
    return "finished"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write run metadata markers into log files")
    sub = parser.add_subparsers(dest="command", required=True)

    header = sub.add_parser("write-header", help="Append a run header to a log file")
    header.add_argument("--log", type=Path, required=True)
    header.add_argument("--pid", type=int, required=True)
    header.add_argument("--command", dest="command_text", required=True)
    header.add_argument("--tmux-session", default=None)
    header.add_argument("--pid-file", default=None)
    header.add_argument("--repo-root", default=None)
    header.add_argument("--cpus", default=None)
    header.add_argument("--started-at", default=None)

    exit_cmd = sub.add_parser("write-exit", help="Append a run exit footer to a log file")
    exit_cmd.add_argument("--log", type=Path, required=True)
    exit_cmd.add_argument("--exit-code", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "write-header":
        log_path = args.log.expanduser().resolve()
        meta = RunLogMeta(
            pid=args.pid,
            command=args.command_text,
            started_at=args.started_at or utc_now_iso(),
            tmux_session=args.tmux_session,
            pid_file=args.pid_file,
            log_file=str(log_path),
            repo_root=args.repo_root,
            cpus=args.cpus,
        )
        write_run_log_header(log_path, meta)
        return 0
    if args.command == "write-exit":
        write_run_log_exit(args.log.expanduser().resolve(), exit_code=args.exit_code)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
