"""Log file discovery and tailing for the TUI Logs tab."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from rl_adaptive_dbs.run_log_meta import (
    RunLogExit,
    RunLogMeta,
    read_run_log_markers,
    run_state_for_log,
    run_status_label,
)

DEFAULT_TAIL_LINES = 200
LOG_SUFFIXES = frozenset({".jsonl", ".log"})
_TAIL_CHUNK_BYTES = 65_536


@dataclass(frozen=True)
class LogFile:
    """A log file shown in the Logs tab file list."""

    path: Path
    source: str
    display_path: str
    mtime: float
    size: int
    is_bookmark: bool = False
    run_meta: RunLogMeta | None = None
    run_exit: RunLogExit | None = None
    run_state: str | None = None


def bookmarks_file(artifacts_dir: Path) -> Path:
    """Persistent bookmark list co-located with training artifacts."""
    return artifacts_dir / ".tui-log-bookmarks.json"


def load_bookmarks(path: Path) -> list[Path]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    bookmarks: list[Path] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            bookmarks.append(Path(item).expanduser())
    return bookmarks


def save_bookmarks(path: Path, bookmarks: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [str(item) for item in bookmarks]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def add_bookmark(bookmarks_path: Path, log_path: Path) -> list[Path]:
    """Append ``log_path`` to bookmarks if missing; return updated list."""
    resolved = log_path.expanduser().resolve()
    bookmarks = load_bookmarks(bookmarks_path)
    if resolved not in {item.expanduser().resolve() for item in bookmarks}:
        bookmarks.append(resolved)
        save_bookmarks(bookmarks_path, bookmarks)
    return bookmarks


def _display_path(path: Path, *, base: Path | None) -> str:
    if base is not None:
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            pass
    return str(path)


def _enrich_log_file(entry: LogFile) -> LogFile:
    if entry.path.suffix.lower() != ".log":
        return entry
    meta, exit_info = read_run_log_markers(entry.path)
    if meta is None and exit_info is None:
        return entry
    state = run_state_for_log(entry.path, meta=meta, exit_info=exit_info)
    return replace(
        entry,
        run_meta=meta,
        run_exit=exit_info,
        run_state=state,
    )


def _copy_log_file(entry: LogFile, **kwargs: object) -> LogFile:
    return replace(entry, **kwargs)


def _scan_tree(root: Path, source: str) -> list[LogFile]:
    if not root.is_dir():
        return []
    entries: list[LogFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in LOG_SUFFIXES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entry = LogFile(
            path=path.resolve(),
            source=source,
            display_path=_display_path(path, base=root),
            mtime=stat.st_mtime,
            size=stat.st_size,
        )
        entries.append(_enrich_log_file(entry))
    return entries


def discover_log_files(
    results_dir: Path,
    artifacts_dir: Path,
    *,
    bookmarks: list[Path] | None = None,
) -> list[LogFile]:
    """Find JSONL and plain logs under results/, artifacts/, and bookmarks."""
    bookmark_paths = bookmarks if bookmarks is not None else load_bookmarks(bookmarks_file(artifacts_dir))
    by_path: dict[Path, LogFile] = {}

    for entry in _scan_tree(results_dir, "results"):
        by_path[entry.path] = entry
    for entry in _scan_tree(artifacts_dir, "artifacts"):
        by_path[entry.path] = entry

    ordered: list[LogFile] = []
    seen_bookmarks: set[Path] = set()
    for raw in bookmark_paths:
        path = raw.expanduser().resolve()
        if path in seen_bookmarks:
            continue
        seen_bookmarks.add(path)
        if path in by_path:
            existing = by_path[path]
            ordered.append(
                _copy_log_file(
                    existing,
                    is_bookmark=True,
                )
            )
            del by_path[path]
            continue
        if not path.is_file():
            ordered.append(
                LogFile(
                    path=path,
                    source="bookmark",
                    display_path=str(path),
                    mtime=0.0,
                    size=0,
                    is_bookmark=True,
                )
            )
            continue
        try:
            stat = path.stat()
        except OSError:
            ordered.append(
                LogFile(
                    path=path,
                    source="bookmark",
                    display_path=str(path),
                    mtime=0.0,
                    size=0,
                    is_bookmark=True,
                )
            )
            continue
        ordered.append(
            _enrich_log_file(
                LogFile(
                    path=path,
                    source="bookmark",
                    display_path=str(path),
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                    is_bookmark=True,
                )
            )
        )

    discovered = sorted(by_path.values(), key=lambda item: item.mtime, reverse=True)
    return ordered + discovered


def filter_log_files(files: list[LogFile], query: str) -> list[LogFile]:
    needle = query.strip().lower()
    if not needle:
        return files
    return [
        item
        for item in files
        if needle in item.display_path.lower()
        or needle in item.source.lower()
        or needle in (item.run_state or "").lower()
        or (
            item.run_meta is not None
            and item.run_meta.tmux_session is not None
            and needle in item.run_meta.tmux_session.lower()
        )
    ]


def select_log_file(files: list[LogFile], path: Path | None) -> LogFile | None:
    if not files:
        return None
    if path is None:
        return files[0]
    target = path.resolve()
    for item in files:
        if item.path.resolve() == target:
            return item
    return files[0]


def cycle_log_file(files: list[LogFile], active: Path | None, delta: int) -> Path | None:
    if not files:
        return None
    paths = [item.path for item in files]
    if active is None:
        return paths[0]
    try:
        index = next(i for i, path in enumerate(paths) if path.resolve() == active.resolve())
    except StopIteration:
        return paths[0]
    return paths[(index + delta) % len(paths)]


def tail_lines(path: Path, *, max_lines: int = DEFAULT_TAIL_LINES) -> tuple[list[str], str | None]:
    """Return the last ``max_lines`` of a text log file."""
    if not path.is_file():
        return [], f"not found: {path}"
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size <= _TAIL_CHUNK_BYTES:
                text = handle.read().decode("utf-8", errors="replace")
            else:
                handle.seek(max(0, size - _TAIL_CHUNK_BYTES))
                text = handle.read().decode("utf-8", errors="replace")
                newline = text.find("\n")
                if newline >= 0:
                    text = text[newline + 1 :]
        lines = text.splitlines()
        return lines[-max_lines:], None
    except OSError as exc:
        return [], str(exc)


def format_log_line(line: str, *, color: bool) -> str:
    if not color:
        return line
    upper = line.upper()
    if "ERROR" in upper:
        return f"[red]{line}[/red]"
    if "WARNING" in upper or " WARN" in upper:
        return f"[yellow]{line}[/yellow]"
    return line


def log_file_rows(files: list[LogFile]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for item in files:
        marker = "★ " if item.is_bookmark else ""
        size = f"{item.size:,} B" if item.size else "missing"
        run_label = run_status_label(
            item.path,
            meta=item.run_meta,
            exit_info=item.run_exit,
            state=item.run_state,
        )
        rows.append((item.source, f"{marker}{item.display_path}", size, run_label or "—"))
    return rows


def log_row_key(item: LogFile) -> str:
    """Stable DataTable row key for a log file."""
    return str(item.path)


def _run_detail_suffix(item: LogFile) -> str:
    label = run_status_label(
        item.path,
        meta=item.run_meta,
        exit_info=item.run_exit,
        state=item.run_state,
    )
    return f"  {label}" if label else ""


def logs_status_line(
    files: list[LogFile],
    opened: LogFile | None = None,
    *,
    highlighted: LogFile | None = None,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> str:
    if not files:
        return "Logs: no files"
    bookmark_count = sum(1 for item in files if item.is_bookmark)
    running_count = sum(1 for item in files if item.run_state == "running")
    tail_note = f"tail {tail_lines} lines"
    parts = [
        f"Logs: {len(files)} files",
        f"bookmarks: {bookmark_count}",
    ]
    if running_count:
        parts.append(f"running: {running_count}")
    parts.append(tail_note)
    if opened is not None:
        parts.append(f"viewing: {opened.display_path}{_run_detail_suffix(opened)}")
    elif highlighted is not None:
        parts.append(
            f"selected: {highlighted.display_path}{_run_detail_suffix(highlighted)} (Enter to open)"
        )
    return "  ".join(parts)


def logs_empty_message(results_dir: Path, artifacts_dir: Path) -> str:
    return (
        f"No .jsonl or .log files under {results_dir}/ or {artifacts_dir}/.\n\n"
        "Run training or benchmarks in another terminal, or press b to bookmark a path."
    )
