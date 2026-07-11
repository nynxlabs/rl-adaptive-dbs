"""Repository path helpers (no benchmarks imports)."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` (or cwd) until ``pyproject.toml`` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            if (candidate / "suites").is_dir():
                return candidate
            return candidate
    return current
