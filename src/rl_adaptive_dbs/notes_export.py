"""Locate the optional notes folder that figure promotion mirrors into.

``--push-kb`` / ``--update-report`` copy replication PNGs and refresh the
Report 3 gallery in a notes folder outside the repo, laid out as
``<notes_dir>/figures/<paper>/...`` and ``<notes_dir>/reports/3.md``.

Resolution order:

1. ``RL_DBS_NOTES_DIR`` or ``export.notes_dir`` in ``.rl-dbs.yaml``.
2. Auto-detect: when a ``figures/<paper>/replications.md`` tracker is a
   symlink into a notes folder, that folder is used.

Without either, the export is unavailable and callers raise a clear error.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rl_adaptive_dbs.user_config import resolve_config

NOTES_DIR_HINT = (
    "no notes folder configured for figure export: set RL_DBS_NOTES_DIR or "
    "export.notes_dir in .rl-dbs.yaml (folder holding figures/<paper>/ and reports/3.md)"
)


def _detect_from_trackers(trackers: Iterable[Path], repo_root: Path) -> Path | None:
    root = repo_root.resolve()
    for tracker in trackers:
        if not tracker.exists():
            continue
        resolved = tracker.resolve()
        if resolved.is_relative_to(root):
            continue
        # <notes_dir>/figures/<paper>/replications.md
        return resolved.parent.parent.parent
    return None


def resolve_notes_dir(trackers: Iterable[Path] = (), *, repo_root: Path) -> Path | None:
    """Configured notes folder, else one detected from symlinked trackers, else ``None``."""
    configured = resolve_config().notes_dir
    if configured is not None:
        return configured.resolve()
    return _detect_from_trackers(trackers, repo_root)


def require_notes_dir(trackers: Iterable[Path] = (), *, repo_root: Path) -> Path:
    notes_dir = resolve_notes_dir(trackers, repo_root=repo_root)
    if notes_dir is None:
        raise FileNotFoundError(NOTES_DIR_HINT)
    return notes_dir
