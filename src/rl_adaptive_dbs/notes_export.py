"""Locate the optional notes folder that figure promotion mirrors into.

``--export-notes`` copies replication PNGs into a notes folder outside the
repo. The notes folder mirrors repo-relative paths: the repo's
``docs/controllers/ddpg/README.md`` is ``<notes_dir>/docs/controllers/ddpg/README.md``
and its images are under ``<notes_dir>/docs/controllers/<controller>/figures/images/``
(``rl_adaptive_dbs.doc_paths``).

Resolution order:

1. ``RL_DBS_NOTES_DIR`` or ``export.notes_dir`` in ``.rl-dbs.yaml``.
2. Auto-detect: when a ``docs/controllers/<controller>/README.md`` tracker is a
   symlink into a notes folder, that folder is used.

Without either, the export is unavailable and callers raise a clear error.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rl_adaptive_dbs.doc_paths import all_trackers, rel_to_root
from rl_adaptive_dbs.user_config import resolve_config

NOTES_DIR_HINT = (
    "no notes folder configured for figure export: set RL_DBS_NOTES_DIR or "
    "export.notes_dir in .rl-dbs.yaml (folder mirroring the repo's docs/, e.g. "
    "docs/controllers/<controller>/)"
)


def _detect_from_trackers(trackers: Iterable[Path], repo_root: Path) -> Path | None:
    """Notes folder behind the first tracker symlinked out of the repo.

    A tracker at repo path ``docs/controllers/<controller>/README.md`` resolving to
    ``<notes_dir>/docs/controllers/<controller>/README.md`` gives ``<notes_dir>``.
    Targets whose tail does not mirror the repo-relative path are skipped.
    """
    root = repo_root.resolve()
    for tracker in trackers:
        if not tracker.exists():
            continue
        resolved = tracker.resolve()
        if resolved.is_relative_to(root):
            continue
        rel = rel_to_root(tracker, repo_root)
        if rel is None:
            continue
        rel_parts = Path(rel).parts
        if resolved.parts[-len(rel_parts) :] != rel_parts:
            continue
        return resolved.parents[len(rel_parts) - 1]
    return None


def resolve_notes_dir(trackers: Iterable[Path] | None = None, *, repo_root: Path) -> Path | None:
    """Configured notes folder, else one detected from symlinked trackers, else ``None``.

    ``trackers`` defaults to every paper tracker under ``repo_root``.
    """
    configured = resolve_config().notes_dir
    if configured is not None:
        return configured.resolve()
    if trackers is None:
        trackers = all_trackers(repo_root)
    return _detect_from_trackers(trackers, repo_root)


def require_notes_dir(trackers: Iterable[Path] | None = None, *, repo_root: Path) -> Path:
    notes_dir = resolve_notes_dir(trackers, repo_root=repo_root)
    if notes_dir is None:
        raise FileNotFoundError(NOTES_DIR_HINT)
    return notes_dir
