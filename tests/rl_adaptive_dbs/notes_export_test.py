"""Tests for the optional notes-folder export location."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rl_adaptive_dbs.notes_export import require_notes_dir, resolve_notes_dir
from rl_adaptive_dbs.user_config import resolve_config


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.delenv("RL_DBS_NOTES_DIR", raising=False)
    monkeypatch.delenv("RL_DBS_CONFIG", raising=False)
    return root


def _tracker(root: Path) -> Path:
    path = root / "figures" / "nguyen" / "replications.md"
    path.parent.mkdir(parents=True)
    return path


def test_unset_raises_hint(repo: Path) -> None:
    tracker = _tracker(repo)
    tracker.write_text("# local tracker\n", encoding="utf-8")
    assert resolve_notes_dir([tracker], repo_root=repo) is None
    with pytest.raises(FileNotFoundError, match="RL_DBS_NOTES_DIR"):
        require_notes_dir([tracker], repo_root=repo)


def test_yaml_key(repo: Path, tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    (repo / ".rl-dbs.yaml").write_text(f"export:\n  notes_dir: {notes}\n", encoding="utf-8")
    assert resolve_config().notes_dir == notes
    assert require_notes_dir(repo_root=repo) == notes.resolve()


def test_env_overrides_yaml(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / ".rl-dbs.yaml").write_text(f"export:\n  notes_dir: {tmp_path / 'a'}\n", encoding="utf-8")
    monkeypatch.setenv("RL_DBS_NOTES_DIR", str(tmp_path / "b"))
    assert require_notes_dir(repo_root=repo) == (tmp_path / "b").resolve()


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_detects_from_symlinked_tracker(repo: Path, tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    target = notes / "figures" / "nguyen" / "replications.md"
    target.parent.mkdir(parents=True)
    target.write_text("# tracker\n", encoding="utf-8")
    tracker = _tracker(repo)
    tracker.symlink_to(target)
    assert require_notes_dir([tracker], repo_root=repo) == notes.resolve()
