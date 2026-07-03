"""Tests for ``.rl-dbs.yaml`` user configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from rl_adaptive_dbs.config_show import show_config
from rl_adaptive_dbs.user_config import (
    find_config_file,
    persist_config_key,
    preview_config_key,
    resolve_config,
)


def test_find_config_file_walks_parents(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    cfg = repo / ".rl-dbs.yaml"
    cfg.write_text("env:\n  beta_threshold: 0.5\n", encoding="utf-8")

    found = find_config_file(nested, repo_root=repo)
    assert found == cfg.resolve()


def test_resolve_config_merges_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".rl-dbs.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "plant": {"dt_ms": 0.02, "backend": "python"},
                "env": {"beta_threshold": 0.4, "biomarker": {"band_hz": [12, 30]}},
                "defaults": {"seed": 7, "results_dir": "out"},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_config(config_path=path)
    assert resolved.plant.dt_ms == 0.02
    assert resolved.plant_backend == "python"
    assert resolved.env.beta_threshold == 0.4
    assert resolved.biomarker_band_hz == (12.0, 30.0)
    assert resolved.default_seed == 7
    assert resolved.results_dir == Path("out")


def test_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".rl-dbs.yaml"
    path.write_text("defaults:\n  seed: 1\n", encoding="utf-8")
    monkeypatch.setenv("RL_DBS_SEED", "99")

    resolved = resolve_config(config_path=path)
    assert resolved.default_seed == 99


def test_plant_backend_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".rl-dbs.yaml"
    path.write_text("plant:\n  backend: matlab\n", encoding="utf-8")
    monkeypatch.setenv("RL_DBS_PLANT_BACKEND", "python")

    resolved = resolve_config(config_path=path)
    assert resolved.plant_backend == "python"


def test_persist_plant_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_path = tmp_path / ".rl-dbs.yaml"

    path, resolved = persist_config_key("plant.backend", "python", config_path=write_path)
    assert path == write_path
    assert resolved.plant_backend == "python"

    payload = show_config(["plant.backend"], config_path=path)
    assert payload["plant.backend"] == "python"


def test_persist_and_show_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_path = tmp_path / ".rl-dbs.yaml"

    path, resolved = persist_config_key("env.beta_t", "0.42", config_path=write_path)
    assert path == write_path
    assert resolved.env.beta_threshold == 0.42

    payload = show_config(["env.beta_t"], config_path=path)
    assert payload["env.beta_threshold"] == 0.42


def test_preview_config_key_without_persist(tmp_path: Path) -> None:
    path = tmp_path / ".rl-dbs.yaml"
    path.write_text("env:\n  beta_threshold: 0.35\n", encoding="utf-8")

    resolved = preview_config_key("plant.dt", "0.02", config_path=path)
    assert resolved.plant.dt_ms == 0.02
    assert resolved.env.beta_threshold == 0.35
    # File on disk unchanged
    assert "dt_ms" not in path.read_text(encoding="utf-8")


def test_unknown_config_key_raises() -> None:
    with pytest.raises(KeyError, match="unknown config key"):
        show_config(["not.a.key"])
