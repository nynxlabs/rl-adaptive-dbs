"""Tests for scripts/matlab/fetch_kumaravelu.py (offline; no download, no MATLAB)."""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "matlab"))

import fetch_kumaravelu as fk  # noqa: E402

from envs.plant.matlab_backend import model_dir  # noqa: E402

BASE = b"a\r\nb\r\nc\r\nd\r\n"
RESULT = b"x\r\na\r\nc\r\nd\r\ny\r\nz\r\n"


def test_patch_round_trip_keeps_crlf() -> None:
    patch = fk.make_patch("f.m", BASE, RESULT)
    assert "\r" not in patch
    assert fk.apply_patch(BASE, patch) == RESULT


def test_patch_rejects_changed_base() -> None:
    patch = fk.make_patch("f.m", BASE, RESULT)
    with pytest.raises(fk.FetchError):
        fk.apply_patch(b"a\r\nB\r\nc\r\nd\r\n", patch)


@pytest.mark.plant
def test_packaged_patches_cover_every_output() -> None:
    patch_dir = fk.default_patch_dir()
    assert patch_dir is not None
    manifest = fk._manifest(patch_dir)
    assert set(manifest) == set(fk.OUTPUTS)
    for name in fk.OUTPUTS:
        patch = fk._patch_path(patch_dir, name)
        if patch.exists():
            assert patch.read_text(encoding="utf-8").startswith(f"--- a/{name}\n")


def test_without_patches_outputs_are_the_upstream_files() -> None:
    upstream = {name: f"{name}\r\n".encode() for name in fk.UPSTREAM_SHA256}
    assert fk.build(upstream, None) == upstream
    assert fk.expected_sha256(None) == fk.UPSTREAM_SHA256


def test_check_without_patches_compares_against_upstream(tmp_path: Path) -> None:
    problems = fk.check(tmp_path, None)
    assert sorted(problems) == sorted(f"missing {tmp_path / name}" for name in fk.UPSTREAM_SHA256)


def test_model_dir_names_fetch_command_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RL_ADAPTIVE_DBS_MATLAB_MODEL", str(tmp_path))
    assert model_dir(require=False) == tmp_path.resolve()
    with pytest.raises(FileNotFoundError, match="fetch_kumaravelu.py"):
        model_dir()


def test_model_dir_defaults_to_repo_reference_material(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RL_ADAPTIVE_DBS_MATLAB_MODEL", raising=False)
    monkeypatch.delenv("RL_ADAPTIVE_DBS_ROOT", raising=False)
    assert model_dir(require=False) == _ROOT.resolve() / "reference-material" / "KumaraveluEtAl2016"
