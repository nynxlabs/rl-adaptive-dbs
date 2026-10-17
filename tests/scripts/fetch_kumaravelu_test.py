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


def test_committed_patches_cover_every_output() -> None:
    manifest = fk._manifest()
    assert set(manifest) == set(fk.OUTPUTS)
    for name in fk.OUTPUTS:
        patch = fk._patch_path(name)
        if patch.exists():
            assert patch.read_text(encoding="utf-8").startswith(f"--- a/{name}\n")


def test_model_dir_names_fetch_command_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RL_ADAPTIVE_DBS_MATLAB_MODEL", str(tmp_path))
    assert model_dir(require=False) == tmp_path.resolve()
    with pytest.raises(FileNotFoundError, match="fetch_kumaravelu.py"):
        model_dir()
