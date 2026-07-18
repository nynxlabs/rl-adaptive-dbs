"""Tests for scripts.lib.train_runtime_guard."""

from __future__ import annotations

import os

import pytest

from rl_adaptive_dbs.thread_limits import DEFAULT_TRAIN_MAX_THREADS, ENV_VAR, THREAD_ENV_KEYS

from scripts.lib.train_runtime_guard import run_main


def test_train_runtime_guard_applies_default_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in THREAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(ENV_VAR, raising=False)

    seen: dict[str, str] = {}

    def main() -> int:
        seen["omp"] = os.environ["OMP_NUM_THREADS"]
        return 0

    code = run_main(main, label="test-train-cap")
    assert code == 0
    assert seen["omp"] == str(DEFAULT_TRAIN_MAX_THREADS)
