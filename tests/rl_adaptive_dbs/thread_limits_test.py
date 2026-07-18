"""Tests for rl_adaptive_dbs.thread_limits."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rl_adaptive_dbs.thread_limits import (
    DEFAULT_PLANT_HEAVY_MAX_THREADS,
    DEFAULT_TRAIN_MAX_THREADS,
    ENV_VAR,
    THREAD_ENV_KEYS,
    apply_max_threads,
    bootstrap_thread_limits,
    default_max_threads_for_cli,
    first_cli_subcommand,
    parse_max_threads_from_argv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_apply_max_threads_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in THREAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    apply_max_threads(3)
    for key in THREAD_ENV_KEYS:
        assert os.environ[key] == "3"


def test_parse_max_threads_from_argv() -> None:
    assert parse_max_threads_from_argv(["train", "--max-threads", "4", "--dry-run"]) == 4
    assert parse_max_threads_from_argv(["train", "--dry-run"]) is None


def test_bootstrap_prefers_flag_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in THREAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ENV_VAR, "8")
    cap = bootstrap_thread_limits(["--max-threads", "2"])
    assert cap == 2
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_bootstrap_reads_env_when_no_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in THREAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ENV_VAR, "5")
    cap = bootstrap_thread_limits([])
    assert cap == 5
    assert os.environ["NUMBA_NUM_THREADS"] == "5"


def test_bootstrap_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in THREAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(ENV_VAR, raising=False)
    cap = bootstrap_thread_limits([], default=DEFAULT_PLANT_HEAVY_MAX_THREADS)
    assert cap == DEFAULT_PLANT_HEAVY_MAX_THREADS
    assert os.environ["OMP_NUM_THREADS"] == str(DEFAULT_PLANT_HEAVY_MAX_THREADS)


def test_first_cli_subcommand_skips_globals() -> None:
    assert first_cli_subcommand(["--verbose", "--seed", "0", "train", "--dry-run"]) == "train"
    assert first_cli_subcommand(["--max-threads", "8", "benchmark", "--suite", "x"]) == "benchmark"
    assert first_cli_subcommand(["info", "controllers"]) == "info"


def test_default_max_threads_for_cli() -> None:
    assert default_max_threads_for_cli(["train", "--dry-run"]) == DEFAULT_TRAIN_MAX_THREADS
    assert default_max_threads_for_cli(["info", "controllers"]) is None


def test_run_module_applies_default_cap_without_flag() -> None:
    script = REPO_ROOT / "tests" / "fixtures" / "thread_limit_probe.py"
    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    for key in THREAD_ENV_KEYS:
        env.pop(key, None)
    code = subprocess.run(
        [sys.executable, "-m", "rl_adaptive_dbs.run", str(script)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert code.returncode == 0, code.stderr
    assert f"OMP_NUM_THREADS={DEFAULT_PLANT_HEAVY_MAX_THREADS}" in code.stdout
    assert f"thread pools capped at {DEFAULT_PLANT_HEAVY_MAX_THREADS}" in code.stdout


def test_run_module_sets_env_before_script(capsys) -> None:
    script = REPO_ROOT / "tests" / "fixtures" / "thread_limit_probe.py"
    code = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_adaptive_dbs.run",
            "--max-threads",
            "3",
            str(script),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert code.returncode == 0, code.stderr
    assert "OMP_NUM_THREADS=3" in code.stdout
    assert "thread pools capped at 3" in code.stdout


def test_run_module_rejects_missing_script() -> None:
    code = subprocess.run(
        [sys.executable, "-m", "rl_adaptive_dbs.run", "scripts/no_such_script.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert code.returncode != 0
    assert "script not found" in code.stderr
