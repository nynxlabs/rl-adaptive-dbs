"""TUI reload / dev-mode helpers."""

from __future__ import annotations

import time
from pathlib import Path

from rl_adaptive_dbs.tui.reload import (
    RESTART_EXIT_CODE,
    child_argv,
    py_files_changed,
    snapshot_py_files,
    tui_package_root,
)


def test_restart_exit_code_is_distinct() -> None:
    assert RESTART_EXIT_CODE not in (0, 1, 130)


def test_child_argv_drops_dev_flag() -> None:
    argv = child_argv(["--dev", "--results-dir", "results/"])
    assert "--dev" not in argv
    assert "--results-dir" in argv
    assert "results/" in argv
    assert argv[0].endswith("rl-dbs-tui") or argv[:2] == [__import__("sys").executable, "-m"]


def test_tui_package_is_runnable_as_module() -> None:
    assert (tui_package_root() / "__main__.py").is_file()


def test_tui_package_root_contains_app_module() -> None:
    root = tui_package_root()
    assert (root / "app.py").is_file()


def test_py_files_changed_detects_edit(tmp_path: Path) -> None:
    module = tmp_path / "pane.py"
    module.write_text("TITLE = 'v1'\n", encoding="utf-8")
    before = snapshot_py_files(tmp_path)
    assert not py_files_changed(before, tmp_path)

    time.sleep(0.02)
    module.write_text("TITLE = 'v2'\n", encoding="utf-8")
    assert py_files_changed(before, tmp_path)


def test_main_rejects_dev_with_ascii(capsys) -> None:
    from rl_adaptive_dbs.tui import main as tui_main

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark_results"
    code = tui_main(["--dev", "--ascii", "--results-dir", str(fixtures)])
    assert code == 0
    assert "mehregan_eval_smoke" in capsys.readouterr().out
