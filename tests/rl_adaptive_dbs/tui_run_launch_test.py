"""Detached launch helper tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rl_adaptive_dbs.run_log_meta import RUN_EXIT_PREFIX, RUN_META_PREFIX
from rl_adaptive_dbs.tui.run_launch import command_text, default_log_path, launch_detached


def test_command_text_quotes_spaces() -> None:
    assert command_text(["echo", "hello world"]) == "echo 'hello world'"


def test_default_log_path_under_artifacts(tmp_path: Path) -> None:
    path = default_log_path(tmp_path / "artifacts", "cli/info")
    assert path.parent == tmp_path / "artifacts" / "tui-runs"
    assert "cli_info" in path.name


def test_launch_detached_writes_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    artifacts = repo / "artifacts"
    log_path = artifacts / "tui-runs" / "probe.log"

    if not Path("/bin/sh").is_file():
        return

    result = launch_detached(
        ["/bin/sh", "-c", "echo launched-by-tui"],
        repo_root=repo,
        artifacts_dir=artifacts,
        recipe_id="test/probe",
        log_path=log_path,
        bookmark=False,
    )
    assert result.pid > 0

    deadline = time.monotonic() + 10.0
    text = ""
    while time.monotonic() < deadline:
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8")
            if RUN_EXIT_PREFIX in text:
                break
        time.sleep(0.1)

    assert RUN_META_PREFIX in text
    assert "launched-by-tui" in text
    assert RUN_EXIT_PREFIX in text
    footer = [line for line in text.splitlines() if line.startswith(RUN_EXIT_PREFIX)][-1]
    payload = json.loads(footer[len(RUN_EXIT_PREFIX) :])
    assert payload["exit_code"] == 0
