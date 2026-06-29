"""Expanded ``rl-dbs`` CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from rl_adaptive_dbs.cli import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark_results"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_info_controllers() -> None:
    code = main(["info", "controllers"])
    assert code == 0


def test_info_json() -> None:
    code = main(["info", "--json"])
    assert code == 0


def test_config_show() -> None:
    code = main(["config", "show", "env.dt_rl", "env.beta_t"])
    assert code == 0


def test_summary_fixture(capsys) -> None:
    code = main(
        [
            "summary",
            "--results-dir",
            str(FIXTURES),
            "--suite-name",
            "mehregan_eval_smoke",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "mehregan_eval_smoke" in out
    assert "ddpg" in out


def test_train_dry_run() -> None:
    code = main(["train", "--controller", "ddpg", "--variant", "paper", "--dry-run"])
    assert code == 0


def test_train_dry_run_checkpoint_path(capsys) -> None:
    code = main(
        [
            "-v",
            "train",
            "--controller",
            "ddpg",
            "--variant",
            "paper",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    assert code == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["checkpoint"] == "artifacts/ddpg/paper_train0.pt"


def test_eval_baseline_dry_run_skipped() -> None:
    """Baseline eval needs env; smoke test uses benchmark runner tests instead."""
    code = main(["info", "variants", "--controller", "baseline"])
    assert code == 0


def test_tui_ascii_mode(capsys) -> None:
    from rl_adaptive_dbs.tui import main as tui_main

    code = tui_main(["--ascii", "--results-dir", str(FIXTURES)])
    assert code == 0
    assert "mehregan_eval_smoke" in capsys.readouterr().out
