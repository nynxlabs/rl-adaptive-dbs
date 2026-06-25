"""``rl-dbs`` CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from rl_adaptive_dbs.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    code = main(
        [
            "-v",
            "benchmark",
            "--suite-name",
            "mehregan_eval_smoke",
            "--results-dir",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )
    assert code == 0
    manifest = tmp_path / "results" / "mehregan_eval_smoke" / "manifest.json"
    assert manifest.is_file()
