"""Benchmark suite YAML loading and run expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.suite import (
    expand_planned_runs,
    load_suite,
    parse_controller_filter,
    resolve_suite_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_smoke_suite() -> None:
    suite = load_suite("mehregan_eval_smoke", repo_root=REPO_ROOT)
    assert suite.name == "mehregan_eval_smoke"
    assert suite.protocol == "mehregan"
    assert suite.eval_steps == 2
    assert suite.seeds == (0, 1)
    assert len(suite.controllers) == 2


def test_expand_planned_runs() -> None:
    suite = load_suite("mehregan_eval_smoke", repo_root=REPO_ROOT)
    planned = expand_planned_runs(suite, repo_root=REPO_ROOT)
    assert len(planned) == 4  # 2 baselines × 2 seeds
    assert all(item.controller == "baseline" for item in planned)
    assert all(item.checkpoint is None for item in planned)


def test_controller_filter() -> None:
    suite = load_suite("mehregan_eval_smoke", repo_root=REPO_ROOT)
    filt = parse_controller_filter("baseline:cdbs-130hz")
    planned = expand_planned_runs(suite, controller_filter=filt, repo_root=REPO_ROOT)
    assert len(planned) == 2
    assert all(item.variant == "cdbs-130hz" for item in planned)


def test_resolve_suite_path_by_name() -> None:
    path = resolve_suite_path("mehregan_eval", repo_root=REPO_ROOT)
    assert path.name == "mehregan_eval.yaml"


def test_resolve_ptq_checkpoint(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "artifacts" / "ddpg"
    ckpt_dir.mkdir(parents=True)
    fp_ckpt = ckpt_dir / "paper_train0.pt"
    fp_ckpt.write_bytes(b"placeholder")
    suite = load_suite("mehregan_eval", repo_root=REPO_ROOT)
    entry = next(e for e in suite.controllers if e.variant == "ptq-int8")
    from benchmarks.suite import _resolve_checkpoint

    path = _resolve_checkpoint(entry, suite, repo_root=tmp_path)
    assert path == fp_ckpt.resolve()


def test_parse_controller_filter_invalid() -> None:
    with pytest.raises(ValueError, match="controller:variant"):
        parse_controller_filter("ddpg")
