"""Training tab data layer tests (no terminal required)."""

from __future__ import annotations

from pathlib import Path

from rl_adaptive_dbs.tui.training_data import (
    discover_training_runs,
    load_training_run,
    parse_training_log,
    return_sparkline_data,
    select_training_run,
    training_status_line,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "training_artifacts"


def test_discover_training_runs() -> None:
    runs = discover_training_runs(FIXTURES)
    assert len(runs) >= 3
    ids = {run.run_id for run in runs}
    assert "ddpg/paper/train_log.jsonl" in ids
    assert "ddpg/pattern_train_smoke.json" in ids
    assert "ddpg/train_paper_seed0.log" in ids


def test_parse_jsonl_run() -> None:
    run = load_training_run(FIXTURES / "ddpg/paper/train_log.jsonl", FIXTURES)
    assert run is not None
    assert run.controller == "ddpg"
    assert run.variant == "paper"
    assert run.current_episode == 3
    assert run.planned_episodes == 10
    assert run.last_return == -5.4


def test_parse_json_array_run() -> None:
    episodes, total, error = parse_training_log(FIXTURES / "ddpg/pattern_train_smoke.json")
    assert error is None
    assert total is None
    assert len(episodes) == 2
    assert episodes[0].return_value == -176.49


def test_parse_text_log() -> None:
    episodes, total, error = parse_training_log(FIXTURES / "ddpg/train_paper_seed0.log")
    assert error is None
    assert total == 10
    assert len(episodes) == 3
    assert episodes[-1].return_value == -18.40


def test_sparkline_and_status() -> None:
    runs = discover_training_runs(FIXTURES)
    run = select_training_run(runs, "ddpg/paper/train_log.jsonl")
    assert run is not None
    spark = return_sparkline_data(run)
    assert spark == [-12.5, -8.1, -5.4]
    line = training_status_line(run, runs)
    assert "ddpg/paper" in line
    assert "3/10" in line
