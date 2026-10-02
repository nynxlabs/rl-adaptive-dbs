"""rl-dbs panel: argument splitting, candidate output redirection, compare helpers."""

from __future__ import annotations

from rl_adaptive_dbs import panel_cli

FLAGS = {
    "--series",
    "--checkpoint",
    "--manifest",
    "--out",
    "--no-update-docs",
    "--abort-on-fail",
    "--export-notes",
    "--update-report",
}


def test_split_script_args_only_after_run() -> None:
    head, tail = panel_cli.split_script_args(
        ["panel", "run", "nguyen/4", "--candidate", "a", "--", "--episodes", "5"]
    )
    assert head == ["panel", "run", "nguyen/4", "--candidate", "a"]
    assert tail == ["--episodes", "5"]
    assert panel_cli.split_script_args(["panel", "compare", "nguyen/4"]) == (
        ["panel", "compare", "nguyen/4"],
        [],
    )


def test_candidate_outputs_redirected_and_promote_skipped() -> None:
    args = panel_cli.build_panel_args("nguyen/4", ["--episodes", "5"], candidate="c1", flags=FLAGS)
    out_dir = str(panel_cli.candidate_dir("nguyen/4", "c1"))
    for flag in ("--series", "--checkpoint", "--manifest", "--out"):
        assert args[args.index(flag) + 1].startswith(out_dir)
    assert "--no-update-docs" in args and "--abort-on-fail" in args
    assert "--export-notes" not in args


def test_explicit_output_flag_wins_and_promote_run_publishes() -> None:
    args = panel_cli.build_panel_args(
        "nguyen/4", ["--series", "/tmp/s.json"], candidate="c1", flags=FLAGS, abort_on_fail=False
    )
    assert args.count("--series") == 1 and "--abort-on-fail" not in args
    promote = panel_cli.build_panel_args("nguyen/4", [], candidate=None, flags=FLAGS)
    assert "--export-notes" in promote and "--update-report" in promote
    assert "--no-update-docs" not in promote


def test_config_diff() -> None:
    assert panel_cli.config_diff({"a": 1, "b": 2}, {"a": 1, "b": 3}) == {"b": 3}
    assert panel_cli.config_diff(None, {"a": 1}) == {}
