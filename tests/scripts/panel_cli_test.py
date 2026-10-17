"""rl-dbs panel: argument splitting, candidate output redirection, compare helpers."""

from __future__ import annotations

from pathlib import Path

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


def test_launch_argv_adds_a_log_window_for_pounce(monkeypatch) -> None:
    monkeypatch.setattr(panel_cli.shutil, "which", lambda name: "/usr/bin/" + name if name == "tmux" else None)
    argv = panel_cli._launch_argv("nguyen-4-main", ["python", "x.py"], Path("/repo/logs/nguyen-4-main.log"))
    assert argv[:9] == ["tmux", "-L", "nguyen-4-main", "new-session", "-d", "-s", "nguyen-4-main", "-n", "run"]
    assert "setsid nohup nice -n 10 python x.py < /dev/null" in argv[9]
    window = argv[argv.index("new-window") :]
    assert window[:5] == ["new-window", "-t", "=nguyen-4-main:", "-n", "log"]
    assert "tail -n +1 -F --pid=" in window[5] and "/repo/logs/nguyen-4-main.log" in window[5]
    assert ["@pounce_readonly", "on"] == argv[argv.index("@pounce_readonly") : argv.index("@pounce_readonly") + 2]
    assert argv[-1] == "rl-dbs · nguyen-4-main"
    # Without a log there is no log window.
    assert "new-window" not in panel_cli._launch_argv("s", ["python", "x.py"])


def test_candidate_redirects_a_cache_dir_panel() -> None:
    flags = {"--cache-dir", "--no-promote", "--parallel-series", "--set"}
    args = panel_cli.build_panel_args("ravivarapu/4a", ["--set", "n_obs=5"], candidate="c1", flags=flags)
    assert args[args.index("--cache-dir") + 1] == str(panel_cli.candidate_dir("ravivarapu/4a", "c1"))
    assert "--no-promote" in args
