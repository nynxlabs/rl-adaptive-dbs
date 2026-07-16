"""Launch follow mode helpers (no terminal required)."""

from __future__ import annotations

from rl_adaptive_dbs.tui.launch_follow import (
    LAUNCH_FOLLOW_ASK,
    LAUNCH_FOLLOW_LOGS,
    LAUNCH_FOLLOW_NONE,
    LAUNCH_FOLLOW_TERMINAL,
    cycle_dialog_follow,
    cycle_launch_follow_setting,
    initial_dialog_follow,
    normalize_launch_follow,
)
from rl_adaptive_dbs.tui.settings_data import TuiSettings, step_setting, update_setting


def test_normalize_launch_follow_defaults_unknown() -> None:
    assert normalize_launch_follow("bogus") == LAUNCH_FOLLOW_LOGS
    assert normalize_launch_follow(LAUNCH_FOLLOW_TERMINAL) == LAUNCH_FOLLOW_TERMINAL


def test_initial_dialog_follow_respects_setting() -> None:
    assert initial_dialog_follow(LAUNCH_FOLLOW_ASK) == LAUNCH_FOLLOW_LOGS
    assert initial_dialog_follow(LAUNCH_FOLLOW_NONE) == LAUNCH_FOLLOW_NONE
    assert initial_dialog_follow(LAUNCH_FOLLOW_TERMINAL) == LAUNCH_FOLLOW_TERMINAL


def test_cycle_dialog_follow_skips_ask() -> None:
    assert cycle_dialog_follow(LAUNCH_FOLLOW_LOGS) == LAUNCH_FOLLOW_TERMINAL
    assert cycle_dialog_follow(LAUNCH_FOLLOW_TERMINAL) == LAUNCH_FOLLOW_NONE
    assert cycle_dialog_follow(LAUNCH_FOLLOW_NONE) == LAUNCH_FOLLOW_LOGS


def test_settings_launch_follow_step_and_update() -> None:
    settings = TuiSettings(launch_follow=LAUNCH_FOLLOW_LOGS)
    stepped = step_setting(settings, "launch_follow", 1)
    assert stepped.launch_follow == LAUNCH_FOLLOW_TERMINAL
    updated = update_setting(settings, "launch_follow", LAUNCH_FOLLOW_ASK)
    assert updated.launch_follow == LAUNCH_FOLLOW_ASK
