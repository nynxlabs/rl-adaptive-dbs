"""Settings tab data layer tests (no terminal required)."""

from __future__ import annotations

from pathlib import Path

from rl_adaptive_dbs.tui.settings_data import (
    TuiSettings,
    load_settings,
    parse_setting_input,
    save_settings,
    settings_file,
    settings_status_line,
    step_setting,
    update_setting,
)


def test_settings_round_trip(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    path = settings_file(artifacts)
    settings = TuiSettings(
        refresh_s=2.5,
        tail_lines=400,
        sparkline_episodes=80,
        color_enabled=True,
    )

    save_settings(path, settings)
    loaded = load_settings(path)
    assert loaded == settings


def test_load_settings_uses_defaults_for_missing_file(tmp_path: Path) -> None:
    defaults = TuiSettings(refresh_s=3.0, color_enabled=True)
    loaded = load_settings(tmp_path / "missing.json", defaults=defaults)
    assert loaded == defaults


def test_update_setting_clamps_values() -> None:
    settings = TuiSettings()
    updated = update_setting(settings, "tail_lines", 10_000)
    assert updated.tail_lines == 2000
    updated = update_setting(settings, "refresh_s", 0.01)
    assert updated.refresh_s == 0.25


def test_step_setting_adjusts_numeric_fields() -> None:
    settings = TuiSettings(refresh_s=1.0, tail_lines=200)
    stepped = step_setting(settings, "refresh_s", 1)
    assert stepped.refresh_s == 1.25
    stepped = step_setting(stepped, "tail_lines", -1)
    assert stepped.tail_lines == 199


def test_parse_setting_input_accepts_color_aliases() -> None:
    assert parse_setting_input("color_enabled", "on") is True
    assert parse_setting_input("color_enabled", "off") is False
    assert parse_setting_input("color_enabled", "maybe") is None


def test_step_setting_toggles_color() -> None:
    settings = TuiSettings(color_enabled=False)
    toggled = step_setting(settings, "color_enabled", 1)
    assert toggled.color_enabled is True


def test_settings_status_line_shows_description() -> None:
    line = settings_status_line(TuiSettings(), selected_key="tail_lines")
    assert line.startswith("Log tail lines:")
    assert "tailing" in line.lower()
