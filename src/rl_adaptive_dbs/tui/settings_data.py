"""Persistent TUI preferences (poll interval, tail size, sparkline window)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rl_adaptive_dbs.tui.logs_data import DEFAULT_TAIL_LINES
from rl_adaptive_dbs.tui.training_data import MAX_SPARKLINE_EPISODES

DEFAULT_REFRESH_S = 1.0

SETTING_KEYS = (
    "refresh_s",
    "tail_lines",
    "sparkline_episodes",
    "color_enabled",
)

SETTING_BOUNDS: dict[str, tuple[float, float] | None] = {
    "refresh_s": (0.25, 60.0),
    "tail_lines": (50.0, 2000.0),
    "sparkline_episodes": (10.0, 200.0),
    "color_enabled": None,
}

SETTING_DESCRIPTIONS: dict[str, str] = {
    "refresh_s": "How often tabs reload data from disk.",
    "tail_lines": "Lines shown when tailing an open log file.",
    "sparkline_episodes": "Episode window for the Training return sparkline.",
    "color_enabled": "Use theme colors instead of monochrome (restart required).",
}


@dataclass(frozen=True)
class TuiSettings:
    """User-editable TUI preferences persisted under ``artifacts/``."""

    refresh_s: float = DEFAULT_REFRESH_S
    tail_lines: int = DEFAULT_TAIL_LINES
    sparkline_episodes: int = MAX_SPARKLINE_EPISODES
    color_enabled: bool = False

    @classmethod
    def from_mapping(cls, raw: object, *, defaults: TuiSettings | None = None) -> TuiSettings:
        base = defaults or cls()
        if not isinstance(raw, dict):
            return base
        merged = asdict(base)
        for key in SETTING_KEYS:
            if key not in raw:
                continue
            value = raw[key]
            if key == "color_enabled":
                if isinstance(value, bool):
                    merged[key] = value
                continue
            if key in {"tail_lines", "sparkline_episodes"}:
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    merged[key] = int(value)
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = float(value)
        return cls(**merged)


def settings_file(artifacts_dir: Path) -> Path:
    """Persistent settings co-located with training artifacts."""
    return artifacts_dir / ".tui-settings.json"


def load_settings(path: Path, *, defaults: TuiSettings | None = None) -> TuiSettings:
    if not path.is_file():
        return defaults or TuiSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults or TuiSettings()
    return TuiSettings.from_mapping(raw, defaults=defaults)


def save_settings(path: Path, settings: TuiSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: getattr(settings, key) for key in SETTING_KEYS}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clamp_setting(key: str, value: float | int | bool) -> float | int | bool:
    """Clamp numeric settings to documented bounds."""
    if key == "color_enabled":
        return bool(value)
    bounds = SETTING_BOUNDS.get(key)
    if bounds is None:
        return value
    low, high = bounds
    if key in {"tail_lines", "sparkline_episodes"}:
        return int(min(max(int(value), int(low)), int(high)))
    return float(min(max(float(value), low), high))


def update_setting(settings: TuiSettings, key: str, value: float | int | bool) -> TuiSettings:
    clamped = clamp_setting(key, value)
    return replace(settings, **{key: clamped})


def settings_table_rows(settings: TuiSettings) -> list[tuple[str, str, str]]:
    """Rows for the Settings tab table: (key, label, display value)."""
    return [
        ("refresh_s", "Poll interval (s)", f"{settings.refresh_s:g}"),
        ("tail_lines", "Log tail lines", str(settings.tail_lines)),
        ("sparkline_episodes", "Training sparkline episodes", str(settings.sparkline_episodes)),
        ("color_enabled", "Color", "on" if settings.color_enabled else "off"),
    ]


def settings_info_lines(
    *,
    results_dir: Path,
    artifacts_dir: Path,
    logs_dir: Path,
    settings_path: Path,
    bookmarks_path: Path,
) -> list[str]:
    return [
        f"results: {results_dir}",
        f"artifacts: {artifacts_dir}",
        f"logs: {logs_dir}",
        f"settings file: {settings_path}",
        f"bookmarks: {bookmarks_path}",
        "Paths come from CLI flags; other values persist in the settings file.",
        "Color changes apply after Ctrl+R restart.",
    ]


def settings_hints_line() -> str:
    return "↑↓ select  Enter edit  + / - increase / decrease  space toggle  Esc cancel edit"


def settings_status_line(settings: TuiSettings, *, selected_key: str | None = None) -> str:
    if selected_key is not None:
        label = next(
            (row[1] for row in settings_table_rows(settings) if row[0] == selected_key),
            selected_key,
        )
        description = SETTING_DESCRIPTIONS.get(selected_key, "")
        return f"{label}: {description}"
    return "Settings"


def parse_setting_input(key: str, text: str) -> float | int | bool | None:
    """Parse user input for a setting; return None on failure."""
    stripped = text.strip().lower()
    if key == "color_enabled":
        if stripped in {"1", "true", "on", "yes", "y"}:
            return True
        if stripped in {"0", "false", "off", "no", "n"}:
            return False
        return None
    try:
        if key in {"tail_lines", "sparkline_episodes"}:
            return int(stripped)
        return float(stripped)
    except ValueError:
        return None


def step_setting(settings: TuiSettings, key: str, delta: int) -> TuiSettings:
    """Increment or decrement a numeric setting by one step."""
    if key == "color_enabled":
        return replace(settings, color_enabled=not settings.color_enabled)
    current = getattr(settings, key)
    if key == "refresh_s":
        step = 0.25 if delta > 0 else -0.25
        return update_setting(settings, key, current + step)
    return update_setting(settings, key, current + delta)
