"""Launch output follow modes for the Run tab."""

from __future__ import annotations

LAUNCH_FOLLOW_LOGS = "logs"
LAUNCH_FOLLOW_TERMINAL = "terminal"
LAUNCH_FOLLOW_NONE = "none"
LAUNCH_FOLLOW_ASK = "ask"

LAUNCH_FOLLOW_MODES = (
    LAUNCH_FOLLOW_LOGS,
    LAUNCH_FOLLOW_TERMINAL,
    LAUNCH_FOLLOW_NONE,
    LAUNCH_FOLLOW_ASK,
)

# Modes selectable in the launch confirmation dialog (``f`` to cycle).
LAUNCH_FOLLOW_DIALOG_MODES = (
    LAUNCH_FOLLOW_LOGS,
    LAUNCH_FOLLOW_TERMINAL,
    LAUNCH_FOLLOW_NONE,
)

LAUNCH_FOLLOW_LABELS: dict[str, str] = {
    LAUNCH_FOLLOW_LOGS: "Logs tab",
    LAUNCH_FOLLOW_TERMINAL: "Terminal (tmux split)",
    LAUNCH_FOLLOW_NONE: "Don't follow",
    LAUNCH_FOLLOW_ASK: "Ask each launch",
}


def normalize_launch_follow(value: object, *, default: str = LAUNCH_FOLLOW_LOGS) -> str:
    if isinstance(value, str) and value in LAUNCH_FOLLOW_MODES:
        return value
    return default


def launch_follow_label(mode: str) -> str:
    return LAUNCH_FOLLOW_LABELS.get(mode, mode)


def initial_dialog_follow(setting: str) -> str:
    """Default follow choice shown in the launch confirmation dialog."""
    if setting == LAUNCH_FOLLOW_ASK:
        return LAUNCH_FOLLOW_LOGS
    if setting in LAUNCH_FOLLOW_DIALOG_MODES:
        return setting
    return LAUNCH_FOLLOW_LOGS


def cycle_dialog_follow(current: str, delta: int = 1) -> str:
    modes = LAUNCH_FOLLOW_DIALOG_MODES
    if current not in modes:
        current = LAUNCH_FOLLOW_LOGS
    index = modes.index(current)
    return modes[(index + delta) % len(modes)]


def cycle_launch_follow_setting(current: str, delta: int = 1) -> str:
    if current not in LAUNCH_FOLLOW_MODES:
        current = LAUNCH_FOLLOW_LOGS
    index = LAUNCH_FOLLOW_MODES.index(current)
    return LAUNCH_FOLLOW_MODES[(index + delta) % len(LAUNCH_FOLLOW_MODES)]
