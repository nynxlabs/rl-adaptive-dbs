"""``rl-dbs config show`` — resolved settings from defaults and ``.rl-dbs.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_adaptive_dbs.user_config import config_show_payload, resolve_config


def show_config(
    keys: list[str] | None = None,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Return resolved config values for requested keys (all if ``keys`` is empty)."""
    resolved = resolve_config(config_path=config_path)
    return config_show_payload(resolved, keys)


def format_config_text(payload: dict[str, Any], *, config_path: Path | None = None) -> str:
    lines = [f"{key}: {value}" for key, value in payload.items()]
    if config_path is not None:
        lines.append(f"config_file: {config_path}")
    elif "config_file" not in payload:
        resolved = resolve_config()
        if resolved.config_path is not None:
            lines.append(f"config_file: {resolved.config_path}")
    return "\n".join(lines)
