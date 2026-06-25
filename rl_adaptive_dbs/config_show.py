"""``rl-dbs config show`` — read defaults from env/plant specs."""

from __future__ import annotations

from typing import Any

from envs.mehregan.config import MehreganEnvConfig
from envs.plant.config import PlantConfig

CONFIG_KEYS: dict[str, tuple[str, Any]] = {
    "plant.dt": ("plant.dt_ms", lambda: PlantConfig().dt_ms),
    "plant.pd": ("plant.pd", lambda: PlantConfig().pd),
    "env.dt_rl": ("env.step_duration_s", lambda: MehreganEnvConfig().step_duration_s),
    "env.beta_t": ("env.beta_threshold", lambda: MehreganEnvConfig().beta_threshold),
    "env.episode_steps": ("env.max_episode_steps", lambda: MehreganEnvConfig().max_episode_steps),
    "env.reward_scale": ("env.reward_scale", lambda: MehreganEnvConfig().reward_scale),
    "env.observation_scale": ("env.observation_scale", lambda: MehreganEnvConfig().observation_scale),
}


def show_config(keys: list[str] | None = None) -> dict[str, Any]:
    """Return resolved config values for requested keys (all if ``keys`` is empty)."""
    selected = keys if keys else list(CONFIG_KEYS.keys())
    out: dict[str, Any] = {}
    for key in selected:
        if key not in CONFIG_KEYS:
            msg = f"unknown config key {key!r}"
            raise KeyError(msg)
        label, getter = CONFIG_KEYS[key]
        out[label] = getter()
    if "env.biomarker.band_hz" in (keys or []) or not keys:
        out["env.biomarker.band_hz"] = [13, 35]
    return out


def format_config_text(payload: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in payload.items())
