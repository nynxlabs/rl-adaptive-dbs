"""User configuration: ``.rl-dbs.yaml`` discovery, merge, and persistence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from rl_adaptive_dbs.paths import find_repo_root
from envs.mehregan.config import MehreganEnvConfig
from envs.plant.config import PlantConfig

PlantBackendName = Literal["matlab", "python"]
DEFAULT_PLANT_BACKEND: PlantBackendName = "matlab"

CONFIG_FILENAMES: tuple[str, ...] = (".rl-dbs.yaml", ".rl-dbs.yml")

# Dot keys accepted by ``rl-dbs config show`` / ``config set`` → nested YAML path.
DOT_KEY_PATHS: dict[str, tuple[str, ...]] = {
    "plant.backend": ("plant", "backend"),
    "plant.dt": ("plant", "dt_ms"),
    "plant.pd": ("plant", "pd"),
    "plant.corstim": ("plant", "corstim"),
    "plant.neurons_per_region": ("plant", "neurons_per_region"),
    "env.dt_rl": ("env", "step_duration_s"),
    "env.beta_t": ("env", "beta_threshold"),
    "env.episode_steps": ("env", "max_episode_steps"),
    "env.reward_scale": ("env", "reward_scale"),
    "env.observation_scale": ("env", "observation_scale"),
    "env.state_length": ("env", "state_length"),
    "env.biomarker.band_hz": ("env", "biomarker", "band_hz"),
    "defaults.seed": ("defaults", "seed"),
    "defaults.results_dir": ("defaults", "results_dir"),
    "defaults.checkpoint_dir": ("defaults", "checkpoint_dir"),
}

DEFAULT_BIOMARKER_BAND_HZ: tuple[float, float] = (13.0, 35.0)


def _parse_plant_backend(raw: Any) -> PlantBackendName:
    if raw is None:
        return DEFAULT_PLANT_BACKEND
    value = str(raw).strip().lower()
    if value in ("matlab", "python"):
        return value  # type: ignore[return-value]
    msg = f"plant.backend must be 'matlab' or 'python', got {raw!r}"
    raise ValueError(msg)


@dataclass(frozen=True)
class ResolvedConfig:
    """Effective settings after defaults, file, and environment overlays."""

    plant: PlantConfig
    plant_backend: PlantBackendName
    env: MehreganEnvConfig
    biomarker_band_hz: tuple[float, float]
    default_seed: int
    results_dir: Path
    checkpoint_dir: Path | None
    config_path: Path | None


def find_config_file(
    start: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Walk from ``start`` (or cwd) up to ``repo_root`` for the first config file."""
    root = repo_root or find_repo_root(start)
    current = (start or Path.cwd()).resolve()
    seen: set[Path] = set()
    for candidate in (current, *current.parents):
        if candidate in seen:
            break
        seen.add(candidate)
        for name in CONFIG_FILENAMES:
            path = candidate / name
            if path.is_file():
                return path.resolve()
        if candidate == root:
            break
    return None


def resolve_config_path(explicit: Path | str | None = None) -> Path | None:
    """Explicit ``--config`` / ``RL_DBS_CONFIG``, else discovered file."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            msg = f"config file not found: {path}"
            raise FileNotFoundError(msg)
        return path
    env_path = os.environ.get("RL_DBS_CONFIG")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            msg = f"RL_DBS_CONFIG file not found: {path}"
            raise FileNotFoundError(msg)
        return path
    return find_config_file()


def load_yaml_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = data
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            msg = f"cannot set nested key {'.'.join(path)}: {key!r} is not a mapping"
            raise TypeError(msg)
        node = child
    node[path[-1]] = value


def _dataclass_overlay(
    cls: type[Any],
    section: dict[str, Any] | None,
    *,
    field_map: dict[str, str] | None = None,
) -> Any:
    if not section:
        return cls()
    valid = {item.name for item in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in section.items():
        target = (field_map or {}).get(key, key)
        if target in valid:
            kwargs[target] = value
    return replace(cls(), **kwargs)


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    if "," in stripped and not stripped.startswith("["):
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        if parts and all(_looks_numeric(part) for part in parts):
            return [_coerce_number(part) for part in parts]
    if _looks_numeric(stripped):
        return _coerce_number(stripped)
    return stripped


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _coerce_number(text: str) -> int | float:
    if "." in text or "e" in text.lower():
        return float(text)
    return int(text)


def _band_hz_tuple(raw: Any) -> tuple[float, float]:
    if raw is None:
        return DEFAULT_BIOMARKER_BAND_HZ
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    msg = f"env.biomarker.band_hz must be a two-element list, got {raw!r}"
    raise ValueError(msg)


def resolve_config(
    *,
    cwd: Path | None = None,
    config_path: Path | None = None,
    file_data: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Merge built-in defaults with YAML file and ``RL_DBS_*`` environment variables."""
    path = config_path if config_path is not None else resolve_config_path()
    data = _deep_merge(load_yaml_file(path), file_data or {})

    plant_section = data.get("plant") if isinstance(data.get("plant"), dict) else {}
    plant = _dataclass_overlay(PlantConfig, plant_section)
    plant_backend = _parse_plant_backend(plant_section.get("backend"))
    if "RL_DBS_PLANT_BACKEND" in os.environ:
        plant_backend = _parse_plant_backend(os.environ["RL_DBS_PLANT_BACKEND"])
    env = _dataclass_overlay(MehreganEnvConfig, data.get("env"))

    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    biomarker = data.get("env", {}).get("biomarker", {}) if isinstance(data.get("env"), dict) else {}
    band = _band_hz_tuple(biomarker.get("band_hz"))

    default_seed = int(defaults.get("seed", 42))
    if "RL_DBS_SEED" in os.environ:
        default_seed = int(os.environ["RL_DBS_SEED"])

    results_dir = Path(str(defaults.get("results_dir", "results")))
    if "RL_DBS_RESULTS_DIR" in os.environ:
        results_dir = Path(os.environ["RL_DBS_RESULTS_DIR"])

    checkpoint_raw = defaults.get("checkpoint_dir")
    checkpoint_dir = Path(str(checkpoint_raw)) if checkpoint_raw else None

    return ResolvedConfig(
        plant=plant,
        plant_backend=plant_backend,
        env=env,
        biomarker_band_hz=band,
        default_seed=default_seed,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        config_path=path,
    )


def config_show_payload(
    resolved: ResolvedConfig,
    keys: list[str] | None = None,
) -> dict[str, Any]:
    """Map resolved settings to ``config show`` labels."""
    all_keys = list(DOT_KEY_PATHS.keys())
    selected = keys if keys else all_keys
    out: dict[str, Any] = {}
    for key in selected:
        if key not in DOT_KEY_PATHS:
            msg = f"unknown config key {key!r}"
            raise KeyError(msg)
        if key == "plant.backend":
            out["plant.backend"] = resolved.plant_backend
        elif key == "plant.dt":
            out["plant.dt_ms"] = resolved.plant.dt_ms
        elif key == "plant.pd":
            out["plant.pd"] = resolved.plant.pd
        elif key == "plant.corstim":
            out["plant.corstim"] = resolved.plant.corstim
        elif key == "plant.neurons_per_region":
            out["plant.neurons_per_region"] = resolved.plant.neurons_per_region
        elif key == "env.dt_rl":
            out["env.step_duration_s"] = resolved.env.step_duration_s
        elif key == "env.beta_t":
            out["env.beta_threshold"] = resolved.env.beta_threshold
        elif key == "env.episode_steps":
            out["env.max_episode_steps"] = resolved.env.max_episode_steps
        elif key == "env.reward_scale":
            out["env.reward_scale"] = resolved.env.reward_scale
        elif key == "env.observation_scale":
            out["env.observation_scale"] = resolved.env.observation_scale
        elif key == "env.state_length":
            out["env.state_length"] = resolved.env.state_length
        elif key == "env.biomarker.band_hz":
            out["env.biomarker.band_hz"] = list(resolved.biomarker_band_hz)
        elif key == "defaults.seed":
            out["defaults.seed"] = resolved.default_seed
        elif key == "defaults.results_dir":
            out["defaults.results_dir"] = str(resolved.results_dir)
        elif key == "defaults.checkpoint_dir":
            out["defaults.checkpoint_dir"] = (
                str(resolved.checkpoint_dir) if resolved.checkpoint_dir else None
            )
    return out


def default_config_write_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    return root / CONFIG_FILENAMES[0]


def persist_config_key(
    key: str,
    value: str,
    *,
    config_path: Path | None = None,
) -> tuple[Path, ResolvedConfig]:
    """Set one dot-key in the user config file and return the updated path + resolved settings."""
    if key not in DOT_KEY_PATHS:
        msg = f"unknown config key {key!r}"
        raise KeyError(msg)
    path = config_path or resolve_config_path() or default_config_write_path()
    parsed = _parse_scalar(value)
    existing = load_yaml_file(path if path.is_file() else None)
    _set_nested(existing, DOT_KEY_PATHS[key], parsed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(existing, handle, sort_keys=False, default_flow_style=False)
    return path, resolve_config(config_path=path)


def preview_config_key(
    key: str,
    value: str,
    *,
    config_path: Path | None = None,
) -> ResolvedConfig:
    """Return resolved settings after applying one dot-key without writing a file."""
    if key not in DOT_KEY_PATHS:
        msg = f"unknown config key {key!r}"
        raise KeyError(msg)
    base_path = config_path or resolve_config_path()
    overlay: dict[str, Any] = {}
    _set_nested(overlay, DOT_KEY_PATHS[key], _parse_scalar(value))
    return resolve_config(config_path=base_path, file_data=overlay)
