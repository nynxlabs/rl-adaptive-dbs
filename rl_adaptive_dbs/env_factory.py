"""Construct Mehregan environments from resolved user configuration."""

from __future__ import annotations

from pathlib import Path

from envs.mehregan.env import MehreganEnv
from envs.plant.matlab_backend import MatlabPlant
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import (
    PlantBackendName,
    ResolvedConfig,
    resolve_config,
)


def build_mehregan_env(
    *,
    resolved: ResolvedConfig | None = None,
    config_path: str | Path | None = None,
    plant_backend: PlantBackendName | None = None,
) -> MehreganEnv:
    """Create ``MehreganEnv`` with plant/env settings from ``.rl-dbs.yaml`` when present."""
    cfg = resolved
    if cfg is None:
        explicit = Path(config_path).expanduser().resolve() if config_path else None
        cfg = resolve_config(config_path=explicit)
    backend = plant_backend if plant_backend is not None else cfg.plant_backend
    if backend == "python":
        plant = PythonPlant(config=cfg.plant)
    elif backend == "matlab":
        plant = MatlabPlant(config=cfg.plant)
    else:
        msg = f"unknown plant_backend: {backend!r}"
        raise ValueError(msg)
    return MehreganEnv(plant=plant, config=cfg.env)
