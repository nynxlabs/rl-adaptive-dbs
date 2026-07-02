"""Construct Mehregan environments from resolved user configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from envs.mehregan.env import MehreganEnv
from envs.plant.matlab_backend import MatlabPlant
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import ResolvedConfig, resolve_config

PlantBackendName = Literal["matlab", "python"]


def build_mehregan_env(
    *,
    resolved: ResolvedConfig | None = None,
    config_path: str | Path | None = None,
    plant_backend: PlantBackendName = "matlab",
) -> MehreganEnv:
    """Create ``MehreganEnv`` with plant/env settings from ``.rl-dbs.yaml`` when present."""
    cfg = resolved
    if cfg is None:
        explicit = Path(config_path).expanduser().resolve() if config_path else None
        cfg = resolve_config(config_path=explicit)
    if plant_backend == "python":
        plant = PythonPlant(config=cfg.plant)
    elif plant_backend == "matlab":
        plant = MatlabPlant(config=cfg.plant)
    else:
        msg = f"unknown plant_backend: {plant_backend!r}"
        raise ValueError(msg)
    return MehreganEnv(plant=plant, config=cfg.env)
