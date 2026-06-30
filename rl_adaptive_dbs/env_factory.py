"""Construct Mehregan environments from resolved user configuration."""

from __future__ import annotations

from pathlib import Path

from envs.mehregan.env import MehreganEnv
from envs.plant.matlab_backend import MatlabPlant
from rl_adaptive_dbs.user_config import ResolvedConfig, resolve_config


def build_mehregan_env(
    *,
    resolved: ResolvedConfig | None = None,
    config_path: str | Path | None = None,
) -> MehreganEnv:
    """Create ``MehreganEnv`` with plant/env settings from ``.rl-dbs.yaml`` when present."""
    cfg = resolved
    if cfg is None:
        explicit = Path(config_path).expanduser().resolve() if config_path else None
        cfg = resolve_config(config_path=explicit)
    plant = MatlabPlant(config=cfg.plant)
    return MehreganEnv(plant=plant, config=cfg.env)
