"""Plant configuration (Kumaravelu et al., 2016 reference defaults)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlantConfig:
    """Non-Gym plant settings per docs/plant.md."""

    dt_ms: float = 0.01
    pd: int = 1
    corstim: int = 0
    neurons_per_region: int = 10
