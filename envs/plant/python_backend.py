"""Native Python Kumaravelu CBGT plant (docs/development/native-plant-port.md)."""

from __future__ import annotations

from envs.plant.config import PlantConfig
from envs.plant.dbs import DbsSpec, create_dbs_current
from envs.plant.matlab_backend import IntegrateResult

import numpy as np


class PythonPlant:
    """NumPy port of Kumaravelu et al. (2016) — drop-in ``PlantBackend``."""

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.config = config or PlantConfig()
        self._rng: np.random.Generator | None = None
        self._seed: int | None = None
        self._iteration = 0

    @property
    def config(self) -> PlantConfig:
        return self._config

    @config.setter
    def config(self, value: PlantConfig) -> None:
        self._config = value

    def close(self) -> None:
        return None

    def __enter__(self) -> PythonPlant:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self, seed: int | None = None) -> PythonPlant:
        self._seed = seed
        self._iteration = 0
        if seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(seed)
        return self

    def integrate(
        self,
        duration_s: float,
        dbs_spec: DbsSpec | None = None,
        *,
        record_spikes: bool = True,
    ) -> IntegrateResult:
        if duration_s <= 0:
            msg = "duration_s must be positive"
            raise ValueError(msg)

        spec = dbs_spec if dbs_spec is not None else DbsSpec.none()
        tmax_ms = duration_s * 1000.0
        self._iteration += 1

        if self._rng is None:
            self.reset(seed=self._seed)

        _ = create_dbs_current(
            spec.frequency_hz,
            tmax_ms=tmax_ms,
            dt_ms=self.config.dt_ms,
        )

        msg = (
            "PythonPlant network integrator not yet implemented "
            "(Phase B — see docs/development/native-plant-port.md)"
        )
        raise NotImplementedError(msg)
