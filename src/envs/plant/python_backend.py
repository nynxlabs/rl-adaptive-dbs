"""Native Python Kumaravelu CBGT plant (docs/development/native-plant-port.md)."""

from __future__ import annotations

from envs.plant.config import PlantConfig
from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from envs.plant.network.matlab_rng import load_cached_init_draws

import numpy as np


class PythonPlant:
    """NumPy port of Kumaravelu et al. (2016) — drop-in ``PlantBackend``."""

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.config = config or PlantConfig()
        self._rng: np.random.Generator | None = None
        self._seed: int | None = None
        self._init_draws: NetworkInitDraws | None = None
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
        self._init_draws = load_cached_init_draws(seed) if seed is not None else None
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
        gpi_spike_buffer_size: int | None = None,
        record_th_spikes: bool = False,
        th_spike_buffer_size: int | None = None,
        record_cor_spikes: bool = False,
        cor_spike_buffer_size: int | None = None,
    ) -> IntegrateResult:
        if duration_s <= 0:
            msg = "duration_s must be positive"
            raise ValueError(msg)

        spec = dbs_spec if dbs_spec is not None else DbsSpec.none()
        tmax_ms = duration_s * 1000.0
        self._iteration += 1

        if self._rng is None:
            self.reset(seed=self._seed)

        if self.config.smc_pulse_source == "cor_spikes":
            record_cor_spikes = True

        return integrate_network(
            config=self.config,
            duration_s=duration_s,
            dbs_spec=spec,
            record_spikes=record_spikes,
            rng=self._rng,
            iteration=self._iteration,
            seed=self._seed,
            init_draws=self._init_draws,
            gpi_spike_buffer_size=gpi_spike_buffer_size,
            record_th_spikes=record_th_spikes,
            th_spike_buffer_size=th_spike_buffer_size,
            record_cor_spikes=record_cor_spikes,
            cor_spike_buffer_size=cor_spike_buffer_size,
        )
