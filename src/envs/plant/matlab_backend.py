"""MATLAB Engine bridge to Kumaravelu et al. (2016) CBGT dynamics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from envs.plant.biomarkers import p_beta
from envs.plant.config import PlantConfig
from envs.plant.dbs import DbsSpec
from envs.plant.spikes import spike_counts, spikes_from_matlab_cell


def _repo_root() -> Path:
    env = os.environ.get("RL_ADAPTIVE_DBS_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def _model_dir() -> Path:
    env = os.environ.get("RL_ADAPTIVE_DBS_MATLAB_MODEL")
    if env:
        return Path(env).resolve()
    return _repo_root() / "reference-material" / "KumaraveluEtAl2016"


@dataclass(frozen=True)
class IntegrateResult:
    """One plant integration segment."""

    gpi_spikes: list[np.ndarray]
    duration_s: float
    dt_ms: float
    pd: int
    dbs_spec: DbsSpec
    seed: int | None
    p_beta: float | None
    info: dict[str, Any]


class MatlabPlant:
    """Non-Gym Kumaravelu plant via matlab.engine (docs/plant.md §7)."""

    def __init__(
        self,
        config: PlantConfig | None = None,
        *,
        engine: Any | None = None,
    ) -> None:
        self.config = config or PlantConfig()
        self._engine = engine
        self._owns_engine = engine is None
        self._seed: int | None = None
        self._iteration = 0

    @property
    def config(self) -> PlantConfig:
        return self._config

    @config.setter
    def config(self, value: PlantConfig) -> None:
        self._config = value

    def _get_engine(self) -> Any:
        if self._engine is None:
            import matlab.engine

            self._engine = matlab.engine.start_matlab()
            self._engine.cd(str(_model_dir()), nargout=0)
        return self._engine

    def close(self) -> None:
        if self._engine is not None and self._owns_engine:
            self._engine.exit()
        self._engine = None

    def __enter__(self) -> MatlabPlant:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self, seed: int | None = None) -> MatlabPlant:
        self._seed = seed
        self._iteration = 0
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
        eng = self._get_engine()

        self._iteration += 1
        seed_arg: float | list[float]
        if self._seed is None:
            seed_arg = []
        else:
            seed_arg = float(self._seed)

        raw = eng.simulate_network_model(
            float(self._iteration),
            float(self.config.pd),
            float(self.config.corstim),
            float(spec.pick_dbs_freq),
            True,
            seed_arg,
            float(tmax_ms),
            nargout=6,
        )
        gpi_cell, dt_ms, tmax_out, pd_out, _pick, dbs_freq_hz = raw

        gpi_spikes: list[np.ndarray] = []
        if record_spikes:
            gpi_spikes = spikes_from_matlab_cell(
                gpi_cell,
                neurons=self.config.neurons_per_region,
            )

        p_beta_val: float | None = None
        if record_spikes:
            p_beta_val = p_beta(
                gpi_spikes,
                dt_ms=float(dt_ms),
                segment_duration_s=duration_s,
            )

        info = {
            "dbs_freq_hz": float(dbs_freq_hz),
            "gpi_spike_counts": spike_counts(gpi_spikes).tolist() if record_spikes else [],
            "tmax_ms": float(tmax_out),
        }
        if p_beta_val is not None:
            info["p_beta"] = p_beta_val
        return IntegrateResult(
            gpi_spikes=gpi_spikes,
            duration_s=duration_s,
            dt_ms=float(dt_ms),
            pd=int(pd_out),
            dbs_spec=spec,
            seed=self._seed,
            p_beta=p_beta_val,
            info=info,
        )
