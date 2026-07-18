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

    def _plant_opts(
        self,
        eng: Any,
        *,
        record_th_spikes: bool,
        dbs_spec: DbsSpec,
        n_steps: int,
    ) -> Any:
        import matlab

        opts = eng.struct(
            "smc_schedule",
            str(self.config.smc_schedule),
            "smc_frequency_hz",
            float(self.config.smc_frequency_hz),
            "smc_amplitude",
            float(self.config.smc_amplitude),
            "smc_pulse_width_ms",
            float(self.config.smc_pulse_width_ms),
            "smc_invgamma_shape",
            float(self.config.smc_invgamma_shape),
            "smc_invgamma_scale_ms",
            float(self.config.smc_invgamma_scale_ms),
            "smc_site",
            str(self.config.smc_site),
            "smc_cortical_amplitude",
            float(self.config.smc_cortical_amplitude),
            "smc_pulse_source",
            str(self.config.smc_pulse_source),
            "return_th_spikes",
            bool(record_th_spikes),
        )
        if dbs_spec.idbs is not None:
            idbs = np.asarray(dbs_spec.idbs, dtype=np.float64).reshape(-1)
            if idbs.size != n_steps:
                msg = (
                    f"dbs_spec.idbs length {idbs.size} != expected {n_steps} "
                    f"for MatlabPlant integrate"
                )
                raise ValueError(msg)
            opts["idbs"] = matlab.double(idbs.reshape(1, -1).tolist())
        return opts

    def integrate(
        self,
        duration_s: float,
        dbs_spec: DbsSpec | None = None,
        *,
        record_spikes: bool = True,
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
        dt_ms = self.config.dt_ms
        n_steps = int(round(tmax_ms / dt_ms)) + 1
        eng = self._get_engine()

        self._iteration += 1
        seed_arg: float | list[float]
        if self._seed is None:
            seed_arg = []
        else:
            seed_arg = float(self._seed)

        plant_opts = self._plant_opts(
            eng,
            record_th_spikes=record_th_spikes,
            dbs_spec=spec,
            n_steps=n_steps,
        )
        nargout = 14 if record_th_spikes else 6
        raw = eng.simulate_network_model(
            float(self._iteration),
            float(self.config.pd),
            float(self.config.corstim),
            float(spec.pick_dbs_freq),
            True,
            seed_arg,
            float(tmax_ms),
            [],
            plant_opts,
            nargout=nargout,
        )
        gpi_cell, dt_ms, tmax_out, pd_out, _pick, dbs_freq_hz = raw[:6]

        gpi_spikes: list[np.ndarray] = []
        if record_spikes:
            gpi_spikes = spikes_from_matlab_cell(
                gpi_cell,
                neurons=self.config.neurons_per_region,
            )

        th_spikes: list[np.ndarray] = []
        smc_pulse_times_s: np.ndarray = np.array([], dtype=np.float64)
        if record_th_spikes:
            th_cell = raw[12]
            th_spikes = spikes_from_matlab_cell(
                th_cell,
                neurons=self.config.neurons_per_region,
            )
            smc_raw = raw[13]
            smc_pulse_times_s = np.asarray(smc_raw, dtype=np.float64).reshape(-1)

        p_beta_val: float | None = None
        if record_spikes:
            p_beta_val = p_beta(
                gpi_spikes,
                dt_ms=float(dt_ms),
                segment_duration_s=duration_s,
            )

        info: dict[str, Any] = {
            "dbs_freq_hz": float(dbs_freq_hz),
            "gpi_spike_counts": spike_counts(gpi_spikes).tolist() if record_spikes else [],
            "tmax_ms": float(tmax_out),
            "smc_frequency_hz": float(self.config.smc_frequency_hz),
            "smc_schedule": str(self.config.smc_schedule),
            "smc_site": str(self.config.smc_site),
            "smc_pulse_source": str(self.config.smc_pulse_source),
            "smc_pulse_times_s": smc_pulse_times_s.tolist(),
            "smc_pulse_count": int(smc_pulse_times_s.size),
        }
        if record_th_spikes:
            info["th_spike_counts"] = spike_counts(th_spikes).tolist()
            info["th_spikes"] = th_spikes
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
