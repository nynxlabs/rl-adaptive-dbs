"""CTX-BG-TH network integrator (Kumaravelu et al., 2016 native port)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from envs.plant.biomarkers import p_beta, resolve_smc_pulse_times, smc_pulse_times_from_trace
from envs.plant.config import (
    BOC_SMC_AMPLITUDE,
    BOC_SMC_INVGAMMA_SCALE_MS,
    BOC_SMC_INVGAMMA_SHAPE,
    BOC_SMC_PULSE_WIDTH_MS,
    PlantConfig,
)
from envs.plant.dbs import DbsSpec, create_dbs_current
from envs.plant.matlab_backend import IntegrateResult
from envs.plant.network import gating as g
from envs.plant.network.synapses import SpikeConvolver, build_synaptic_kernels
from envs.plant.spikes import find_spike_times, spike_counts

try:
    from envs.plant.network.numba_loop import (
        N_CONV,
        gpi_spikes_from_buffer,
        numba_loop_available,
        run_cbgt_loop,
    )
except ImportError:  # pragma: no cover
    numba_loop_available = lambda: False  # type: ignore[assignment,misc]

POPULATION_NAMES = (
    "ctx_exc",
    "ctx_inh",
    "str_direct",
    "str_indirect",
    "stn",
    "gpe",
    "gpi",
    "thalamus",
)

DEFAULT_VOLTAGE_MEAN: float = -62.0
DEFAULT_VOLTAGE_STD: float = 5.0
STR_VOLTAGE_MEAN: float = -63.8

SPIKE_SYN_THRESHOLD_MV: float = -10.0

# Izhikevich regular-spiking / fast-spiking parameters (simulate_network_model.m)
_AE: float = 0.02
_BE: float = 0.2
_CE: float = -65.0
_DE: float = 8.0
_AI: float = 0.1
_BI: float = 0.2
_CI: float = -65.0
_DI: float = 2.0

_CM: float = 1.0
_GL = np.array([0.05, 0.35, 0.1, 0.1], dtype=np.float64)
_EL = np.array([-70.0, -60.0, -65.0, -67.0], dtype=np.float64)
_GNA = np.array([3.0, 49.0, 120.0, 100.0], dtype=np.float64)
_ENA = np.array([50.0, 60.0, 55.0, 50.0], dtype=np.float64)
_GK = np.array([5.0, 57.0, 30.0, 80.0], dtype=np.float64)
_EK = np.array([-75.0, -90.0, -80.0, -100.0], dtype=np.float64)
_GT = np.array([5.0, 5.0, 0.5], dtype=np.float64)
_ET: float = 0.0
_GCA = np.array([0.0, 2.0, 0.15], dtype=np.float64)
_ECA = np.array([0.0, 140.0, 120.0], dtype=np.float64)
_EM: float = -100.0
_GAHP = np.array([0.0, 20.0, 10.0], dtype=np.float64)
_K1 = np.array([0.0, 15.0, 10.0], dtype=np.float64)
_KCA = np.array([0.0, 22.5, 15.0], dtype=np.float64)
_GA: float = 5.0
_GL_STN: float = 15.0
_GCAK: float = 1.0

_KCA_STN: float = 2e-3
_Z: float = 2.0
_F: float = 96485.0
_CAO: float = 2000.0
_R: float = 8314.0
_T: float = 298.0
_ALP: float = 1.0 / (_Z * _F)
_CON: float = (_R * _T) / (_Z * _F)

_ESYN = np.array([-85.0, 0.0, -85.0, 0.0, -85.0, -85.0, -80.0], dtype=np.float64)
_TAU: float = 5.0
_TAU_I: float = 13.0
_GPEAK: float = 0.43

_GGITH: float = 0.112
_GGESN: float = 0.5
_GSTRGPE: float = 0.5
_GSTRGPI: float = 0.5
_GGIGI: float = 0.5
_GM: float = 1.0
_GGABA: float = 0.1
_GCORINDRSTR: float = 0.07
_GIE: float = 0.2
_GTHCOR: float = 0.15
_GEi: float = 0.1

_IAPPTH: float = 1.2
_IAPPGPI: float = 3.0

_STN_TD2: float = 130.0
_STN_TR2: float = 2.0
_GPE_TR: float = 30.0


@dataclass
class NetworkState:
    """Current membrane and synaptic state for each population."""

    populations: Mapping[str, np.ndarray]
    time_ms: float


@dataclass(frozen=True)
class NetworkInitDraws:
    """Fixed initialization draws exported from MATLAB for equivalence tests."""

    v1: np.ndarray
    v2: np.ndarray
    v3: np.ndarray
    v4: np.ndarray
    v5: np.ndarray
    v6: np.ndarray
    perms: tuple[np.ndarray, ...]
    gcorsna: np.ndarray
    gcorsnn: np.ndarray
    gcordrstr: np.ndarray
    ggege: np.ndarray
    gsngen: np.ndarray
    gsngea: np.ndarray
    gsngi: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> NetworkInitDraws:
        data = np.load(path)
        perms = tuple(data[f"perm_{index}"] for index in range(15))
        return cls(
            v1=data["v1"],
            v2=data["v2"],
            v3=data["v3"],
            v4=data["v4"],
            v5=data["v5"],
            v6=data["v6"],
            perms=perms,
            gcorsna=data["gcorsna"],
            gcorsnn=data["gcorsnn"],
            gcordrstr=data["gcordrstr"],
            ggege=data["ggege"],
            gsngen=data["gsngen"],
            gsngea=data["gsngea"],
            gsngi=data["gsngi"],
        )


def _create_cortical_stimulus(tmax_ms: float, dt_ms: float) -> np.ndarray:
    """Port of ``Iappco`` pulse when ``corstim==1`` (simulate_network_model.m)."""
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    iappco = np.zeros(n_steps, dtype=np.float64)
    start_idx = int(round(1000.0 / dt_ms))
    end_idx = int(round((1000.0 + 0.3) / dt_ms))
    iappco[start_idx - 1 : end_idx] = 350.0
    return iappco


def create_periodic_cortical_stimulus(
    frequency_hz: float,
    *,
    tmax_ms: float,
    dt_ms: float,
    amplitude: float = 350.0,
    pulse_width_ms: float = 0.3,
) -> np.ndarray:
    """Periodic cortical ``Iappco`` pulses (legacy probe helper)."""
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    iappco = np.zeros(n_steps, dtype=np.float64)
    if frequency_hz <= 0.0:
        return iappco
    isi_ms = 1000.0 / frequency_hz
    pulse_steps = max(1, int(round(pulse_width_ms / dt_ms)))
    onset_ms = 0.0
    while onset_ms <= tmax_ms:
        start_idx = int(round(onset_ms / dt_ms))
        end_idx = min(start_idx + pulse_steps, n_steps)
        if start_idx < n_steps:
            iappco[start_idx:end_idx] = amplitude
        onset_ms += isi_ms
    return iappco


def create_periodic_thalamic_smc(
    frequency_hz: float,
    *,
    tmax_ms: float,
    dt_ms: float,
    amplitude: float = BOC_SMC_AMPLITUDE,
    pulse_width_ms: float = BOC_SMC_PULSE_WIDTH_MS,
) -> np.ndarray:
    """BoC-style SMC pulse component for ``Iappth`` (Gao: embedded in TH activations)."""
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    smc = np.zeros(n_steps, dtype=np.float64)
    if frequency_hz <= 0.0:
        return smc
    isi_ms = 1000.0 / frequency_hz
    pulse_steps = max(1, int(round(pulse_width_ms / dt_ms)))
    onset_ms = 0.0
    while onset_ms <= tmax_ms:
        start_idx = int(round(onset_ms / dt_ms))
        end_idx = min(start_idx + pulse_steps, n_steps)
        if start_idx < n_steps:
            smc[start_idx:end_idx] = amplitude
        onset_ms += isi_ms
    return smc


def create_boc_invgamma_thalamic_smc(
    *,
    tmax_ms: float,
    dt_ms: float,
    rng: np.random.Generator,
    amplitude: float = BOC_SMC_AMPLITUDE,
    pulse_width_ms: float = BOC_SMC_PULSE_WIDTH_MS,
    invgamma_shape: float = BOC_SMC_INVGAMMA_SHAPE,
    invgamma_scale_ms: float = BOC_SMC_INVGAMMA_SCALE_MS,
) -> np.ndarray:
    """BoC inverse-gamma SMC pulse component (Jovanov platform / Gao EI protocol)."""
    from scipy.stats import invgamma

    n_steps = int(round(tmax_ms / dt_ms)) + 1
    smc = np.zeros(n_steps, dtype=np.float64)
    pulse_steps = max(1, int(round(pulse_width_ms / dt_ms)))
    onset_ms = 0.0
    while onset_ms <= tmax_ms:
        start_idx = int(round(onset_ms / dt_ms))
        end_idx = min(start_idx + pulse_steps, n_steps)
        if start_idx < n_steps:
            smc[start_idx:end_idx] = amplitude
        period_ms = float(
            invgamma.rvs(invgamma_shape, scale=invgamma_scale_ms, random_state=rng)
        )
        period_ms = max(period_ms, pulse_width_ms + 1e-3)
        onset_ms += period_ms
    return smc


def _build_smc_pulse_component(
    config: PlantConfig,
    *,
    tmax_ms: float,
    dt_ms: float,
    rng: np.random.Generator,
    amplitude: float | None = None,
) -> np.ndarray:
    """BoC or periodic SMC pulse component (rising edges define SMCτ)."""
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    amp = float(config.smc_amplitude if amplitude is None else amplitude)
    schedule = config.smc_schedule
    if schedule == "off" and config.smc_frequency_hz > 0.0:
        schedule = "periodic"

    if schedule == "boc":
        return create_boc_invgamma_thalamic_smc(
            tmax_ms=tmax_ms,
            dt_ms=dt_ms,
            rng=rng,
            amplitude=amp,
            pulse_width_ms=config.smc_pulse_width_ms,
            invgamma_shape=config.smc_invgamma_shape,
            invgamma_scale_ms=config.smc_invgamma_scale_ms,
        )
    hz = config.smc_frequency_hz if schedule == "periodic" else 10.0
    return create_periodic_thalamic_smc(
        hz,
        tmax_ms=tmax_ms,
        dt_ms=dt_ms,
        amplitude=amp,
        pulse_width_ms=config.smc_pulse_width_ms,
    )


def _resolve_iappco(
    *,
    corstim: int,
    config: PlantConfig,
    tmax_ms: float,
    dt_ms: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Build cortical drive, effective ``corstim``, and SMCτ component for EI."""
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    smc_trace = np.zeros(n_steps, dtype=np.float64)
    if config.smc_enabled() and config.smc_site == "cortical":
        smc_trace = _build_smc_pulse_component(
            config,
            tmax_ms=tmax_ms,
            dt_ms=dt_ms,
            rng=rng,
            amplitude=config.smc_cortical_amplitude,
        )
        return smc_trace.copy(), 1, smc_trace
    if corstim == 1:
        return _create_cortical_stimulus(tmax_ms, dt_ms), 1, smc_trace
    return np.zeros(n_steps, dtype=np.float64), 0, smc_trace


def _resolve_iappth(
    *,
    config: PlantConfig,
    tmax_ms: float,
    dt_ms: float,
    rng: np.random.Generator,
    baseline: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build thalamic drive and SMC pulse component when ``smc_site='thalamic'``."""
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    base = float(config.iappth_baseline if baseline is None else baseline)
    if not config.smc_enabled() or config.smc_site != "thalamic":
        zeros = np.zeros(n_steps, dtype=np.float64)
        return np.full(n_steps, base, dtype=np.float64), zeros

    smc = _build_smc_pulse_component(
        config,
        tmax_ms=tmax_ms,
        dt_ms=dt_ms,
        rng=rng,
    )
    iappth = np.full(n_steps, base, dtype=np.float64) + smc
    return iappth, smc


def _spike_convolver_step(
    conv: SpikeConvolver,
    v_prev: np.ndarray,
    v_curr: np.ndarray,
    syn_funcs: tuple[np.ndarray, ...],
    *,
    threshold: float = SPIKE_SYN_THRESHOLD_MV,
) -> tuple[np.ndarray, ...]:
    cross = (v_prev < threshold) & (v_curr > threshold)
    for j in np.flatnonzero(cross):
        conv.on_spike(int(j))
    outputs = tuple(conv.evaluate_all(sf) for sf in syn_funcs)
    conv.step()
    return outputs


def initialize_network_state(
    config: PlantConfig, rng: np.random.Generator
) -> NetworkState:
    """Draw initial voltages for every population (Kumaravelu ICs)."""

    n = config.neurons_per_region

    def _draw_th_like() -> np.ndarray:
        return rng.normal(DEFAULT_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)

    def _draw_str_like() -> np.ndarray:
        return rng.normal(STR_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)

    populations = {
        "thalamus": _draw_th_like(),
        "stn": _draw_th_like(),
        "gpe": _draw_th_like(),
        "gpi": _draw_th_like(),
        "str_indirect": _draw_str_like(),
        "str_direct": _draw_str_like(),
        "ctx_exc": np.full(n, _CE, dtype=np.float64),
        "ctx_inh": np.full(n, _CI, dtype=np.float64),
    }
    return NetworkState(populations=populations, time_ms=0.0)


DEFAULT_GPI_SPIKE_BUFFER = 512


def integrate_network(
    *,
    config: PlantConfig,
    duration_s: float,
    dbs_spec: DbsSpec,
    record_spikes: bool,
    rng: np.random.Generator,
    iteration: int,
    seed: int | None = None,
    init_draws: NetworkInitDraws | None = None,
    return_traces: tuple[str, ...] = (),
    debug_steps: tuple[int, ...] = (),
    gpi_spike_buffer_size: int | None = None,
    record_th_spikes: bool = False,
    th_spike_buffer_size: int | None = None,
    record_cor_spikes: bool = False,
    cor_spike_buffer_size: int | None = None,
) -> IntegrateResult:
    """Advance the CBGT network for one segment (``CTX_BG_TH_network`` port)."""

    if config.smc_pulse_source == "cor_spikes":
        record_cor_spikes = True

    if rng is None:
        raise ValueError("rng must be initialized before integrating the network")

    dt_ms = config.dt_ms
    n = config.neurons_per_region
    pd = int(config.pd)
    corstim = int(config.corstim)
    smc_frequency_hz = float(config.smc_frequency_hz)
    tmax_ms = duration_s * 1000.0
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    t_ms = np.arange(n_steps, dtype=np.float64) * dt_ms
    dt = dt_ms

    if dbs_spec.idbs is not None:
        # Option C: precomputed fixed-mean pattern trace (TASK-84). Must sit on
        # the same 0:dt_ms:tmax_ms grid the integrator uses for this segment.
        idbs = np.asarray(dbs_spec.idbs, dtype=np.float64)
        if idbs.shape != (n_steps,):
            msg = (
                f"dbs_spec.idbs length {idbs.shape} != expected ({n_steps},) "
                f"for duration_s={duration_s}, dt_ms={dt_ms}"
            )
            raise ValueError(msg)
    else:
        idbs = create_dbs_current(
            dbs_spec.frequency_hz,
            tmax_ms=tmax_ms,
            dt_ms=dt_ms,
        )
    iappco, corstim, smc_trace_co = _resolve_iappco(
        corstim=corstim,
        config=config,
        tmax_ms=tmax_ms,
        dt_ms=dt_ms,
        rng=rng,
    )
    iappth, smc_trace_th = _resolve_iappth(
        config=config,
        tmax_ms=tmax_ms,
        dt_ms=dt_ms,
        rng=rng,
    )
    smc_trace = smc_trace_co if config.smc_site == "cortical" else smc_trace_th

    # --- Initial voltages (MATLAB draw order: v1..v6) ---
    if init_draws is not None:
        v1, v2, v3, v4, v5, v6 = (
            init_draws.v1,
            init_draws.v2,
            init_draws.v3,
            init_draws.v4,
            init_draws.v5,
            init_draws.v6,
        )
    else:
        v1 = rng.normal(DEFAULT_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)
        v2 = rng.normal(DEFAULT_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)
        v3 = rng.normal(DEFAULT_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)
        v4 = rng.normal(DEFAULT_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)
        v5 = rng.normal(STR_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)
        v6 = rng.normal(STR_VOLTAGE_MEAN, DEFAULT_VOLTAGE_STD, size=n)

    vth = np.asarray(v1, dtype=np.float64).copy()
    vsn = np.asarray(v2, dtype=np.float64).copy()
    vge = np.asarray(v3, dtype=np.float64).copy()
    vgi_curr = np.asarray(v4, dtype=np.float64).copy()
    vstr_indr = np.asarray(v5, dtype=np.float64).copy()
    vstr_dr = np.asarray(v6, dtype=np.float64).copy()
    ve = np.full(n, _CE, dtype=np.float64)
    vi = np.full(n, _CI, dtype=np.float64)

    ue = np.full(n, _BE * _CE, dtype=np.float64)
    ui = np.full(n, _BI * _CI, dtype=np.float64)

    trace_names = frozenset(return_traces)
    trace_vth = np.empty((n, n_steps), dtype=np.float64) if "vth" in trace_names else None
    trace_vsn = np.empty((n, n_steps), dtype=np.float64) if "vsn" in trace_names else None
    trace_vge = np.empty((n, n_steps), dtype=np.float64) if "vge" in trace_names else None
    trace_vgi = np.empty((n, n_steps), dtype=np.float64) if "vgi" in trace_names else None
    trace_vstr_indr = (
        np.empty((n, n_steps), dtype=np.float64) if "vstr_indr" in trace_names else None
    )
    if trace_vth is not None:
        trace_vth[:, 0] = vth
    if trace_vsn is not None:
        trace_vsn[:, 0] = vsn
    if trace_vge is not None:
        trace_vge[:, 0] = vge
    if trace_vgi is not None:
        trace_vgi[:, 0] = vgi_curr
    if trace_vstr_indr is not None:
        trace_vstr_indr[:, 0] = vstr_indr

    gpi_spike_lists: list[list[float]] | None = None
    if record_spikes and trace_vgi is None:
        gpi_spike_lists = [[] for _ in range(n)]

    th_spike_lists: list[list[float]] | None = None
    if record_th_spikes:
        th_spike_lists = [[] for _ in range(n)]

    cor_spike_lists: list[list[float]] | None = None
    if record_cor_spikes:
        cor_spike_lists = [[] for _ in range(2 * n)]

    # --- Wiring permutations (15 randperm draws) ---
    if init_draws is not None:
        perm = list(init_draws.perms)
    else:
        perm = [rng.permutation(n) for _ in range(15)]
    (
        all_idx,
        bll,
        cll,
        dll,
        ell,
        fll,
        gll,
        hll,
        ill,
        jll,
        kll,
        lll,
        mll,
        nll,
        oll,
    ) = perm

    # --- Heterogeneous conductances ---
    if init_draws is not None:
        gcorsna = init_draws.gcorsna
        gcorsnn = init_draws.gcorsnn
        gcordrstr = init_draws.gcordrstr
        ggege = init_draws.ggege
        gsngen = init_draws.gsngen
        gsngea = init_draws.gsngea
        gsngi = init_draws.gsngi
    else:
        gcorsna = 0.3 * rng.random(n)
        gcorsnn = 0.003 * rng.random(n)
        gcordrstr = (0.07 - 0.044 * pd) + 0.001 * rng.random(n)
        ggege = rng.random(n)

        gsngen = np.zeros(n, dtype=np.float64)
        gsngen[rng.permutation(n)[:2]] = 0.002 * rng.random(2)
        gsngea = np.zeros(n, dtype=np.float64)
        gsngea[rng.permutation(n)[:2]] = 0.3 * rng.random(2)
        gsngi = np.zeros(n, dtype=np.float64)
        gsngi[rng.permutation(n)[:5]] = 0.15

    # --- Gating / channel state at t=0 ---
    N3 = g.gpe_ninf(vge)
    N4 = g.gpe_ninf(vgi_curr)
    H1 = g.th_hinf(vth)
    H3 = g.gpe_hinf(vge)
    H4 = g.gpe_hinf(vgi_curr)
    R1 = g.th_rinf(vth)
    R3 = g.gpe_rinf(vge)
    R4 = g.gpe_rinf(vgi_curr)
    CA2 = 0.1
    CA3 = CA2
    CA4 = CA2

    N2 = g.stn_ninf(vsn)
    H2 = g.stn_hinf(vsn)
    M2 = g.stn_minf(vsn)
    A2 = g.stn_ainf(vsn)
    B2 = g.stn_binf(vsn)
    C2 = g.stn_cinf(vsn)
    D2 = g.stn_d2inf(vsn)
    D1 = g.stn_d1inf(vsn)
    P2 = g.stn_pinf(vsn)
    Q2 = g.stn_qinf(vsn)
    R2 = g.stn_rinf(vsn)
    CAsn2 = np.full(n, 0.005, dtype=np.float64)

    m5 = g.alpham(vstr_indr) / (g.alpham(vstr_indr) + g.betam(vstr_indr))
    h5 = g.alphah(vstr_indr) / (g.alphah(vstr_indr) + g.betah(vstr_indr))
    n5 = g.alphan(vstr_indr) / (g.alphan(vstr_indr) + g.betan(vstr_indr))
    p5 = g.alphap(vstr_indr) / (g.alphap(vstr_indr) + g.betap(vstr_indr))

    m6 = g.alpham(vstr_dr) / (g.alpham(vstr_dr) + g.betam(vstr_dr))
    h6 = g.alphah(vstr_dr) / (g.alphah(vstr_dr) + g.betah(vstr_dr))
    n6 = g.alphan(vstr_dr) / (g.alphan(vstr_dr) + g.betan(vstr_dr))
    p6 = g.alphap(vstr_dr) / (g.alphap(vstr_dr) + g.betap(vstr_dr))

    # --- Synaptic filter states ---
    S2a = np.zeros(n, dtype=np.float64)
    S21a = np.zeros(n, dtype=np.float64)
    S2b = np.zeros(n, dtype=np.float64)
    S21b = np.zeros(n, dtype=np.float64)
    S2an = np.zeros(n, dtype=np.float64)
    S21an = np.zeros(n, dtype=np.float64)
    S3a = np.zeros(n, dtype=np.float64)
    S31a = np.zeros(n, dtype=np.float64)
    S3b = np.zeros(n, dtype=np.float64)
    S31b = np.zeros(n, dtype=np.float64)
    S32b = np.zeros(n, dtype=np.float64)
    S3c = np.zeros(n, dtype=np.float64)
    S31c = np.zeros(n, dtype=np.float64)
    S32c = np.zeros(n, dtype=np.float64)
    S4 = np.zeros(n, dtype=np.float64)
    S5 = np.zeros(n, dtype=np.float64)
    S51 = np.zeros(n, dtype=np.float64)
    S52 = np.zeros(n, dtype=np.float64)
    S53 = np.zeros(n, dtype=np.float64)
    S54 = np.zeros(n, dtype=np.float64)
    S55 = np.zeros(n, dtype=np.float64)
    S56 = np.zeros(n, dtype=np.float64)
    S57 = np.zeros(n, dtype=np.float64)
    S58 = np.zeros(n, dtype=np.float64)
    S59 = np.zeros(n, dtype=np.float64)
    S9 = np.zeros(n, dtype=np.float64)
    S6a = np.zeros(n, dtype=np.float64)
    S6b = np.zeros(n, dtype=np.float64)
    S6bn = np.zeros(n, dtype=np.float64)
    S61b = np.zeros(n, dtype=np.float64)
    S61bn = np.zeros(n, dtype=np.float64)
    S91 = np.zeros(n, dtype=np.float64)
    S92 = np.zeros(n, dtype=np.float64)
    S93 = np.zeros(n, dtype=np.float64)
    S94 = np.zeros(n, dtype=np.float64)
    S95 = np.zeros(n, dtype=np.float64)
    S96 = np.zeros(n, dtype=np.float64)
    S97 = np.zeros(n, dtype=np.float64)
    S98 = np.zeros(n, dtype=np.float64)
    S99 = np.zeros(n, dtype=np.float64)
    S7 = np.zeros(n, dtype=np.float64)
    S8 = np.zeros(n, dtype=np.float64)
    S1a = np.zeros(n, dtype=np.float64)
    S1b = np.zeros(n, dtype=np.float64)
    S1c = np.zeros(n, dtype=np.float64)
    Z1a = np.zeros(n, dtype=np.float64)
    Z1b = np.zeros(n, dtype=np.float64)

    kernels = build_synaptic_kernels(dt_ms)
    conv_th = SpikeConvolver(n, dt_ms)
    conv_stn = SpikeConvolver(n, dt_ms)
    conv_gpe = SpikeConvolver(n, dt_ms)
    conv_gpi = SpikeConvolver(n, dt_ms)
    conv_str_indr = SpikeConvolver(n, dt_ms)
    conv_str_dr = SpikeConvolver(n, dt_ms)
    conv_cor = SpikeConvolver(n, dt_ms)

    # Precomputed circular shifts (same as np.roll on length-n wiring vectors).
    _ar = np.arange(n, dtype=np.intp)
    _roll_m1 = (_ar + 1) % n
    _roll_p1 = (_ar - 1) % n
    _roll_m2 = (_ar + 2) % n
    _roll_p2 = (_ar - 2) % n
    _roll_m3 = (_ar + 3) % n
    _roll_m4 = (_ar + 4) % n
    _roll_m5 = (_ar + 5) % n
    _roll_m6 = (_ar + 6) % n
    _roll_m7 = (_ar + 7) % n
    _roll_m8 = (_ar + 8) % n
    _roll_m9 = (_ar + 9) % n

    iappgpe = 3.0 - 2.0 * corstim * (1 - pd)
    uce_scale = _GPEAK / (_TAU * np.exp(-1.0)) / dt
    gpi_spike_threshold = -20.0
    th_spike_threshold = -20.0
    v1_prev = np.empty(n, dtype=np.float64)
    v2_prev = np.empty(n, dtype=np.float64)
    v3_prev = np.empty(n, dtype=np.float64)
    v4_prev = np.empty(n, dtype=np.float64)
    v5_prev = np.empty(n, dtype=np.float64)
    v6_prev = np.empty(n, dtype=np.float64)
    v7_prev = np.empty(n, dtype=np.float64)
    v8_prev = np.empty(n, dtype=np.float64)

    debug_snapshots: dict[int, dict[str, Any]] = {}
    debug_step_set = frozenset(debug_steps)

    use_numba = (
        numba_loop_available()
        and not debug_steps
        and not return_traces
        and trace_vgi is None
    )
    numba_gpi_buf: np.ndarray | None = None
    numba_gpi_counts: np.ndarray | None = None
    numba_th_buf: np.ndarray | None = None
    numba_th_counts: np.ndarray | None = None
    numba_cor_buf: np.ndarray | None = None
    numba_cor_counts: np.ndarray | None = None

    gpi_buf_len = (
        DEFAULT_GPI_SPIKE_BUFFER
        if gpi_spike_buffer_size is None
        else max(DEFAULT_GPI_SPIKE_BUFFER, int(gpi_spike_buffer_size))
    )
    th_buf_len = (
        DEFAULT_GPI_SPIKE_BUFFER
        if th_spike_buffer_size is None
        else max(DEFAULT_GPI_SPIKE_BUFFER, int(th_spike_buffer_size))
    )
    cor_buf_len = (
        DEFAULT_GPI_SPIKE_BUFFER
        if cor_spike_buffer_size is None
        else max(DEFAULT_GPI_SPIKE_BUFFER, int(cor_spike_buffer_size))
    )

    if use_numba:
        spike_idx = np.zeros((N_CONV, n, 512), dtype=np.int32)
        spike_n = np.zeros((N_CONV, n), dtype=np.int32)
        numba_gpi_buf = np.zeros((n, gpi_buf_len), dtype=np.float64)
        numba_gpi_counts = np.zeros(n, dtype=np.int32)
        if record_th_spikes:
            numba_th_buf = np.zeros((n, th_buf_len), dtype=np.float64)
            numba_th_counts = np.zeros(n, dtype=np.int32)
        else:
            numba_th_buf = np.zeros((n, 1), dtype=np.float64)
            numba_th_counts = np.zeros(n, dtype=np.int32)
        if record_cor_spikes:
            numba_cor_buf = np.zeros((2 * n, cor_buf_len), dtype=np.float64)
            numba_cor_counts = np.zeros(2 * n, dtype=np.int32)
        else:
            numba_cor_buf = np.zeros((1, 1), dtype=np.float64)
            numba_cor_counts = np.zeros(1, dtype=np.int32)
        ca3_state = np.full(n, float(CA3), dtype=np.float64)
        ca4_state = np.full(n, float(CA4), dtype=np.float64)
        run_cbgt_loop(
            n_steps,
            dt,
            n,
            pd,
            corstim,
            conv_th.max_index,
            idbs,
            iappco,
            iappth,
            float(config.ggith),
            t_ms,
            all_idx,
            bll,
            cll,
            dll,
            ell,
            fll,
            gll,
            hll,
            ill,
            jll,
            kll,
            lll,
            mll,
            nll,
            oll,
            _roll_m1,
            _roll_p1,
            _roll_m2,
            _roll_p2,
            _roll_m3,
            _roll_m4,
            _roll_m5,
            _roll_m6,
            _roll_m7,
            _roll_m8,
            _roll_m9,
            gcorsna,
            gcorsnn,
            gcordrstr,
            ggege,
            gsngen,
            gsngea,
            gsngi,
            kernels["syn_func_th"],
            kernels["syn_func_stn_gpea"],
            kernels["syn_func_stn_gpen"],
            kernels["syn_func_stn_gpi"],
            kernels["syn_func_gpe_stn"],
            kernels["syn_func_gpe_gpi"],
            kernels["syn_func_gpe_gpe"],
            kernels["syn_func_gpi_th"],
            kernels["syn_func_str_indr"],
            kernels["syn_func_str_dr"],
            kernels["syn_func_cor_d2"],
            kernels["syn_func_cor_stn_a"],
            kernels["syn_func_cor_stn_n"],
            vth,
            vsn,
            vge,
            vgi_curr,
            vstr_indr,
            vstr_dr,
            ve,
            vi,
            ue,
            ui,
            H1,
            R1,
            N2,
            H2,
            M2,
            A2,
            B2,
            C2,
            D2,
            D1,
            P2,
            Q2,
            R2,
            CAsn2,
            N3,
            H3,
            R3,
            ca3_state,
            N4,
            H4,
            R4,
            ca4_state,
            m5,
            h5,
            n5,
            p5,
            m6,
            h6,
            n6,
            p6,
            S2a,
            S21a,
            S2b,
            S2an,
            S21an,
            S3a,
            S31a,
            S3b,
            S31b,
            S32b,
            S3c,
            S31c,
            S32c,
            S4,
            S5,
            S51,
            S52,
            S53,
            S54,
            S55,
            S56,
            S57,
            S58,
            S59,
            S9,
            S6a,
            S6b,
            S6bn,
            S61b,
            S61bn,
            S91,
            S92,
            S93,
            S94,
            S95,
            S96,
            S97,
            S98,
            S99,
            S7,
            S8,
            S1a,
            S1b,
            S1c,
            Z1a,
            Z1b,
            spike_idx,
            spike_n,
            numba_gpi_buf,
            numba_gpi_counts,
            numba_th_buf,
            numba_th_counts,
            record_th_spikes,
            numba_cor_buf,
            numba_cor_counts,
            record_cor_spikes,
            iappgpe,
            uce_scale,
        )

    # --- Main Euler loop: Python step=1..n_steps-1 ↔ MATLAB i=2:length(t) ---
    if not use_numba:
        for step in range(1, n_steps):
            np.copyto(v1_prev, vth)
            np.copyto(v2_prev, vsn)
            np.copyto(v3_prev, vge)
            np.copyto(v4_prev, vgi_curr)
            np.copyto(v5_prev, vstr_indr)
            np.copyto(v6_prev, vstr_dr)
            np.copyto(v7_prev, ve)
            np.copyto(v8_prev, vi)
            V1 = v1_prev
            V2 = v2_prev
            V3 = v3_prev
            V4 = v4_prev
            V5 = v5_prev
            V6 = v6_prev
            V7 = v7_prev
            V8 = v8_prev

            # Synaptic delay / wiring shifts
            S21a = S2a[_roll_p1]
            S21an = S2an[_roll_p1]
            S21b = S2b[_roll_p1]
            S31a = S3a[_roll_m1]
            S31b = S3b[_roll_m1]
            S31c = S3c[_roll_m1]
            S32c = S3c[_roll_p2]
            S32b = S3b[_roll_p2]

            S11cr = S1c[all_idx]
            S12cr = S1c[bll]
            S13cr = S1c[cll]
            S14cr = S1c[dll]
            S11br = S1b[ell]
            S12br = S1b[fll]
            S13br = S1b[gll]
            S14br = S1b[hll]
            S11ar = S1a[ill]
            S12ar = S1a[jll]
            S13ar = S1a[kll]
            S14ar = S1a[lll]
            S81r = S8[mll]
            S82r = S8[nll]
            S83r = S8[oll]

            S51 = S5[_roll_m1]
            S52 = S5[_roll_m2]
            S53 = S5[_roll_m3]
            S54 = S5[_roll_m4]
            S55 = S5[_roll_m5]
            S56 = S5[_roll_m6]
            S57 = S5[_roll_m7]
            S58 = S5[_roll_m8]
            S59 = S5[_roll_m9]

            S61b = S6b[_roll_m1]
            S61bn = S6bn[_roll_m1]
            S91 = S9[_roll_m1]
            S92 = S9[_roll_m2]
            S93 = S9[_roll_m3]
            S94 = S9[_roll_m4]
            S95 = S9[_roll_m5]
            S96 = S9[_roll_m6]
            S97 = S9[_roll_m7]
            S98 = S9[_roll_m8]
            S99 = S9[_roll_m9]

            # Instantaneous gating
            m1 = g.th_minf(V1)
            m3 = g.gpe_minf(V3)
            m4 = g.gpe_minf(V4)
            n3 = g.gpe_ninf(V3)
            n4 = g.gpe_ninf(V4)
            h1 = g.th_hinf(V1)
            h3 = g.gpe_hinf(V3)
            h4 = g.gpe_hinf(V4)
            p1 = g.th_pinf(V1)
            a3 = g.gpe_ainf(V3)
            a4 = g.gpe_ainf(V4)
            s3 = g.gpe_sinf(V3)
            s4 = g.gpe_sinf(V4)
            r1 = g.th_rinf(V1)
            r3 = g.gpe_rinf(V3)
            r4 = g.gpe_rinf(V4)

            tn3 = g.gpe_taun(V3)
            tn4 = g.gpe_taun(V4)
            th1 = g.th_tauh(V1)
            th3 = g.gpe_tauh(V3)
            th4 = g.gpe_tauh(V4)
            tr1 = g.th_taur(V1)

            n2 = g.stn_ninf(V2)
            m2 = g.stn_minf(V2)
            h2 = g.stn_hinf(V2)
            a2 = g.stn_ainf(V2)
            b2 = g.stn_binf(V2)
            c2 = g.stn_cinf(V2)
            d2 = g.stn_d2inf(V2)
            d1 = g.stn_d1inf(V2)
            p2 = g.stn_pinf(V2)
            q2 = g.stn_qinf(V2)
            r2 = g.stn_rinf(V2)

            tn2 = g.stn_taun(V2)
            tm2 = g.stn_taum(V2)
            th2 = g.stn_tauh(V2)
            ta2 = g.stn_taua(V2)
            tb2 = g.stn_taub(V2)
            tc2 = g.stn_tauc(V2)
            td1 = g.stn_taud1(V2)
            tp2 = g.stn_taup(V2)
            tq2 = g.stn_tauq(V2)

            ecasn = _CON * np.log(_CAO / CAsn2)

            # --- Thalamic currents ---
            il1 = _GL[0] * (V1 - _EL[0])
            ina1 = _GNA[0] * (m1**3) * H1 * (V1 - _ENA[0])
            ik1 = _GK[0] * ((0.75 * (1.0 - H1)) ** 4) * (V1 - _EK[0])
            it1 = _GT[0] * (p1**2) * R1 * (V1 - _ET)
            igith = float(config.ggith) * (V1 - _ESYN[5]) * S4

            # --- STN currents ---
            ina2 = _GNA[1] * (M2**3) * H2 * (V2 - _ENA[1])
            ik2 = _GK[1] * (N2**4) * (V2 - _EK[1])
            ia2 = _GA * (A2**2) * B2 * (V2 - _EK[1])
            il2_stn = _GL_STN * (C2**2) * D1 * D2 * (V2 - ecasn)
            it2 = _GT[1] * (P2**2) * Q2 * (V2 - ecasn)
            icak2 = _GCAK * (R2**2) * (V2 - _EK[1])
            il2 = _GL[1] * (V2 - _EL[1])
            igesn = _GGESN * ((V2 - _ESYN[0]) * (S3a + S31a))
            icorsnampa = gcorsna * (V2 - _ESYN[1]) * (S6b + S61b)
            icorsnnmda = gcorsnn * (V2 - _ESYN[1]) * (S6bn + S61bn)

            # --- GPe currents ---
            il3 = _GL[2] * (V3 - _EL[2])
            ik3 = _GK[2] * (N3**4) * (V3 - _EK[2])
            ina3 = _GNA[2] * (m3**3) * H3 * (V3 - _ENA[2])
            it3 = _GT[2] * (a3**3) * R3 * (V3 - _ECA[2])
            ica3 = _GCA[2] * (s3**2) * (V3 - _ECA[2])
            iahp3 = _GAHP[2] * (V3 - _EK[2]) * (CA3 / (CA3 + _K1[2]))
            isngeampa = gsngea * ((V3 - _ESYN[1]) * (S2a + S21a))
            isngenmda = gsngen * ((V3 - _ESYN[1]) * (S2an + S21an))
            igege = (0.25 * (pd * 3 + 1)) * ggege * ((V3 - _ESYN[2]) * (S31c + S32c))
            istrgpe = _GSTRGPE * (V3 - _ESYN[5]) * (
                S5 + S51 + S52 + S53 + S54 + S55 + S56 + S57 + S58 + S59
            )

            if step in debug_step_set:
                debug_snapshots[step] = {
                    "V2": V2.copy(),
                    "V3": V3.copy(),
                    "S2a": S2a.copy(),
                    "S21a": S21a.copy(),
                    "S2an": S2an.copy(),
                    "S21an": S21an.copy(),
                    "S3c": S3c.copy(),
                    "S31c": S31c.copy(),
                    "S32c": S32c.copy(),
                    "N3": N3.copy(),
                    "H3": H3.copy(),
                    "R3": R3.copy(),
                    "CA3": CA3.copy(),
                    "isngeampa": isngeampa.copy(),
                    "isngenmda": isngenmda.copy(),
                    "igege": igege.copy(),
                    "istrgpe": istrgpe.copy(),
                    "ik3": ik3.copy(),
                    "ina3": ina3.copy(),
                    "il3": il3.copy(),
                    "it3": it3.copy(),
                    "ica3": ica3.copy(),
                    "iahp3": iahp3.copy(),
                    "stn_spike_times": [list(t) for t in conv_stn._times],
                    "gpe_spike_times": [list(t) for t in conv_gpe._times],
                }

            # --- GPi currents ---
            il4 = _GL[2] * (V4 - _EL[2])
            ik4 = _GK[2] * (N4**4) * (V4 - _EK[2])
            ina4 = _GNA[2] * (m4**3) * H4 * (V4 - _ENA[2])
            it4 = _GT[2] * (a4**3) * R4 * (V4 - _ECA[2])
            ica4 = _GCA[2] * (s4**2) * (V4 - _ECA[2])
            iahp4 = _GAHP[2] * (V4 - _EK[2]) * (CA4 / (CA4 + _K1[2]))
            isngi = gsngi * ((V4 - _ESYN[3]) * (S2b + S21b))
            igigi = _GGIGI * ((V4 - _ESYN[4]) * (S31b + S32b))
            istrgpi = _GSTRGPI * (V4 - _ESYN[5]) * (
                S9 + S91 + S92 + S93 + S94 + S95 + S96 + S97 + S98 + S99
            )

            # --- Striatum D2 ---
            ina5 = _GNA[3] * (m5**3) * h5 * (V5 - _ENA[3])
            ik5 = _GK[3] * (n5**4) * (V5 - _EK[3])
            il5 = _GL[3] * (V5 - _EL[3])
            im5 = (2.6 - 1.1 * pd) * _GM * p5 * (V5 - _EM)
            igaba5 = (_GGABA / 4.0) * (V5 - _ESYN[6]) * (S11cr + S12cr + S13cr + S14cr)
            icorstr5 = _GCORINDRSTR * (V5 - _ESYN[1]) * S6a

            # --- Striatum D1 ---
            ina6 = _GNA[3] * (m6**3) * h6 * (V6 - _ENA[3])
            ik6 = _GK[3] * (n6**4) * (V6 - _EK[3])
            il6 = _GL[3] * (V6 - _EL[3])
            im6 = (2.6 - 1.1 * pd) * _GM * p6 * (V6 - _EM)
            igaba6 = (_GGABA / 3.0) * (V6 - _ESYN[6]) * (S81r + S82r + S83r)
            icorstr6 = gcordrstr * (V6 - _ESYN[1]) * S6a

            # --- Cortex ---
            iie = _GIE * (V7 - _ESYN[0]) * (S11br + S12br + S13br + S14br)
            ithcor = _GTHCOR * (V7 - _ESYN[1]) * S7
            iei = _GEi * (V8 - _ESYN[1]) * (S11ar + S12ar + S13ar + S14ar)

            # --- Thalamus update ---
            vth[:] = V1 + dt * (
                (1.0 / _CM) * (-il1 - ik1 - ina1 - it1 - igith + iappth[step])
            )
            H1 = H1 + dt * ((h1 - H1) / th1)
            R1 = R1 + dt * ((r1 - R1) / tr1)
            if th_spike_lists is not None:
                cross_th = (V1 <= th_spike_threshold) & (vth > th_spike_threshold)
                if cross_th.any():
                    t_spike_s = t_ms[step - 1] / 1000.0
                    for neuron_index in np.flatnonzero(cross_th):
                        th_spike_lists[neuron_index].append(t_spike_s)

            (S7,) = _spike_convolver_step(
                conv_th,
                V1,
                vth,
                (kernels["syn_func_th"],),
            )

            # --- STN update ---
            vsn[:] = V2 + dt * (
                (1.0 / _CM)
                * (
                    -ina2
                    - ik2
                    - ia2
                    - il2_stn
                    - it2
                    - icak2
                    - il2
                    - igesn
                    - icorsnampa
                    - icorsnnmda
                    + idbs[step]
                )
            )
            N2 = N2 + dt * ((n2 - N2) / tn2)
            H2 = H2 + dt * ((h2 - H2) / th2)
            M2 = M2 + dt * ((m2 - M2) / tm2)
            A2 = A2 + dt * ((a2 - A2) / ta2)
            B2 = B2 + dt * ((b2 - B2) / tb2)
            C2 = C2 + dt * ((c2 - C2) / tc2)
            D2 = D2 + dt * ((d2 - D2) / _STN_TD2)
            D1 = D1 + dt * ((d1 - D1) / td1)
            P2 = P2 + dt * ((p2 - P2) / tp2)
            Q2 = Q2 + dt * ((q2 - Q2) / tq2)
            R2 = R2 + dt * ((r2 - R2) / _STN_TR2)
            CAsn2 = CAsn2 + dt * ((-_ALP * (il2_stn + it2)) - (_KCA_STN * CAsn2))

            S2a, S2an, S2b = _spike_convolver_step(
                conv_stn,
                V2,
                vsn,
                (
                    kernels["syn_func_stn_gpea"],
                    kernels["syn_func_stn_gpen"],
                    kernels["syn_func_stn_gpi"],
                ),
            )

            # --- GPe update ---
            vge[:] = V3 + dt * (
                (1.0 / _CM)
                * (-il3 - ik3 - ina3 - it3 - ica3 - iahp3 - isngeampa - isngenmda - igege - istrgpe + iappgpe)
            )
            N3 = N3 + dt * (0.1 * (n3 - N3) / tn3)
            H3 = H3 + dt * (0.05 * (h3 - H3) / th3)
            R3 = R3 + dt * (1.0 * (r3 - R3) / _GPE_TR)
            CA3 = CA3 + dt * (1e-4 * (-ica3 - it3 - _KCA[2] * CA3))

            S3a, S3b, S3c = _spike_convolver_step(
                conv_gpe,
                V3,
                vge,
                (
                    kernels["syn_func_gpe_stn"],
                    kernels["syn_func_gpe_gpi"],
                    kernels["syn_func_gpe_gpe"],
                ),
            )

            # --- GPi update ---
            vgi_curr[:] = V4 + dt * (
                (1.0 / _CM)
                * (-il4 - ik4 - ina4 - it4 - ica4 - iahp4 - isngi - igigi - istrgpi + _IAPPGPI)
            )
            if trace_vgi is not None:
                trace_vgi[:, step] = vgi_curr
            if gpi_spike_lists is not None:
                cross = (V4 <= gpi_spike_threshold) & (vgi_curr > gpi_spike_threshold)
                if cross.any():
                    t_spike_s = t_ms[step - 1] / 1000.0
                    for neuron_index in np.flatnonzero(cross):
                        gpi_spike_lists[neuron_index].append(t_spike_s)

            N4 = N4 + dt * (0.1 * (n4 - N4) / tn4)
            H4 = H4 + dt * (0.05 * (h4 - H4) / th4)
            R4 = R4 + dt * (1.0 * (r4 - R4) / _GPE_TR)
            CA4 = CA4 + dt * (1e-4 * (-ica4 - it4 - _KCA[2] * CA4))

            (S4,) = _spike_convolver_step(
                conv_gpi,
                V4,
                vgi_curr,
                (kernels["syn_func_gpi_th"],),
            )

            # --- Striatum D2 ---
            vstr_indr[:] = V5 + (dt / _CM) * (-ina5 - ik5 - il5 - im5 - igaba5 - icorstr5)
            m5 = m5 + dt * (g.alpham(V5) * (1.0 - m5) - g.betam(V5) * m5)
            h5 = h5 + dt * (g.alphah(V5) * (1.0 - h5) - g.betah(V5) * h5)
            n5 = n5 + dt * (g.alphan(V5) * (1.0 - n5) - g.betan(V5) * n5)
            p5 = p5 + dt * (g.alphap(V5) * (1.0 - p5) - g.betap(V5) * p5)
            S1c = S1c + dt * ((g.Ggaba(V5) * (1.0 - S1c)) - (S1c / _TAU_I))

            (S5,) = _spike_convolver_step(
                conv_str_indr,
                V5,
                vstr_indr,
                (kernels["syn_func_str_indr"],),
            )

            # --- Striatum D1 ---
            vstr_dr[:] = V6 + (dt / _CM) * (-ina6 - ik6 - il6 - im6 - igaba6 - icorstr6)
            m6 = m6 + dt * (g.alpham(V6) * (1.0 - m6) - g.betam(V6) * m6)
            h6 = h6 + dt * (g.alphah(V6) * (1.0 - h6) - g.betah(V6) * h6)
            n6 = n6 + dt * (g.alphan(V6) * (1.0 - n6) - g.betan(V6) * n6)
            p6 = p6 + dt * (g.alphap(V6) * (1.0 - p6) - g.betap(V6) * p6)
            S8 = S8 + dt * ((g.Ggaba(V6) * (1.0 - S8)) - (S8 / _TAU_I))

            (S9,) = _spike_convolver_step(
                conv_str_dr,
                V6,
                vstr_dr,
                (kernels["syn_func_str_dr"],),
            )

            # --- Excitatory cortex (Izhikevich) ---
            ve_new = V7 + dt * ((0.04 * (V7**2)) + (5.0 * V7) + 140.0 - ue - iie - ithcor + iappco[step])
            ue_new = ue + dt * (_AE * ((_BE * V7) - ue))
            for j in range(n):
                if V7[j] >= 30.0:
                    ve_new[j] = _CE
                    ue_new[j] = ue[j] + _DE
                    conv_cor.on_spike(j)
            ve[:] = ve_new
            ue[:] = ue_new
            if cor_spike_lists is not None:
                cross_ve = (V7 <= th_spike_threshold) & (ve > th_spike_threshold)
                if cross_ve.any():
                    t_spike_s = t_ms[step - 1] / 1000.0
                    for neuron_index in np.flatnonzero(cross_ve):
                        cor_spike_lists[neuron_index].append(t_spike_s)

            S6a = conv_cor.evaluate_all(kernels["syn_func_cor_d2"])
            S6b = conv_cor.evaluate_all(kernels["syn_func_cor_stn_a"])
            S6bn = conv_cor.evaluate_all(kernels["syn_func_cor_stn_n"])
            conv_cor.step()

            ace = (V7 < SPIKE_SYN_THRESHOLD_MV) & (ve > SPIKE_SYN_THRESHOLD_MV)
            uce = np.zeros(n, dtype=np.float64)
            uce[ace] = uce_scale
            S1a = S1a + dt * Z1a
            z1adot = uce - (2.0 / _TAU) * Z1a - (1.0 / (_TAU**2)) * S1a
            Z1a = Z1a + dt * z1adot

            # --- Inhibitory cortex ---
            vi_new = V8 + dt * ((0.04 * (V8**2)) + (5.0 * V8) + 140.0 - ui - iei + iappco[step])
            ui_new = ui + dt * (_AI * ((_BI * V8) - ui))
            for j in range(n):
                if V8[j] >= 30.0:
                    vi_new[j] = _CI
                    ui_new[j] = ui[j] + _DI
            vi[:] = vi_new
            ui[:] = ui_new
            if cor_spike_lists is not None:
                cross_vi = (V8 <= th_spike_threshold) & (vi > th_spike_threshold)
                if cross_vi.any():
                    t_spike_s = t_ms[step - 1] / 1000.0
                    for neuron_index in np.flatnonzero(cross_vi):
                        cor_spike_lists[n + neuron_index].append(t_spike_s)

            aci = (V8 < SPIKE_SYN_THRESHOLD_MV) & (vi > SPIKE_SYN_THRESHOLD_MV)
            uci = np.zeros(n, dtype=np.float64)
            uci[aci] = uce_scale
            S1b = S1b + dt * Z1b
            z1bdot = uci - (2.0 / _TAU) * Z1b - (1.0 / (_TAU**2)) * S1b
            Z1b = Z1b + dt * z1bdot

            if trace_vth is not None:
                trace_vth[:, step] = vth
            if trace_vsn is not None:
                trace_vsn[:, step] = vsn
            if trace_vge is not None:
                trace_vge[:, step] = vge
            if trace_vstr_indr is not None:
                trace_vstr_indr[:, step] = vstr_indr

            if trace_vstr_indr is not None:
                trace_vstr_indr[:, step] = vstr_indr

    gpi_spikes: list[np.ndarray] = []
    th_spikes: list[np.ndarray] = []
    cor_spikes: list[np.ndarray] = []
    p_beta_val: float | None = None
    if record_spikes:
        if use_numba and numba_gpi_buf is not None and numba_gpi_counts is not None:
            gpi_spikes = gpi_spikes_from_buffer(numba_gpi_buf, numba_gpi_counts)
        elif gpi_spike_lists is not None:
            gpi_spikes = [
                np.asarray(times, dtype=np.float64) for times in gpi_spike_lists
            ]
        elif trace_vgi is not None:
            gpi_spikes = find_spike_times(trace_vgi, t_ms, n)
        p_beta_val = p_beta(
            gpi_spikes,
            dt_ms=dt_ms,
            segment_duration_s=duration_s,
        )
    if record_th_spikes:
        if use_numba and numba_th_buf is not None and numba_th_counts is not None:
            th_spikes = gpi_spikes_from_buffer(numba_th_buf, numba_th_counts)
        elif th_spike_lists is not None:
            th_spikes = [
                np.asarray(times, dtype=np.float64) for times in th_spike_lists
            ]
        elif trace_vth is not None:
            th_spikes = find_spike_times(trace_vth, t_ms, n)
    if record_cor_spikes:
        if use_numba and numba_cor_buf is not None and numba_cor_counts is not None:
            cor_spikes = gpi_spikes_from_buffer(numba_cor_buf, numba_cor_counts)
        elif cor_spike_lists is not None:
            cor_spikes = [
                np.asarray(times, dtype=np.float64) for times in cor_spike_lists
            ]

    drive_pulse_times = smc_pulse_times_from_trace(smc_trace, dt_ms=dt_ms)
    smc_pulse_times = resolve_smc_pulse_times(
        drive_pulse_times_s=drive_pulse_times,
        cor_spikes=cor_spikes if cor_spikes else None,
        pulse_source=config.smc_pulse_source,
        pulse_width_ms=config.smc_pulse_width_ms,
    )

    info: dict[str, Any] = {
        "dbs_freq_hz": float(dbs_spec.frequency_hz),
        "gpi_spike_counts": spike_counts(gpi_spikes).tolist() if record_spikes else [],
        "tmax_ms": float(tmax_ms),
        "iteration": iteration,
        "smc_frequency_hz": smc_frequency_hz,
        "smc_schedule": config.smc_schedule,
        "smc_site": config.smc_site,
        "smc_pulse_source": config.smc_pulse_source,
        "smc_drive_times_s": drive_pulse_times.tolist(),
        "smc_pulse_times_s": smc_pulse_times.tolist(),
        "smc_pulse_count": int(smc_pulse_times.size),
    }
    if record_th_spikes:
        info["th_spike_counts"] = spike_counts(th_spikes).tolist()
        info["th_spikes"] = th_spikes
    if record_cor_spikes:
        info["cor_spike_counts"] = spike_counts(cor_spikes).tolist()
        info["cor_spikes"] = cor_spikes
    if p_beta_val is not None:
        info["p_beta"] = p_beta_val
    if debug_snapshots:
        info["debug_snapshots"] = debug_snapshots

    traces: dict[str, np.ndarray] = {}
    if return_traces:
        if trace_vgi is not None and "vgi" in trace_names:
            traces["vgi"] = trace_vgi.copy()
        if trace_vsn is not None and "vsn" in trace_names:
            traces["vsn"] = trace_vsn.copy()
        if trace_vge is not None and "vge" in trace_names:
            traces["vge"] = trace_vge.copy()
        if trace_vth is not None and "vth" in trace_names:
            traces["vth"] = trace_vth.copy()
        if trace_vstr_indr is not None and "vstr_indr" in trace_names:
            traces["vstr_indr"] = trace_vstr_indr.copy()
        if traces:
            info["traces"] = traces

    return IntegrateResult(
        gpi_spikes=gpi_spikes,
        duration_s=duration_s,
        dt_ms=dt_ms,
        pd=pd,
        dbs_spec=dbs_spec,
        seed=seed,
        p_beta=p_beta_val,
        info=info,
    )
