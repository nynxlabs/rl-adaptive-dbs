"""Synaptic kernels and spike-time convolution for the Kumaravelu CBGT network.

Precomputed ``syn_func_*`` arrays and ``SpikeConvolver`` mirror ``simulate_network_model.m``
(lines 276–346 and ``t_list_*`` update logic) for the native plant **Phase B** port.
"""

from __future__ import annotations

from collections import deque

import numpy as np

T_A_MS: float = 1000.0
TAU_MS: float = 5.0
TAU_I_MS: float = 13.0
GPEAK: float = 0.43
GPEAK1: float = 0.3

__all__ = [
    "GPEAK",
    "GPEAK1",
    "T_A_MS",
    "TAU_I_MS",
    "TAU_MS",
    "SpikeConvolver",
    "build_synaptic_kernels",
]


def build_synaptic_kernels(dt_ms: float) -> dict[str, np.ndarray]:
    """Return precomputed synaptic kernel arrays indexed from 0 (MATLAB index minus one).

    Array length matches ``int(T_A_MS / dt_ms) + 1``, equivalent to MATLAB ``0:dt:t_a``.
    """
    dt = float(dt_ms)
    t_vec = np.arange(0.0, T_A_MS + dt * 0.5, dt, dtype=np.float64)
    expected_len = int(T_A_MS / dt) + 1
    if t_vec.size != expected_len:
        t_vec = np.linspace(0.0, T_A_MS, expected_len, dtype=np.float64)

    const = GPEAK / (TAU_MS * np.exp(-1.0))
    const1 = GPEAK1 / (TAU_MS * np.exp(-1.0))
    const2 = GPEAK1 / (TAU_MS * np.exp(-1.0))

    active = lambda t_lo: (t_vec >= t_lo) & (t_vec <= T_A_MS)

    # TH-CTX
    t_d_th_cor = 5.0
    syn_func_th = (
        const
        * (t_vec - t_d_th_cor)
        * np.exp(-(t_vec - t_d_th_cor) / TAU_MS)
        * active(t_d_th_cor)
    )

    # STN-GPe
    t_d_stn_gpe = 2.0
    taudstngpea = 2.5
    taurstngpea = 0.4
    taudstngpen = 67.0
    taurstngpen = 2.0
    tpeakstngpea = t_d_stn_gpe + (
        (taudstngpea * taurstngpea) / (taudstngpea - taurstngpea)
    ) * np.log(taudstngpea / taurstngpea)
    fstngpea = 1.0 / (
        np.exp(-(tpeakstngpea - t_d_stn_gpe) / taudstngpea)
        - np.exp(-(tpeakstngpea - t_d_stn_gpe) / taurstngpea)
    )
    syn_func_stn_gpea = (
        GPEAK
        * fstngpea
        * (
            np.exp(-(t_vec - t_d_stn_gpe) / taudstngpea)
            - np.exp(-(t_vec - t_d_stn_gpe) / taurstngpea)
        )
        * active(t_d_stn_gpe)
    )
    tpeakstngpen = t_d_stn_gpe + (
        (taudstngpen * taurstngpen) / (taudstngpen - taurstngpen)
    ) * np.log(taudstngpen / taurstngpen)
    fstngpen = 1.0 / (
        np.exp(-(tpeakstngpen - t_d_stn_gpe) / taudstngpen)
        - np.exp(-(tpeakstngpen - t_d_stn_gpe) / taurstngpen)
    )
    syn_func_stn_gpen = (
        GPEAK
        * fstngpen
        * (
            np.exp(-(t_vec - t_d_stn_gpe) / taudstngpen)
            - np.exp(-(t_vec - t_d_stn_gpe) / taurstngpen)
        )
        * active(t_d_stn_gpe)
    )

    # STN-GPi
    t_d_stn_gpi = 1.5
    syn_func_stn_gpi = (
        const
        * (t_vec - t_d_stn_gpi)
        * np.exp(-(t_vec - t_d_stn_gpi) / TAU_MS)
        * active(t_d_stn_gpi)
    )

    # GPe-STN
    t_d_gpe_stn = 4.0
    taudg = 7.7
    taurg = 0.4
    tpeakg = t_d_gpe_stn + (
        (taudg * taurg) / (taudg - taurg)
    ) * np.log(taudg / taurg)
    fg = 1.0 / (
        np.exp(-(tpeakg - t_d_gpe_stn) / taudg)
        - np.exp(-(tpeakg - t_d_gpe_stn) / taurg)
    )
    syn_func_gpe_stn = (
        GPEAK1
        * fg
        * (
            np.exp(-(t_vec - t_d_gpe_stn) / taudg)
            - np.exp(-(t_vec - t_d_gpe_stn) / taurg)
        )
        * active(t_d_gpe_stn)
    )

    # GPe-GPi
    t_d_gpe_gpi = 3.0
    syn_func_gpe_gpi = (
        const1
        * (t_vec - t_d_gpe_gpi)
        * np.exp(-(t_vec - t_d_gpe_gpi) / TAU_MS)
        * active(t_d_gpe_gpi)
    )

    # GPe-GPe
    t_d_gpe_gpe = 1.0
    syn_func_gpe_gpe = (
        const1
        * (t_vec - t_d_gpe_gpe)
        * np.exp(-(t_vec - t_d_gpe_gpe) / TAU_MS)
        * active(t_d_gpe_gpe)
    )

    # GPi-TH
    t_d_gpi_th = 5.0
    syn_func_gpi_th = (
        const1
        * (t_vec - t_d_gpi_th)
        * np.exp(-(t_vec - t_d_gpi_th) / TAU_MS)
        * active(t_d_gpi_th)
    )

    # Indirect Str-GPe
    t_d_d2_gpe = 5.0
    syn_func_str_indr = (
        const2
        * (t_vec - t_d_d2_gpe)
        * np.exp(-(t_vec - t_d_d2_gpe) / TAU_MS)
        * active(t_d_d2_gpe)
    )

    # Direct Str-GPi
    t_d_d1_gpi = 4.0
    syn_func_str_dr = (
        const2
        * (t_vec - t_d_d1_gpi)
        * np.exp(-(t_vec - t_d_d1_gpi) / TAU_MS)
        * active(t_d_d1_gpi)
    )

    # Cortex-Indirect Str
    t_d_cor_d2 = 5.1
    syn_func_cor_d2 = (
        const
        * (t_vec - t_d_cor_d2)
        * np.exp(-(t_vec - t_d_cor_d2) / TAU_MS)
        * active(t_d_cor_d2)
    )

    # Cortex-STN
    t_d_cor_stn = 5.9
    taudn = 90.0
    taurn = 2.0
    tauda = 2.49
    taura = 0.5
    tpeaka = t_d_cor_stn + (
        (tauda * taura) / (tauda - taura)
    ) * np.log(tauda / taura)
    fa = 1.0 / (
        np.exp(-(tpeaka - t_d_cor_stn) / tauda)
        - np.exp(-(tpeaka - t_d_cor_stn) / taura)
    )
    syn_func_cor_stn_a = (
        GPEAK
        * fa
        * (
            np.exp(-(t_vec - t_d_cor_stn) / tauda)
            - np.exp(-(t_vec - t_d_cor_stn) / taura)
        )
        * active(t_d_cor_stn)
    )
    tpeakn = t_d_cor_stn + (
        (taudn * taurn) / (taudn - taurn)
    ) * np.log(taudn / taurn)
    fn = 1.0 / (
        np.exp(-(tpeakn - t_d_cor_stn) / taudn)
        - np.exp(-(tpeakn - t_d_cor_stn) / taurn)
    )
    syn_func_cor_stn_n = (
        GPEAK
        * fn
        * (
            np.exp(-(t_vec - t_d_cor_stn) / taudn)
            - np.exp(-(t_vec - t_d_cor_stn) / taurn)
        )
        * active(t_d_cor_stn)
    )

    return {
        "syn_func_th": syn_func_th,
        "syn_func_stn_gpea": syn_func_stn_gpea,
        "syn_func_stn_gpen": syn_func_stn_gpen,
        "syn_func_stn_gpi": syn_func_stn_gpi,
        "syn_func_gpe_stn": syn_func_gpe_stn,
        "syn_func_gpe_gpi": syn_func_gpe_gpi,
        "syn_func_gpe_gpe": syn_func_gpe_gpe,
        "syn_func_gpi_th": syn_func_gpi_th,
        "syn_func_str_indr": syn_func_str_indr,
        "syn_func_str_dr": syn_func_str_dr,
        "syn_func_cor_d2": syn_func_cor_d2,
        "syn_func_cor_stn_a": syn_func_cor_stn_a,
        "syn_func_cor_stn_n": syn_func_cor_stn_n,
    }


class SpikeConvolver:
    """Per-neuron spike index lists matching MATLAB ``t_list_*`` bookkeeping.

    Spike indices are **1-based** internally (append ``1`` on spike). Kernel arrays
    from :func:`build_synaptic_kernels` are **0-based**; :meth:`evaluate` maps
    ``index -> syn_func[index - 1]``.
    """

    def __init__(self, n_neurons: int, dt_ms: float, *, t_a_ms: float = T_A_MS) -> None:
        if n_neurons < 1:
            raise ValueError("n_neurons must be at least 1")
        self.n_neurons = n_neurons
        self.dt_ms = float(dt_ms)
        self.t_a_ms = float(t_a_ms)
        self.max_index = int(t_a_ms / self.dt_ms)
        self._times: list[deque[int]] = [deque() for _ in range(n_neurons)]

    def on_spike(self, neuron_index: int) -> None:
        """Record a spike for ``neuron_index`` (0-based neuron id)."""
        self._times[neuron_index].append(1)

    def step(self) -> None:
        """Advance all spike indices by one and trim expired entries."""
        max_idx = self.max_index
        for times in self._times:
            if not times:
                continue
            for i in range(len(times)):
                times[i] += 1
            if times[0] == max_idx:
                times.popleft()

    def evaluate(self, neuron_index: int, syn_func: np.ndarray) -> float:
        """Sum kernel samples for all active spikes on one neuron."""
        times = self._times[neuron_index]
        if not times:
            return 0.0
        total = 0.0
        for index in times:
            total += syn_func[index - 1]
        return total

    def evaluate_all(self, syn_func: np.ndarray) -> np.ndarray:
        """Vectorized kernel sum for every neuron (one synaptic kernel)."""
        out = np.zeros(self.n_neurons, dtype=np.float64)
        for j in range(self.n_neurons):
            times = self._times[j]
            if not times:
                continue
            total = 0.0
            for index in times:
                total += syn_func[index - 1]
            out[j] = total
        return out
