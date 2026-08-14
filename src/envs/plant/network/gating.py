"""Voltage-dependent gating helpers for the Kumaravelu et al. (2016) CBGT network.

Ported from ``simulate_network_model.m`` local functions (lines 860–1038) as part
of the native plant **Phase B** integrator work (see ``docs/development/native-plant-port.md``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Ggaba",
    "ah",
    "alphah",
    "alpham",
    "alphan",
    "alphap",
    "betah",
    "betam",
    "betan",
    "betap",
    "bh",
    "gpe_ainf",
    "gpe_hinf",
    "gpe_minf",
    "gpe_ninf",
    "gpe_rinf",
    "gpe_sinf",
    "gpe_tauh",
    "gpe_taun",
    "stn_ainf",
    "stn_binf",
    "stn_cinf",
    "stn_d1inf",
    "stn_d2inf",
    "stn_hinf",
    "stn_minf",
    "stn_ninf",
    "stn_pinf",
    "stn_qinf",
    "stn_rinf",
    "stn_taua",
    "stn_taub",
    "stn_tauc",
    "stn_taud1",
    "stn_tauh",
    "stn_taum",
    "stn_taun",
    "stn_taup",
    "stn_tauq",
    "th_hinf",
    "th_minf",
    "th_pinf",
    "th_rinf",
    "th_tauh",
    "th_taur",
]


def _logistic(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid (avoids exp overflow on extreme voltages)."""
    # Callers pass float64 state vectors; skip redundant asarray in the hot loop.
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def gpe_ainf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 57.0) / 2.0))


def gpe_hinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 58.0) / 12.0))


def gpe_minf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 37.0) / 10.0))


def gpe_ninf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 50.0) / 14.0))


def gpe_rinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 70.0) / 2.0))


def gpe_sinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 2.0))


def gpe_tauh(V: np.ndarray) -> np.ndarray:
    return 0.05 + 0.27 / (1.0 + np.exp(-(V + 40.0) / -12.0))


def gpe_taun(V: np.ndarray) -> np.ndarray:
    return 0.05 + 0.27 / (1.0 + np.exp(-(V + 40.0) / -12.0))


def th_hinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 41.0) / 4.0))


def th_minf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 37.0) / 7.0))


def th_pinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 60.0) / 6.2))


def th_rinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 84.0) / 4.0))


def ah(V: np.ndarray) -> np.ndarray:
    """Rate ``a`` used inside ``th_tauh`` (distinct from ``alphah``)."""
    return 0.128 * np.exp(-(V + 46.0) / 18.0)


def bh(V: np.ndarray) -> np.ndarray:
    """Rate ``b`` used inside ``th_tauh`` (distinct from ``betah``)."""
    return 4.0 / (1.0 + np.exp(-(V + 23.0) / 5.0))


def _vtrap_1mexp(x: np.ndarray, k: float) -> np.ndarray:
    """``x / (1 - exp(-x/k))``; limit ``k`` as ``x -> 0`` (HH singularity)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    thresh = 1e-6 * abs(k)
    small = np.abs(x) < thresh
    out[small] = k
    xs = x[~small]
    out[~small] = xs / (1.0 - np.exp(-xs / k))
    return out


def _vtrap_expm1(x: np.ndarray, k: float) -> np.ndarray:
    """``x / (exp(x/k) - 1)``; limit ``k`` as ``x -> 0``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    thresh = 1e-6 * abs(k)
    small = np.abs(x) < thresh
    out[small] = k
    xs = x[~small]
    out[~small] = xs / (np.exp(xs / k) - 1.0)
    return out


def _vtrap_1mexp_pos(x: np.ndarray, k: float) -> np.ndarray:
    """``x / (1 - exp(x/k))``; limit ``-k`` as ``x -> 0``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    thresh = 1e-6 * abs(k)
    small = np.abs(x) < thresh
    out[small] = -k
    xs = x[~small]
    out[~small] = xs / (1.0 - np.exp(xs / k))
    return out


def th_tauh(V: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(ah(V) + bh(V), 1e-12)


def th_taur(V: np.ndarray) -> np.ndarray:
    return 0.15 * (28.0 + np.exp(-(V + 25.0) / 10.5))


def alphah(V: np.ndarray) -> np.ndarray:
    return 0.128 * np.exp((-50.0 - V) / 18.0)


def alpham(V: np.ndarray) -> np.ndarray:
    return 0.32 * _vtrap_1mexp(54.0 + np.asarray(V, dtype=np.float64), 4.0)


def alphan(V: np.ndarray) -> np.ndarray:
    return 0.032 * _vtrap_1mexp(52.0 + np.asarray(V, dtype=np.float64), 5.0)


def alphap(V: np.ndarray) -> np.ndarray:
    return 3.209e-4 * _vtrap_1mexp(30.0 + np.asarray(V, dtype=np.float64), 9.0)


def betah(V: np.ndarray) -> np.ndarray:
    return 4.0 / (1.0 + np.exp((-27.0 - V) / 5.0))


def betan(V: np.ndarray) -> np.ndarray:
    return 0.5 * np.exp((-57.0 - V) / 40.0)


def betam(V: np.ndarray) -> np.ndarray:
    return 0.28 * _vtrap_expm1(np.asarray(V, dtype=np.float64) + 27.0, 5.0)


def betap(V: np.ndarray) -> np.ndarray:
    return -3.209e-4 * _vtrap_1mexp_pos(
        30.0 + np.asarray(V, dtype=np.float64), 9.0
    )


def Ggaba(V: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 + np.tanh(V / 4.0))


def stn_ainf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 45.0) / 14.7))


def stn_binf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 90.0) / 7.5))


def stn_cinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 30.6) / 5.0))


def stn_d1inf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 60.0) / 7.5))


def stn_d2inf(V: np.ndarray) -> np.ndarray:
    return _logistic(-(V - 0.1) / 0.02)


def stn_hinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 45.5) / 6.4))


def stn_minf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 40.0) / 8.0))


def stn_ninf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 41.0) / 14.0))


def stn_pinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 56.0) / 6.7))


def stn_qinf(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((V + 85.0) / 5.8))


def stn_rinf(V: np.ndarray) -> np.ndarray:
    return _logistic((V - 0.17) / 0.08)


def stn_taua(V: np.ndarray) -> np.ndarray:
    return 1.0 + 1.0 / (1.0 + np.exp(-(V + 40.0) / -0.5))


def stn_taub(V: np.ndarray) -> np.ndarray:
    return 200.0 / (
        np.exp(-(V + 60.0) / -30.0) + np.exp(-(V + 40.0) / 10.0)
    )


def stn_tauc(V: np.ndarray) -> np.ndarray:
    return 45.0 + 10.0 / (
        np.exp(-(V + 27.0) / -20.0) + np.exp(-(V + 50.0) / 15.0)
    )


def stn_taud1(V: np.ndarray) -> np.ndarray:
    return 400.0 + 500.0 / (
        np.exp(-(V + 40.0) / -15.0) + np.exp(-(V + 20.0) / 20.0)
    )


def stn_tauh(V: np.ndarray) -> np.ndarray:
    return 24.5 / (
        np.exp(-(V + 50.0) / -15.0) + np.exp(-(V + 50.0) / 16.0)
    )


def stn_taum(V: np.ndarray) -> np.ndarray:
    return 0.2 + 3.0 / (1.0 + np.exp(-(V + 53.0) / -0.7))


def stn_taun(V: np.ndarray) -> np.ndarray:
    return 11.0 / (
        np.exp(-(V + 40.0) / -40.0) + np.exp(-(V + 40.0) / 50.0)
    )


def stn_taup(V: np.ndarray) -> np.ndarray:
    return 5.0 + 0.33 / (
        np.exp(-(V + 27.0) / -10.0) + np.exp(-(V + 102.0) / 15.0)
    )


def stn_tauq(V: np.ndarray) -> np.ndarray:
    return 400.0 / (
        np.exp(-(V + 50.0) / -15.0) + np.exp(-(V + 50.0) / 16.0)
    )
