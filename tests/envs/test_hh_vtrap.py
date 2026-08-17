"""Hodgkin–Huxley vtrap: finite rates at the classic 0/0 voltages."""

from __future__ import annotations

import numpy as np
import pytest

from envs.plant.network import gating as g


def test_striatal_rates_finite_at_singularities() -> None:
    assert np.isfinite(g.alpham(np.array([-54.0]))[0])
    assert np.isfinite(g.betam(np.array([-27.0]))[0])
    assert np.isfinite(g.alphan(np.array([-52.0]))[0])
    assert np.isfinite(g.alphap(np.array([-30.0]))[0])
    assert np.isfinite(g.betap(np.array([-30.0]))[0])


def test_vtrap_limits_match_lhopital() -> None:
    # alpham: 0.32 * (V+54)/(1-exp(-(V+54)/4)) → 0.32 * 4 as V → -54
    assert g.alpham(np.array([-54.0]))[0] == pytest.approx(0.32 * 4.0)
    # betam: 0.28 * (V+27)/(exp((V+27)/5)-1) → 0.28 * 5 as V → -27
    assert g.betam(np.array([-27.0]))[0] == pytest.approx(0.28 * 5.0)
