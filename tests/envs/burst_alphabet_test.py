"""Mehregan burst pattern menu: exact mean rate, distinct patterns."""

from __future__ import annotations

import numpy as np
import pytest

from envs.mehregan.pattern_alternatives import BurstPatternAlphabet


def _onsets(trace: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(np.r_[0.0, np.asarray(trace)]) > 0)


@pytest.mark.parametrize(("mean_hz", "n_pulses"), [(45.0, 90), (30.0, 60)])
def test_exact_layout_holds_the_mean_rate(mean_hz: float, n_pulses: int) -> None:
    alphabet = BurstPatternAlphabet(mean_hz=mean_hz)
    assert alphabet.n_actions == 21
    patterns = [tuple(_onsets(alphabet.idbs_for_pattern(i))) for i in range(alphabet.n_actions)]
    assert {len(p) for p in patterns} == {n_pulses}
    assert len(set(patterns)) == alphabet.n_actions


def test_regular_pattern_has_no_step_boundary_double_pulse() -> None:
    alphabet = BurstPatternAlphabet(mean_hz=45.0)
    onsets = _onsets(alphabet.idbs_for_pattern(0)) * alphabet.dt_ms
    period_ms = 1000.0 / 45.0
    gap_to_next_step = alphabet.step_duration_s * 1000.0 - onsets[-1]
    assert gap_to_next_step == pytest.approx(period_ms, abs=0.5)  # grid rounding


def test_legacy_layout_unchanged() -> None:
    legacy = BurstPatternAlphabet(mean_hz=45.0, n_patterns=41, layout="legacy")
    assert legacy.n_actions == 41
    assert len(_onsets(legacy.idbs_for_pattern(1))) == 77
    np.testing.assert_array_equal(legacy.idbs_for_pattern(21), legacy.idbs_for_pattern(1))


def test_curated_menus_hold_the_rate_and_drop_beta_bursts() -> None:
    from envs.mehregan.pattern_alternatives import CuratedPatternAlphabet

    full = CuratedPatternAlphabet(mean_hz=45.0)
    nobeta = CuratedPatternAlphabet(mean_hz=45.0, exclude_beta_bursts=True, n_random=20)
    assert full.n_actions == 21
    assert nobeta.n_actions == 1 + 12 + 20  # sizes 2 / 3 (22.5 / 15 Hz bursts) dropped
    for alphabet in (full, nobeta):
        counts = {len(_onsets(alphabet.idbs_for_pattern(i))) for i in range(alphabet.n_actions)}
        assert counts == {90}
    np.testing.assert_array_equal(full.idbs_for_pattern(5), BurstPatternAlphabet(mean_hz=45.0).idbs_for_pattern(5))
