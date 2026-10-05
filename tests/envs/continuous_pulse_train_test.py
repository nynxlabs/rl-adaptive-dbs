"""Phase-continuous DBS pulse train across carried plant segments."""

from __future__ import annotations

import numpy as np

from envs.plant.dbs import ContinuousPulseTrain, create_dbs_current


def _stitch(train: ContinuousPulseTrain, segments: list[tuple[float, float, float]], tmax_ms: float) -> np.ndarray:
    """Concatenate the applied samples (1..N) of each segment after its sample 0."""
    out: list[np.ndarray] = []
    for freq, pw, amp in segments:
        idbs, state = train.segment(frequency_hz=freq, pulse_width_ms=pw, amplitude=amp, tmax_ms=tmax_ms)
        out.append(idbs if not out else idbs[1:])
        train.commit(state)
    return np.concatenate(out)


def test_constant_triple_matches_one_long_train() -> None:
    # 45 Hz ISI (2222 samples) does not divide 100 ms, so a per-segment restart would differ.
    stitched = _stitch(ContinuousPulseTrain(), [(45.0, 0.3, 300.0)] * 5, tmax_ms=100.0)
    whole = create_dbs_current(45.0, tmax_ms=500.0, pulse_width_ms=0.3, amplitude=300.0)
    np.testing.assert_array_equal(stitched, whole)


def test_pulse_crossing_boundary_finishes_at_its_own_amplitude() -> None:
    # 100 Hz ISI is 1000 samples; a 10.05 ms segment (1005 samples) ends 5 samples into
    # the 1.5 ms (150-sample) pulse at global 1000, so 144 samples spill over.
    train = ContinuousPulseTrain()
    idbs0, s0 = train.segment(frequency_hz=100.0, pulse_width_ms=1.5, amplitude=200.0, tmax_ms=10.05)
    train.commit(s0)
    idbs1, _ = train.segment(frequency_hz=100.0, pulse_width_ms=1.5, amplitude=50.0, tmax_ms=10.05)
    assert idbs0[1000:].tolist() == [200.0] * 6
    # Segment 1 starts at global 1005; the old pulse covers globals 1005..1149.
    assert idbs1[:145].tolist() == [200.0] * 145
    assert idbs1[145] == 0.0
    # Next onset at global 2000 uses the new amplitude.
    assert idbs1[2000 - 1005] == 50.0


def test_frequency_change_keeps_last_onset_and_off_restarts_train() -> None:
    train = ContinuousPulseTrain()
    _, s0 = train.segment(frequency_hz=40.0, pulse_width_ms=0.3, amplitude=300.0, tmax_ms=100.0)
    train.commit(s0)  # onsets at 0, 2500, 5000, 7500, 10000 (last = segment end)
    idbs1, s1 = train.segment(frequency_hz=50.0, pulse_width_ms=0.3, amplitude=300.0, tmax_ms=100.0)
    # Last onset at global 10000 (= local 0); 50 Hz ISI 2000 puts the next at local 2000.
    assert idbs1[1:30].tolist() == [300.0] * 29
    assert idbs1[30:2000].max() == 0.0
    assert idbs1[2000] == 300.0
    train.commit(s1)
    idbs2, s2 = train.segment(frequency_hz=0.0, pulse_width_ms=0.3, amplitude=300.0, tmax_ms=100.0)
    train.commit(s2)
    assert idbs2[1:].max() == 300.0  # only the tail of the pulse at the boundary
    idbs3, _ = train.segment(frequency_hz=50.0, pulse_width_ms=0.3, amplitude=300.0, tmax_ms=100.0)
    assert idbs3[0] == 300.0 and idbs3[2000] == 300.0
