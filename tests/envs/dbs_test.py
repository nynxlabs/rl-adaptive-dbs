"""Public DBS helpers (no plant needed)."""

from __future__ import annotations

import numpy as np
import pytest

from envs.plant.dbs import stn_dbs_gains


def test_stn_dbs_gains_by_distance() -> None:
    np.testing.assert_array_equal(stn_dbs_gains(10, 1.0), np.ones(10))
    gains = stn_dbs_gains(10, 4.0)
    assert gains[0] == 1.0
    assert gains[-1] == pytest.approx(1 / 16)
    assert np.all(np.diff(gains) < 0)
    with pytest.raises(ValueError):
        stn_dbs_gains(10, 0.5)
