"""Fig 5a spike count excludes the stimulated STN and GPi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_adaptive_dbs.panel import load_script_module

_PLOT = Path(__file__).resolve().parents[2] / "scripts" / "figures" / "papers" / "nguyen" / "5" / "plot.py"


def test_network_spikes_drops_stn_and_gpi_columns() -> None:
    fig5 = load_script_module("nguyen_fig5_plot_test", _PLOT)
    by_pop = [[1, 2, 3, 4, 100, 5, 6, 7], [0, 0, 0, 0, 50, 0, 1, 0]]
    np.testing.assert_allclose(fig5.network_spikes({"episode_spikes_per_step_by_population": by_pop}), [22, 0])


def test_network_spikes_rejects_gpi_only_series() -> None:
    fig5 = load_script_module("nguyen_fig5_plot_test", _PLOT)
    with pytest.raises(ValueError, match="8 CBGT populations"):
        fig5.network_spikes({"episode_spikes_per_step_by_population": [[3.0], [4.0]]})
