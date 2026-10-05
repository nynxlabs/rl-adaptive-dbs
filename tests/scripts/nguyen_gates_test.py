"""Tests for Nguyen digitization-anchored paper gates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
sys.path.insert(0, str(_DIG))

from nguyen_gates import (  # noqa: E402
    curves_path,
    fig3_gates,
    fig5_spikes_energy_gates,
    fig6_power_gates,
    fig6_training_gates,
    fig7_eval_gates,
    load_curves,
)

ARTIFACT = Path("artifacts/figures/papers/nguyen/paper_digitization")


@pytest.mark.skipif(not (ARTIFACT / "curves_fig4_reward.json").exists(), reason="no digitization")
def test_load_curves_fig4_reward():
    paper = load_curves("fig4_reward")
    assert "Smoothed" in paper or "Raw" in paper


@pytest.mark.skipif(not (ARTIFACT / "samples.json").exists(), reason="no fig3 samples")
def test_fig3_gates_on_cached_samples():
    samples = __import__("json").loads(
        Path("artifacts/figures/papers/nguyen/3/samples.json").read_text()
    )
    report = fig3_gates(samples)
    assert report["gates"]["ordering_pd_on_above_pd_off"]
    assert report["pass"]


@pytest.mark.skipif(not (ARTIFACT / "curves_fig5_spikes.json").exists(), reason="no digitization")
def test_fig5_paper_self_consistent():
    paper_s = load_curves("fig5_spikes")
    paper_e = load_curves("fig5_energy")
    sx, sy = paper_s["Spike Count"]
    ex, ey = paper_e["Smoothed"] if "Smoothed" in paper_e else paper_e["Raw"]
    n = int(min(sx[-1], ex[-1])) + 1
    spikes = np.interp(np.arange(n), sx, sy)
    energies = np.interp(np.arange(n), ex, ey)
    report = fig5_spikes_energy_gates(spikes, energies)
    # The paper's own curves must pass required and report-only rows alike.
    report["gates"] = {**report["gates"], **report["report"]}
    assert "spike_mean_near_paper" in report["report"]
    assert report["gates"]["spike_mean_near_paper"]
    assert report["gates"]["spike_stays_near_800"]
    assert report["gates"]["energy_early_near_paper"]
    assert report["gates"]["energy_mid_near_paper"]
    assert report["gates"]["energy_late_near_paper"]
    assert report["gates"]["energy_mean_near_paper"]
    assert report["gates"]["energy_mid_ramp_near_paper"]
    assert report["gates"]["energy_trend_near_paper"]
    assert report["gates"]["energy_monotonic_rise"]
    assert report["gates"]["energy_late_above_early"]
    assert report["pass"]


@pytest.mark.skipif(not (ARTIFACT / "curves_fig6_power.json").exists(), reason="no digitization")
def test_fig6_paper_self_consistent():
    ab = load_curves("fig6_power")
    amp = load_curves("fig6_amp")
    freq = load_curves("fig6_freq")
    pw = load_curves("fig6_pw")
    n = 501
    abx, aby = ab["Smoothed"] if "Smoothed" in ab else ab["Raw"]
    ampx, ampy = amp["Smoothed"] if "Smoothed" in amp else amp["Raw"]
    freqx, freqy = freq["Smoothed"] if "Smoothed" in freq else freq["Raw"]
    pwx, pwy = pw["Smoothed"] if "Smoothed" in pw else pw["Raw"]
    episodes = np.arange(n, dtype=float)
    ab_interp = np.interp(episodes, abx, aby)
    amp_interp = np.interp(episodes, ampx, ampy)
    freq_interp = np.interp(episodes, freqx, freqy)
    pw_interp = np.interp(episodes, pwx, pwy)

    power_rep = fig6_power_gates(ab_interp)
    assert power_rep["gates"]["alpha_beta_early_above_theta"]
    assert power_rep["gates"]["alpha_beta_early_near_paper"]
    assert power_rep["gates"]["alpha_beta_mid_near_paper"]
    assert power_rep["gates"]["alpha_beta_drops_by_100"]
    assert power_rep["gates"]["alpha_beta_post100_below_theta"]
    assert power_rep["gates"]["alpha_beta_post100_near_paper"]
    assert power_rep["gates"]["alpha_beta_late_below_theta"]
    assert power_rep["gates"]["alpha_beta_late_near_paper"]
    assert power_rep["gates"]["alpha_beta_mean_near_paper"]
    assert power_rep["gates"]["alpha_beta_monotonic_drop"]
    assert power_rep["gates"]["alpha_beta_trend_near_paper"]
    assert power_rep["gates"]["alpha_beta_drop_magnitude_near_paper"]
    assert power_rep["gates"]["alpha_beta_series_has_variance"]
    assert power_rep["gates"]["alpha_beta_late_stable"]
    assert power_rep["pass"]

    report = fig6_training_gates(
        ab_interp,
        amp_interp,
        freq_interp,
        pw_interp,
    )
    report["gates"] = {**report["gates"], **report["report"]}
    assert "pw_late_near_paper" in report["report"]
    assert report["gates"]["freq_late_stable"]
    assert report["gates"]["alpha_beta_decreases_like_paper"]
    assert report["gates"]["alpha_beta_late_below_theta"]
    assert report["gates"]["alpha_beta_late_near_paper"]
    assert report["gates"]["amp_late_near_paper"]
    assert report["gates"]["freq_late_near_paper"]
    assert report["gates"]["pw_late_near_paper"]
    assert report["gates"]["late_params_stable"]
    assert report["pass"]


@pytest.mark.skipif(not (ARTIFACT / "curves_fig7.json").exists(), reason="no digitization")
def test_fig7_paper_self_consistent():
    paper = load_curves("fig7")
    ax, ay = paper["average"]
    sort_idx = np.argsort(ax)
    ax, ay = ax[sort_idx], ay[sort_idx]
    steps = np.arange(26)
    interp_y = np.interp(steps, ax, ay)
    trajectories = [interp_y.tolist() for _ in range(3)]
    report = fig7_eval_gates(trajectories)
    assert report["gates"]["eval_protocol_ok"]
    assert report["gates"]["overall_mean_near_paper"]
    assert report["pass"]


def test_curves_path_stems():
    assert curves_path("fig4_reward").name == "curves_fig4_reward.json"
