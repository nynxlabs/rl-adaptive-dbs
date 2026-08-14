"""Tests for Ravivarapu Fig. 4a digitization gates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
sys.path.insert(0, str(_DIG))

from ravivarapu_gates import (  # noqa: E402
    load_curves,
    ravivarapu_fig4a_attach_tiered_pass,
    ravivarapu_fig4a_gates,
    ravivarapu_fig4a_digitization_gates,
    ravivarapu_inference_gates,
)


def test_fig4a_paper_self_passes_full_shape_gates():
    paper = load_curves("fig4a")
    x_b, y_b = paper["Baseline"]
    x_s, y_s = paper["SEA-DBS"]
    episodes = np.arange(150, dtype=float)
    report = ravivarapu_fig4a_digitization_gates(
        np.interp(episodes, x_b, y_b),
        np.interp(episodes, x_s, y_s),
        n_expected=150,
    )
    assert report["pass"]
    assert report["gates"]["progressive_decline_baseline"]
    assert report["gates"]["progressive_decline_sea"]
    assert report["gates"]["gap_widens_mid_to_late"]


def test_fig4a_rejects_baseline_mid_plateau_and_gap_closing():
    episodes = np.arange(150, dtype=float)
    baseline = np.concatenate(
        [
            np.linspace(0.46, 0.467, 40),
            np.linspace(0.467, 0.453, 80),
            np.linspace(0.453, 0.424, 30),
        ]
    )
    sea = np.concatenate(
        [
            np.linspace(0.453, 0.454, 40),
            np.linspace(0.454, 0.393, 80),
            np.linspace(0.390, 0.385, 30),
        ]
    )
    report = ravivarapu_fig4a_digitization_gates(baseline, sea, n_expected=150)
    assert not report["gates"]["progressive_decline_baseline"]


def test_fig4a_paper_self_passes_endpoint_separation_gates():
    paper = load_curves("fig4a")
    x_b, y_b = paper["Baseline"]
    x_s, y_s = paper["SEA-DBS"]
    episodes = np.arange(150, dtype=float)
    report = ravivarapu_fig4a_digitization_gates(
        np.interp(episodes, x_b, y_b),
        np.interp(episodes, x_s, y_s),
        n_expected=150,
    )
    assert report["gates"]["final_window_gap_near_paper"]
    assert np.isfinite(report["metrics"]["endpoint_gap"])


def test_fig4a_requires_substantial_final_mean_gap():
    episodes = np.arange(150, dtype=float)
    baseline = np.linspace(0.48, 0.40, episodes.size)
    sea = np.linspace(0.47, 0.35, episodes.size)

    report = ravivarapu_fig4a_gates(baseline, sea, n_expected=150)

    assert report["gates"]["final_window_gap_substantial"]
    assert report["metrics"]["final_window_gap"] >= 0.03


def test_fig4a_rejects_barely_separated_final_means():
    episodes = np.arange(150, dtype=float)
    baseline = np.linspace(0.48, 0.38, episodes.size)
    sea = np.linspace(0.47, 0.35, episodes.size)

    report = ravivarapu_fig4a_gates(baseline, sea, n_expected=150)

    assert not report["gates"]["final_window_gap_substantial"]


def test_fig4a_tiered_pass_full_implies_shape_pass():
    from ravivarapu_gates import RAVIVARAPU_FIG4A_GATE_TIER

    flat = {key: True for key in RAVIVARAPU_FIG4A_GATE_TIER}
    tiered = ravivarapu_fig4a_attach_tiered_pass(flat)
    assert tiered["shape_pass"]
    assert tiered["pass"]


def test_fig4a_tiered_shape_pass_without_full_polish():
    from ravivarapu_gates import RAVIVARAPU_FIG4A_GATE_TIER

    flat = {key: (tier == "shape") for key, tier in RAVIVARAPU_FIG4A_GATE_TIER.items()}
    tiered = ravivarapu_fig4a_attach_tiered_pass(flat)
    assert tiered["shape_pass"]
    assert not tiered["pass"]


def test_inference_shape_gates_pass_on_split_decline():
    baseline = np.linspace(0.46, 0.36, 11)
    sea = np.linspace(0.46, 0.31, 11)
    report = ravivarapu_inference_gates(baseline, sea, carrier_hz=50.0)
    assert report["pass"]
    assert report["gates"]["baseline_declines"]
    assert report["gates"]["paper_declines"]
    assert report["gates"]["paper_end_below_baseline"]
    assert report["gates"]["paper_steeper_drop"]


def test_inference_shape_gates_any_net_drop_counts():
    """Modest 50 Hz always-on drop (~0.033) is still a decline."""
    y = np.linspace(0.461, 0.428, 11)
    report = ravivarapu_inference_gates(y, y, carrier_hz=50.0)
    assert report["gates"]["baseline_declines"]
    assert report["gates"]["paper_declines"]
    assert not report["gates"]["paper_end_below_baseline"]
    assert not report["gates"]["paper_steeper_drop"]
    assert not report["pass"]


def test_inference_shape_gates_reject_rise():
    y = np.linspace(0.461, 0.468, 11)
    report = ravivarapu_inference_gates(y, y, carrier_hz=30.0)
    assert not report["gates"]["baseline_declines"]
    assert not report["gates"]["paper_declines"]
    assert not report["pass"]


def test_inference_5b_weaker_than_50hz():
    b30 = np.linspace(0.46, 0.40, 11)
    s30 = np.linspace(0.46, 0.39, 11)
    b50 = np.linspace(0.46, 0.36, 11)
    s50 = np.linspace(0.46, 0.31, 11)
    report = ravivarapu_inference_gates(
        b30,
        s30,
        carrier_hz=30.0,
        sea_trace_50hz=s50,
        baseline_trace_50hz=b50,
    )
    assert report["pass"]
    assert report["gates"]["weaker_than_50hz_sea"]
    assert report["gates"]["weaker_than_50hz_baseline"]


def test_inference_early_window_passes_on_five_pulse_50hz():
    """100 ms / 5-pulse 50 Hz floor matches digitized steps 0–5 to 0.03 MAE."""
    sea = [
        0.4606,
        0.4241,
        0.4120,
        0.4059,
        0.4023,
        0.3877,
        0.3877,
        0.3877,
        0.3877,
        0.3877,
        0.3877,
    ]
    baseline = [
        0.4606,
        0.4241,
        0.4120,
        0.4059,
        0.4023,
        0.4030,
        0.4000,
        0.4000,
        0.4000,
        0.4000,
        0.4000,
    ]
    report = ravivarapu_inference_gates(baseline, sea, carrier_hz=50.0)
    if "early_mae_sea" not in report["gates"]:
        return
    assert report["gates"]["early_mae_sea"]
    assert report["gates"]["early_mae_baseline"]
    assert report["gates"]["early_sea_declines"]
    assert report["gates"]["early_baseline_declines"]
    assert report["gates"]["early_sea_below_baseline"]


def test_inference_early_window_rejects_four_pulse_floor():
    """62 ms / 4-pulse 50 Hz drop (~0.033) is too shallow for steps 0–5."""
    y = np.linspace(0.461, 0.428, 11)
    report = ravivarapu_inference_gates(y, y, carrier_hz=50.0)
    if "early_sea_declines" not in report["gates"]:
        return
    assert not report["gates"]["early_sea_declines"]
    assert not report["pass"]
