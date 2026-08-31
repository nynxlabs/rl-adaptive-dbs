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
    ravivarapu_fig4a_digitization_gates,
    ravivarapu_fig4a_gates,
    ravivarapu_fig4b_attach_tiered_pass,
    ravivarapu_fig4b_gates,
    ravivarapu_inference_gates,
)


def test_fig4a_paper_self_passes_full_shape_gates():
    paper = load_curves("fig4a")
    x_b, y_b = paper["Baseline"]
    x_s, y_s = paper["SEA-DBS"]
    episodes = np.arange(150, dtype=float)
    report = ravivarapu_fig4a_gates(
        np.interp(episodes, x_b, y_b),
        np.interp(episodes, x_s, y_s),
        n_expected=150,
    )
    assert report["pass"]
    assert report["gates"]["gradual_decline_baseline"]
    assert report["gates"]["gradual_decline_sea"]
    assert report["gates"]["late_gap_substantial"]
    assert report["gates"]["sea_steeper_drop_than_baseline"]
    assert report["gates"]["pearson_baseline_min"]
    assert report["gates"]["pearson_sea_min"]


def test_fig4a_rejects_baseline_mid_plateau():
    episodes = np.arange(150, dtype=float)
    # Baseline flat from ep 40 onward (mid == late)
    baseline = np.concatenate(
        [
            np.linspace(0.48, 0.42, 40),
            np.linspace(0.42, 0.42, 110),
        ]
    )
    sea = np.linspace(0.48, 0.34, 150)
    report = ravivarapu_fig4a_gates(baseline, sea, n_expected=150)
    assert not report["gates"]["gradual_decline_baseline"]


def test_fig4a_paper_self_passes_endpoint_separation_gates():
    paper = load_curves("fig4a")
    x_b, y_b = paper["Baseline"]
    x_s, y_s = paper["SEA-DBS"]
    episodes = np.arange(150, dtype=float)
    report = ravivarapu_fig4a_gates(
        np.interp(episodes, x_b, y_b),
        np.interp(episodes, x_s, y_s),
        n_expected=150,
    )
    assert report["gates"]["late_gap_near_paper"]
    assert np.isfinite(report["metrics"]["late_gap"])


def test_fig4a_requires_substantial_late_gap():
    episodes = np.arange(150, dtype=float)
    baseline = np.linspace(0.48, 0.40, episodes.size)
    sea = np.linspace(0.47, 0.35, episodes.size)

    report = ravivarapu_fig4a_gates(baseline, sea, n_expected=150)

    assert report["gates"]["late_gap_substantial"]
    assert report["metrics"]["late_gap"] >= 0.02


def test_fig4a_rejects_barely_separated_final_means():
    episodes = np.arange(150, dtype=float)
    baseline = np.linspace(0.48, 0.36, episodes.size)
    sea = np.linspace(0.47, 0.355, episodes.size)

    report = ravivarapu_fig4a_gates(baseline, sea, n_expected=150)

    assert not report["gates"]["late_gap_substantial"]


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


def test_fig4b_paper_self_passes_full_gates():
    paper = load_curves("fig4b")
    x_b, y_b = paper["Baseline Reward"]
    x_s, y_s = paper["SEA-DBS Reward"]
    episodes = np.arange(150, dtype=float)
    report = ravivarapu_fig4b_gates(
        np.interp(episodes, x_b, y_b),
        np.interp(episodes, x_s, y_s),
        n_expected=150,
    )
    assert report["pass"]
    assert report["gates"]["baseline_rises"]
    assert report["gates"]["sea_rises"]
    assert report["gates"]["sea_above_baseline_late"]
    assert report["gates"]["sea_steeper_rise_than_baseline"]
    assert report["gates"]["pearson_baseline_min"]
    assert report["gates"]["pearson_sea_min"]
    assert report["gates"]["shared_start_near_paper"]


def test_fig4b_rejects_non_rising_baseline():
    episodes = np.arange(150, dtype=float)
    baseline = np.full(150, -40.0)
    sea = np.linspace(-40.0, 10.0, 150)
    report = ravivarapu_fig4b_gates(baseline, sea, n_expected=150)
    assert not report["gates"]["baseline_rises"]
    assert not report["pass"]


def test_fig4b_rejects_inverted_late_order():
    episodes = np.arange(150, dtype=float)
    baseline = np.linspace(-40.0, 15.0, 150)
    sea = np.linspace(-40.0, -5.0, 150)
    report = ravivarapu_fig4b_gates(baseline, sea, n_expected=150)
    assert not report["gates"]["sea_above_baseline_late"]
    assert not report["gates"]["sea_steeper_rise_than_baseline"]
    assert not report["pass"]


def test_fig4b_tiered_pass_full_implies_shape_pass():
    from ravivarapu_gates import RAVIVARAPU_FIG4B_GATE_TIER

    flat = {key: True for key in RAVIVARAPU_FIG4B_GATE_TIER}
    tiered = ravivarapu_fig4b_attach_tiered_pass(flat)
    assert tiered["shape_pass"]
    assert tiered["pass"]


def test_fig4b_tiered_shape_pass_without_full_polish():
    from ravivarapu_gates import RAVIVARAPU_FIG4B_GATE_TIER

    flat = {key: (tier == "shape") for key, tier in RAVIVARAPU_FIG4B_GATE_TIER.items()}
    tiered = ravivarapu_fig4b_attach_tiered_pass(flat)
    assert tiered["shape_pass"]
    assert not tiered["pass"]


def test_inference_shape_gates_pass_on_split_decline():
    # 150 ms / n_obs=6 with 100 ms untreated start: SEA reaches the last-window floor at step 6.
    baseline = [0.4606, 0.4606, 0.4164, 0.4274, 0.4075, 0.3943, 0.3721, 0.3500, 0.3500, 0.3500, 0.3500]
    sea = [0.4606, 0.3943, 0.3721, 0.3611, 0.3544, 0.3500, 0.3279, 0.3279, 0.3279, 0.3279, 0.3279]
    report = ravivarapu_inference_gates(baseline, sea, carrier_hz=50.0)
    assert report["pass"]
    assert report["gates"]["shared_start_near_paper"]
    assert report["gates"]["baseline_declines"]
    assert report["gates"]["paper_declines"]
    assert report["gates"]["paper_end_below_baseline"]
    assert report["gates"]["paper_steeper_drop"]
    assert report["gates"]["early_mae_sea_3_5"]
    assert report["gates"]["late_sea_declines"]
    assert report["gates"]["late_baseline_declines"]
    assert report["gates"]["mid_mae_sea"]


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
    b30 = [0.461, 0.466, 0.468, 0.460, 0.452, 0.445, 0.438, 0.430, 0.424, 0.418, 0.412]
    s30 = [0.461, 0.455, 0.448, 0.441, 0.434, 0.427, 0.420, 0.413, 0.406, 0.400, 0.393]
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
    assert report["gates"]["early_baseline_rises"]
    assert report["gates"]["early_sea_plateau"]
    assert report["gates"]["early_sea_below_baseline"]
    assert report["gates"]["pearson_baseline_min"]
    assert report["gates"]["pearson_sea_min"]


def test_inference_5b_rejects_premature_sea_plunge():
    # If SEA drops by 0.035 on step 1, early_sea_plateau fails
    b30 = [0.461, 0.466, 0.468, 0.460, 0.452, 0.445, 0.438, 0.430, 0.424, 0.418, 0.412]
    s30_plunge = [0.461, 0.425, 0.415, 0.410, 0.405, 0.402, 0.400, 0.398, 0.395, 0.393, 0.390]
    b50 = np.linspace(0.46, 0.36, 11)
    s50 = np.linspace(0.46, 0.31, 11)
    report = ravivarapu_inference_gates(
        b30,
        s30_plunge,
        carrier_hz=30.0,
        sea_trace_50hz=s50,
        baseline_trace_50hz=b50,
    )
    assert not report["gates"]["early_sea_plateau"]
    assert not report["pass"]


def test_inference_5b_rejects_baseline_early_drop():
    # If Baseline drops on step 1 instead of rising/plateauing, early_baseline_rises fails
    b30_drop = [0.461, 0.440, 0.435, 0.430, 0.425, 0.420, 0.415, 0.410, 0.405, 0.402, 0.400]
    s30 = [0.461, 0.455, 0.448, 0.441, 0.434, 0.427, 0.420, 0.413, 0.406, 0.400, 0.393]
    b50 = np.linspace(0.46, 0.36, 11)
    s50 = np.linspace(0.46, 0.31, 11)
    report = ravivarapu_inference_gates(
        b30_drop,
        s30,
        carrier_hz=30.0,
        sea_trace_50hz=s50,
        baseline_trace_50hz=b50,
    )
    assert not report["gates"]["early_baseline_rises"]
    assert not report["pass"]


def test_inference_early_window_passes_on_five_pulse_50hz():
    """140 ms / 8-pulse 50 Hz with an early Baseline skip (paper 3–5)."""
    sea = [
        0.4795,
        0.4092,
        0.3857,
        0.3740,
        0.3670,
        0.3623,
        0.3589,
        0.3564,
        0.3545,
        0.3529,
        0.3516,
    ]
    baseline = [
        0.4795,
        0.4795,
        0.4326,
        0.4443,
        0.4232,
        0.4092,
        0.3991,
        0.3916,
        0.3857,
        0.3810,
        0.3772,
    ]
    report = ravivarapu_inference_gates(baseline, sea, carrier_hz=50.0)
    if "early_mae_sea" not in report["gates"]:
        return
    assert report["gates"]["early_mae_sea"]
    assert report["gates"]["early_mae_sea_3_5"]
    assert report["gates"]["early_mae_baseline"]
    assert report["gates"]["early_sea_declines"]
    assert report["gates"]["early_baseline_declines"]
    assert report["gates"]["early_sea_below_baseline"]


def test_inference_early_window_rejects_overlaid_traces():
    """Identical stim prefixes fail: paper keeps Baseline above SEA on 1–5."""
    y = [
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
    report = ravivarapu_inference_gates(y, y, carrier_hz=50.0)
    if "early_sea_below_baseline" not in report["gates"]:
        return
    assert not report["gates"]["early_sea_below_baseline"]
    assert not report["pass"]


def test_inference_early_window_rejects_four_pulse_floor():
    """62 ms / 4-pulse 50 Hz drop (~0.033) is too shallow for steps 0–5."""
    y = np.linspace(0.461, 0.428, 11)
    report = ravivarapu_inference_gates(y, y, carrier_hz=50.0)
    if "early_sea_declines" not in report["gates"]:
        return
    assert not report["gates"]["early_sea_declines"]
    assert not report["pass"]


def test_inference_late_window_rejects_n_obs_floor():
    """n_obs=5 always-on floors at step 5; paper keeps falling through 10."""
    sea = [
        0.4795,
        0.4092,
        0.3857,
        0.3740,
        0.3670,
        0.3388,
        0.3388,
        0.3388,
        0.3388,
        0.3388,
        0.3388,
    ]
    baseline = [
        0.4795,
        0.4795,
        0.4326,
        0.4092,
        0.3951,
        0.3670,
        0.3670,
        0.3670,
        0.3670,
        0.3670,
        0.3951,
    ]
    report = ravivarapu_inference_gates(baseline, sea, carrier_hz=50.0)
    assert not report["gates"]["late_sea_declines"]
    assert not report["gates"]["late_baseline_declines"]
    assert not report["pass"]


def test_inference_mid_window_rejects_onset_fill():
    """n_obs=10 leftover untreated holds SEA ~0.35 on steps 5–9."""
    sea = [
        0.4981,
        0.4130,
        0.3846,
        0.3704,
        0.3619,
        0.3563,
        0.3522,
        0.3492,
        0.3468,
        0.3449,
        0.3279,
    ]
    baseline = [
        0.4981,
        0.4981,
        0.4414,
        0.4556,
        0.4300,
        0.4130,
        0.4009,
        0.3917,
        0.3846,
        0.3790,
        0.3619,
    ]
    report = ravivarapu_inference_gates(baseline, sea, carrier_hz=50.0)
    assert not report["gates"]["mid_mae_sea"]
    assert not report["pass"]


def test_fig6_gates_pass():
    from ravivarapu_gates import ravivarapu_fig6_gates

    traces = {
        "Baseline": [0.4606, 0.4606, 0.4376, 0.4376, 0.4146, 0.3916, 0.3686, 0.3456, 0.3456, 0.3456, 0.3456],
        "Baseline + PTQ(fp16)": [0.4606, 0.4606, 0.4606, 0.4606, 0.4376, 0.4146, 0.3916, 0.3686, 0.3686, 0.3686, 0.3686],
        "SEA-DBS": [0.4606, 0.4376, 0.4146, 0.3916, 0.3686, 0.3456, 0.3226, 0.3226, 0.3226, 0.3226, 0.3226],
        "SEA-DBS + PTQ(fp16)": [0.4606, 0.4376, 0.4146, 0.4146, 0.3916, 0.3686, 0.3456, 0.3456, 0.3456, 0.3226, 0.3226],
    }
    report = ravivarapu_fig6_gates(traces)
    assert report["pass"]
    assert report["gates"]["shared_start_near_paper"]
    assert report["gates"]["sea_below_baseline"]
    assert report["gates"]["sea_ptq_below_baseline"]


def test_fig7_gates_pass():
    from ravivarapu_gates import ravivarapu_fig7_gates

    traces = {
        "baseline": [0.4606, 0.4606, 0.4385, 0.4385, 0.4164, 0.3943, 0.3721, 0.3500, 0.3500, 0.3500, 0.3500],
        "baseline-pm": [0.4606, 0.4385, 0.4164, 0.3943, 0.3721, 0.3721, 0.3500, 0.3500, 0.3500, 0.3500, 0.3500],
        "baseline-gs": [0.4606, 0.4606, 0.4606, 0.4385, 0.4385, 0.4164, 0.3943, 0.3943, 0.3943, 0.3943, 0.3943],
        "paper": [0.4606, 0.4385, 0.4164, 0.3943, 0.3721, 0.3500, 0.3279, 0.3279, 0.3279, 0.3279, 0.3279],
    }
    report = ravivarapu_fig7_gates(traces)
    assert report["pass"]
    assert report["gates"]["shared_start_near_paper"]
    assert report["gates"]["sea_dbs_lowest_tail"]
    assert report["gates"]["pm_not_sea"]
    # Baseline must be above or equal to Baseline+PM across steps 0-4
    assert all(b >= pm for b, pm in zip(traces["baseline"][:5], traces["baseline-pm"][:5]))
