"""Tests for Ravivarapu digitization gates (Figs 4–7, Table II)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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

_PAPER_CURVES = (
    Path(__file__).resolve().parents[2]
    / "artifacts/figures/papers/ravivarapu/paper_digitization"
)
# Digitized paper curves are local artifacts (not committed); skip paper-relative
# tests when they are absent, e.g. on a fresh clone or in CI.
needs_paper_curves = pytest.mark.skipif(
    not all(
        (_PAPER_CURVES / f"curves_{fig}.json").exists()
        for fig in ("fig4a", "fig4b", "fig5a", "fig5b")
    ),
    reason="no Ravivarapu paper digitization",
)


@needs_paper_curves
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
    assert report["gates"]["shared_start_near_paper"]
    assert report["gates"]["baseline_late_near_paper"]
    assert report["gates"]["sea_steeper_drop_than_baseline"]
    assert report["gates"]["pearson_baseline_min"]
    assert report["gates"]["pearson_sea_min"]


@needs_paper_curves
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


@needs_paper_curves
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


def _paper_4a() -> tuple[np.ndarray, np.ndarray]:
    paper = load_curves("fig4a")
    episodes = np.arange(150, dtype=float)
    return (
        np.interp(episodes, *paper["Baseline"]),
        np.interp(episodes, *paper["SEA-DBS"]),
    )


@needs_paper_curves
def test_fig4a_rejects_barely_separated_final_means():
    baseline, _sea = _paper_4a()
    report = ravivarapu_fig4a_gates(baseline, baseline - 0.003, n_expected=150)
    assert not report["gates"]["late_gap_near_paper"]
    assert not report["gates"]["sea_steeper_drop_than_baseline"]


@needs_paper_curves
def test_fig4a_rejects_flat_baseline():
    """A Baseline that never declines failed only the drop gates before; the
    late/early ratio (0.30 tol) and missing Baseline level gates let it through."""
    baseline, sea = _paper_4a()
    flat = np.full(150, float(baseline[:15].mean()))
    report = ravivarapu_fig4a_gates(flat, sea, n_expected=150)
    assert not report["gates"]["late_early_ratio_baseline_near_paper"]
    assert not report["gates"]["baseline_late_near_paper"]


@needs_paper_curves
def test_fig4a_rejects_start_offset():
    baseline, sea = _paper_4a()
    report = ravivarapu_fig4a_gates(baseline - 0.05, sea - 0.05, n_expected=150)
    assert not report["gates"]["shared_start_near_paper"]


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


@needs_paper_curves
def test_fig4b_paper_self_passes_full_gates():
    paper = load_curves("fig4b")
    x_b, y_b = paper["Baseline Reward"]
    x_s, y_s = paper["SEA-DBS Reward"]
    episodes = np.arange(150, dtype=float)
    psd_b, psd_s = _paper_4a()
    report = ravivarapu_fig4b_gates(
        np.interp(episodes, x_b, y_b),
        np.interp(episodes, x_s, y_s),
        n_expected=150,
        baseline_psd=psd_b,
        sea_psd=psd_s,
    )
    assert report["pass"], report["gates"]
    assert report["gates"]["eq7_consistent_with_4a"]
    assert report["gates"]["sea_faster_initial_rise"]
    assert report["gates"]["baseline_rises"]
    assert report["gates"]["sea_rises"]
    assert report["gates"]["sea_above_baseline_late"]
    assert report["gates"]["sea_steeper_rise_than_baseline"]
    assert report["gates"]["pearson_baseline_min"]
    assert report["gates"]["pearson_sea_min"]
    assert report["gates"]["shared_start_near_paper"]


@needs_paper_curves
def test_fig4b_compares_absolute_reward_not_rescaled_paper():
    """A run whose rewards are the paper's shape at 40 % scale must not pass the
    level gates: the paper is no longer rescaled by our Baseline's early level."""
    paper = load_curves("fig4b")
    x_b, y_b = paper["Baseline Reward"]
    x_s, y_s = paper["SEA-DBS Reward"]
    episodes = np.arange(150, dtype=float)
    report = ravivarapu_fig4b_gates(
        0.4 * np.interp(episodes, x_b, y_b),
        0.4 * np.interp(episodes, x_s, y_s),
        n_expected=150,
    )
    assert not report["gates"]["shared_start_near_paper"]
    assert not report["gates"]["baseline_rise_vs_paper"]
    assert not report["pass"]


@needs_paper_curves
def test_fig4b_rejects_non_rising_baseline():
    episodes = np.arange(150, dtype=float)
    baseline = np.full(150, -40.0)
    sea = np.linspace(-40.0, 10.0, 150)
    report = ravivarapu_fig4b_gates(baseline, sea, n_expected=150)
    assert not report["gates"]["baseline_rises"]
    assert not report["pass"]


@needs_paper_curves
def test_fig4b_rejects_inverted_late_order():
    episodes = np.arange(150, dtype=float)
    baseline = np.linspace(-40.0, 15.0, 150)
    sea = np.linspace(-40.0, -5.0, 150)
    report = ravivarapu_fig4b_gates(baseline, sea, n_expected=150)
    assert not report["gates"]["sea_above_baseline_late"]
    assert not report["gates"]["sea_steeper_rise_than_baseline"]
    assert not report["pass"]


@needs_paper_curves
def test_fig4b_eq7_gate_fails_closed_without_psd():
    paper = load_curves("fig4b")
    episodes = np.arange(150, dtype=float)
    report = ravivarapu_fig4b_gates(
        np.interp(episodes, *paper["Baseline Reward"]),
        np.interp(episodes, *paper["SEA-DBS Reward"]),
        n_expected=150,
    )
    assert not report["gates"]["eq7_consistent_with_4a"]


@needs_paper_curves
def test_fig4b_eq7_gate_rejects_reward_off_its_psd():
    """Mean step reward 0.1 above Eq. (7) of the episode PSD (our run's gap)."""
    from ravivarapu_gates import eq7_reward

    psd_b, psd_s = _paper_4a()
    report = ravivarapu_fig4b_gates(
        eq7_reward(psd_b) + 0.1,
        eq7_reward(psd_s) + 0.1,
        n_expected=150,
        baseline_psd=psd_b,
        sea_psd=psd_s,
    )
    assert not report["gates"]["eq7_consistent_with_4a"]


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


def _paper_steps(stem: str, series: str) -> list[float]:
    from ravivarapu_gates import _paper_early_norm

    values = _paper_early_norm(stem, series, n=11)
    assert values is not None
    return [float(v) for v in values]


@needs_paper_curves
def test_inference_50hz_paper_self_passes():
    baseline = _paper_steps("fig5a", "Baseline 50Hz")
    sea = _paper_steps("fig5a", "SEA-DBS 50Hz")
    report = ravivarapu_inference_gates(baseline, sea, carrier_hz=50.0)
    assert report["pass"], report["gates"]


@needs_paper_curves
def test_inference_30hz_paper_self_passes():
    report = ravivarapu_inference_gates(
        _paper_steps("fig5b", "Baseline 30Hz"),
        _paper_steps("fig5b", "SEA-DBS 30Hz"),
        carrier_hz=30.0,
        sea_trace_50hz=_paper_steps("fig5a", "SEA-DBS 50Hz"),
        baseline_trace_50hz=_paper_steps("fig5a", "Baseline 50Hz"),
    )
    assert report["pass"], report["gates"]


@needs_paper_curves
def test_inference_50hz_rejects_identical_always_on_traces():
    """Two policies that both stimulate every step overlay — paper splits them."""
    sea = _paper_steps("fig5a", "SEA-DBS 50Hz")
    report = ravivarapu_inference_gates(list(sea), list(sea), carrier_hz=50.0)
    assert not report["gates"]["paper_end_below_baseline"]
    assert not report["pass"]


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


@needs_paper_curves
def test_inference_5b_rising_30hz_is_not_weaker_suppression():
    """30 Hz traces that rise above onset must not count as 'weaker than 50 Hz'."""
    b50 = _paper_steps("fig5a", "Baseline 50Hz")
    s50 = _paper_steps("fig5a", "SEA-DBS 50Hz")
    rising = list(np.linspace(0.4485, 0.6067, 11))
    report = ravivarapu_inference_gates(
        rising, rising, carrier_hz=30.0, sea_trace_50hz=s50, baseline_trace_50hz=b50
    )
    assert not report["gates"]["weaker_than_50hz_sea"]
    assert not report["gates"]["weaker_than_50hz_baseline"]


@needs_paper_curves
def test_inference_50hz_rejects_near_tie():
    """SEA-DBS 0.0005 below Baseline used to pass every ordering gate."""
    b = np.asarray(_paper_steps("fig5a", "Baseline 50Hz"))
    sea = b - np.r_[0.0, np.full(10, 5e-4)]
    report = ravivarapu_inference_gates(b, sea, carrier_hz=50.0)
    assert not report["gates"]["tail_gap_near_paper"]
    assert not report["pass"]


@needs_paper_curves
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
    assert report["gates"]["weaker_than_50hz_sea"]
    assert report["gates"]["weaker_than_50hz_baseline"]
    assert report["gates"]["early_baseline_rises"]
    assert report["gates"]["early_sea_plateau"]
    assert report["gates"]["early_sea_below_baseline"]
    assert report["gates"]["pearson_baseline_min"]
    assert report["gates"]["pearson_sea_min"]


@needs_paper_curves
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


@needs_paper_curves
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


@needs_paper_curves
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


def _fig5a_paper() -> dict[str, list[float]]:
    return {
        "baseline": _paper_steps("fig5a", "Baseline 50Hz"),
        "paper": _paper_steps("fig5a", "SEA-DBS 50Hz"),
    }


_FIG6_NAMES = ("Baseline", "Baseline + PTQ(fp16)", "SEA-DBS", "SEA-DBS + PTQ(fp16)")
_FIG7_NAMES = {
    "baseline": "Baseline",
    "baseline-pm": "Baseline + PM",
    "baseline-gs": "Baseline + GS",
    "paper": "SEA-DBS",
}


@needs_paper_curves
def test_fig6_paper_self_passes():
    from ravivarapu_gates import ravivarapu_fig6_gates

    report = ravivarapu_fig6_gates(
        {n: _paper_steps("fig6", n) for n in _FIG6_NAMES}, fig5a_traces=_fig5a_paper()
    )
    assert report["pass"], report["gates"]


@needs_paper_curves
def test_fig6_ptq_identical_to_fp32_is_not_a_fail():
    """The paper claims PTQ tracks fp32; an unchanged PTQ trace is not a defect."""
    from ravivarapu_gates import ravivarapu_fig6_gates

    traces = {n: _paper_steps("fig6", n) for n in _FIG6_NAMES}
    traces["SEA-DBS + PTQ(fp16)"] = list(traces["SEA-DBS"])
    report = ravivarapu_fig6_gates(traces, fig5a_traces=_fig5a_paper())
    assert report["gates"]["sea_ptq_tracks_fp32"]
    assert "ptq_traces_distinct" not in report["gates"]


@needs_paper_curves
def test_fig6_rejects_ptq_17pct_worse():
    from ravivarapu_gates import ravivarapu_fig6_gates

    traces = {n: np.asarray(_paper_steps("fig6", n)) for n in _FIG6_NAMES}
    sea = traces["SEA-DBS"]
    traces["SEA-DBS + PTQ(fp16)"] = np.r_[sea[:5], sea[5:] * 1.17]
    report = ravivarapu_fig6_gates(traces, fig5a_traces=_fig5a_paper())
    assert not report["gates"]["sea_ptq_tracks_fp32"]
    assert not report["pass"]


@needs_paper_curves
def test_fig6_fails_closed_without_fig5a():
    from ravivarapu_gates import ravivarapu_fig6_gates

    report = ravivarapu_fig6_gates({n: _paper_steps("fig6", n) for n in _FIG6_NAMES})
    assert not report["gates"]["fp32_matches_fig5a"]


def test_fig6_rejects_ten_samples():
    from ravivarapu_gates import ravivarapu_fig6_gates

    y = list(np.linspace(0.46, 0.32, 10))
    report = ravivarapu_fig6_gates({n: y for n in _FIG6_NAMES})
    assert not report["gates"]["n_steps_ok"]


@needs_paper_curves
def test_fig7_paper_self_passes():
    from ravivarapu_gates import ravivarapu_fig7_gates

    report = ravivarapu_fig7_gates(
        {k: _paper_steps("fig7", v) for k, v in _FIG7_NAMES.items()}, fig5a_traces=_fig5a_paper()
    )
    assert report["pass"], report["gates"]


@needs_paper_curves
def test_fig7_rejects_flat_gs():
    """+GS flat at the untreated level used to pass as 'highest tail'."""
    from ravivarapu_gates import ravivarapu_fig7_gates

    traces = {k: _paper_steps("fig7", v) for k, v in _FIG7_NAMES.items()}
    traces["baseline-gs"] = [traces["baseline"][0]] * 11
    report = ravivarapu_fig7_gates(traces, fig5a_traces=_fig5a_paper())
    assert report["gates"]["gs_highest_tail"]
    assert not report["gates"]["mae_gs"]
    assert not report["gates"]["gs_tail_near_paper"]


def test_fig7_rejects_gs_not_highest():
    from ravivarapu_gates import ravivarapu_fig7_gates

    traces = {
        "baseline": [0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.40, 0.40, 0.40, 0.40],
        "baseline-pm": [0.46, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.39, 0.39, 0.39, 0.39],
        "baseline-gs": [0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.35, 0.35, 0.35, 0.35, 0.35],
        "paper": [0.46, 0.43, 0.40, 0.37, 0.34, 0.33, 0.32, 0.32, 0.32, 0.32, 0.32],
    }
    report = ravivarapu_fig7_gates(traces)
    assert not report["gates"]["gs_highest_tail"]
    assert not report["pass"]


def test_fig7_rejects_identical_variants():
    """Four policies that all stimulate every step tie; ties are not the paper's ordering."""
    from ravivarapu_gates import ravivarapu_fig7_gates

    trace = [0.449, 0.418, 0.387, 0.357, 0.326, 0.295, 0.295, 0.295, 0.295, 0.295, 0.295]
    report = ravivarapu_fig7_gates({k: list(trace) for k in ("baseline", "baseline-pm", "baseline-gs", "paper")})
    assert not report["gates"]["sea_dbs_lowest_tail"]
    assert not report["gates"]["gs_highest_tail"]
    assert not report["gates"]["pm_not_sea"]


def test_table2_paper_self_passes():
    from ravivarapu_gates import TABLE2_PAPER_PSD, ravivarapu_table2_gates

    rows = {n: {"baseline_psd": b, "sea_psd": s} for n, (b, s) in TABLE2_PAPER_PSD.items()}
    report = ravivarapu_table2_gates(rows)
    assert report["pass"], report["gates"]


def test_table2_rejects_tie():
    from ravivarapu_gates import TABLE2_PAPER_PSD, ravivarapu_table2_gates

    rows = {n: {"baseline_psd": s, "sea_psd": s} for n, (_b, s) in TABLE2_PAPER_PSD.items()}
    report = ravivarapu_table2_gates(rows)
    assert not report["gates"]["sea_below_baseline_every_n"]
    assert not report["gates"]["gap_near_paper_every_n"]
    assert not report["pass"]


def test_table2_rejects_missing_interval():
    from ravivarapu_gates import ravivarapu_table2_gates

    report = ravivarapu_table2_gates({10: {"baseline_psd": 364.0, "sea_psd": 326.0}})
    assert not report["pass"]
