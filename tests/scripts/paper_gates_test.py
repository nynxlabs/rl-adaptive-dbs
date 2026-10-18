"""Tests for Mehregan digitization-anchored paper gates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
sys.path.insert(0, str(_DIG))

from paper_gates import (  # noqa: E402
    fig4b_paper_per_episode,
    fig4b_paper_x_to_episode,
    fig1b_gates,
    fig2_time_gates,
    fig4a_gates,
    fig4b_gates,
    fig5_efficacy_gates,
    fig6_quant_gates,
    load_refined,
    refined_path,
    rel_close,
    ratio_close,
    window_mean,
)

ARTIFACT = Path("artifacts/figures/papers/mehregan")
_EFFICACY = Path(__file__).resolve().parents[2] / "scripts/figures/papers/mehregan/efficacy_panel.py"


def _has(panel: str, stem: str = "curves_wpd_refined") -> bool:
    return (ARTIFACT / panel / "paper_digitization" / f"{stem}.json").exists()


def _failed(report: dict) -> dict:
    return {k: v for k, v in report["gates"].items() if not v}


@pytest.mark.skipif(not (ARTIFACT / "1b/paper_digitization/curves_wpd_refined.json").exists(), reason="no digitization artifact")
def test_load_refined_1b_has_three_series():
    series = load_refined(refined_path("1b"))
    assert set(series) >= {"pd", "healthy", "pd_130hz"}
    x, y = series["pd"]
    assert x.size == y.size >= 4


def test_window_mean_and_rel_close():
    x = np.linspace(0, 10, 11)
    y = np.arange(11, dtype=float)
    assert window_mean(x, y, lo=5, hi=7) == pytest.approx(6.0)
    assert rel_close(100.0, 110.0, tol=0.15)
    assert not rel_close(100.0, 150.0, tol=0.15)
    assert ratio_close(2.0, 4.0, 3.0, 6.0, tol=0.05)


@pytest.mark.parametrize("panel", ["2a", "2b"])
def test_fig2_paper_self_consistent(panel: str):
    if not _has(panel):
        pytest.skip("no digitization artifact")
    paper = load_refined(refined_path(panel))
    # Feeding paper as replication must pass every gate.
    report = fig2_time_gates(
        paper,
        paper=paper,
        panel=panel,
        rep_untreated="pd",
        rep_treated="PD 130Hz Treatment",
    )
    assert report["gates"]["treated_below_untreated_late"]
    assert report["gates"]["late_ratio_near_paper"]
    assert {"untreated_flat", "t50_near_paper", "treated_shape_r", "gap_shape_r"} <= set(report["gates"])
    assert report["pass"], _failed(report)


@pytest.mark.skipif(not _has("2a"), reason="no digitization artifact")
def test_fig2a_late_onset_fails_t50():
    paper = load_refined(refined_path("2a"))
    tx, ty = paper["PD 130Hz Treatment"]
    t = np.linspace(0.0, 12.0, 241)
    pre = float(np.mean(ty[tx <= 2.0]))
    # Same suppression, but the drop starts 3 s late.
    delayed = np.where(t < 5.0, pre, np.interp(t - 3.0, tx, ty))
    rep = {"pd": paper["pd"], "pd_130hz": (t, delayed)}
    report = fig2_time_gates(rep, paper=paper, panel="2a")
    assert not report["gates"]["t50_near_paper"]
    assert not report["pass"]


@pytest.mark.skipif(not _has("2b"), reason="no digitization artifact")
def test_fig2b_drifting_untreated_fails_flat():
    paper = load_refined(refined_path("2b"))
    ux, uy = paper["pd"]
    drift = np.where(ux <= 2.0, 1.0, 1.0 - 0.06 * (ux - 2.0))  # −60% by 12 s
    rep = {"pd": (ux, uy * drift), "pd_130hz": paper["PD 130Hz Treatment"]}
    report = fig2_time_gates(rep, paper=paper, panel="2b")
    assert not report["gates"]["untreated_flat"]
    assert not report["pass"]


@pytest.mark.skipif(not (ARTIFACT / "4a/paper_digitization/curves_wpd_refined.json").exists(), reason="no digitization artifact")
def test_fig4a_paper_self_drop():
    paper = load_refined(refined_path("4a"))
    x, y = paper["training"]
    # Build a step-indexed trace by interpolating paper onto 0..300
    steps = np.arange(300, dtype=float)
    trace = np.interp(steps, x, y)
    report = fig4a_gates(trace)
    assert report["gates"]["overall_trend_down"]
    assert report["gates"]["drop_vs_paper"]
    assert report["gates"]["ep0_near_paper"]
    assert {"episode_shape_r", "per_episode_near_paper", "no_late_rebound"} <= set(report["gates"])
    assert report["pass"], _failed(report)


@pytest.mark.skipif(not _has("4a"), reason="no digitization artifact")
def test_fig4a_late_rebound_and_cliff_fail():
    x, y = load_refined(refined_path("4a"))["training"]
    trace = np.interp(np.arange(300, dtype=float), x, y)
    rebound = trace.copy()
    rebound[210:] += 0.08  # eps 7–9 climb back toward the start level
    report = fig4a_gates(rebound)
    assert not report["gates"]["no_late_rebound"]
    assert not report["gates"]["per_episode_near_paper"]
    cliff = trace.copy()
    cliff[120:150] -= 0.15  # mid fade far deeper than the paper's (two-sided gate)
    report = fig4a_gates(cliff)
    assert not report["gates"]["mid_fade_vs_paper"]


@pytest.mark.skipif(
    not (ARTIFACT / "4b/paper_digitization/curves_wpd_refined_psd.json").exists(),
    reason="no digitization artifact",
)
def test_fig4b_paper_self_late_floor():
    psd = load_refined(refined_path("4b", stem="curves_wpd_refined_psd"))
    rew = load_refined(refined_path("4b", stem="curves_wpd_refined_reward"))
    # 10 vertices on the paper's 0–8 axis → episodes 0–9.
    beta = fig4b_paper_per_episode(psd[next(iter(psd))], 10)
    reward = fig4b_paper_per_episode(rew[next(iter(rew))], 10)
    report = fig4b_gates(reward, beta)
    assert report["gates"]["late_beta_near_paper"]
    assert report["gates"]["late_reward_near_zero"]
    assert report["gates"]["ep0_beta_near_paper"]
    assert report["gates"]["plot_style"]
    assert report["pass"], _failed(report)
    # Nine episodes (the old default) no longer satisfies the panel.
    assert not fig4b_gates(reward[:9], beta[:9])["gates"]["plot_style"]


def test_fig4b_paper_x_maps_to_episode():
    assert fig4b_paper_x_to_episode(np.array([0.0, 8.0 / 9.0, 8.0]))[1:] == pytest.approx([1.0, 9.0])


@pytest.mark.skipif(not _has("4b", "curves_wpd_refined_psd"), reason="no digitization artifact")
def test_fig4b_deep_dip_and_positive_rewards_fail():
    psd = load_refined(refined_path("4b", stem="curves_wpd_refined_psd"))
    beta = fig4b_paper_per_episode(psd[next(iter(psd))], 10)
    # Shape of the shipped seed-0 run: ep 1 dives to −194, late episodes go positive.
    reward = np.array([-81.0, -193.7, -102.7, -12.3, -16.5, -24.3, -17.2, 16.7, 1.9, 1.2])
    report = fig4b_gates(reward, beta)
    assert not report["gates"]["reward_floor_not_deeper"]
    assert not report["gates"]["max_reward_le_cap"]
    assert not report["gates"]["early_reward_near_paper"]
    over = beta.copy()
    over[7] *= 0.78  # one over-suppressed episode
    report = fig4b_gates(fig4b_paper_per_episode(
        load_refined(refined_path("4b", stem="curves_wpd_refined_reward"))["Reward"], 10), over)
    assert not report["gates"]["per_episode_psd_near_paper"]
    assert not report["gates"]["psd_floor_not_deeper"]


@pytest.mark.skipif(not (ARTIFACT / "5b/paper_digitization/curves_wpd_refined.json").exists(), reason="no digitization artifact")
def test_fig5b_ratios_from_paper_means():
    paper = load_refined(refined_path("5b"))
    means = {
        name: window_mean(*xy, lo=4.0)
        for name, xy in [
            ("no_stim", paper["PD no stim"]),
            ("trained", paper["Fully Trained 30Hz"]),
            ("periodic", paper["Periodic 30Hz"]),
        ]
    }
    # Remap keys for builder
    report = fig5_efficacy_gates(
        {
            "no_stim": means["no_stim"],
            "trained": means["trained"],
            "periodic": means["periodic"],
        },
        panel="5b",
    )
    assert report["gates"]["trained_below_no_stim"]
    assert report["pass"]


@pytest.mark.skipif(not (ARTIFACT / "1b/paper_digitization/curves_wpd_refined.json").exists(), reason="no digitization artifact")
def test_fig1b_ordering_on_paper():
    paper = load_refined(refined_path("1b"))
    report = fig1b_gates(paper, paper=paper)
    assert report["gates"]["pd_gt_healthy"]
    assert report["gates"]["pd_130_lt_pd"]
    assert {"pd_peak_freq_near_paper", "low_over_beta_near_paper", "pd_shape_r"} <= set(report["gates"])
    assert report["pass"], _failed(report)


@pytest.mark.skipif(not _has("1b"), reason="no digitization artifact")
def test_fig1b_fine_resolution_paper_passes_after_smoothing():
    """A 0.1 Hz resampled copy of the paper (replication resolution) still passes."""
    paper = load_refined(refined_path("1b"))
    f = np.arange(0.2, 50.6, 0.0977)
    fine = {k: (f, np.interp(f, *paper[k])) for k in paper}
    report = fig1b_gates(fine, paper=paper)
    assert report["pass"], _failed(report)


@pytest.mark.skipif(not _has("1b"), reason="no digitization artifact")
def test_fig1b_shifted_pd_peak_fails():
    paper = load_refined(refined_path("1b"))
    rep = dict(paper)
    x, y = paper["pd"]
    rep["pd"] = (x, np.interp(x - 8.0, x, y))  # PD peak moved 8 Hz up
    report = fig1b_gates(rep, paper=paper)
    assert not report["gates"]["pd_peak_freq_near_paper"]
    assert not report["pass"]


_FIG5_NAMES = {
    "5a": {
        "no_stim": "PD no stim",
        "trained": "Fully Trained 45Hz",
        "periodic": "Periodic 45Hz",
        "cdbs_130": "Periodic 130Hz",
    },
    "5b": {
        "no_stim": "PD no stim",
        "trained": "Fully Trained 30Hz",
        "periodic": "Periodic 30Hz",
    },
}


def _paper_means(panel: str, names: dict[str, str]) -> dict[str, float]:
    paper = load_refined(refined_path(panel))
    return {key: window_mean(*paper[name], lo=4.0) for key, name in names.items()}


@pytest.mark.skipif(not (ARTIFACT / "5a/paper_digitization/curves_wpd_refined.json").exists(), reason="no digitization artifact")
def test_fig5a_paper_self_passes():
    means = _paper_means("5a", _FIG5_NAMES["5a"])
    report = fig5_efficacy_gates(means, panel="5a", require_cdbs=True)
    assert report["pass"], report["gates"]


@pytest.fixture(scope="module")
def efficacy_panel():
    from rl_adaptive_dbs.panel import load_script_module

    return load_script_module("mehregan_efficacy_panel_test", _EFFICACY)


def _fig5_paper_inputs(panel: str) -> tuple[dict, dict]:
    paper = load_refined(refined_path(panel))
    traces = {k: paper[v] for k, v in _FIG5_NAMES[panel].items()}
    means = {f"{k}_mean": window_mean(*xy, lo=4.0) for k, xy in traces.items()}
    means["trained_closed_loop"] = True
    return means, traces


@pytest.mark.parametrize("panel", ["5a", "5b"])
def test_fig5_efficacy_panel_full_gate_set_passes_on_paper(efficacy_panel, panel: str):
    if not _has(panel):
        pytest.skip("no digitization artifact")
    cfg = efficacy_panel.FIG5A if panel == "5a" else efficacy_panel.FIG5B
    means, traces = _fig5_paper_inputs(panel)
    gates = efficacy_panel.efficacy_gates(cfg, means, traces)
    flags = {k: v for k, v in gates.items() if isinstance(v, bool) and k != "pass"}
    expected = {
        "shared_baseline", "trained_closed_loop", "prestim_level_near_paper",
        "trained_onset_ratio_near_paper", "trained_shape_r", "periodic_shape_r",
        "trained_periodic_gap_near_paper",
    }
    if panel == "5a":
        expected |= {"cdbs_lowest", "cdbs_no_stim_ratio_near_paper"}
    assert expected <= set(flags)
    assert gates["pass"], {k: v for k, v in flags.items() if not v}
    # Without traces the verdict cannot pass.
    assert not efficacy_panel.efficacy_gates(cfg, means)["pass"]


@pytest.mark.parametrize("panel", ["5a", "5b"])
def test_fig5_trained_equal_periodic_fails_gap(efficacy_panel, panel: str):
    if not _has(panel):
        pytest.skip("no digitization artifact")
    cfg = efficacy_panel.FIG5A if panel == "5a" else efficacy_panel.FIG5B
    means, traces = _fig5_paper_inputs(panel)
    # Trained policy that just reproduces periodic stimulation.
    traces["trained"] = traces["periodic"]
    means["trained_mean"] = means["periodic_mean"]
    gates = efficacy_panel.efficacy_gates(cfg, means, traces)
    assert not gates["trained_periodic_gap_near_paper"]
    assert not gates["pass"]


@pytest.mark.parametrize("panel", ["5a", "5b"])
def test_fig5_non_learning_trained_fails_ratio(efficacy_panel, panel: str):
    if not _has(panel):
        pytest.skip("no digitization artifact")
    cfg = efficacy_panel.FIG5A if panel == "5a" else efficacy_panel.FIG5B
    means, traces = _fig5_paper_inputs(panel)
    # A random-like pattern near the untreated level (trained/no-stim ~0.99).
    means["trained_mean"] = 0.99 * means["no_stim_mean"]
    gates = efficacy_panel.efficacy_gates(cfg, means, traces)
    assert not gates["trained_no_stim_ratio_near_paper"]


@pytest.mark.skipif(not _has("5a"), reason="no digitization artifact")
def test_fig5a_offset_trained_baseline_fails_shared(efficacy_panel):
    means, traces = _fig5_paper_inputs("5a")
    t, y = traces["trained"]
    traces["trained"] = (t, np.where(t <= 2.0, y * 1.08, y))  # trained pre-onset 8% high
    gates = efficacy_panel.efficacy_gates(efficacy_panel.FIG5A, means, traces)
    assert not gates["shared_baseline"]


@pytest.mark.parametrize(
    ("panel", "fp32_name"), [("6a", "Fully Trained 45Hz"), ("6b", "Fully Trained 30Hz")]
)
def test_fig6_paper_self_passes(panel: str, fp32_name: str):
    if not (ARTIFACT / f"{panel}/paper_digitization/curves_wpd_refined.json").exists():
        pytest.skip("no digitization artifact")
    paper = load_refined(refined_path(panel))
    traces = {
        "fp32": paper[fp32_name],
        "ptq_int8": paper["PTQ, INT8"],
        "ptq_fp16": paper["PTQ, FP16"],
        "qat": paper["QAT"],
    }
    closed = {k: True for k in traces}
    report = fig6_quant_gates(traces, panel=panel, closed_loop=closed)
    assert report["pass"], {k: v for k, v in report["gates"].items() if not v}


def test_fig6_rejects_open_loop_and_suppressing_qat():
    t = np.linspace(0.0, 12.0, 61)
    pre = np.where(t <= 2.0, 470.0, 0.0)
    suppressed = pre + np.where(t > 2.0, 360.0, 0.0)
    traces = {k: (t, suppressed) for k in ("fp32", "ptq_int8", "ptq_fp16", "qat")}
    closed = {"fp32": True, "ptq_int8": True, "ptq_fp16": True, "qat": False}
    report = fig6_quant_gates(traces, panel="6a", paper={}, closed_loop=closed)
    assert not report["gates"]["all_closed_loop"]
    assert not report["gates"]["qat_elevated_vs_fp32"]
    assert not report["pass"]


def test_fig6_qat_seed_band_needs_majority_and_median():
    t = np.linspace(0.0, 12.0, 61)
    pre = 500.0

    def trace(late_over_pre: float) -> tuple[np.ndarray, np.ndarray]:
        return t, np.where(t <= 2.0, pre, pre * late_over_pre)

    fp32 = trace(0.78)
    traces = {"fp32": fp32, "ptq_int8": fp32, "ptq_fp16": fp32}
    paper = {"QAT": trace(1.0), "fp32": fp32, "PTQ, INT8": fp32, "PTQ, FP16": fp32}
    lucky = {str(s): trace(r) for s, r in enumerate((1.26, 0.59, 1.26, 0.59, 0.59, 0.89, 0.69))}
    report = fig6_quant_gates(traces, qat_seeds=lucky, paper=paper)
    assert report["metrics"]["qat_paper_like_count"] == 3
    assert not report["gates"]["qat_majority_not_below_paper"]
    assert not report["gates"]["qat_median_near_paper"]
    robust = {str(s): trace(r) for s, r in enumerate((0.95, 1.05, 0.59, 1.0, 0.9, 1.1, 0.69))}
    report = fig6_quant_gates(traces, qat_seeds=robust, paper=paper)
    assert report["gates"]["qat_majority_not_below_paper"]
    assert report["gates"]["qat_median_near_paper"]
    assert report["gates"]["qat_elevated_vs_fp32"]


@pytest.mark.parametrize(("panel", "fp32_name"), [("6a", "Fully Trained 45Hz"), ("6b", "Fully Trained 30Hz")])
def test_fig6_ptq_identical_to_fp32_fails_alters(panel: str, fp32_name: str):
    if not _has(panel):
        pytest.skip("no digitization artifact")
    paper = load_refined(refined_path(panel))
    fp32 = paper[fp32_name]
    traces = {"fp32": fp32, "ptq_int8": fp32, "ptq_fp16": fp32, "qat": paper["QAT"]}
    report = fig6_quant_gates(traces, panel=panel, closed_loop={k: True for k in traces})
    assert not report["gates"]["ptq_int8_alters_trace"]
    assert not report["gates"]["ptq_fp16_alters_trace"]
    # PTQ "near fp32" still holds — the alteration gates are what catch it.
    assert report["gates"]["ptq_int8_near_fp32"] and report["gates"]["ptq_fp16_near_fp32"]
    assert not report["pass"]


@pytest.mark.skipif(not _has("6a"), reason="no digitization artifact")
def test_fig6a_qat_tracking_fp32_fails_gap():
    paper = load_refined(refined_path("6a"))
    traces = {
        "fp32": paper["Fully Trained 45Hz"],
        "ptq_int8": paper["PTQ, INT8"],
        "ptq_fp16": paper["PTQ, FP16"],
        "qat": paper["Fully Trained 45Hz"],
    }
    report = fig6_quant_gates(traces, panel="6a")
    assert not report["gates"]["qat_fp32_gap_vs_paper"]
