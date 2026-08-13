"""Tests for Mehregan digitization-anchored paper gates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
sys.path.insert(0, str(_DIG))

from paper_gates import (  # noqa: E402
    fig1b_gates,
    fig2_time_gates,
    fig4a_gates,
    fig4b_gates,
    fig5_efficacy_gates,
    load_refined,
    refined_path,
    rel_close,
    ratio_close,
    window_mean,
)

ARTIFACT = Path("artifacts/figures/papers/mehregan")


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


@pytest.mark.skipif(not (ARTIFACT / "2a/paper_digitization/curves_wpd_refined.json").exists(), reason="no digitization artifact")
def test_fig2a_paper_self_consistent():
    paper = load_refined(refined_path("2a"))
    # Feeding paper as replication should pass ordering + ratio gates.
    report = fig2_time_gates(paper, paper=paper, panel="2a")
    assert report["gates"]["treated_below_untreated_late"]
    assert report["gates"]["late_ratio_near_paper"]
    assert report["pass"]


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


@pytest.mark.skipif(
    not (ARTIFACT / "4b/paper_digitization/curves_wpd_refined_psd.json").exists(),
    reason="no digitization artifact",
)
def test_fig4b_paper_self_late_floor():
    psd = load_refined(refined_path("4b", stem="curves_wpd_refined_psd"))
    rew = load_refined(refined_path("4b", stem="curves_wpd_refined_reward"))
    px, py = psd[next(iter(psd))]
    rx, ry = rew[next(iter(rew))]
    episodes = np.arange(9, dtype=float)
    beta = np.interp(episodes, px, py)
    reward = np.interp(episodes, rx, ry)
    report = fig4b_gates(reward, beta)
    assert report["gates"]["late_beta_above_threshold"]
    assert report["gates"]["late_beta_near_paper"]
    assert report["gates"]["late_reward_near_zero"]
    assert report["pass"]


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
    assert report["pass"]
