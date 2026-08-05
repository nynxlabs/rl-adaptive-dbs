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
    fig4_training_gates,
    fig5_spikes_energy_gates,
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


@pytest.mark.skipif(not (ARTIFACT / "curves_fig4_reward.json").exists(), reason="no digitization")
def test_fig4_paper_self_consistent():
    paper_r = load_curves("fig4_reward")
    paper_l = load_curves("fig4_length")
    rx, ry = paper_r["Smoothed"] if "Smoothed" in paper_r else paper_r["Raw"]
    lx, ly = paper_l["Smoothed"] if "Smoothed" in paper_l else paper_l["Raw"]
    n = int(min(rx[-1], lx[-1], 500)) + 1
    rewards = np.interp(np.arange(n), rx, ry)
    lengths = np.interp(np.arange(n), lx, ly)
    report = fig4_training_gates(rewards, lengths, max_episode_steps=25)
    assert report["gates"]["reward_improves_like_paper"]
    assert report["gates"]["length_decreases_like_paper"]


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
    assert report["gates"]["spike_mean_near_paper"]
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
    report = fig6_training_gates(
        np.interp(episodes, abx, aby),
        np.interp(episodes, ampx, ampy),
        np.interp(episodes, freqx, freqy),
        np.interp(episodes, pwx, pwy),
    )
    assert report["gates"]["alpha_beta_decreases_like_paper"]
    assert report["pass"]


@pytest.mark.skipif(not (ARTIFACT / "curves_fig7.json").exists(), reason="no digitization")
def test_fig7_paper_self_consistent():
    paper = load_curves("fig7")
    ax, ay = paper["average"]
    trajectories = [ay.tolist() for _ in range(3)]
    report = fig7_eval_gates(trajectories)
    assert report["gates"]["eval_protocol_ok"]
    assert report["gates"]["overall_mean_near_paper"]
    assert report["pass"]


def test_curves_path_stems():
    assert curves_path("fig4_reward").name == "curves_fig4_reward.json"
