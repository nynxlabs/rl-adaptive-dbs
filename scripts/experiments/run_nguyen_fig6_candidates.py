#!/usr/bin/env python3
"""Run Nguyen Fig 6 parallel candidate tuning runs (Candidate 1, 2, 3).

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_nguyen_fig6_candidates.py --candidate 1
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_nguyen_fig6_candidates.py --candidate 2
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_nguyen_fig6_candidates.py --candidate 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer
from controllers.snn.config import (
    BIOMARKER_THRESHOLD,
    INIT_AMPLITUDE_NA_PER_CM2,
    INIT_FREQUENCY_HZ,
    INIT_PULSE_WIDTH_MS,
    SNNConfig,
    fig4_nguyen_config,
)
from controllers.snn.networks import DSQN
from controllers.snn.trainer import (
    DSQNTrainer,
    TrainResult,
    save_checkpoint,
)

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from nguyen_gates import (  # noqa: E402
    attach_digitization,
    fig4_training_gates,
    fig5_spikes_energy_gates,
    fig6_training_gates,
)

_OVERLAY = Path(__file__).resolve().parents[2] / "scripts" / "figures" / "papers" / "paper_overlay.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("figure_paper_overlay", _OVERLAY)
assert _spec and _spec.loader
_paper_overlay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paper_overlay)

_PLOT_AXES = Path(__file__).resolve().parents[2] / "scripts" / "figures" / "papers" / "plot_axes.py"
_pa_spec = importlib.util.spec_from_file_location("figure_plot_axes", _PLOT_AXES)
assert _pa_spec and _pa_spec.loader
_figure_plot_axes = importlib.util.module_from_spec(_pa_spec)
_pa_spec.loader.exec_module(_figure_plot_axes)
data_ylim = _figure_plot_axes.data_ylim

OUT_DIR = Path("artifacts/figures/papers/nguyen/experiments")
FIG_DIR = Path("figures/nguyen/images/experiments")

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "font.size": 10,
}


def moving_average(y: np.ndarray, window: int = 20) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    window = max(1, int(window))
    if window == 1 or y.size == 1:
        return y.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(y.astype(float), (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: y.size]


def get_candidate_config(candidate: int, seed: int = 0) -> SNNConfig:
    base = fig4_nguyen_config(seed=seed, num_episodes=500)
    if candidate == 1:
        # Candidate 1: Baseline refined with nominal 25-step alpha-beta logging
        return base
    elif candidate == 2:
        # Candidate 2: Frequency smoothing (freq_sens=15, early=3, threshold_reward=350, warm_zone=200)
        return replace(
            base,
            frequency_sensitivity=15.0,
            frequency_sensitivity_early=3.0,
            frequency_sensitivity_early_episodes=50,
            threshold_reward=350.0,
            warm_zone_upper=200.0,
            warm_zone_bonus_coef=180.0,
            alpha_beta_progress_coef=2800.0,
        )
    elif candidate == 3:
        # Candidate 3: Direct Paper Target Alignment (pw_min=1.0, freq_sens=12, amp_sens=8)
        return replace(
            base,
            pulse_width_min=1.0,
            frequency_sensitivity=12.0,
            frequency_sensitivity_early=4.0,
            frequency_sensitivity_early_episodes=50,
            amplitude_sensitivity=8.0,
            threshold_reward=320.0,
            alpha_beta_progress_coef=3000.0,
            warm_zone_upper=210.0,
        )
    elif candidate == 4:
        # Candidate 4: Earlier Exploration Transition (accelerate@800 steps -> ~ep 60, floor by ep 85)
        return replace(
            base,
            epsilon_accelerate_after_steps=800,
            epsilon_accelerate_decay_steps=300,
            pulse_width_min_ramp_end_episode=95,
        )
    elif candidate == 5:
        # Candidate 5: Sharp Transition + Strong Progress Guidance
        return replace(
            base,
            epsilon_accelerate_after_steps=850,
            epsilon_accelerate_decay_steps=250,
            alpha_beta_progress_coef=3500.0,
            threshold_reward=350.0,
            pulse_width_min_ramp_end_episode=90,
        )
    elif candidate == 6:
        # Candidate 6: Fast PW Ramp + Transition at ep 70
        return replace(
            base,
            epsilon_accelerate_after_steps=750,
            epsilon_accelerate_decay_steps=300,
            frequency_sensitivity_early_episodes=40,
            pulse_width_min_early_episodes=40,
            pulse_width_min_ramp_end_episode=85,
            alpha_beta_progress_coef=3200.0,
        )
    elif candidate == 7:
        # Candidate 7: Frequency max 95 Hz, sensitivity 10 Hz, accelerated transition
        return replace(
            base,
            frequency_max=95.0,
            frequency_sensitivity=10.0,
            frequency_sensitivity_early=4.0,
            pulse_width_max=1.25,
            epsilon_accelerate_after_steps=800,
            epsilon_accelerate_decay_steps=300,
            pulse_width_min_ramp_end_episode=95,
        )
    elif candidate == 8:
        # Candidate 8: Frequency max 90 Hz, sensitivity 10 Hz, tight PW bounds (0.95-1.15)
        return replace(
            base,
            frequency_max=90.0,
            frequency_sensitivity=10.0,
            frequency_sensitivity_early=3.0,
            pulse_width_min=0.95,
            pulse_width_max=1.15,
            pulse_width_min_ramp_end_episode=90,
            epsilon_accelerate_after_steps=800,
            epsilon_accelerate_decay_steps=280,
            alpha_beta_progress_coef=3000.0,
        )
    elif candidate == 9:
        # Candidate 9: Frequency max 85 Hz, sensitivity 8 Hz, paper exact target
        return replace(
            base,
            frequency_max=85.0,
            frequency_sensitivity=8.0,
            frequency_sensitivity_early=3.0,
            pulse_width_min=0.95,
            pulse_width_max=1.10,
            pulse_width_min_ramp_end_episode=85,
            epsilon_accelerate_after_steps=800,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=350.0,
            alpha_beta_progress_coef=3200.0,
        )
    elif candidate == 10:
        # Candidate 10: Balanced frequency max 100 Hz, step 15 Hz, pw_min 1.02
        return replace(
            base,
            frequency_max=100.0,
            frequency_sensitivity=15.0,
            frequency_sensitivity_early=4.0,
            pulse_width_min=1.02,
            pulse_width_max=1.20,
            pulse_width_min_ramp_end_episode=90,
            epsilon_accelerate_after_steps=800,
            epsilon_accelerate_decay_steps=280,
            threshold_reward=320.0,
            alpha_beta_progress_coef=3000.0,
        )
    elif candidate == 11:
        # Candidate 11: Frequency max 105 Hz, pw_min 1.05, high progress guidance
        return replace(
            base,
            frequency_max=105.0,
            frequency_sensitivity=15.0,
            frequency_sensitivity_early=4.0,
            pulse_width_min=1.05,
            pulse_width_max=1.20,
            pulse_width_min_ramp_end_episode=90,
            epsilon_accelerate_after_steps=800,
            epsilon_accelerate_decay_steps=260,
            threshold_reward=350.0,
            alpha_beta_progress_coef=3500.0,
        )
    elif candidate == 12:
        # Candidate 12: Frequency max 98 Hz, step 12 Hz, pw_min 1.00
        return replace(
            base,
            frequency_max=98.0,
            frequency_sensitivity=12.0,
            frequency_sensitivity_early=4.0,
            pulse_width_min=1.00,
            pulse_width_max=1.18,
            pulse_width_min_ramp_end_episode=85,
            epsilon_accelerate_after_steps=750,
            epsilon_accelerate_decay_steps=280,
            threshold_reward=330.0,
            alpha_beta_progress_coef=3200.0,
        )
    elif candidate == 13:
        # Candidate 13: Fast PW Floor Ramp ep 30->75, f_max=105 Hz, w_min=1.02 ms
        return replace(
            base,
            frequency_max=105.0,
            frequency_sensitivity=15.0,
            frequency_sensitivity_early=5.0,
            frequency_sensitivity_early_episodes=40,
            pulse_width_min=1.02,
            pulse_width_min_early=0.05,
            pulse_width_min_early_episodes=30,
            pulse_width_min_ramp_end_episode=75,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=350.0,
            alpha_beta_progress_coef=3500.0,
        )
    elif candidate == 14:
        # Candidate 14: Balanced Fast Ramp ep 35->80, f_max=100 Hz, w_min=1.05 ms
        return replace(
            base,
            frequency_max=100.0,
            frequency_sensitivity=15.0,
            frequency_sensitivity_early=5.0,
            frequency_sensitivity_early_episodes=40,
            pulse_width_min=1.05,
            pulse_width_min_early=0.05,
            pulse_width_min_early_episodes=35,
            pulse_width_min_ramp_end_episode=80,
            epsilon_accelerate_after_steps=750,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=350.0,
            alpha_beta_progress_coef=3200.0,
        )
    elif candidate == 15:
        # Candidate 15: Direct Paper Target + Early Transition ep 35->80, f_max=100 Hz, w_min=1.00 ms
        return replace(
            base,
            frequency_max=100.0,
            frequency_sensitivity=12.0,
            frequency_sensitivity_early=5.0,
            frequency_sensitivity_early_episodes=35,
            pulse_width_min=1.00,
            pulse_width_min_early=0.05,
            pulse_width_min_early_episodes=35,
            pulse_width_min_ramp_end_episode=80,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=350.0,
            alpha_beta_progress_coef=3500.0,
        )
    elif candidate == 16:
        # Candidate 16: Cand 12 Shape + Deep Suppression (f_max=115 Hz, A_min=250 nA, w_min=1.05 ms)
        return replace(
            base,
            frequency_max=115.0,
            frequency_sensitivity=12.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=250.0,
            pulse_width_min=1.05,
            pulse_width_max=1.20,
            pulse_width_min_ramp_end_episode=85,
            epsilon_accelerate_after_steps=750,
            epsilon_accelerate_decay_steps=280,
            threshold_reward=340.0,
            alpha_beta_progress_coef=3200.0,
        )
    elif candidate == 17:
        # Candidate 17: Cand 12 Shape + Max Suppression (f_max=120 Hz, A_min=260 nA, w_min=1.08 ms)
        return replace(
            base,
            frequency_max=120.0,
            frequency_sensitivity=12.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=260.0,
            pulse_width_min=1.08,
            pulse_width_max=1.22,
            pulse_width_min_ramp_end_episode=85,
            epsilon_accelerate_after_steps=750,
            epsilon_accelerate_decay_steps=280,
            threshold_reward=350.0,
            alpha_beta_progress_coef=3500.0,
        )
    elif candidate == 18:
        # Candidate 18: Cand 12 Shape + Fast Intra-Episode Ramp (f_max=110 Hz, A_min=255 nA, kappa=3600)
        return replace(
            base,
            frequency_max=110.0,
            frequency_sensitivity=14.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=255.0,
            pulse_width_min=1.05,
            pulse_width_max=1.20,
            pulse_width_min_ramp_end_episode=80,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=260,
            threshold_reward=360.0,
            alpha_beta_progress_coef=3600.0,
        )
    elif candidate == 19:
        # Candidate 19: Cand 18 Structure + f_max=135 Hz, step 18 Hz, target ~100 a.u.
        return replace(
            base,
            frequency_max=135.0,
            frequency_sensitivity=18.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=260.0,
            pulse_width_min=1.05,
            pulse_width_max=1.20,
            pulse_width_min_ramp_end_episode=80,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=380.0,
            alpha_beta_progress_coef=3800.0,
        )
    elif candidate == 20:
        # Candidate 20: Cand 18 Structure + f_max=145 Hz, step 20 Hz, max suppression
        return replace(
            base,
            frequency_max=145.0,
            frequency_sensitivity=20.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=260.0,
            pulse_width_min=1.08,
            pulse_width_max=1.22,
            pulse_width_min_ramp_end_episode=80,
            epsilon_accelerate_after_steps=650,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=400.0,
            alpha_beta_progress_coef=4000.0,
        )
    elif candidate == 21:
        # Candidate 21: Cand 18 Structure + f_max=130 Hz, step 16 Hz, w_min=1.10 ms
        return replace(
            base,
            frequency_max=130.0,
            frequency_sensitivity=16.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=260.0,
            pulse_width_min=1.10,
            pulse_width_max=1.22,
            pulse_width_min_ramp_end_episode=80,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=380.0,
            alpha_beta_progress_coef=3600.0,
        )
    elif candidate == 22:
        # Candidate 22: Cand 20 Polish - f_max=102 Hz, w_min=1.12 ms, ramp_end=105
        return replace(
            base,
            frequency_max=102.0,
            frequency_sensitivity=16.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=260.0,
            pulse_width_min=1.12,
            pulse_width_max=1.22,
            pulse_width_min_ramp_end_episode=105,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=380.0,
            alpha_beta_progress_coef=3800.0,
        )
    elif candidate == 23:
        # Candidate 23: Cand 20 Polish - f_max=100 Hz, w_min=1.15 ms, ramp_end=110
        return replace(
            base,
            frequency_max=100.0,
            frequency_sensitivity=15.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=260.0,
            pulse_width_min=1.15,
            pulse_width_max=1.25,
            pulse_width_min_ramp_end_episode=110,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=400.0,
            alpha_beta_progress_coef=4000.0,
        )
    elif candidate == 24:
        # Candidate 24: Cand 20 Polish - f_max=104 Hz, w_min=1.10 ms, ramp_end=100
        return replace(
            base,
            frequency_max=104.0,
            frequency_sensitivity=16.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=260.0,
            pulse_width_min=1.10,
            pulse_width_max=1.20,
            pulse_width_min_ramp_end_episode=100,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=380.0,
            alpha_beta_progress_coef=3800.0,
        )
    elif candidate == 25:
        # Candidate 25: Enhanced Charge - f_max=98 Hz, A_min=265 nA, w_min=1.15 ms, ramp_end=105
        return replace(
            base,
            frequency_max=98.0,
            frequency_sensitivity=15.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=265.0,
            amplitude_max=300.0,
            pulse_width_min=1.15,
            pulse_width_max=1.25,
            pulse_width_min_ramp_end_episode=105,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=380.0,
            alpha_beta_progress_coef=3800.0,
        )
    elif candidate == 26:
        # Candidate 26: Strong Charge - f_max=96 Hz, A_min=270 nA, w_min=1.18 ms, ramp_end=105
        return replace(
            base,
            frequency_max=96.0,
            frequency_sensitivity=14.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=270.0,
            amplitude_max=300.0,
            pulse_width_min=1.18,
            pulse_width_max=1.28,
            pulse_width_min_ramp_end_episode=105,
            epsilon_accelerate_after_steps=700,
            epsilon_accelerate_decay_steps=250,
            threshold_reward=400.0,
            alpha_beta_progress_coef=4000.0,
        )
    elif candidate == 27:
        # Candidate 27: Balanced Deep Suppression - f_max=100 Hz, A_min=262 nA, w_min=1.16 ms, ramp_end=100
        return replace(
            base,
            frequency_max=100.0,
            frequency_sensitivity=16.0,
            frequency_sensitivity_early=4.0,
            amplitude_min=262.0,
            amplitude_max=295.0,
            pulse_width_min=1.16,
            pulse_width_max=1.25,
            pulse_width_min_ramp_end_episode=100,
            epsilon_accelerate_after_steps=680,
            epsilon_accelerate_decay_steps=240,
            threshold_reward=380.0,
            alpha_beta_progress_coef=3800.0,
        )
    else:
        raise ValueError(f"Unknown candidate {candidate}")


def plot_fig6(series: dict[str, Any], out_path: Path, title_suffix: str = "") -> None:
    plt.rcParams.update(STYLE)
    ab = np.asarray(series["episode_alpha_beta_means"], dtype=float)
    amp = np.asarray(series["episode_amplitudes"], dtype=float)
    freq = np.asarray(series["episode_frequencies"], dtype=float)
    pw = np.asarray(series["episode_pulse_widths"], dtype=float)
    episodes = np.arange(ab.size, dtype=float)
    ab_smooth = moving_average(ab, 20)
    amp_smooth = moving_average(amp, 20)
    freq_smooth = moving_average(freq, 20)
    pw_smooth = moving_average(pw, 20)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.5), sharex=True)

    ax0 = axes[0]
    ax0.plot(episodes, ab, color="#9ecae1", linewidth=0.8, alpha=0.85, label="Raw")
    ax0.plot(episodes, ab_smooth, color="#08519c", linewidth=2.0, label="Smoothed")
    ax0.axhline(BIOMARKER_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2, label="θ=150")
    ax0.set_ylabel("α–β Power")
    ax0.set_title(f"GPi α–β Oscillation Power {title_suffix}".strip())
    ax0.grid(True, linestyle="--", alpha=0.6)

    # Panel (b) with three distinct y-axes: Frequency (left), Amplitude (right), Pulse width (offset right)
    ax_freq = axes[1]
    ax_amp = ax_freq.twinx()
    ax_pw = ax_freq.twinx()

    ax_pw.spines["right"].set_position(("axes", 1.14))

    # Frequency: raw + rolling average
    ax_freq.plot(episodes, freq, color="#9ecae1", linewidth=0.8, alpha=0.55, label="_freq_raw")
    ax_freq.plot(episodes, freq_smooth, color=_paper_overlay.NGUYEN_FREQ, linewidth=2.0, label="Frequency")

    # Amplitude: raw + rolling average
    ax_amp.plot(episodes, amp, color="#fb9a99", linewidth=0.8, alpha=0.55, label="_amp_raw")
    ax_amp.plot(episodes, amp_smooth, color=_paper_overlay.NGUYEN_AMP, linewidth=2.0, label="Amplitude")

    # Pulse width: raw + rolling average
    ax_pw.plot(episodes, pw, color="#a1d99b", linewidth=0.8, alpha=0.55, label="_pw_raw")
    ax_pw.plot(episodes, pw_smooth, color=_paper_overlay.NGUYEN_PW, linewidth=2.0, label="Pulse width")

    _paper_overlay.overlay_nguyen_fig6(ax0, ax_freq, ax_amp, ax_pw)

    # Style axes
    ax_freq.set_xlabel("Episode")
    ax_freq.set_ylabel("Frequency (Hz)", color=_paper_overlay.NGUYEN_FREQ)
    ax_freq.tick_params(axis="y", labelcolor=_paper_overlay.NGUYEN_FREQ)
    ax_freq.spines["left"].set_color(_paper_overlay.NGUYEN_FREQ)
    ax_freq.grid(True, linestyle="--", alpha=0.6)

    ax_amp.set_ylabel("Amplitude (nA/cm²)", color=_paper_overlay.NGUYEN_AMP)
    ax_amp.tick_params(axis="y", labelcolor=_paper_overlay.NGUYEN_AMP)
    ax_amp.spines["right"].set_color(_paper_overlay.NGUYEN_AMP)
    ax_amp.grid(False)

    ax_pw.set_ylabel("Pulse width (ms)", color=_paper_overlay.NGUYEN_PW)
    ax_pw.tick_params(axis="y", labelcolor=_paper_overlay.NGUYEN_PW)
    ax_pw.spines["right"].set_color(_paper_overlay.NGUYEN_PW)
    ax_pw.grid(False)

    ax_freq.set_title("DBS Parameters")

    # Limits with data_ylim to include replication + paper curves
    freq_curves = _paper_overlay.load_panel_curves(_paper_overlay.NGUYEN_DIG / "curves_fig6_freq.json")
    _, py_f = _paper_overlay.pick_series(freq_curves, "Smoothed", "Raw")
    amp_curves = _paper_overlay.load_panel_curves(_paper_overlay.NGUYEN_DIG / "curves_fig6_amp.json")
    _, py_a = _paper_overlay.pick_series(amp_curves, "Smoothed", "Raw")
    pw_curves = _paper_overlay.load_panel_curves(_paper_overlay.NGUYEN_DIG / "curves_fig6_pw.json")
    _, py_p = _paper_overlay.pick_series(pw_curves, "Smoothed", "Raw")

    ax_freq.set_ylim(data_ylim(freq, py_f, pad_frac=0.08))
    ax_amp.set_ylim(data_ylim(amp, py_a, pad_frac=0.08))
    ax_pw.set_ylim(data_ylim(pw, py_p, pad_frac=0.08))

    _paper_overlay.place_legend(ax0, fontsize=8)

    handles_labels = [
        (line, line.get_label())
        for line in ax_freq.get_lines() + ax_amp.get_lines() + ax_pw.get_lines()
        if line.get_label() and not line.get_label().startswith("_")
    ]
    if handles_labels:
        handles, labels = zip(*handles_labels)
        _paper_overlay.place_legend(
            ax_freq,
            handles=list(handles),
            labels=list(labels),
            fontsize=8,
            ncol=3,
            loc="upper center",
        )

    last_ep = max(0, ab.size - 1)
    for ax in (axes[0], ax_freq):
        ax.set_xlim(0.0, float(last_ep))
        if last_ep >= 100:
            ax.set_xticks(np.arange(0, last_ep + 1, 100))

    fig.subplots_adjust(left=0.10, right=0.84, top=0.95, bottom=0.08, hspace=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = get_candidate_config(args.candidate, seed=args.seed)
    if args.smoke:
        cfg = replace(cfg, num_episodes=2)

    cand_dir = OUT_DIR / f"cand{args.candidate}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    series_path = cand_dir / "series.json"
    ckpt_path = cand_dir / "checkpoint.pt"
    manifest_path = cand_dir / "manifest.json"
    fig_path = FIG_DIR / f"cand{args.candidate}_fig6.png"

    print(f"=== Starting Nguyen Fig 6 Candidate {args.candidate} (seed={args.seed}, episodes={cfg.num_episodes}) ===", flush=True)
    t0 = time.perf_counter()

    env = NguyenEnvAdapter(config=cfg)
    dsqn = DSQN(cfg)
    buffer = ReplayBuffer(config=cfg)
    trainer = DSQNTrainer(dsqn=dsqn, buffer=buffer, config=cfg)

    result = trainer.train_episodes(
        env,
        checkpoint_path=ckpt_path,
        checkpoint_interval=50,
    )
    elapsed = time.perf_counter() - t0

    series_payload = {
        "seed": cfg.seed,
        "num_episodes": cfg.num_episodes,
        "max_episode_steps": cfg.max_episode_steps,
        "episode_rewards": result.episode_rewards,
        "episode_lengths": result.episode_lengths,
        "episode_spike_totals": result.episode_spike_totals,
        "episode_energies": result.episode_energies,
        "episode_alpha_beta_means": result.episode_alpha_beta_means,
        "episode_early_stops": result.episode_early_stops,
        "episode_amplitudes": result.episode_amplitudes,
        "episode_frequencies": result.episode_frequencies,
        "episode_pulse_widths": result.episode_pulse_widths,
        "update_count": trainer.update_count,
        "smoke": args.smoke,
        "config": {
            "frequency_sensitivity": cfg.frequency_sensitivity,
            "frequency_sensitivity_early": cfg.frequency_sensitivity_early,
            "pulse_width_min": cfg.pulse_width_min,
            "threshold_reward": cfg.threshold_reward,
            "alpha_beta_progress_coef": cfg.alpha_beta_progress_coef,
        },
    }
    series_path.write_text(json.dumps(series_payload, indent=2) + "\n", encoding="utf-8")

    # Evaluate gates
    fig6_gates = fig6_training_gates(
        result.episode_alpha_beta_means,
        result.episode_amplitudes,
        result.episode_frequencies,
        result.episode_pulse_widths,
    )
    fig4_gates = fig4_training_gates(
        result.episode_rewards,
        result.episode_lengths,
        max_episode_steps=cfg.max_episode_steps,
    )
    fig5_gates = fig5_spikes_energy_gates(
        result.episode_spike_totals,
        result.episode_energies,
    )

    plot_fig6(series_payload, fig_path, title_suffix=f"(Candidate {args.candidate})")

    manifest = {
        "candidate": args.candidate,
        "num_episodes": cfg.num_episodes,
        "elapsed_s": elapsed,
        "fig6_gates": fig6_gates,
        "fig4_gates": fig4_gates,
        "fig5_gates": fig5_gates,
        "pass": bool(fig6_gates.get("pass", False)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nCandidate {args.candidate} finished in {elapsed:.1f}s. Gate pass: {manifest['pass']}", flush=True)
    print("Fig 6 Gates:", flush=True)
    for k, v in fig6_gates.get("gates", {}).items():
        print(f"  {k:45s}: {v}", flush=True)

    return 0 if manifest["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
