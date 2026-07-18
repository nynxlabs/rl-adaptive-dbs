#!/usr/bin/env python3
"""Deprecated — use ``scripts/figures/papers/1/5a/plot.py`` (skip_regular is default).

Quick plot: skip_regular 45Hz eval — 3 conditions (no stim, trained irregular, regular periodic).

Generates a step-function P_beta comparison matching the paper's fig 5a layout.
Uses 0.2s step duration, seed 0.

Run:
  cd ~/neuroengineering/rl-adaptive-dbs && source .venv/bin/activate
  python3 scripts/figures/papers/1/5a/plot_skip_regular.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from controllers.ddpg import evaluate
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.mehregan.env import MehreganEnv
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.patterns import PatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

CHECKPOINT = Path("artifacts/figures/papers/1/4a/checkpoint_skip_regular_02s.pt")
OUT_DIR = Path("figures/papers/1/5a")
OUT_FILE = OUT_DIR / "efficacy_45hz_skip_regular.png"

STEP_DURATION_S = 0.2
MEAN_HZ = 45.0
SEED = 0
EVAL_STEPS = 50  # 50 × 0.2s = 10s stimulation
STIM_ONSET_STEP = 10  # 10 × 0.2s = 2s baseline

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.size": 10,
}


def _make_irr_env():
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=0.02)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        step_duration_s=STEP_DURATION_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=EVAL_STEPS + STIM_ONSET_STEP,
        skip_regular=True,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ, step_duration_s=STEP_DURATION_S,
        dt_ms=plant_cfg.dt_ms, skip_regular=True,
    )
    return MehreganEnv(
        plant=PythonPlant(config=plant_cfg), config=env_cfg, alphabet=alphabet,
    )


def _make_reg_env():
    """Env with all 41 patterns (for regular periodic baseline)."""
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=0.02)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        step_duration_s=STEP_DURATION_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=EVAL_STEPS + STIM_ONSET_STEP,
        skip_regular=False,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ, step_duration_s=STEP_DURATION_S,
        dt_ms=plant_cfg.dt_ms, skip_regular=False,
    )
    return MehreganEnv(
        plant=PythonPlant(config=plant_cfg), config=env_cfg, alphabet=alphabet,
    )


def _make_nostim_env():
    """Env with scalar frequency action space (action 0 = no stim)."""
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=0.02)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        step_duration_s=STEP_DURATION_S,
        action_space_mode="scalar_frequency",
        max_episode_steps=EVAL_STEPS + STIM_ONSET_STEP,
    )
    return MehreganEnv(
        plant=PythonPlant(config=plant_cfg), config=env_cfg, alphabet=PatternAlphabet(),
    )


def _run_series(env, *, actions: list[int], seed: int) -> list[float]:
    """Run a series of actions and return per-step raw P_beta."""
    env.reset(seed=seed)
    pbs = []
    for a in actions:
        obs, reward, term, trunc, info = env.step(a)
        pbs.append(info["p_beta_raw"])
    return pbs


def _run_trained_series(checkpoint_path: str, *, seed: int, total_steps: int) -> list[float]:
    """Load checkpoint actor and run greedy actions for total_steps."""
    import torch
    from controllers.ddpg import load_actor
    actor, _cfg = load_actor(checkpoint_path)
    actor.eval()

    env = _make_irr_env()
    obs, info = env.reset(seed=seed)
    pbs = []
    for _ in range(total_steps):
        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = actor(state_t)
            action = int(logits.argmax(dim=-1).item())
        obs, reward, term, trunc, info = env.step(action)
        pbs.append(info["p_beta_raw"])
    return pbs


def main():
    plt.rcParams.update(STYLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_steps = STIM_ONSET_STEP + EVAL_STEPS  # 10 + 50 = 60 steps
    time_axis = np.arange(total_steps) * STEP_DURATION_S  # 0.0 to 12.0s
    stim_onset_s = STIM_ONSET_STEP * STEP_DURATION_S  # 2.0s

    # --- 1. No stimulation (action 0 = 0 Hz) ---
    print("Running no-stim baseline...", flush=True)
    env_nostim = _make_nostim_env()
    pbs_nostim = _run_series(env_nostim, actions=[0] * total_steps, seed=SEED)

    # --- 2. Regular periodic 45 Hz (pattern 0) ---
    print("Running regular periodic 45Hz...", flush=True)
    env_reg = _make_reg_env()
    pbs_reg = _run_series(env_reg, actions=[0] * total_steps, seed=SEED)

    # --- 3. Trained irregular (skip_regular checkpoint) ---
    print("Evaluating trained policy (greedy, 60 steps)...", flush=True)
    pbs_trained = _run_trained_series(str(CHECKPOINT), seed=SEED, total_steps=total_steps)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(time_axis[:STIM_ONSET_STEP], pbs_nostim[:STIM_ONSET_STEP],
            color="black", linewidth=1.0, alpha=0.5)
    ax.plot(time_axis, pbs_nostim,
            color="black", linewidth=1.5, label="PD no stim")
    ax.plot(time_axis, pbs_reg,
            color="#ff7f0e", linewidth=1.5, label=f"Periodic {int(MEAN_HZ)} Hz")
    ax.plot(time_axis, pbs_trained,
            color="#2ca02c", linewidth=1.5, label=f"Trained {int(MEAN_HZ)} Hz (irregular)")

    ax.axvline(stim_onset_s, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPi $P_\\beta$ (raw PSD)")
    ax.set_title("Fig 5a — Post-train efficacy @ 45 Hz (skip_regular, 0.2s steps)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, time_axis[-1])

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_FILE}", flush=True)

    # Print summary
    trained_mean = np.mean(pbs_trained[STIM_ONSET_STEP:])
    reg_mean = np.mean(pbs_reg[STIM_ONSET_STEP:])
    nostim_mean = np.mean(pbs_nostim[STIM_ONSET_STEP:])
    print(f"\n=== Summary (post-onset means) ===")
    print(f"No stim:         {nostim_mean:.1f}")
    print(f"Regular 45 Hz:   {reg_mean:.1f}")
    print(f"Trained (irreg): {trained_mean:.1f}")
    print(f"Trained > Regular? {trained_mean > reg_mean} {'✅' if trained_mean > reg_mean else '❌'}")
    print(f"Trained < No stim? {trained_mean < nostim_mean} {'✅' if trained_mean < nostim_mean else '❌'}")


if __name__ == "__main__":
    main()
