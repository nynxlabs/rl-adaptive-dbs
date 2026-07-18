#!/usr/bin/env python3
"""Fig 4a diagnostic — three traces on one panel.

Compares per-step PSD(x10³) over 300 training steps (10×30):

  1. **No STN DBS** — open-loop rollout, DbsSpec.none() every step
  2. **Pattern 0 (regular 45 Hz)** — open-loop, action 0 every step
  3. **DDPG + softmax exploration** — online training (45 Hz init)

Run:
  uv run python scripts/probes/fig4a_three_trace_diagnostic.py
  uv run python scripts/probes/fig4a_three_trace_diagnostic.py --episodes 5  # faster
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from controllers.ddpg.config import fig4a_ddpg_config
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.dbs import DbsSpec
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACTS = Path("artifacts/probes")
OUT_PNG = Path("figures/papers/1/4a/three_trace_diagnostic.png")
OUT_JSON = ARTIFACTS / "fig4a_three_trace_diagnostic.json"

PAPER_DT_MS = 0.02
MEAN_HZ = 45.0
STEPS_PER_EPISODE = 30
DEFAULT_SEED = 0
SCALE = 1000.0

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
}
SERIES = {
    "no_stim": {"label": "No STN DBS (open loop)", "color": "#d62728", "linewidth": 1.0},
    "pattern_0": {"label": "Pattern 0 — regular 45 Hz (open loop)", "color": "#ff7f0e", "linewidth": 1.0},
    "explore_train": {"label": "DDPG + softmax exploration", "color": "#1f6f6f", "linewidth": 1.2},
}


def _make_env(*, seed: int) -> tuple[MehreganEnv, Any]:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
    )
    plant = PythonPlant(config=plant_cfg)
    env = MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)
    return env, plant_cfg


def rollout_no_stim(env: MehreganEnv, *, seed: int, num_episodes: int) -> list[float]:
    trace: list[float] = []
    for episode in range(num_episodes):
        _obs, info = env.reset(seed=seed + episode)
        for _ in range(STEPS_PER_EPISODE):
            result = env._integrate_segment(DbsSpec.none())
            trace.append(float(result.p_beta) / SCALE)
    return trace


def rollout_pattern_0(env: MehreganEnv, *, seed: int, num_episodes: int) -> list[float]:
    trace: list[float] = []
    for episode in range(num_episodes):
        env.reset(seed=seed + episode)
        for _ in range(STEPS_PER_EPISODE):
            _obs, _r, _term, _trunc, info = env.step(0)
            trace.append(float(info["p_beta_norm"]))
    return trace


def train_explore_trace(
    env: MehreganEnv,
    *,
    seed: int,
    num_episodes: int,
) -> tuple[list[float], list[int], dict[str, Any]]:
    config = fig4a_ddpg_config(
        seed=seed,
        num_episodes=num_episodes,
        max_episode_steps=STEPS_PER_EPISODE,
    )
    trainer = DDPGTrainer(env, config)
    beta_trace: list[float] = []
    actions: list[int] = []
    env_step = trainer._random_warmup()

    for episode in range(num_episodes):
        state, _info = env.reset(seed=seed + episode)
        trainer._update_obs_stats(state)
        terminated = truncated = False
        while not (terminated or truncated):
            action, logits = trainer._select_action(state, env_step=env_step)
            env_step += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            trainer._update_obs_stats(next_state)
            beta_trace.append(float(info["p_beta_norm"]))
            actions.append(int(action))
            normalized_reward = trainer._normalize_reward(reward)
            dw = float(info.get("dw", 1.0 if truncated else 0.0))
            trainer.buffer.add(
                state=state,
                action=action,
                action_logits=logits,
                reward=normalized_reward,
                next_state=next_state,
                dw=dw,
            )
            state = next_state
            if len(trainer.buffer) >= config.min_buffer_size:
                for _ in range(config.update_frequency):
                    trainer._update_step()

    unique = len(set(actions))
    dominant = int(max(set(actions), key=actions.count)) if actions else -1
    meta = {
        "unique_actions": unique,
        "dominant_action": dominant,
        "dominant_fraction": actions.count(dominant) / len(actions) if actions else 0.0,
        "exploration_mode": config.exploration_mode,
        "temperature_start": config.exploration_temperature_start,
        "temperature_end": config.exploration_temperature_end,
        "logit_noise_std": config.logit_noise_std,
    }
    return beta_trace, actions, meta


def _window_means(trace: list[float], window: int = 30) -> tuple[float, float]:
    arr = np.asarray(trace, dtype=float)
    if arr.size < window:
        return float("nan"), float("nan")
    return float(arr[:window].mean()), float(arr[-window:].mean())


def plot_panel(traces: dict[str, list[float]], *, out_path: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    steps = np.arange(max(len(t) for t in traces.values()))
    ax.set_xlim(0, max(300, int(steps[-1]) if steps.size else 300))
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSD(x10³)")
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.8)
    ax.axhline(0.375, color="#888888", linestyle=":", linewidth=0.8, label="paper ~end (0.375)")
    ax.axhline(0.5, color="#aaaaaa", linestyle=":", linewidth=0.8, label="paper ~start (0.5)")

    for key, meta in SERIES.items():
        y = np.asarray(traces[key], dtype=float)
        x = np.arange(y.size)
        ax.plot(x, y, color=meta["color"], linewidth=meta["linewidth"], label=meta["label"])

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--skip-train", action="store_true", help="Only rollouts (fast)")
    args = parser.parse_args()

    expected = args.episodes * STEPS_PER_EPISODE
    print(f"Fig 4a three-trace diagnostic — {args.episodes} episodes ({expected} steps)", flush=True)

    # --- rollouts ---
    env, plant_cfg = _make_env(seed=args.seed)
    t0 = time.time()
    try:
        print("  [1/3] no STN DBS rollout...", flush=True)
        no_stim = rollout_no_stim(env, seed=args.seed, num_episodes=args.episodes)
        print(f"        mean={np.mean(no_stim):.3f}  first={no_stim[0]:.3f}", flush=True)

        print("  [2/3] pattern 0 open loop...", flush=True)
        pattern_0 = rollout_pattern_0(env, seed=args.seed, num_episodes=args.episodes)
        print(f"        mean={np.mean(pattern_0):.3f}  first={pattern_0[0]:.3f}", flush=True)
    finally:
        env.close()

    explore_train: list[float] = []
    train_meta: dict[str, Any] = {"skipped": True}
    train_actions: list[int] = []

    if not args.skip_train:
        env2, _ = _make_env(seed=args.seed)
        try:
            print("  [3/3] DDPG + softmax training...", flush=True)
            explore_train, train_actions, train_meta = train_explore_trace(
                env2, seed=args.seed, num_episodes=args.episodes,
            )
            train_meta["skipped"] = False
            print(
                f"        mean={np.mean(explore_train):.3f}  "
                f"unique_actions={train_meta['unique_actions']}  "
                f"dominant={train_meta['dominant_action']}",
                flush=True,
            )
        finally:
            env2.close()

    traces = {"no_stim": no_stim, "pattern_0": pattern_0}
    if explore_train:
        traces["explore_train"] = explore_train

    plot_panel(traces, out_path=args.out_png)

    payload: dict[str, Any] = {
        "figure": "fig4a_three_trace_diagnostic",
        "seed": args.seed,
        "num_episodes": args.episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "plant_dt_ms": plant_cfg.dt_ms,
        "mean_hz": MEAN_HZ,
        "elapsed_s": time.time() - t0,
        "traces": traces,
        "summary": {
            "no_stim": {
                "n": len(no_stim),
                "mean": float(np.mean(no_stim)),
                "first": float(no_stim[0]),
                "last": float(no_stim[-1]),
                "start_window_mean": _window_means(no_stim)[0],
                "end_window_mean": _window_means(no_stim)[1],
            },
            "pattern_0": {
                "n": len(pattern_0),
                "mean": float(np.mean(pattern_0)),
                "first": float(pattern_0[0]),
                "last": float(pattern_0[-1]),
                "start_window_mean": _window_means(pattern_0)[0],
                "end_window_mean": _window_means(pattern_0)[1],
            },
        },
        "training": train_meta,
        "output_png": str(args.out_png),
    }
    if explore_train:
        payload["summary"]["explore_train"] = {
            "n": len(explore_train),
            "mean": float(np.mean(explore_train)),
            "first": float(explore_train[0]),
            "last": float(explore_train[-1]),
            "start_window_mean": _window_means(explore_train)[0],
            "end_window_mean": _window_means(explore_train)[1],
            "trend_down": _window_means(explore_train)[1] < _window_means(explore_train)[0],
        }
        payload["train_actions"] = train_actions

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out_png}", flush=True)
    print(f"wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
