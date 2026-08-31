#!/usr/bin/env python3
"""Run Mehregan Fig 4a Action Persistence candidates (Option C, Iteration 2).

Candidates:
  - Candidate 1 (cand1_k3_alr9e4_warmup15): Action persistence K=3 steps,
    softmax tau 2.0 -> 1.0, actor_lr=9.0e-4, warmup=15.
  - Candidate 2 (cand2_k3_alr8e4_tau18_09): Action persistence K=3 steps,
    softmax tau 1.8 -> 0.9, actor_lr=8.0e-4, warmup=20.
  - Candidate 3 (cand3_k2_alr8e4_tau16_07): Action persistence K=2 steps,
    softmax tau 1.6 -> 0.7, actor_lr=8.5e-4, warmup=20.

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_mehregan_fig4a_candidates.py --candidate 1
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_mehregan_fig4a_candidates.py --candidate 2
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_mehregan_fig4a_candidates.py --candidate 3
"""
from __future__ import annotations

import argparse
import importlib.util
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
import torch

from controllers.ddpg.checkpoint import save_checkpoint
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.thread_limits import apply_max_threads
from rl_adaptive_dbs.user_config import resolve_config

apply_max_threads(1)

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig4a_gates  # noqa: E402

_OVERLAY = Path(__file__).resolve().parents[2] / "scripts" / "figures" / "papers" / "paper_overlay.py"
_spec = importlib.util.spec_from_file_location("figure_paper_overlay", _OVERLAY)
assert _spec and _spec.loader
_paper_overlay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paper_overlay)

ARTIFACTS_ROOT = Path("artifacts/figures/papers/mehregan/experiments")
FIGURES_ROOT = Path("figures/mehregan/images/experiments")

PAPER_DT_MS = 0.02
MEAN_HZ = 45.0
STATE_LENGTH = 1
NUM_EPISODES = 10
STEPS_PER_EPISODE = 30
DEFAULT_SEED = 0

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "font.size": 10,
}


def _make_env(
    *,
    seed: int,
    state_length: int = STATE_LENGTH,
    jitter_fraction: float = 0.5,
    plant_integration_mode: str = "continuous",
) -> tuple[MehreganEnv, Any]:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=state_length,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        plant_integration_mode=plant_integration_mode,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
        jitter_fraction=jitter_fraction,
    )
    plant = PythonPlant(config=plant_cfg)
    env = MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)
    _ = seed
    return env, plant_cfg


def run_training_persistent(
    env: MehreganEnv,
    *,
    config: DDPGConfig,
    action_persistence: int = 3,
) -> tuple[list[float], list[int], list[float], DDPGTrainer]:
    trainer = DDPGTrainer(env, config)
    beta_trace: list[float] = []
    actions: list[int] = []
    episode_rewards: list[float] = []
    env_step = 0

    for episode in range(config.num_episodes):
        state, info0 = env.reset(seed=config.seed + episode)
        trainer._update_obs_stats(state)
        episode_reward = float(info0.get("reward", 0.0))
        terminated = truncated = False
        action = None
        logits = None
        steps_remaining_on_action = 0

        while not (terminated or truncated):
            if steps_remaining_on_action <= 0 or action is None:
                action, logits = trainer._select_action(state, env_step=env_step)
                steps_remaining_on_action = action_persistence
            steps_remaining_on_action -= 1

            env_step += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            trainer._update_obs_stats(next_state)
            beta_trace.append(float(info["p_beta_norm"]))
            actions.append(int(action))
            episode_reward += float(reward)
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
        episode_rewards.append(episode_reward)
        trainer._env_step = env_step
        print(
            f"  episode {episode + 1}/{config.num_episodes} "
            f"reward={episode_reward:.2f} mean_beta={np.mean(beta_trace[-30:]):.4f}",
            flush=True,
        )

    return beta_trace, actions, episode_rewards, trainer


def analyze_roughness_and_spikes(beta_trace: list[float], actions: list[int]) -> dict[str, Any]:
    """Compute roughness, step-to-step differences, and spike counts."""
    trace = np.array(beta_trace, dtype=float)
    diffs = np.abs(np.diff(trace))
    roughness = float(np.mean(np.diff(trace, n=2)**2)) if len(trace) > 2 else float("nan")
    spikes = []
    for s in range(1, min(len(trace) - 1, 120)):
        prev_v = trace[s - 1]
        curr_v = trace[s]
        next_v = trace[s + 1]
        if (prev_v - curr_v > 0.08) and (next_v - curr_v > 0.08):
            spikes.append({
                "step": s,
                "episode": s // 30,
                "beta": float(curr_v),
                "prev_beta": float(prev_v),
                "next_beta": float(next_v),
                "action": int(actions[s]),
            })
    return {
        "mean_diff": float(diffs.mean()),
        "max_diff": float(diffs.max()),
        "roughness": roughness,
        "n_early_spikes": len(spikes),
        "spikes": spikes,
    }


def plot_candidate_trace(
    beta_trace: list[float],
    out_path: Path,
    *,
    candidate_title: str,
    gates_pass: bool,
) -> None:
    y = np.asarray(beta_trace, dtype=float)
    x = np.arange(y.size, dtype=float)
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(x, y, color="#1f77b4", linewidth=1.1, label="Replication (45 Hz)")
    _paper_overlay.overlay_mehregan_fig4a(ax)
    y0 = min(0.15, float(y.min()) * 0.95)
    y1 = max(0.65, float(y.max()) * 1.05)
    ax.set_xlim(0, 300)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Steps")
    ax.set_ylabel(r"PSD($x10^3$)")
    status_str = "PASS" if gates_pass else "FAIL"
    ax.set_title(f"{candidate_title} [{status_str}]", fontsize=10)
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    fig.tight_layout()
    _paper_overlay.place_legend(ax, loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_candidate(cand_id: int, seed: int = DEFAULT_SEED) -> int:
    print(f"============================================================")
    print(f"Starting Mehregan Fig 4a Option C Candidate {cand_id} (seed={seed})")
    print(f"============================================================", flush=True)

    cand_dir = ARTIFACTS_ROOT / f"cand{cand_id}"
    fig_dir = FIGURES_ROOT
    cand_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if cand_id == 1:
        cand_name = "Candidate 1 (cand1_k3_alr9e4_warmup15: K=3, lr=9.0e-4, warmup=15)"
        persistence = 3
        config = DDPGConfig(
            variant="paper",
            seed=seed,
            num_episodes=NUM_EPISODES,
            max_episode_steps=STEPS_PER_EPISODE,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=MEAN_HZ,
            exploration_mode="softmax",
            init_bias_scale=0.0,
            exploration_temperature_start=2.0,
            exploration_temperature_end=1.0,
            logit_noise_std=0.05,
            entropy_coeff=0.08,
            critic_action_input="one_hot",
            critic_warmup_steps=15,
            actor_lr=9.0e-4,
            log_episodes=True,
        )
    elif cand_id == 2:
        cand_name = "Candidate 2 (cand2_k3_alr8e4_tau18_09: K=3, tau 1.8->0.9, lr=8.0e-4, warmup=20)"
        persistence = 3
        config = DDPGConfig(
            variant="paper",
            seed=seed,
            num_episodes=NUM_EPISODES,
            max_episode_steps=STEPS_PER_EPISODE,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=MEAN_HZ,
            exploration_mode="softmax",
            init_bias_scale=0.0,
            exploration_temperature_start=1.8,
            exploration_temperature_end=0.9,
            logit_noise_std=0.05,
            entropy_coeff=0.08,
            critic_action_input="one_hot",
            critic_warmup_steps=20,
            actor_lr=8.0e-4,
            log_episodes=True,
        )
    elif cand_id == 3:
        cand_name = "Candidate 3 (cand3_k2_alr8e4_tau16_07: K=2, tau 1.6->0.7, lr=8.5e-4, warmup=20)"
        persistence = 2
        config = DDPGConfig(
            variant="paper",
            seed=seed,
            num_episodes=NUM_EPISODES,
            max_episode_steps=STEPS_PER_EPISODE,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=MEAN_HZ,
            exploration_mode="softmax",
            init_bias_scale=0.0,
            exploration_temperature_start=1.6,
            exploration_temperature_end=0.7,
            logit_noise_std=0.05,
            entropy_coeff=0.08,
            critic_action_input="one_hot",
            critic_warmup_steps=20,
            actor_lr=8.5e-4,
            log_episodes=True,
        )
    else:
        print(f"Unknown candidate {cand_id}", file=sys.stderr)
        return 1

    t0 = time.time()
    env, _ = _make_env(
        seed=seed,
        jitter_fraction=0.5,
        plant_integration_mode="continuous",
    )

    try:
        beta_trace, actions, episode_rewards, trainer = run_training_persistent(
            env,
            config=config,
            action_persistence=persistence,
        )
    finally:
        env.close()

    elapsed = time.time() - t0
    print(f"[{cand_name}] Training completed in {elapsed:.1f}s", flush=True)

    # Gates and Roughness analysis
    gate_pack = fig4a_gates(beta_trace)
    roughness_info = analyze_roughness_and_spikes(beta_trace, actions)

    dig_gates_pass = gate_pack.get("pass", False)
    all_pass = dig_gates_pass

    print(f"\n--- Gate Results for {cand_name} ---")
    for k, v in gate_pack["gates"].items():
        print(f"  {k}: {v}")
    print(f"  mean_diff: {roughness_info['mean_diff']:.4f} (paper=0.0062)")
    print(f"  max_diff: {roughness_info['max_diff']:.4f} (paper=0.0270)")
    print(f"  roughness: {roughness_info['roughness']:.6f} (paper=0.000068)")
    print(f"  n_early_spikes: {roughness_info['n_early_spikes']}")
    print(f"  => OVERALL GATES PASS: {all_pass}\n")

    # Plot
    png_path = fig_dir / f"training_beta_cand{cand_id}.png"
    plot_candidate_trace(beta_trace, png_path, candidate_title=cand_name, gates_pass=all_pass)
    print(f"Wrote plot to {png_path}")

    # Save cache
    series_path = cand_dir / "series.json"
    manifest_path = cand_dir / "manifest.json"
    ckpt_path = cand_dir / "checkpoint.pt"

    save_checkpoint(
        path=ckpt_path,
        actor=trainer.actor,
        config=config,
        state_length=STATE_LENGTH,
        n_actions=env.action_space.n,
        critic=trainer.critic,
        trainer=trainer,
        extra={
            "candidate": cand_id,
            "candidate_name": cand_name,
            "gates_pass": all_pass,
            "elapsed_s": elapsed,
        },
    )

    manifest_data = {
        "candidate": cand_id,
        "candidate_name": cand_name,
        "seed": seed,
        "elapsed_s": elapsed,
        "gates_pass": all_pass,
        "digitization_gates": gate_pack["gates"],
        "roughness_analysis": roughness_info,
        "gate_metrics": gate_pack.get("metrics", {}),
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    series_data = {
        "candidate": cand_id,
        "candidate_name": cand_name,
        "seed": seed,
        "elapsed_s": elapsed,
        "beta_norm_trace": beta_trace,
        "actions": actions,
        "episode_rewards": episode_rewards,
        "gates": gate_pack["gates"],
        "gates_pass": all_pass,
        "roughness_analysis": roughness_info,
    }
    series_path.write_text(json.dumps(series_data, indent=2))
    print(f"Wrote artifacts to {cand_dir}", flush=True)

    return 0 if all_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    return run_candidate(args.candidate, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
