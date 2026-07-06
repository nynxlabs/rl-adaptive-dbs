#!/usr/bin/env python3
"""Sweep STN pattern actions and record P_beta / reward trajectories (TASK-67 plant diag)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.patterns import PatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config


@dataclass
class ActionSweep:
    action: int
    frequency_hz: float
    steps: list[dict[str, float]]
    mean_p_beta_raw: float
    mean_p_beta_norm: float
    mean_reward: float
    final_p_beta_raw: float
    final_p_beta_norm: float


def run_sweep(
    *,
    seed: int,
    steps_per_action: int,
    actions: list[int] | None,
    state_length: int,
) -> dict:
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(state_length=state_length)
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    alphabet = PatternAlphabet()
    action_list = actions if actions is not None else list(range(alphabet.n_actions))
    results: list[ActionSweep] = []

    try:
        for action in action_list:
            env.reset(seed=seed)
            step_rows: list[dict[str, float]] = []
            for _ in range(steps_per_action):
                _obs, reward, _term, _trunc, info = env.step(action)
                step_rows.append(
                    {
                        "p_beta_raw": float(info["p_beta_raw"]),
                        "p_beta_norm": float(info["p_beta_norm"]),
                        "reward": float(reward),
                        "dbs_freq_hz": float(info["dbs_freq_hz"]),
                    }
                )
            p_raw = [r["p_beta_raw"] for r in step_rows]
            p_norm = [r["p_beta_norm"] for r in step_rows]
            rewards = [r["reward"] for r in step_rows]
            spec = alphabet.to_dbs_spec(action)
            results.append(
                ActionSweep(
                    action=action,
                    frequency_hz=spec.frequency_hz,
                    steps=step_rows,
                    mean_p_beta_raw=float(np.mean(p_raw)),
                    mean_p_beta_norm=float(np.mean(p_norm)),
                    mean_reward=float(np.mean(rewards)),
                    final_p_beta_raw=float(p_raw[-1]),
                    final_p_beta_norm=float(p_norm[-1]),
                )
            )
    finally:
        env.close()

    mean_norm = [r.mean_p_beta_norm for r in results]
    mean_raw = [r.mean_p_beta_raw for r in results]
    mean_rew = [r.mean_reward for r in results]
    final_norm = [r.final_p_beta_norm for r in results]

    return {
        "seed": seed,
        "state_length": state_length,
        "steps_per_action": steps_per_action,
        "n_actions": len(results),
        "summary": {
            "p_beta_norm_mean_min": float(np.min(mean_norm)),
            "p_beta_norm_mean_max": float(np.max(mean_norm)),
            "p_beta_norm_mean_span": float(np.max(mean_norm) - np.min(mean_norm)),
            "p_beta_raw_mean_min": float(np.min(mean_raw)),
            "p_beta_raw_mean_max": float(np.max(mean_raw)),
            "p_beta_raw_mean_span": float(np.max(mean_raw) - np.min(mean_raw)),
            "reward_mean_min": float(np.min(mean_rew)),
            "reward_mean_max": float(np.max(mean_rew)),
            "reward_mean_span": float(np.max(mean_rew) - np.min(mean_rew)),
            "final_p_beta_norm_min": float(np.min(final_norm)),
            "final_p_beta_norm_max": float(np.max(final_norm)),
            "final_p_beta_norm_span": float(np.max(final_norm) - np.min(final_norm)),
            "best_reward_action": int(results[int(np.argmax(mean_rew))].action),
            "worst_reward_action": int(results[int(np.argmin(mean_rew))].action),
            "lowest_p_beta_action": int(results[int(np.argmin(mean_norm))].action),
            "highest_p_beta_action": int(results[int(np.argmax(mean_norm))].action),
        },
        "actions": [asdict(r) for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5, help="Steps per action (same action held)")
    parser.add_argument("--state-length", type=int, default=15)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ddpg/plant_response_sweep.json"),
    )
    args = parser.parse_args()

    payload = run_sweep(
        seed=args.seed,
        steps_per_action=args.steps,
        actions=None,
        state_length=args.state_length,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    s = payload["summary"]
    print(json.dumps(s, indent=2))
    print(
        f"p_beta_norm span={s['p_beta_norm_mean_span']:.4f} "
        f"reward span={s['reward_mean_span']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
