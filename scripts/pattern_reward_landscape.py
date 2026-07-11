#!/usr/bin/env python3
"""Sweep all fixed-mean patterns for single-step reward / beta (TASK-105).

Runs one 2 s RL step per pattern (41 actions at 45 Hz mean) from a common
plant reset, ranks outcomes vs pattern 0 (regular periodic train — paper init,
not no-stimulation).

Run: uv run python scripts/pattern_reward_landscape.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config


def describe_pattern(action: int, *, mean_hz: float) -> str:
    """Human-readable semantics for reporting (pattern 0 is stimulated, not off)."""
    if action == 0:
        return (
            f"pattern 0: regular {mean_hz:g} Hz periodic train "
            "(paper init target; NOT no-stimulation)"
        )
    return f"pattern {action}: irregular train at {mean_hz:g} Hz mean rate"


@dataclass
class PatternStep:
    action: int
    semantics: str
    reward: float
    p_beta_raw: float
    p_beta_norm: float
    dbs_freq_hz: float
    reward_delta_vs_pattern0: float
    p_beta_norm_delta_vs_pattern0: float


def run_landscape(
    *,
    seed: int,
    mean_hz: float,
    state_length: int,
) -> dict:
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(
        state_length=state_length,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=mean_hz,
    )
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    alphabet = env.alphabet
    if not isinstance(alphabet, FixedMeanPatternAlphabet):
        msg = "expected FixedMeanPatternAlphabet in fixed_mean_pattern mode"
        raise TypeError(msg)

    # Pre-warm idbs cache so the sweep timing reflects plant integration only.
    for i in range(alphabet.n_actions):
        alphabet.idbs_for_pattern(i)

    rows: list[PatternStep] = []
    try:
        for action in range(alphabet.n_actions):
            env.reset(seed=seed)
            _obs, reward, _term, _trunc, info = env.step(action)
            rows.append(
                PatternStep(
                    action=action,
                    semantics=describe_pattern(action, mean_hz=mean_hz),
                    reward=float(reward),
                    p_beta_raw=float(info["p_beta_raw"]),
                    p_beta_norm=float(info["p_beta_norm"]),
                    dbs_freq_hz=float(info["dbs_freq_hz"]),
                    reward_delta_vs_pattern0=0.0,
                    p_beta_norm_delta_vs_pattern0=0.0,
                )
            )
            print(
                f"  action {action:2d}: reward={reward:+.4f} "
                f"p_beta_norm={float(info['p_beta_norm']):.4f}",
                flush=True,
            )
    finally:
        env.close()

    p0 = rows[0]
    for row in rows:
        row.reward_delta_vs_pattern0 = row.reward - p0.reward
        row.p_beta_norm_delta_vs_pattern0 = row.p_beta_norm - p0.p_beta_norm

    rewards = [r.reward for r in rows]
    betas = [r.p_beta_norm for r in rows]
    reward_order = np.argsort(rewards)[::-1]  # higher reward is better
    beta_order = np.argsort(betas)  # lower beta is better

    def rank_of(action: int, order: np.ndarray) -> int:
        return int(np.where(order == action)[0][0]) + 1

    patterns = []
    for row in rows:
        entry = asdict(row)
        entry["reward_rank"] = rank_of(row.action, reward_order)
        entry["p_beta_norm_rank"] = rank_of(row.action, beta_order)
        patterns.append(entry)

    best_reward_action = int(reward_order[0])
    best_beta_action = int(beta_order[0])

    return {
        "task": "TASK-105",
        "seed": seed,
        "mean_hz": mean_hz,
        "state_length": state_length,
        "n_patterns": alphabet.n_actions,
        "pattern0_semantics": describe_pattern(0, mean_hz=mean_hz),
        "summary": {
            "pattern0_reward": float(p0.reward),
            "pattern0_p_beta_norm": float(p0.p_beta_norm),
            "pattern0_reward_rank": rank_of(0, reward_order),
            "pattern0_p_beta_norm_rank": rank_of(0, beta_order),
            "best_reward_action": best_reward_action,
            "best_reward": float(rewards[best_reward_action]),
            "best_reward_semantics": describe_pattern(best_reward_action, mean_hz=mean_hz),
            "lowest_p_beta_action": best_beta_action,
            "lowest_p_beta_norm": float(betas[best_beta_action]),
            "reward_span": float(max(rewards) - min(rewards)),
            "p_beta_norm_span": float(max(betas) - min(betas)),
            "irregular_beat_pattern0_on_reward": best_reward_action != 0,
            "pattern0_is_best_reward": best_reward_action == 0,
        },
        "patterns": patterns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean-hz", type=float, default=45.0)
    parser.add_argument("--state-length", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ddpg/pattern_reward_landscape.json"),
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=== Pattern reward landscape (41 patterns, 1 step each) ===", flush=True)
    payload = run_landscape(
        seed=args.seed,
        mean_hz=args.mean_hz,
        state_length=args.state_length,
    )
    payload["elapsed_s"] = round(time.time() - t0, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    s = payload["summary"]
    print(json.dumps(s, indent=2), flush=True)
    print(
        f"reward_span={s['reward_span']:.4f} "
        f"pattern0_rank={s['pattern0_reward_rank']}/{payload['n_patterns']} "
        f"elapsed={payload['elapsed_s']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
