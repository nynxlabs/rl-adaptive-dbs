#!/usr/bin/env python3
"""TASK-104: pattern-mode reward landscape probes (1-step + 30-step rollouts).

1. 41-pattern single-step sweep → artifacts/ddpg/pattern_reward_landscape_1step.json
2. 30-step constant-policy rollouts → artifacts/ddpg/pattern_reward_landscape_30step.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACTS = Path("artifacts/ddpg")
OUT_1STEP = ARTIFACTS / "pattern_reward_landscape_1step.json"
OUT_30STEP = ARTIFACTS / "pattern_reward_landscape_30step.json"
PROBE_SEED = 42
ROLLOUT_SEED = 1000
MAX_STEPS = 30


def _make_env(*, state_length: int = 1) -> MehreganEnv:
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(
        state_length=state_length,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=45.0,
        max_episode_steps=MAX_STEPS,
    )
    plant = PythonPlant(config=resolved.plant)
    return MehreganEnv(plant=plant, config=env_cfg)


def run_1step_sweep(env: MehreganEnv, *, seed: int) -> dict:
    n_actions = env.action_space.n
    per_action: list[dict] = []
    for action in range(n_actions):
        env.reset(seed=seed)
        _obs, reward, _term, _trunc, info = env.step(action)
        per_action.append(
            {
                "action": action,
                "reward": float(reward),
                "p_beta_norm": float(info.get("p_beta_norm", 0.0)),
            }
        )
        print(f"  action {action:2d}: reward={reward:+.4f}", flush=True)

    rewards = [r["reward"] for r in per_action]
    best_idx = int(np.argmax(rewards))
    return {
        "task": "TASK-104",
        "probe": "1step_sweep",
        "seed": seed,
        "state_length": env.config.state_length,
        "action_space_mode": env.config.action_space_mode,
        "pattern_mean_hz": env.config.pattern_mean_hz,
        "n_actions": n_actions,
        "per_action": per_action,
        "best_action": best_idx,
        "best_reward": float(rewards[best_idx]),
        "worst_action": int(np.argmin(rewards)),
        "worst_reward": float(min(rewards)),
        "reward_spread": float(max(rewards) - min(rewards)),
        "pattern0_reward": float(per_action[0]["reward"]),
        "pattern0_is_global_best": best_idx == 0,
        "nonzero_beats_pattern0": [
            r["action"]
            for r in per_action
            if r["action"] != 0 and r["reward"] > per_action[0]["reward"]
        ],
    }


def _rollout_constant(env: MehreganEnv, *, seed: int, action_fn) -> dict:
    state, _info = env.reset(seed=seed)
    total_reward = 0.0
    step_rewards: list[float] = []
    actions: list[int] = []
    done = False
    while not done:
        action = int(action_fn(len(actions)))
        actions.append(action)
        state, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        step_rewards.append(float(reward))
        done = terminated or truncated
    return {
        "total_reward": total_reward,
        "mean_step_reward": float(np.mean(step_rewards)),
        "actions": actions,
        "step_rewards": step_rewards,
    }


def run_30step_rollouts(env: MehreganEnv, *, seed: int) -> dict:
    rng = np.random.default_rng(seed + 7)
    policies = {
        "pattern_0_regular": lambda _step: 0,
        "pattern_10_mid_irregular": lambda _step: 10,
        "pattern_40_max_jitter": lambda _step: 40,
        "random_irregular": lambda _step: int(rng.integers(1, 41)),
    }
    results: dict[str, dict] = {}
    for name, action_fn in policies.items():
        print(f"=== 30-step rollout: {name} ===", flush=True)
        rollout = _rollout_constant(env, seed=seed, action_fn=action_fn)
        print(
            f"  total_reward={rollout['total_reward']:+.4f} "
            f"mean_step={rollout['mean_step_reward']:+.4f}",
            flush=True,
        )
        results[name] = rollout

    p0_total = results["pattern_0_regular"]["total_reward"]
    beats_p0 = {
        name: data["total_reward"]
        for name, data in results.items()
        if name != "pattern_0_regular" and data["total_reward"] > p0_total
    }
    return {
        "task": "TASK-104",
        "probe": "30step_constant_policy",
        "seed": seed,
        "state_length": env.config.state_length,
        "action_space_mode": env.config.action_space_mode,
        "pattern_mean_hz": env.config.pattern_mean_hz,
        "max_steps": MAX_STEPS,
        "policies": results,
        "pattern0_total_reward": p0_total,
        "any_nonzero_beats_pattern0": beats_p0,
        "pattern0_is_best": len(beats_p0) == 0,
    }


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== TASK-104: 41-pattern 1-step sweep ===", flush=True)
    env = _make_env(state_length=1)
    try:
        sweep = run_1step_sweep(env, seed=PROBE_SEED)
        sweep["elapsed_s"] = round(time.time() - t0, 1)
        OUT_1STEP.write_text(json.dumps(sweep, indent=2) + "\n")
        print(f"\nWrote {OUT_1STEP}", flush=True)
        print(
            f"best_action={sweep['best_action']} "
            f"pattern0_is_global_best={sweep['pattern0_is_global_best']}",
            flush=True,
        )

        t1 = time.time()
        print("\n=== TASK-104: 30-step constant-policy rollouts ===", flush=True)
        rollouts = run_30step_rollouts(env, seed=ROLLOUT_SEED)
        rollouts["elapsed_s"] = round(time.time() - t1, 1)
        OUT_30STEP.write_text(json.dumps(rollouts, indent=2) + "\n")
        print(f"\nWrote {OUT_30STEP}", flush=True)
        print(
            f"pattern0_is_best_30step={rollouts['pattern0_is_best']} "
            f"beaters={list(rollouts['any_nonzero_beats_pattern0'].keys())}",
            flush=True,
        )
    finally:
        env.close()

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    from scripts.lib.train_runtime_guard import run_main

    sys.exit(run_main(main, label="task104-pattern-probes"))
