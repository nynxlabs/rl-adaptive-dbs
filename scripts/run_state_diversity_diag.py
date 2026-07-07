#!/usr/bin/env python3
"""Quick diagnostic: does the plant produce distinguishable states?

Collects states from random actions and checks:
1. Per-element variance across states
2. Are states actually different enough for the CNN to distinguish?
3. What does the reward landscape look like as a function of state?
"""

import json
import sys
import numpy as np

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.reward import mehregan_reward
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

SEED = 0
STATE_LENGTH = 15


def main():
    resolved = resolve_config()
    config = MehreganEnvConfig(state_length=STATE_LENGTH)
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=config)

    # Strategy 1: Single-step states from different actions
    print("=== Single-step state diversity ===", flush=True)
    single_states = []
    for action in range(0, 41, 5):
        obs, _ = env.reset(seed=SEED)
        for _ in range(5):  # warm up the window a bit
            obs, _, _, _, _ = env.step(0)
        obs, _, _, _, _ = env.step(action)
        single_states.append({"action": action, "obs": obs.tolist(), "mean": float(np.mean(obs))})
        print(f"  action={action:2d} mean(obs)={np.mean(obs):.4f} obs[-3:]={obs[-3:].tolist()}", flush=True)

    states_array = np.array([s["obs"] for s in single_states])
    per_elem_std = states_array.std(axis=0)
    print(f"\n  Per-element std across actions: {per_elem_std}", flush=True)
    print(f"  Mean per-element std: {per_elem_std.mean():.6f}", flush=True)
    print(f"  Max per-element std: {per_elem_std.max():.6f}", flush=True)

    # Strategy 2: Multi-step states — 15 steps of the same action
    print("\n=== Multi-step state convergence ===", flush=True)
    for action in [0, 20, 40]:
        obs, _ = env.reset(seed=SEED)
        states = [obs.tolist()]
        for step in range(29):
            obs, _, _, _, _ = env.step(action)
            states.append(obs.tolist())
        states_arr = np.array(states)
        # Check how state evolves over the episode
        means = [np.mean(s) for s in states]
        print(f"  action={action}: mean(obs) over steps: {[f'{m:.4f}' for m in means[:5]]} ... {[f'{m:.4f}' for m in means[-3:]]}", flush=True)
        print(f"    final state range: [{states_arr[-1].min():.4f}, {states_arr[-1].max():.4f}]", flush=True)
        print(f"    final state std: {states_arr[-1].std():.6f}", flush=True)

    # Strategy 3: Compare states from different actions after convergence
    print("\n=== Converged state comparison ===", flush=True)
    converged_states = []
    for action in [0, 10, 20, 30, 40]:
        obs, _ = env.reset(seed=SEED)
        for _ in range(29):
            obs, _, _, _, _ = env.step(action)
        converged_states.append({"action": action, "obs": obs.tolist(), "mean": float(np.mean(obs))})

    conv_arr = np.array([s["obs"] for s in converged_states])
    conv_per_elem_std = conv_arr.std(axis=0)
    print(f"  Converged states per-element std: {conv_per_elem_std}", flush=True)
    print(f"  Mean per-element std: {conv_per_elem_std.mean():.6f}", flush=True)

    for s in converged_states:
        r = mehregan_reward(np.array(s["obs"]), beta_threshold=0.35, reward_scale=10.0)
        print(f"  action={s['action']:2d} mean(obs)={s['mean']:.4f} reward={r:.4f}", flush=True)

    # Strategy 4: What frequency gives p_beta closest to threshold?
    print("\n=== Optimal action search ===", flush=True)
    best_action = None
    best_dist = float("inf")
    for action in range(41):
        obs, _ = env.reset(seed=SEED)
        for _ in range(29):
            obs, _, _, _, _ = env.step(action)
        mean_obs = float(np.mean(obs))
        dist = abs(mean_obs - 0.35)
        if dist < best_dist:
            best_dist = dist
            best_action = action
        if action % 5 == 0:
            print(f"  action={action:2d} mean(obs)={mean_obs:.4f} |dist_to_0.35|={dist:.4f}", flush=True)
    print(f"\n  Best action: {best_action} (dist={best_dist:.4f})", flush=True)

    env.close()


if __name__ == "__main__":
    main()
