#!/usr/bin/env python3
"""Diagnostic: reward landscape and state diversity under random policy."""

import os, sys, json, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from envs.mehregan.env import MehreganEnv
from envs.mehregan.config import MehreganEnvConfig
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

resolved = resolve_config()
env_cfg = MehreganEnvConfig(state_length=15)
plant = PythonPlant(config=resolved.plant)
env = MehreganEnv(plant=plant, config=env_cfg)

# 1. Random episode
state, info = env.reset(seed=0)
print(f"state_shape={state.shape}")
print(f"obs_scale={env_cfg.observation_scale} beta_thresh={env_cfg.beta_threshold}")
print(f"init_state: {[round(float(x), 6) for x in state]}")

states = [state]
actions_taken = []
rewards = []
infos = [info]

for step in range(30):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    states.append(next_state)
    actions_taken.append(action)
    rewards.append(reward)
    infos.append(info)
    if terminated or truncated:
        break

states = np.array(states)
print(f"\nstate_range=[{states.min():.6f}, {states.max():.6f}]")
for i in range(min(10, len(states))):
    print(f"  step{i}: mean={states[i].mean():.6f} std={states[i].std():.6f}")

pbetas = [x["p_beta_raw"] for x in infos]
print(f"\np_beta_raw: {[round(p, 1) for p in pbetas[:15]]}")
print(f"rewards: {[round(r, 3) for r in rewards[:15]]}")
print(f"reward_range=[{min(rewards):.3f}, {max(rewards):.3f}] reward_std={np.std(rewards):.6f}")
print(f"unique_actions={sorted(set(actions_taken))}")

# 2. Action comparison: same starting state, different actions (reuse single env)
print("\n=== ACTION COMPARISON ===")
plant2 = PythonPlant(config=resolved.plant)
env_cfg2 = MehreganEnvConfig(state_length=15)
env2 = MehreganEnv(plant=plant2, config=env_cfg2)
s2, _ = env2.reset(seed=42)
# 5 warmup steps with 45Hz (action 27)
for _ in range(5):
    env2.step(27)

# Get current state snapshot
pre_state = env2._obs_window.copy()
pre_state_arr = np.array(pre_state, dtype=np.float32)
print(f"pre_action_state: mean={pre_state_arr.mean():.6f}")

# Now try 10 different actions from this state (reset to same point each time)
# Actually we can't reset to mid-episode. So let's just do one action at a time
# by saving and restoring, OR we note that the P_beta for this step is what matters
# The reward depends on the OBSERVATION (window) which includes the new p_beta

# Let's do: from the current state, step once with each of several actions
# But we can only step once. So let's try the same seed, same warmup, different action
action_results = {}
for a in [0, 5, 10, 15, 20, 25, 27, 30, 35, 40, 45, 49]:
    plant_t = PythonPlant(config=resolved.plant)
    env_t = MehreganEnv(plant=plant_t, config=MehreganEnvConfig(state_length=15))
    s_t, _ = env_t.reset(seed=42)
    for _ in range(5):
        env_t.step(27)
    s_next, r, t, tr, info = env_t.step(a)
    action_results[a] = {"reward": round(float(r), 4), "p_beta": round(info["p_beta_raw"], 1), "state_mean": round(float(s_next.mean()), 6)}
    env_t.close()

for a in sorted(action_results.keys()):
    ar = action_results[a]
    print(f"  action={a:>2d} reward={ar['reward']:.4f} p_beta={ar['p_beta']:.1f} state_mean={ar['state_mean']:.6f}")

rews = [action_results[a]["reward"] for a in action_results]
print(f"reward spread across actions: {max(rews) - min(rews):.6f}")

env2.close()
env.close()
print("\nDIAGNOSTIC COMPLETE")
