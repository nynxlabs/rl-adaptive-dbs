#!/usr/bin/env python3
"""Retrain DDPG at 45 Hz with 0.2s steps and 40 irregular patterns (skip_regular).

Paper-faithful setup:
- step_duration_s=0.2 (matches paper's fine-grained step function)
- 40 patterns (pattern 0 / regular periodic excluded)
- 30 steps per episode (6s training episodes)
- 10 episodes, softmax exploration
- Seed 0

Run in tmux:
  tmux new-session -d -s retrain-45hz \\
    "setsid nohup bash -c 'cd ~/neuroengineering/rl-adaptive-dbs && source .venv/bin/activate && python3 scripts/retrain_45hz_skip_regular.py >> logs/retrain-45hz.log 2>&1' < /dev/null"
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllers.ddpg import DDPGConfig, train, save_checkpoint
from controllers.ddpg.config import fig4a_ddpg_config
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

STEP_DURATION_S = 0.2
MEAN_HZ = 45.0
SEED = 0
NUM_EPISODES = 10
STEPS_PER_EPISODE = 30  # 30 × 0.2s = 6s per episode
EVAL_STEPS = 50  # 50 × 0.2s = 10s stimulation + 2s baseline = 12s display

OUTPUT_DIR = Path("artifacts/figures/papers/1/4a")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_skip_regular_02s.pt"


def main():
    t0 = time.time()
    print(f"=== RETRAIN 45Hz: skip_regular=True, step_duration={STEP_DURATION_S}s ===", flush=True)
    print(f"Patterns: 40 irregular (no regular periodic)", flush=True)
    print(f"Episodes: {NUM_EPISODES}, Steps/ep: {STEPS_PER_EPISODE} ({STEPS_PER_EPISODE * STEP_DURATION_S}s)", flush=True)
    print(f"Eval steps: {EVAL_STEPS} ({EVAL_STEPS * STEP_DURATION_S}s stimulation + 2s baseline)", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print("", flush=True)

    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=0.02)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        step_duration_s=STEP_DURATION_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=True,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=STEP_DURATION_S,
        dt_ms=plant_cfg.dt_ms,
        skip_regular=True,
    )
    plant = PythonPlant(config=plant_cfg)
    env = MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)

    print(f"Action space: {alphabet.n_actions} patterns (skip_regular=True)", flush=True)
    print(f"Step duration: {env_cfg.step_duration_s}s", flush=True)
    print(f"Plant dt: {plant_cfg.dt_ms}ms", flush=True)
    print("", flush=True)

    ddpg_cfg = fig4a_ddpg_config(
        seed=SEED,
        num_episodes=NUM_EPISODES,
        max_episode_steps=STEPS_PER_EPISODE,
    )

    print("Starting training...", flush=True)
    result = train(config=ddpg_cfg, env=env, checkpoint_path=CHECKPOINT_PATH)

    elapsed = time.time() - t0
    print(f"\n=== TRAINING COMPLETE ({elapsed:.0f}s) ===", flush=True)

    # Report action distribution
    if hasattr(result, 'episode_actions') and result.episode_actions:
        all_actions = [a for ep in result.episode_actions for a in ep]
        unique = set(all_actions)
        print(f"Unique actions seen: {len(unique)} / {alphabet.n_actions}", flush=True)
        print(f"Actions: {sorted(unique)}", flush=True)

    print(f"\nCheckpoint: {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    main()
