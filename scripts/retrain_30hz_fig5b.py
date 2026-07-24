#!/usr/bin/env python3
"""Retrain DDPG at 30 Hz for Fig 5b (burst alphabet; pattern 0 kept).

Paper-faithful stack (matches passing Fig 4a profile):
- step_duration_s=0.2, state_length=1, fixed_mean_pattern
- 10 episodes × 30 steps, softmax + one_hot critic, seed 0
- **BurstPatternAlphabet** (Fig 5b convention): fixed 30 Hz mean pulse count,
  patterns 1–40 are high-rate clusters (60–120 Hz) + silence. Open-loop oracle
  at dt=0.02: 32/41 beat no-stim (vs 0/41 for ±1/3 ISI jitter).

Run in tmux:
  tmux new-session -d -s retrain-30hz \\
    "setsid nohup uv run python scripts/retrain_30hz_fig5b.py >> logs/retrain-30hz.log 2>&1 < /dev/null"
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllers.ddpg import train
from controllers.ddpg.config import fig4a_ddpg_config
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

STEP_DURATION_S = 0.2
MEAN_HZ = 30.0
SEED = 0
NUM_EPISODES = 10
STEPS_PER_EPISODE = 30
PAPER_DT_MS = 0.02
ALPHABET_NAME = "burst"  # BurstPatternAlphabet — Fig 5b redesign

OUTPUT_DIR = Path("artifacts/figures/papers/1/5b")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"


def main() -> None:
    t0 = time.time()
    print(f"=== RETRAIN 30Hz Fig 5b: step_duration={STEP_DURATION_S}s ===", flush=True)
    print(
        f"Alphabet: {ALPHABET_NAME} (41 patterns; pattern 0 = regular 30 Hz)",
        flush=True,
    )
    print(f"Episodes: {NUM_EPISODES}, Steps/ep: {STEPS_PER_EPISODE}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print("", flush=True)

    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        step_duration_s=STEP_DURATION_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=False,
    )
    alphabet = BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=STEP_DURATION_S,
        dt_ms=plant_cfg.dt_ms,
    )
    env = MehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=alphabet,
    )

    print(f"Action space: {alphabet.n_actions} patterns", flush=True)
    print(f"Plant dt: {plant_cfg.dt_ms}ms", flush=True)
    print("", flush=True)

    ddpg_cfg = fig4a_ddpg_config(
        seed=SEED,
        num_episodes=NUM_EPISODES,
        max_episode_steps=STEPS_PER_EPISODE,
        pattern_mean_hz=MEAN_HZ,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting training...", flush=True)
    result = train(config=ddpg_cfg, env=env, checkpoint_path=CHECKPOINT_PATH)

    elapsed = time.time() - t0
    print(f"\n=== TRAINING COMPLETE ({elapsed:.0f}s) ===", flush=True)
    if hasattr(result, "episode_actions") and result.episode_actions:
        all_actions = [a for ep in result.episode_actions for a in ep]
        print(f"Unique actions: {len(set(all_actions))} / {alphabet.n_actions}", flush=True)
    print(f"Checkpoint: {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    main()
