#!/usr/bin/env python3
"""Pattern-mode DDPG training — v2 with exploration fix (TASK-83).

v1 collapsed to unique_actions=1 because temp=1.0 over 41 actions produced
near-uniform softmax, making the Q-gradient too weak to overcome random
logit bias.

Fix strategy:
  - Temperature decay 5.0 → 0.3 (starts exploratory, converges to sharp policy)
  - Logit noise std=0.1 (maintains diversity during early training)
  - Entropy regularization coeff=0.01 (penalizes premature logit collapse)
  - Random warmup 200 steps (seeds replay buffer with diverse transitions)
  - 50 episodes (more time for 41-action Q-learning to converge)

Run: uv run python scripts/run_pattern_train_v2.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.networks import Actor
from controllers.ddpg.trainer import DDPGTrainer, TrainMetrics
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from scripts.pattern_reward_landscape import describe_pattern
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACTS = Path("artifacts/ddpg")
CHECKPOINT = ARTIFACTS / "pattern_train_v2.pt"
RESULTS = ARTIFACTS / "pattern_train_v2.json"
FINAL_RESULTS = ARTIFACTS / "pattern_train_v2_final.json"
LOGFILE = ARTIFACTS / "pattern_train_v2.log"


def _save_checkpoint(trainer: DDPGTrainer, episode: int) -> None:
    torch.save({
        "episode": episode,
        "actor": trainer.actor.state_dict(),
        "critic": trainer.critic.state_dict(),
        "actor_target": trainer.actor_target.state_dict(),
        "critic_target": trainer.critic_target.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "buffer": trainer.buffer,
        "obs_count": trainer._obs_count,
        "obs_mean": trainer._obs_mean,
        "obs_m2": trainer._obs_m2,
        "warmup_steps_done": trainer._warmup_steps_done,
    }, CHECKPOINT)


def _load_checkpoint(trainer: DDPGTrainer) -> int:
    if not CHECKPOINT.exists():
        return 0
    ckpt = torch.load(CHECKPOINT, map_location=trainer.device, weights_only=False)
    trainer.actor.load_state_dict(ckpt["actor"])
    trainer.critic.load_state_dict(ckpt["critic"])
    trainer.actor_target.load_state_dict(ckpt["actor_target"])
    trainer.critic_target.load_state_dict(ckpt["critic_target"])
    trainer.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    trainer.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    trainer.buffer = ckpt["buffer"]
    trainer._obs_count = ckpt["obs_count"]
    trainer._obs_mean = ckpt["obs_mean"]
    trainer._obs_m2 = ckpt["obs_m2"]
    trainer._warmup_steps_done = ckpt["warmup_steps_done"]
    return ckpt["episode"] + 1


def _analyze_policy(actor: Actor, state_length: int, *, n_samples: int = 1000) -> dict:
    actions = []
    for _ in range(n_samples):
        state = np.random.randn(state_length).astype(np.float32) * 0.1 + 0.4
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = actor(state_t)
            action = int(torch.argmax(logits, dim=-1).item())
            actions.append(action)
    from collections import Counter
    counts = Counter(actions)
    return {
        "unique_actions_offline": len(counts),
        "dominant_action": counts.most_common(1)[0][0],
        "dominant_fraction": counts.most_common(1)[0][1] / n_samples,
        "action_distribution": dict(sorted(counts.items())),
    }


def _rollout_actions(env: MehreganEnv, actor: Actor, seed: int) -> list[int]:
    state, _ = env.reset(seed=seed)
    actions = []
    done = False
    while not done:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = actor(state_t)
            action = int(torch.argmax(logits, dim=-1).item())
        actions.append(action)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return actions


def _append_result(result: dict) -> None:
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text())
    else:
        data = []
    data.append(result)
    RESULTS.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    resolved = resolve_config()
    mean_hz = 45.0
    num_episodes = 20  # first pass — extend to 50 if unique_actions > 1 by ep 15

    env_cfg = MehreganEnvConfig(
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=mean_hz,
    )
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    alphabet = env.alphabet

    try:
        config = DDPGConfig(
            variant="paper",
            seed=0,
            num_episodes=num_episodes,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=mean_hz,
            # Exploration fix: temperature decay 5.0 → 0.3
            exploration_mode="softmax",
            exploration_temperature_start=5.0,
            exploration_temperature_end=0.3,
            # Critic: one_hot for correct Q-targets under exploration
            critic_action_input="one_hot",
            critic_warmup_steps=100,
            # Reward
            reward_normalize=False,
            critic_loss_fn="mse",
            # Anti-collapse: logit noise + entropy regularization
            logit_noise_std=0.1,
            entropy_coeff=0.01,
            # Random warmup to seed replay buffer with diverse transitions
            random_warmup_steps=200,
            log_episodes=True,
        )

        trainer = DDPGTrainer(env, config)

        # Warmup replay buffer (random actions)
        warmup_steps = trainer._random_warmup()
        env_step = warmup_steps

        # Resume from checkpoint if available
        start_ep = _load_checkpoint(trainer)
        if start_ep > 0:
            print(f"Resuming from episode {start_ep} (checkpoint found)", flush=True)
            for i in range(start_ep):
                env.reset(seed=config.seed + i)

        print(f"=== Pattern-mode DDPG training v2 (exploration fix) ===", flush=True)
        print(f"  action_space_mode={env_cfg.action_space_mode}", flush=True)
        print(f"  mean_hz={mean_hz}, n_patterns={alphabet.n_actions}", flush=True)
        print(f"  state_length={env_cfg.state_length}", flush=True)
        print(f"  episodes={num_episodes}, starting from {start_ep}", flush=True)
        print(f"  temp_decay: {config.exploration_temperature_start} → {config.exploration_temperature_end}", flush=True)
        print(f"  logit_noise_std={config.logit_noise_std}", flush=True)
        print(f"  entropy_coeff={config.entropy_coeff}", flush=True)
        print(f"  random_warmup_steps={config.random_warmup_steps}", flush=True)
        print(flush=True)

        t0 = time.time()
        episode_rewards = []

        for episode in range(start_ep, num_episodes):
            state, _info = env.reset(seed=config.seed + episode)
            trainer._update_obs_stats(state)
            episode_reward = float(_info.get("reward", 0.0))
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, logits = trainer._select_action(state, env_step=env_step)
                env_step += 1
                next_state, reward, terminated, truncated, info = env.step(action)
                trainer._update_obs_stats(next_state)
                dw = float(info.get("dw", 1.0 if truncated else 0.0))
                normalized_reward = trainer._normalize_reward(reward)

                trainer.buffer.add(
                    state=state,
                    action=action,
                    action_logits=logits,
                    reward=normalized_reward,
                    next_state=next_state,
                    dw=dw,
                )
                episode_reward += reward
                state = next_state
                steps += 1
                if config.log_episodes and steps % 5 == 0:
                    print(
                        f"  episode {episode + 1}/{num_episodes} "
                        f"step {steps}/{config.max_episode_steps}",
                        flush=True,
                    )

                if len(trainer.buffer) >= config.min_buffer_size:
                    for _ in range(config.update_frequency):
                        c_loss, a_loss = trainer._update_step()

            episode_rewards.append(episode_reward)

            # Log temperature for monitoring
            current_temp = config.exploration_temperature_start + \
                (config.exploration_temperature_end - config.exploration_temperature_start) * \
                min(1.0, env_step / max(1, config.num_episodes * config.max_episode_steps))

            print(
                f"episode {episode + 1}/{num_episodes} "
                f"reward={episode_reward:.2f} steps={steps} temp={current_temp:.3f}",
                flush=True,
            )

            # Save checkpoint + result after every episode
            _save_checkpoint(trainer, episode)
            _append_result({
                "episode": episode + 1,
                "reward": float(episode_reward),
                "steps": steps,
                "temp": round(current_temp, 3),
                "timestamp": time.time(),
            })

            gc.collect()

        # Final analysis
        elapsed = time.time() - t0
        offline = _analyze_policy(trainer.actor, env_cfg.state_length)
        rollout = _rollout_actions(env, trainer.actor, seed=1000)

        dominant = int(offline["dominant_action"])
        final = {
            "task": "pattern-action-space-train-v2",
            "option": "C",
            "fix": "temp_decay_5_to_0.3+logit_noise_0.1+entropy_0.01+warmup_200",
            "action_space_mode": env_cfg.action_space_mode,
            "mean_hz": mean_hz,
            "n_patterns": alphabet.n_actions,
            "state_length": env_cfg.state_length,
            "episodes": num_episodes,
            "seed": config.seed,
            "elapsed_s": round(elapsed, 1),
            "final_reward": float(episode_rewards[-1]),
            "unique_actions_offline": int(offline["unique_actions_offline"]),
            "dominant_action": dominant,
            "dominant_action_semantics": describe_pattern(dominant, mean_hz=mean_hz),
            "dominant_fraction": float(offline["dominant_fraction"]),
            "action_distribution_offline": offline["action_distribution"],
            "unique_actions_rollout": len(set(rollout)),
            "rollout_actions": rollout,
            "episode_rewards": [float(r) for r in episode_rewards],
            "acceptance_pass": len(set(rollout)) > 1 and int(offline["unique_actions_offline"]) > 1,
        }

        FINAL_RESULTS.write_text(json.dumps(final, indent=2) + "\n")

        print(json.dumps({k: v for k, v in final.items() if k not in ("rollout_actions", "episode_rewards")}, indent=2), flush=True)
        print(f"\nrollout_unique={final['unique_actions_rollout']} offline_unique={final['unique_actions_offline']}", flush=True)
        if final["acceptance_pass"]:
            print("\n✓ ACCEPTANCE PASS", flush=True)
        else:
            print("\n✗ ACCEPTANCE FAIL", flush=True)
        return 0 if final["acceptance_pass"] else 1

    finally:
        env.close()


if __name__ == "__main__":
    from scripts.train_runtime_guard import run_main

    sys.exit(run_main(main, label="pattern-train-v2"))
