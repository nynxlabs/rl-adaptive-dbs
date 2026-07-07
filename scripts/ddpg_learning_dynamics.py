#!/usr/bin/env python3
"""Instrument DDPG training for learning-dynamics investigation (TASK-72).

Logs critic Q discrimination, actor gradient flow, replay diversity, reward vs Q
scale, and logit evolution during a short training probe. Does not replace full
retraining — use for diagnosis only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from controllers.ddpg.buffer import ReplayBuffer
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.networks import Actor, Critic, soft_update
from controllers.ddpg.quantization import unwrap_actor
from controllers.ddpg.trainer import DDPGTrainer, TrainMetrics, TrainResult
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.reward import mehregan_reward
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from tests.envs.mock_plant import MockPlant


@dataclass
class UpdateDiagnostics:
    critic_loss: float
    actor_loss: float
    reward_mean: float
    reward_std: float
    q_pred_mean: float
    q_pred_std: float
    q_target_mean: float
    q_std_onehot_actions: float
    q_std_stored_logits: float
    actor_grad_encoder_norm: float
    actor_grad_head_norm: float
    critic_grad_norm: float


@dataclass
class EpisodeDiagnostics:
    episode: int
    episode_reward: float
    replay_unique_actions: int
    replay_actions_with_samples: int
    replay_total_transitions: int
    logit_unique_argmax: int
    logit_margin_mean: float
    logit_margin_min: float
    updates: list[UpdateDiagnostics] = field(default_factory=list)


def canonical_action_logits(n_actions: int, action: int, *, peak: float = 3.0) -> torch.Tensor:
    logits = torch.zeros(n_actions, dtype=torch.float32)
    logits[action] = peak
    return logits


def measure_q_std_onehot(
    critic: Critic,
    states: torch.Tensor,
    *,
    n_actions: int,
    peak: float = 3.0,
) -> float:
    """Std of Q(s, canonical_logits_a) across actions, averaged over states."""
    critic.eval()
    stds: list[float] = []
    with torch.no_grad():
        for state in states:
            q_vals = []
            for action in range(n_actions):
                logits = canonical_action_logits(n_actions, action, peak=peak).to(states.device)
                q_vals.append(float(critic(state.unsqueeze(0), logits.unsqueeze(0)).item()))
            stds.append(float(np.std(q_vals)))
    return float(np.mean(stds)) if stds else 0.0


def probe_logit_collapse(actor: Actor, state_length: int, *, n_samples: int = 200) -> dict[str, float | int]:
    actor.eval()
    actions: list[int] = []
    margins: list[float] = []
    with torch.no_grad():
        for _ in range(n_samples):
            window = np.random.uniform(0.25, 0.65, size=state_length).astype(np.float32)
            state_t = torch.as_tensor(window).unsqueeze(0)
            logits = actor(state_t).squeeze(0).numpy()
            sorted_logits = np.sort(logits)
            margins.append(float(sorted_logits[-1] - sorted_logits[-2]))
            actions.append(int(np.argmax(logits)))
    return {
        "unique_argmax": len(set(actions)),
        "logit_margin_mean": float(np.mean(margins)),
        "logit_margin_min": float(np.min(margins)),
    }


class InstrumentedDDPGTrainer(DDPGTrainer):
    """DDPG trainer that records learning-dynamics diagnostics."""

    def __init__(self, *args: Any, log_every: int = 50, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_every = log_every
        self._update_count = 0
        self.episode_diagnostics: list[EpisodeDiagnostics] = []
        self._current_episode_updates: list[UpdateDiagnostics] = []

    def _update_step(self) -> tuple[float, float]:
        batch = self.buffer.sample(self.config.batch_size)

        states = self._to_tensor(batch.state)
        next_states = self._to_tensor(batch.next_state)
        stored_logits = self._to_tensor(batch.action_logits)
        rewards = self._to_tensor(batch.reward)
        dw = self._to_tensor(batch.dw)

        with torch.no_grad():
            next_logits = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_logits)
            q_target = rewards + self.config.gamma * (1.0 - dw) * next_q

        q_pred = self.critic(states, stored_logits)
        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = float(
            torch.sqrt(
                sum(p.grad.norm().pow(2) for p in self.critic.parameters() if p.grad is not None)
            ).item()
        )
        self.critic_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad_(False)

        actor_logits = self.actor(states)
        actor_loss = -self.critic(states, actor_logits).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        enc_grad = float(
            torch.sqrt(
                sum(
                    p.grad.norm().pow(2)
                    for p in self.actor.encoder.parameters()
                    if p.grad is not None
                )
            ).item()
        )
        head_grad = float(
            torch.sqrt(
                sum(p.grad.norm().pow(2) for p in self.actor.head.parameters() if p.grad is not None)
            ).item()
        )
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad_(True)

        soft_update(self.critic_target, self.critic, self.config.tau)
        soft_update(self.actor_target, self.actor, self.config.tau)

        self._update_count += 1
        if self._update_count % self.log_every == 0:
            n_actions = int(self.env.action_space.n)
            q_std_onehot = measure_q_std_onehot(
                self.critic,
                states[: min(8, len(states))],
                n_actions=n_actions,
            )
            with torch.no_grad():
                q_on_stored = self.critic(states, stored_logits)
                q_std_stored = float(q_on_stored.std().item())
            self._current_episode_updates.append(
                UpdateDiagnostics(
                    critic_loss=float(critic_loss.item()),
                    actor_loss=float(actor_loss.item()),
                    reward_mean=float(rewards.mean().item()),
                    reward_std=float(rewards.std().item()),
                    q_pred_mean=float(q_pred.mean().item()),
                    q_pred_std=float(q_pred.std().item()),
                    q_target_mean=float(q_target.mean().item()),
                    q_std_onehot_actions=q_std_onehot,
                    q_std_stored_logits=q_std_stored,
                    actor_grad_encoder_norm=enc_grad,
                    actor_grad_head_norm=head_grad,
                    critic_grad_norm=critic_grad_norm,
                )
            )

        return float(critic_loss.item()), float(actor_loss.item())

    def train(self) -> TrainResult:
        metrics = TrainMetrics()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        env_step = 0
        state_length = int(self.env.observation_space.shape[0])

        for episode in range(self.config.num_episodes):
            self._current_episode_updates = []
            state, _info = self.env.reset(seed=self.config.seed + episode)
            episode_reward = float(_info.get("reward", 0.0))
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, logits = self._select_action(state, env_step=env_step)
                env_step += 1
                next_state, reward, terminated, truncated, info = self.env.step(action)
                dw = float(info.get("dw", 1.0 if truncated else 0.0))
                self.buffer.add(
                    state=state,
                    action=action,
                    action_logits=logits,
                    reward=reward,
                    next_state=next_state,
                    dw=dw,
                )
                episode_reward += reward
                state = next_state
                if len(self.buffer) >= self.config.min_buffer_size:
                    for _ in range(self.config.update_frequency):
                        self._update_step()
            metrics.episode_rewards.append(episode_reward)

            counts = self.buffer.action_counts()
            logit_probe = probe_logit_collapse(self.actor, state_length)
            self.episode_diagnostics.append(
                EpisodeDiagnostics(
                    episode=episode,
                    episode_reward=episode_reward,
                    replay_unique_actions=int(np.count_nonzero(counts)),
                    replay_actions_with_samples=int(np.sum(counts > 0)),
                    replay_total_transitions=len(self.buffer),
                    logit_unique_argmax=int(logit_probe["unique_argmax"]),
                    logit_margin_mean=float(logit_probe["logit_margin_mean"]),
                    logit_margin_min=float(logit_probe["logit_margin_min"]),
                    updates=list(self._current_episode_updates),
                )
            )

        return TrainResult(
            actor=unwrap_actor(self.actor),
            policy=self.actor,
            critic=self.critic,
            metrics=metrics,
            config=self.config,
        )


def summarize_diagnostics(episodes: list[EpisodeDiagnostics]) -> dict[str, Any]:
    all_updates = [u for ep in episodes for u in ep.updates]
    if not all_updates:
        return {"n_logged_updates": 0}

    def mean_attr(name: str) -> float:
        return float(np.mean([getattr(u, name) for u in all_updates]))

    first_ep = episodes[0]
    last_ep = episodes[-1]
    return {
        "n_logged_updates": len(all_updates),
        "reward_std_mean": mean_attr("reward_std"),
        "q_pred_std_mean": mean_attr("q_pred_std"),
        "q_std_onehot_actions_mean": mean_attr("q_std_onehot_actions"),
        "q_std_stored_logits_mean": mean_attr("q_std_stored_logits"),
        "actor_grad_encoder_norm_mean": mean_attr("actor_grad_encoder_norm"),
        "actor_grad_head_norm_mean": mean_attr("actor_grad_head_norm"),
        "critic_grad_norm_mean": mean_attr("critic_grad_norm"),
        "first_episode": {
            "replay_unique_actions": first_ep.replay_unique_actions,
            "logit_unique_argmax": first_ep.logit_unique_argmax,
            "logit_margin_mean": first_ep.logit_margin_mean,
        },
        "last_episode": {
            "replay_unique_actions": last_ep.replay_unique_actions,
            "logit_unique_argmax": last_ep.logit_unique_argmax,
            "logit_margin_mean": last_ep.logit_margin_mean,
        },
        "logit_margin_delta": last_ep.logit_margin_mean - first_ep.logit_margin_mean,
    }


def reward_q_scale_reference() -> dict[str, float]:
    s_vals = np.linspace(0.2, 0.6, 41)
    rewards = [mehregan_reward(np.array([s])) for s in s_vals]
    return {
        "synthetic_reward_range": float(max(rewards) - min(rewards)),
        "plant_reward_span_5step": 2.741322304461603,
    }


def run_probe(
    *,
    episodes: int,
    state_length: int,
    seed: int,
    mock: bool,
    exploration_mode: str,
    log_every: int,
) -> dict[str, Any]:
    if mock:
        env = MehreganEnv(
            plant=MockPlant(),
            config=MehreganEnvConfig(max_episode_steps=5, state_length=state_length),
        )
    else:
        resolved = resolve_config()
        env = MehreganEnv(
            plant=PythonPlant(config=resolved.plant),
            config=MehreganEnvConfig(state_length=state_length),
        )

    config = DDPGConfig(
        variant="paper",
        seed=seed,
        num_episodes=episodes,
        exploration_mode=exploration_mode,
        exploration_temperature_start=2.0,
        exploration_temperature_end=0.5,
        exploration_epsilon_start=0.3,
        exploration_epsilon_end=0.05,
        min_buffer_size=8 if mock else 16,
        batch_size=8 if mock else 16,
        max_episode_steps=5 if mock else 30,
    )

    try:
        trainer = InstrumentedDDPGTrainer(env, config, log_every=log_every)
        result = trainer.train()
        summary = summarize_diagnostics(trainer.episode_diagnostics)
        return {
            "config": {
                "episodes": episodes,
                "state_length": state_length,
                "seed": seed,
                "mock": mock,
                "exploration_mode": exploration_mode,
                "log_every": log_every,
            },
            "reward_q_scale_reference": reward_q_scale_reference(),
            "summary": summary,
            "episodes": [
                {
                    "episode": ep.episode,
                    "episode_reward": ep.episode_reward,
                    "replay_unique_actions": ep.replay_unique_actions,
                    "replay_actions_with_samples": ep.replay_actions_with_samples,
                    "replay_total_transitions": ep.replay_total_transitions,
                    "logit_unique_argmax": ep.logit_unique_argmax,
                    "logit_margin_mean": ep.logit_margin_mean,
                    "logit_margin_min": ep.logit_margin_min,
                    "updates": [u.__dict__ for u in ep.updates],
                }
                for ep in trainer.episode_diagnostics
            ],
            "final_episode_rewards": [float(r) for r in result.metrics.episode_rewards],
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--state-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Use MockPlant (fast smoke)")
    parser.add_argument("--exploration-mode", choices=("epsilon", "softmax"), default="softmax")
    parser.add_argument("--log-every", type=int, default=20, help="Log diagnostics every N updates")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ddpg/learning_dynamics_task72.json"),
    )
    args = parser.parse_args()

    out = run_probe(
        episodes=args.episodes,
        state_length=args.state_length,
        seed=args.seed,
        mock=args.mock,
        exploration_mode=args.exploration_mode,
        log_every=args.log_every,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
