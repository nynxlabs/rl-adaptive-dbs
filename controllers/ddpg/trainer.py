"""DDPG training loop (Mehregan Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controllers.ddpg.buffer import ReplayBuffer
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.quantization import unwrap_actor, wrap_actor_for_training
from controllers.ddpg.networks import Actor, Critic, clone_module, hard_update, soft_update
from envs.mehregan.baselines import baseline_action

if TYPE_CHECKING:
    from envs.mehregan.env import MehreganEnv


@dataclass
class TrainMetrics:
    episode_rewards: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)


@dataclass
class TrainResult:
    actor: Actor
    policy: nn.Module
    critic: Critic
    metrics: TrainMetrics
    config: DDPGConfig


class DDPGTrainer:
    """Online actor–critic trainer on ``MehreganEnv``."""

    def __init__(
        self,
        env: MehreganEnv,
        config: DDPGConfig,
        *,
        actor: Actor | None = None,
        critic: Critic | None = None,
    ) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(config.device)

        state_length = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)

        self.actor = actor or Actor(
            state_length=state_length,
            n_actions=n_actions,
            conv_channels=config.conv_channels,
            shrink_dim=config.shrink_dim,
        )
        self.actor = wrap_actor_for_training(self.actor, config.variant)
        self.critic = critic or Critic(
            state_length=state_length,
            n_actions=n_actions,
            conv_channels=config.conv_channels,
            shrink_dim=config.shrink_dim,
        )
        self.actor_target = clone_module(self.actor)
        self.critic_target = clone_module(self.critic)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

        init_action = baseline_action(config.init_baseline, env.alphabet)
        self.actor.init_toward_action(init_action, bias_scale=config.init_bias_scale)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        state_shape = (state_length,)
        self.buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_shape=state_shape,
            n_actions=n_actions,
            seed=config.seed,
        )
        self._n_actions = n_actions

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device, dtype=torch.float32)

    def _exploration_fraction(self, env_step: int) -> float:
        total_steps = self.config.num_episodes * self.env.config.max_episode_steps
        if total_steps <= 1:
            return 1.0
        return min(1.0, env_step / total_steps)

    def _exploration_epsilon(self, env_step: int) -> float:
        frac = self._exploration_fraction(env_step)
        start = self.config.exploration_epsilon_start
        end = self.config.exploration_epsilon_end
        return start + frac * (end - start)

    def _exploration_temperature(self, env_step: int) -> float:
        frac = self._exploration_fraction(env_step)
        start = self.config.exploration_temperature_start
        end = self.config.exploration_temperature_end
        return start + frac * (end - start)

    def _select_action(self, state: np.ndarray, *, env_step: int) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            state_t = self._to_tensor(state).unsqueeze(0)
            logits = self.actor(state_t)
            logits_np = logits.squeeze(0).cpu().numpy()
            if self.config.exploration_mode == "softmax":
                temp = self._exploration_temperature(env_step)
                probs = F.softmax(logits / temp, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
            else:
                epsilon = self._exploration_epsilon(env_step)
                if np.random.random() < epsilon:
                    action = int(self.env.action_space.sample())
                else:
                    action_t, _ = Actor.select_action(logits)
                    action = int(action_t.item())
        return action, logits_np

    def _action_features(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        stored_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.critic_action_input == "logits":
            return stored_logits
        return F.one_hot(actions, num_classes=self._n_actions).to(dtype=torch.float32)

    def _q_all_actions(self, critic: Critic, states: torch.Tensor) -> torch.Tensor:
        """Q(s, a) for every discrete action; shape (batch, n_actions)."""
        batch = states.shape[0]
        n_actions = self._n_actions
        states_exp = states.unsqueeze(1).expand(-1, n_actions, -1).reshape(batch * n_actions, -1)
        eye = torch.eye(n_actions, device=states.device, dtype=torch.float32)
        action_features = eye.unsqueeze(0).expand(batch, -1, -1).reshape(batch * n_actions, n_actions)
        q_values = critic(states_exp, action_features)
        return q_values.reshape(batch, n_actions)

    def _actor_q_expectation(self, states: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.config.critic_action_input == "logits":
            return self.critic(states, logits)
        q_all = self._q_all_actions(self.critic, states)
        probs = F.softmax(logits, dim=-1)
        return (probs * q_all).sum(dim=-1)

    def _update_step(self) -> tuple[float, float]:
        batch = self.buffer.sample(self.config.batch_size)

        states = self._to_tensor(batch.state)
        next_states = self._to_tensor(batch.next_state)
        actions = torch.as_tensor(batch.action, device=self.device, dtype=torch.long)
        stored_logits = self._to_tensor(batch.action_logits)
        rewards = self._to_tensor(batch.reward)
        dw = self._to_tensor(batch.dw)

        action_features = self._action_features(
            states=states,
            actions=actions,
            stored_logits=stored_logits,
        )

        with torch.no_grad():
            if self.config.critic_action_input == "logits":
                next_logits = self.actor_target(next_states)
                next_q = self.critic_target(next_states, next_logits)
            else:
                next_logits = self.actor_target(next_states)
                next_actions = torch.argmax(next_logits, dim=-1)
                next_features = F.one_hot(next_actions, num_classes=self._n_actions).to(
                    dtype=torch.float32,
                )
                next_q = self.critic_target(next_states, next_features)
            q_target = rewards + self.config.gamma * (1.0 - dw) * next_q

        q_pred = self.critic(states, action_features)
        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad_(False)

        actor_logits = self.actor(states)
        actor_loss = -self._actor_q_expectation(states, actor_logits).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad_(True)

        soft_update(self.critic_target, self.critic, self.config.tau)
        soft_update(self.actor_target, self.actor, self.config.tau)

        return float(critic_loss.item()), float(actor_loss.item())

    def train(self) -> TrainResult:
        metrics = TrainMetrics()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        env_step = 0

        for episode in range(self.config.num_episodes):
            state, _info = self.env.reset(seed=self.config.seed + episode)
            episode_reward = float(_info.get("reward", 0.0))
            steps = 0

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
                steps += 1

                if len(self.buffer) >= self.config.min_buffer_size:
                    for _ in range(self.config.update_frequency):
                        c_loss, a_loss = self._update_step()
                        metrics.critic_losses.append(c_loss)
                        metrics.actor_losses.append(a_loss)

            metrics.episode_rewards.append(episode_reward)
            metrics.episode_steps.append(steps)
            if self.config.log_episodes:
                print(
                    f"episode {episode + 1}/{self.config.num_episodes} "
                    f"reward={episode_reward:.2f} steps={steps}",
                    flush=True,
                )

        return TrainResult(
            actor=unwrap_actor(self.actor),
            policy=self.actor,
            critic=self.critic,
            metrics=metrics,
            config=self.config,
        )


def train_ddpg(
    env: MehreganEnv,
    config: DDPGConfig | None = None,
    **kwargs: Any,
) -> TrainResult:
    """Train DDPG on ``env`` and return the trained actor."""
    cfg = config or DDPGConfig()
    trainer = DDPGTrainer(env, cfg, **kwargs)
    return trainer.train()
