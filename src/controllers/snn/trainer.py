"""DSQN trainer — DQN updates on output membrane potentials (Nguyen §III.B)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from controllers.snn.actions import select_action
from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer, Transition
from controllers.snn.config import SNNConfig
from controllers.snn.networks import DSQN


@dataclass(frozen=True)
class TrainMetrics:
    episode: int
    epsilon: float
    buffer_size: int
    loss: float | None = None


@dataclass
class TrainResult:
    """Outcome of a DSQN training run."""

    config: SNNConfig
    dsqn: DSQN
    metrics: list[TrainMetrics] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    update_count: int = 0


def _decode_factored_indices(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode base-3 factored action indices (0–26) into three group indices."""
    i0 = actions // 9
    rem = actions % 9
    i1 = rem // 3
    i2 = rem % 3
    return i0, i1, i2


def _q_from_membrane(
    membrane: torch.Tensor,
    actions: torch.Tensor,
    *,
    action_scheme: str,
) -> torch.Tensor:
    """Gather Q(s,a) from output membrane potentials."""
    if action_scheme == "joint":
        return membrane.gather(1, actions.view(-1, 1)).squeeze(1)

    if action_scheme == "factored":
        grouped = membrane.view(-1, 3, 3)
        i0, i1, i2 = _decode_factored_indices(actions)
        q0 = grouped[:, 0, :].gather(1, i0.view(-1, 1)).squeeze(1)
        q1 = grouped[:, 1, :].gather(1, i1.view(-1, 1)).squeeze(1)
        q2 = grouped[:, 2, :].gather(1, i2.view(-1, 1)).squeeze(1)
        return q0 + q1 + q2

    msg = f"unknown action_scheme {action_scheme!r}"
    raise ValueError(msg)


def _max_q_from_membrane(membrane: torch.Tensor, *, action_scheme: str) -> torch.Tensor:
    """max_a Q(s', a) from membrane potentials."""
    if action_scheme == "joint":
        return membrane.max(dim=1).values

    if action_scheme == "factored":
        grouped = membrane.view(-1, 3, 3)
        return grouped.max(dim=-1).values.sum(dim=-1)

    msg = f"unknown action_scheme {action_scheme!r}"
    raise ValueError(msg)


class DSQNTrainer:
    """DQN-style trainer: control from spike counts, Q from membrane potentials."""

    def __init__(
        self,
        dsqn: DSQN,
        buffer: ReplayBuffer,
        config: SNNConfig | None = None,
        *,
        target_dsqn: DSQN | None = None,
    ) -> None:
        self.config = (config or SNNConfig()).with_variant_defaults()
        self.dsqn = dsqn
        self.buffer = buffer
        self.target_dsqn = target_dsqn if target_dsqn is not None else deepcopy(dsqn)
        self.target_dsqn.eval()
        for param in self.target_dsqn.parameters():
            param.requires_grad_(False)

        self.device = torch.device(self.config.device)
        self.dsqn.to(self.device)
        self.target_dsqn.to(self.device)
        self.optimizer = torch.optim.Adam(self.dsqn.parameters(), lr=self.config.learning_rate)
        self._rng = np.random.default_rng(self.config.seed)
        self._total_steps = 0
        self._update_count = 0
        self._last_loss: float | None = None

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def last_loss(self) -> float | None:
        return self._last_loss

    def current_epsilon(self) -> float:
        cfg = self.config
        if cfg.epsilon_decay_steps <= 0:
            return cfg.epsilon_end
        progress = min(1.0, self._total_steps / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)

    def note_step(self) -> None:
        self._total_steps += 1

    def hard_update_target(self) -> None:
        self.target_dsqn.load_state_dict(self.dsqn.state_dict())

    def maybe_update(self) -> bool:
        """Run a gradient step when replay cadence is met."""
        if not self.buffer.ready_for_update():
            return False
        if len(self.buffer) < self.config.batch_size:
            return False
        self.train_step()
        self.buffer.mark_updated()
        if (
            self.config.target_update_period > 0
            and self._update_count % self.config.target_update_period == 0
        ):
            self.hard_update_target()
        return True

    def train_step(self) -> float:
        """One minibatch Bellman update on output membrane Q-values."""
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size)
        states = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_state, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)

        self.dsqn.train()
        q_out = self.dsqn(states)
        q_sa = _q_from_membrane(q_out.membrane, actions, action_scheme=cfg.action_scheme)

        with torch.no_grad():
            next_out = self.target_dsqn(next_states)
            next_max = _max_q_from_membrane(next_out.membrane, action_scheme=cfg.action_scheme)
            target = rewards + cfg.gamma * (1.0 - done) * next_max

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self._update_count += 1
        self._last_loss = float(loss.detach().cpu())
        return self._last_loss

    def act(self, observation: np.ndarray, *, explore: bool = True) -> tuple[int, np.ndarray]:
        """ε-greedy action from spike counts; returns (replay_index, MultiDiscrete indices)."""
        flat = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        tensor = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
        self.dsqn.eval()
        with torch.no_grad():
            out = self.dsqn(tensor)
        counts = out.spike_counts.detach().cpu().numpy().reshape(-1)
        epsilon = self.current_epsilon() if explore else 0.0
        action_index, ternary = select_action(
            counts,
            config=self.config,
            epsilon=epsilon,
            rng=self._rng,
        )
        # Adapter MultiDiscrete expects {0,1,2}; ternary is {-1,0,1}.
        indices = (ternary + 1).astype(np.int64)
        return action_index, indices

    def train_episodes(self, env: NguyenEnvAdapter) -> TrainResult:
        """Run ``num_episodes`` of interaction + DQN updates."""
        cfg = self.config
        result = TrainResult(config=cfg, dsqn=self.dsqn)

        for episode in range(cfg.num_episodes):
            obs, _info = env.reset(seed=cfg.seed + episode)
            episode_reward = 0.0
            steps = 0
            for _ in range(cfg.max_episode_steps):
                flat = np.asarray(obs, dtype=np.float32).reshape(-1)
                action_index, indices = self.act(obs, explore=True)
                next_obs, reward, terminated, truncated, _step_info = env.step(indices)
                done = bool(terminated or truncated)
                self.buffer.add(
                    Transition(
                        state=flat,
                        action=int(action_index),
                        reward=float(reward),
                        next_state=np.asarray(next_obs, dtype=np.float32).reshape(-1),
                        done=done,
                    )
                )
                self.note_step()
                self.maybe_update()
                episode_reward += float(reward)
                steps += 1
                obs = next_obs
                if done:
                    break

            metrics = TrainMetrics(
                episode=episode,
                epsilon=self.current_epsilon(),
                buffer_size=len(self.buffer),
                loss=self._last_loss,
            )
            result.metrics.append(metrics)
            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(steps)
            if cfg.log_episodes:
                print(
                    f"episode {episode + 1}/{cfg.num_episodes} "
                    f"reward={episode_reward:.3f} steps={steps} "
                    f"eps={metrics.epsilon:.3f} loss={metrics.loss}",
                    flush=True,
                )

        result.update_count = self._update_count
        return result


def save_checkpoint(
    path: str | Path,
    *,
    dsqn: DSQN,
    config: SNNConfig,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dsqn_state_dict": dsqn.state_dict(),
        "config": config,
        "controller": "snn",
        "variant": config.variant,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def train_dsqn(
    env: NguyenEnvAdapter | None = None,
    config: SNNConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    plant: Any | None = None,
) -> TrainResult:
    """Train DSQN on ``NguyenEnvAdapter`` and optionally save a checkpoint."""
    cfg = (config or SNNConfig()).with_variant_defaults()
    owns_env = env is None
    if env is None:
        env = NguyenEnvAdapter(plant=plant, config=cfg)
    try:
        dsqn = DSQN(cfg)
        buffer = ReplayBuffer(cfg, seed=cfg.seed)
        trainer = DSQNTrainer(dsqn, buffer, cfg)
        result = trainer.train_episodes(env)
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                dsqn=result.dsqn,
                config=cfg,
                optimizer=trainer.optimizer,
                extra={
                    "episode_rewards": result.episode_rewards,
                    "episode_lengths": result.episode_lengths,
                    "update_count": result.update_count,
                },
            )
        return result
    finally:
        if owns_env:
            env.close()
