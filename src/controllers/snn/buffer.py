"""DSQN replay buffer — (s, a, r, s', done) tuples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from controllers.snn.config import SNNConfig


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class TransitionBatch:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    """Ring buffer for Nguyen DSQN transitions."""

    def __init__(self, config: SNNConfig | None = None, *, seed: int | None = None) -> None:
        cfg = (config or SNNConfig()).with_variant_defaults()
        self.config = cfg
        self.capacity = cfg.replay_capacity
        state_dim = cfg.flat_observation_dim
        self._rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self._states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._done = np.zeros(self.capacity, dtype=np.bool_)
        self._pos = 0
        self._size = 0
        self._since_update = 0

    def __len__(self) -> int:
        return self._size

    @property
    def transitions_since_update(self) -> int:
        return self._since_update

    def add(self, transition: Transition) -> None:
        state = np.asarray(transition.state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(transition.next_state, dtype=np.float32).reshape(-1)
        expected = self.config.flat_observation_dim
        if state.shape != (expected,) or next_state.shape != (expected,):
            msg = f"state vectors must have shape ({expected},)"
            raise ValueError(msg)

        idx = self._pos
        self._states[idx] = state
        self._next_states[idx] = next_state
        self._actions[idx] = transition.action
        self._rewards[idx] = transition.reward
        self._done[idx] = transition.done
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._since_update += 1

    def ready_for_update(self) -> bool:
        return self._since_update >= self.config.replay_update_cadence

    def mark_updated(self) -> None:
        self._since_update = 0

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size > self._size:
            msg = f"batch_size {batch_size} exceeds buffer size {self._size}"
            raise ValueError(msg)
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            state=self._states[indices],
            action=self._actions[indices],
            reward=self._rewards[indices],
            next_state=self._next_states[indices],
            done=self._done[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "states": self._states.copy(),
            "next_states": self._next_states.copy(),
            "actions": self._actions.copy(),
            "rewards": self._rewards.copy(),
            "done": self._done.copy(),
            "pos": int(self._pos),
            "size": int(self._size),
            "since_update": int(self._since_update),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._states[:] = np.asarray(state["states"], dtype=np.float32)
        self._next_states[:] = np.asarray(state["next_states"], dtype=np.float32)
        self._actions[:] = np.asarray(state["actions"], dtype=np.int64)
        self._rewards[:] = np.asarray(state["rewards"], dtype=np.float32)
        self._done[:] = np.asarray(state["done"], dtype=np.bool_)
        self._pos = int(state["pos"])
        self._size = int(state["size"])
        self._since_update = int(state.get("since_update", 0))
        self._rng.bit_generator.state = state["rng_state"]
