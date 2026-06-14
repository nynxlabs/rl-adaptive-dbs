"""Replay buffer for Algorithm 1 transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    action_logits: np.ndarray
    reward: float
    next_state: np.ndarray
    dw: float


class ReplayBuffer:
    """Ring buffer storing $(s, a, a_{logit}, R, s', dw)$ tuples."""

    def __init__(
        self,
        *,
        capacity: int,
        state_shape: tuple[int, ...],
        n_actions: int,
        seed: int = 0,
    ) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._logits = np.zeros((capacity, n_actions), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dw = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        action_logits: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        dw: float,
    ) -> None:
        idx = self._pos
        self._states[idx] = state
        self._next_states[idx] = next_state
        self._actions[idx] = action
        self._logits[idx] = action_logits
        self._rewards[idx] = reward
        self._dw[idx] = dw
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        if self._size < batch_size:
            msg = f"buffer size {self._size} < batch_size {batch_size}"
            raise ValueError(msg)
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return Transition(
            state=self._states[indices],
            action=self._actions[indices],
            action_logits=self._logits[indices],
            reward=self._rewards[indices],
            next_state=self._next_states[indices],
            dw=self._dw[indices],
        )
