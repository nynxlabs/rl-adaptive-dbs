"""Replay buffer for Algorithm 1 (stores r_hat)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    action_logits: np.ndarray
    reward: float
    r_hat: float
    next_state: np.ndarray
    dw: float


class ReplayBuffer:
    """Ring buffer for $(s, a, a_{logit}, r, \\hat{r}, s', dw)$."""

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
        self._r_hat = np.zeros(capacity, dtype=np.float32)
        self._dw = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        idx = self._pos
        self._states[idx] = transition.state
        self._next_states[idx] = transition.next_state
        self._actions[idx] = transition.action
        self._logits[idx] = transition.action_logits
        self._rewards[idx] = transition.reward
        self._r_hat[idx] = transition.r_hat
        self._dw[idx] = transition.dw
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
            r_hat=self._r_hat[indices],
            next_state=self._next_states[indices],
            dw=self._dw[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "states": self._states.copy(),
            "next_states": self._next_states.copy(),
            "actions": self._actions.copy(),
            "logits": self._logits.copy(),
            "rewards": self._rewards.copy(),
            "r_hat": self._r_hat.copy(),
            "dw": self._dw.copy(),
            "pos": int(self._pos),
            "size": int(self._size),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._states[:] = np.asarray(state["states"], dtype=np.float32)
        self._next_states[:] = np.asarray(state["next_states"], dtype=np.float32)
        self._actions[:] = np.asarray(state["actions"], dtype=np.int64)
        self._logits[:] = np.asarray(state["logits"], dtype=np.float32)
        self._rewards[:] = np.asarray(state["rewards"], dtype=np.float32)
        self._r_hat[:] = np.asarray(state["r_hat"], dtype=np.float32)
        self._dw[:] = np.asarray(state["dw"], dtype=np.float32)
        self._pos = int(state["pos"])
        self._size = int(state["size"])
        self._rng.bit_generator.state = state["rng_state"]
