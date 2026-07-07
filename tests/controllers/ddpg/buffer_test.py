"""Replay buffer unit tests."""

from __future__ import annotations

import numpy as np

from controllers.ddpg.buffer import ReplayBuffer


def test_buffer_ring_overwrite() -> None:
    buf = ReplayBuffer(capacity=4, state_shape=(2,), n_actions=3, seed=0)
    for i in range(6):
        state = np.full(2, float(i), dtype=np.float32)
        buf.add(
            state=state,
            action=i,
            action_logits=np.ones(3, dtype=np.float32) * i,
            reward=float(i),
            next_state=state + 1.0,
            dw=0.0,
        )
    assert len(buf) == 4
    batch = buf.sample(4)
    assert batch.state.shape == (4, 2)
    assert batch.action_logits.shape == (4, 3)


def test_buffer_action_counts() -> None:
    buf = ReplayBuffer(capacity=8, state_shape=(2,), n_actions=5, seed=0)
    for action in (0, 1, 1, 3):
        state = np.zeros(2, dtype=np.float32)
        buf.add(
            state=state,
            action=action,
            action_logits=np.zeros(5, dtype=np.float32),
            reward=0.0,
            next_state=state,
            dw=0.0,
        )
    counts = buf.action_counts()
    assert counts.tolist() == [1, 2, 0, 1, 0]
