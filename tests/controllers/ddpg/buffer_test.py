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
