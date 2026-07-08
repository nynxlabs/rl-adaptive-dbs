"""DSQN trainer scaffold — training loop deferred to a follow-up task."""

from __future__ import annotations

from dataclasses import dataclass

from controllers.snn.buffer import ReplayBuffer
from controllers.snn.config import SNNConfig
from controllers.snn.networks import DSQN


@dataclass(frozen=True)
class TrainMetrics:
    episode: int
    epsilon: float
    buffer_size: int


class DSQNTrainer:
    """DQN-style trainer shell; gradient updates are not implemented yet."""

    def __init__(
        self,
        dsqn: DSQN,
        buffer: ReplayBuffer,
        config: SNNConfig | None = None,
    ) -> None:
        self.dsqn = dsqn
        self.buffer = buffer
        self.config = (config or SNNConfig()).with_variant_defaults()
        self._total_steps = 0

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def current_epsilon(self) -> float:
        cfg = self.config
        if cfg.epsilon_decay_steps <= 0:
            return cfg.epsilon_end
        progress = min(1.0, self._total_steps / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)

    def note_step(self) -> None:
        self._total_steps += 1

    def maybe_update(self) -> bool:
        """Return True when the replay cadence (128) is met; no-op until training lands."""
        if not self.buffer.ready_for_update():
            return False
        self.buffer.mark_updated()
        return True

    def train_step(self) -> None:
        msg = "DSQN training is not implemented yet — scaffold only (TASK-95)"
        raise NotImplementedError(msg)
