"""Mehregan et al. environment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MehreganEnvConfig:
    """Paper §IV.A.1 defaults ([environment.md](../../docs/environment.md))."""

    step_duration_s: float = 2.0
    max_episode_steps: int = 30
    beta_threshold: float = 0.35
    reward_scale: float = 10.0
    # Raw $P_\beta$ is ~400–500 without stimulation (Mehregan Fig. 4); Fig. 3c uses $\beta_t=0.35$.
    observation_scale: float = 1000.0
    state_length: int = 1  # paper original; state_length > 1 requires obs preprocessing (TASK-67)
