"""Mehregan et al. environment configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
    from envs.mehregan.patterns import PatternAlphabet


@dataclass(frozen=True)
class MehreganEnvConfig:
    """Paper §IV.A.1 defaults ([environment.md](../../docs/environment.md))."""

    step_duration_s: float = 2.0
    max_episode_steps: int = 30
    beta_threshold: float = 0.35
    reward_scale: float = 10.0
    # Raw $P_\\beta$ is ~400–500 without stimulation (Mehregan Fig. 4); Fig. 3c uses $\\beta_t=0.35$.
    observation_scale: float = 1000.0
    state_length: int = 1  # paper original; state_length > 1 requires obs preprocessing (TASK-67)
    action_space_mode: str = "scalar_frequency"  # scalar_frequency | fixed_mean_pattern
    pattern_mean_hz: float = 45.0  # mean stimulation rate for fixed_mean_pattern mode
    skip_regular: bool = False  # True → exclude pattern 0 (regular periodic) from agent action space


def make_alphabet(
    config: MehreganEnvConfig,
) -> "PatternAlphabet | FixedMeanPatternAlphabet":
    """Create the action-space alphabet from the env config.

    Returns :class:`PatternAlphabet` (scalar-frequency, default) or
    :class:`FixedMeanPatternAlphabet` (Option C) depending on
    ``config.action_space_mode``.
    """
    if config.action_space_mode == "fixed_mean_pattern":
        from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
        return FixedMeanPatternAlphabet(
            mean_hz=config.pattern_mean_hz,
            step_duration_s=config.step_duration_s,
            skip_regular=config.skip_regular,
        )
    if config.action_space_mode == "scalar_frequency":
        from envs.mehregan.patterns import PatternAlphabet
        return PatternAlphabet()
    msg = f"unknown action_space_mode {config.action_space_mode!r}"
    raise ValueError(msg)
