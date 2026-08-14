"""Within-step / continuous-plant Mehregan environment configuration (extension)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dataclasses import dataclass

if TYPE_CHECKING:
    from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
    from envs.mehregan.patterns import PatternAlphabet

StateMode = Literal["scalar", "within_step", "multi_step_history"]
RewardStateMode = Literal["observation_mean", "full_segment"]
PlantIntegrationMode = Literal["disconnected", "continuous"]

STATE_MODES: tuple[StateMode, ...] = ("scalar", "within_step", "multi_step_history")
REWARD_STATE_MODES: tuple[RewardStateMode, ...] = ("observation_mean", "full_segment")
PLANT_INTEGRATION_MODES: tuple[PlantIntegrationMode, ...] = (
    "disconnected",
    "continuous",
)


def resolve_state_mode(*, state_mode: str, state_length: int) -> StateMode:
    """Resolve effective state mode (legacy: scalar + state_length>1 → multi_step_history)."""
    if state_mode not in STATE_MODES:
        msg = f"unknown state_mode {state_mode!r}; expected one of {STATE_MODES}"
        raise ValueError(msg)
    if state_mode == "scalar" and state_length > 1:
        return "multi_step_history"
    return state_mode  # type: ignore[return-value]


@dataclass(frozen=True)
class WithinStepEnvConfig:
    """Paper §IV.A.1 defaults ([environment.md](../../docs/environment.md))."""

    step_duration_s: float = 2.0
    max_episode_steps: int = 30
    beta_threshold: float = 0.35
    reward_scale: float = 10.0
    # Raw $P_\\beta$ is ~400–500 without stimulation (Mehregan Fig. 4); Fig. 3c uses $\\beta_t=0.35$.
    observation_scale: float = 1000.0
    state_length: int = 1
    # scalar: one Pβ per step (state_length must be 1).
    # within_step: L sub-window Pβ samples inside each step (paper CNN path).
    # multi_step_history: rolling deque of past whole-step scalars (TASK-67 diagnostic).
    state_mode: str = "scalar"
    # observation_mean: Eq. (8) uses mean(observation) [default].
    # full_segment: Eq. (8) uses whole-step Pβ / observation_scale (decoupled from obs vector).
    reward_state_mode: str = "observation_mean"
    action_space_mode: str = "scalar_frequency"  # scalar_frequency | fixed_mean_pattern
    pattern_mean_hz: float = 45.0  # mean stimulation rate for fixed_mean_pattern mode
    skip_regular: bool = False  # True → exclude pattern 0 (regular periodic) from agent action space
    plant_dt_ms: float | None = None  # biomarker dt; None → plant integrate result dt_ms
    # disconnected: each RL step is a cold integrate from episode IC (legacy default).
    # continuous: sequential 2 s PythonPlant carry (Alg. 1).
    plant_integration_mode: str = "disconnected"
    # No-DBS segment at episode start before first action (defaults to step_duration_s).
    pre_stim_duration_s: float | None = None


def make_alphabet(
    config: WithinStepEnvConfig,
    *,
    plant_dt_ms: float | None = None,
) -> "PatternAlphabet | FixedMeanPatternAlphabet":
    """Create the action-space alphabet from the env config.

    Returns :class:`PatternAlphabet` (scalar-frequency, default) or
    :class:`FixedMeanPatternAlphabet` (Option C) depending on
    ``config.action_space_mode``.
    """
    dt_ms = plant_dt_ms if plant_dt_ms is not None else config.plant_dt_ms
    if config.action_space_mode == "fixed_mean_pattern":
        from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet

        kwargs: dict[str, float] = {
            "mean_hz": config.pattern_mean_hz,
            "step_duration_s": config.step_duration_s,
        }
        if dt_ms is not None:
            kwargs["dt_ms"] = dt_ms
        return FixedMeanPatternAlphabet(
            skip_regular=config.skip_regular,
            **kwargs,
        )
    if config.action_space_mode == "scalar_frequency":
        from envs.mehregan.patterns import PatternAlphabet

        return PatternAlphabet()
    msg = f"unknown action_space_mode {config.action_space_mode!r}"
    raise ValueError(msg)
