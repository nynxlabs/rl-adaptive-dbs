"""Shared helpers for mid-training checkpoint resume."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class ConfigMismatchError(ValueError):
    """Raised when a checkpoint config disagrees with the active train config."""


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    msg = f"cannot serialize config type {type(config)!r}"
    raise TypeError(msg)


def validate_material_fields(
    saved: dict[str, Any],
    active: dict[str, Any],
    fields: tuple[str, ...],
    *,
    label: str = "config",
) -> None:
    mismatches: list[str] = []
    for field in fields:
        saved_val = saved.get(field)
        active_val = active.get(field)
        if saved_val != active_val:
            mismatches.append(f"{field}: checkpoint={saved_val!r} active={active_val!r}")
    if mismatches:
        detail = "; ".join(mismatches)
        msg = f"checkpoint {label} mismatch — {detail}"
        raise ConfigMismatchError(msg)


# Compared on resume; ``num_episodes`` may increase (train more episodes).
RESUME_SKIP_EQUALITY_FIELDS: frozenset[str] = frozenset({"num_episodes"})


def validate_resume_config_fields(
    saved: dict[str, Any],
    active: dict[str, Any],
    fields: tuple[str, ...],
    *,
    label: str,
    resume_start: int,
) -> None:
    """Compare material fields; allow ``num_episodes`` target to grow."""
    compare = tuple(f for f in fields if f not in RESUME_SKIP_EQUALITY_FIELDS)
    validate_material_fields(saved, active, compare, label=label)
    active_total = active.get("num_episodes")
    if active_total is not None and int(active_total) < int(resume_start):
        msg = (
            f"checkpoint {label}: num_episodes={active_total!r} "
            f"< resume_start={resume_start}"
        )
        raise ConfigMismatchError(msg)


def infer_completed_episodes(
    payload: dict[str, Any],
    *,
    metrics_path: Path | None = None,
    start_episode: int | None = None,
) -> int:
    """Return episode index to continue from (0 = fresh train)."""
    if start_episode is not None:
        return int(start_episode)

    extra = payload.get("extra")
    if isinstance(extra, dict):
        if "completed_episodes" in extra:
            return int(extra["completed_episodes"])
        rewards = extra.get("episode_rewards")
        if isinstance(rewards, list):
            return len(rewards)

    top_rewards = payload.get("episode_rewards")
    if isinstance(top_rewards, list):
        return len(top_rewards)

    if metrics_path is not None and metrics_path.is_file():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        rewards = data.get("episode_rewards")
        if isinstance(rewards, list):
            return len(rewards)

    return 0


SNN_MATERIAL_FIELDS: tuple[str, ...] = (
    "variant",
    "seed",
    "max_episode_steps",
    "alpha_beta_threshold",
    "subthreshold_steps_required",
    "energy_penalty",
    "threshold_reward",
    "alpha_beta_progress_coef",
    "warm_zone_upper",
    "warm_zone_bonus_coef",
    "truncation_penalty",
    "amplitude_sensitivity",
    "frequency_sensitivity",
    "pulse_width_sensitivity",
    "action_scheme",
    "hidden_size",
    "n_action_outputs",
    "sequence_steps",
    "neurons_per_region",
    "n_regions",
    "gamma",
    "learning_rate",
    "batch_size",
    "replay_capacity",
    "epsilon_start",
    "epsilon_end",
)

DDPG_MATERIAL_FIELDS: tuple[str, ...] = (
    "variant",
    "seed",
    "max_episode_steps",
    "action_space_mode",
    "pattern_mean_hz",
    "exploration_mode",
    "exploration_epsilon_start",
    "exploration_epsilon_end",
    "exploration_temperature_start",
    "exploration_temperature_end",
    "init_bias_scale",
    "critic_action_input",
    "logit_noise_std",
    "entropy_coeff",
    "reward_normalize",
    "obs_normalize",
    "random_warmup_steps",
    "critic_warmup_steps",
    "gamma",
    "tau",
    "batch_size",
    "buffer_capacity",
    "min_buffer_size",
    "update_frequency",
    "conv1_out",
    "conv2_out",
    "fc_hidden",
    "pool_kernel",
)

SEA_DBS_MATERIAL_FIELDS: tuple[str, ...] = (
    "variant",
    "seed",
    "max_episode_steps",
    "n_actions",
    "state_dim",
    "hidden_size",
    "use_predictive_model",
    "use_gumbel_softmax",
    "gamma",
    "batch_size",
    "buffer_capacity",
    "min_buffer_size",
    "update_frequency",
    "epsilon_start",
    "epsilon_end",
    "gs_tau0",
    "gs_tau_min",
    "gs_lambda",
    "pm_warmup_steps",
    "episode_psd_metric",
    "fixed_episode_seed",
    "dbs_burst_ms",
)
