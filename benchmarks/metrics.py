"""Normalize rollout payloads to core benchmark metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.mehregan.baselines import default_baselines


def _stim_frequency_mean(payload: dict[str, Any], *, variant: str) -> float:
    stim = payload.get("stim_freq_hz")
    if stim:
        return float(np.mean(stim))
    if variant in default_baselines():
        freq = default_baselines()[variant].dbs_spec.frequency_hz
        return float(freq) if freq > 0 else 0.0
    return float("nan")


def rollout_to_core_metrics(
    payload: dict[str, Any],
    *,
    controller: str,
    variant: str,
    seed: int,
    run_id: str,
    protocol: str,
    step_duration_s: float,
    metrics_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map a rollout or eval dict to [benchmarking.md](../docs/benchmarking.md) §4 core metrics."""
    p_beta = [float(x) for x in payload.get("p_beta", [])]
    steps = int(payload.get("steps", max(0, len(payload.get("actions", [])))))
    reward_sum = float(
        payload.get("reward_sum", payload.get("total_reward", float("nan")))
    )
    episode_length_s = (len(p_beta) if p_beta else steps + 1) * step_duration_s

    metrics: dict[str, Any] = {
        "controller": controller,
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "protocol": protocol,
        "p_beta_mean": float(np.mean(p_beta)) if p_beta else float("nan"),
        "p_beta_final": float(p_beta[-1]) if p_beta else float("nan"),
        "reward_sum": reward_sum,
        "stim_frequency_mean": _stim_frequency_mean(payload, variant=variant),
        "episode_length": steps,
        "episode_length_s": episode_length_s,
    }
    if metrics_extra:
        metrics["metrics_extra"] = dict(metrics_extra)
    return metrics


def rollout_timeseries(payload: dict[str, Any], *, step_duration_s: float) -> dict[str, list[Any]]:
    """Per-segment series for optional ``timeseries/`` output."""
    p_beta = [float(x) for x in payload.get("p_beta", [])]
    rewards = [float(x) for x in payload.get("rewards", [])]
    t = [index * step_duration_s for index in range(len(p_beta))]
    series: dict[str, list[Any]] = {"t": t, "p_beta": p_beta}
    if rewards:
        series["reward"] = rewards
    stim = payload.get("stim_freq_hz")
    if stim:
        series["stim_freq_hz"] = [float(x) for x in stim]
    actions = payload.get("actions")
    if actions is not None:
        series["actions"] = [int(x) for x in actions]
    return series
