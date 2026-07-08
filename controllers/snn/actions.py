"""Action selection from DSQN spike counts (Nguyen §III.B)."""

from __future__ import annotations

import numpy as np

from controllers.snn.config import SNNConfig

TERNARY_FROM_INDEX: tuple[int, ...] = (-1, 0, 1)


def decode_factored_action(action_indices: np.ndarray) -> np.ndarray:
    """Map three indices in ``{0, 1, 2}`` to ternary ``{-1, 0, 1}``."""
    indices = np.asarray(action_indices, dtype=np.int64).reshape(-1)
    if indices.shape != (3,):
        msg = f"expected 3 action indices, got shape {indices.shape}"
        raise ValueError(msg)
    if np.any((indices < 0) | (indices > 2)):
        msg = f"action indices must be in {{0, 1, 2}}, got {indices.tolist()}"
        raise ValueError(msg)
    return np.array([TERNARY_FROM_INDEX[int(i)] for i in indices], dtype=np.int64)


def decode_joint_action(action_index: int) -> np.ndarray:
    """Map a single joint index in ``[0, 8]`` to three ternary deltas."""
    if not 0 <= action_index <= 8:
        msg = f"joint action index must be in [0, 8], got {action_index}"
        raise ValueError(msg)
    rem = int(action_index)
    indices = []
    for _ in range(3):
        indices.append(rem % 3)
        rem //= 3
    return decode_factored_action(np.array(list(reversed(indices)), dtype=np.int64))


def select_action(
    spike_counts: np.ndarray,
    *,
    config: SNNConfig | None = None,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray]:
    """Argmax on spike counts with optional ε-greedy exploration.

    Returns ``(action_index, ternary_deltas)`` where ``action_index`` is the
    discrete index used for replay (joint: 0–8; factored: flattened group argmax).
    """
    cfg = (config or SNNConfig()).with_variant_defaults()
    counts = np.asarray(spike_counts, dtype=np.float64).reshape(-1)
    if counts.shape != (cfg.n_action_outputs,):
        msg = f"expected {cfg.n_action_outputs} spike counts, got shape {counts.shape}"
        raise ValueError(msg)

    gen = rng if rng is not None else np.random.default_rng()
    explore = epsilon > 0.0 and gen.random() < epsilon

    if cfg.action_scheme == "joint":
        greedy = int(np.argmax(counts))
        action_index = int(gen.integers(cfg.n_action_outputs)) if explore else greedy
        return action_index, decode_joint_action(action_index)

    if cfg.action_scheme == "factored":
        grouped = counts.reshape(3, 3)
        greedy_indices = np.argmax(grouped, axis=1)
        if explore:
            chosen_indices = gen.integers(0, 3, size=3)
        else:
            chosen_indices = greedy_indices
        ternary = decode_factored_action(chosen_indices)
        # Replay index: base-3 encoding of chosen ternary indices.
        action_index = int(chosen_indices[0] * 9 + chosen_indices[1] * 3 + chosen_indices[2])
        return action_index, ternary

    msg = f"unknown action_scheme {cfg.action_scheme!r}"
    raise ValueError(msg)
