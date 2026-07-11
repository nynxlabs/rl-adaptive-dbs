"""Spike-matrix observation encoder (Nguyen Eq. (4))."""

from __future__ import annotations

import numpy as np

from controllers.snn.config import SNNConfig


class SpikeObservationEncoder:
    """Binary spike presence matrix in [0, 1]^{n × N} from per-neuron spike times."""

    def __init__(self, config: SNNConfig | None = None) -> None:
        self.config = (config or SNNConfig()).with_variant_defaults()

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.config.observation_shape

    def encode(
        self,
        spike_times_per_neuron: list[np.ndarray],
        *,
        duration_s: float,
    ) -> np.ndarray:
        """Build a binary matrix with shape ``(sequence_steps, n_neurons)``."""
        n_steps, n_neurons = self.output_shape
        if len(spike_times_per_neuron) != n_neurons:
            msg = (
                f"expected {n_neurons} neuron spike trains, got {len(spike_times_per_neuron)}"
            )
            raise ValueError(msg)
        if duration_s <= 0:
            msg = "duration_s must be positive"
            raise ValueError(msg)

        obs = np.zeros((n_steps, n_neurons), dtype=np.float32)
        bin_edges = np.linspace(0.0, duration_s, n_steps + 1)
        for neuron_idx, times in enumerate(spike_times_per_neuron):
            if times.size == 0:
                continue
            times = np.asarray(times, dtype=float).reshape(-1)
            bin_idx = np.searchsorted(bin_edges[1:], times, side="right")
            bin_idx = np.clip(bin_idx, 0, n_steps - 1)
            obs[bin_idx, neuron_idx] = 1.0
        return obs

    def flatten(self, observation: np.ndarray) -> np.ndarray:
        """Flatten ``(n, N)`` spike matrix to a 1-D feature vector for DSQN input."""
        expected = self.output_shape
        if observation.shape != expected:
            msg = f"observation shape {observation.shape} != expected {expected}"
            raise ValueError(msg)
        return observation.reshape(-1).astype(np.float32, copy=False)
