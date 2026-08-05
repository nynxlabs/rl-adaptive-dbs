"""LIF layers and DSQN topology (Nguyen §III.B)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from controllers.snn.config import SNNConfig


@dataclass(frozen=True)
class LIFOutput:
    """Spike counts for control and membrane potentials for Q-learning."""

    spike_counts: torch.Tensor
    membrane: torch.Tensor


class LIFLayer(nn.Module):
    """Linear + leaky integrate-and-fire with unrolled internal timesteps."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        leak: float = 0.95,
        threshold: float = 1.0,
        unroll_steps: int = 5,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.leak = leak
        self.threshold = threshold
        self.unroll_steps = unroll_steps

    def forward(
        self,
        x: torch.Tensor,
        membrane: torch.Tensor | None = None,
    ) -> LIFOutput:
        if x.ndim != 2:
            msg = f"expected input shape (batch, features), got {tuple(x.shape)}"
            raise ValueError(msg)
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype
        if membrane is None:
            membrane = torch.zeros(batch, self.linear.out_features, device=device, dtype=dtype)
        else:
            membrane = membrane.to(device=device, dtype=dtype)

        spike_counts = torch.zeros(batch, self.linear.out_features, device=device, dtype=dtype)
        for _ in range(self.unroll_steps):
            drive = self.linear(x)
            membrane = self.leak * membrane + drive
            spikes = (membrane > self.threshold).to(dtype)
            spike_counts = spike_counts + spikes
            membrane = membrane - spikes * self.threshold
        return LIFOutput(spike_counts=spike_counts, membrane=membrane)


class DSQN(nn.Module):
    """Three-layer LIF DSQN: input → hidden (128) → 9 action outputs."""

    def __init__(self, config: SNNConfig | None = None) -> None:
        super().__init__()
        cfg = (config or SNNConfig()).with_variant_defaults()
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        self.config = cfg
        self.input_layer = LIFLayer(
            cfg.flat_observation_dim,
            cfg.hidden_size,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            unroll_steps=cfg.internal_unroll_steps,
        )
        self.hidden_layer = LIFLayer(
            cfg.hidden_size,
            cfg.hidden_size,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            unroll_steps=cfg.internal_unroll_steps,
        )
        self.output_layer = LIFLayer(
            cfg.hidden_size,
            cfg.n_action_outputs,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            unroll_steps=cfg.internal_unroll_steps,
        )

    def forward(self, observation: torch.Tensor) -> LIFOutput:
        """Forward pass from flattened spike observation to action-head outputs."""
        if observation.ndim != 2:
            msg = f"expected observation shape (batch, features), got {tuple(observation.shape)}"
            raise ValueError(msg)
        if observation.shape[1] != self.config.flat_observation_dim:
            msg = (
                f"expected {self.config.flat_observation_dim} input features, "
                f"got {observation.shape[1]}"
            )
            raise ValueError(msg)

        hidden_input = self.input_layer(observation)
        hidden = self.hidden_layer(hidden_input.spike_counts)
        return self.output_layer(hidden.spike_counts)
