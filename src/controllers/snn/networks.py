"""LIF layers and DSQN topology (Nguyen §III.B)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from controllers.snn.config import SNNConfig


@dataclass(frozen=True)
class LIFOutput:
    """Spike counts for control and membrane potentials for Q-learning."""

    spike_counts: torch.Tensor
    membrane: torch.Tensor


class _ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward (snnTorch ``surrogate.atan``, alpha 2)."""

    alpha = 2.0

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, u: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(u)
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        (u,) = ctx.saved_tensors
        a = _ATanSpike.alpha
        return grad_output * (a / 2) / (1 + (math.pi / 2 * a * u) ** 2)


def _spike(u: torch.Tensor, surrogate: str) -> torch.Tensor:
    if surrogate == "atan":
        return _ATanSpike.apply(u)
    if surrogate == "none":
        return (u > 0).to(u.dtype)
    msg = f"surrogate_gradient must be 'none' or 'atan', got {surrogate!r}"
    raise ValueError(msg)


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
        surrogate: str = "none",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.leak = leak
        self.threshold = threshold
        self.unroll_steps = unroll_steps
        self.surrogate = surrogate

    def step(self, x: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One LIF step: leak, integrate, spike, reset by subtraction (reset not differentiated)."""
        membrane = self.leak * membrane + self.linear(x)
        spikes = _spike(membrane - self.threshold, self.surrogate)
        return spikes, membrane - spikes.detach() * self.threshold

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
            spikes, membrane = self.step(x, membrane)
            spike_counts = spike_counts + spikes
        return LIFOutput(spike_counts=spike_counts, membrane=membrane)


class DSQN(nn.Module):
    """Three-layer LIF DSQN: input → hidden (128) → 9 action outputs."""

    def __init__(self, config: SNNConfig | None = None) -> None:
        super().__init__()
        cfg = (config or SNNConfig()).with_variant_defaults()
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        self.config = cfg
        if cfg.snn_input_mode not in ("static", "sequence"):
            msg = f"snn_input_mode must be 'static' or 'sequence', got {cfg.snn_input_mode!r}"
            raise ValueError(msg)
        in_features = cfg.flat_observation_dim
        if cfg.snn_input_mode == "sequence":
            in_features = cfg.observation_shape[1]
        self.input_layer = LIFLayer(
            in_features,
            cfg.hidden_size,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            unroll_steps=cfg.internal_unroll_steps,
            surrogate=cfg.surrogate_gradient,
        )
        self.hidden_layer = LIFLayer(
            cfg.hidden_size,
            cfg.hidden_size,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            unroll_steps=cfg.internal_unroll_steps,
            surrogate=cfg.surrogate_gradient,
        )
        self.output_layer = LIFLayer(
            cfg.hidden_size,
            cfg.n_action_outputs,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            unroll_steps=cfg.internal_unroll_steps,
            surrogate=cfg.surrogate_gradient,
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

        if self.config.snn_input_mode == "sequence":
            return self._forward_sequence(observation)
        hidden_input = self.input_layer(observation)
        hidden = self.hidden_layer(hidden_input.spike_counts)
        return self.output_layer(hidden.spike_counts)

    def _forward_sequence(self, observation: torch.Tensor) -> LIFOutput:
        """Row t of the (n × N) spike matrix drives step t; all layers step together."""
        steps, n_neurons = self.config.observation_shape
        x = observation.reshape(observation.shape[0], steps, n_neurons)
        layers = (self.input_layer, self.hidden_layer, self.output_layer)
        membranes = [
            torch.zeros(x.shape[0], layer.linear.out_features, device=x.device, dtype=x.dtype)
            for layer in layers
        ]
        spike_counts = torch.zeros_like(membranes[-1])
        for t in range(steps):
            spikes = x[:, t]
            for i, layer in enumerate(layers):
                spikes, membranes[i] = layer.step(spikes, membranes[i])
            spike_counts = spike_counts + spikes
        return LIFOutput(spike_counts=spike_counts, membrane=membranes[-1])
