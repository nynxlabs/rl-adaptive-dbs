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
        reset: str = "post",
    ) -> None:
        super().__init__()
        if reset not in ("post", "eq2"):
            msg = f"lif_reset must be 'post' or 'eq2', got {reset!r}"
            raise ValueError(msg)
        self.linear = nn.Linear(in_features, out_features)
        self.leak = leak
        self.threshold = threshold
        self.unroll_steps = unroll_steps
        self.surrogate = surrogate
        self.reset = reset

    def step(self, x: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One LIF step: leak, integrate, spike, reset by subtraction (reset not differentiated).

        ``reset="eq2"`` carries U[t] itself and subtracts the previous step's spike on the
        next step, U[t] = βU[t−1] + WX[t] − S[t−1]θ (Eq. 2), with S[t−1] recomputed from
        U[t−1]. ``"post"`` stores U − Sθ, which the next step then leaks.
        """
        if self.reset == "eq2":
            prev_spikes = (membrane > self.threshold).to(membrane.dtype)
            membrane = self.leak * membrane + self.linear(x) - prev_spikes * self.threshold
            spikes = _spike(membrane - self.threshold, self.surrogate)
            return spikes, membrane
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
    """LIF DSQN: input → ``hidden_layers`` × hidden (128) → 9 action outputs.

    §III.B states one hidden layer (``hidden_layers=1``); ``hidden_layers=2`` is the
    pre-Oct 17 2026 topology (``input_layer`` + ``hidden_layer``), kept for old checkpoints.
    """

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
        self._pool = int(cfg.snn_input_avgpool) if cfg.snn_input_avgpool > 1 else 0
        if self._pool:
            if cfg.snn_input_mode != "sequence" or in_features % self._pool:
                msg = (
                    f"snn_input_avgpool={cfg.snn_input_avgpool} needs sequence input with a "
                    f"neuron count divisible by it, got {cfg.snn_input_mode!r} / {in_features}"
                )
                raise ValueError(msg)
            in_features //= self._pool
        if cfg.hidden_layers not in (1, 2):
            msg = f"hidden_layers must be 1 or 2, got {cfg.hidden_layers!r}"
            raise ValueError(msg)
        lif = {
            "leak": cfg.lif_leak,
            "threshold": cfg.lif_threshold,
            "unroll_steps": cfg.internal_unroll_steps,
            "surrogate": cfg.surrogate_gradient,
            "reset": cfg.lif_reset,
        }
        self.input_layer = LIFLayer(in_features, cfg.hidden_size, **lif)
        self.hidden_layer = (
            LIFLayer(cfg.hidden_size, cfg.hidden_size, **lif) if cfg.hidden_layers == 2 else None
        )
        self.output_layer = LIFLayer(cfg.hidden_size, cfg.n_action_outputs, **lif)

    @property
    def layers(self) -> tuple[LIFLayer, ...]:
        if self.hidden_layer is None:
            return (self.input_layer, self.output_layer)
        return (self.input_layer, self.hidden_layer, self.output_layer)

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
        out = self.input_layer(observation)
        for layer in self.layers[1:]:
            out = layer(out.spike_counts)
        return out

    def _forward_sequence(self, observation: torch.Tensor) -> LIFOutput:
        """Row t of the (n × N) spike matrix drives step t; all layers step together."""
        steps, n_neurons = self.config.observation_shape
        x = observation.reshape(observation.shape[0], steps, n_neurons)
        if self._pool:
            x = x.reshape(x.shape[0], steps, n_neurons // self._pool, self._pool).mean(dim=-1)
        layers = self.layers
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
