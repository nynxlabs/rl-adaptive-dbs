"""Mehregan §III.D quantization — PTQ (FP16/INT8) and QAT fake-quant actor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from controllers.ddpg.networks import Actor

PTQ_VARIANTS = frozenset({"ptq-fp16", "ptq-int8"})
QUANTIZED_VARIANTS = PTQ_VARIANTS | {"qat"}


def is_ptq_variant(variant: str) -> bool:
    return variant in PTQ_VARIANTS


def is_quantized_variant(variant: str) -> bool:
    return variant in QUANTIZED_VARIANTS


def fp_source_variant(variant: str) -> str:
    """Checkpoint slug for full-precision weights used by PTQ variants."""
    if variant in PTQ_VARIANTS:
        return "paper"
    return variant


class QATActor(nn.Module):
    """Actor with fake-quant input and dequant logits (Mehregan §III.D)."""

    def __init__(self, actor: Actor) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.actor = actor
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.quant(state)
        logits = self.actor(x)
        return self.dequant(logits)

    def init_toward_action(self, action_index: int) -> None:
        self.actor.init_toward_action(action_index)


def unwrap_actor(module: nn.Module) -> Actor:
    if isinstance(module, QATActor):
        return module.actor
    if isinstance(module, Actor):
        return module
    msg = f"expected Actor or QATActor, got {type(module).__name__}"
    raise TypeError(msg)


def wrap_actor_for_training(actor: Actor, variant: str) -> nn.Module:
    if variant == "qat":
        return QATActor(actor)
    return actor


def apply_ptq(actor: Actor, variant: str) -> nn.Module:
    """Post-training quantization for inference-only eval."""
    if variant == "ptq-fp16":
        return actor.half()
    if variant == "ptq-int8":
        actor_cpu = actor.cpu().eval()
        return torch.ao.quantization.quantize_dynamic(
            actor_cpu,
            {nn.Linear},
            dtype=torch.qint8,
        )
    msg = f"not a PTQ variant: {variant!r}"
    raise ValueError(msg)


def prepare_actor_for_eval(
    actor: Actor,
    variant: str,
    *,
    device: str = "cpu",
) -> nn.Module:
    """Return an actor module ready for ``run_mehregan_eval``."""
    if variant in PTQ_VARIANTS:
        prepared = apply_ptq(actor, variant)
        if variant == "ptq-int8":
            return prepared
        return prepared.to(device)
    return actor.to(device)


def actor_state_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32
