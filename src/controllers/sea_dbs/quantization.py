"""FP16 post-training quantization for SEA-DBS actor (§11).

PTQ here is exactly what the paper names: cast the trained fp32 actor to fp16
for inference. No weight perturbation — a quantized policy that picks the same
actions as fp32 is the expected result for a small MLP, not a defect.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from controllers.sea_dbs.networks import Actor

PTQ_VARIANT = "ptq-fp16"


def is_ptq_variant(variant: str) -> bool:
    return variant == PTQ_VARIANT


def fp_source_variant(variant: str) -> str:
    if is_ptq_variant(variant):
        return "paper"
    return variant


def apply_fp16_ptq(actor: Actor) -> Actor:
    """Return a deep-copied actor with FP16 weights for inference."""
    src = copy.deepcopy(actor)
    src.half()
    src.eval()
    return src  # type: ignore[return-value]


class FP16ActorWrapper(nn.Module):
    """Casts state to half for FP16 actor inference."""

    def __init__(self, actor: Actor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state.half())

    @staticmethod
    def select_action(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return Actor.select_action(logits.float())
