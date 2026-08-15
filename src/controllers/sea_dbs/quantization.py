"""FP16 post-training quantization for SEA-DBS actor (§11)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from controllers.sea_dbs.networks import Actor

PTQ_VARIANT = "ptq-fp16"
# Same default as Mehregan Fig 6a fp16 (distinct closed-loop path vs fp32).
DEFAULT_PTQ_WEIGHT_NOISE = 0.03
DEFAULT_PTQ_NOISE_SEED = 11


def is_ptq_variant(variant: str) -> bool:
    return variant == PTQ_VARIANT


def fp_source_variant(variant: str) -> str:
    if is_ptq_variant(variant):
        return "paper"
    return variant


def perturb_float_parameters(
    module: nn.Module,
    *,
    scale: float,
    seed: int,
) -> nn.Module:
    """Deep-copy ``module`` and add Gaussian noise to floating parameters."""
    if scale < 0:
        msg = f"ptq weight noise scale must be >= 0, got {scale}"
        raise ValueError(msg)
    out = copy.deepcopy(module)
    if scale == 0:
        return out
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for param in out.parameters():
            if not torch.is_floating_point(param):
                continue
            noise = torch.randn(
                param.shape,
                generator=generator,
                dtype=param.dtype,
                device=param.device,
            )
            param.add_(noise * scale)
    return out


def apply_fp16_ptq(
    actor: Actor,
    *,
    weight_noise: float = 0.0,
    noise_seed: int = DEFAULT_PTQ_NOISE_SEED,
) -> Actor:
    """Return a deep-copied actor with FP16 weights for inference.

    ``weight_noise`` is Gaussian perturbation applied to fp32 weights *before*
    ``.half()`` (Mehregan Fig 6a/6b PTQ split). Scale 0 keeps a plain copy.
    """
    src = (
        perturb_float_parameters(actor, scale=float(weight_noise), seed=int(noise_seed))
        if float(weight_noise) > 0
        else copy.deepcopy(actor)
    )
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
