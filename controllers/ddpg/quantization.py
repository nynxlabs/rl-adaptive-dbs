"""Mehregan §III.D quantization — PTQ (FP16/INT8) and QAT fake-quant actor."""

from __future__ import annotations

import platform

import torch
import torch.nn as nn

from controllers.ddpg.networks import Actor

PTQ_VARIANTS = frozenset({"ptq-fp16", "ptq-int8"})
QUANTIZED_VARIANTS = PTQ_VARIANTS | {"qat"}

_QAT_LAYER_TYPES = (nn.Conv1d, nn.Linear)


def is_ptq_variant(variant: str) -> bool:
    return variant in PTQ_VARIANTS


def is_quantized_variant(variant: str) -> bool:
    return variant in QUANTIZED_VARIANTS


def fp_source_variant(variant: str) -> str:
    """Checkpoint slug for full-precision weights used by PTQ variants."""
    if variant in PTQ_VARIANTS:
        return "paper"
    return variant


def qat_backend() -> str:
    """Quantized-engine backend for eager-mode QAT (``fbgemm`` on x86, ``qnnpack`` on ARM)."""
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "qnnpack"
    return "fbgemm"


def _default_qat_qconfig() -> torch.ao.quantization.QConfig:
    backend = qat_backend()
    torch.backends.quantized.engine = backend
    return torch.ao.quantization.get_default_qat_qconfig(backend)


def _assign_qat_qconfig(module: nn.Module, qconfig: torch.ao.quantization.QConfig) -> None:
    module.qconfig = qconfig
    for submodule in module.modules():
        if isinstance(submodule, _QAT_LAYER_TYPES):
            submodule.qconfig = qconfig


class QATActor(nn.Module):
    """Actor with fake-quant input and dequant logits (Mehregan §III.D)."""

    def __init__(self, actor: Actor, *, prepared: bool = True) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.actor = actor
        self.dequant = torch.ao.quantization.DeQuantStub()
        if prepared:
            self._enable_fake_quant()

    def _enable_fake_quant(self) -> None:
        qconfig = _default_qat_qconfig()
        _assign_qat_qconfig(self, qconfig)
        torch.ao.quantization.prepare_qat(self, inplace=True)

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


def is_qat_prepared(module: nn.Module) -> bool:
    return isinstance(module, QATActor) and hasattr(module.quant, "activation_post_process")


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
    qat_state_dict: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Return an actor module ready for ``run_mehregan_eval``."""
    if variant in PTQ_VARIANTS:
        prepared = apply_ptq(actor, variant)
        if variant == "ptq-int8":
            return prepared
        return prepared.to(device)
    if variant == "qat":
        qat = QATActor(actor)
        if qat_state_dict is not None:
            qat.load_state_dict(qat_state_dict)
        return qat.to(device)
    return actor.to(device)


def actor_state_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32
