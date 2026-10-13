"""Where does 10-episode QAT land across training seeds? (Mehregan Fig 6a/6b)

QAT trains with the paper recipe; its logits saturate at the fake-quant
clamp, so the deployed argmax often ties and resolves to the lowest tied
index. This probe trains QAT per seed and records the eval action, the size
of the tie, and the closed-loop late/pre level against fp32-like suppression.
Per episode it also logs the logit head's fake-quant clamp, the head's
pre-quant output range, and the tie at a reference state. Diagnostic only.

``--observer`` swaps the activation observer (paper-silent):
``ema`` = PyTorch QAT default (moving-average min/max, ``--averaging-constant``
0.01), ``minmax`` = running min/max that never lags the outputs.
``--qconfig qnnpack`` instead uses PyTorch's stock qnnpack QAT qconfig
unchanged (8-bit activations, per-tensor symmetric weights) on the qnnpack
engine. ``--init-from CKPT`` fine-tunes from a full-precision checkpoint
(actor, critic and both targets) instead of training QAT from scratch.

  uv run python -m rl_adaptive_dbs.run scripts/probes/mehregan_qat_seeds.py \
      --mean-hz 45 --seeds 1-3 --ckpt-dir artifacts/probes/qat45 --out artifacts/probes/qat45_1-3.json
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import controllers.ddpg.quantization as quantization
from rl_adaptive_dbs.panel import load_script_module

_PP = load_script_module(
    "mehregan_paper_protocol",
    Path(__file__).resolve().parents[1] / "figures/papers/mehregan/paper_protocol.py",
)

REF_STATE = 0.45  # typical P_beta/1000 at the reset segment


def _qconfig_factory(observer: str, averaging_constant: float):
    def make() -> torch.ao.quantization.QConfig:
        torch.backends.quantized.engine = quantization.qat_backend()
        default = torch.ao.quantization.get_default_qat_qconfig(quantization.qat_backend())
        if observer == "minmax":
            activation = torch.ao.quantization.FakeQuantize.with_args(
                observer=torch.ao.quantization.MinMaxObserver,
                quant_min=0, quant_max=255, dtype=torch.quint8, reduce_range=False,
            )
        else:
            activation = torch.ao.quantization.FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.ao.quantization.MovingAverageMinMaxObserver,
                averaging_constant=averaging_constant,
                quant_min=0, quant_max=255, dtype=torch.quint8, reduce_range=False,
            )
        return torch.ao.quantization.QConfig(activation=activation, weight=default.weight)

    return make


def _stock_qnnpack_qconfig() -> torch.ao.quantization.QConfig:
    torch.backends.quantized.engine = "qnnpack"
    return torch.ao.quantization.get_default_qat_qconfig("qnnpack")


def _init_from(path: Path):
    """Trainer hook: load fp32 actor/critic weights (fake-quant state starts fresh)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)

    def hook(trainer: Any) -> None:
        for module in (trainer.actor, trainer.actor_target):
            missing, unexpected = quantization.unwrap_actor(module).load_state_dict(
                payload["actor_state_dict"], strict=False
            )
            bad = [k for k in missing if "fake_quant" not in k and "post_process" not in k]
            if bad or unexpected:
                raise RuntimeError(f"fp32 → QAT load mismatch: {bad} {unexpected}")
        trainer.critic.load_state_dict(payload["critic_state_dict"])
        trainer.critic_target.load_state_dict(
            payload.get("critic_target_state_dict", payload["critic_state_dict"])
        )

    return hook


def _head_diag(policy: torch.nn.Module) -> dict[str, Any]:
    """Clamp, pre-quant head output and tie at ``REF_STATE`` (observers off)."""
    probe = copy.deepcopy(policy)
    probe.apply(torch.ao.quantization.disable_observer)
    head = probe.actor.head
    fq = head.activation_post_process
    scale, zp = float(fq.scale), int(fq.zero_point)
    captured: dict[str, torch.Tensor] = {}
    head.register_forward_pre_hook(lambda m, i: captured.__setitem__("x", i[0]))
    with torch.no_grad():
        q = probe(torch.tensor([[REF_STATE]])).squeeze(0).numpy()
        raw = torch.nn.functional.linear(
            captured["x"], head.weight_fake_quant(head.weight), head.bias
        ).squeeze(0).numpy()
    hi, lo = (fq.quant_max - zp) * scale, (fq.quant_min - zp) * scale
    tied = np.flatnonzero(np.isclose(q, q.max()))
    return {
        "clamp_lo": lo,
        "clamp_hi": hi,
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "n_raw_above_clamp": int(np.sum(raw > hi)),
        "n_raw_below_clamp": int(np.sum(raw < lo)),
        "n_distinct": int(len(np.unique(np.round(q / scale)))),
        "n_tied_at_max": int(len(tied)),
        "argmax": int(q.argmax()),
        "tied": tied.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-hz", type=float, required=True)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--freeze-observer-after", type=int, default=None)
    ap.add_argument("--observer", choices=("ema", "minmax"), default="ema")
    ap.add_argument("--averaging-constant", type=float, default=0.01)
    ap.add_argument("--qconfig", choices=("int8", "qnnpack"), default="int8")
    ap.add_argument("--init-from", type=Path, default=None, help="fine-tune from this fp32 ckpt")
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="keep QAT checkpoints here")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    quantization._default_qat_qconfig = (
        _stock_qnnpack_qconfig
        if args.qconfig == "qnnpack"
        else _qconfig_factory(args.observer, args.averaging_constant)
    )
    knobs = _PP.apply_overrides(_PP.PaperKnobs(), args.overrides)
    lo, _, hi = args.seeds.partition("-")
    alphabet = _PP.make_alphabet(args.mean_hz, knobs)
    rows = {}
    for seed in range(int(lo), int(hi or lo) + 1):
        diag: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = args.ckpt_dir or Path(tmp)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = ckpt_dir / f"qat_{int(args.mean_hz)}hz_seed{seed}.pt"
            record = _PP.train_paper(
                mean_hz=args.mean_hz, seed=seed, knobs=knobs, variant="qat", checkpoint_path=ckpt,
                qat_freeze_observer_after=args.freeze_observer_after,
                episode_hook=lambda tr, ep: diag.append({"episode": ep, **_head_diag(tr.actor)}),
                trainer_hook=_init_from(args.init_from) if args.init_from else None,
            )
            qat = _PP.load_qat_actor(ckpt)
        for d in diag:
            print(f"  ep{d['episode']}: {({k: v for k, v in d.items() if k != 'tied'})}", flush=True)
        cond = _PP.run_condition(
            seed=args.eval_seed, alphabet=alphabet, policy=_PP.module_policy(qat)
        )
        logits = np.asarray(cond["logits"])
        n_tied = [int(np.sum(np.isclose(r, r.max()))) for r in logits]
        steps = _PP.STEPS_PER_EPISODE
        row = {
            "actions": cond["actions"],
            "n_tied_at_max": n_tied[0],
            "n_tied_per_step": n_tied,
            "tied_first_step": np.flatnonzero(np.isclose(logits[0], logits[0].max())).tolist(),
            "pre": _PP.pre_mean(cond),
            "late": _PP.post_mean(cond, lo=4.0),
            "trace": cond["trace"],
            "time_s": cond["time_s"],
            "train_unique_per_episode": [
                len(set(record["actions"][i:i + steps]))
                for i in range(0, len(record["actions"]), steps)
            ],
            "train_mean_beta_per_episode": [
                float(np.mean(record["beta_trace"][i:i + steps]))
                for i in range(0, len(record["beta_trace"]), steps)
            ],
            "episode_rewards": record["episode_rewards"],
            "head_diag": diag,
            "checkpoint": str(ckpt) if args.ckpt_dir else None,
        }
        row["late_over_pre"] = row["late"] / row["pre"]
        rows[str(seed)] = row
        print(
            f"seed {seed}: actions={row['actions']} tied={n_tied} "
            f"late/pre={row['late_over_pre']:.3f}",
            flush=True,
        )
    out = {
        "mean_hz": args.mean_hz,
        "knobs": knobs.as_dict(),
        "observer": args.observer,
        "averaging_constant": args.averaging_constant,
        "freeze_observer_after": args.freeze_observer_after,
        "qconfig": args.qconfig,
        "init_from": str(args.init_from) if args.init_from else None,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
