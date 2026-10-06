"""Mehregan et al. Fig 6a / 6b — PTQ and QAT efficacy (shared by both panels).

§IV.A.3: the fully trained fp32 model (Fig 5a / 5b) is post-training quantized
to fp16 and int8 (PyTorch ``quantize_dynamic``), and a separate model is trained
with quantization-aware training "with the same settings" (10 episodes). All
four run the §IV.A.2 eval on the same seed: 2 s reset, then five 2 s steps of
each actor closed loop.

Fig 6a uses the Fig 4a 45 Hz checkpoint; Fig 6b uses the Fig 5b 30 Hz
checkpoint. ``--train-qat`` trains the QAT model with ``paper_protocol``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_adaptive_dbs import panel as _panel
from rl_adaptive_dbs.panel import load_script_module

_HERE = Path(__file__).resolve().parent
_PP = load_script_module("mehregan_paper_protocol", _HERE / "paper_protocol.py")

_DIG = _HERE.parents[2] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig6_quant_gates  # noqa: E402

_overlay_import = load_script_module("figure_overlay_import", _HERE.parent / "overlay_import.py")
_paper_overlay = _overlay_import.load_paper_overlay()
_figure_promote = load_script_module("figure_promote", _HERE.parent / "promote.py")
_resume_cli = load_script_module("figure_resume_cli", _HERE.parent / "resume_cli.py")

SERIES = ("fp32", "ptq_int8", "ptq_fp16", "qat")
STIM_ONSET_S = _PP.ONSET_DISPLAY_S
TIME_MAX_S = _PP.DISPLAY_S
STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10}


@dataclass(frozen=True)
class QuantPanel:
    panel: str
    mean_hz: float
    out_stem: str
    fp32_checkpoint: Path
    fp32_label: str
    y_min: float
    y_max: float

    @property
    def figures_dir(self) -> Path:
        return Path(f"figures/mehregan/images/{self.panel}")

    @property
    def cache_dir(self) -> Path:
        return Path(f"artifacts/figures/papers/mehregan/{self.panel}")


FIG6A = QuantPanel(
    panel="6a",
    mean_hz=45.0,
    out_stem="ptq_qat_45hz",
    fp32_checkpoint=Path("artifacts/figures/papers/mehregan/4a/checkpoint.pt"),
    fp32_label="Fully Trained 45Hz",
    y_min=250.0,
    y_max=550.0,
)

FIG6B = QuantPanel(
    panel="6b",
    mean_hz=30.0,
    out_stem="ptq_qat_30hz",
    fp32_checkpoint=Path("artifacts/figures/papers/mehregan/5b/checkpoint.pt"),
    fp32_label="Fully Trained 30Hz",
    y_min=300.0,
    y_max=550.0,
)


def _style(cfg: QuantPanel) -> dict[str, dict[str, Any]]:
    return {
        "fp32": {"label": cfg.fp32_label, "color": "#2ecc40", "ls": "-", "z": 4},
        "ptq_int8": {"label": "PTQ, INT8", "color": "#1f3fd1", "ls": "-", "z": 3},
        "ptq_fp16": {"label": "PTQ, FP16", "color": "#7b2d8e", "ls": "-", "z": 2},
        "qat": {"label": "QAT", "color": "#e8731a", "ls": "--", "z": 1},
    }


def _linked_png(path: Path) -> Path:
    paper = path.parent / "paper.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not paper.is_symlink() or path.exists() or path.is_symlink():
        return path
    target = paper.resolve().parent / path.name
    if not target.exists():
        target.touch()
    path.symlink_to(target)
    return path


def _logit_drift(fp32: list[list[float]], other: list[list[float]]) -> float:
    """Max |logit difference| over the eval steps (report only)."""
    a = np.asarray(fp32, dtype=float)
    b = np.asarray(other, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    return float(np.max(np.abs(a - b)))


def run_eval(
    cfg: QuantPanel, *, fp32_checkpoint: Path, qat_checkpoint: Path, seed: int, knobs: Any
) -> dict[str, Any]:
    alphabet = _PP.make_alphabet(cfg.mean_hz, knobs)
    fp32 = _PP.load_fp32_actor(fp32_checkpoint)
    modules: dict[str, torch.nn.Module] = {
        "fp32": fp32,
        "ptq_int8": _PP.ptq_actor(fp32, "int8"),
        "ptq_fp16": _PP.ptq_actor(fp32, "fp16"),
        "qat": _PP.load_qat_actor(qat_checkpoint),
    }
    t0 = time.time()
    conditions = {
        name: _PP.run_condition(seed=seed, alphabet=alphabet, policy=_PP.module_policy(mod))
        for name, mod in modules.items()
    }
    for name, cond in conditions.items():
        print(f"{name} actions {cond['actions']}", flush=True)
    drift = {
        name: _logit_drift(conditions["fp32"]["logits"], conditions[name]["logits"])
        for name in ("ptq_int8", "ptq_fp16")
    }
    return {
        "figure": f"mehregan_fig{cfg.panel}",
        "protocol": "paper_closed_loop",
        "mean_hz": cfg.mean_hz,
        "seed": seed,
        "knobs": knobs.as_dict(),
        "fp32_checkpoint": str(fp32_checkpoint),
        "qat_checkpoint": str(qat_checkpoint),
        "plant_dt_ms": _PP.PAPER_DT_MS,
        "conditions": conditions,
        "ptq_logit_drift": drift,
        "elapsed_s": round(time.time() - t0, 1),
    }


def quant_gates(cfg: QuantPanel, payload: dict[str, Any]) -> dict[str, Any]:
    conds = payload["conditions"]
    traces = {
        k: (np.asarray(v["time_s"], dtype=float), np.asarray(v["trace"], dtype=float))
        for k, v in conds.items()
    }
    closed = {k: bool(v.get("closed_loop")) and bool(v.get("actions")) for k, v in conds.items()}
    return fig6_quant_gates(traces, panel=cfg.panel, closed_loop=closed)


def plot_panel(cfg: QuantPanel, payload: dict[str, Any], *, out_path: Path) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    style = _style(cfg)
    flat: list[float] = []
    for key in SERIES:
        cond = payload["conditions"][key]
        s = style[key]
        ax.plot(
            cond["time_s"], cond["trace"], color=s["color"], linestyle=s["ls"],
            linewidth=1.6, zorder=s["z"], label=s["label"],
        )
        flat.extend(cond["trace"])
    ax.axvline(STIM_ONSET_S, color="#888888", linestyle="--", linewidth=1.2, zorder=0)
    getattr(_paper_overlay, f"overlay_mehregan_fig{cfg.panel}")(ax)
    lo = min(cfg.y_min, float(np.floor((min(flat) - 20.0) / 50.0) * 50.0))
    hi = max(cfg.y_max, float(np.ceil((max(flat) + 20.0) / 50.0) * 50.0))
    ax.set_xlim(0.0, TIME_MAX_S)
    ax.set_ylim(lo, hi)
    ax.set_xticks(np.arange(0.0, TIME_MAX_S + 1e-9, 2.0))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("PSD")
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    fig.tight_layout()
    _paper_overlay.place_legend(ax, loc="lower left", fontsize=8, ncol=1)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    conds = payload["conditions"]
    return {
        "out": str(out_path),
        "y_min": lo,
        "y_max": hi,
        **{f"{k}_post_mean": _PP.post_mean(conds[k], lo=4.0) for k in SERIES},
        **{f"{k}_pre_mean": _PP.pre_mean(conds[k]) for k in SERIES},
        "actions": {k: conds[k]["actions"] for k in SERIES},
        "ptq_logit_drift": payload.get("ptq_logit_drift"),
    }


def main(cfg: QuantPanel, doc: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--seed", type=int, default=0, help="Eval seed (shared by all series)")
    parser.add_argument("--train-seed", type=int, default=0, help="QAT training seed")
    parser.add_argument("--fp32-checkpoint", type=Path, default=cfg.fp32_checkpoint)
    parser.add_argument("--qat-checkpoint", type=Path, default=cfg.cache_dir / "qat_checkpoint.pt")
    parser.add_argument("--qat-series", type=Path, default=cfg.cache_dir / "qat_series.json")
    parser.add_argument("--eval-json", type=Path, default=cfg.cache_dir / "eval.json")
    parser.add_argument("--manifest", type=Path, default=cfg.cache_dir / "manifest.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--train-qat", action="store_true",
        help="Train the 10-episode QAT model (paper Alg. 1 + fake quant) first",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="FIELD=VALUE",
        help="Paper-silent knob override (repeatable); must match the fp32 training",
    )
    parser.add_argument("--no-update-docs", dest="update_docs", action="store_false")
    parser.set_defaults(update_docs=True)
    _resume_cli.add_export_notes_arg(parser)
    _resume_cli.add_update_report3_arg(parser)
    args = parser.parse_args()
    _resume_cli.configure_promote_publish(args, _figure_promote)

    if args.out is None:
        cfg.figures_dir.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(cfg.figures_dir, cfg.out_stem)
        args.out = _linked_png(args.out)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)

    qat_training: dict[str, Any] | None = None
    if args.plot_only:
        if not args.eval_json.exists():
            print(f"missing eval JSON: {args.eval_json}", file=sys.stderr)
            return 2
        payload = json.loads(args.eval_json.read_text())
    else:
        knobs = _PP.apply_overrides(_PP.PaperKnobs(), args.overrides)
        if args.train_qat:
            record = _PP.train_paper(
                mean_hz=cfg.mean_hz,
                seed=args.train_seed,
                knobs=knobs,
                variant="qat",
                checkpoint_path=args.qat_checkpoint,
                extra={"figure": f"mehregan_fig{cfg.panel}_qat", "png_version": png_version},
            )
            args.qat_series.parent.mkdir(parents=True, exist_ok=True)
            args.qat_series.write_text(json.dumps(record, indent=2) + "\n")
            shutil.copyfile(
                args.qat_checkpoint,
                args.qat_checkpoint.with_name(f"{args.qat_checkpoint.stem}_v{png_version}.pt"),
            )
            qat_training = {
                "series": str(args.qat_series),
                "episode_rewards": record["episode_rewards"],
                "unique_actions": record["unique_actions"],
                "action_counts": record["action_counts"],
            }
        for path in (args.fp32_checkpoint, args.qat_checkpoint):
            if not path.exists():
                print(f"missing checkpoint: {path}", file=sys.stderr)
                return 2
        payload = run_eval(
            cfg,
            fp32_checkpoint=args.fp32_checkpoint,
            qat_checkpoint=args.qat_checkpoint,
            seed=args.seed,
            knobs=knobs,
        )
        if qat_training is not None:
            payload["qat_training"] = qat_training
        args.eval_json.parent.mkdir(parents=True, exist_ok=True)
        args.eval_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.eval_json}", flush=True)

    panel = plot_panel(cfg, payload, out_path=args.out)
    dig = quant_gates(cfg, payload)
    gates = dict(dig["gates"])
    panel["gates"] = gates
    manifest = {
        "figure": f"mehregan_fig{cfg.panel}",
        "protocol": payload.get("protocol"),
        "mean_hz": cfg.mean_hz,
        "seed": payload.get("seed"),
        "knobs": payload.get("knobs"),
        "plant_dt_ms": payload.get("plant_dt_ms"),
        "eval_json": str(args.eval_json),
        "fp32_checkpoint": payload.get("fp32_checkpoint"),
        "qat_checkpoint": payload.get("qat_checkpoint"),
        "qat_training": payload.get("qat_training"),
        "output_png": _figure_promote.repo_rel_posix(args.out),
        "png_version": png_version,
        "panel": panel,
        "gates": gates,
        "all_pass": bool(dig["pass"]),
        "paper_gate_metrics": dig["metrics"],
        "paper_ref": dig["paper_ref"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = _panel.stamp_manifest(manifest)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out}\nwrote {args.manifest}", flush=True)
    print(
        "all_pass={} ".format(manifest["all_pass"])
        + " ".join(f"{k}={panel[f'{k}_post_mean']:.1f}" for k in SERIES),
        flush=True,
    )
    print(gates, flush=True)

    if args.update_docs:
        promote = getattr(_figure_promote, f"promote_{cfg.panel}")
        updated = promote(
            manifest=manifest, eval_path=args.eval_json, png_path=args.out, update_docs=True
        )
        print(f"updated docs caption: {updated.get('caption')}", flush=True)
    return 0 if manifest["all_pass"] else 1
