"""Mehregan et al. Fig 6a / 6b — PTQ and QAT efficacy (shared by both panels).

§IV.A.3: the fully trained fp32 model (Fig 5a / 5b) is post-training quantized
to fp16 and int8 (PyTorch ``quantize_dynamic``), and a separate model is trained
with quantization-aware training "with the same settings" (10 episodes). All
run the §IV.A.2 eval on the same seed: 2 s reset, then five 2 s steps of each
actor closed loop.

Fig 6a uses the Fig 4a 45 Hz checkpoint; Fig 6b uses the Fig 5b 30 Hz
checkpoint. ``--train-qat`` trains the QAT model with ``paper_protocol`` once
per QAT training seed (``--qat-seeds``, default 0–14, one subprocess per seed).
10-episode int8 QAT does not converge — its logits saturate at the fake-quant
clamp and the deployed argmax resolves a tie — so the pattern it plays varies
by training seed. QAT is therefore shown and gated as a seed band (thin line
per seed, median in bold), not as one draw.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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

SERIES = ("fp32", "ptq_int8", "ptq_fp16")
DEFAULT_QAT_SEEDS = "0-14"
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

    @property
    def plot_script(self) -> Path:
        return _HERE / self.panel / "plot.py"


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
        "qat": {"label": "QAT", "color": "#e8731a", "ls": "-", "z": 1},
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


def parse_seeds(spec: str) -> list[int]:
    """``"0-6"`` / ``"0,2,5"`` / ``"3"`` → sorted unique seeds."""
    seeds: set[int] = set()
    for part in spec.split(","):
        lo, sep, hi = part.strip().partition("-")
        seeds.update(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return sorted(seeds)


def qat_checkpoint_path(qat_dir: Path, seed: int) -> Path:
    return qat_dir / f"qat_checkpoint_seed{seed}.pt"


def train_one_qat_seed(cfg: QuantPanel, *, seed: int, qat_dir: Path, knobs: Any) -> int:
    """Train one 10-episode QAT seed (worker entry of ``--train-qat``)."""
    record = _PP.train_paper(
        mean_hz=cfg.mean_hz,
        seed=seed,
        knobs=knobs,
        variant="qat",
        checkpoint_path=qat_checkpoint_path(qat_dir, seed),
        extra={"figure": f"mehregan_fig{cfg.panel}_qat"},
    )
    (qat_dir / f"qat_series_seed{seed}.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


def train_qat_seeds(
    cfg: QuantPanel, *, seeds: list[int], qat_dir: Path, overrides: list[str], workers: int
) -> dict[str, Any]:
    """Train every QAT seed in its own single-threaded subprocess, ``workers`` at a time."""
    qat_dir.mkdir(parents=True, exist_ok=True)
    pending = list(seeds)
    running: dict[int, tuple[subprocess.Popen, Any]] = {}
    failed: dict[int, int] = {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < max(1, workers):
            seed = pending.pop(0)
            log = (qat_dir / f"qat_train_seed{seed}.log").open("w")
            cmd = [
                sys.executable, "-m", "rl_adaptive_dbs.run", str(cfg.plot_script),
                "--train-qat-seed", str(seed), "--qat-dir", str(qat_dir),
            ]
            for pair in overrides:
                cmd += ["--set", pair]
            running[seed] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"QAT seed {seed}: training (log {log.name})", flush=True)
        for seed, (proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[seed]
            if proc.returncode:
                failed[seed] = proc.returncode
            print(f"QAT seed {seed}: exit {proc.returncode} ({time.time() - t0:.0f} s)", flush=True)
        time.sleep(2.0)
    if failed:
        msg = f"QAT training failed for seeds {failed} — see {qat_dir}/qat_train_seed*.log"
        raise RuntimeError(msg)
    out: dict[str, Any] = {}
    for seed in seeds:
        record = json.loads((qat_dir / f"qat_series_seed{seed}.json").read_text())
        out[str(seed)] = {
            "series": str(qat_dir / f"qat_series_seed{seed}.json"),
            "episode_rewards": record["episode_rewards"],
            "unique_actions": record["unique_actions"],
        }
    return out


def run_eval(
    cfg: QuantPanel, *, fp32_checkpoint: Path, qat_checkpoints: dict[int, Path], seed: int,
    knobs: Any,
) -> dict[str, Any]:
    alphabet = _PP.make_alphabet(cfg.mean_hz, knobs)
    fp32 = _PP.load_fp32_actor(fp32_checkpoint)
    modules: dict[str, torch.nn.Module] = {
        "fp32": fp32,
        "ptq_int8": _PP.ptq_actor(fp32, "int8"),
        "ptq_fp16": _PP.ptq_actor(fp32, "fp16"),
    }
    t0 = time.time()
    conditions = {
        name: _PP.run_condition(seed=seed, alphabet=alphabet, policy=_PP.module_policy(mod))
        for name, mod in modules.items()
    }
    for name, cond in conditions.items():
        print(f"{name} actions {cond['actions']}", flush=True)
    qat_seeds: dict[str, dict[str, Any]] = {}
    for train_seed, path in sorted(qat_checkpoints.items()):
        cond = _PP.run_condition(
            seed=seed, alphabet=alphabet, policy=_PP.module_policy(_PP.load_qat_actor(path))
        )
        logits = np.asarray(cond["logits"], dtype=float)
        cond["n_tied_at_max"] = [int(np.sum(np.isclose(r, r.max()))) for r in logits]
        cond["checkpoint"] = str(path)
        qat_seeds[str(train_seed)] = cond
        print(
            f"qat seed {train_seed} actions {cond['actions']} tied {cond['n_tied_at_max']}",
            flush=True,
        )
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
        "plant_dt_ms": _PP.PAPER_DT_MS,
        "conditions": conditions,
        "qat_seeds": qat_seeds,
        "ptq_logit_drift": drift,
        "elapsed_s": round(time.time() - t0, 1),
    }


def _xy(cond: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(cond["time_s"], dtype=float), np.asarray(cond["trace"], dtype=float)


def _closed(cond: dict[str, Any]) -> bool:
    return bool(cond.get("closed_loop")) and bool(cond.get("actions"))


def quant_gates(cfg: QuantPanel, payload: dict[str, Any]) -> dict[str, Any]:
    conds = payload["conditions"]
    qat = payload["qat_seeds"]
    traces = {k: _xy(conds[k]) for k in SERIES}
    closed = {k: _closed(conds[k]) for k in SERIES}
    closed["qat"] = bool(qat) and all(_closed(c) for c in qat.values())
    return fig6_quant_gates(
        traces,
        qat_seeds={s: _xy(c) for s, c in qat.items()},
        panel=cfg.panel,
        closed_loop=closed,
    )


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
    qat = payload["qat_seeds"]
    q = style["qat"]
    t = np.asarray(next(iter(qat.values()))["time_s"], dtype=float)
    band = np.vstack([np.asarray(c["trace"], dtype=float) for c in qat.values()])
    ax.fill_between(
        t, band.min(axis=0), band.max(axis=0), color=q["color"], alpha=0.12, linewidth=0,
        zorder=0.5, label=f"QAT seed range ({len(qat)} train seeds)",
    )
    for row in band:
        ax.plot(t, row, color=q["color"], linewidth=0.7, alpha=0.45, zorder=0.8)
    ax.plot(
        t, np.median(band, axis=0), color=q["color"], linewidth=2.0, zorder=q["z"],
        label="QAT (median of seeds)",
    )
    flat.extend(band.ravel().tolist())
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
    qat_post = {s: _PP.post_mean(c, lo=4.0) for s, c in qat.items()}
    return {
        "out": str(out_path),
        "y_min": lo,
        "y_max": hi,
        **{f"{k}_post_mean": _PP.post_mean(conds[k], lo=4.0) for k in SERIES},
        **{f"{k}_pre_mean": _PP.pre_mean(conds[k]) for k in SERIES},
        "qat_post_mean": float(np.median(list(qat_post.values()))),
        "qat_seed_post_mean": qat_post,
        "actions": {k: conds[k]["actions"] for k in SERIES},
        "qat_seed_actions": {s: c["actions"] for s, c in qat.items()},
        "qat_seed_n_tied_at_max": {s: c.get("n_tied_at_max") for s, c in qat.items()},
        "ptq_logit_drift": payload.get("ptq_logit_drift"),
    }


def main(cfg: QuantPanel, doc: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--seed", type=int, default=0, help="Eval seed (shared by all series)")
    parser.add_argument(
        "--qat-seeds", default=DEFAULT_QAT_SEEDS,
        help="QAT training seeds shown as the QAT band (e.g. 0-14 or 0,2,5)",
    )
    parser.add_argument(
        "--qat-workers", type=int, default=3,
        help="QAT seeds trained in parallel (one single-threaded process each)",
    )
    parser.add_argument("--fp32-checkpoint", type=Path, default=cfg.fp32_checkpoint)
    parser.add_argument(
        "--qat-dir", type=Path, default=cfg.cache_dir / "qat",
        help="Per-seed QAT checkpoints, training series and logs",
    )
    parser.add_argument("--train-qat-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-json", type=Path, default=cfg.cache_dir / "eval.json")
    parser.add_argument("--manifest", type=Path, default=cfg.cache_dir / "manifest.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--train-qat", action="store_true",
        help="Train the 10-episode QAT model (paper Alg. 1 + fake quant) per QAT seed first",
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
    if args.train_qat_seed is not None:
        knobs = _PP.apply_overrides(_PP.PaperKnobs(), args.overrides)
        args.qat_dir.mkdir(parents=True, exist_ok=True)
        return train_one_qat_seed(cfg, seed=args.train_qat_seed, qat_dir=args.qat_dir, knobs=knobs)
    _resume_cli.configure_promote_publish(args, _figure_promote)
    qat_seed_list = parse_seeds(args.qat_seeds)

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
        qat_checkpoints = {s: qat_checkpoint_path(args.qat_dir, s) for s in qat_seed_list}
        if args.train_qat:
            qat_training = train_qat_seeds(
                cfg, seeds=qat_seed_list, qat_dir=args.qat_dir, overrides=args.overrides,
                workers=args.qat_workers,
            )
            archive = args.qat_dir / f"v{png_version}"
            archive.mkdir(parents=True, exist_ok=True)
            for path in qat_checkpoints.values():
                shutil.copyfile(path, archive / path.name)
        for path in (args.fp32_checkpoint, *qat_checkpoints.values()):
            if not path.exists():
                print(f"missing checkpoint: {path} (run with --train-qat)", file=sys.stderr)
                return 2
        payload = run_eval(
            cfg,
            fp32_checkpoint=args.fp32_checkpoint,
            qat_checkpoints=qat_checkpoints,
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
        "qat_seeds": sorted(int(s) for s in payload["qat_seeds"]),
        "qat_checkpoints": {s: c.get("checkpoint") for s, c in payload["qat_seeds"].items()},
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
        + " ".join(f"{k}={panel[f'{k}_post_mean']:.1f}" for k in (*SERIES, "qat")),
        flush=True,
    )
    metrics = dig["metrics"]
    print(
        f"QAT paper-like {metrics['qat_paper_like_count']}/{metrics['qat_seed_count']} seeds; "
        "late/pre by seed "
        + " ".join(f"{s}:{v:.2f}" for s, v in metrics["qat_seed_late_over_pre"].items()),
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
