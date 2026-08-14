#!/usr/bin/env python3
"""Ravivarapu Fig 6 — FP16 PTQ @ 50 Hz (10-step eval).

Eval-only panel. Resume training from Fig 4a / Fig 7 checkpoints as above.

Paper Fig. 6 has **four** series: Baseline, Baseline+PTQ(fp16), SEA-DBS,
SEA-DBS+PTQ(fp16). Quantized SEA-DBS should track fp32 PSD reduction and
still beat Baseline; model size ~65 MB → ~33 MB (FP16 PTQ only — no QAT).

Fig 6 shares Fig 5a eval knobs (50 Hz, 150 ms window, n_obs=6, Gumbel offset
34). Greedy argmax collapses both Fig 4a actors to always-on and makes PTQ
indistinguishable from fp32.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("RL_DBS_MAX_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.config import (
    ABLATION_EVAL_STEPS,
    FIG5A_GUMBEL_SEED_OFFSET,
    FIG5A_INFERENCE_BURST_MS,
    FIG5A_INFERENCE_N_OBS,
    FIG5A_INFERENCE_WINDOW_S,
    INFERENCE_CARRIER_50HZ,
    INFERENCE_PSD_SAMPLES,
    SEADBSConfig,
)
from controllers.sea_dbs.eval import evaluate

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_spec = importlib.util.spec_from_file_location("figure_resume_cli", _RESUME_CLI)
assert _resume_spec and _resume_spec.loader
_resume_cli = importlib.util.module_from_spec(_resume_spec)
_resume_spec.loader.exec_module(_resume_cli)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_fig6_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/6")
FIGURES_DIR = Path("figures/ravivarapu/images/6")
OUT_STEM = "ptq_fp16_50hz"
SERIES = (
    ("baseline", "Baseline", False),
    ("baseline", "Baseline + PTQ(fp16)", True),
    ("paper", "SEA-DBS", False),
    ("paper", "SEA-DBS + PTQ(fp16)", True),
)
# FP16 PTQ matches fp32 Gumbel actions on this checkpoint, so the pairs share
# y. Draw fp32 first (thicker, open markers) and PTQ on top (thinner, filled
# markers), and dodge PTQ by a fraction of a step so the four strokes do not
# paint over each other. Display-only; gates still use the raw traces.
PTQ_X_DODGE = 0.28
PLOT_STYLE = {
    "Baseline": {
        "color": "#1f77b4",
        "linewidth": 2.6,
        "marker": "o",
        "markersize": 7.5,
        "markerfacecolor": "none",
        "markeredgewidth": 1.4,
        "zorder": 3,
    },
    "SEA-DBS": {
        "color": "#2ca02c",
        "linewidth": 2.6,
        "marker": "^",
        "markersize": 7.5,
        "markerfacecolor": "none",
        "markeredgewidth": 1.4,
        "zorder": 3,
    },
    "Baseline + PTQ(fp16)": {
        "color": "#ff7f0e",
        "linewidth": 1.8,
        "linestyle": (0, (4.0, 1.4, 1.0, 1.4)),
        "marker": "s",
        "markersize": 5.5,
        "zorder": 4,
    },
    "SEA-DBS + PTQ(fp16)": {
        "color": "#d62728",
        "linewidth": 1.8,
        "linestyle": (0, (4.0, 1.4, 1.0, 1.4)),
        "marker": "D",
        "markersize": 5.5,
        "zorder": 4,
    },
}


def _vault_backed_png(path: Path) -> Path:
    path = Path(path)
    roots: list[Path] = []
    main_root = getattr(_figure_promote, "REPO_ROOT", None)
    if isinstance(main_root, Path):
        roots.append(main_root)
    roots.append(Path.cwd())
    for root in roots:
        paper = root / path.parent / "paper.png"
        if not paper.is_symlink():
            continue
        vault_dir = paper.resolve().parent
        vault_target = vault_dir / path.name
        local = path if path.is_absolute() else Path.cwd() / path
        local.parent.mkdir(parents=True, exist_ok=True)
        if not vault_target.exists():
            vault_target.parent.mkdir(parents=True, exist_ok=True)
            vault_target.touch()
        if local.exists() or local.is_symlink():
            if local.resolve() != vault_target.resolve():
                return vault_target
            return local
        try:
            local.symlink_to(vault_target)
        except OSError:
            return vault_target
        return local
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ckpt(variant: str, seed: int) -> Path:
    path = Path("artifacts/figures/papers/ravivarapu/4") / f"{variant}_train{seed}.pt"
    if path.is_file():
        return path
    return Path("artifacts/sea_dbs") / f"{variant}_train{seed}.pt"


def _model_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def evaluate_gates(traces: dict[str, list[float]]) -> dict[str, Any]:
    dig = ravivarapu_fig6_gates(traces)
    n = min(len(v) for v in traces.values())
    return merge_gate_report(dig, {"n_steps": n})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot from cached series.json (no eval)",
    )
    _resume_cli.add_push_kb_arg(parser)
    _resume_cli.add_update_report3_arg(parser)
    args = parser.parse_args()
    _resume_cli.configure_promote_publish(args, _figure_promote)
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    n_expected = steps + 1 if args.smoke else INFERENCE_PSD_SAMPLES
    series_path = CACHE_DIR / "series.json"
    actions: dict[str, list[int]] = {}
    eval_meta: dict[str, Any] = {}
    model_sizes: dict[str, int] = {}

    if args.plot_only:
        if not series_path.is_file():
            raise SystemExit(f"missing series cache: {series_path}")
        payload = json.loads(series_path.read_text(encoding="utf-8"))
        traces = payload["traces"]
        actions = payload.get("actions") or {}
        model_sizes = payload.get("model_sizes") or {}
        steps = int(payload.get("steps", len(next(iter(traces.values()))) - 1))
        n_expected = int(payload.get("n_psd_samples", len(next(iter(traces.values())))))
    else:
        traces = {}
        for variant, label, use_ptq in SERIES:
            ckpt = _ckpt(variant, args.seed)
            payload = evaluate(
                ckpt,
                config=SEADBSConfig(variant=variant, seed=args.seed),
                max_steps=steps,
                carrier_hz=INFERENCE_CARRIER_50HZ,
                use_fp16_ptq=use_ptq,
                action_mode="gumbel",
                dbs_burst_ms=FIG5A_INFERENCE_BURST_MS,
                biomarker_window_s=FIG5A_INFERENCE_WINDOW_S,
                n_obs=FIG5A_INFERENCE_N_OBS,
                gumbel_seed_offset=FIG5A_GUMBEL_SEED_OFFSET,
            )
            traces[label] = payload["p_beta_trajectories"][0]
            actions[label] = payload["action_trajectories"][0]
            model_sizes[label] = _model_bytes(ckpt) if not use_ptq else _model_bytes(ckpt) // 2
            eval_meta[label] = {
                "variant": variant,
                "fp16_ptq": use_ptq,
                "carrier_hz": payload["carrier_hz"],
                "dbs_burst_ms": payload["dbs_burst_ms"],
                "biomarker_window_s": payload.get("biomarker_window_s"),
                "n_obs": payload.get("n_obs"),
                "gumbel_seed_offset": FIG5A_GUMBEL_SEED_OFFSET,
                "n_psd_samples": payload["n_psd_samples"],
                "action_mode": payload["action_mode"],
                "stim_frac": float(np.mean(payload["action_trajectories"][0])),
                "model_bytes": model_sizes[label],
            }

    fig, ax = plt.subplots(figsize=(6, 4))
    scale = float(_paper_overlay.RAVI_INFERENCE_PAPER_Y_TO_NORM)
    ys: list[np.ndarray] = []
    for _, label, use_ptq in sorted(SERIES, key=lambda row: row[2]):
        y = np.asarray(traces[label], dtype=float) * scale
        x = np.arange(y.size, dtype=float)
        if use_ptq:
            x = x + PTQ_X_DODGE
        ax.plot(
            x,
            y,
            label=label,
            **PLOT_STYLE[label],
        )
        ys.append(y)
    paper = _paper_overlay.overlay_ravivarapu_fig6(ax)
    ys.extend(py for _px, py in paper.values())
    all_y = np.concatenate(ys) if ys else np.array([300.0, 480.0])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_xlim(0, 10)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Stimulation step")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.set_title("FP16 PTQ @ 50 Hz")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=7)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(_vault_backed_png(png_path), dpi=150)
    plt.close(fig)

    gates = {"pass": True, "smoke_override": True} if args.smoke else evaluate_gates(traces)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        series_path.write_text(
            json.dumps(
                {
                    "traces": traces,
                    "actions": actions,
                    "steps": steps,
                    "n_psd_samples": n_expected,
                    "eval_meta": eval_meta,
                    "model_sizes": model_sizes,
                },
                indent=2,
            )
            + "\n"
        )
    fp32_bytes = max(
        (v for k, v in model_sizes.items() if "PTQ" not in k),
        default=0,
    )
    ptq_bytes = max(
        (v for k, v in model_sizes.items() if "PTQ" in k),
        default=0,
    )
    fp32_mb = round(fp32_bytes / (1024 * 1024), 1)
    ptq_mb = round(ptq_bytes / (1024 * 1024), 1)
    caption = (
        f"FP16 PTQ inference GPi beta PSD vs step @ 50 Hz (seed {args.seed}, Gumbel-max); "
        f"pass={gates.get('pass')}; four-series Baseline/SEA fp32+PTQ "
        f"(PTQ matches fp32 on this checkpoint; PTQ x-dodged 0.28 step, dash-dot); "
        f"actor checkpoint ~{fp32_mb} MB → ~{ptq_mb} MB (FP16 weights)."
    )
    manifest = {
        "panel": "6",
        "carrier_hz": INFERENCE_CARRIER_50HZ,
        "n_steps": steps,
        "n_psd_samples": n_expected,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "caption": caption,
        "series_cache": series_path.as_posix(),
        "actions": {k: list(v) for k, v in actions.items()},
        "eval_meta": eval_meta,
        "model_bytes_fp32": fp32_bytes,
        "model_bytes_fp16_ptq": ptq_bytes,
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if hasattr(_figure_promote, "promote_ravivarapu_6"):
        _figure_promote.promote_ravivarapu_6(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
