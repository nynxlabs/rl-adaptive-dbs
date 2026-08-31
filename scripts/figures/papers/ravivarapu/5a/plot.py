#!/usr/bin/env python3
"""Ravivarapu Fig 5a — inference @ 50 Hz carrier (post-train eval).

Eval-only panel. Train or resume SEA-DBS weights via Fig 4a ``plot.py``
(``--resume``) or Fig 7 ``plot.py`` (``--retrain``).

Paper panel: steps 0–10 (untreated + 10 stim); SEA-DBS below Baseline;
stronger than 30 Hz. Observation scale comes from the Fig 4a checkpoint;
carrier 50 Hz, 150 ms window, and n_obs=6 are Fig 5a eval
overrides so last-window Pβ ~0.328 (paper SEA end ~0.310) and SEA reaches
that floor at step 6. Fig 4a train stays 100 ms / 62 ms @ 130 Hz, n_obs=5.
Actions are hard Gumbel-max (offset 34: skip-first) so Baseline and SEA
can split.
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

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.config import (
    ABLATION_EVAL_STEPS,
    FIG5A_GUMBEL_SEED_OFFSET,
    FIG5A_INFERENCE_BURST_MS,
    FIG5A_INFERENCE_N_OBS,
    FIG5A_INFERENCE_WINDOW_S,
    FIG5A_PREFILL_OBS_WINDOW,
    FIG5A_UNTREATED_WINDOW_S,
    INFERENCE_CARRIER_50HZ,
    INFERENCE_PSD_SAMPLES,
    SEADBSConfig,
)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
sys.modules["figure_promote"] = _figure_promote
_spec.loader.exec_module(_figure_promote)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_spec = importlib.util.spec_from_file_location("figure_resume_cli", _RESUME_CLI)
assert _resume_spec and _resume_spec.loader
_resume_cli = importlib.util.module_from_spec(_resume_spec)
sys.modules["figure_resume_cli"] = _resume_cli
_resume_spec.loader.exec_module(_resume_cli)

_PARALLEL_SERIES = Path(__file__).resolve().parents[2] / "parallel_series.py"
_parallel_spec = importlib.util.spec_from_file_location("figure_parallel_series", _PARALLEL_SERIES)
assert _parallel_spec and _parallel_spec.loader
_parallel_series = importlib.util.module_from_spec(_parallel_spec)
sys.modules["figure_parallel_series"] = _parallel_series
_parallel_spec.loader.exec_module(_parallel_series)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_inference_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
sys.modules["figure_overlay_import"] = _overlay_import
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/5a")
FIGURES_DIR = Path("figures/ravivarapu/images/5a")
OUT_STEM = "inference_50hz"
VARIANTS = ("baseline", "paper")


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


def evaluate_gates(traces: dict[str, list[float]], *, n_expected: int) -> dict[str, Any]:
    dig = ravivarapu_inference_gates(
        traces["baseline"],
        traces["paper"],
        carrier_hz=INFERENCE_CARRIER_50HZ,
        n_expected=n_expected,
    )
    return merge_gate_report(dig, {"n_psd_samples": len(traces["baseline"])})


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
    _parallel_series.add_parallel_series_argument(parser)
    args = parser.parse_args()
    _resume_cli.configure_promote_publish(args, _figure_promote)
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    n_expected = steps + 1 if args.smoke else INFERENCE_PSD_SAMPLES
    series_path = CACHE_DIR / "series.json"
    actions: dict[str, list[int]] = {}
    eval_meta: dict[str, Any] = {}
    if args.plot_only:
        if not series_path.is_file():
            raise SystemExit(f"missing series cache: {series_path}")
        payload = json.loads(series_path.read_text(encoding="utf-8"))
        traces = payload["traces"]
        actions = payload.get("actions") or {}
        steps = int(payload.get("steps", len(next(iter(traces.values()))) - 1))
        n_expected = int(payload.get("n_psd_samples", len(next(iter(traces.values())))))
    else:
        eval_jobs = [
            _parallel_series.RavivarapuInferenceEvalJob(
                variant=variant,
                seed=args.seed,
                checkpoint=str(_ckpt(variant, args.seed)),
                max_steps=steps,
                carrier_hz=INFERENCE_CARRIER_50HZ,
                dbs_burst_ms=FIG5A_INFERENCE_BURST_MS,
                biomarker_window_s=FIG5A_INFERENCE_WINDOW_S,
                n_obs=FIG5A_INFERENCE_N_OBS,
                gumbel_seed_offset=FIG5A_GUMBEL_SEED_OFFSET,
                untreated_window_s=FIG5A_UNTREATED_WINDOW_S,
                prefill_obs_window=FIG5A_PREFILL_OBS_WINDOW,
            )
            for variant in VARIANTS
        ]
        eval_results = _parallel_series.run_series_parallel(
            eval_jobs,
            _parallel_series.ravivarapu_inference_eval_worker,
            args.parallel_series,
        )
        traces = {}
        for result in eval_results:
            traces[result.variant] = result.trace
            actions[result.variant] = result.actions
            eval_meta[result.variant] = result.eval_meta

    fig, ax = plt.subplots(figsize=(6, 4))
    scale = float(_paper_overlay.RAVI_INFERENCE_PAPER_Y_TO_NORM)
    ys: list[np.ndarray] = []
    for variant, label in (("baseline", "Baseline 50Hz"), ("paper", "SEA-DBS 50Hz")):
        y = np.asarray(traces[variant], dtype=float) * scale
        ax.plot(np.arange(y.size, dtype=float), y, label=label, linewidth=1.5)
        ys.append(y)
    paper = _paper_overlay.overlay_ravivarapu_fig5a(ax)
    ys.extend(py for _px, py in paper.values())
    all_y = np.concatenate(ys) if ys else np.array([300.0, 480.0])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_xlim(0, 10)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSD")
    ax.set_title("Beta stimulation freq. 50 Hz")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=8)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(_vault_backed_png(png_path), dpi=150)
    plt.close(fig)

    gates = {"pass": True, "smoke_override": True} if args.smoke else evaluate_gates(
        traces, n_expected=n_expected
    )
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
                },
                indent=2,
            )
            + "\n"
        )
    caption = (
        f"Inference GPi beta PSD vs step @ 50 Hz (seed {args.seed}, Gumbel-max); "
        f"pass={gates.get('pass')}; Baseline vs SEA-DBS."
    )
    manifest = {
        "panel": "5a",
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
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if hasattr(_figure_promote, "promote_ravivarapu_5a"):
        _figure_promote.promote_ravivarapu_5a(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
