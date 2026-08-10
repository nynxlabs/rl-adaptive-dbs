#!/usr/bin/env python3
"""Nguyen et al. Figure 5 — CBGT spikes and DBS energy over training.

Reads the shared Fig. 4 train cache (``artifacts/figures/papers/nguyen/4/series.json``).
Panel (a) per-episode CBGT spike totals; (b) per-episode DBS energy (Eq. (6) sum).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/5/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/5/plot.py --refresh-train
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from nguyen_gates import attach_digitization, fig5_spikes_energy_gates  # noqa: E402

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

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_import_spec and _overlay_import_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_import_spec)
_overlay_import_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

import matplotlib.pyplot as plt
import numpy as np

FIG4_CACHE = Path("artifacts/figures/papers/nguyen/4")
FIG4_SERIES = FIG4_CACHE / "series.json"
FIG4_MANIFEST = FIG4_CACHE / "manifest.json"
FIGURES_DIR = Path("figures/nguyen/images/5")
CACHE_DIR = Path("artifacts/figures/papers/nguyen/5")
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
OUT_STEM = "spikes_energy"
SMOOTH_WINDOW = 20
ENERGY_FLAT_FRAC = 0.01
# Paper Fig. 5 axis bands (one-seed qualitative).
PAPER_SPIKE_MIN = 400.0
PAPER_SPIKE_MAX = 950.0
PAPER_ENERGY_MIN = 300.0
PAPER_ENERGY_MAX = 3200.0

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#cccccc",
    "font.size": 10,
}


def _vault_backed_png(path: Path) -> Path:
    path = Path(path)
    paper = path.parent / "paper.png"
    if not paper.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    vault_dir = paper.resolve().parent
    vault_target = vault_dir / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return path
    if not vault_target.exists():
        vault_target.touch()
    path.symlink_to(vault_target)
    return path


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    window = max(1, int(window))
    if window == 1 or y.size == 1:
        return y.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(y.astype(float), (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: y.size]


def load_series(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Fig 4 series cache: {path}")
    series = json.loads(path.read_text(encoding="utf-8"))
    for key in ("episode_spike_totals", "episode_energies"):
        if key not in series or len(series[key]) != len(series.get("episode_rewards", [])):
            msg = (
                f"series missing {key!r}; re-run Fig 4 train with spike/energy logging:\n"
                "  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py"
            )
            raise ValueError(msg)
    return series


def evaluate_gates(series: dict[str, Any], *, fig4_manifest: dict[str, Any] | None) -> dict[str, Any]:
    spikes = np.asarray(series["episode_spike_totals"], dtype=float)
    energies = np.asarray(series["episode_energies"], dtype=float)
    n = int(spikes.size)
    shared_train = bool(
        fig4_manifest is not None
        and fig4_manifest.get("gates", {}).get("pass")
        and int(series.get("num_episodes", 0)) == int(fig4_manifest.get("gates", {}).get("n_episodes", -1))
    )
    spike_std = float(np.std(spikes)) if n else 0.0
    energy_std = float(np.std(energies)) if n else 0.0
    energy_mean = float(np.mean(energies)) if n else 0.0
    spike_mean = float(np.mean(spikes)) if n else 0.0
    spike_series_has_variance = spike_std > 0.0
    energy_series_has_variance = energy_std > 0.0
    energy_not_constant = energy_std > ENERGY_FLAT_FRAC * max(abs(energy_mean), 1.0)
    spike_in_paper_band = PAPER_SPIKE_MIN <= spike_mean <= PAPER_SPIKE_MAX
    energy_in_paper_band = (
        PAPER_ENERGY_MIN <= energy_mean <= PAPER_ENERGY_MAX
        and float(np.max(energies)) <= PAPER_ENERGY_MAX * 1.1
    )
    gates = {
        "n_episodes": n,
        "shared_train": shared_train,
        "spike_series_has_variance": spike_series_has_variance,
        "energy_series_has_variance": energy_series_has_variance,
        "energy_not_constant": energy_not_constant,
        "spike_in_paper_band": spike_in_paper_band,
        "energy_in_paper_band": energy_in_paper_band,
        "spike_mean": spike_mean,
        "energy_mean": energy_mean,
        "spike_std": spike_std,
        "energy_std": energy_std,
    }
    gates["pass"] = bool(
        shared_train
        and spike_series_has_variance
        and energy_series_has_variance
        and energy_not_constant
        and spike_in_paper_band
        and energy_in_paper_band
    )
    if series.get("smoke"):
        gates["pass"] = True
        gates["smoke_override"] = True
        return gates

    dig = fig5_spikes_energy_gates(spikes, energies)
    return attach_digitization(gates, dig)


def plot_series(series: dict[str, Any], out_path: Path, *, smooth_window: int) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    spikes = np.asarray(series["episode_spike_totals"], dtype=float)
    energies = np.asarray(series["episode_energies"], dtype=float)
    episodes = np.arange(spikes.size, dtype=float)

    spike_smooth = moving_average(spikes, smooth_window)
    energy_smooth = moving_average(energies, smooth_window)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(episodes, spikes, color="#7b6ba8", linewidth=0.9, alpha=0.85, label="Raw")
    ax0.plot(episodes, spike_smooth, color="#4a148c", linewidth=1.8, label="Smoothed")
    ax0.set_ylabel("Spike Count")
    ax0.set_title("CBGT Network Spikes")
    ax0.grid(True, linestyle="--", alpha=0.6)

    ax1 = axes[1]
    ax1.plot(episodes, energies, color="#b2df8a", linewidth=0.8, alpha=0.85, label="Raw")
    ax1.plot(episodes, energy_smooth, color="#1b7837", linewidth=2.0, label="Smoothed")
    paper_y = _paper_overlay.overlay_nguyen_fig5(ax0, axes[1])
    spike_hi = max(
        PAPER_SPIKE_MAX + 50.0,
        float(np.nanmax(spikes)) + 50.0 if spikes.size else PAPER_SPIKE_MAX,
        float(np.nanmax(paper_y["spikes"][0])) + 50.0,
    )
    ax0.set_ylim(PAPER_SPIKE_MIN - 50.0, spike_hi)
    _paper_overlay.place_legend(ax0, fontsize=8, loc="upper right")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Energy (a.u.)")
    ax1.set_title("DBS Energy Consumption")
    energy_hi = max(
        PAPER_ENERGY_MAX + 200.0,
        float(np.nanmax(energies)) + 200.0 if energies.size else PAPER_ENERGY_MAX,
        float(np.nanmax(paper_y["energy"][0])) + 200.0,
        float(np.nanmax(paper_y["energy"][1])) + 200.0,
    )
    ax1.set_ylim(0.0, energy_hi)
    ax1.grid(True, linestyle="--", alpha=0.6)
    _paper_overlay.place_legend(ax1, fontsize=8, loc="upper right")

    last_ep = max(0, spikes.size - 1)
    for ax in axes:
        ax.set_xlim(0.0, float(last_ep))
        if last_ep >= 100:
            ax.set_xticks(np.arange(0, last_ep + 1, 100))
        else:
            ax.set_xticks(np.arange(0, last_ep + 1, max(1, last_ep // 5)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "n_episodes": int(spikes.size),
        "spike_min": float(spikes.min()) if spikes.size else float("nan"),
        "spike_max": float(spikes.max()) if spikes.size else float("nan"),
        "energy_min": float(energies.min()) if energies.size else float("nan"),
        "energy_max": float(energies.max()) if energies.size else float("nan"),
        "smooth_window": int(smooth_window),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=FIG4_SERIES)
    parser.add_argument("--fig4-manifest", type=Path, default=FIG4_MANIFEST)
    parser.add_argument(
        "--refresh-train",
        action="store_true",
        help="re-run Fig 4 train (writes shared series.json) before plotting",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-update-docs", action="store_true")
    _resume_cli.add_training_resume_args(parser)
    args = parser.parse_args(argv)
    _resume_cli.configure_promote_publish(args, _figure_promote)

    if args.refresh_train:
        fig4_argv: list[str] = []
        if args.resume is not None:
            fig4_argv.extend(["--resume", str(args.resume)])
        if args.start_episode is not None:
            fig4_argv.extend(["--start-episode", str(args.start_episode)])
        fig4_argv.extend(["--checkpoint-interval", str(args.checkpoint_interval)])
        fig4_path = Path(__file__).resolve().parent.parent / "4" / "plot.py"
        spec = importlib.util.spec_from_file_location("nguyen_fig4_plot", fig4_path)
        assert spec and spec.loader
        fig4_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fig4_mod)
        rc = int(fig4_mod.main(fig4_argv))
        if rc != 0:
            return rc

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _vault_backed_png(args.out)

    t0 = time.perf_counter()
    series = load_series(args.series)
    fig4_manifest = None
    if args.fig4_manifest.is_file():
        fig4_manifest = json.loads(args.fig4_manifest.read_text(encoding="utf-8"))

    gates = evaluate_gates(series, fig4_manifest=fig4_manifest)
    panel = plot_series(series, args.out, smooth_window=args.smooth_window)

    caption = (
        f"Fig 4 shared train {series['num_episodes']} ep, seed={series['seed']}; "
        f"spike_mean={gates['spike_mean']:.0f}, energy_mean={gates['energy_mean']:.1f}; "
        f"pass={gates['pass']}"
    )
    manifest = {
        "panel": "2/5",
        "out": args.out.as_posix(),
        "series": args.series.as_posix(),
        "fig4_manifest": args.fig4_manifest.as_posix(),
        "gates": gates,
        "panel_stats": panel,
        "elapsed_s": time.perf_counter() - t0,
        "png_version": png_version,
        "caption": caption,
        "smoke": bool(series.get("smoke")),
    }
    write_json(args.manifest, manifest)

    if not args.no_update_docs:
        updated = _figure_promote.promote_nguyen_5(
            manifest=manifest,
            png_path=args.out,
        )
        print(f"updated comparison doc: {updated['doc']}", flush=True)

    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    if png_version is not None:
        print(f"output PNG version={png_version}", flush=True)
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
