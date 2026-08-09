#!/usr/bin/env python3
"""Mehregan et al. (paper 1) Figure 2a — cDBS effect on GPi beta-band power over time.

Replicates the paper panel showing GPi beta power (Eq. 1, 13–35 Hz) during a
12 s parkinsonian simulation:

  - 0–2 s: no STN DBS (shared baseline for both traces)
  - 2–12 s: either no treatment (red) or 130 Hz conventional cDBS (blue)

**Default (dense trailing):** integrate **14 s** (2 s pre-roll + 12 s display). Plot
axis is simulation time **minus 2 s**. STN DBS at **sim 4 s** (= display **2 s**).
**0.2 s** samples with a **2 s** trailing $P_\\beta$ window (overlapping, not step
bins). Biomarker windows end at **sim 14 s** (= display **12 s**); last window
``[12, 14]``. Fig 2a passes a larger Numba GPI spike buffer so recording is not
truncated before sim 14.

**Paper segment mode:** ``--sampling segment`` — six whole-segment 2 s bins
(Mehregan §IV.A.1) as a step plot.

Run:
  uv run python scripts/figures/papers/mehregan/2a/plot.py
  uv run python scripts/figures/papers/mehregan/2a/plot.py --plot-only
  uv run python scripts/figures/papers/mehregan/2a/plot.py --sampling segment
  uv run python scripts/figures/papers/mehregan/2a/plot.py --no-update-docs  # skip figures/mehregan/replications.md refresh

Plant efficacy simulation only — no RL training; checkpoint resume is not applicable.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

import matplotlib.pyplot as plt
import numpy as np

from envs.plant import DbsSpec, PlantConfig, PythonPlant
from envs.plant.biomarkers import SpectrumParams, p_beta
from envs.plant.dbs import create_dbs_current

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig2_time_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

FIGURES_DIR = Path("figures/mehregan/images/2a")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/2a")
DEFAULT_SERIES = CACHE_DIR / "series.json"
DEFAULT_OUT = FIGURES_DIR / "beta_power.png"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
DISPLAY_S = 12.0
DBS_ONSET_S = 2.0
WARMUP_S = 2.0
WINDOW_S = 2.0
STEP_S = 0.2
SEGMENT_S = 2.0
INTEGRATE_S = WARMUP_S + DISPLAY_S
DBS_ONSET_SIM = WARMUP_S + DBS_ONSET_S
BIOMARKER_SIM_END_MAX = WARMUP_S + DISPLAY_S  # sim 14 — display t=12 → window [12, 14]
# Numba default GPI buffer is 512 spikes/neuron (~12.8 s at 40 Hz). Fig 2a integrates 14 s.
FIG2A_GPI_SPIKE_HEADROOM_HZ = 60.0
FIG2A_GPI_SPIKE_BUFFER_MARGIN = 64
DEFAULT_SEEDS: tuple[int, ...] = (0,)


def fig2a_gpi_spike_buffer_size(*, integrate_s: float = INTEGRATE_S) -> int:
    """GPI spike cap per neuron for long Fig 2a integrates (Numba path only)."""
    return max(
        512,
        int(np.ceil(integrate_s * FIG2A_GPI_SPIKE_HEADROOM_HZ)) + FIG2A_GPI_SPIKE_BUFFER_MARGIN,
    )

SamplingMode = Literal["trailing", "segment"]

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.size": 10,
}
SERIES = {
    "pd_no_treatment": {
        "label": "PD no Treatment",
        "color": "#d62728",
    },
    "pd_130hz": {
        "label": "PD 130Hz Treatment",
        "color": "#1f77b4",
    },
}


def parse_seeds(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        msg = "--seeds must list at least one integer"
        raise ValueError(msg)
    return tuple(int(p) for p in parts)


def dbs_spec_with_onset(
    *,
    onset_s: float,
    frequency_hz: float,
    duration_s: float,
    dt_ms: float,
) -> DbsSpec:
    """STN drive: off until ``onset_s``, then periodic ``frequency_hz`` (0 = always off)."""
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    if frequency_hz > 0.0 and onset_s < duration_s:
        stim = create_dbs_current(
            frequency_hz,
            tmax_ms=(duration_s - onset_s) * 1000.0,
            dt_ms=dt_ms,
        )
        onset_idx = int(round(onset_s * 1000.0 / dt_ms))
        end = min(onset_idx + stim.size, n_steps)
        idbs[onset_idx:end] = stim[: end - onset_idx]
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(frequency_hz).pick_dbs_freq,
            idbs=idbs,
            mean_hz=frequency_hz,
        )
    return DbsSpec.none()


def _log(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(msg, file=sys.stderr, flush=True)


def segment_edges(duration_s: float, segment_s: float) -> np.ndarray:
    n_segments = int(round(duration_s / segment_s))
    expected = duration_s / segment_s
    if not np.isclose(n_segments, expected, rtol=0.0, atol=1e-9):
        msg = f"duration_s={duration_s} must be divisible by segment_s={segment_s}"
        raise ValueError(msg)
    return np.linspace(0.0, duration_s, n_segments + 1)


def sample_times(step_s: float, *, duration_s: float = DISPLAY_S) -> np.ndarray:
    n_steps = int(round(duration_s / step_s))
    if not np.isclose(n_steps * step_s, duration_s, rtol=0.0, atol=1e-9):
        msg = f"duration_s={duration_s} must be divisible by step_s={step_s}"
        raise ValueError(msg)
    return np.linspace(0.0, duration_s, n_steps + 1)


def trailing_window_sim(
    display_t: float,
    *,
    window_s: float = WINDOW_S,
    warmup_s: float = WARMUP_S,
    sim_window_end_cap: float = BIOMARKER_SIM_END_MAX,
) -> tuple[float, float]:
    """Map display time to a trailing sim-time window (``display = sim - warmup``).

    Window end is capped at ``sim_window_end_cap`` (sim 14 s at display 12) so the
    last sample uses ``[12, 14]`` on the simulation clock.
    """
    end = min(warmup_s + display_t, sim_window_end_cap)
    start = end - window_s
    return max(0.0, start), end


def display_segment_edges_sim(segment_s: float, *, duration_s: float = DISPLAY_S) -> np.ndarray:
    """RL segment bin edges on the simulation clock (display edges + warmup)."""
    return segment_edges(duration_s, segment_s) + WARMUP_S


def _slice_spikes(
    gpi_spikes: list[np.ndarray],
    *,
    t_start: float,
    t_end: float,
    inclusive_end: bool,
) -> list[np.ndarray]:
    sub_spikes: list[np.ndarray] = []
    for spikes in gpi_spikes:
        arr = np.asarray(spikes, dtype=float).reshape(-1)
        if arr.size == 0:
            sub_spikes.append(arr)
            continue
        if inclusive_end:
            mask = (arr >= t_start) & (arr <= t_end)
        else:
            mask = (arr >= t_start) & (arr < t_end)
        sub_spikes.append(arr[mask] - t_start)
    return sub_spikes


def segment_p_beta(
    gpi_spikes: list[np.ndarray],
    *,
    dt_ms: float,
    edges: np.ndarray,
    label: str = "trace",
    verbose: bool = True,
) -> np.ndarray:
    """One whole-segment $P_\\beta$ per RL step (Mehregan §IV.A.1)."""
    spectrum = SpectrumParams.from_dt_ms(dt_ms)
    n_segments = edges.size - 1
    values = np.zeros(n_segments, dtype=float)
    t0_all = time.monotonic()
    _log(f"    [{label}] segment P_beta: {n_segments} x {edges[1] - edges[0]:.1f}s windows", verbose=verbose)
    for idx in range(n_segments):
        t_start = float(edges[idx])
        t_end = float(edges[idx + 1])
        seg_len = t_end - t_start
        inclusive_end = idx == n_segments - 1
        sub_spikes = _slice_spikes(
            gpi_spikes,
            t_start=t_start,
            t_end=t_end,
            inclusive_end=inclusive_end,
        )
        values[idx] = p_beta(
            sub_spikes,
            dt_ms=dt_ms,
            segment_duration_s=seg_len,
            params=spectrum,
        )
        _log(
            f"    [{label}] segment {idx + 1}/{n_segments} "
            f"[{t_start:.0f},{t_end:.0f})s  P_beta={values[idx]:.1f}",
            verbose=verbose,
        )
    if verbose:
        _log(
            f"    [{label}] segments done in {time.monotonic() - t0_all:.1f}s "
            f"(range {float(values.min()):.1f}..{float(values.max()):.1f})",
            verbose=True,
        )
    return values


def trailing_p_beta(
    gpi_spikes: list[np.ndarray],
    *,
    dt_ms: float,
    times: np.ndarray,
    window_s: float = WINDOW_S,
    label: str = "trace",
    verbose: bool = True,
    progress_every: int = 10,
) -> np.ndarray:
    """Trailing-window $P_\\beta$ on the display axis (dense time series)."""
    spectrum = SpectrumParams.from_dt_ms(dt_ms)
    n_samples = times.size
    values = np.zeros(n_samples, dtype=float)
    step_s = float(times[1] - times[0]) if n_samples > 1 else STEP_S
    t0_all = time.monotonic()
    _log(
        f"    [{label}] trailing P_beta: {n_samples} samples, "
        f"{window_s:.1f}s window every {step_s:.1f}s",
        verbose=verbose,
    )
    for idx, display_t in enumerate(times):
        t_start, t_end = trailing_window_sim(
            float(display_t),
            window_s=window_s,
        )
        seg_len = t_end - t_start
        inclusive_end = idx == n_samples - 1
        sub_spikes = _slice_spikes(
            gpi_spikes,
            t_start=t_start,
            t_end=t_end,
            inclusive_end=inclusive_end,
        )
        values[idx] = p_beta(
            sub_spikes,
            dt_ms=dt_ms,
            segment_duration_s=seg_len,
            params=spectrum,
        )
        if verbose and (idx == 0 or idx == n_samples - 1 or (idx + 1) % progress_every == 0):
            _log(
                f"    [{label}] sample {idx + 1}/{n_samples} "
                f"t={display_t:.1f}s sim[{t_start:.1f},{t_end:.1f}) "
                f"P_beta={values[idx]:.1f}",
                verbose=True,
            )
    if verbose:
        _log(
            f"    [{label}] trailing done in {time.monotonic() - t0_all:.1f}s "
            f"(range {float(values.min()):.1f}..{float(values.max()):.1f})",
            verbose=True,
        )
    return values


def step_plot_xy(edges: np.ndarray, segment_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand segment bins to a steps-post polyline through ``t=0`` … ``duration``."""
    y = np.concatenate([segment_values, [segment_values[-1]]])
    return edges, y


def simulate_trace(
    plant: PythonPlant,
    *,
    seed: int,
    stim_hz_after_onset: float,
    sampling: SamplingMode,
    segment_s: float,
    edges: np.ndarray,
    times: np.ndarray,
    window_s: float,
    label: str,
    verbose: bool = True,
) -> np.ndarray:
    """Integrate warmup + display; return biomarker series for one condition."""
    plant.config = PlantConfig(pd=1)
    plant.reset(seed=seed)
    dt_ms = plant.config.dt_ms
    dbs_onset_sim = DBS_ONSET_SIM
    if stim_hz_after_onset > 0.0:
        spec = dbs_spec_with_onset(
            onset_s=dbs_onset_sim,
            frequency_hz=stim_hz_after_onset,
            duration_s=INTEGRATE_S,
            dt_ms=dt_ms,
        )
        stim_note = f"130 Hz from display t={DBS_ONSET_S:.0f}s (sim {dbs_onset_sim:.0f}s)"
    else:
        spec = DbsSpec.none()
        stim_note = "no DBS"
    _log(
        f"    [{label}] integrate {INTEGRATE_S:.0f}s plant "
        f"({WARMUP_S:.0f}s pre-roll + {DISPLAY_S:.0f}s display, seed {seed}, {stim_note})...",
        verbose=verbose,
    )
    t_integrate0 = time.monotonic()
    result = plant.integrate(
        INTEGRATE_S,
        spec,
        gpi_spike_buffer_size=fig2a_gpi_spike_buffer_size(integrate_s=INTEGRATE_S),
    )
    integrate_elapsed = time.monotonic() - t_integrate0
    if not result.gpi_spikes:
        msg = "plant integrate did not record GPi spikes"
        raise RuntimeError(msg)
    spike_count = sum(int(np.asarray(s).size) for s in result.gpi_spikes)
    _log(
        f"    [{label}] integrate done in {integrate_elapsed:.1f}s ({spike_count:,} GPi spikes)",
        verbose=verbose,
    )
    if sampling == "segment":
        return segment_p_beta(
            result.gpi_spikes,
            dt_ms=result.dt_ms,
            edges=edges,
            label=label,
            verbose=verbose,
        )
    return trailing_p_beta(
        result.gpi_spikes,
        dt_ms=result.dt_ms,
        times=times,
        window_s=window_s,
        label=label,
        verbose=verbose,
    )


def simulate_seed(
    plant: PythonPlant,
    *,
    seed: int,
    sampling: SamplingMode,
    segment_s: float,
    edges: np.ndarray,
    times: np.ndarray,
    window_s: float,
    verbose: bool = True,
) -> dict[str, Any]:
    _log(f"  seed {seed}: condition 1/2 — PD no treatment", verbose=verbose)
    p_none = simulate_trace(
        plant,
        seed=seed,
        stim_hz_after_onset=0.0,
        sampling=sampling,
        segment_s=segment_s,
        edges=edges,
        times=times,
        window_s=window_s,
        label="pd_no_treatment",
        verbose=verbose,
    )
    _log(f"  seed {seed}: condition 2/2 — PD 130 Hz cDBS", verbose=verbose)
    p_130 = simulate_trace(
        plant,
        seed=seed,
        stim_hz_after_onset=130.0,
        sampling=sampling,
        segment_s=segment_s,
        edges=edges,
        times=times,
        window_s=window_s,
        label="pd_130hz",
        verbose=verbose,
    )
    if sampling == "segment":
        _log(
            f"  seed {seed}: done "
            f"(seg1 [0,2)s P_beta={p_none[0]:.1f}; seg2 [2,4)s "
            f"no_tx={p_none[1]:.1f} hz130={p_130[1]:.1f})",
            verbose=verbose,
        )
    else:
        _log(
            f"  seed {seed}: done "
            f"(t=0 P_beta={p_none[0]:.1f}; t={DBS_ONSET_S:.0f}s "
            f"no_tx={p_none[int(DBS_ONSET_S / (times[1] - times[0]))]:.1f} "
            f"hz130={p_130[int(DBS_ONSET_S / (times[1] - times[0]))]:.1f}; "
            f"t={DISPLAY_S:.0f}s no_tx={p_none[-1]:.1f} hz130={p_130[-1]:.1f})",
            verbose=verbose,
        )
    row: dict[str, Any] = {
        "seed": seed,
        "time_s": times.tolist(),
        "pd_no_treatment": p_none.tolist(),
        "pd_130hz": p_130.tolist(),
    }
    if sampling == "segment":
        display_edges = edges - WARMUP_S
        plot_x, _ = step_plot_xy(display_edges, p_none)
        row["segment_edges"] = display_edges.tolist()
        row["segment_edges_sim"] = edges.tolist()
        row["segment_values"] = {
            "pd_no_treatment": p_none.tolist(),
            "pd_130hz": p_130.tolist(),
        }
        row["time_s"] = plot_x.tolist()
    return row


def mean_trace(seed_rows: list[dict[str, Any]], key: str) -> list[float]:
    stack = np.asarray([row[key] for row in seed_rows], dtype=float)
    return stack.mean(axis=0).tolist()


def save_series(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_series(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _plot_xy_from_cache(cache: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray, str]:
    sampling = cache.get("sampling", "segment")
    values = np.asarray(cache["traces"][key], dtype=float)
    if sampling == "trailing":
        times = np.asarray(cache["time_s"], dtype=float)
        if times.size != values.size:
            msg = "trailing cache time_s and traces length mismatch"
            raise ValueError(msg)
        return times, values, "line"
    edges = np.asarray(cache["segment_edges"], dtype=float)
    if values.size > edges.size - 1:
        msg = "segment cache trace length does not match segment_edges"
        raise ValueError(msg)
    x, y = step_plot_xy(edges, values)
    return x, y, "steps-post"


def plot_fig2a(
    cache: dict[str, Any],
    *,
    out_path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

    sampling = cache.get("sampling", "segment")
    plot_style = "line" if sampling == "trailing" else "steps-post"
    peak = 0.0
    for key in ("pd_no_treatment", "pd_130hz"):
        meta = SERIES[key]
        x, y, drawstyle = _plot_xy_from_cache(cache, key)
        peak = max(peak, float(np.max(y)))
        plot_kwargs: dict[str, Any] = {
            "color": meta["color"],
            "linewidth": 1.2,
            "label": meta["label"],
        }
        if drawstyle == "steps-post":
            plot_kwargs["drawstyle"] = "steps-post"
        ax.plot(x, y, **plot_kwargs)

    paper_y = _paper_overlay.overlay_mehregan_fig2(ax)
    for _, (py, _) in paper_y.items():
        peak = max(peak, float(np.max(py)))

    ax.axvline(
        DBS_ONSET_S,
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        zorder=0,
    )

    ymin = y_min if y_min is not None else 100.0
    ymax = y_max if y_max is not None else float(np.ceil(max(peak * 1.05, 600.0) / 50.0) * 50.0)

    ax.set_xlim(0.0, DISPLAY_S)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("PSD")
    ax.legend(loc="upper center", fontsize=9, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    n_points = len(cache["traces"]["pd_no_treatment"])
    return {
        "out": str(out_path),
        "y_min": ymin,
        "y_max": ymax,
        "dbs_onset_s": DBS_ONSET_S,
        "n_points": n_points,
        "plot_style": plot_style,
        "p_beta_peak": peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds (default: 0; mean when multiple)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--series",
        type=Path,
        default=DEFAULT_SERIES,
        help="Cached trace JSON (written on simulate; read with --plot-only)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot from --series cache (seconds; no plant simulation)",
    )
    parser.add_argument(
        "--sampling",
        choices=("trailing", "segment"),
        default="trailing",
        help="trailing = dense 0.2 s samples (default); segment = 6x2 s step plot",
    )
    parser.add_argument(
        "--step-s",
        type=float,
        default=STEP_S,
        help="Display-axis sample interval for trailing mode (default: 0.2 s)",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=WINDOW_S,
        help="Trailing P_beta window length (default: 2 s)",
    )
    parser.add_argument(
        "--segment-s",
        type=float,
        default=SEGMENT_S,
        help="RL step length for segment mode (default: 2 s)",
    )
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress incremental progress logs (stderr)",
    )
    parser.add_argument(
        "--no-update-docs",
        action="store_true",
        help="Skip figures/mehregan/replications.md caption refresh",
    )
    args = parser.parse_args()

    sampling: SamplingMode = args.sampling
    verbose = not args.quiet
    if sampling == "trailing":
        times = sample_times(args.step_s)
        edges = display_segment_edges_sim(args.segment_s)
        n_segments = None
    else:
        times = sample_times(args.segment_s, duration_s=DISPLAY_S)
        edges = display_segment_edges_sim(args.segment_s)
        n_segments = edges.size - 1

    if args.plot_only:
        if not args.series.exists():
            print(f"missing series cache: {args.series}", file=sys.stderr)
            print("Run without --plot-only once to build the cache.", file=sys.stderr)
            return 2
        cache = load_series(args.series)
        seeds = tuple(cache.get("seeds", args.seeds))
        print(f"loaded traces from {args.series}", file=sys.stderr, flush=True)
    else:
        seeds = args.seeds
        per_seed: list[dict[str, Any]] = []
        with PythonPlant() as plant:
            for seed in seeds:
                if sampling == "trailing":
                    _log(
                        f"simulating seed {seed} "
                        f"({INTEGRATE_S:.0f}s sim = {WARMUP_S:.0f}s pre-roll + {DISPLAY_S:.0f}s display, "
                        f"{times.size} trailing samples every {args.step_s:.1f}s, "
                        f"{args.window_s:.1f}s window)...",
                        verbose=verbose,
                    )
                else:
                    _log(
                        f"simulating seed {seed} "
                        f"({DISPLAY_S:.0f} s, {n_segments} segments x {args.segment_s:.1f} s)...",
                        verbose=verbose,
                    )
                per_seed.append(
                    simulate_seed(
                        plant,
                        seed=seed,
                        sampling=sampling,
                        segment_s=args.segment_s,
                        edges=edges,
                        times=times,
                        window_s=args.window_s,
                        verbose=verbose,
                    )
                )

        traces = {
            "pd_no_treatment": mean_trace(per_seed, "pd_no_treatment"),
            "pd_130hz": mean_trace(per_seed, "pd_130hz"),
        }
        cache = {
            "figure": "mehregan_fig2a",
            "sampling": sampling,
            "duration_s": DISPLAY_S,
            "dbs_onset_s": DBS_ONSET_S,
            "warmup_s": WARMUP_S,
            "display_offset_s": WARMUP_S,
            "dbs_onset_sim_s": DBS_ONSET_SIM,
            "biomarker_sim_end_max_s": BIOMARKER_SIM_END_MAX,
            "integrate_s": INTEGRATE_S,
            "seeds": list(seeds),
            "time_s": times.tolist(),
            "traces": traces,
            "per_seed": per_seed,
        }
        if sampling == "trailing":
            cache["step_s"] = args.step_s
            cache["window_s"] = args.window_s
            cache["n_samples"] = int(times.size)
        else:
            display_edges = segment_edges(DISPLAY_S, args.segment_s)
            plot_x, _ = step_plot_xy(display_edges, np.asarray(traces["pd_no_treatment"]))
            cache["segment_s"] = args.segment_s
            cache["n_segments"] = n_segments
            cache["segment_edges"] = display_edges.tolist()
            cache["segment_edges_sim"] = edges.tolist()
            cache["time_s"] = plot_x.tolist()
        save_series(cache, args.series)
        _log(f"wrote series cache {args.series}", verbose=verbose)

    _log(f"plotting {args.out}...", verbose=verbose)
    panel = plot_fig2a(
        cache,
        out_path=args.out,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    times = np.asarray(cache["time_s"], dtype=float)
    traces = cache["traces"]
    paper_gates = fig2_time_gates(
        {
            "pd": (times, np.asarray(traces["pd_no_treatment"], dtype=float)),
            "pd_130hz": (times, np.asarray(traces["pd_130hz"], dtype=float)),
        },
        panel="2a",
    )
    panel["gates"] = paper_gates["gates"]
    panel["gates_pass"] = paper_gates["pass"]
    panel["paper_gate_metrics"] = paper_gates["metrics"]

    manifest = {
        "figure": "mehregan_fig2a",
        "sampling": cache.get("sampling", sampling),
        "duration_s": cache.get("duration_s", DISPLAY_S),
        "dbs_onset_s": cache.get("dbs_onset_s", DBS_ONSET_S),
        "warmup_s": cache.get("warmup_s", WARMUP_S),
        "display_offset_s": cache.get("display_offset_s", WARMUP_S),
        "dbs_onset_sim_s": cache.get("dbs_onset_sim_s", DBS_ONSET_SIM),
        "biomarker_sim_end_max_s": cache.get("biomarker_sim_end_max_s", BIOMARKER_SIM_END_MAX),
        "integrate_s": cache.get("integrate_s", INTEGRATE_S),
        "step_s": cache.get("step_s", args.step_s if sampling == "trailing" else None),
        "window_s": cache.get("window_s", args.window_s if sampling == "trailing" else None),
        "segment_s": cache.get("segment_s", args.segment_s if sampling == "segment" else None),
        "n_samples": cache.get("n_samples"),
        "n_segments": cache.get("n_segments", n_segments),
        "segment_edges": cache.get("segment_edges"),
        "seeds": list(cache.get("seeds", args.seeds)),
        "series_cache": str(args.series),
        "plot_only": args.plot_only,
        "panel": panel,
        "gates": paper_gates["gates"],
        "gates_pass": paper_gates["pass"],
        "paper_ref": paper_gates["paper_ref"],
        "time_s": cache.get("time_s"),
        "traces": cache.get("traces"),
        "per_seed": cache.get("per_seed"),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    _log(f"wrote manifest {args.manifest}", verbose=verbose)
    _log(f"gates_pass={paper_gates['pass']} gates={paper_gates['gates']}", verbose=True)

    if not args.no_update_docs:
        updated = _figure_promote.promote_2a(
            manifest=manifest,
            series_path=args.series,
            png_path=args.out,
        )
        _log(f"updated comparison doc: {updated['doc']}", verbose=True)

    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if paper_gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
