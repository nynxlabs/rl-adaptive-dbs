#!/usr/bin/env python3
"""Mehregan et al. (paper 1) Figure 2b — cDBS effect on Error Index over time.

Replicates the paper panel showing windowed Error Index (EI, Eq. 2) during a
12 s parkinsonian simulation with So-style SMC pulses into TH (path A):

  - 0–2 s: no STN DBS (shared baseline for both traces)
  - 2–12 s: either no treatment (red) or 130 Hz conventional cDBS (blue)

Default plant knobs (2026-07-12 path A): ``smc_site=thalamic``,
``iappth_baseline=0`` (So et al. 2012 pulses-only TH drive; not Kumaravelu's
constant Iappth), ``ggith=0.112``, BoC inverse-gamma SMC (3.5 µA/cm², 5 ms).
EI metric: Mehregan Eq. 2 / Gao et al. ICCPS 2020. Same trailing-window protocol
as Fig 2a (14 s sim, 2 s pre-roll, 0.2 s samples, 2 s EI window).

Run:
  uv run --group figures python scripts/figures/papers/mehregan/2b/plot.py
  uv run --group figures python scripts/figures/papers/mehregan/2b/plot.py --plot-only

Each run writes ``figures/mehregan/images/2b/error_index_vN.png`` (N auto-increments)
and updates the replication image link in ``figures/mehregan/replications.md``.

Plant EI simulation only — no RL training; checkpoint resume is not applicable.
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

from envs.plant import DbsSpec, MatlabPlant, PlantConfig, PythonPlant
from envs.plant.biomarkers import DEFAULT_EI_WINDOW_S, error_index
from envs.plant.config import (
    BOC_SMC_AMPLITUDE,
    BOC_SMC_CORTICAL_AMPLITUDE,
    KUMARAVELU_GGITH,
)
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

FIGURES_DIR = Path("figures/mehregan/images/2b")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/2b")
DEFAULT_SERIES = CACHE_DIR / "series.json"
OUT_STEM = "error_index"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
DISPLAY_S = 12.0
DBS_ONSET_S = 2.0
WARMUP_S = 2.0
WINDOW_S = DEFAULT_EI_WINDOW_S
STEP_S = 0.2
SEGMENT_S = 2.0
INTEGRATE_S = WARMUP_S + DISPLAY_S
DBS_ONSET_SIM = WARMUP_S + DBS_ONSET_S
BIOMARKER_SIM_END_MAX = WARMUP_S + DISPLAY_S
FIG2_SPIKE_HEADROOM_HZ = 60.0
FIG2_SPIKE_BUFFER_MARGIN = 64
DEFAULT_SMC_HZ = 10.0
DEFAULT_SMC_SCHEDULE = "boc"
# Path A (So-style): SMC pulses into TH, no cerebellar bias (iappth_baseline=0).
DEFAULT_SMC_SITE = "thalamic"
DEFAULT_IAPPTH_BASELINE = 0.0
DEFAULT_GGITH = KUMARAVELU_GGITH
DEFAULT_SMC_AMPLITUDE = BOC_SMC_AMPLITUDE
DEFAULT_BACKEND = "python"
DEFAULT_SEEDS: tuple[int, ...] = (0,)
# Paper Fig 2b frame is 0–0.4; used when auto limits still fit all series.
PAPER_Y_MIN = 0.0
PAPER_Y_MAX = 0.4

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


def fig2_spike_buffer_size(*, integrate_s: float = INTEGRATE_S) -> int:
    return max(
        512,
        int(np.ceil(integrate_s * FIG2_SPIKE_HEADROOM_HZ)) + FIG2_SPIKE_BUFFER_MARGIN,
    )


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
    end = min(warmup_s + display_t, sim_window_end_cap)
    start = end - window_s
    return max(0.0, start), end


def trailing_ei(
    th_spikes: list[np.ndarray],
    smc_pulse_times_s: np.ndarray,
    *,
    times: np.ndarray,
    window_s: float = WINDOW_S,
    n_neurons: int,
    label: str = "trace",
    verbose: bool = True,
    progress_every: int = 10,
) -> np.ndarray:
    n_samples = times.size
    values = np.zeros(n_samples, dtype=float)
    step_s = float(times[1] - times[0]) if n_samples > 1 else STEP_S
    t0_all = time.monotonic()
    _log(
        f"    [{label}] trailing EI: {n_samples} samples, "
        f"{window_s:.1f}s window every {step_s:.1f}s",
        verbose=verbose,
    )
    for idx, display_t in enumerate(times):
        t_start, t_end = trailing_window_sim(float(display_t), window_s=window_s)
        inclusive_end = idx == n_samples - 1
        values[idx] = error_index(
            th_spikes,
            smc_pulse_times_s,
            t_start=t_start,
            t_end=t_end,
            inclusive_pulse_end=inclusive_end,
            n_neurons=n_neurons,
        )
        if verbose and (idx == 0 or idx == n_samples - 1 or (idx + 1) % progress_every == 0):
            _log(
                f"    [{label}] sample {idx + 1}/{n_samples} "
                f"t={display_t:.1f}s sim[{t_start:.1f},{t_end:.1f}] "
                f"EI={values[idx]:.3f}",
                verbose=True,
            )
    if verbose:
        _log(
            f"    [{label}] trailing done in {time.monotonic() - t0_all:.1f}s "
            f"(range {float(values.min()):.3f}..{float(values.max()):.3f})",
            verbose=True,
        )
    return values


def make_plant(backend: str) -> PythonPlant | MatlabPlant:
    if backend == "matlab":
        return MatlabPlant()
    if backend == "python":
        return PythonPlant()
    msg = f"unsupported plant backend: {backend!r} (use python or matlab)"
    raise ValueError(msg)


def simulate_trace(
    plant: PythonPlant | MatlabPlant,
    *,
    seed: int,
    smc_schedule: str,
    smc_site: str,
    smc_pulse_source: str,
    smc_cortical_amplitude: float,
    smc_amplitude: float,
    iappth_baseline: float,
    ggith: float,
    smc_hz: float,
    stim_hz_after_onset: float,
    times: np.ndarray,
    window_s: float,
    label: str,
    verbose: bool = True,
) -> np.ndarray:
    plant.config = PlantConfig(
        pd=1,
        corstim=0,
        smc_schedule=smc_schedule,  # type: ignore[arg-type]
        smc_site=smc_site,  # type: ignore[arg-type]
        smc_pulse_source=smc_pulse_source,  # type: ignore[arg-type]
        smc_cortical_amplitude=smc_cortical_amplitude,
        smc_amplitude=smc_amplitude,
        smc_frequency_hz=smc_hz if smc_schedule == "periodic" else 0.0,
        iappth_baseline=iappth_baseline,
        ggith=ggith,
    )
    plant.reset(seed=seed)
    dt_ms = plant.config.dt_ms
    if stim_hz_after_onset > 0.0:
        spec = dbs_spec_with_onset(
            onset_s=DBS_ONSET_SIM,
            frequency_hz=stim_hz_after_onset,
            duration_s=INTEGRATE_S,
            dt_ms=dt_ms,
        )
        stim_note = f"130 Hz from display t={DBS_ONSET_S:.0f}s (sim {DBS_ONSET_SIM:.0f}s)"
    else:
        spec = DbsSpec.none()
        stim_note = "no DBS"
    smc_note = (
        f"SMC {smc_hz:g} Hz periodic Iappco"
        if smc_schedule == "periodic" and use_cortical
        else f"SMC {smc_hz:g} Hz periodic Iappth"
        if smc_schedule == "periodic"
        else f"SMC {smc_schedule} BoC inv-gamma {smc_site} ({smc_pulse_source})"
    )
    _log(
        f"    [{label}] integrate {INTEGRATE_S:.0f}s "
        f"({smc_note}, seed {seed}, {stim_note})...",
        verbose=verbose,
    )
    t_integrate0 = time.monotonic()
    buf = fig2_spike_buffer_size(integrate_s=INTEGRATE_S)
    result = plant.integrate(
        INTEGRATE_S,
        spec,
        record_spikes=False,
        record_th_spikes=True,
        th_spike_buffer_size=buf,
        cor_spike_buffer_size=buf if smc_pulse_source == "cor_spikes" else None,
    )
    integrate_elapsed = time.monotonic() - t_integrate0
    th_spikes = result.info.get("th_spikes")
    smc_times = np.asarray(result.info.get("smc_pulse_times_s", []), dtype=float)
    if not th_spikes:
        msg = "plant integrate did not record thalamic spikes"
        raise RuntimeError(msg)
    spike_count = sum(int(np.asarray(s).size) for s in th_spikes)
    _log(
        f"    [{label}] integrate done in {integrate_elapsed:.1f}s "
        f"({spike_count:,} TH spikes, {smc_times.size} SMC pulses)",
        verbose=verbose,
    )
    return trailing_ei(
        th_spikes,
        smc_times,
        times=times,
        window_s=window_s,
        n_neurons=plant.config.neurons_per_region,
        label=label,
        verbose=verbose,
    )


def simulate_seed(
    plant: PythonPlant | MatlabPlant,
    *,
    seed: int,
    smc_schedule: str,
    smc_site: str,
    smc_pulse_source: str,
    smc_cortical_amplitude: float,
    smc_amplitude: float,
    iappth_baseline: float,
    ggith: float,
    smc_hz: float,
    times: np.ndarray,
    window_s: float,
    verbose: bool = True,
) -> dict[str, Any]:
    _log(f"  seed {seed}: condition 1/2 — PD no treatment", verbose=verbose)
    ei_none = simulate_trace(
        plant,
        seed=seed,
        smc_schedule=smc_schedule,
        smc_site=smc_site,
        smc_pulse_source=smc_pulse_source,
        smc_cortical_amplitude=smc_cortical_amplitude,
        smc_amplitude=smc_amplitude,
        iappth_baseline=iappth_baseline,
        ggith=ggith,
        smc_hz=smc_hz,
        stim_hz_after_onset=0.0,
        times=times,
        window_s=window_s,
        label="pd_no_treatment",
        verbose=verbose,
    )
    _log(f"  seed {seed}: condition 2/2 — PD 130 Hz cDBS", verbose=verbose)
    ei_130 = simulate_trace(
        plant,
        seed=seed,
        smc_schedule=smc_schedule,
        smc_site=smc_site,
        smc_pulse_source=smc_pulse_source,
        smc_cortical_amplitude=smc_cortical_amplitude,
        smc_amplitude=smc_amplitude,
        iappth_baseline=iappth_baseline,
        ggith=ggith,
        smc_hz=smc_hz,
        stim_hz_after_onset=130.0,
        times=times,
        window_s=window_s,
        label="pd_130hz",
        verbose=verbose,
    )
    _log(
        f"  seed {seed}: done "
        f"(t=0 EI no_tx={ei_none[0]:.3f} hz130={ei_130[0]:.3f}; "
        f"t={DISPLAY_S:.0f}s no_tx={ei_none[-1]:.3f} hz130={ei_130[-1]:.3f})",
        verbose=verbose,
    )
    return {
        "seed": seed,
        "time_s": times.tolist(),
        "pd_no_treatment": ei_none.tolist(),
        "pd_130hz": ei_130.tolist(),
    }


def mean_trace(seed_rows: list[dict[str, Any]], key: str) -> list[float]:
    stack = np.asarray([row[key] for row in seed_rows], dtype=float)
    return stack.mean(axis=0).tolist()


def save_series(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_series(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _auto_ylim(*series: np.ndarray) -> tuple[float, float]:
    """Y limits covering every replication + paper series (pad; prefer paper 0–0.4)."""
    chunks = [np.asarray(s, dtype=float).ravel() for s in series if s is not None and np.size(s)]
    if not chunks:
        return PAPER_Y_MIN, PAPER_Y_MAX
    vals = np.concatenate(chunks)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return PAPER_Y_MIN, PAPER_Y_MAX
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(hi - lo, 1e-6)
    pad = max(0.02, 0.05 * span)
    ymin = max(0.0, lo - pad)
    ymax = hi + pad
    # Keep the paper frame when it still shows everything.
    if ymin >= PAPER_Y_MIN and ymax <= PAPER_Y_MAX:
        return PAPER_Y_MIN, PAPER_Y_MAX
    if ymin <= 0.05:
        ymin = PAPER_Y_MIN
    return ymin, ymax


def plot_fig2b(
    cache: dict[str, Any],
    *,
    out_path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

    repl_ys: list[np.ndarray] = []
    for key in ("pd_no_treatment", "pd_130hz"):
        meta = SERIES[key]
        x = np.asarray(cache["time_s"], dtype=float)
        y = np.asarray(cache["traces"][key], dtype=float)
        repl_ys.append(y)
        ax.plot(x, y, color=meta["color"], linewidth=1.2, label=meta["label"])

    paper_y = _paper_overlay.overlay_mehregan_fig2b(ax)
    paper_ys = [py for _, (py, _) in paper_y.items()]

    ax.axvline(DBS_ONSET_S, color="#888888", linestyle="--", linewidth=1.2, zorder=0)

    auto_ymin, auto_ymax = _auto_ylim(*repl_ys, *paper_ys)
    ymin = y_min if y_min is not None else auto_ymin
    ymax = y_max if y_max is not None else auto_ymax
    peak = float(max((float(np.max(y)) for y in (*repl_ys, *paper_ys) if y.size), default=0.0))

    ax.set_xlim(0.0, DISPLAY_S)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Error Index")
    _paper_overlay.place_legend(ax, loc="upper center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "out": str(out_path),
        "y_min": ymin,
        "y_max": ymax,
        "dbs_onset_s": DBS_ONSET_S,
        "n_points": len(cache["traces"]["pd_no_treatment"]),
        "plot_style": "line",
        "ei_peak": peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds (default: 0; mean when multiple)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: auto-increment "
            f"{FIGURES_DIR.as_posix()}/{OUT_STEM}_vN.png"
        ),
    )
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--step-s", type=float, default=STEP_S)
    parser.add_argument("--window-s", type=float, default=WINDOW_S)
    parser.add_argument("--smc-hz", type=float, default=DEFAULT_SMC_HZ)
    parser.add_argument(
        "--smc-schedule",
        choices=("off", "boc", "periodic"),
        default=DEFAULT_SMC_SCHEDULE,
        help="SMC probe schedule (default: boc = BoC inverse-gamma)",
    )
    parser.add_argument(
        "--smc-site",
        choices=("thalamic", "cortical"),
        default=DEFAULT_SMC_SITE,
        help="SMC injection site (default: thalamic So-style path A)",
    )
    parser.add_argument(
        "--iappth-baseline",
        type=float,
        default=DEFAULT_IAPPTH_BASELINE,
        help="Constant TH bias before SMC pulses (So-style Fig 2b: 0; Kumaravelu: 1.2)",
    )
    parser.add_argument(
        "--ggith",
        type=float,
        default=DEFAULT_GGITH,
        help=f"GPi→TH conductance (default {DEFAULT_GGITH})",
    )
    parser.add_argument(
        "--smc-amplitude",
        type=float,
        default=DEFAULT_SMC_AMPLITUDE,
        help=f"BoC pulse amplitude on Iappth when --smc-site=thalamic (default {DEFAULT_SMC_AMPLITUDE})",
    )
    parser.add_argument(
        "--smc-cortical-amplitude",
        type=float,
        default=BOC_SMC_CORTICAL_AMPLITUDE,
        help=f"BoC pulse amplitude on Iappco when --smc-site=cortical (default {BOC_SMC_CORTICAL_AMPLITUDE})",
    )
    parser.add_argument(
        "--smc-pulse-source",
        choices=("drive", "cor_spikes"),
        default="drive",
        help="SMCτ for EI: scheduled drive rising edges or Cor spike alignment (Gao)",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "matlab"),
        default=DEFAULT_BACKEND,
        help="Plant backend (default: python; matlab uses Kumaravelu reference)",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Y-axis lower limit (default: auto from replication + paper digitization)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Y-axis upper limit (default: auto from replication + paper digitization)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-update-docs", action="store_true")
    args = parser.parse_args()

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _vault_backed_png(args.out)

    verbose = not args.quiet
    times = sample_times(args.step_s)

    if args.plot_only:
        if not args.series.exists():
            print(f"missing series cache: {args.series}", file=sys.stderr)
            return 2
        cache = load_series(args.series)
        print(f"loaded traces from {args.series}", file=sys.stderr, flush=True)
    else:
        seeds = args.seeds
        per_seed: list[dict[str, Any]] = []
        with make_plant(args.backend) as plant:
            for seed in seeds:
                _log(
                    f"simulating seed {seed} "
                    f"({INTEGRATE_S:.0f}s sim, SMC {args.smc_schedule} {args.smc_site} "
                    f"base={args.iappth_baseline:g} ggith={args.ggith:g}, "
                    f"backend {args.backend}, {times.size} trailing samples)...",
                    verbose=verbose,
                )
                per_seed.append(
                    simulate_seed(
                        plant,
                        seed=seed,
                        smc_schedule=args.smc_schedule,
                        smc_site=args.smc_site,
                        smc_pulse_source=args.smc_pulse_source,
                        smc_cortical_amplitude=args.smc_cortical_amplitude,
                        smc_amplitude=args.smc_amplitude,
                        iappth_baseline=args.iappth_baseline,
                        ggith=args.ggith,
                        smc_hz=args.smc_hz,
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
            "figure": "mehregan_fig2b",
            "sampling": "trailing",
            "duration_s": DISPLAY_S,
            "dbs_onset_s": DBS_ONSET_S,
            "warmup_s": WARMUP_S,
            "display_offset_s": WARMUP_S,
            "dbs_onset_sim_s": DBS_ONSET_SIM,
            "biomarker_sim_end_max_s": BIOMARKER_SIM_END_MAX,
            "integrate_s": INTEGRATE_S,
            "smc_frequency_hz": args.smc_hz,
            "smc_schedule": args.smc_schedule,
            "smc_site": args.smc_site,
            "smc_pulse_source": args.smc_pulse_source,
            "smc_cortical_amplitude": args.smc_cortical_amplitude,
            "smc_amplitude": args.smc_amplitude,
            "iappth_baseline": args.iappth_baseline,
            "ggith": args.ggith,
            "plant_backend": args.backend,
            "corstim": 0,
            "path_a_so_style": args.smc_site == "thalamic" and args.iappth_baseline == 0.0,
            "seeds": list(seeds),
            "time_s": times.tolist(),
            "traces": traces,
            "per_seed": per_seed,
            "step_s": args.step_s,
            "window_s": args.window_s,
            "n_samples": int(times.size),
        }
        save_series(cache, args.series)
        _log(f"wrote series cache {args.series}", verbose=verbose)

    panel = plot_fig2b(
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
        panel="2b",
    )
    panel["gates"] = paper_gates["gates"]
    panel["gates_pass"] = paper_gates["pass"]
    panel["paper_gate_metrics"] = paper_gates["metrics"]

    manifest = {
        "figure": "mehregan_fig2b",
        "sampling": cache.get("sampling", "trailing"),
        "duration_s": cache.get("duration_s", DISPLAY_S),
        "dbs_onset_s": cache.get("dbs_onset_s", DBS_ONSET_S),
        "warmup_s": cache.get("warmup_s", WARMUP_S),
        "smc_frequency_hz": cache.get("smc_frequency_hz", args.smc_hz),
        "smc_schedule": cache.get("smc_schedule", args.smc_schedule),
        "smc_site": cache.get("smc_site", args.smc_site),
        "smc_pulse_source": cache.get("smc_pulse_source", args.smc_pulse_source),
        "smc_cortical_amplitude": cache.get(
            "smc_cortical_amplitude", args.smc_cortical_amplitude
        ),
        "smc_amplitude": cache.get("smc_amplitude", args.smc_amplitude),
        "iappth_baseline": cache.get("iappth_baseline", args.iappth_baseline),
        "ggith": cache.get("ggith", args.ggith),
        "path_a_so_style": cache.get("path_a_so_style"),
        "plant_backend": cache.get("plant_backend", args.backend),
        "corstim": cache.get("corstim", 0),
        "integrate_s": cache.get("integrate_s", INTEGRATE_S),
        "step_s": cache.get("step_s", args.step_s),
        "window_s": cache.get("window_s", args.window_s),
        "n_samples": cache.get("n_samples"),
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
        "png_version": png_version,
        "output_png": _figure_promote.repo_rel_posix(args.out),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    _log(f"gates_pass={paper_gates['pass']} gates={paper_gates['gates']}", verbose=True)

    if not args.no_update_docs:
        updated = _figure_promote.promote_2b(
            manifest=manifest,
            series_path=args.series,
            png_path=args.out,
        )
        _log(f"updated comparison doc: {updated['doc']}", verbose=True)

    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if paper_gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
