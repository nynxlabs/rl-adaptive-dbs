#!/usr/bin/env python3
"""Mehregan et al. (paper 1) Figure 5b — post-train efficacy @ 30 Hz.

Paper-protocol eval: **12 s** display with **2 s** shared baseline, then stimulation.
Default plot uses the same **0.2 s trailing / 2 s window** biomarker protocol as Fig 2a
(14 s integrate, 2 s pre-roll). ``--sampling segment`` keeps the legacy 2 s RL step plot.

Three series on the **raw PSD** scale:

  1. PD no stim (black) — real rollout, not a flat fake baseline
  2. Fully trained 30 Hz pattern policy (green)
  3. Periodic 30 Hz / pattern 0 (orange)

**Paired workflow (default):** eval + plot from 30 Hz checkpoint
(``artifacts/figures/papers/1/5b/checkpoint.pt``). **BurstPatternAlphabet**
(41 patterns; pattern 0 = regular 30 Hz; irregulars = 60–120 Hz clusters at
fixed mean rate — Fig 5b redesign). Train with ``scripts/retrain_30hz_fig5b.py``.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/1/5b/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/1/5b/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/1/5b/plot.py --sampling segment

Each run writes ``figures/papers/1/5b/efficacy_30hz_vN.png`` (N auto-increments) and updates
``docs/figures/paper_1.md``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
from envs.plant import DbsSpec, PlantConfig, PythonPlant
from envs.plant.dbs import create_dbs_current

from scripts.run_task108_paper_protocol_eval import run_eval

_FIG2A_PATH = Path(__file__).resolve().parents[1] / "2a" / "plot.py"
_fig2a_spec = importlib.util.spec_from_file_location("fig2a_plot", _FIG2A_PATH)
assert _fig2a_spec and _fig2a_spec.loader
_fig2a = importlib.util.module_from_spec(_fig2a_spec)
_fig2a_spec.loader.exec_module(_fig2a)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

FIGURES_DIR = Path("figures/papers/1/5b")
CACHE_DIR = Path("artifacts/figures/papers/1/5b")
DEFAULT_EVAL = CACHE_DIR / "eval.json"
DEFAULT_CHECKPOINT = CACHE_DIR / "checkpoint.pt"
DEFAULT_LANDSCAPE = Path("artifacts/ddpg/pattern_reward_landscape_30hz.json")
OUT_STEM = "efficacy_30hz"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"

MEAN_HZ = 30.0
PAPER_DT_MS = 0.02
DEFAULT_SEED = 0
EVAL_STEPS = 5

SEGMENT_S = 2.0
N_SEGMENTS = 6
TIME_MAX_S = 12.0
STIM_ONSET_S = 2.0
TRAILING_RL_STEP_S = 0.2
TRAILING_STIM_STEPS = int((TIME_MAX_S - STIM_ONSET_S) / TRAILING_RL_STEP_S)
# Y-limits default to auto-fit from plotted traces; override with --y-min / --y-max.
DEFAULT_Y_MIN: float | None = None
DEFAULT_Y_MAX: float | None = None

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
    "no_stim": {"label": "PD no stim", "color": "#111111", "linestyle": "-", "linewidth": 1.5, "zorder": 1},
    "trained": {
        "label": "Fully Trained 30Hz",
        "color": "#2ca02c",
        "linestyle": "--",
        "linewidth": 2.2,
        "zorder": 4,
    },
    "periodic": {"label": "Periodic 30Hz", "color": "#ff7f0e", "linestyle": "-", "linewidth": 1.5, "zorder": 2},
}


def _vault_backed_png(path: Path) -> Path:
    """Ensure replication PNGs land in the vault via ``paper.png``'s target dir.

    Worktrees often materialize ``paper.png`` as a real file (not a symlink). In that
    case, fall back to the main checkout's ``paper.png`` symlink so we still write
    the vault path and leave a worktree symlink for local viewing.
    """
    path = Path(path)
    paper = path.parent / "paper.png"
    if not paper.is_symlink():
        main = _figure_promote.main_checkout_root(_REPO_ROOT)
        main_paper = main / "figures" / "papers" / "1" / "5b" / "paper.png"
        if main_paper.is_symlink():
            paper = main_paper
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
    vault_dir = paper.resolve().parent
    vault_target = vault_dir / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() and path.resolve() == vault_target.resolve():
        return path
    if path.exists() or path.is_symlink():
        # Real file in a worktree from an earlier run — replace with vault symlink.
        if path.is_file() and not path.is_symlink():
            if not vault_target.exists():
                import shutil

                shutil.copy2(path, vault_target)
            path.unlink()
            path.symlink_to(vault_target)
            return path
        return path
    if not vault_target.exists():
        vault_target.touch()
    path.symlink_to(vault_target)
    # Also expose on main checkout figures/ when plotting from a worktree.
    main = _figure_promote.main_checkout_root(_REPO_ROOT)
    main_link = main / "figures" / "papers" / "1" / "5b" / path.name
    if main_link.parent.is_dir() and not main_link.exists():
        try:
            main_link.symlink_to(vault_target)
        except OSError:
            pass
    return path


def _segment_times() -> np.ndarray:
    return np.arange(0.0, TIME_MAX_S, SEGMENT_S)


def _step_series(p_beta: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(p_beta) != N_SEGMENTS:
        msg = f"expected {N_SEGMENTS} P_beta samples, got {len(p_beta)}"
        raise ValueError(msg)
    x = np.concatenate([_segment_times(), [TIME_MAX_S]])
    y = np.concatenate([p_beta, [p_beta[-1]]])
    return x, y


def _policy_p_beta(policies: dict[str, Any], key: str) -> list[float]:
    data = policies[key]
    if "error" in data:
        msg = f"policy {key!r} has error: {data['error']}"
        raise KeyError(msg)
    return [float(v) for v in data["p_beta"]]


def _find_trained_key(policies: dict[str, Any]) -> str | None:
    trained = [
        k
        for k, v in policies.items()
        if k.startswith("trained_ddpg_") and isinstance(v, dict) and "p_beta" in v
    ]
    return trained[0] if trained else None


def _post_onset_mean(trace: list[float]) -> float:
    if len(trace) < 2:
        return float("nan")
    return float(np.mean(trace[1:]))


def _post_onset_mean_trailing(times: list[float] | np.ndarray, trace: list[float] | np.ndarray) -> float:
    t = np.asarray(times, dtype=float)
    y = np.asarray(trace, dtype=float)
    mask = t >= STIM_ONSET_S - 1e-9
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y[mask]))


def _baseline_at_onset(times: list[float] | np.ndarray, trace: list[float] | np.ndarray) -> float:
    t = np.asarray(times, dtype=float)
    y = np.asarray(trace, dtype=float)
    idx = int(np.argmin(np.abs(t - 0.0)))
    return float(y[idx])


def _integrate_idbs(
    *,
    duration_s: float,
    dt_ms: float,
    onset_sim_s: float,
    segment_actions: list[int] | None,
    alphabet: BurstPatternAlphabet | None,
    scalar_hz: float | None = None,
    rl_step_s: float = SEGMENT_S,
) -> np.ndarray:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    onset_idx = int(round(onset_sim_s * 1000.0 / dt_ms))
    if scalar_hz is not None and scalar_hz > 0.0:
        stim = create_dbs_current(
            scalar_hz,
            tmax_ms=(duration_s - onset_sim_s) * 1000.0,
            dt_ms=dt_ms,
        )
        end = min(onset_idx + stim.size, n_steps)
        idbs[onset_idx:end] = stim[: end - onset_idx]
        return idbs
    if not segment_actions or alphabet is None:
        return idbs
    step_samples = int(round(rl_step_s * 1000.0 / dt_ms))
    for seg_i, action in enumerate(segment_actions):
        seg = np.asarray(alphabet.idbs_for_action(action), dtype=np.float64)
        start = onset_idx + seg_i * step_samples
        end = min(start + seg.size, n_steps)
        if start >= n_steps:
            break
        idbs[start:end] = seg[: end - start]
    return idbs


def _trained_actions_fine(
    checkpoint: Path,
    *,
    seed: int,
) -> list[int]:
    import torch
    from dataclasses import replace

    from controllers.ddpg import load_actor
    from envs.mehregan.config import MehreganEnvConfig
    from rl_adaptive_dbs.user_config import resolve_config

    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig()
    plant = PythonPlant(config=plant_cfg)
    dt_ms = float(PAPER_DT_MS)
    alphabet = BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=TRAILING_RL_STEP_S,
        dt_ms=dt_ms,
    )
    plant.config = PlantConfig(pd=1, dt_ms=dt_ms)
    plant.reset(seed=seed)
    plant.integrate(_fig2a.DBS_ONSET_SIM, DbsSpec.none())

    actor, _ = load_actor(checkpoint)
    actor.eval()
    obs_scale = env_cfg.observation_scale
    obs = np.zeros((1,), dtype=np.float32)
    actions: list[int] = []
    for _ in range(TRAILING_STIM_STEPS):
        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = int(actor(state_t).argmax(dim=-1).item())
        actions.append(action)
        spec = alphabet.to_dbs_spec(action)
        result = plant.integrate(TRAILING_RL_STEP_S, spec)
        if result.p_beta is None:
            msg = "plant integrate missing p_beta during fine action rollout"
            raise RuntimeError(msg)
        obs = np.array([result.p_beta / obs_scale], dtype=np.float32)
    return actions


def _trailing_condition_trace(
    plant: PythonPlant,
    *,
    seed: int,
    label: str,
    segment_actions: list[int] | None,
    alphabet: BurstPatternAlphabet | None,
    scalar_hz: float | None,
    times: np.ndarray,
    rl_step_s: float = SEGMENT_S,
) -> np.ndarray:
    dt_ms = plant.config.dt_ms
    idbs = _integrate_idbs(
        duration_s=_fig2a.INTEGRATE_S,
        dt_ms=dt_ms,
        onset_sim_s=_fig2a.DBS_ONSET_SIM,
        segment_actions=segment_actions,
        alphabet=alphabet,
        scalar_hz=scalar_hz,
        rl_step_s=rl_step_s,
    )
    plant.config = PlantConfig(pd=1, dt_ms=dt_ms)
    plant.reset(seed=seed)
    if scalar_hz is None and not segment_actions:
        spec = DbsSpec.none()
    elif scalar_hz is not None:
        spec = DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(scalar_hz).pick_dbs_freq,
            idbs=idbs,
            mean_hz=scalar_hz,
        )
    else:
        spec = DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(MEAN_HZ).pick_dbs_freq,
            idbs=idbs,
            mean_hz=MEAN_HZ,
        )
    print(f"  trailing integrate: {label} (seed {seed})", flush=True)
    result = plant.integrate(
        _fig2a.INTEGRATE_S,
        spec,
        gpi_spike_buffer_size=_fig2a.fig2a_gpi_spike_buffer_size(integrate_s=_fig2a.INTEGRATE_S),
    )
    if not result.gpi_spikes:
        msg = f"plant integrate did not record GPi spikes for {label}"
        raise RuntimeError(msg)
    return _fig2a.trailing_p_beta(
        result.gpi_spikes,
        dt_ms=result.dt_ms,
        times=times,
        window_s=_fig2a.WINDOW_S,
        label=label,
        verbose=True,
    )


def _run_trailing_eval(
    *,
    seed: int,
    checkpoint: Path,
    eval_path: Path,
) -> dict[str, Any]:
    t0 = time.time()
    times = _fig2a.sample_times(_fig2a.STEP_S, duration_s=_fig2a.DISPLAY_S)
    plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=PAPER_DT_MS))
    dt_ms = float(PAPER_DT_MS)
    alphabet = BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=TRAILING_RL_STEP_S,
        dt_ms=dt_ms,
    )
    periodic_actions = [0] * TRAILING_STIM_STEPS
    trained_actions = _trained_actions_fine(checkpoint, seed=seed)

    print("Fig 5b trailing eval — 0.2 s samples, 2 s window (Fig 2a protocol)", flush=True)
    traces = {
        "no_stim": _trailing_condition_trace(
            plant,
            seed=seed,
            label="no_stim",
            segment_actions=None,
            alphabet=None,
            scalar_hz=None,
            times=times,
        ),
        "periodic": _trailing_condition_trace(
            plant,
            seed=seed,
            label="periodic_30hz",
            segment_actions=periodic_actions,
            alphabet=alphabet,
            scalar_hz=None,
            times=times,
            rl_step_s=TRAILING_RL_STEP_S,
        ),
        "trained": _trailing_condition_trace(
            plant,
            seed=seed,
            label="trained",
            segment_actions=trained_actions,
            alphabet=alphabet,
            scalar_hz=None,
            times=times,
            rl_step_s=TRAILING_RL_STEP_S,
        ),
    }
    plant.close()

    payload: dict[str, Any] = {
        "figure": "mehregan_fig5b",
        "sampling": "trailing",
        "mean_hz": MEAN_HZ,
        "seed": seed,
        "plant_dt_ms": PAPER_DT_MS,
        "skip_regular": False,
        "checkpoint": str(checkpoint),
        "integrate_s": _fig2a.INTEGRATE_S,
        "warmup_s": _fig2a.WARMUP_S,
        "display_s": _fig2a.DISPLAY_S,
        "step_s": _fig2a.STEP_S,
        "window_s": _fig2a.WINDOW_S,
        "stim_onset_display_s": STIM_ONSET_S,
        "rl_step_s": TRAILING_RL_STEP_S,
        "eval_steps": EVAL_STEPS,
        "trained_actions": trained_actions,
        "time_s": times.tolist(),
        "traces": {key: values.tolist() for key, values in traces.items()},
        "elapsed_s": round(time.time() - t0, 2),
    }
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote trailing eval JSON {eval_path}", flush=True)
    return payload


def fig5b_pass(panel: dict[str, Any]) -> dict[str, Any]:
    """Qualitative gates for Fig 5b (post-onset means, raw PSD)."""
    trained = panel.get("trained_mean")
    if trained is None:
        return {
            "shared_baseline": False,
            "trained_below_no_stim": False,
            "trained_below_periodic": False,
            "periodic_above_no_stim": False,
            "pass": False,
            "note": "missing trained policy",
        }
    no_stim = panel["no_stim_mean"]
    periodic = panel["periodic_mean"]
    baseline_delta = abs(panel["baseline_no_stim"] - panel["baseline_periodic"])
    shared_baseline = baseline_delta < 25.0
    trained_below_no_stim = trained < no_stim
    trained_below_periodic = trained < periodic
    periodic_above_no_stim = periodic > no_stim
    return {
        "shared_baseline": shared_baseline,
        "baseline_delta": baseline_delta,
        "trained_below_no_stim": trained_below_no_stim,
        "trained_below_periodic": trained_below_periodic,
        "periodic_above_no_stim": periodic_above_no_stim,
        "pass": shared_baseline
        and trained_below_no_stim
        and trained_below_periodic
        and periodic_above_no_stim,
        "no_stim_mean": no_stim,
        "trained_mean": trained,
        "periodic_mean": periodic,
    }


def _ylim_for_traces(
    traces: list[list[float]],
    *,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[float, float, list[float]]:
    """Choose y-limits from data (default) or explicit overrides.

    When ``y_min`` / ``y_max`` are ``None``, pad the finite trace extrema and
    snap to a nice 50-unit grid. Explicit values pin that end of the axis.
    """
    flat = [v for trace in traces for v in trace if np.isfinite(v)]
    if not flat:
        lo = 100.0 if y_min is None else float(y_min)
        hi = 700.0 if y_max is None else float(y_max)
        if hi <= lo:
            hi = lo + 200.0
        ticks = [float(t) for t in np.arange(lo, hi + 1e-9, 100.0)]
        return lo, hi, ticks

    pad = 20.0
    data_lo = float(np.min(flat)) - pad
    data_hi = float(np.max(flat)) + pad
    lo = float(y_min) if y_min is not None else data_lo
    hi = float(y_max) if y_max is not None else data_hi
    # Keep non-negative beta power readable; do not force a fixed paper band.
    if y_min is None:
        lo = max(0.0, lo)
    lo = float(np.floor(lo / 50.0) * 50.0)
    hi = float(np.ceil(hi / 50.0) * 50.0)
    if hi <= lo:
        hi = lo + 200.0
    step = 50.0 if (hi - lo) <= 400.0 else 100.0
    ticks = [float(t) for t in np.arange(lo, hi + 1e-9, step)]
    if not ticks or ticks[-1] < hi - 1e-9:
        ticks.append(hi)
    return lo, hi, ticks


def plot_fig5b(
    payload: dict[str, Any],
    *,
    out_path: Path,
    y_min: float | None = DEFAULT_Y_MIN,
    y_max: float | None = DEFAULT_Y_MAX,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    sampling = payload.get("sampling", "segment")
    trained_key: str | None = None

    if sampling == "trailing":
        times = np.asarray(payload["time_s"], dtype=float)
        trace_map = payload["traces"]
        no_stim = trace_map["no_stim"]
        periodic = trace_map["periodic"]
        trained = trace_map.get("trained")
        trained_key = "trained" if trained is not None else None
        plotted: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "no_stim": (times, np.asarray(no_stim, dtype=float)),
            "periodic": (times, np.asarray(periodic, dtype=float)),
        }
        if trained is not None:
            plotted["trained"] = (times, np.asarray(trained, dtype=float))
        y0, y1, yticks = _ylim_for_traces(
            [list(v) for v in trace_map.values()],
            y_min=y_min,
            y_max=y_max,
        )
        plot_order = ("no_stim", "periodic", "trained")
        for key in plot_order:
            if key not in plotted:
                continue
            meta = SERIES[key]
            x, y = plotted[key]
            ax.plot(
                x,
                y,
                color=meta["color"],
                linestyle=meta["linestyle"],
                linewidth=meta["linewidth"],
                zorder=meta["zorder"],
                label=meta["label"],
            )
        baseline_no_stim = _baseline_at_onset(times, no_stim)
        baseline_periodic = _baseline_at_onset(times, periodic)
        no_stim_mean = _post_onset_mean_trailing(times, no_stim)
        periodic_mean = _post_onset_mean_trailing(times, periodic)
        trained_mean = (
            _post_onset_mean_trailing(times, trained) if trained is not None else None
        )
        trained_equals_periodic = (
            trained is not None and np.allclose(trained, periodic, rtol=0.0, atol=1.0)
        )
    else:
        policies = payload["paper_protocol_policies"]

        if "no_stim" not in policies:
            msg = "eval JSON missing no_stim baseline (re-run eval with scalar baselines)"
            raise KeyError(msg)

        no_stim = _policy_p_beta(policies, "no_stim")
        periodic = _policy_p_beta(policies, "pattern0_regular")
        trained_key = _find_trained_key(policies)
        if trained_key is None:
            print(
                "warning: no trained_ddpg_* policy — skipping green trace",
                file=sys.stderr,
            )
            trained = None
        else:
            trained = _policy_p_beta(policies, trained_key)

        plotted_seg: dict[str, list[float]] = {
            "no_stim": no_stim,
            "periodic": periodic,
        }
        if trained is not None:
            plotted_seg["trained"] = trained
        y0, y1, yticks = _ylim_for_traces(list(plotted_seg.values()), y_min=y_min, y_max=y_max)

        plot_order = ("no_stim", "periodic", "trained")
        for key in plot_order:
            if key not in plotted_seg:
                continue
            meta = SERIES[key]
            x, y = _step_series(plotted_seg[key])
            ax.step(
                x,
                y,
                where="post",
                color=meta["color"],
                linestyle=meta["linestyle"],
                linewidth=meta["linewidth"],
                zorder=meta["zorder"],
                label=meta["label"],
            )
        baseline_no_stim = float(no_stim[0])
        baseline_periodic = float(periodic[0])
        no_stim_mean = _post_onset_mean(no_stim)
        periodic_mean = _post_onset_mean(periodic)
        trained_mean = _post_onset_mean(trained) if trained is not None else None
        trained_equals_periodic = trained is not None and trained == periodic

    ax.axvline(STIM_ONSET_S, color="#888888", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xlim(0.0, TIME_MAX_S)
    ax.set_ylim(y0, y1)
    ax.set_yticks(yticks)
    ax.set_xticks(np.arange(0.0, TIME_MAX_S + 1e-9, 2.0))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("PSD")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=1)
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    panel = {
        "out": str(out_path),
        "sampling": sampling,
        "trained_key": trained_key,
        "trained_equals_periodic": trained_equals_periodic,
        "y_min": y0,
        "y_max": y1,
        "baseline_no_stim": baseline_no_stim,
        "baseline_periodic": baseline_periodic,
        "no_stim_mean": no_stim_mean,
        "trained_mean": trained_mean,
        "periodic_mean": periodic_mean,
    }
    if trained_mean is None:
        panel["gates"] = {
            "shared_baseline": abs(panel["baseline_no_stim"] - panel["baseline_periodic"]) < 25.0,
            "trained_below_no_stim": False,
            "trained_below_periodic": False,
            "periodic_above_no_stim": panel["periodic_mean"] > panel["no_stim_mean"],
            "pass": False,
            "note": "missing trained policy",
        }
    else:
        panel["gates"] = fig5b_pass(panel)
    return panel


def _run_paper_eval(
    *,
    seed: int,
    landscape: Path,
    checkpoints: list[Path],
    eval_path: Path,
    sampling: str,
) -> dict[str, Any]:
    if sampling == "trailing":
        if not checkpoints or not checkpoints[0].exists():
            msg = f"missing checkpoint for trailing eval: {checkpoints}"
            raise FileNotFoundError(msg)
        return _run_trailing_eval(
            seed=seed,
            checkpoint=checkpoints[0],
            eval_path=eval_path,
        )
    if not landscape.exists():
        msg = f"missing landscape JSON: {landscape}"
        raise FileNotFoundError(msg)
    t0 = time.time()
    payload = run_eval(
        mean_hz=MEAN_HZ,
        landscape_path=landscape,
        checkpoints=checkpoints,
        seed=seed,
        eval_steps=EVAL_STEPS,
        plant_dt_ms=PAPER_DT_MS,
        include_scalar_baselines=True,
        skip_regular=False,
    )
    payload["sampling"] = "segment"
    payload["elapsed_s"] = round(time.time() - t0, 2)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote eval JSON {eval_path}", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--landscape", type=Path, default=DEFAULT_LANDSCAPE)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output PNG (default: next {OUT_STEM}_vN.png)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Y-axis minimum (default: auto from traces)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Y-axis maximum (default: auto from traces)",
    )
    parser.add_argument(
        "--sampling",
        choices=("trailing", "segment"),
        default="trailing",
        help="Biomarker sampling: trailing 0.2 s (Fig 2a default) or 2 s RL segment steps",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-update-docs", dest="update_docs", action="store_false")
    parser.set_defaults(update_docs=True)
    args = parser.parse_args()

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)

    if args.plot_only:
        if not args.eval_json.exists():
            print(f"missing eval JSON: {args.eval_json}", file=sys.stderr)
            return 2
        payload = json.loads(args.eval_json.read_text())
    else:
        if not args.checkpoint.exists():
            print(f"missing checkpoint: {args.checkpoint}", file=sys.stderr)
            print(
                "Train first: tmux new-session -d -s retrain-30hz "
                '"setsid nohup uv run python scripts/retrain_30hz_fig5b.py '
                '>> logs/retrain-30hz.log 2>&1 < /dev/null"',
                file=sys.stderr,
            )
            return 2
        payload = _run_paper_eval(
            seed=args.seed,
            landscape=args.landscape,
            checkpoints=[args.checkpoint],
            eval_path=args.eval_json,
            sampling=args.sampling,
        )

    args.out = _vault_backed_png(args.out)
    panel = plot_fig5b(payload, out_path=args.out, y_min=args.y_min, y_max=args.y_max)
    gates = panel["gates"]

    manifest = {
        "figure": "mehregan_fig5b",
        "mean_hz": MEAN_HZ,
        "seed": payload.get("seed", args.seed),
        "plant_dt_ms": payload.get("plant_dt_ms", PAPER_DT_MS),
        "skip_regular": False,
        "sampling": payload.get("sampling", args.sampling),
        "eval_json": str(args.eval_json),
        "checkpoint": str(args.checkpoint) if args.checkpoint.exists() else None,
        "output_png": _figure_promote.repo_rel_posix(args.out),
        "png_version": png_version,
        "panel": panel,
        "gates": gates,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.manifest}", flush=True)
    if png_version is not None:
        print(f"output PNG version={png_version}", flush=True)
    trained_note = (
        f"{panel['trained_mean']:.1f}"
        if panel.get("trained_mean") is not None
        else "n/a"
    )
    print(
        f"gates pass={gates['pass']} trained={trained_note} "
        f"no_stim={panel['no_stim_mean']:.1f} periodic={panel['periodic_mean']:.1f}",
        flush=True,
    )

    if args.update_docs:
        updated = _figure_promote.promote_5b(
            manifest=manifest,
            eval_path=args.eval_json,
            png_path=args.out,
            update_docs=True,
        )
        print(f"updated docs caption: {updated.get('caption')}", flush=True)

    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
