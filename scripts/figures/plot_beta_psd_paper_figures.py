#!/usr/bin/env python3
"""TASK-144: Publication-style beta PSD time-series plots (Mehregan Fig 5/6 style).

Loads paper-protocol eval JSON from run_task108_paper_protocol_eval.py and plots
P_beta (13–35 Hz GPi power) over the 2 s baseline + 5×2 s stimulation protocol.

Run:
  uv run python scripts/figures/plot_beta_psd_paper_figures.py
  uv run python scripts/figures/plot_beta_psd_paper_figures.py --json-30hz path/to/30hz.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path("artifacts/ddpg")
SEGMENT_S = 2.0
N_SEGMENTS = 6  # baseline + 5 stim steps
TIME_MAX_S = 12.0

# Paper-like dark theme with high-contrast series colors.
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#30363d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
}
SERIES_COLORS = {
    "no_stimulation": "#f85149",
    "pattern0": "#58a6ff",
    "irregular": "#d29922",
    "trained": "#3fb950",
    "ptq": "#a371f7",
    "qat": "#f778ba",
}


def _segment_times() -> np.ndarray:
    """Left edges of each 2 s segment (0, 2, …, 10)."""
    return np.arange(0, TIME_MAX_S, SEGMENT_S)


def _step_series(p_beta: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Step-function (x, y) holding each segment's P_beta for 2 s."""
    if len(p_beta) != N_SEGMENTS:
        msg = f"expected {N_SEGMENTS} P_beta samples, got {len(p_beta)}"
        raise ValueError(msg)
    x = np.concatenate([_segment_times(), [TIME_MAX_S]])
    y = np.concatenate([p_beta, [p_beta[-1]]])
    return x, y


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _policy_p_beta(policies: dict[str, Any], key: str) -> list[float]:
    data = policies[key]
    if "error" in data:
        msg = f"policy {key!r} has error: {data['error']}"
        raise KeyError(msg)
    return [float(v) for v in data["p_beta"]]


def _find_trained_key(policies: dict[str, Any], prefer: str | None = None) -> str | None:
    if prefer and prefer in policies and "p_beta" in policies[prefer]:
        return prefer
    trained = [
        k
        for k, v in policies.items()
        if k.startswith("trained_ddpg_") and isinstance(v, dict) and "p_beta" in v
    ]
    if not trained:
        return None
    if prefer:
        for k in trained:
            if prefer in k:
                return k
    return trained[0]


def plot_fig5_panel(
    payload: dict[str, Any],
    *,
    title: str,
    series: list[tuple[str, str, str]],
    out_path: Path,
    trained_prefer: str | None = None,
) -> dict[str, Any]:
    """Plot one Fig 5 panel. series = [(policy_key, label, color_key), ...]."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    policies = payload["paper_protocol_policies"]
    no_stim = float(payload["no_stim_1step"]["p_beta_raw"])
    ax.axhline(
        no_stim,
        color=SERIES_COLORS["no_stimulation"],
        linestyle="--",
        linewidth=1.5,
        label="No stimulation (baseline)",
        alpha=0.85,
    )

    plotted: list[str] = []
    for policy_key, label, color_key in series:
        if policy_key not in policies:
            continue
        p_beta = _policy_p_beta(policies, policy_key)
        x, y = _step_series(p_beta)
        ax.step(
            x,
            y,
            where="post",
            color=SERIES_COLORS[color_key],
            linewidth=2.0,
            label=label,
        )
        plotted.append(policy_key)

    trained_key = _find_trained_key(policies, trained_prefer)
    if trained_key and trained_key not in plotted:
        p_beta = _policy_p_beta(policies, trained_key)
        x, y = _step_series(p_beta)
        ax.step(
            x,
            y,
            where="post",
            color=SERIES_COLORS["trained"],
            linewidth=2.0,
            label=f"Trained RL ({trained_key.removeprefix('trained_ddpg_')})",
        )
        plotted.append(trained_key)

    ax.set_xlim(0, TIME_MAX_S)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\beta$ PSD (13–35 Hz, GPi)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "out": str(out_path),
        "plotted": plotted,
        "trained_key": trained_key,
        "no_stim_p_beta": no_stim,
    }


def plot_fig6_panel(
    payloads: list[tuple[str, dict[str, Any]]],
    *,
    title: str,
    out_path: Path,
) -> dict[str, Any]:
    """Overlay multiple trained/quantized policies (Fig 6 style)."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    color_cycle = [SERIES_COLORS["trained"], SERIES_COLORS["ptq"], SERIES_COLORS["qat"]]
    plotted: list[str] = []

    for i, (label, payload) in enumerate(payloads):
        policies = payload.get("paper_protocol_policies", {})
        trained_key = _find_trained_key(policies)
        if not trained_key:
            continue
        p_beta = _policy_p_beta(policies, trained_key)
        x, y = _step_series(p_beta)
        color = color_cycle[i % len(color_cycle)]
        ax.step(x, y, where="post", color=color, linewidth=2.0, label=label)
        plotted.append(label)

    if payload := (payloads[0][1] if payloads else None):
        no_stim = float(payload["no_stim_1step"]["p_beta_raw"])
        ax.axhline(
            no_stim,
            color=SERIES_COLORS["no_stimulation"],
            linestyle="--",
            linewidth=1.5,
            label="No stimulation",
            alpha=0.85,
        )

    ax.set_xlim(0, TIME_MAX_S)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\beta$ PSD (13–35 Hz, GPi)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {"out": str(out_path), "plotted": plotted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-45hz",
        type=Path,
        default=ARTIFACTS / "task108_paper_protocol_45hz.json",
    )
    parser.add_argument(
        "--json-30hz",
        type=Path,
        default=ARTIFACTS / "task108_paper_protocol_30hz.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ARTIFACTS,
    )
    parser.add_argument(
        "--fig6-json",
        type=Path,
        action="append",
        default=[],
        help="Extra eval JSON for Fig 6 overlays (label:auto from filename)",
    )
    args = parser.parse_args()

    summary: dict[str, Any] = {"panels": {}}

    if args.json_45hz.exists():
        p45 = _load_payload(args.json_45hz)
        summary["panels"]["fig5a_45hz"] = plot_fig5_panel(
            p45,
            title="Fig 5a — 45 Hz mean rate (paper protocol)",
            series=[
                ("pattern0_regular", "Periodic 45 Hz (pattern 0)", "pattern0"),
            ],
            out_path=args.out_dir / "fig5a_beta_psd_45hz.png",
            trained_prefer="trained_ddpg_pattern_train_v2",
        )
    else:
        print(f"skip 45 Hz: missing {args.json_45hz}", file=sys.stderr)

    if args.json_30hz.exists():
        p30 = _load_payload(args.json_30hz)
        irregular_key = "best_open_loop_irregular"
        for k in p30["paper_protocol_policies"]:
            if k.startswith("best_open_loop_irregular"):
                irregular_key = k
                break
        summary["panels"]["fig5b_30hz"] = plot_fig5_panel(
            p30,
            title="Fig 5b — 30 Hz mean rate (paper protocol)",
            series=[
                ("pattern0_regular", "Periodic 30 Hz (pattern 0)", "pattern0"),
                (irregular_key, "Best irregular open-loop (landscape)", "irregular"),
            ],
            out_path=args.out_dir / "fig5b_beta_psd_30hz.png",
            trained_prefer="trained_ddpg_pattern_train_30hz",
        )
    else:
        print(f"skip 30 Hz: missing {args.json_30hz}", file=sys.stderr)

    fig6_sources: list[tuple[str, dict[str, Any]]] = []
    for path in args.fig6_json:
        if path.exists():
            fig6_sources.append((path.stem, _load_payload(path)))

    if fig6_sources:
        summary["panels"]["fig6"] = plot_fig6_panel(
            fig6_sources,
            title="Fig 6 — Quantized vs full-precision (paper protocol)",
            out_path=args.out_dir / "fig6_beta_psd_quantized.png",
        )

    manifest_path = args.out_dir / "fig5_beta_psd_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
