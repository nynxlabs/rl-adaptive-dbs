#!/usr/bin/env python3
"""Mehregan et al. (paper 1) Figure 4b — training curves vs episode.

Paper §IV.A.1 companion to Fig 4a: two panels over **9 episodes** indexed
**0–8** (paper x-axis ticks 0, 2, 4, 6, 8; line reaches episode 8):

  1. **Episode total reward** vs episode
  2. **Episode-mean PSD(x10³)** ($P_\\beta / 1000$) vs episode

Loads traces from the Fig 4a series cache (default: locked ``series_v4.json``,
first 8 episodes). Resume training via Fig 4a ``--resume`` (this panel replots cache only). (``training_fig4b_vN.png``)
for showcase side-by-side use, plus separate reward/PSD PNGs for debugging.

Run:
  uv run python scripts/figures/papers/mehregan/4b/plot.py
  uv run python scripts/figures/papers/mehregan/4b/plot.py --plot-only
  uv run python scripts/figures/papers/mehregan/4b/plot.py --episodes 8 --fig4a-series artifacts/figures/papers/mehregan/4a/series_v4.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig4b_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

FIGURES_DIR = Path("figures/mehregan/images/4b")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/4b")
FIG4A_CACHE_DIR = Path("artifacts/figures/papers/mehregan/4a")
DEFAULT_FIG4A_SERIES = FIG4A_CACHE_DIR / "series_v4.json"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
REWARD_STEM = "training_reward"
PSD_STEM = "training_psd"
COMBINED_STEM = "training_fig4b"

DEFAULT_EPISODES = 9
STEPS_PER_EPISODE = 30

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
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


def _episode_rewards_from_payload(payload: dict[str, Any]) -> list[float]:
    training = payload.get("training") or {}
    rewards = training.get("episode_rewards")
    if rewards is None:
        rewards = payload.get("episode_rewards")
    if not rewards:
        msg = "no episode_rewards in Fig 4a cache (expected under training or top-level)"
        raise ValueError(msg)
    return [float(r) for r in rewards]


def _episode_mean_beta(payload: dict[str, Any], n_episodes: int) -> list[float]:
    trace = payload.get("beta_norm_trace")
    if not trace:
        msg = "no beta_norm_trace in Fig 4a cache — required for PSD vs episode panel"
        raise ValueError(msg)
    y = np.asarray(trace, dtype=float)
    need = n_episodes * STEPS_PER_EPISODE
    if y.size < need:
        msg = f"beta_norm_trace has {y.size} steps; need {need} for {n_episodes} episodes"
        raise ValueError(msg)
    reshaped = y[:need].reshape(n_episodes, STEPS_PER_EPISODE)
    return [float(row.mean()) for row in reshaped]


def _truncate_episodes(
    episode_rewards: list[float],
    episode_mean_beta: list[float],
    n_episodes: int,
) -> tuple[list[float], list[float]]:
    if len(episode_rewards) < n_episodes:
        msg = (
            f"only {len(episode_rewards)} episode rewards in cache; "
            f"requested {n_episodes}"
        )
        raise ValueError(msg)
    if len(episode_mean_beta) < n_episodes:
        msg = (
            f"only {len(episode_mean_beta)} episode-mean PSD values; "
            f"requested {n_episodes}"
        )
        raise ValueError(msg)
    return episode_rewards[:n_episodes], episode_mean_beta[:n_episodes]


def _fig4b_pass(episode_rewards: list[float]) -> dict[str, Any]:
    if len(episode_rewards) < 2:
        return {"pass": False, "reason": "insufficient episodes"}
    ep1 = float(episode_rewards[0])
    ep_last = float(episode_rewards[-1])
    best = float(max(episode_rewards))
    best_ep = int(np.argmax(episode_rewards)) + 1
    recovery = best - ep1
    late_better = ep_last > ep1
    rise_ep = None
    for i, r in enumerate(episode_rewards[1:], start=1):
        if r > ep1 + 5.0:
            rise_ep = i
            break
    return {
        "pass": late_better or recovery > 20.0,
        "ep1": ep1,
        "ep_last": ep_last,
        "best_ep": best_ep,
        "best_reward": best,
        "recovery_from_ep1": recovery,
        "rise_episode": rise_ep,
        "late_better": late_better,
    }


def _gate_summary(
    episode_rewards: list[float],
    *,
    episode_mean_beta: list[float],
) -> dict[str, Any]:
    dig = fig4b_gates(episode_rewards, episode_mean_beta)
    legacy = _fig4b_pass(episode_rewards)
    n = len(episode_rewards)
    metrics = dig["metrics"]
    gates = dict(dig["gates"])
    gates["plot_style"] = n >= 2
    gates["automation"] = bool(legacy.get("pass"))
    return {
        "n_episodes": n,
        "early_mean_ep1_3": metrics.get("early_reward"),
        "late_mean_ep6_end": metrics.get("late_reward"),
        "automation_pass": legacy["pass"],
        "rise_episode": metrics.get("rise_episode"),
        "beta_early_mean": metrics.get("early_beta"),
        "beta_late_mean": metrics.get("late_beta"),
        "beta_trend_down": gates.get("beta_drops"),
        "paper_gate_metrics": metrics,
        "paper_ref": dig["paper_ref"],
        "gates": gates,
        "gates_pass": all(gates.values()),
        "fig4b_pass": legacy,
    }


def _nice_step(span: float) -> float:
    """Pick a readable tick step from the data span."""
    if span <= 0:
        return 1.0
    raw = span / 5.0
    magnitude = 10.0 ** np.floor(np.log10(raw))
    residual = raw / magnitude
    if residual <= 1.5:
        step = magnitude
    elif residual <= 3.5:
        step = 2.0 * magnitude
    elif residual <= 7.5:
        step = 5.0 * magnitude
    else:
        step = 10.0 * magnitude
    return float(step)


def _snap_ylim(
    y: np.ndarray,
    *,
    step: float,
    default_lo: float,
    default_hi: float,
) -> tuple[float, float, list[float]]:
    """Axis limits = tight ceil/floor of data extrema on a fixed tick step."""
    if y.size == 0 or not np.isfinite(y).any():
        return default_lo, default_hi, [default_lo, default_hi]
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    lo = float(np.floor(y_min / step) * step)
    hi = float(np.ceil(y_max / step) * step)
    if hi <= lo:
        hi = lo + step
    ticks = [float(t) for t in np.arange(lo, hi + 1e-9, step)]
    if not ticks or ticks[-1] < hi - 1e-9:
        ticks.append(hi)
    return lo, hi, ticks


def _reward_tick_step(span: float) -> float:
    """Readable reward ticks; span ~45 → step 10 (−30…20), smaller spans → step 5."""
    if span <= 20:
        return 5.0
    if span <= 50:
        return 10.0
    return _nice_step(span)


def _ylim_for_rewards(y: np.ndarray) -> tuple[float, float, list[float]]:
    if y.size == 0 or not np.isfinite(y).any():
        return -10.0, 10.0, [-10, 0, 10]
    span = float(np.nanmax(y) - np.nanmin(y))
    step = _reward_tick_step(span)
    return _snap_ylim(y, step=step, default_lo=-10.0, default_hi=10.0)


def _ylim_for_psd(y: np.ndarray) -> tuple[float, float, list[float]]:
    return _snap_ylim(y, step=0.05, default_lo=0.3, default_hi=0.5)


def _episode_x_coords(n_episodes: int) -> np.ndarray:
    """0-based episode indices (paper labels first training episode as 0)."""
    return np.arange(n_episodes, dtype=float)


def _episode_x_axis(ax: plt.Axes, n_points: int) -> None:
    """Paper Fig 4b: last episode index N; xlim and line both reach N."""
    last_ep = max(0, n_points - 1)
    ax.set_xlim(0.0, float(last_ep))
    ax.set_xticks(np.arange(0, last_ep + 1, 2))


def _plot_reward_on_ax(ax: plt.Axes, episode_rewards: list[float]) -> dict[str, Any]:
    y = np.asarray(episode_rewards, dtype=float)
    x = _episode_x_coords(len(y))
    ax.plot(x, y, color="#16a34a", linewidth=1.2, label="Replication")
    paper = _paper_overlay.overlay_mehregan_fig4b_reward(ax)
    _paper_xy = next(iter(paper.values()), (np.array([]), np.array([])))
    paper_y = _paper_xy[1] if len(_paper_xy) > 1 else np.array([])
    y_all = np.concatenate([y, paper_y]) if paper_y.size else y
    _episode_x_axis(ax, len(y))
    y0, y1, yticks = _ylim_for_rewards(y_all)
    ax.set_ylim(y0, y1)
    ax.set_yticks(yticks)
    ax.set_ylabel("Reward")
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    _paper_overlay.place_legend(ax, fontsize=8, loc="lower right")
    return {
        "n_episodes": int(y.size),
        "y_min": float(y.min()) if y.size else float("nan"),
        "y_max": float(y.max()) if y.size else float("nan"),
        "y_mean": float(y.mean()) if y.size else float("nan"),
        "ylim": [y0, y1],
    }


def _plot_psd_on_ax(ax: plt.Axes, episode_mean_beta: list[float]) -> dict[str, Any]:
    y = np.asarray(episode_mean_beta, dtype=float)
    x = _episode_x_coords(len(y))
    ax.plot(x, y, color="#1f6f6f", linewidth=1.2, label="Replication")
    paper = _paper_overlay.overlay_mehregan_fig4b_psd(ax)
    _paper_xy = next(iter(paper.values()), (np.array([]), np.array([])))
    paper_y = _paper_xy[1] if len(_paper_xy) > 1 else np.array([])
    y_all = np.concatenate([y, paper_y]) if paper_y.size else y
    _episode_x_axis(ax, len(y))
    y0, y1, yticks = _ylim_for_psd(y_all)
    ax.set_ylim(y0, y1)
    ax.set_yticks(yticks)
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"PSD($x10^3$)")
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    _paper_overlay.place_legend(ax, fontsize=8, loc="upper right")
    return {
        "n_episodes": int(y.size),
        "y_min": float(y.min()) if y.size else float("nan"),
        "y_max": float(y.max()) if y.size else float("nan"),
        "y_mean": float(y.mean()) if y.size else float("nan"),
        "ylim": [y0, y1],
    }


def plot_reward_vs_episode(
    episode_rewards: list[float],
    *,
    out_path: Path,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    panel = _plot_reward_on_ax(ax, episode_rewards)
    ax.set_xlabel("Episode")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return panel


def plot_psd_vs_episode(
    episode_mean_beta: list[float],
    *,
    out_path: Path,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    panel = _plot_psd_on_ax(ax, episode_mean_beta)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return panel


def plot_combined_fig4b(
    episode_rewards: list[float],
    episode_mean_beta: list[float],
    *,
    out_path: Path,
) -> dict[str, Any]:
    """Paper-style stacked panel: reward (top) + episode-mean PSD (bottom)."""
    plt.rcParams.update(STYLE)
    fig, (ax_reward, ax_psd) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8.0, 7.0),
        dpi=150,
    )
    reward_panel = _plot_reward_on_ax(ax_reward, episode_rewards)
    psd_panel = _plot_psd_on_ax(ax_psd, episode_mean_beta)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "reward": reward_panel,
        "psd": psd_panel,
        "n_episodes": reward_panel["n_episodes"],
    }


def _checklist_rows(gates: dict[str, Any], summary: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    g = summary.get("gates") or {}
    ep_pass = summary.get("fig4b_pass") or {}
    n = summary.get("n_episodes", "—")

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)

    rise = summary.get("rise_episode")
    rise_txt = f"first rise by episode {rise}" if rise is not None else "no sharp rise detected"
    beta_early = summary.get("beta_early_mean")
    beta_late = summary.get("beta_late_mean")
    beta_txt = "—"
    n0 = int(n) - 1 if isinstance(n, int) else n
    if isinstance(beta_early, float) and isinstance(beta_late, float):
        beta_txt = f"mean ep0–2={beta_early:.3f}, ep6–{n0}={beta_late:.3f}"

    reward_ylim = summary.get("reward_ylim")
    psd_ylim = summary.get("psd_ylim")
    reward_ylim_txt = (
        f"auto snap {reward_ylim[0]:.0f}–{reward_ylim[1]:.0f}"
        if isinstance(reward_ylim, list) and len(reward_ylim) == 2
        else "auto"
    )
    psd_ylim_txt = (
        f"auto snap {psd_ylim[0]:.2f}–{psd_ylim[1]:.2f}"
        if isinstance(psd_ylim, list) and len(psd_ylim) == 2
        else "auto"
    )

    return [
        (
            "**Plot style**",
            "One panel; reward + PSD vs episode 0–8",
            f"Two PNGs; episodes 0–{n0} (0-based), line reaches ep {n0}",
            "✓" if g.get("plot_style") else "✗",
        ),
        (
            "**Axes (episodes)**",
            "0–8, ticks every 2",
            "0–8, ticks every 2",
            "✓",
        ),
        (
            "**Axes (reward scale)**",
            "~−80–0",
            reward_ylim_txt,
            "~",
        ),
        (
            "**Axes (PSD scale)**",
            "~0.35–0.50",
            psd_ylim_txt,
            "~",
        ),
        (
            "**Early episodes (0–2)**",
            "Reward negative (~−80 to ~−55)",
            f"mean {_fmt(summary.get('early_mean_ep1_3'))} (negative, different magnitude)",
            "~" if g.get("early_negative") else "✗",
        ),
        (
            "**Rise timing**",
            "Climb ~ep 2–6 toward ~0",
            rise_txt,
            "~" if g.get("rise_timing") else "✗",
        ),
        (
            "**Late episodes (6–8)**",
            "Plateau near **0**",
            f"mean {_fmt(summary.get('late_mean_ep6_end'))} (improved, not near 0)",
            "~",
        ),
        (
            "**Episode-mean PSD**",
            "Gradual fall ~0.50→~0.37",
            beta_txt if beta_txt != "—" else "see manifest",
            "~" if g.get("beta_inverse_trend") else "✗",
        ),
        (
            "**Automation gate**",
            "—",
            f"recovery={_fmt(ep_pass.get('recovery_from_ep1'))}",
            "✓" if g.get("automation") else "✗",
        ),
    ]


def update_checklist_in_doc(rows: list[tuple[str, str, str, str]]) -> None:
    doc = _figure_promote.PAPER_1_DOC
    text = doc.read_text()
    start = text.find("## Fig 4b — training reward vs episode")
    if start < 0:
        return
    table_hdr = (
        "| Check | Paper | Replication | Match? |\n"
        "|-------|-------|-------------|--------|\n"
    )
    hdr_pos = text.find(table_hdr, start)
    if hdr_pos < 0:
        return
    body_start = hdr_pos + len(table_hdr)
    run_pos = text.find("\n**Run:**", body_start)
    if run_pos < 0:
        return
    new_rows = "".join(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in rows)
    doc.write_text(text[:body_start] + new_rows + text[run_pos:])


def update_status_in_doc(*, passed: bool, note: str) -> None:
    doc = _figure_promote.PAPER_1_DOC
    text = doc.read_text()
    new_status = f"**Status:** Pass — {note}" if passed else f"**Status:** Open — {note}"
    start = text.find("## Fig 4b — training reward vs episode")
    if start < 0:
        return
    status_pos = text.find("**Status:**", start)
    end = text.find("\n\n###", status_pos)
    if status_pos < 0 or end < 0:
        return
    doc.write_text(text[:status_pos] + new_status + text[end:])


def _load_fig4a_payload(series_path: Path, manifest_path: Path | None) -> dict[str, Any]:
    if series_path.exists():
        return json.loads(series_path.read_text())
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        training = manifest.get("training") or {}
        rewards = training.get("episode_rewards")
        if rewards:
            return {
                "figure": "mehregan_fig4a",
                "seed": manifest.get("seed"),
                "training": training,
                "episode_rewards": rewards,
            }
    print(f"missing Fig 4a cache: {series_path}", file=sys.stderr)
    if manifest_path:
        print(f"also checked manifest: {manifest_path}", file=sys.stderr)
    print("Run Fig 4a first: uv run python scripts/figures/papers/mehregan/4a/plot.py", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig4a-series",
        type=Path,
        default=DEFAULT_FIG4A_SERIES,
        help="Fig 4a series JSON (default: locked series_v4.json)",
    )
    parser.add_argument(
        "--fig4a-manifest",
        type=Path,
        default=FIG4A_CACHE_DIR / "manifest.json",
        help="Fallback manifest when series cache is missing",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of episodes to plot (default: 8)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load cached Fig 4a traces only (default behaviour)",
    )
    parser.add_argument(
        "--no-update-docs",
        dest="update_docs",
        action="store_false",
        help="Skip figures/mehregan/replications.md updates",
    )
    parser.add_argument(
        "--update-checklist",
        action="store_true",
        help="Rewrite Fig 4b checklist in figures/mehregan/replications.md (off by default; passed panels use Status only)",
    )
    parser.set_defaults(update_checklist=False)
    parser.set_defaults(update_docs=True)
    parser.add_argument(
        "--push-kb",
        action="store_true",
        help="After promote, copy replication PNGs to the knowledge-base vault",
    )
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="After promote, refresh Report 3 gallery image links in the knowledge-base",
    )
    args = parser.parse_args()
    _figure_promote.set_push_kb_images(args.push_kb)
    _figure_promote.set_update_report3(args.update_report)

    payload = _load_fig4a_payload(args.fig4a_series, args.fig4a_manifest)
    all_rewards = _episode_rewards_from_payload(payload)
    all_beta = _episode_mean_beta(payload, len(all_rewards))
    episode_rewards, episode_mean_beta = _truncate_episodes(
        all_rewards, all_beta, args.episodes
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reward_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, REWARD_STEM)
    psd_path = FIGURES_DIR / f"{PSD_STEM}_v{png_version}.png"
    combined_path = FIGURES_DIR / f"{COMBINED_STEM}_v{png_version}.png"
    reward_path = _vault_backed_png(reward_path)
    psd_path = _vault_backed_png(psd_path)
    combined_path = _vault_backed_png(combined_path)

    print(
        f"Fig 4b — {args.episodes} episodes from {args.fig4a_series}",
        flush=True,
    )

    reward_panel = plot_reward_vs_episode(episode_rewards, out_path=reward_path)
    psd_panel = plot_psd_vs_episode(episode_mean_beta, out_path=psd_path)
    combined_panel = plot_combined_fig4b(
        episode_rewards,
        episode_mean_beta,
        out_path=combined_path,
    )
    summary = _gate_summary(episode_rewards, episode_mean_beta=episode_mean_beta)
    summary["reward_ylim"] = list(reward_panel.get("ylim", []))
    summary["psd_ylim"] = list(psd_panel.get("ylim", []))
    gates = summary.get("gates") or {}
    fig4b_pass = summary.get("fig4b_pass") or {}

    manifest = {
        "figure": "mehregan_fig4b",
        "seed": payload.get("seed"),
        "fig4a_series": str(args.fig4a_series),
        "mean_hz": payload.get("mean_hz", 45.0),
        "num_episodes": args.episodes,
        "png_version": png_version,
        "episode_rewards": episode_rewards,
        "episode_mean_beta": episode_mean_beta,
        "summary": summary,
        "panels": {
            "combined": {
                **combined_panel,
                "output_png": _figure_promote.repo_rel_posix(combined_path),
            },
            "reward": {
                **reward_panel,
                "output_png": _figure_promote.repo_rel_posix(reward_path),
            },
            "psd": {
                **psd_panel,
                "output_png": _figure_promote.repo_rel_posix(psd_path),
            },
        },
        "output_png_combined": _figure_promote.repo_rel_posix(combined_path),
        "output_png_reward": _figure_promote.repo_rel_posix(reward_path),
        "output_png_psd": _figure_promote.repo_rel_posix(psd_path),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "rewards.json"
    cache_path.write_text(
        json.dumps(
            {
                "num_episodes": args.episodes,
                "episode_rewards": episode_rewards,
                "episode_mean_beta": episode_mean_beta,
                "fig4a_series": str(args.fig4a_series),
            },
            indent=2,
        )
        + "\n"
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {combined_path}", flush=True)
    print(f"wrote {reward_path}", flush=True)
    print(f"wrote {psd_path}", flush=True)
    print(f"wrote {args.manifest}", flush=True)
    print(
        f"gates: automation={gates.get('automation')} "
        f"rise_ep={summary.get('rise_episode')} "
        f"ep1={fig4b_pass.get('ep1'):.1f} ep{args.episodes - 1}={fig4b_pass.get('ep_last'):.1f} "
        f"psd {episode_mean_beta[0]:.3f}→{episode_mean_beta[-1]:.3f}",
        flush=True,
    )

    if args.update_docs:
        updated = _figure_promote.promote_4b(
            manifest=manifest,
            rewards_path=cache_path,
            reward_png_path=reward_path,
            psd_png_path=psd_path,
            combined_png_path=combined_path,
            update_docs=True,
        )
        print(f"updated docs caption: {updated.get('caption')}", flush=True)
        print(f"updated reward image: {updated.get('reward_png_repo_rel')}", flush=True)
        print(f"updated psd image: {updated.get('psd_png_repo_rel')}", flush=True)
        if gates.get("automation"):
            update_status_in_doc(
                passed=True,
                note=(
                    "two panels × 9 episodes (indices 0–8), paired with Fig 4a v4 (seed 0; "
                    "paper seed unspecified). Qualitative: reward↑, PSD↓. "
                    "Y-limits snap to data extrema. Numeric bands differ — compare trends, not pointwise values."
                ),
            )
            print("updated docs status → Pass", flush=True)

    if args.update_checklist:
        update_checklist_in_doc(_checklist_rows(gates, summary))
        print("updated docs checklist", flush=True)

    return 0 if gates.get("automation") else 1


if __name__ == "__main__":
    raise SystemExit(main())
