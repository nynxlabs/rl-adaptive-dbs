#!/usr/bin/env python3
"""Ravivarapu Fig 4a — training beta PSD vs episode (Baseline vs SEA-DBS).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py --smoke
"""
from __future__ import annotations

from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import config_from_dict, fig4_ravivarapu_config
from dataclasses import asdict, dataclass, replace

from controllers.sea_dbs.checkpoint import load_checkpoint, save_checkpoint
from controllers.sea_dbs.trainer import SEA_DBSTrainer

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import (  # noqa: E402
    merge_gate_report,
    ravivarapu_fig4a_attach_tiered_pass,
    ravivarapu_fig4a_gates,
)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_cli = load_script_module("figure_resume_cli", _RESUME_CLI)

_PARALLEL_SERIES = Path(__file__).resolve().parents[2] / "parallel_series.py"
_parallel_series = load_script_module("figure_parallel_series", _PARALLEL_SERIES)

_OVERLAY = Path(__file__).resolve().parents[2] / "paper_overlay.py"
_paper_overlay = load_script_module("figure_paper_overlay", _OVERLAY)

_COMMON = Path(__file__).resolve().parents[1] / "common.py"
_common = load_script_module("ravivarapu_common", _COMMON)

FIGURES_DIR = Path("docs/controllers/sea_dbs/figures/images/4a")
CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/4")
SHARED_SERIES = CACHE_DIR / "series.json"
DEFAULT_MANIFEST = CACHE_DIR / "manifest_4a.json"
OUT_STEM = "training_psd"
DEFAULT_SEED = 0
VARIANTS = ("baseline", "paper")
DEFAULT_TRAIN_EPISODES = 150
# Paper-silent display convention: ship PNG uses a rolling mean so single-seed
# roughness matches digitized paper (~std Δ≈0.004). Gates / series.json stay raw.
DISPLAY_ROLL_WINDOW = 10

# Replication traces; paper overlays use lightened dashed versions of these colors.
REPL_BASELINE_COLOR = "#1f77b4"
REPL_SEA_COLOR = "#d62728"


def parse_overrides(items: list[str] | None) -> tuple[tuple[str, Any], ...]:
    """Typed ``--set FIELD=VALUE`` overrides for ``SEADBSConfig`` (applied to every variant)."""
    from dataclasses import fields

    from controllers.sea_dbs.config import SEADBSConfig

    types = {f.name: f.type for f in fields(SEADBSConfig)}
    defaults = SEADBSConfig()
    out: list[tuple[str, Any]] = []
    for item in items or []:
        name, sep, raw = item.partition("=")
        if not sep or name not in types or name in {"variant", "seed"}:
            raise SystemExit(f"--set expects FIELD=VALUE with a shared SEADBSConfig field, got {item!r}")
        current = getattr(defaults, name)
        if isinstance(current, bool):
            if raw.lower() not in {"true", "false"}:
                raise SystemExit(f"--set {name}: expected true/false, got {raw!r}")
            value: Any = raw.lower() == "true"
        elif raw.lower() == "none":
            value = None
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float) or current is None:
            value = float(raw)
        else:
            value = raw
        out.append((name, value))
    return tuple(out)


def train_variant(
    variant: str,
    *,
    seed: int,
    smoke: bool,
    num_episodes: int | None,
    overrides: tuple[tuple[str, Any], ...] = (),
    resume_path: Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
) -> dict[str, Any]:
    cfg = replace(fig4_ravivarapu_config(seed=seed, variant=variant), **dict(overrides))
    if smoke:
        cfg = cfg.for_smoke(episodes=3, max_steps=5)
    elif num_episodes is not None:
        cfg = replace(cfg, num_episodes=num_episodes)

    ckpt = CACHE_DIR / f"{variant}_train{seed}.pt"
    if resume_path is not None:
        ckpt = resume_path

    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        resume_start = 0
        if resume_path is not None:
            payload = load_checkpoint(resume_path, device=cfg.device)
            saved_cfg = config_from_dict(payload["sea_dbs_config"])
            from controllers.sea_dbs.checkpoint import infer_sea_dbs_start_episode, validate_resume_config

            metrics_path = resume_path.with_suffix(".metrics.json")
            resume_start = infer_sea_dbs_start_episode(
                payload,
                metrics_path=metrics_path,
                start_episode=start_episode,
            )
            validate_resume_config(saved_cfg, cfg, resume_start=resume_start)
            trainer.actor.load_state_dict(payload["actor_state_dict"])
            trainer.critic.load_state_dict(payload["critic_state_dict"])
            trainer.load_resume_state(payload)

        result = trainer.train_episodes(
            start_episode=resume_start,
            checkpoint_path=ckpt,
            checkpoint_interval=checkpoint_interval,
        )
        save_checkpoint(
            ckpt,
            actor=result.actor,
            critic=result.critic,
            config=cfg,
            predictive_model=result.predictive_model,
            trainer=trainer,
            extra={
                "completed_episodes": len(result.episode_rewards),
                "episode_rewards": result.episode_rewards,
                "episode_psd": result.episode_psd,
                "episode_stim_frac": result.episode_stim_frac,
            },
        )
        return {
            "variant": variant,
            "seed": seed,
            "episode_rewards": result.episode_rewards,
            "episode_psd": result.episode_psd,
            "episode_stim_frac": result.episode_stim_frac,
            "config": asdict(cfg),
            "num_episodes": cfg.num_episodes,
            "smoke": smoke,
            "checkpoint": ckpt.as_posix(),
        }
    finally:
        env.close()


@dataclass(frozen=True)
class _TrainVariantJob:
    variant: str
    seed: int
    smoke: bool
    num_episodes: int | None
    resume_path: str | None
    start_episode: int | None
    checkpoint_interval: int
    overrides: tuple[tuple[str, Any], ...] = ()


def _train_variant_worker(job: _TrainVariantJob) -> dict[str, Any]:
    _parallel_series.bootstrap_worker_threads()
    resume = Path(job.resume_path) if job.resume_path else None
    return train_variant(
        job.variant,
        seed=job.seed,
        smoke=job.smoke,
        num_episodes=job.num_episodes,
        resume_path=resume,
        start_episode=job.start_episode,
        checkpoint_interval=job.checkpoint_interval,
        overrides=job.overrides,
    )


def train_all(
    *,
    seed: int,
    smoke: bool,
    num_episodes: int | None,
    resume: Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
    train_variants: tuple[str, ...] = VARIANTS,
    parallel_series: int = 0,
    overrides: tuple[tuple[str, Any], ...] = (),
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series: dict[str, Any] = {
        "seed": seed,
        "smoke": smoke,
        "overrides": {k: v for k, v in overrides},
        "variants": {},
    }
    cached: dict[str, Any] = {}
    if SHARED_SERIES.is_file():
        try:
            cached = json.loads(SHARED_SERIES.read_text(encoding="utf-8")).get("variants") or {}
        except json.JSONDecodeError:
            cached = {}
    jobs: list[_TrainVariantJob] = []
    job_variants: list[str] = []
    for variant in VARIANTS:
        if variant not in train_variants:
            if variant not in cached:
                raise SystemExit(
                    f"--variants omitted {variant!r} but {SHARED_SERIES} has no cached series"
                )
            print(f"reusing cached variant={variant} from {SHARED_SERIES}", flush=True)
            series["variants"][variant] = cached[variant]
            continue
        print(f"training variant={variant} seed={seed} smoke={smoke}", flush=True)
        variant_resume = resume
        if resume is not None and variant != "paper" and str(resume).find(variant) < 0:
            candidate = CACHE_DIR / f"{variant}_train{seed}.pt"
            variant_resume = candidate if candidate.is_file() else None
        resume_file: Path | None = None
        if variant_resume is not None and variant_resume.is_file():
            resume_file = variant_resume
        jobs.append(
            _TrainVariantJob(
                variant=variant,
                seed=seed,
                smoke=smoke,
                num_episodes=num_episodes,
                resume_path=resume_file.as_posix() if resume_file else None,
                start_episode=start_episode,
                checkpoint_interval=checkpoint_interval,
                overrides=overrides,
            )
        )
        job_variants.append(variant)
    if jobs:
        results = _parallel_series.run_series_parallel(
            jobs,
            _train_variant_worker,
            parallel_series,
        )
        for variant, result in zip(job_variants, results, strict=True):
            series["variants"][variant] = result
    SHARED_SERIES.write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8")
    return series


def evaluate_gates(series: dict[str, Any]) -> dict[str, Any]:
    if series.get("smoke"):
        return {"pass": False, "shape_pass": False, "smoke_override": True}
    baseline = series["variants"]["baseline"]["episode_psd"]
    sea = series["variants"]["paper"]["episode_psd"]
    report = ravivarapu_fig4a_gates(baseline, sea, n_expected=DEFAULT_TRAIN_EPISODES)
    merged = merge_gate_report(report, {"n_episodes": min(len(baseline), len(sea))})
    return ravivarapu_fig4a_attach_tiered_pass(merged)


def plot_series(series: dict[str, Any], png_path: Path, *, promote: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    paper_y: list[np.ndarray] = []
    roll_w = DISPLAY_ROLL_WINDOW
    repl_styles = (
        ("baseline", f"Baseline (DDPG, roll{roll_w})", REPL_BASELINE_COLOR),
        ("paper", f"SEA-DBS (roll{roll_w})", REPL_SEA_COLOR),
    )
    for variant, label, color in repl_styles:
        raw = np.asarray(series["variants"][variant]["episode_psd"], dtype=float)
        episodes = np.arange(1, len(raw) + 1)
        # Raw episodes (what the gates read) drawn faintly under the rolling mean.
        ax.plot(episodes, raw, linewidth=0.6, alpha=0.3, color=color)
        psd = _common.rolling_mean(raw, roll_w)
        ax.plot(episodes, psd, label=label, linewidth=1.6 if variant == "baseline" else 1.8, color=color)
        paper_y.extend([np.asarray(psd, dtype=float), raw])
    paper_overlay_y = _paper_overlay.overlay_ravivarapu_fig4a(ax)
    paper_y.extend(paper_overlay_y[name][0] for name in ("Baseline", "SEA-DBS"))
    y_all = np.concatenate([np.ravel(y) for y in paper_y if y.size])
    if y_all.size:
        ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        pad = max(0.02, 0.05 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_common.linked_png(png_path, _figure_promote, promote=promote), dpi=150)
    plt.close(fig)


def main() -> int:
    global CACHE_DIR, SHARED_SERIES, DEFAULT_MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=None,
        help="Train only these variants; omitted ones are reused from series.json",
    )
    _resume_cli.add_training_resume_args(parser)
    _parallel_series.add_parallel_series_argument(parser)
    _common.add_no_promote_arg(parser)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Override a shared SEADBSConfig field for every variant (repeatable; candidates)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Write checkpoints / series / manifests here instead of {CACHE_DIR} (candidates)",
    )
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    if args.cache_dir is not None:
        CACHE_DIR = args.cache_dir
        SHARED_SERIES = CACHE_DIR / "series.json"
        DEFAULT_MANIFEST = CACHE_DIR / "manifest_4a.json"
        if not args.no_promote:
            raise SystemExit("--cache-dir is for candidates; pass --no-promote too")
    if args.no_promote:
        args.export_notes = False
        args.update_report = False
    _resume_cli.configure_promote_publish(args, _figure_promote)

    t0 = time.time()
    if args.plot_only:
        if not SHARED_SERIES.is_file():
            raise SystemExit(f"missing cache: {SHARED_SERIES}")
        series = json.loads(SHARED_SERIES.read_text(encoding="utf-8"))
    else:
        train_variants = tuple(args.variants) if args.variants else VARIANTS
        series = train_all(
            seed=args.seed,
            smoke=args.smoke,
            num_episodes=args.episodes,
            resume=args.resume,
            start_episode=args.start_episode,
            checkpoint_interval=args.checkpoint_interval,
            train_variants=train_variants,
            parallel_series=args.parallel_series,
            overrides=overrides,
        )

    gates = evaluate_gates(series)
    if args.cache_dir is not None:
        # Candidates keep their plot beside their cache, out of the ship image folder.
        png_path, png_version = Path(CACHE_DIR) / f"{OUT_STEM}.png", None
    else:
        png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    plot_series(series, png_path, promote=not args.no_promote)

    caption = (
        f"Training mean GPi beta PSD vs episode (seed {args.seed}); "
        f"shape_pass={gates['shape_pass']} pass={gates['pass']}; "
        f"Baseline vs full SEA-DBS (PM+GS); one shared training config; "
        f"display roll{DISPLAY_ROLL_WINDOW} (gates on raw)."
    )
    stim = {
        v: series["variants"][v].get("episode_stim_frac") or [] for v in VARIANTS
    }
    manifest = {
        "panel": "4a",
        "seed": args.seed,
        "smoke": args.smoke or series.get("smoke", False),
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "elapsed_s": round(time.time() - t0, 1),
        "caption": caption,
        "series_cache": SHARED_SERIES.as_posix(),
        "display_roll_window": DISPLAY_ROLL_WINDOW,
        "train_config": "fig4_ravivarapu_config (shared, no per-variant knobs)",
        "config_overrides": series.get("overrides") or {},
        "stim_frac_windows": {
            v: {
                "early_0_15": float(np.mean(f[:15])) if len(f) >= 15 else None,
                "mid_40_80": float(np.mean(f[40:80])) if len(f) >= 80 else None,
                "late_120_150": float(np.mean(f[120:150])) if len(f) >= 150 else None,
            }
            for v, f in stim.items()
        },
    }
    manifest = _panel.stamp_manifest(manifest)
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.no_promote and hasattr(_figure_promote, "promote_ravivarapu_4a"):
        _figure_promote.promote_ravivarapu_4a(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")

    return _panel.exit_code(manifest, smoke=bool(getattr(args, "smoke", False)))

if __name__ == "__main__":
    raise SystemExit(main())
