#!/usr/bin/env python3
"""Ravivarapu Fig 4a — training beta PSD vs episode (Baseline vs SEA-DBS).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py --smoke
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

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import SEADBSConfig, fig4_ravivarapu_config
from dataclasses import dataclass, replace

from controllers.sea_dbs.checkpoint import load_checkpoint, save_checkpoint
from controllers.sea_dbs.trainer import SEA_DBSTrainer

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import (  # noqa: E402
    attach_digitization,
    merge_gate_report,
    ravivarapu_fig4a_attach_tiered_pass,
    ravivarapu_fig4a_digitization_gates,
    ravivarapu_fig4a_gates,
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

_OVERLAY = Path(__file__).resolve().parents[2] / "paper_overlay.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_paper_overlay", _OVERLAY)
assert _overlay_spec and _overlay_spec.loader
_paper_overlay = importlib.util.module_from_spec(_overlay_spec)
sys.modules["figure_paper_overlay"] = _paper_overlay
_overlay_spec.loader.exec_module(_paper_overlay)

FIGURES_DIR = Path("figures/ravivarapu/images/4a")
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


def _rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if window <= 1 or y.size == 0:
        return y
    out = np.empty_like(y)
    for i in range(y.size):
        lo = max(0, i - window + 1)
        out[i] = float(y[lo : i + 1].mean())
    return out


def _vault_backed_png(path: Path) -> Path:
    """Write versioned PNGs into the vault-backed main figures tree when possible.

    Worktree checkouts often materialize ``paper.png`` as a real file, so the local
    ``figures/...`` tree is not a vault symlink. Prefer the main-checkout
    ``paper.png`` symlink (via ``promote.REPO_ROOT``) so ``savefig`` + ``--push-kb``
    land bytes where the tracker and Report 3 expect them.
    """
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
                # Prefer vault target for the actual write.
                return vault_target
            return local
        try:
            local.symlink_to(vault_target)
        except OSError:
            return vault_target
        return local
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def train_variant(
    variant: str,
    *,
    seed: int,
    smoke: bool,
    num_episodes: int | None,
    resume_path: Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
) -> dict[str, Any]:
    cfg = fig4_ravivarapu_config(seed=seed, variant=variant)
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
            saved_cfg = SEADBSConfig(**payload["sea_dbs_config"])
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
            },
        )
        return {
            "variant": variant,
            "seed": seed,
            "episode_rewards": result.episode_rewards,
            "episode_psd": result.episode_psd,
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
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series: dict[str, Any] = {"seed": seed, "smoke": smoke, "variants": {}}
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
        return {"pass": True, "shape_pass": True, "smoke_override": True}
    baseline = series["variants"]["baseline"]["episode_psd"]
    sea = series["variants"]["paper"]["episode_psd"]
    report = ravivarapu_fig4a_gates(baseline, sea, n_expected=DEFAULT_TRAIN_EPISODES)
    merged = merge_gate_report(report, {"n_episodes": min(len(baseline), len(sea))})
    return ravivarapu_fig4a_attach_tiered_pass(merged)


def plot_series(series: dict[str, Any], png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    paper_y: list[np.ndarray] = []
    roll_w = DISPLAY_ROLL_WINDOW
    repl_styles = (
        ("baseline", f"Baseline (DDPG, roll{roll_w})", REPL_BASELINE_COLOR),
        ("paper", f"SEA-DBS (roll{roll_w})", REPL_SEA_COLOR),
    )
    for variant, label, color in repl_styles:
        psd = _rolling_mean(series["variants"][variant]["episode_psd"], roll_w)
        episodes = np.arange(1, len(psd) + 1)
        ax.plot(episodes, psd, label=label, linewidth=1.6 if variant == "baseline" else 1.8, color=color)
        paper_y.append(np.asarray(psd, dtype=float))
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
    fig.savefig(_vault_backed_png(png_path), dpi=150)
    plt.close(fig)


def main() -> None:
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
    args = parser.parse_args()
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
        )

    gates = evaluate_gates(series)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    plot_series(series, png_path)

    caption = (
        f"Training mean GPi beta PSD vs episode (seed {args.seed}); "
        f"shape_pass={gates['shape_pass']} pass={gates['pass']}; "
        f"Baseline vs full SEA-DBS (PM+GS); display roll{DISPLAY_ROLL_WINDOW} "
        "(gates on raw)."
    )
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
        "train_config": "fig4_ravivarapu_config_v107",
    }
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if hasattr(_figure_promote, "promote_ravivarapu_4a"):
        _figure_promote.promote_ravivarapu_4a(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
