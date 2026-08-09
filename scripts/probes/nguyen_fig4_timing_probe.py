#!/usr/bin/env python3
"""Nguyen Fig 4 timing-shape probe — cached series counterfactuals + config sweep.

Part A loads ``artifacts/figures/papers/nguyen/4/series.json`` and tests whether
a **left-shifted** convergence (pad early exploration, trim late tail) would pass
``fig4_timing_shape_gates`` — diagnostic only, not a ship shortcut.

Part B runs short DSQN trains (default 150 ep) over ε-decay / shaping knobs and
scores ``shape_pass`` plus timing gate metrics via the panel ``evaluate_gates``.

Examples::

  uv run python -m rl_adaptive_dbs.run scripts/probes/nguyen_fig4_timing_probe.py --series-only
  uv run python -m rl_adaptive_dbs.run scripts/probes/nguyen_fig4_timing_probe.py --quick
  uv run python -m rl_adaptive_dbs.run scripts/probes/nguyen_fig4_timing_probe.py --shaping
  uv run python -m rl_adaptive_dbs.run scripts/probes/nguyen_fig4_timing_probe.py --episodes 150
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "digitization"))

from controllers.snn.adapter import NguyenEnvAdapter  # noqa: E402
from controllers.snn.buffer import ReplayBuffer  # noqa: E402
from controllers.snn.config import fig4_nguyen_config  # noqa: E402
from controllers.snn.networks import DSQN  # noqa: E402
from controllers.snn.trainer import DSQNTrainer  # noqa: E402
from nguyen_gates import fig4_timing_shape_gates  # noqa: E402

_PLOT = ROOT / "scripts" / "figures" / "papers" / "nguyen" / "4" / "plot.py"
_spec = importlib.util.spec_from_file_location("nguyen_4_plot", _PLOT)
assert _spec and _spec.loader
_plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plot)

DEFAULT_SERIES = ROOT / "artifacts/figures/papers/nguyen/4/series.json"
OUT_DIR = ROOT / "artifacts/probes"
TIMING_KEYS = (
    "length_mid_glide_like_paper",
    "length_post100_plateau",
    "reward_post100_plateau",
)


def _load_series(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _series_payload(
    rewards: np.ndarray,
    lengths: np.ndarray,
    *,
    template: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": template.get("seed", 0),
        "num_episodes": int(rewards.size),
        "max_episode_steps": int(template.get("max_episode_steps", 25)),
        "episode_rewards": rewards.tolist(),
        "episode_lengths": lengths.astype(int).tolist(),
        "smoke": False,
    }


def _timing_failures(gates: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    for group in ("reward", "length"):
        g = gates.get(group) or {}
        for key in TIMING_KEYS:
            if key in g and g[key] is False:
                fails.append(f"{group}.{key}")
    return fails


def _phase_stats(lengths: np.ndarray, rewards: np.ndarray) -> dict[str, float]:
    w = 20
    kernel = np.ones(w, dtype=float) / float(w)
    ls = np.convolve(lengths.astype(float), kernel, mode="valid")
    rs = np.convolve(rewards.astype(float), kernel, mode="valid")
    xs = np.arange(w - 1, len(lengths), dtype=float)

    def med(a: float, b: float) -> float:
        m = (xs >= a) & (xs < b)
        return float(np.median(ls[m])) if np.any(m) else float("nan")

    return {
        "len_0_50": med(0, 50),
        "len_50_100": med(50, 100),
        "len_100_150": med(100, 150),
        "rew_100_150": float(np.median(rs[(xs >= 100) & (xs < 150)])) if np.any((xs >= 100) & (xs < 150)) else float("nan"),
    }


def evaluate_series(series: dict[str, Any]) -> dict[str, Any]:
    return _plot.evaluate_gates(series, max_episode_steps=int(series.get("max_episode_steps", 25)))


def series_shift_left(
    rewards: np.ndarray,
    lengths: np.ndarray,
    shift: int,
    *,
    max_steps: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad ``shift`` synthetic exploration episodes; drop ``shift`` from the tail."""
    n = int(rewards.size)
    if shift <= 0:
        return rewards.copy(), lengths.copy()
    if shift >= n:
        msg = f"shift {shift} >= n {n}"
        raise ValueError(msg)
    early_rew = float(np.median(rewards[: min(50, n)]))
    pad_rew = np.full(shift, early_rew, dtype=float)
    pad_len = np.full(shift, max_steps, dtype=float)
    return (
        np.concatenate([pad_rew, rewards[shift:]]),
        np.concatenate([pad_len, lengths[shift:]]),
    )


def run_series_diagnostics(series_path: Path) -> list[dict[str, Any]]:
    template = _load_series(series_path)
    rewards = np.asarray(template["episode_rewards"], dtype=float)
    lengths = np.asarray(template["episode_lengths"], dtype=float)
    max_steps = int(template.get("max_episode_steps", 25))
    rows: list[dict[str, Any]] = []

    print("=== Part A: cached series counterfactuals ===", flush=True)
    print(f"source: {series_path}  n={rewards.size}", flush=True)

    baseline_gates = evaluate_series(template)
    timing = fig4_timing_shape_gates(lengths, rewards)
    row0 = {
        "name": "baseline",
        "shift": 0,
        "shape_pass": baseline_gates.get("shape_pass"),
        "timing_fails": _timing_failures(baseline_gates),
        "timing_metrics": timing.get("metrics", {}),
        "phase": _phase_stats(lengths, rewards),
    }
    rows.append(row0)
    print(row0, flush=True)

    for shift in (25, 50, 75, 100, 125):
        if shift >= rewards.size - 120:
            continue
        rw, ln = series_shift_left(rewards, lengths, shift, max_steps=max_steps)
        payload = _series_payload(rw, ln, template=template)
        gates = evaluate_series(payload)
        timing = fig4_timing_shape_gates(ln, rw)
        row = {
            "name": f"shift_left_{shift}",
            "shift": shift,
            "shape_pass": gates.get("shape_pass"),
            "timing_fails": _timing_failures(gates),
            "timing_metrics": timing.get("metrics", {}),
            "phase": _phase_stats(ln, rw),
        }
        rows.append(row)
        print(row, flush=True)

    return rows


def _train_cfg(cfg) -> tuple[list[float], list[int]]:
    env = NguyenEnvAdapter(config=cfg)
    try:
        dsqn = DSQN(cfg)
        buffer = ReplayBuffer(cfg, seed=cfg.seed)
        trainer = DSQNTrainer(dsqn, buffer, cfg)
        result = trainer.train_episodes(env)
        return result.episode_rewards, result.episode_lengths
    finally:
        env.close()


def _eps_ep100_base(episodes: int):
    """v23 default: eps decay 2200 / end 0.05, tu=2, v9 shaping."""
    return fig4_nguyen_config(seed=0, num_episodes=episodes)


def train_variants(
    *,
    episodes: int,
    quick: bool,
    shaping: bool,
) -> list[tuple[str, Any]]:
    if shaping:
        base = _eps_ep100_base(episodes)
        return [
            ("v23_baseline", base),
            ("trunc_800k", replace(base, truncation_penalty=800_000.0)),
            ("trunc_1m", replace(base, truncation_penalty=1_000_000.0)),
            ("prog_2500", replace(base, alpha_beta_progress_coef=2_500.0)),
            ("prog_3000", replace(base, alpha_beta_progress_coef=3_000.0)),
            ("warm_200", replace(base, warm_zone_bonus_coef=200.0)),
            ("tu3", replace(base, subthreshold_steps_required=3)),
        ]

    base = fig4_nguyen_config(seed=0, num_episodes=episodes)
    variants: list[tuple[str, Any]] = [
        ("v23_baseline", base),
        (
            "eps_fast_2000",
            replace(base, epsilon_decay_steps=2_000, epsilon_end=0.10),
        ),
        (
            "eps_fast_2500",
            replace(base, epsilon_decay_steps=2_500, epsilon_end=0.10),
        ),
        (
            "eps_ep100_2200",
            replace(base, epsilon_decay_steps=2_200, epsilon_end=0.05),
        ),
    ]
    if not quick:
        variants.extend(
            [
                (
                    "eps_slow_end005",
                    replace(base, epsilon_decay_steps=3_500, epsilon_end=0.05),
                ),
                (
                    "eps_fast_end005",
                    replace(base, epsilon_decay_steps=2_500, epsilon_end=0.05),
                ),
                (
                    "weaker_prog_1500",
                    replace(
                        base,
                        epsilon_decay_steps=2_500,
                        epsilon_end=0.10,
                        alpha_beta_progress_coef=1_500.0,
                    ),
                ),
                (
                    "strong_trunc_800k",
                    replace(
                        base,
                        epsilon_decay_steps=2_500,
                        epsilon_end=0.10,
                        truncation_penalty=800_000.0,
                    ),
                ),
                (
                    "tu2_fast_eps",
                    replace(
                        base,
                        epsilon_decay_steps=2_500,
                        epsilon_end=0.10,
                        subthreshold_steps_required=2,
                    ),
                ),
            ]
        )
    return variants


def run_train_sweep(
    *,
    episodes: int,
    quick: bool,
    shaping: bool,
) -> list[dict[str, Any]]:
    mode = "shaping" if shaping else ("quick" if quick else "full")
    print(f"\n=== Part B: train sweep ({episodes} ep, mode={mode}) ===", flush=True)
    rows: list[dict[str, Any]] = []
    for name, cfg in train_variants(episodes=episodes, quick=quick, shaping=shaping):
        t0 = time.perf_counter()
        print(f"\n--- {name} ---", flush=True)
        print(
            f"  eps_decay={cfg.epsilon_decay_steps} eps_end={cfg.epsilon_end} "
            f"prog={cfg.alpha_beta_progress_coef} trunc={cfg.truncation_penalty} "
            f"warm={cfg.warm_zone_bonus_coef} tu={cfg.subthreshold_steps_required}",
            flush=True,
        )
        rewards, lengths = _train_cfg(cfg)
        payload = _series_payload(
            np.asarray(rewards, dtype=float),
            np.asarray(lengths, dtype=float),
            template={"seed": cfg.seed, "max_episode_steps": cfg.max_episode_steps},
        )
        gates = evaluate_series(payload)
        timing = fig4_timing_shape_gates(lengths, rewards)
        late_start = min(150, max(51, len(rewards) - 30))
        row = {
            "name": name,
            "episodes": episodes,
            "shape_pass": gates.get("shape_pass"),
            "pass": gates.get("pass"),
            "timing_fails": _timing_failures(gates),
            "timing_metrics": timing.get("metrics", {}),
            "phase": _phase_stats(np.asarray(lengths), np.asarray(rewards)),
            "late_len": float(np.mean(lengths[late_start:])),
            "late_reward": float(np.mean(rewards[late_start:])),
            "early_stops": int(np.sum(np.asarray(lengths) < cfg.max_episode_steps)),
            "time_s": time.perf_counter() - t0,
            "config": {
                "epsilon_decay_steps": cfg.epsilon_decay_steps,
                "epsilon_end": cfg.epsilon_end,
                "alpha_beta_progress_coef": cfg.alpha_beta_progress_coef,
                "truncation_penalty": cfg.truncation_penalty,
                "warm_zone_bonus_coef": cfg.warm_zone_bonus_coef,
                "subthreshold_steps_required": cfg.subthreshold_steps_required,
            },
        }
        rows.append(row)
        print(row, flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--series-only", action="store_true")
    parser.add_argument("--train", action="store_true", help="run Part B train sweep")
    parser.add_argument("--quick", action="store_true", help="4 epsilon train variants only")
    parser.add_argument(
        "--shaping",
        action="store_true",
        help="7-variant shaping sweep on v23 eps_ep100_2200 base",
    )
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out: dict[str, Any] = {"series_path": args.series.as_posix()}
    if args.series_only:
        out["series_diagnostics"] = (
            run_series_diagnostics(args.series) if args.series.is_file() else []
        )
    elif args.shaping:
        out["series_diagnostics"] = []
    elif args.series.is_file():
        out["series_diagnostics"] = run_series_diagnostics(args.series)
    else:
        print(f"missing series: {args.series}", flush=True)
        out["series_diagnostics"] = []

    run_train = args.train or not args.series_only
    if run_train:
        out["train_sweep"] = run_train_sweep(
            episodes=args.episodes,
            quick=args.quick,
            shaping=args.shaping,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.shaping:
        tag = "shaping"
    elif args.quick:
        tag = "quick"
    else:
        tag = "full"
    out_path = args.out or OUT_DIR / f"nguyen_fig4_timing_probe_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)

    if run_train and out.get("train_sweep"):
        def _rank_key(r: dict[str, Any]) -> tuple:
            phase = r.get("phase") or {}
            len_100_150 = phase.get("len_100_150", 99.0)
            return (
                int(bool(r.get("shape_pass"))),
                -len(r.get("timing_fails") or []),
                -float(len_100_150) if isinstance(len_100_150, (int, float)) else -99.0,
            )

        best = sorted(out["train_sweep"], key=_rank_key, reverse=True)
        print("\n=== Train sweep ranking (top 3) ===", flush=True)
        for row in best[:3]:
            print(row["name"], row.get("shape_pass"), row.get("timing_fails"), row.get("phase"), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
