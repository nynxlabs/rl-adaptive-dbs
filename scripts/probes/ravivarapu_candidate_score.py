#!/usr/bin/env python3
"""Score Ravivarapu Fig 4 candidates on every downstream panel gate set.

Given one or more Fig 4a candidate caches (``--cache-dir`` runs of
``scripts/figures/papers/ravivarapu/4a/plot.py``, one per training seed), this:

- scores Figs 4a / 4b on each seed and, with several caches, on the **seed mean**
  (episode-by-episode mean of the raw series across seeds — a candidate reading
  of the paper's smooth curve, labelled as such; the gate code and thresholds are
  the panel's own, unchanged);
- rolls each seed's checkpoints out on the Fig 5a protocol (50 Hz, plant seed 0,
  10 steps) with argmax actions (the shipped reading) and with hardened
  Gumbel-Softmax / Gumbel-max actions (candidate reading, torch seed 0 — fixed);
- optionally runs the Table II reseed rollout with a partial window carry
  (``--table2-keep``) on the first cache's checkpoints.

Writes ``score.json`` (and, for a seed mean, ``series.json`` + versioned PNGs)
under ``--out``.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_candidate_score.py \\
      --out artifacts/figures/papers/ravivarapu/candidates/<name>_mean \\
      artifacts/figures/papers/ravivarapu/candidates/<name>_s{0,1,2}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from rl_adaptive_dbs.panel import load_script_module

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "digitization"))

from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, INFERENCE_CARRIER_50HZ, SEADBSConfig  # noqa: E402
from controllers.sea_dbs.eval import (  # noqa: E402
    TABLE2_RESEED_INTERVALS,
    evaluate_ablation_steps,
    evaluate_reseeded,
)
from ravivarapu_gates import ravivarapu_inference_gates, ravivarapu_table2_gates  # noqa: E402

PANELS = ROOT / "scripts" / "figures" / "papers" / "ravivarapu"
_p4a = load_script_module("ravi_score_4a", PANELS / "4a" / "plot.py")
_p4b = load_script_module("ravi_score_4b", PANELS / "4b" / "plot.py")
_pt2 = load_script_module("ravi_score_t2", PANELS / "table2" / "plot.py")
_promote = load_script_module("ravi_score_promote", ROOT / "scripts" / "figures" / "papers" / "promote.py")
VARIANTS = ("baseline", "paper")


_NOT_GATES = {"pass", "shape_pass", "smoke_override"}


def _count(gates: dict[str, Any]) -> tuple[int, int, list[str]]:
    src = gates.get("gates") if isinstance(gates.get("gates"), dict) else gates
    g = {k: v for k, v in src.items() if isinstance(v, (bool, np.bool_)) and k not in _NOT_GATES}
    fails = sorted(k for k, v in g.items() if not v)
    return len(g) - len(fails), len(g), fails


def _windows(x: list[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    return {
        "early": float(a[:15].mean()),
        "mid": float(a[40:80].mean()),
        "midlate": float(a[70:110].mean()),
        "late": float(a[120:150].mean()),
    }


def mean_series(series_list: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"seed": "mean", "smoke": False, "variants": {}}
    out["seeds"] = [s.get("seed") for s in series_list]
    out["overrides"] = series_list[0].get("overrides") or {}
    for v in VARIANTS:
        entries = [s["variants"][v] for s in series_list]
        out["variants"][v] = {
            "variant": v,
            "config": entries[0].get("config") or {},
            **{
                key: np.mean([e[key] for e in entries], axis=0).tolist()
                for key in ("episode_psd", "episode_rewards", "episode_stim_frac")
            },
        }
    return out


def score_fig4(series: dict[str, Any]) -> dict[str, Any]:
    g4a = _p4a.evaluate_gates(series)
    g4b = _p4b.evaluate_gates(series)
    p4a, n4a, f4a = _count(g4a)
    p4b, n4b, f4b = _count(g4b)
    return {
        "4a": f"{p4a}/{n4a}",
        "4b": f"{p4b}/{n4b}",
        "4a_fails": f4a,
        "4b_fails": f4b,
        "psd_windows": {v: _windows(series["variants"][v]["episode_psd"]) for v in VARIANTS},
        "stim_windows": {v: _windows(series["variants"][v]["episode_stim_frac"]) for v in VARIANTS},
        "pearson": {
            "baseline": (g4a.get("gate_metrics") or {}).get("pearson_baseline"),
            "sea": (g4a.get("gate_metrics") or {}).get("pearson_sea"),
        },
    }


def score_5a(cache: Path, seed: int, action_mode: str) -> dict[str, Any]:
    traces: dict[str, list[float]] = {}
    actions: dict[str, list[int]] = {}
    for v in VARIANTS:
        out = evaluate_ablation_steps(
            cache / f"{v}_train{seed}.pt",
            config=SEADBSConfig(variant=v, seed=0),
            n_steps=ABLATION_EVAL_STEPS,
            carrier_hz=INFERENCE_CARRIER_50HZ,
            action_mode=action_mode,
        )
        traces[v] = out["p_beta_trajectories"][0]
        actions[v] = out["action_trajectories"][0]
    rep = ravivarapu_inference_gates(
        traces["baseline"], traces["paper"], carrier_hz=INFERENCE_CARRIER_50HZ, n_expected=ABLATION_EVAL_STEPS + 1
    )
    p, n, fails = _count(rep)
    return {"5a": f"{p}/{n}", "fails": fails, "traces": traces, "actions": actions}


def score_table2(cache: Path, seed: int, keep: float) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for n in TABLE2_RESEED_INTERVALS:
        for v in VARIANTS:
            r = evaluate_reseeded(
                cache / f"{v}_train{seed}.pt",
                reseed_every=n,
                config=SEADBSConfig(variant=v, seed=0),
                carrier_hz=INFERENCE_CARRIER_50HZ,
                keep_window_frac=keep,
            )
            r.pop("p_beta_norm", None), r.pop("rewards", None), r.pop("actions", None)
            cells[f"{v}_{n}"] = r
    rows = _pt2._rows(cells)
    rep = ravivarapu_table2_gates(rows)
    p, n_g, fails = _count(rep)
    return {"table2": f"{p}/{n_g}", "fails": fails, "rows": {str(k): v for k, v in rows.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: first cache)")
    parser.add_argument("--no-5a", action="store_true")
    parser.add_argument("--table2-keep", type=float, nargs="*", default=[])
    args = parser.parse_args()
    out_dir = args.out or args.caches[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    series_list = []
    report: dict[str, Any] = {"caches": [c.as_posix() for c in args.caches], "per_seed": {}}
    for cache in args.caches:
        s = json.loads((cache / "series.json").read_text(encoding="utf-8"))
        series_list.append(s)
        seed = int(s["seed"])
        entry: dict[str, Any] = {"cache": cache.as_posix(), **score_fig4(s)}
        if not args.no_5a:
            entry["5a_argmax"] = score_5a(cache, seed, "argmax")
            entry["5a_gumbel"] = score_5a(cache, seed, "gumbel")
        report["per_seed"][str(seed)] = entry
        print(
            f"seed {seed}: 4a {entry['4a']} 4b {entry['4b']}"
            + (
                f" 5a argmax {entry['5a_argmax']['5a']} gumbel {entry['5a_gumbel']['5a']}"
                f" | gumbel actions B {entry['5a_gumbel']['actions']['baseline']}"
                f" S {entry['5a_gumbel']['actions']['paper']}"
                if not args.no_5a
                else ""
            ),
            flush=True,
        )

    if len(series_list) > 1:
        mean = mean_series(series_list)
        (out_dir / "series.json").write_text(json.dumps(mean, indent=2) + "\n", encoding="utf-8")
        report["seed_mean"] = score_fig4(mean)
        png4a, _ = _promote.next_versioned_png(out_dir, "training_psd_seedmean")
        png4b, _ = _promote.next_versioned_png(out_dir, "training_reward_seedmean")
        _p4a.plot_series(mean, png4a, promote=False)
        _p4b.plot_series(mean, png4b, promote=False)
        report["seed_mean"]["png"] = [png4a.as_posix(), png4b.as_posix()]
        sm = report["seed_mean"]
        print(f"seed mean ({len(series_list)} seeds): 4a {sm['4a']} 4b {sm['4b']} fails 4a {sm['4a_fails']}", flush=True)
        print(f"  4b fails {sm['4b_fails']}", flush=True)

    first = args.caches[0]
    seed0 = int(series_list[0]["seed"])
    report["table2"] = {}
    for keep in args.table2_keep:
        t2 = score_table2(first, seed0, keep)
        report["table2"][str(keep)] = t2
        print(f"table2 keep={keep}: {t2['table2']} fails {t2['fails']}", flush=True)
        for n, row in t2["rows"].items():
            print(f"  n={n}: " + " ".join(f"{k}={v:.3f}" for k, v in row.items()), flush=True)

    (out_dir / "score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
