"""Run Nguyen Fig 4 trains for several variants × seeds, then gate Figs 4–6 on each.

Diagnostic driver for multi-seed comparisons (docs/figures/nguyen/4.md): every job
trains ``scripts/figures/papers/nguyen/4/plot.py`` with ``--set`` overrides into its own
candidate folder, then replots Figs 5 and 6 from that series with ``--plot-only``.
Nothing touches the panel's official series, manifests, tracker or PNG versions.

    uv run python -m rl_adaptive_dbs.run scripts/probes/nguyen_seed_batch.py \
        --batch audit1 --seeds 0-7 --parallel 8 \
        --variant base --variant "tu5:subthreshold_steps_required=5"

Variant syntax: ``NAME`` or ``NAME:FIELD=VALUE,FIELD=VALUE``. ``--summary-only``
re-reads finished jobs and prints the per-variant table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "figures" / "papers" / "nguyen"


def _main_checkout() -> Path:
    """Candidate outputs live in the main checkout so they outlive a worktree."""
    common = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent


CAND_ROOT = _main_checkout() / "artifacts" / "figures" / "papers" / "nguyen" / "4" / "candidates"


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_variant(text: str) -> tuple[str, list[str]]:
    name, _, rest = text.partition(":")
    sets = [s for s in rest.split(",") if s]
    return name, sets


def run_job(
    batch: str,
    name: str,
    sets: list[str],
    seed: int,
    episodes: int,
    *,
    fig7: bool = False,
    fig7_sets: list[str] | None = None,
) -> dict:
    out = CAND_ROOT / f"{batch}-{name}-s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    # Hard per-run cap: a train takes ~10–15 min at 8 in parallel, an eval ~3 min.
    py = ["timeout", "3600", sys.executable, "-m", "rl_adaptive_dbs.run", "--max-threads", "1"]
    log = out / "train.log"
    if not (out / "series.json").is_file():
        cmd = py + [
            str(SCRIPTS / "4" / "plot.py"),
            "--seed", str(seed),
            "--episodes", str(episodes),
            "--series", str(out / "series.json"),
            "--checkpoint", str(out / "checkpoint.pt"),
            "--manifest", str(out / "manifest.json"),
            "--out", str(out / "fig4.png"),
            "--no-update-docs",
            "--allow-repeat",
        ]
        for s in sets:
            cmd += ["--set", s]
        with log.open("w") as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=False)
    for panel in ("5", "6"):
        if (out / f"fig{panel}_manifest.json").is_file() or not (out / "series.json").is_file():
            continue
        cmd = py + [
            str(SCRIPTS / panel / "plot.py"),
            "--plot-only",
            "--series", str(out / "series.json"),
            "--fig4-manifest", str(out / "manifest.json"),
            "--manifest", str(out / f"fig{panel}_manifest.json"),
            "--out", str(out / f"fig{panel}.png"),
            "--no-update-docs",
        ]
        with (out / f"fig{panel}.log").open("w") as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if fig7 and (out / "checkpoint.pt").is_file() and not (out / "fig7_manifest.json").is_file():
        cmd = py + [
            str(SCRIPTS / "7" / "plot.py"),
            "--checkpoint", str(out / "checkpoint.pt"),
            "--fig4-manifest", str(out / "manifest.json"),
            "--eval-cache", str(out / "eval7.json"),
            "--manifest", str(out / "fig7_manifest.json"),
            "--out", str(out / "fig7.png"),
            "--no-update-docs",
        ]
        for s in fig7_sets or []:
            cmd += ["--set", s]
        with (out / "fig7.log").open("w") as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return summarize_job(out)


def _progress_ep(series: np.ndarray, start: float, end: float, frac: float) -> float | None:
    """First episode where the 20-ep moving average has covered ``frac`` of start→end."""
    w = 20
    sm = np.convolve(series, np.ones(w) / w, mode="valid")
    span = end - start
    if abs(span) < 1e-9:
        return None
    prog = (sm - start) / span
    hit = np.nonzero(prog >= frac)[0]
    return float(hit[0] + w // 2) if hit.size else None


def summarize_job(out: Path) -> dict:
    rec: dict = {"dir": str(out)}
    try:
        s = json.loads((out / "series.json").read_text())
    except (OSError, json.JSONDecodeError):
        rec["missing"] = True
        return rec
    m4 = json.loads((out / "manifest.json").read_text())
    rec["fig4"] = bool(m4.get("gates_pass"))
    fails = list(m4["gates"]["reward"].get("failed", [])) + list(m4["gates"]["length"].get("failed", []))
    rec["fig4_failed"] = fails
    for panel in ("5", "6", "7"):
        p = out / f"fig{panel}_manifest.json"
        if p.is_file():
            mp = json.loads(p.read_text())
            g = mp.get("gates", {})
            rec[f"fig{panel}"] = bool(mp.get("gates_pass"))
            rec[f"fig{panel}_failed"] = list(g.get("failed", []))
    p7 = out / "fig7_manifest.json"
    if p7.is_file():
        mt = json.loads(p7.read_text()).get("gates", {}).get("paper_gate_metrics", {})
        rec["fig7_mid"] = mt.get("mid_mean")
        rec["fig7_late"] = mt.get("late_mean")
        rec["fig7_start"] = mt.get("start_val")
    L = np.asarray(s["episode_lengths"], float)
    R = np.asarray(s["episode_rewards"], float)
    late = slice(350, 500)
    rec.update(
        start_len=float(np.median(L[:50])),
        late_len=float(L[late].mean()),
        late150_len=float(L[150:].mean()),
        t10=_progress_ep(L, L[:50].mean(), L[150:].mean(), 0.1),
        t50=_progress_ep(L, L[:50].mean(), L[150:].mean(), 0.5),
        t90=_progress_ep(L, L[:50].mean(), L[150:].mean(), 0.9),
        early_reward=float(R[:50].mean()),
        late_reward=float(R[150:].mean()),
        late350_reward=float(R[late].mean()),
        max_reward=float(R.max()),
        late_pw=float(np.mean(s["episode_pulse_widths"][350:])),
        late_freq=float(np.mean(s["episode_frequencies"][350:])),
        late_amp=float(np.mean(s["episode_amplitudes"][350:])),
        late_energy=float(np.mean(s["episode_energies"][350:])),
        late_ab=float(np.mean(s["episode_alpha_beta_means"][350:])),
        late_timeouts=float(np.mean(~np.asarray(s["episode_early_stops"][150:], bool))),
    )
    return rec


def table(results: dict[str, list[dict]]) -> str:
    cols = [
        ("fig4", "F4"), ("fig46", "F4–6"), ("t10", "t10"), ("t50", "t50"), ("t90", "t90"),
        ("late_len", "len350"), ("late_pw", "pw"), ("late_freq", "Hz"), ("late_amp", "amp"),
        ("late_energy", "E"), ("late_ab", "αβ"), ("late_reward", "R150"), ("max_reward", "Rmax"),
        ("fig7", "F7"), ("fig7_mid", "F7 mid"), ("fig7_late", "F7 late"),
    ]
    lines = ["| variant | n | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for name, recs in results.items():
        recs = [r for r in recs if not r.get("missing")]
        if not recs:
            lines.append(f"| {name} | 0 |" + " |" * len(cols))
            continue
        row = [name, str(len(recs))]
        for key, _ in cols:
            if key == "fig4":
                row.append(str(sum(r["fig4"] for r in recs)))
            elif key == "fig7":
                row.append(str(sum(bool(r.get("fig7")) for r in recs)))
            elif key == "fig46":
                row.append(str(sum(r["fig4"] and r.get("fig5") and r.get("fig6") for r in recs)))
            else:
                vals = [r[key] for r in recs if r.get(key) is not None]
                if not vals:
                    row.append("—")
                elif key in ("late_pw",):
                    row.append(f"{np.mean(vals):.2f}")
                elif key in ("late_len",):
                    row.append(f"{np.mean(vals):.1f}")
                elif key in ("late_reward", "max_reward"):
                    row.append(f"{np.mean(vals) / 1e3:+.1f}k")
                else:
                    row.append(f"{np.mean(vals):.0f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("BATCH_PARALLEL", "4")))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--fig7", action="store_true", help="also run the 50-episode Fig 7 eval per job")
    ap.add_argument("--fig7-set", action="append", default=[], metavar="FIELD=VALUE",
                    help="eval-only SNNConfig override for the Fig 7 eval (repeatable)")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    variants = [parse_variant(v) for v in args.variant]
    results: dict[str, list[dict]] = {name: [] for name, _ in variants}
    if args.summary_only:
        for name, _ in variants:
            for seed in seeds:
                results[name].append(summarize_job(CAND_ROOT / f"{args.batch}-{name}-s{seed}"))
    else:
        jobs = [(name, sets, seed) for seed in seeds for name, sets in variants]
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(run_job, args.batch, n, st, sd, args.episodes, fig7=args.fig7, fig7_sets=args.fig7_set): n for n, st, sd in jobs}
            for fut, name in futs.items():
                rec = fut.result()
                results[name].append(rec)
                print(f"done {rec['dir']}: fig4={rec.get('fig4')} fig5={rec.get('fig5')} fig6={rec.get('fig6')}", flush=True)
    summary = CAND_ROOT / f"{args.batch}-summary.json"
    summary.write_text(json.dumps(results, indent=2) + "\n")
    print(table(results))
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
