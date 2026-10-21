"""Mehregan et al. Fig 5a / 5b — post-train efficacy (shared by both panels).

§IV.A.2: after training, the model runs on a fixed seed — 2 s reset, then five
2 s steps of the trained policy — next to fixed periodic stimulation on the same
seed. Fig 5a (45 Hz): PD no stim, fully trained, periodic 45 Hz, periodic
130 Hz. Fig 5b (30 Hz): PD no stim, fully trained, periodic 30 Hz.

The trained series is the paper's own model: Fig 5a evaluates the Fig 4a
checkpoint; Fig 5b trains its 30 Hz model with the same recipe
(``paper_protocol.train_paper``). The policy runs closed loop: each step plays
``argmax`` of the actor on the previous step's $P_\\beta$.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from envs.plant.dbs import DbsSpec
from rl_adaptive_dbs import doc_paths
from rl_adaptive_dbs import panel as _panel
from rl_adaptive_dbs.panel import load_script_module

_HERE = Path(__file__).resolve().parent
_PP = load_script_module("mehregan_paper_protocol", _HERE / "paper_protocol.py")

_DIG = _HERE.parents[2] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig5_efficacy_gates  # noqa: E402

_overlay_import = load_script_module("figure_overlay_import", _HERE.parent / "overlay_import.py")
_paper_overlay = _overlay_import.load_paper_overlay()
_figure_promote = load_script_module("figure_promote", _HERE.parent / "promote.py")
_resume_cli = load_script_module("figure_resume_cli", _HERE.parent / "resume_cli.py")

LATE_LO_S = 4.0  # paper-gate window (digitized late means use t >= 4 s)
STIM_ONSET_S = _PP.ONSET_DISPLAY_S
TIME_MAX_S = _PP.DISPLAY_S



@dataclass(frozen=True)
class EfficacyPanel:
    panel: str
    mean_hz: float
    out_stem: str
    default_checkpoint: Path
    trains_own_model: bool
    with_cdbs: bool
    trained_label: str
    periodic_label: str
    y_min: float
    y_max: float

    @property
    def figures_dir(self) -> Path:
        return Path(doc_paths.figure_images_rel("mehregan", self.panel))

    @property
    def cache_dir(self) -> Path:
        return Path(f"artifacts/figures/papers/mehregan/{self.panel}")


FIG5A = EfficacyPanel(
    panel="5a",
    mean_hz=45.0,
    out_stem="efficacy_45hz",
    default_checkpoint=Path("artifacts/figures/papers/mehregan/4a/checkpoint.pt"),
    trains_own_model=False,
    with_cdbs=True,
    trained_label="Fully Trained 45Hz",
    periodic_label="Periodic 45Hz",
    y_min=100.0,
    y_max=600.0,
)

FIG5B = EfficacyPanel(
    panel="5b",
    mean_hz=30.0,
    out_stem="efficacy_30hz",
    default_checkpoint=Path("artifacts/figures/papers/mehregan/5b/checkpoint.pt"),
    trains_own_model=True,
    with_cdbs=False,
    trained_label="Fully Trained 30Hz",
    periodic_label="Periodic 30Hz",
    y_min=300.0,
    y_max=700.0,
)


def _series_style(cfg: EfficacyPanel) -> dict[str, dict[str, Any]]:
    return {
        "no_stim": {"label": "PD no stim", "color": "#111111", "lw": 1.5, "z": 1},
        "cdbs_130": {"label": "Periodic 130Hz", "color": "#bcbd22", "lw": 1.5, "z": 2},
        "periodic": {"label": cfg.periodic_label, "color": "#ff7f0e", "lw": 1.5, "z": 3},
        "trained": {"label": cfg.trained_label, "color": "#2ca02c", "lw": 2.2, "z": 4},
    }


_LINKED = load_script_module("figure_linked", Path(__file__).resolve().parents[1] / "_linked.py")


def _linked_png(path: Path) -> Path:
    """New versioned PNGs follow the panel folder's notes link (``papers/_linked.py``)."""
    return _LINKED.linked_png(path)


def run_eval(cfg: EfficacyPanel, *, checkpoint: Path, seed: int, knobs: Any) -> dict[str, Any]:
    alphabet = _PP.make_alphabet(cfg.mean_hz, knobs)
    periodic = _PP.regular_spec(alphabet)
    if _PP.is_perpulse_checkpoint(checkpoint):
        # Per-pulse candidate (deviates from the paper): play slot arrays.
        from controllers.ddpg.action_designs import SlotAlphabet

        actor, design, payload = _PP.load_perpulse(checkpoint)
        policy = _PP.perpulse_policy(actor, design, state_len=payload["state_len"], knobs=payload["knobs"])
        alphabet = SlotAlphabet(cfg.mean_hz, step_s=_PP.STEP_S, dt_ms=_PP.PAPER_DT_MS)
    else:
        policy = _PP.module_policy(_PP.load_fp32_actor(checkpoint))
    t0 = time.time()
    conditions: dict[str, dict[str, Any]] = {
        "no_stim": _PP.run_condition(seed=seed, alphabet=alphabet),
        "periodic": _PP.run_condition(seed=seed, alphabet=alphabet, fixed_spec=periodic),
        "trained": _PP.run_condition(seed=seed, alphabet=alphabet, policy=policy),
    }
    if cfg.with_cdbs:
        conditions["cdbs_130"] = _PP.run_condition(
            seed=seed, alphabet=alphabet, fixed_spec=DbsSpec.from_frequency_hz(130.0)
        )
    print(f"trained actions {conditions['trained']['actions']}", flush=True)
    return {
        "figure": f"mehregan_fig{cfg.panel}",
        "protocol": "paper_closed_loop",
        "mean_hz": cfg.mean_hz,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "knobs": knobs.as_dict(),
        "plant_dt_ms": _PP.PAPER_DT_MS,
        "eval_steps": _PP.EVAL_STEPS,
        "conditions": conditions,
        "elapsed_s": round(time.time() - t0, 1),
    }


def panel_means(payload: dict[str, Any]) -> dict[str, Any]:
    conds = payload["conditions"]
    out: dict[str, Any] = {
        f"{k}_mean": _PP.post_mean(v, lo=LATE_LO_S) for k, v in conds.items()
    }
    out.update({f"{k}_pre": _PP.pre_mean(v) for k, v in conds.items()})
    out["trained_actions"] = conds["trained"]["actions"]
    out["trained_closed_loop"] = bool(conds["trained"].get("closed_loop"))
    out["late_lo_s"] = LATE_LO_S
    return out


def payload_traces(payload: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{series: (time_s, P_beta)}`` from an eval payload (``eval.json``)."""
    return {
        k: (np.asarray(v["time_s"], dtype=float), np.asarray(v["trace"], dtype=float))
        for k, v in payload["conditions"].items()
    }


def efficacy_gates(
    cfg: EfficacyPanel,
    means: dict[str, Any],
    traces: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Panel verdict from late means + the eval traces (:func:`payload_traces`).

    The trace-level gates (shared baseline across every series, pre-onset level,
    onset speed, shape r, trained − periodic gap) need ``traces``; without them
    the verdict fails on ``traces_present``.
    """
    required = ("no_stim_mean", "trained_mean", "periodic_mean")
    if any(means.get(k) is None for k in required):
        # Manifest from before the closed-loop eval — re-run the panel.
        return {"means_present": False, "pass": False}
    dig = fig5_efficacy_gates(
        {
            "no_stim": means["no_stim_mean"],
            "trained": means["trained_mean"],
            "periodic": means["periodic_mean"],
            "cdbs_130": means.get("cdbs_130_mean"),
        },
        traces=traces,
        panel=cfg.panel,
        onset=STIM_ONSET_S,
        late_lo=LATE_LO_S,
        t_end=TIME_MAX_S,
        require_cdbs=cfg.with_cdbs,
    )
    gates = dict(dig["gates"])
    if traces is None:
        gates["traces_present"] = False
    gates["trained_closed_loop"] = bool(means.get("trained_closed_loop"))
    return {
        **gates,
        "pass": all(gates.values()),
        "paper_gate_metrics": dig.get("metrics"),
        "paper_ref": dig.get("paper_ref"),
    }


def plot_panel(cfg: EfficacyPanel, payload: dict[str, Any], *, out_path: Path) -> dict[str, Any]:
    _paper_overlay.apply_figure_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    style = _series_style(cfg)
    flat: list[float] = []
    for key in ("no_stim", "cdbs_130", "periodic", "trained"):
        cond = payload["conditions"].get(key)
        if cond is None:
            continue
        s = style[key]
        ax.plot(
            cond["time_s"], cond["trace"], color=s["color"], linewidth=s["lw"],
            zorder=s["z"], label=s["label"],
        )
        flat.extend(cond["trace"])
    ax.axvline(STIM_ONSET_S, color="#888888", linestyle="--", linewidth=1.2, zorder=0)
    overlay = getattr(_paper_overlay, f"overlay_mehregan_fig{cfg.panel}")
    overlay(ax)
    lo = min(cfg.y_min, float(np.floor((min(flat) - 20.0) / 50.0) * 50.0))
    hi = max(cfg.y_max, float(np.ceil((max(flat) + 20.0) / 50.0) * 50.0))
    ax.set_xlim(0.0, TIME_MAX_S)
    ax.set_ylim(lo, hi)
    ax.set_xticks(np.arange(0.0, TIME_MAX_S + 1e-9, 2.0))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("PSD")
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    # Legend outside the plot, on the right: the traces fill the whole plot area.
    _paper_overlay.place_legend(
        ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8, ncol=1
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    means = panel_means(payload)
    return {"out": str(out_path), "y_min": lo, "y_max": hi, **means}


def main(cfg: EfficacyPanel, doc: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--seed", type=int, default=0, help="Eval seed (fixed, shared by all series)")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=cfg.default_checkpoint)
    parser.add_argument("--eval-json", type=Path, default=cfg.cache_dir / "eval.json")
    parser.add_argument("--manifest", type=Path, default=cfg.cache_dir / "manifest.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    if cfg.trains_own_model:
        parser.add_argument("--series", type=Path, default=cfg.cache_dir / "series.json")
        parser.add_argument(
            "--train",
            action="store_true",
            help=f"Train the {cfg.mean_hz:g} Hz model (paper Alg. 1) into --checkpoint first",
        )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="FIELD=VALUE",
        help="Paper-silent knob override (repeatable); must match the checkpoint's training",
    )
    parser.add_argument("--no-update-docs", dest="update_docs", action="store_false")
    parser.set_defaults(update_docs=True)
    _resume_cli.add_export_notes_arg(parser)
    args = parser.parse_args()
    _resume_cli.configure_promote_publish(args, _figure_promote)

    if args.out is None:
        cfg.figures_dir.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(cfg.figures_dir, cfg.out_stem)
        args.out = _linked_png(args.out)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)

    training: dict[str, Any] | None = None
    if args.plot_only:
        if not args.eval_json.exists():
            print(f"missing eval JSON: {args.eval_json}", file=sys.stderr)
            return 2
        payload = json.loads(args.eval_json.read_text())
    else:
        knobs = _PP.apply_overrides(_PP.PaperKnobs(), args.overrides)
        if cfg.trains_own_model and args.train:
            record = _PP.train_paper(
                mean_hz=cfg.mean_hz,
                seed=args.train_seed,
                knobs=knobs,
                checkpoint_path=args.checkpoint,
                extra={"figure": f"mehregan_fig{cfg.panel}", "png_version": png_version},
            )
            args.series.parent.mkdir(parents=True, exist_ok=True)
            args.series.write_text(json.dumps(record, indent=2) + "\n")
            shutil.copyfile(
                args.checkpoint,
                args.checkpoint.with_name(f"{args.checkpoint.stem}_v{png_version}.pt"),
            )
            training = {
                "series": str(args.series),
                "episode_rewards": record["episode_rewards"],
                "unique_actions": record["unique_actions"],
                "action_counts": record["action_counts"],
            }
        if not args.checkpoint.exists():
            print(f"missing checkpoint: {args.checkpoint}", file=sys.stderr)
            return 2
        payload = run_eval(cfg, checkpoint=args.checkpoint, seed=args.seed, knobs=knobs)
        if training is not None:
            payload["training"] = training
        args.eval_json.parent.mkdir(parents=True, exist_ok=True)
        args.eval_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.eval_json}", flush=True)

    panel = plot_panel(cfg, payload, out_path=args.out)
    gates = efficacy_gates(cfg, panel, payload_traces(payload))
    panel["gates"] = gates
    manifest = {
        "figure": f"mehregan_fig{cfg.panel}",
        "protocol": payload.get("protocol"),
        "mean_hz": cfg.mean_hz,
        "seed": payload.get("seed"),
        "knobs": payload.get("knobs"),
        "plant_dt_ms": payload.get("plant_dt_ms"),
        "sampling": "trailing",
        "eval_json": str(args.eval_json),
        "checkpoint": payload.get("checkpoint"),
        "output_png": _figure_promote.repo_rel_posix(args.out),
        "png_version": png_version,
        "training": payload.get("training"),
        "panel": panel,
        "gates": gates,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = _panel.stamp_manifest(manifest)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = " ".join(
        f"{k}={panel[f'{k}_mean']:.1f}" for k in payload["conditions"] if f"{k}_mean" in panel
    )
    print(f"wrote {args.out}\nwrote {args.manifest}", flush=True)
    print(f"gates pass={gates['pass']} {summary} actions={panel['trained_actions']}", flush=True)
    print({k: v for k, v in gates.items() if isinstance(v, bool)}, flush=True)

    if args.update_docs:
        promote = getattr(_figure_promote, f"promote_{cfg.panel}")
        updated = promote(
            manifest=manifest, eval_path=args.eval_json, png_path=args.out, update_docs=True
        )
        print(f"updated docs caption: {updated.get('caption')}", flush=True)
    return 0 if gates["pass"] else 1
