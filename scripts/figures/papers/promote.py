"""Refresh the paper replication docs after paper figure plot scripts run.

Plot scripts write replication PNGs under
``docs/controllers/<controller>/figures/images/<panel>/`` and JSON caches under
``artifacts/figures/papers/``. This module updates the docs those runs feed
(paths: ``rl_adaptive_dbs.doc_paths``):

- each panel's page ``docs/controllers/<controller>/figures/<panel>.md`` —
  ``caption-<id>`` / ``gates-<id>`` marker blocks, the ``**Status:**`` line
  after the caption, and the replication image links (relative to the page,
  ``images/<panel>/…``);
- the paper tracker ``docs/controllers/<controller>/README.md`` — the
  ``summary`` block and panel table.

While panel blocks are still moving out of the tracker, every per-panel edit
looks in the figure page first and falls back to the tracker (``DocSet``).

When run from a git worktree under ``.worktrees/``, shared doc/figure paths resolve to
the **main checkout** (whose docs may be symlinks into a notes folder) so promote does
not leave a detached worktree copy of the docs.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

from rl_adaptive_dbs import doc_paths
from rl_adaptive_dbs.doc_paths import DocSet


def main_checkout_root(checkout: Path | None = None) -> Path:
    """Return the main repo checkout if ``checkout`` is ``.../.worktrees/<name>``."""
    return doc_paths.main_checkout_root(checkout or Path(__file__).resolve().parents[3])


CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = main_checkout_root(CHECKOUT_ROOT)


def resolve_tracker(paper: str, checkout: Path | None = None) -> Path:
    """Tracker ``docs/controllers/<controller>/README.md`` that plot/promote should update.

    Prefers the main checkout path (possibly a symlink into a notes folder) so worktree runs update
    the shared doc the main branch and any notes app see.
    """
    return doc_paths.tracker_path(paper, main_checkout_root(checkout or CHECKOUT_ROOT))


def resolve_figure_page(paper: str, panel: str, checkout: Path | None = None) -> Path:
    """Figure page ``docs/controllers/<controller>/figures/<panel>.md`` (main checkout)."""
    return doc_paths.figure_page(paper, panel, main_checkout_root(checkout or CHECKOUT_ROOT))


def panel_docs(paper: str, panel: str) -> DocSet:
    """Editable ``[figure page, tracker]`` pair for one panel (page first)."""
    return DocSet.for_panel(_tracker_for(paper), panel, root=REPO_ROOT)


def _image_rel(paper: str, panel: str, name: str) -> str:
    """Repo-relative ``docs/controllers/<controller>/figures/images/<panel>/<name>``."""
    return (doc_paths.figure_images_rel(paper, panel) / name).as_posix()


def resolve_paper_1_doc(checkout: Path | None = None) -> Path:
    """Mehregan tracker ``docs/controllers/ddpg/README.md``."""
    return resolve_tracker("mehregan", checkout)


PAPER_1_DOC = resolve_paper_1_doc()


def refresh_mehregan_gate_tables(*, update_docs: bool = True) -> None:
    """Refresh per-panel gate tables (with live Pass column) in the Mehregan pages + tracker."""
    if not update_docs or not PAPER_1_DOC.exists():
        return
    digitization = Path(__file__).resolve().parents[2] / "digitization"
    path = digitization / "mehregan_gate_status.py"
    if not path.is_file():
        return
    spec = importlib.util.spec_from_file_location("mehregan_gate_status", path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    # dataclasses need the module registered before exec when loaded via importlib.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.refresh_gate_tables(PAPER_1_DOC)


def resolve_nguyen_doc(checkout: Path | None = None) -> Path:
    """Nguyen tracker ``docs/controllers/snn/README.md``."""
    return resolve_tracker("nguyen", checkout)


PAPER_NGUYEN_DOC = resolve_nguyen_doc()
# Back-compat alias used by promote_nguyen_2_3
PAPER_2_DOC = PAPER_NGUYEN_DOC
PAPER_NGUYEN_2_3_MANIFEST = "artifacts/figures/papers/nguyen/3/manifest.json"
PAPER_NGUYEN_2_3_REPLICATION_ALT = "Replication Fig 3"
PAPER_NGUYEN_3_PNG = _image_rel("nguyen", "3", "alpha_beta_dist.png")
PAPER_NGUYEN_4_MANIFEST = "artifacts/figures/papers/nguyen/4/manifest.json"
PAPER_NGUYEN_4_REPLICATION_ALT = "Replication Fig 4"  # legacy single-image trackers
PAPER_NGUYEN_4_REPLICATION_BEST_ALT = "Replication Fig 4 — best"
PAPER_NGUYEN_4_REPLICATION_LATEST_ALT = "Replication Fig 4 — latest"
PAPER_NGUYEN_4_PNG = _image_rel("nguyen", "4", "training_reward_length.png")
PAPER_NGUYEN_5_MANIFEST = "artifacts/figures/papers/nguyen/5/manifest.json"
PAPER_NGUYEN_5_REPLICATION_ALT = "Replication Fig 5"
PAPER_NGUYEN_6_MANIFEST = "artifacts/figures/papers/nguyen/6/manifest.json"
PAPER_NGUYEN_6_REPLICATION_ALT = "Replication Fig 6"
PAPER_NGUYEN_7_MANIFEST = "artifacts/figures/papers/nguyen/7/manifest.json"
PAPER_NGUYEN_7_REPLICATION_ALT = "Replication Fig 7"

# Nguyen Fig 4 gate tiers (must match ``nguyen_gates.FIG4_*_KEYS``; a test checks it):
# shape = transition timing / ep 0–200 shape; full = rest of the required set;
# info = tier 2, logged only (values live under ``gates[group]["tier2"]``).
NGUYEN_FIG4_GATE_TIER: dict[str, dict[str, str]] = {
    "reward": {
        "reward_t50_timing": "shape",
        "reward_pearson_0_200": "shape",
        "reward_pearson_0_500": "full",
        "reward_relative_gain": "full",
        "reward_late_plateau_hold": "full",
        "reward_leads_length": "full",
        "reward_scale_raw_d": "info",
    },
    "length": {
        "length_late_level": "full",
        "length_t50_timing": "shape",
        "length_pearson_0_200": "shape",
        "length_start_at_horizon": "shape",
        "length_pearson_0_500": "full",
        "length_ratio": "full",
        "length_late_plateau_hold": "full",
        "late_timeouts": "full",
        "length_late_trend": "full",
    },
}

# Nguyen gate rows: (manifest key, description). ``pass`` = all non-info rows; ``shape_pass`` = shape tier only.
# Tolerances: smallest clean value at which ≥ 90 % of synthetic faithful runs (digitized paper
# curve + one seed's residual) pass — scripts/probes/nguyen_gate_calibration.py.
NGUYEN_GATE_GROUPS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "4": {
        "reward": [
            ("reward_t50_timing", "50 % / 90 % progress ep within ±45 / ±35 of paper 71 / 86"),
            ("reward_pearson_0_200", "smoothed reward vs paper, Pearson r ep 0–200 ≥ 0.938"),
            ("reward_pearson_0_500", "smoothed reward vs paper, Pearson r ep 0–500 ≥ 0.942"),
            ("reward_relative_gain", "(R̄150–500 − R̄0–50) / abs(R̄0–50) within ±0.026 of paper 0.974"),
            ("reward_late_plateau_hold", "min smoothed progress ep 150–500 ≥ paper 0.959 − 0.058"),
            ("reward_leads_length", "10 % progress lead, length t10 − reward t10 ≥ paper 36 − 35 (reward leads)"),
            ("reward_scale_raw_d", "tier 2: first-50 mean within ±35% of paper −487.5k (raw d only)"),
        ],
        "length": [
            ("length_late_level", "raw mean ep 150–500 / 350–500 within ±1.7 / ±1.9 steps of paper 8.57 / 8.19"),
            ("length_t50_timing", "50 % / 90 % progress ep within ±10 / ±60 of paper 76 / 89"),
            ("length_pearson_0_200", "smoothed length vs paper, Pearson r ep 0–200 ≥ 0.941"),
            ("length_start_at_horizon", "raw median ep 0–50 ≥ paper 24.9 − 0.5"),
            ("length_pearson_0_500", "smoothed length vs paper, Pearson r ep 0–500 ≥ 0.925"),
            ("length_ratio", "L̄150–500 / L̄0–50 within ±0.087 of paper 0.344"),
            ("length_late_plateau_hold", "min smoothed progress ep 150–500 ≥ paper 0.922 − 0.30"),
            ("late_timeouts", "timeout share ep 150–500 ≤ 0.86 % (≤ 3 of 350; paper 0)"),
            ("length_late_trend", "L̄350–500 / L̄150–350 within ±0.15 of paper 0.925 (late length keeps falling)"),
        ],
    },
}

NGUYEN_GATE_ROWS: dict[str, list[tuple[str, str]]] = {
    "3": [
        ("ordering_pd_on_above_pd_off", "PD On median above PD Off median"),
        ("paper_pd_off_q1_near_paper", "digitization — PD Off Q1 within ±10% of paper"),
        ("paper_pd_off_median_near_paper", "digitization — PD Off median within ±10% of paper"),
        ("paper_pd_off_q3_near_paper", "digitization — PD Off Q3 within ±10% of paper"),
        ("paper_pd_off_mean_near_paper", "digitization — PD Off mean within ±10% of paper"),
        ("paper_pd_on_q1_near_paper", "digitization — PD On Q1 within ±10% of paper"),
        ("paper_pd_on_median_near_paper", "digitization — PD On median within ±10% of paper"),
        ("paper_pd_on_q3_near_paper", "digitization — PD On Q3 within ±10% of paper"),
        ("paper_pd_on_mean_near_paper", "digitization — PD On mean within ±10% of paper"),
        ("paper_pd_off_q1_marginally_above_theta", "PD Off Q1 in (150, 187.5] — §IV: Q1 \"marginally higher than 150\" (paper 167.8)"),
    ],
    "5": [
        ("shared_train", "Fig 4 passed + same n_episodes"),
        ("paper_spike_trend_near_paper", "digitization — spikes S̄350–500 / S̄0–50 within ±0.099 of paper 1.00"),
        ("paper_energy_pearson_0_500", "digitization — smoothed energy vs paper, Pearson r ep 0–500 ≥ 0.79"),
        ("paper_energy_rise_timing", "digitization — energy 50 % / 90 % rise ep within ±25 / ±40 of paper 73 / 88"),
        ("paper_energy_mid_ramp_near_paper", "digitization — Ē55–75 / Ē0–50 within ±0.23 of paper 1.15"),
        ("paper_energy_trend_near_paper", "digitization — Ē350–500 / Ē0–50 within ±0.60 of paper 1.80"),
        ("paper_energy_overshoot", "digitization — Ē150–350 / Ē350–500 within ±0.087 of paper 1.16 (late overshoot)"),
    ],
    "6": [
        ("shared_train", "Fig 4 passed + shared train"),
        ("paper_alpha_beta_early_above_theta", "digitization — early α–β > θ=150 (ep 27–50)"),
        ("paper_alpha_beta_post100_below_theta", "digitization — α–β ≤ θ=150 (ep 100–250)"),
        ("paper_alpha_beta_late_below_theta", "digitization — α–β ≤ θ=150 (ep 350–500)"),
        ("paper_alpha_beta_early_near_paper", "digitization — α–β ep 27–50 within ±95 of paper 287"),
        ("paper_alpha_beta_mid_near_paper", "digitization — α–β ep 50–100 within ±52 of paper 221"),
        ("paper_alpha_beta_post100_near_paper", "digitization — α–β ep 100–250 within ±6.4 of paper 110"),
        ("paper_alpha_beta_late_near_paper", "digitization — α–β ep 350–500 within ±5.3 of paper 109"),
        ("paper_alpha_beta_mean_near_paper", "digitization — α–β ep 27–500 within ±11 of paper 130"),
        ("paper_alpha_beta_pearson", "digitization — smoothed α–β vs paper, Pearson r ep 27–500 ≥ 0.909"),
        ("paper_alpha_beta_t50_timing", "digitization — α–β 50 % drop ep within ±35 of paper 87"),
        ("paper_alpha_beta_drop_fraction", "digitization — (early − mid) / (early − late) within ±0.34 of paper 0.37"),
        ("paper_alpha_beta_trend_near_paper", "digitization — late / early α–β within ±0.17 of paper 0.38"),
        ("paper_alpha_beta_drop_magnitude_near_paper", "digitization — early − late α–β within ±93 of paper 178"),
        ("paper_alpha_beta_late_cv", "digitization — α–β CV ep 350–500 within ±0.025 of paper 0.16"),
        ("paper_freq_early_near_paper", "digitization — frequency ep 0–50 within ±42 Hz of paper 59.1"),
        ("paper_freq_mid_near_paper", "digitization — frequency ep 50–100 within ±37 Hz of paper 93.7"),
        ("paper_freq_late_near_paper", "digitization — frequency ep 350–500 within ±6.2 Hz of paper 81.3"),
        ("paper_freq_t50_timing", "digitization — frequency 50 % rise ep (from ep 0–10) within ±45 of paper 26"),
        ("paper_freq_pearson", "digitization — smoothed frequency vs paper, Pearson r ep 0–500 ≥ 0.32"),
        ("paper_freq_late_cv", "digitization — frequency CV ep 350–500 within ±0.025 of paper 0.076"),
    ],
    "7": [
        ("checkpoint_lineage_ok", "Fig 4 train passed"),
        ("paper_eval_protocol_ok", "digitization — 50 episodes, each with the reset reading + 25 steps (paper §IV)"),
        ("paper_step_series_finite", "digitization — step series finite"),
        ("paper_start_near_paper", "digitization — step 0 within ±36.4 (paper single-episode SD) of paper 153.2"),
        ("paper_peak_step_timing", "digitization — peak step within ±1 of paper (2.1)"),
        ("paper_peak_power_near_paper", "digitization — peak within ±49.7 (SD at paper peak step) of paper 278.4"),
        ("paper_early_mean_near_paper", "digitization — steps 0–5 mean within ±41.8 (mean SD) of paper 223.0"),
        ("paper_mid_mean_near_paper", "digitization — steps 6–15 mean within ±29.1 (mean SD) of paper 161.4"),
        ("paper_late_mean_near_paper", "digitization — steps 18–25 mean within ±32.2 (mean SD) of paper 152.2"),
        ("paper_overall_mean_near_paper", "digitization — steps 0–25 mean within ±32.9 (mean SD) of paper 172.3"),
        ("paper_peak_to_late_drop", "digitization — peak − late within ±59.2 (SDs in quadrature) of paper 126.2"),
        ("paper_late_peak_ratio_near_paper", "digitization — late / peak within ±0.15 (propagated SD) of paper 0.547"),
        ("paper_pearson_shape_ok", "digitization — mean-trace Pearson r vs paper ≥ 0.80 (paper shifted one step: 0.77)"),
        ("paper_below_fig3_pd_median", "overall mean below Fig 3 PD On (our median and paper ~295)"),
    ],
}

_PENDING = "pending maintainer decision"
_SUPERSEDED = "superseded by the calibrated rows above"

# Logged in the table but never part of ``pass`` (docs/controllers/snn/figures/5.md, 6.md).
NGUYEN_REPORT_ROWS: dict[str, list[tuple[str, str]]] = {
    "5": [
        ("spike_in_paper_band", f"absolute level ({_PENDING}) — mean spikes 400–950/ep"),
        ("energy_in_paper_band", f"absolute level ({_PENDING}) — mean energy 300–3200/ep, max ≤ 3520"),
        ("paper_spike_early_near_paper", f"absolute level ({_PENDING}) — spikes ep 0–50 within ±25 % of paper"),
        ("paper_spike_mid_near_paper", f"absolute level ({_PENDING}) — spikes ep 55–75 within ±25 % of paper"),
        ("paper_spike_late_near_paper", f"absolute level ({_PENDING}) — spikes ep 350–500 within ±25 % of paper"),
        ("paper_spike_mean_near_paper", f"absolute level ({_PENDING}) — spike mean within ±25 % of paper"),
        ("paper_spike_stays_near_800", f"absolute level ({_PENDING}) — spike windows all in 600–1000"),
        ("paper_energy_early_near_paper", f"absolute level ({_PENDING}) — energy ep 0–50 within ±25 % of paper"),
        ("paper_energy_mid_near_paper", f"absolute level ({_PENDING}) — energy ep 55–75 within ±25 % of paper"),
        ("paper_energy_late_near_paper", f"absolute level ({_PENDING}) — energy ep 350–500 within ±25 % of paper"),
        ("paper_energy_mean_near_paper", f"absolute level ({_PENDING}) — energy mean within ±25 % of paper"),
        ("paper_energy_monotonic_rise", f"{_SUPERSEDED} — energy early < mid < late"),
        ("paper_energy_late_above_early", f"{_SUPERSEDED} — energy late > early × 1.25"),
        ("spike_series_has_variance", f"{_SUPERSEDED} — spike series has variance"),
        ("energy_not_constant", f"{_SUPERSEDED} — energy not constant"),
    ],
    "6": [
        ("paper_amp_late_near_paper", f"amplitude ({_PENDING}) — late amplitude within ±25 % of paper"),
        ("paper_pw_late_near_paper", f"pulse width ({_PENDING}) — late pulse width within ±25 % of paper"),
        ("paper_params_left_init", f"amplitude / pulse width ({_PENDING}) — amp / freq / pw each >5% off init"),
        ("paper_late_params_stable", f"amplitude / pulse width ({_PENDING}) — std last 50 ep ≤ 20% of mean"),
        ("paper_amp_left_init", f"amplitude ({_PENDING}) — >5% off init"),
        ("paper_pw_left_init", f"pulse width ({_PENDING}) — >5% off init"),
        ("paper_alpha_beta_drops_by_100", f"{_SUPERSEDED} — α–β ep 50–100 < ep 0–50 − 15"),
        ("paper_alpha_beta_monotonic_drop", f"{_SUPERSEDED} — α–β early > mid > late"),
        ("paper_alpha_beta_series_has_variance", f"{_SUPERSEDED} — α–β series has variance"),
        ("paper_alpha_beta_late_stable", f"{_SUPERSEDED} — α–β std ep 350–500 ≤ 35 % of mean"),
        ("paper_alpha_beta_decreases_like_paper", f"{_SUPERSEDED} — α–β late < early"),
        ("paper_freq_left_init", f"{_SUPERSEDED} — frequency >5% off init"),
        ("paper_freq_late_stable", f"{_SUPERSEDED} — frequency std last 50 ep ≤ 20% of mean"),
    ],
}

NGUYEN_SUMMARY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("3", "Fig 3", "GPi α–β distribution (PD Off vs PD On)"),
    ("4", "Fig 4", "Training reward + episode length"),
    ("5", "Fig 5", "CBGT spikes + DBS energy over training"),
    ("6", "Fig 6", "α–β + DBS parameters over training"),
    ("7", "Fig 7", "50-episode eval (25 steps)"),
)

NGUYEN_MANIFEST_BY_PANEL: dict[str, str] = {
    "3": PAPER_NGUYEN_2_3_MANIFEST,
    "4": PAPER_NGUYEN_4_MANIFEST,
    "5": PAPER_NGUYEN_5_MANIFEST,
    "6": PAPER_NGUYEN_6_MANIFEST,
    "7": PAPER_NGUYEN_7_MANIFEST,
}

PAPER_1B_PNG = _image_rel("mehregan", "1b", "gpi_psd.png")
PAPER_1B_MANIFEST = "artifacts/figures/papers/mehregan/1b/manifest.json"
PAPER_2A_PNG = _image_rel("mehregan", "2a", "beta_power.png")
PAPER_2A_MANIFEST = "artifacts/figures/papers/mehregan/2a/manifest.json"
PAPER_2B_PNG = _image_rel("mehregan", "2b", "error_index.png")
PAPER_2B_MANIFEST = "artifacts/figures/papers/mehregan/2b/manifest.json"
PAPER_4A_PNG = _image_rel("mehregan", "4a", "training_beta.png")
PAPER_4A_MANIFEST = "artifacts/figures/papers/mehregan/4a/manifest.json"
PAPER_4B_PNG = _image_rel("mehregan", "4b", "training_reward.png")
PAPER_4B_COMBINED_PNG = _image_rel("mehregan", "4b", "training_fig4b.png")
PAPER_4B_MANIFEST = "artifacts/figures/papers/mehregan/4b/manifest.json"
PAPER_1B_REF = _image_rel("mehregan", "1b", "digitized_v2.png")
PAPER_2A_REF = _image_rel("mehregan", "2a", "digitized_v2.png")
PAPER_2B_REF = _image_rel("mehregan", "2b", "digitized_v2.png")
PAPER_1B_REPLICATION_ALT = "Replication Fig 1b"
PAPER_2A_REPLICATION_ALT = "Replication Fig 2a"
PAPER_2B_REPLICATION_ALT = "Replication Fig 2b"
PAPER_4A_REF = _image_rel("mehregan", "4a", "digitized_v2.png")
PAPER_4A_REPLICATION_ALT = "Replication Fig 4a"
PAPER_4B_REF = _image_rel("mehregan", "4b", "digitized_v2.png")
PAPER_4B_REPLICATION_ALT = "Replication Fig 4b reward"
PAPER_4B_PSD_REPLICATION_ALT = "Replication Fig 4b PSD"
PAPER_5A_PNG = _image_rel("mehregan", "5a", "efficacy_45hz.png")
PAPER_5A_MANIFEST = "artifacts/figures/papers/mehregan/5a/manifest.json"
PAPER_5A_REPLICATION_ALT = "Replication Fig 5a"
PAPER_5B_PNG = _image_rel("mehregan", "5b", "efficacy_30hz.png")
PAPER_5B_MANIFEST = "artifacts/figures/papers/mehregan/5b/manifest.json"
PAPER_5B_REPLICATION_ALT = "Replication Fig 5b"
PAPER_6A_PNG = _image_rel("mehregan", "6a", "ptq_qat_45hz.png")
PAPER_6A_MANIFEST = "artifacts/figures/papers/mehregan/6a/manifest.json"
PAPER_6A_REF = _image_rel("mehregan", "6a", "digitized_v2.png")
PAPER_6A_REPLICATION_ALT = "Replication Fig 6a"
PAPER_6B_PNG = _image_rel("mehregan", "6b", "ptq_qat_30hz.png")
PAPER_6B_MANIFEST = "artifacts/figures/papers/mehregan/6b/manifest.json"
PAPER_6B_REF = _image_rel("mehregan", "6b", "digitized_v2.png")
PAPER_6B_REPLICATION_ALT = "Replication Fig 6b"

_VERSIONED_PNG_RE = re.compile(r"^(?P<stem>.+)_v(?P<ver>\d+)\.png$")


def latest_png_version(directory: Path, stem: str) -> int:
    """Highest ``{stem}_vN.png`` version in ``directory``, or 0 if none exist."""
    if not directory.is_dir():
        return 0
    best = 0
    for path in directory.iterdir():
        if not path.is_file() and not path.is_symlink():
            continue
        match = _VERSIONED_PNG_RE.match(path.name)
        if match is None or match.group("stem") != stem:
            continue
        best = max(best, int(match.group("ver")))
    return best


def next_versioned_png(directory: Path, stem: str) -> tuple[Path, int]:
    """Allocate the next ``{stem}_vN.png`` path (N = max existing + 1)."""
    version = latest_png_version(directory, stem) + 1
    return directory / f"{stem}_v{version}.png", version


def parse_png_version(path: Path) -> int | None:
    match = _VERSIONED_PNG_RE.match(path.name)
    return int(match.group("ver")) if match else None


def repo_rel_posix(path: Path) -> str:
    """Normalize a figure path to repo-relative POSIX (``docs/controllers/…`` / ``artifacts/…``)."""
    path = Path(path)
    if path.is_absolute():
        for root in (CHECKOUT_ROOT, REPO_ROOT):
            rel = doc_paths.rel_to_root(path, root)
            if rel is not None:
                return rel
    return doc_paths.repo_rel_posix(path)


def _push_notes_image(ship_repo_rel: str) -> None:
    """Copy one ``docs/controllers/<controller>/figures/images/…`` PNG into the notes folder."""
    try:
        push_path = Path(__file__).resolve().parent / "export_notes_images.py"
        spec = importlib.util.spec_from_file_location("export_notes_images", push_path)
        if spec is None or spec.loader is None:
            return
        split = doc_paths.split_figures_rel(ship_repo_rel)
        if split is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        paper, rel = split
        mod.push_image(paper, rel, dry_run=False)
    except Exception:
        pass


def materialize_ship_png(source: Path, ship_repo_rel: str) -> Path:
    """Copy a versioned replication PNG (and its dark variant) to a stable ship path
    (repo + linked notes folder).

    Gallery embeds use unversioned ship names (e.g.
    ``training_beta.png``) so links from notes apps stay valid after each
    ``_vN`` promote. ``ship_repo_rel`` is repo-relative
    (``docs/controllers/<controller>/figures/images/<panel>/<name>.png``).
    """
    source = Path(source).resolve()
    if not source.is_file():
        msg = f"ship png source missing: {source}"
        raise FileNotFoundError(msg)
    ship_repo_rel = ship_repo_rel.replace("\\", "/")
    pairs = [(source, ship_repo_rel)]
    dark_source = Path(dark_rel(str(source)))
    if dark_source.is_file():
        pairs.append((dark_source, dark_rel(ship_repo_rel)))
    for src, rel in pairs:
        repo_dest = REPO_ROOT / rel
        repo_dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != repo_dest.resolve():
            if repo_dest.is_symlink():
                repo_dest.unlink()
            shutil.copy2(src, repo_dest)
        _push_notes_image(rel)
    return REPO_ROOT / ship_repo_rel


def materialize_docs_figure_papers() -> int:
    """No-op: replication PNGs live under ``docs/controllers/<controller>/figures/images/`` only."""
    return 0


# Every plot saves a light PNG and a dark variant beside it (``figure_style.dark_path``:
# ``x_v3.png`` → ``x_v3_dark.png``); the docs show the pair as one ``<picture>``.
DARK_SUFFIX = "_dark"
_DARK_MEDIA = "(prefers-color-scheme: dark)"


def dark_rel(repo_rel: str) -> str:
    """Repo-relative path of the dark variant of a light PNG."""
    path = Path(repo_rel)
    return path.with_name(f"{path.stem}{DARK_SUFFIX}{path.suffix}").as_posix()


def has_dark_variant(repo_rel: str) -> bool:
    """Whether the dark variant of ``repo_rel`` exists (this checkout or main)."""
    rel = dark_rel(repo_rel)
    return any((root / rel).is_file() for root in (CHECKOUT_ROOT, REPO_ROOT))


def _html_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def picture_markup(alt: str, href: str, dark_href: str | None, *, attrs: str = "") -> str:
    """Doc markup for one replication image: a light/dark ``<picture>`` on one line.

    ``attrs`` are extra ``<img>`` attributes (e.g. `` width="280"``). Without a dark
    variant this is a plain ``![alt](href)``.
    """
    if dark_href is None:
        if attrs:
            return f'<img src="{href}" alt="{_html_attr(alt)}"{attrs}>'
        return f"![{alt}]({href})"
    return (
        f'<picture><source media="{_DARK_MEDIA}" srcset="{dark_href}">'
        f'<img src="{href}" alt="{_html_attr(alt)}"{attrs}></picture>'
    )


def _image_pattern(alt: str) -> re.Pattern[str]:
    """``![alt](…)`` or a ``<picture>`` whose ``<img>`` has ``alt`` (``attrs`` group: extras)."""
    md = rf"!\[{re.escape(alt)}\]\([^)]+\)"
    html = (
        rf'<picture><source media="{re.escape(_DARK_MEDIA)}" srcset="[^"]*">'
        rf'<img src="[^"]*" alt="{re.escape(_html_attr(alt))}"(?P<attrs>[^>]*)></picture>'
    )
    return re.compile(rf"(?:{md}|{html})")


def _image_markup(docs: DocSet, doc: Path, *, alt: str, repo_rel: str, attrs: str = "") -> str:
    href = docs.href(repo_rel, doc)
    dark_href = docs.href(dark_rel(repo_rel), doc) if has_dark_variant(repo_rel) else None
    return picture_markup(alt, href, dark_href, attrs=attrs)


def set_image_link(
    docs: DocSet,
    *,
    alt: str,
    repo_rel: str,
    required: bool = True,
) -> Path | None:
    """Point the image with ``alt`` at ``repo_rel`` (and its dark variant) in the first doc
    that has it.

    Finds ``![alt](…)`` or the ``<picture>`` form and writes the ``<picture>`` form
    (:func:`picture_markup`). Hrefs are relative to that doc (``images/<panel>/…``
    from a figure page, ``figures/images/<panel>/…`` from the tracker).
    """
    pattern = _image_pattern(alt)
    doc = docs.search(pattern)
    if doc is None:
        if required:
            msg = f"missing markdown image for alt={alt!r} in {docs.missing_label()}"
            raise ValueError(msg)
        return None
    docs.sub(
        pattern,
        lambda m: _image_markup(docs, doc, alt=alt, repo_rel=repo_rel, attrs=m.group("attrs") or ""),
        count=1,
        path=doc,
    )
    return doc


def replace_placeholder(docs: DocSet, placeholder: str, *, alt: str, repo_rel: str) -> Path | None:
    """Swap a ``*Not yet generated.*`` placeholder for the image (:func:`picture_markup`)."""
    return docs.replace(
        placeholder,
        lambda doc: _image_markup(docs, doc, alt=alt, repo_rel=repo_rel),
        count=1,
    )


def _tracker_for(paper: str) -> Path:
    return {
        "mehregan": PAPER_1_DOC,
        "nguyen": PAPER_NGUYEN_DOC,
        "ravivarapu": PAPER_RAVIVARAPU_DOC,
    }[paper]


def _today() -> str:
    return date.today().isoformat()


def _fmt_metric(value: Any, digits: int = 1) -> str:
    """Format a gate metric for a caption; ``n/a`` when missing or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number or number in (float("inf"), float("-inf")):
        return "n/a"
    return f"{number:.{digits}f}"


def _caption_1b(manifest: dict[str, Any]) -> str:
    seeds = [int(s) for s in (manifest.get("seeds") or [])]
    if not seeds:
        seed_note = "seeds n/a"
    elif len(seeds) == 1:
        seed_note = f"seed {seeds[0]}"
    elif seeds == list(range(seeds[0], seeds[-1] + 1)):
        seed_note = f"seeds {seeds[0]}–{seeds[-1]} mean"
    else:
        seed_note = f"seeds {','.join(map(str, seeds))} mean"
    duration = _fmt_metric(manifest.get("duration_s", 10.0), 0)
    panel = manifest.get("panel") or {}
    metrics = manifest.get("paper_gate_metrics") or panel.get("paper_gate_metrics") or {}
    beta_ours = metrics.get("beta_ours") or {}
    beta_paper = metrics.get("beta_paper") or {}
    peak_ours = metrics.get("peak_hz_ours") or {}
    peak_paper = metrics.get("peak_hz_paper") or {}
    band_keys = ("healthy", "pd", "pd_130hz")
    ours = "/".join(_fmt_metric(beta_ours.get(k)) for k in band_keys)
    paper = "/".join(_fmt_metric(beta_paper.get(k)) for k in band_keys)
    head = f"GPi PSD, {duration} s sim, {seed_note}"
    beta_part = f"β[13–35 Hz] healthy/PD/130 Hz = {ours} (paper {paper})"
    peak_part = (
        f"PD peak {_fmt_metric(peak_ours.get('pd'), 0)} Hz "
        f"(paper {_fmt_metric(peak_paper.get('pd'), 0)})"
    )
    version = manifest.get("png_version")
    version_part = f"v{version}, " if version is not None else ""
    return f"{head}; {beta_part}; {peak_part}, {version_part}{_today()}"


def _caption_2a(manifest: dict[str, Any]) -> str:
    seeds = manifest.get("seeds", [])
    seed_note = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {','.join(map(str, seeds))}"
    sampling = manifest.get("sampling", "trailing")
    if manifest.get("protocol") == "eval" and sampling != "segment":
        step = manifest.get("step_s", 0.2)
        window = manifest.get("window_s", 2.0)
        protocol = (
            "Fig 5a eval protocol (2 s pre-roll, 2 s reset, five 2 s carried steps; "
            f"{step:g} s trailing / {window:g} s window)"
        )
    elif sampling == "segment":
        seg = manifest.get("segment_s", 2)
        protocol = f"{seg:.0f} s whole-segment bins"
    else:
        step = manifest.get("step_s", 0.2)
        window = manifest.get("window_s", 2.0)
        warmup = manifest.get("warmup_s")
        if warmup:
            protocol = (
                f"14 s sim (2 s pre-roll), plot = sim − 2 s, "
                f"{step:g} s trailing / {window:g} s window (end sim 14 s)"
            )
        else:
            protocol = f"{step:g} s trailing / {window:g} s window"
    return f"{protocol}, {seed_note} ({_today()})"


def papers_tracker_image_link(png_path: Path, *, doc: Path | None = None) -> str:
    """Markdown image href from ``doc`` (a tracker or figure page) to ``png_path``."""
    target = doc or PAPER_1_DOC
    return DocSet([target], root=REPO_ROOT).href(repo_rel_posix(Path(png_path)), target)


def _load_nguyen_manifest(manifest_rel: str) -> dict[str, Any] | None:
    path = REPO_ROOT / manifest_rel
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _gate_pass_cell(gates: dict[str, Any], key: str) -> str:
    if key not in gates:
        return "—"
    value = gates[key]
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "—"


def _fig4_gate_values(gates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reward = gates.get("reward")
    length = gates.get("length")
    if isinstance(reward, dict) and isinstance(length, dict):
        return {"reward": reward, "length": length}
    legacy_reward = {k: gates[k] for k, _ in NGUYEN_GATE_GROUPS["4"]["reward"] if k in gates}
    legacy_length = {k: gates[k] for k, _ in NGUYEN_GATE_GROUPS["4"]["length"] if k in gates}
    return {"reward": legacy_reward, "length": legacy_length}


FIG4_INFORMATIONAL_KEYS = frozenset(
    key for tiers in NGUYEN_FIG4_GATE_TIER.values() for key, tier in tiers.items() if tier == "info"
)


def _fig4_tier_keys(group_name: str, tier: str) -> tuple[str, ...]:
    tiers = NGUYEN_FIG4_GATE_TIER[group_name]
    return tuple(key for key, row_tier in tiers.items() if row_tier == tier)


def _group_rows_pass(
    group_values: dict[str, Any],
    rows: list[tuple[str, str]],
    *,
    skip: frozenset[str] | None = None,
    only: frozenset[str] | None = None,
) -> bool | None:
    skip = skip or frozenset()
    seen = False
    for key, _ in rows:
        if key in skip:
            continue
        if only is not None and key not in only:
            continue
        if key not in group_values:
            continue
        value = group_values[key]
        if not isinstance(value, bool):
            continue
        seen = True
        if not value:
            return False
    return True if seen else None


def _fig4_group_shape_pass(group_name: str, group_values: dict[str, Any]) -> bool | None:
    if isinstance(group_values.get("shape_pass"), bool):
        return bool(group_values["shape_pass"])
    shape_keys = frozenset(_fig4_tier_keys(group_name, "shape"))
    return _group_rows_pass(
        group_values,
        NGUYEN_GATE_GROUPS["4"][group_name],
        only=shape_keys,
    )


def _fig4_group_full_pass(group_name: str, group_values: dict[str, Any]) -> bool | None:
    if isinstance(group_values.get("pass"), bool):
        return bool(group_values["pass"])
    full_keys = frozenset(
        key for key, tier in NGUYEN_FIG4_GATE_TIER[group_name].items() if tier in {"shape", "full"}
    )
    return _group_rows_pass(
        group_values,
        NGUYEN_GATE_GROUPS["4"][group_name],
        skip=FIG4_INFORMATIONAL_KEYS,
        only=full_keys,
    )


def _fig4_tier_pass_cell(
    group_name: str,
    group_values: dict[str, Any],
    key: str,
    *,
    column: str,
) -> str:
    tier = NGUYEN_FIG4_GATE_TIER[group_name].get(key, "full")
    if tier == "info":
        return "—"
    if column == "shape" and tier == "full":
        return "—"
    return _gate_pass_cell(group_values, key)


def _nguyen_fig4_overall_pass(gates: dict[str, Any], *, shape: bool) -> bool | None:
    groups = _fig4_gate_values(gates)
    if shape and isinstance(gates.get("shape_pass"), bool):
        return bool(gates["shape_pass"])
    if not shape and isinstance(gates.get("pass"), bool):
        return bool(gates["pass"])
    reward_pass = _fig4_group_shape_pass("reward", groups["reward"]) if shape else _fig4_group_full_pass(
        "reward", groups["reward"]
    )
    length_pass = _fig4_group_shape_pass("length", groups["length"]) if shape else _fig4_group_full_pass(
        "length", groups["length"]
    )
    if reward_pass is None and length_pass is None:
        return None
    if reward_pass is False or length_pass is False:
        return False
    if reward_pass is True and length_pass is True:
        return True
    return None


def _nguyen_all_gates_pass(gates: dict[str, Any], panel: str) -> bool | None:
    if panel == "4":
        return _nguyen_fig4_overall_pass(gates, shape=False)
    rows = NGUYEN_GATE_ROWS[panel]
    if not gates:
        return None
    # Every listed row is required: a key the manifest lacks (stale manifest, or a gate
    # that was not computed) fails the panel instead of being skipped.
    for key, _ in rows:
        if gates.get(key) is not True:
            return False
    return True


def _format_nguyen_fig4_gates_table(
    manifest: dict[str, Any] | None,
    *,
    manifest_rel: str,
) -> str:
    if manifest is None:
        shape_cell = "—"
        full_cell = "—"
        source = f"no manifest at `{manifest_rel}`"
        groups = {"reward": {}, "length": {}}
    else:
        gate_values = manifest.get("gates") or {}
        groups = _fig4_gate_values(gate_values)
        overall_shape = _nguyen_fig4_overall_pass(gate_values, shape=True)
        overall_full = _nguyen_fig4_overall_pass(gate_values, shape=False)
        shape_cell = "yes" if overall_shape else ("no" if overall_shape is False else "—")
        full_cell = "yes" if overall_full else ("no" if overall_full is False else "—")
        source = f"`{manifest_rel}`"

    lines = [
        f"**Gates set** ({source}; **`shape_pass`**: {shape_cell}, **`pass`**: {full_cell}, {_today()}). "
        "**`shape_pass`**: transition timing and ep 0–200 shape. Ship exit: **`pass`** "
        "(all required gates, scale-free, anchored to the digitized paper curves; tolerances "
        "calibrated so ≥ 90 % of synthetic faithful runs pass each gate). "
        "Both subplot groups required; tier-2 rows are logged in the manifest only.",
        "",
    ]
    for group_name, title in (("reward", "Reward (panel a)"), ("length", "Length (panel b)")):
        group_values = groups[group_name]
        shape_pass = _fig4_group_shape_pass(group_name, group_values)
        full_pass = _fig4_group_full_pass(group_name, group_values)
        shape_cell = "yes" if shape_pass else ("no" if shape_pass is False else "—")
        full_cell = "yes" if full_pass else ("no" if full_pass is False else "—")
        lines.extend(
            [
                f"### {title} (`shape_pass`: {shape_cell} | `pass`: {full_cell})",
                "",
                "| Key | Description | Shape | Full |",
                "|-----|-------------|-------|------|",
            ]
        )
        for key, desc in NGUYEN_GATE_GROUPS["4"][group_name]:
            lines.append(
                f"| `{key}` | {desc} | "
                f"{_fig4_tier_pass_cell(group_name, group_values, key, column='shape')} | "
                f"{_fig4_tier_pass_cell(group_name, group_values, key, column='full')} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_nguyen_gates_table(
    panel: str,
    manifest: dict[str, Any] | None,
    *,
    manifest_rel: str,
) -> str:
    if panel == "4":
        return _format_nguyen_fig4_gates_table(manifest, manifest_rel=manifest_rel)
    rows = NGUYEN_GATE_ROWS[panel]
    if manifest is None:
        overall_cell = "—"
        source = f"no manifest at `{manifest_rel}`"
        gate_values: dict[str, Any] = {}
    else:
        gate_values = manifest.get("gates") or {}
        overall = _nguyen_all_gates_pass(gate_values, panel)
        overall_cell = "yes" if overall else ("no" if overall is False else "—")
        source = f"`{manifest_rel}`"

    report_rows = NGUYEN_REPORT_ROWS.get(panel, [])
    note = (
        "Every row is required for exit." if not report_rows else
        "Rows marked *report only* are logged and never change `pass`; the rest are required."
    )
    lines = [
        f"**Gates set** ({source}; overall **`pass`**: {overall_cell}, {_today()}). {note}",
        "",
        "| Key | Description | Pass |",
        "|-----|-------------|------|",
    ]
    for key, desc in rows:
        lines.append(f"| `{key}` | {desc} | {_gate_pass_cell(gate_values, key)} |")
    for key, desc in report_rows:
        lines.append(f"| `{key}` | *report only* — {desc} | {_gate_pass_cell(gate_values, key)} |")
    return "\n".join(lines)


def _nguyen_summary_status(panel: str, manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return "Open"
    gates = manifest.get("gates") or {}
    overall = _nguyen_all_gates_pass(gates, panel)
    if overall is True:
        version = manifest.get("png_version")
        if version is not None:
            return f"Pass (rep v{version})"
        return "Pass"
    if overall is False:
        if panel == "4":
            if gates.get("shape_pass"):
                return "Shape OK (full open)"
            groups = _fig4_gate_values(gates)
            for group_name, rows in NGUYEN_GATE_GROUPS["4"].items():
                shape_keys = frozenset(_fig4_tier_keys(group_name, "shape"))
                for key, _ in rows:
                    if key in FIG4_INFORMATIONAL_KEYS:
                        continue
                    if key in shape_keys and groups[group_name].get(key) is False:
                        return f"Fail (`{group_name} shape:{key}`)"
            for group_name, rows in NGUYEN_GATE_GROUPS["4"].items():
                for key, _ in rows:
                    if key in FIG4_INFORMATIONAL_KEYS:
                        continue
                    if groups[group_name].get(key) is False:
                        return f"Fail (`{group_name}:{key}`)"
        failed = [key for key, _ in NGUYEN_GATE_ROWS[panel] if gates.get(key) is False]
        if failed:
            return f"Fail (`{failed[0]}`)"
        return "Fail"
    return "Open"


def render_nguyen_summary_table() -> str:
    lines = [
        "| Panel | Description | Status |",
        "|-------|-------------|--------|",
    ]
    for panel, label, description in NGUYEN_SUMMARY_ROWS:
        manifest_rel = NGUYEN_MANIFEST_BY_PANEL[panel]
        manifest = _load_nguyen_manifest(manifest_rel)
        status = _nguyen_summary_status(panel, manifest)
        lines.append(f"| {label} | {description} | {status} |")
    return "\n".join(lines)


def refresh_nguyen_summary_in_text(text: str) -> str:
    """Refresh the ``summary`` block of the Nguyen tracker text."""
    block = render_nguyen_summary_table()
    pattern = re.compile(
        r"(<!-- summary:start -->)(.*?)(<!-- summary:end -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError("missing summary markers in the Nguyen tracker")

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n{block}\n{match.group(3)}"

    return pattern.sub(_repl, text, count=1)


def refresh_nguyen_gate_tables_in_docs(docs: DocSet) -> None:
    """Refresh the tracker summary and every ``gates-<panel>`` block from on-disk manifests.

    Gate blocks are written in each panel's figure page, or in the tracker when
    the page does not have the block yet. Edits stay in ``docs`` until ``save()``.
    """
    tracker = PAPER_NGUYEN_DOC
    text = docs.text(tracker)
    if text is not None:
        docs.set_text(tracker, refresh_nguyen_summary_in_text(text))
    for panel, manifest_rel in NGUYEN_MANIFEST_BY_PANEL.items():
        marker = f"gates-{panel}"
        view = docs.panel_view(tracker, panel)
        if not view.has_marker(marker):
            continue
        manifest = _load_nguyen_manifest(manifest_rel)
        body = _format_nguyen_gates_table(panel, manifest, manifest_rel=manifest_rel)
        view.replace_marker(marker, body)


def refresh_nguyen_gate_tables(*, update_docs: bool = True) -> None:
    """Refresh Nguyen gate tables (figure pages) and summary (tracker) from manifest JSON."""
    if not update_docs or not PAPER_NGUYEN_DOC.exists():
        return
    docs = DocSet([PAPER_NGUYEN_DOC], root=REPO_ROOT)
    refresh_nguyen_gate_tables_in_docs(docs)
    docs.save()


def _caption_2b(manifest: dict[str, Any]) -> str:
    seeds = manifest.get("seeds", [])
    seed_note = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {','.join(map(str, seeds))}"
    step = manifest.get("step_s", 0.2)
    window = manifest.get("window_s", 2.0)
    smc_schedule = manifest.get("smc_schedule", "periodic")
    smc_hz = manifest.get("smc_frequency_hz", 10.0)
    backend = manifest.get("plant_backend", "python")
    smc_site = manifest.get("smc_site", "thalamic")
    smc_pulse_source = manifest.get("smc_pulse_source", "drive")
    if smc_schedule == "boc":
        site = "Iappco" if smc_site == "cortical" else "Iappth"
        smc_note = f"SMC BoC inv-gamma {site}"
        if smc_pulse_source == "cor_spikes":
            smc_note += ", Cor-spike SMCτ"
    elif smc_schedule == "periodic":
        smc_note = f"SMC {smc_hz:g} Hz periodic"
    else:
        smc_note = "SMC off"
    version = manifest.get("png_version")
    protocol = (
        f"14 s sim (2 s pre-roll), plot = sim − 2 s, "
        f"{step:g} s trailing / {window:g} s EI window (end sim 14 s), "
        f"{smc_note}, backend {backend}"
    )
    bits = [protocol, seed_note]
    if version is not None:
        bits.append(f"v{version}")
    return f"{', '.join(bits)} ({_today()})"


def _caption_4a(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    mean_hz = manifest.get("mean_hz", 45)
    knobs = manifest.get("knobs") or {}
    version = manifest.get("png_version")
    summary = manifest.get("summary") or {}
    early = summary.get("early_mean_0_130")
    late = summary.get("late_mean_150_end")
    ep0 = (summary.get("paper_gate_metrics") or {}).get("ep0_mean")
    bits = [f"{mean_hz:g} Hz, paper Alg. 1 (critic on logits, lr 5e-4/1e-3)"]
    if knobs:
        bits.append(
            f"{knobs.get('alphabet')} alphabet, logit noise σ={knobs.get('logit_noise_std'):g}"
        )
    bits.append(f"seed {seed}")
    if version is not None:
        bits.append(f"v{version}")
    if isinstance(ep0, (int, float)):
        bits.append(f"ep0={ep0:.3f}")
    if isinstance(early, (int, float)) and isinstance(late, (int, float)):
        bits.append(f"early={early:.3f} late={late:.3f}")
    return f"{', '.join(bits)} ({_today()})"


def _caption_4b(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    fig4a_series = manifest.get("fig4a_series", "artifacts/figures/papers/mehregan/4a/series.json")
    version = manifest.get("png_version")
    n_ep = manifest.get("num_episodes", 8)
    summary = manifest.get("summary") or {}
    fig4b_pass = summary.get("fig4b_pass") or {}
    ep1 = fig4b_pass.get("ep1")
    ep_last = fig4b_pass.get("ep_last")
    rise = summary.get("rise_episode")
    beta = manifest.get("episode_mean_beta") or []
    bits = [
        f"{n_ep} episodes",
        "45 Hz fixed_mean_pattern (Fig 4a paired run)",
        f"seed {seed}",
        f"source {Path(str(fig4a_series)).name}",
    ]
    if version is not None:
        bits.append(f"v{version}")
    if isinstance(ep1, (int, float)) and isinstance(ep_last, (int, float)):
        n0 = int(manifest.get("num_episodes", 8)) - 1
        bits.append(f"reward ep0={ep1:.1f} ep{n0}={ep_last:.1f}")
    if rise is not None:
        bits.append(f"rise_ep={rise}")
    if len(beta) >= 2:
        bits.append(f"psd {beta[0]:.3f}→{beta[-1]:.3f}")
    if summary.get("automation_pass"):
        bits.append("gate pass")
    return f"{', '.join(bits)} ({_today()})"


def _caption_block(caption: str, manifest_path: str) -> str:
    return f"**Caption:** {caption}\n\n**Manifest:** `{manifest_path}`"


def _set_caption(
    docs: DocSet,
    marker: str,
    caption: str,
    manifest_path: str,
    *,
    required: bool = False,
) -> None:
    """Write a ``caption-<id>`` block (figure page first, then the tracker).

    With ``required=False`` a missing block is skipped; otherwise it raises.
    """
    if not required and not docs.has_marker(marker):
        return
    docs.replace_marker(marker, _caption_block(caption, manifest_path))


_EXPORT_NOTES_IMAGES = False


def set_export_notes_images(enabled: bool) -> None:
    """Enable/disable notes-folder copy after promote (panel scripts: ``--export-notes``)."""
    global _EXPORT_NOTES_IMAGES
    _EXPORT_NOTES_IMAGES = bool(enabled)


def export_notes_images_enabled() -> bool:
    return _EXPORT_NOTES_IMAGES


def _export_notes_images_after_promote(
    *png_paths: Path,
    update_docs: bool = True,
) -> None:
    """Copy replication PNGs into the notes folder (``<notes_dir>/figures``)."""
    if not export_notes_images_enabled():
        return
    if not update_docs and not png_paths:
        return
    try:
        push_path = Path(__file__).resolve().parent / "export_notes_images.py"
        spec = importlib.util.spec_from_file_location("export_notes_images", push_path)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for png in png_paths:
            mod.push_repo_rel_png(repo_rel_posix(png))
        if update_docs and not png_paths:
            mod.push_all(verbose=False)
    except Exception as exc:
        print(f"warning: export_notes_images failed: {exc}", file=sys.stderr)


def _after_promote_publish(*png_paths: Path, update_docs: bool = True) -> None:
    _export_notes_images_after_promote(*png_paths, update_docs=update_docs)


def _promote_mehregan_docs(
    panel: str,
    *,
    caption: str,
    manifest_path: str,
    alt: str,
    repo_rel: str,
    caption_required: bool = False,
    placeholder: str | None = None,
) -> None:
    """Caption + replication image link for one Mehregan panel (figure page, else tracker)."""
    docs = panel_docs("mehregan", panel)
    if not docs.exists():
        return
    _set_caption(docs, f"caption-{panel}", caption, manifest_path, required=caption_required)
    if placeholder:
        replace_placeholder(docs, placeholder, alt=alt, repo_rel=repo_rel)
    set_image_link(docs, alt=alt, repo_rel=repo_rel)
    docs.save()


def promote_1b(
    *,
    manifest: dict[str, Any],
    curves_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 1b caption + replication image link (``docs/controllers/ddpg/figures/1b.md``)."""
    caption = _caption_1b(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _promote_mehregan_docs(
            "1b",
            caption=caption,
            manifest_path=PAPER_1B_MANIFEST,
            alt=PAPER_1B_REPLICATION_ALT,
            repo_rel=repo_rel,
        caption_required=True,
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_1B_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_1B_MANIFEST,
        "curves": str(curves_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_2a(
    *,
    manifest: dict[str, Any],
    series_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 2a caption + replication image link (``docs/controllers/ddpg/figures/2a.md``)."""
    caption = _caption_2a(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _promote_mehregan_docs(
            "2a",
            caption=caption,
            manifest_path=PAPER_2A_MANIFEST,
            alt=PAPER_2A_REPLICATION_ALT,
            repo_rel=repo_rel,
        caption_required=True,
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_2A_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_2A_MANIFEST,
        "series": str(series_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_2b(
    *,
    manifest: dict[str, Any],
    series_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 2b caption + replication image link (``docs/controllers/ddpg/figures/2b.md``)."""
    caption = _caption_2b(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _promote_mehregan_docs(
            "2b",
            caption=caption,
            manifest_path=PAPER_2B_MANIFEST,
            alt=PAPER_2B_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_2B_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_2B_MANIFEST,
        "series": str(series_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_4b(
    *,
    manifest: dict[str, Any],
    rewards_path: Path,
    reward_png_path: Path,
    psd_png_path: Path,
    update_docs: bool = True,
    png_path: Path | None = None,
    combined_png_path: Path | None = None,
) -> dict[str, str]:
    """Refresh Fig 4b caption + replication image links (``docs/controllers/ddpg/figures/4b.md``)."""
    if png_path is not None:
        reward_png_path = png_path
    caption = _caption_4b(manifest)
    reward_rel = repo_rel_posix(reward_png_path)
    psd_rel = repo_rel_posix(psd_png_path)
    docs = panel_docs("mehregan", "4b")
    if update_docs and docs.exists():
        _set_caption(docs, "caption-4b", caption, PAPER_4B_MANIFEST)
        doc = docs.marker_doc("caption-4b")
        text = docs.text(doc) if doc is not None else None
        if text is not None:
            # Rebuild the "### Replication" block above the caption. In the tracker only
            # under its "## Fig 4b" section; a figure page holds just this panel.
            anchor = text.find("## Fig 4b")
            if anchor >= 0 or doc != PAPER_1_DOC:
                repl_start = text.find("### Replication\n\n", max(anchor, 0))
                if repl_start >= 0:
                    repl_start += len("### Replication\n\n")
                    caption_start = text.find("\n<!-- caption-4b:start -->", repl_start)
                    if caption_start >= 0:
                        repl_block = (
                            f"**Reward vs episode**\n\n"
                            f"![{PAPER_4B_REPLICATION_ALT}]({docs.href(reward_rel, doc)})\n\n"
                            f"**Episode-mean PSD vs episode**\n\n"
                            f"![{PAPER_4B_PSD_REPLICATION_ALT}]({docs.href(psd_rel, doc)})\n"
                        )
                        docs.set_text(doc, text[:repl_start] + repl_block + text[caption_start:])
        set_image_link(docs, alt=PAPER_4B_REPLICATION_ALT, repo_rel=reward_rel)
        set_image_link(docs, alt=PAPER_4B_PSD_REPLICATION_ALT, repo_rel=psd_rel, required=False)
        docs.save()
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(reward_png_path, PAPER_4B_PNG)
    if combined_png_path is not None:
        materialize_ship_png(combined_png_path, PAPER_4B_COMBINED_PNG)
    _after_promote_publish(reward_png_path, psd_png_path, update_docs=update_docs)
    return {
        "png": str(reward_png_path),
        "reward_png_repo_rel": reward_rel,
        "psd_png_repo_rel": psd_rel,
        "manifest": PAPER_4B_MANIFEST,
        "rewards": str(rewards_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_4a(
    *,
    manifest: dict[str, Any],
    series_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 4a caption + replication image link (``docs/controllers/ddpg/figures/4a.md``)."""
    caption = _caption_4a(manifest)
    repo_rel = repo_rel_posix(png_path)
    materialize_ship_png(png_path, PAPER_4A_PNG)
    if update_docs:
        _promote_mehregan_docs(
            "4a",
            caption=caption,
            manifest_path=PAPER_4A_MANIFEST,
            alt=PAPER_4A_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_4A_MANIFEST,
        "series": str(series_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def _caption_5a(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = manifest.get("gates") or panel.get("gates") or {}
    ckpt = manifest.get("checkpoint") or manifest.get("fig4a_checkpoint") or "see manifest"
    bits = [
        "45 Hz paper-protocol eval",
        f"seed {seed}",
        f"checkpoint={Path(str(ckpt)).name}",
    ]
    if manifest.get("skip_regular"):
        bits.append("skip_regular")
    if manifest.get("sampling") == "trailing":
        bits.append("0.2s trailing")
    version = manifest.get("png_version")
    if version is not None:
        bits.append(f"v{version}")
    trained_mean = panel.get("trained_mean")
    if isinstance(trained_mean, (int, float)):
        bits.append(f"trained_mean={trained_mean:.0f}")
    bits.append(f"no_stim_mean={panel.get('no_stim_mean', float('nan')):.0f}")
    periodic_mean = panel.get("periodic_mean")
    if isinstance(periodic_mean, (int, float)):
        bits.append(f"periodic_mean={periodic_mean:.0f}")
    if gates.get("trained_above_periodic"):
        bits.append("trained>periodic")
    elif panel.get("trained_equals_periodic"):
        bits.append("trained≡periodic")
    if gates.get("pass"):
        bits.append("gates pass")
    else:
        bits.append("gates open")
    return f"{', '.join(bits)} ({_today()})"


def _caption_5b(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = manifest.get("gates") or panel.get("gates") or {}
    ckpt = manifest.get("checkpoint") or "see manifest"
    bits = [
        "30 Hz paper-protocol eval",
        f"seed {seed}",
        f"checkpoint={Path(str(ckpt)).name}",
    ]
    if manifest.get("sampling") == "trailing":
        bits.append("0.2s trailing")
    version = manifest.get("png_version")
    if version is not None:
        bits.append(f"v{version}")
    trained_mean = panel.get("trained_mean")
    if isinstance(trained_mean, (int, float)):
        bits.append(f"trained_mean={trained_mean:.0f}")
    bits.append(f"no_stim_mean={panel.get('no_stim_mean', float('nan')):.0f}")
    periodic_mean = panel.get("periodic_mean")
    if isinstance(periodic_mean, (int, float)):
        bits.append(f"periodic_mean={periodic_mean:.0f}")
    if gates.get("trained_below_periodic") and gates.get("trained_below_no_stim"):
        bits.append("trained<both")
    elif panel.get("trained_equals_periodic"):
        bits.append("trained≡periodic")
    if gates.get("pass"):
        bits.append("gates pass")
    else:
        bits.append("gates open")
    return f"{', '.join(bits)} ({_today()})"


def promote_5b(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 5b caption + replication image link (``docs/controllers/ddpg/figures/5b.md``)."""
    caption = _caption_5b(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _promote_mehregan_docs(
            "5b",
            caption=caption,
            manifest_path=PAPER_5B_MANIFEST,
            alt=PAPER_5B_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_5B_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_5B_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_5a(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 5a caption + replication image link (``docs/controllers/ddpg/figures/5a.md``)."""
    caption = _caption_5a(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _promote_mehregan_docs(
            "5a",
            caption=caption,
            manifest_path=PAPER_5A_MANIFEST,
            alt=PAPER_5A_REPLICATION_ALT,
            repo_rel=repo_rel,
        placeholder='*Not yet generated.* Target: `docs/controllers/ddpg/figures/images/5a/efficacy_45hz.png`',
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_5A_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_5A_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def _qat_band_caption_bits(manifest: dict[str, Any]) -> list[str]:
    """Fig 6 QAT seed band: paper-like seed count and median late/pre."""
    metrics = manifest.get("paper_gate_metrics") or {}
    n = metrics.get("qat_seed_count")
    if not n:
        return []
    rel = (metrics.get("ours_late_over_pre") or {}).get("qat")
    bits = [f"QAT paper-like {metrics.get('qat_paper_like_count', 0)}/{n} train seeds"]
    if rel is not None:
        bits.append(f"QAT median late/pre {rel:.2f}")
    return bits


def _caption_6a(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = panel.get("gates") or {}
    bits = [
        "45 Hz paper-protocol eval",
        f"eval seed {seed}",
        f"fp32_post={panel.get('fp32_post_mean', 0):.0f}",
        f"QAT median post={panel.get('qat_post_mean', 0):.0f}",
    ]
    bits.extend(_qat_band_caption_bits(manifest))
    if gates.get("ptq_fp16_near_fp32") and gates.get("ptq_int8_near_fp32"):
        bits.append("PTQ tracks fp32")
    if gates.get("qat_elevated_vs_fp32"):
        bits.append("QAT elevated")
    version = manifest.get("png_version")
    if version is not None:
        bits.append(f"v{version}")
    bits.append(_today())
    return ", ".join(bits)


def promote_6a(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 6a caption + replication image link (``docs/controllers/ddpg/figures/6a.md``)."""
    caption = _caption_6a(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _promote_mehregan_docs(
            "6a",
            caption=caption,
            manifest_path=PAPER_6A_MANIFEST,
            alt=PAPER_6A_REPLICATION_ALT,
            repo_rel=repo_rel,
        placeholder='*Not yet generated.* Target: `docs/controllers/ddpg/figures/images/6a/ptq_qat_45hz.png`',
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_6A_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_6A_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def _caption_6b(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = panel.get("gates") or {}
    bits = [
        "30 Hz paper-protocol eval",
        f"eval seed {seed}",
        f"fp32_post={panel.get('fp32_post_mean', 0):.0f}",
        f"QAT median post={panel.get('qat_post_mean', 0):.0f}",
    ]
    bits.extend(_qat_band_caption_bits(manifest))
    if gates.get("ptq_fp16_near_fp32") and gates.get("ptq_int8_near_fp32"):
        bits.append("PTQ tracks fp32")
    if gates.get("qat_elevated_vs_fp32"):
        bits.append("QAT elevated")
    version = manifest.get("png_version")
    if version is not None:
        bits.append(f"v{version}")
    bits.append(_today())
    return ", ".join(bits)


def promote_6b(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 6b caption + replication image link (``docs/controllers/ddpg/figures/6b.md``)."""
    caption = _caption_6b(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _promote_mehregan_docs(
            "6b",
            caption=caption,
            manifest_path=PAPER_6B_MANIFEST,
            alt=PAPER_6B_REPLICATION_ALT,
            repo_rel=repo_rel,
        placeholder='*Not yet generated.* Target: `docs/controllers/ddpg/figures/images/6b/ptq_qat_30hz.png`',
        )
    refresh_mehregan_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_6B_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_6B_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_nguyen_2_3(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 3 caption + replication image link (``docs/controllers/snn/figures/3.md``)."""
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    repo_rel = repo_rel_posix(png_path)
    docs = panel_docs("nguyen", "3")
    link_doc = docs.marker_doc("caption-3") or docs.find(f"![{PAPER_NGUYEN_2_3_REPLICATION_ALT}]")
    link = docs.href(repo_rel, link_doc or PAPER_NGUYEN_DOC)
    if update_docs and docs.exists():
        caption_doc = docs.replace_marker(
            "caption-3",
            _caption_block(caption, PAPER_NGUYEN_2_3_MANIFEST),
        )
        set_image_link(docs, alt=PAPER_NGUYEN_2_3_REPLICATION_ALT, repo_rel=repo_rel)
        _pass = bool(manifest.get("gates", {}).get("pass"))
        _status = (
            f"**Status:** Pass — 500 × 100 ms samples; see `{png_path.name}`."
            if _pass
            else f"**Status:** Open — see `{png_path.name}`."
        )
        docs.sub(
            r"(<!-- caption-3:end -->\s*\n)\*\*Status:\*\*[^\n]+",
            lambda m: m.group(1) + _status,
            count=1,
            path=caption_doc,
        )
        refresh_nguyen_gate_tables_in_docs(docs)
        docs.save()
    materialize_ship_png(png_path, PAPER_NGUYEN_3_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_link": link,
        "manifest": PAPER_NGUYEN_2_3_MANIFEST,
        "caption": caption,
        "doc": str(link_doc or PAPER_NGUYEN_DOC),
    }


def promote_nguyen_4(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 4 caption + replication image link (``docs/controllers/snn/figures/4.md``)."""
    caption = manifest.get("caption") or "see manifest"
    version = manifest.get("png_version")
    if version is not None:
        caption = f"{caption} (v{version})"
    repo_rel = repo_rel_posix(png_path)
    docs = panel_docs("nguyen", "4")
    link_doc = docs.marker_doc("caption-4-latest") or docs.marker_doc("caption-4")
    link = docs.href(repo_rel, link_doc or PAPER_NGUYEN_DOC)
    if update_docs and docs.exists():
        latest_caption = _caption_block(caption, PAPER_NGUYEN_4_MANIFEST)
        if docs.has_marker("caption-4-latest"):
            caption_doc = docs.replace_marker("caption-4-latest", latest_caption)
        else:
            caption_doc = docs.replace_marker("caption-4", latest_caption)
        # Dual-image page: update latest; legacy single-alt docs optional.
        if docs.find(f"![{PAPER_NGUYEN_4_REPLICATION_LATEST_ALT}]") is not None:
            set_image_link(docs, alt=PAPER_NGUYEN_4_REPLICATION_LATEST_ALT, repo_rel=repo_rel)
        else:
            set_image_link(docs, alt=PAPER_NGUYEN_4_REPLICATION_ALT, repo_rel=repo_rel, required=False)
            set_image_link(
                docs, alt=PAPER_NGUYEN_4_REPLICATION_BEST_ALT, repo_rel=repo_rel, required=False
            )
        # Refresh Fig 4 Status line (any prior Open/Pass wording after caption-4).
        _pass = bool(manifest.get("gates", {}).get("pass"))
        _shape = bool(manifest.get("gates", {}).get("shape_pass"))
        gates = manifest.get("gates", {})
        length = gates.get("length", {})
        late_len = length.get("late_length_mean")
        late_s = f"{late_len:.1f}" if isinstance(late_len, (int, float)) else "?"
        _status = (
            f"**Status:** Pass — see manifest gates (`{png_path.name}`)."
            if _pass
            else (
                f"**Status:** Timing shape open — latest **v{version}** "
                f"(`late_len={late_s}`, `shape_pass={_shape}`); see manifest gates."
                if version is not None
                else f"**Status:** Open — see manifest gates (`{png_path.name}`)."
            )
        )
        docs.sub(
            r"(<!-- caption-4(?:-latest)?:end -->\s*\n)\*\*Status:\*\*[^\n]+",
            lambda m: m.group(1) + _status,
            count=1,
            path=caption_doc,
        )
        # Keep the tracker summary-table Status column in sync for the Fig 4 row.
        docs.sub(
            r"(\| Fig 4 — training reward \+ episode length \|[^|]+\|[^|]+\|[^|]+\| )(Open|Pass)( \|)",
            lambda m: m.group(1) + ("Pass" if _pass else "Open") + m.group(3),
            count=1,
        )
        replace_placeholder(
            docs,
            "*Not yet generated.* Target: `docs/controllers/snn/figures/images/4/training_reward_length.png`",
            alt="Replication Fig 4",
            repo_rel=repo_rel,
        )
        refresh_nguyen_gate_tables_in_docs(docs)
        docs.save()
    materialize_ship_png(png_path, PAPER_NGUYEN_4_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_link": link,
        "manifest": PAPER_NGUYEN_4_MANIFEST,
        "caption": caption,
        "doc": str(link_doc or PAPER_NGUYEN_DOC),
    }


def _promote_nguyen_panel(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    caption_marker: str,
    manifest_path: str,
    replication_alt: str,
    status_row_pattern: str,
    placeholder: str | None = None,
    update_docs: bool = True,
) -> dict[str, str]:
    """Shared Nguyen refresh for Fig 5–7 (figure page ``figures/<n>.md``, else tracker)."""
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    panel = caption_marker.removeprefix("caption-")
    repo_rel = repo_rel_posix(png_path)
    docs = panel_docs("nguyen", panel)
    link_doc = docs.marker_doc(caption_marker)
    link = docs.href(repo_rel, link_doc or PAPER_NGUYEN_DOC)
    if update_docs and docs.exists():
        caption_doc = docs.replace_marker(caption_marker, _caption_block(caption, manifest_path))
        if placeholder:
            replace_placeholder(docs, placeholder, alt=replication_alt, repo_rel=repo_rel)
        set_image_link(docs, alt=replication_alt, repo_rel=repo_rel)
        _pass = bool(manifest.get("gates", {}).get("pass"))
        _status = (
            f"**Status:** Pass — see manifest gates (`{png_path.name}`)."
            if _pass
            else f"**Status:** Open — see manifest gates (`{png_path.name}`)."
        )
        docs.sub(
            rf"(<!-- {caption_marker}:end -->\s*\n)\*\*Status:\*\*[^\n]+",
            lambda m: m.group(1) + _status,
            count=1,
            path=caption_doc,
        )
        docs.sub(
            status_row_pattern,
            lambda m: m.group(1) + ("Pass" if _pass else "Open") + m.group(3),
            count=1,
        )
        refresh_nguyen_gate_tables_in_docs(docs)
        docs.save()
    _after_promote_publish(png_path, update_docs=update_docs)
    return {
        "png": str(png_path),
        "png_link": link,
        "manifest": manifest_path,
        "caption": caption,
        "doc": str(link_doc or PAPER_NGUYEN_DOC),
    }


def promote_nguyen_5(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 5 caption + replication image link."""
    return _promote_nguyen_panel(
        manifest=manifest,
        png_path=png_path,
        caption_marker="caption-5",
        manifest_path=PAPER_NGUYEN_5_MANIFEST,
        replication_alt=PAPER_NGUYEN_5_REPLICATION_ALT,
        status_row_pattern=(
            r"(\| Fig 5 — CBGT spikes \+ DBS energy over training \|[^|]+\|[^|]+\|[^|]+\| )(Open|Pass)( \|)"
        ),
        placeholder="*Not yet generated from a passing Fig 4 train.* Target: `docs/controllers/snn/figures/images/5/spikes_energy_vN.png`",
        update_docs=update_docs,
    )


def promote_nguyen_6(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 6 caption + replication image link."""
    return _promote_nguyen_panel(
        manifest=manifest,
        png_path=png_path,
        caption_marker="caption-6",
        manifest_path=PAPER_NGUYEN_6_MANIFEST,
        replication_alt=PAPER_NGUYEN_6_REPLICATION_ALT,
        status_row_pattern=(
            r"(\| Fig 6 — α–β \+ DBS parameters over training \|[^|]+\|[^|]+\|[^|]+\| )(Open|Pass)( \|)"
        ),
        placeholder="*Not yet generated.* Target: `docs/controllers/snn/figures/images/6/alpha_beta_params.png`",
        update_docs=update_docs,
    )


def promote_nguyen_7(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 7 caption + replication image link."""
    return _promote_nguyen_panel(
        manifest=manifest,
        png_path=png_path,
        caption_marker="caption-7",
        manifest_path=PAPER_NGUYEN_7_MANIFEST,
        replication_alt=PAPER_NGUYEN_7_REPLICATION_ALT,
        status_row_pattern=(
            r"(\| Fig 7 — 50-episode eval \(25 steps\) \|[^|]+\|[^|]+\|[^|]+\| )(Open|Pass)( \|)"
        ),
        placeholder="*Not yet generated.* Target: `docs/controllers/snn/figures/images/7/eval_50ep.png`",
        update_docs=update_docs,
    )


def resolve_ravivarapu_doc(checkout: Path | None = None) -> Path:
    """Ravivarapu tracker ``docs/controllers/sea-dbs/README.md``."""
    return resolve_tracker("ravivarapu", checkout)


PAPER_RAVIVARAPU_DOC = resolve_ravivarapu_doc()


def require_notes_dir() -> Path:
    """Notes folder for ``--export-notes`` (``rl_adaptive_dbs.notes_export``)."""
    from rl_adaptive_dbs.notes_export import require_notes_dir as _require_notes_dir

    return _require_notes_dir(doc_paths.all_trackers(REPO_ROOT), repo_root=REPO_ROOT)


PAPER_RAVIVARAPU_4A_PNG = _image_rel("ravivarapu", "4a", "training_psd.png")


def refresh_ravivarapu_gate_tables(*, update_docs: bool = True) -> None:
    """Refresh per-panel gate tables (with live Pass column) in the Ravivarapu tracker."""
    if not update_docs or not PAPER_RAVIVARAPU_DOC.exists():
        return
    digitization = Path(__file__).resolve().parents[2] / "digitization"
    path = digitization / "ravivarapu_gate_status.py"
    if not path.is_file():
        return
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("ravivarapu_gate_status", path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.refresh_gate_tables(PAPER_RAVIVARAPU_DOC)


def _ravivarapu_4a_status_line(manifest: dict[str, Any]) -> str:
    gates = manifest.get("gates") or {}
    ver = manifest.get("png_version")
    rep = f" (rep v{ver})" if ver is not None else ""
    manifest_rel = "artifacts/figures/papers/ravivarapu/4/manifest_4a.json"
    train = manifest.get("train_config") or "fig4_ravivarapu_config"
    roll = manifest.get("display_roll_window")
    roll_note = f"; display roll{roll} (gates on raw)" if roll else ""
    if gates.get("pass"):
        return (
            f"**Status:** **Pass**{rep} — `shape_pass` and full `pass`; "
            f"`{train}`{roll_note}; "
            "paper overlay (black baseline, grey SEA-DBS). "
            f"Manifest `{manifest_rel}`."
        )
    if gates.get("shape_pass"):
        return (
            f"**Status:** Shape OK (full open){rep} — see manifest gates. "
            f"Manifest `{manifest_rel}`."
        )
    return (
        f"**Status:** Fail{rep} — see manifest gates (`shape_pass` / `pass`). "
        f"Manifest `{manifest_rel}`."
    )


def _promote_ravivarapu_docs(
    panel: str,
    *,
    caption_marker: str,
    caption: str,
    manifest_path: str,
    replication_alt: str,
    repo_rel: str,
    status_line: str | None,
    status_after_caption: bool = True,
    placeholder: str | None = None,
    extra: Any = None,
) -> None:
    """Caption, replication image link and Status line for one Ravivarapu panel.

    Edits go to ``figures/<panel>.md`` when it holds the target, else the tracker.
    ``status_after_caption`` rewrites the first ``**Status:**`` line after the
    caption block; otherwise the first one in the doc that holds the caption.
    """
    docs = panel_docs("ravivarapu", panel)
    if not docs.exists():
        return
    _set_caption(docs, caption_marker, caption, manifest_path)
    if placeholder:
        replace_placeholder(docs, placeholder, alt=replication_alt, repo_rel=repo_rel)
    set_image_link(docs, alt=replication_alt, repo_rel=repo_rel, required=False)
    if extra is not None:
        extra(docs)
    if status_line is not None:
        doc = docs.marker_doc(caption_marker) or docs.find("**Status:**")
        text = docs.text(doc) if doc is not None else None
        if text is not None:
            end = f"<!-- {caption_marker}:end -->"
            if status_after_caption:
                head, sep, tail = text.partition(end)
                if sep:
                    tail = re.sub(r"(?m)^\*\*Status:\*\*.*$", lambda _m: status_line, tail, count=1)
                    docs.set_text(doc, head + sep + tail)
            else:
                docs.set_text(
                    doc, re.sub(r"(?m)^\*\*Status:\*\*.*$", lambda _m: status_line, text, count=1)
                )
    docs.save()


def promote_ravivarapu_4a(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    # Rewrite the panel Status line on every promote — including Pass→Pass version
    # bumps (the old Open-only replace left a stale ``rep vN``).
    _promote_ravivarapu_docs(
        "4a",
        caption_marker="caption-4a",
        caption=caption,
        manifest_path="artifacts/figures/papers/ravivarapu/4/manifest_4a.json",
        replication_alt="Replication Fig 4a",
        repo_rel=repo_rel_posix(png_path),
        status_line=_ravivarapu_4a_status_line(manifest),
        status_after_caption=False,
        placeholder="*Not yet generated.* Target: `docs/controllers/sea-dbs/figures/images/4a/training_psd.png`",
    )
    refresh_ravivarapu_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_RAVIVARAPU_4A_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {"png": str(png_path), "caption": caption, "doc": str(PAPER_RAVIVARAPU_DOC)}


def promote_ravivarapu_4b(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    status = "Pass" if manifest.get("gates", {}).get("pass") else "Open"

    def _status(docs: DocSet) -> None:
        docs.replace(
            "**Status:** Open — pair with Fig 4a locked run.",
            f"**Status:** {status} — see manifest gates.",
        )

    _promote_ravivarapu_docs(
        "4b",
        caption_marker="caption-4b",
        caption=caption,
        manifest_path="artifacts/figures/papers/ravivarapu/4/manifest_4b.json",
        replication_alt="Replication Fig 4b",
        repo_rel=repo_rel_posix(png_path),
        status_line=None,
        placeholder="*Not yet generated.* Target: `docs/controllers/sea-dbs/figures/images/4b/training_reward.png`",
        extra=_status,
    )
    refresh_ravivarapu_gate_tables(update_docs=update_docs)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {"png": str(png_path), "caption": caption, "doc": str(PAPER_RAVIVARAPU_DOC)}


PAPER_RAVIVARAPU_5A_PNG = _image_rel("ravivarapu", "5a", "inference_50hz.png")
PAPER_RAVIVARAPU_5B_PNG = _image_rel("ravivarapu", "5b", "inference_30hz.png")
PAPER_RAVIVARAPU_6_PNG = _image_rel("ravivarapu", "6", "ptq_fp16_50hz.png")
PAPER_RAVIVARAPU_7_PNG = _image_rel("ravivarapu", "7", "ablation_psd.png")


def _ravivarapu_inference_status_line(manifest: dict[str, Any], *, panel: str) -> str:
    gates = manifest.get("gates") or {}
    ver = manifest.get("png_version")
    rep = f" (rep v{ver})" if ver is not None else ""
    hz = manifest.get("carrier_hz")
    manifest_rel = f"artifacts/figures/papers/ravivarapu/{panel}/manifest.json"
    if gates.get("smoke_override"):
        return (
            f"**Status:** Open{rep} — smoke override; not a ship pass. "
            f"Manifest `{manifest_rel}`."
        )
    if gates.get("pass"):
        hz_s = f"{hz:g}" if isinstance(hz, (int, float)) else "?"
        return (
            f"**Status:** **Pass**{rep} — inference @ {hz_s} Hz; "
            f"Manifest `{manifest_rel}`."
        )
    failed = [k for k, v in gates.items() if v is False and k not in {"pass", "smoke_override"}]
    fail_note = ", ".join(failed[:4]) if failed else "see gates"
    return (
        f"**Status:** Fail{rep} (`{fail_note}`). Manifest `{manifest_rel}`."
    )


def _promote_ravivarapu_inference(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    panel: str,
    caption_marker: str,
    replication_alt: str,
    ship_alias: str,
    update_docs: bool = True,
) -> dict[str, str]:
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    if update_docs:
        _promote_ravivarapu_docs(
            panel,
            caption_marker=caption_marker,
            caption=caption,
            manifest_path=f"artifacts/figures/papers/ravivarapu/{panel}/manifest.json",
            replication_alt=replication_alt,
            repo_rel=repo_rel_posix(png_path),
            status_line=_ravivarapu_inference_status_line(manifest, panel=panel),
        )
    refresh_ravivarapu_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, ship_alias)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {"png": str(png_path), "caption": caption, "doc": str(PAPER_RAVIVARAPU_DOC)}


PAPER_RAVIVARAPU_T2_PNG = _image_rel("ravivarapu", "table2", "avg_psd.png")


def promote_ravivarapu_table2(
    *, manifest: dict[str, Any], png_path: Path, update_docs: bool = True
) -> dict[str, str]:
    return _promote_ravivarapu_inference(
        manifest=manifest,
        png_path=png_path,
        panel="table2",
        caption_marker="caption-t2",
        replication_alt="Replication Table II",
        ship_alias=PAPER_RAVIVARAPU_T2_PNG,
        update_docs=update_docs,
    )


def promote_ravivarapu_5a(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    return _promote_ravivarapu_inference(
        manifest=manifest,
        png_path=png_path,
        panel="5a",
        caption_marker="caption-5a",
        replication_alt="Replication Fig 5a",
        ship_alias=PAPER_RAVIVARAPU_5A_PNG,
        update_docs=update_docs,
    )


def promote_ravivarapu_5b(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    return _promote_ravivarapu_inference(
        manifest=manifest,
        png_path=png_path,
        panel="5b",
        caption_marker="caption-5b",
        replication_alt="Replication Fig 5b",
        ship_alias=PAPER_RAVIVARAPU_5B_PNG,
        update_docs=update_docs,
    )


def _ravivarapu_fig6_status_line(manifest: dict[str, Any]) -> str:
    gates = manifest.get("gates") or {}
    ver = manifest.get("png_version")
    rep = f" (rep v{ver})" if ver is not None else ""
    manifest_rel = "artifacts/figures/papers/ravivarapu/6/manifest.json"
    if gates.get("smoke_override"):
        return (
            f"**Status:** Open{rep} — smoke override; not a ship pass. "
            f"Manifest `{manifest_rel}`."
        )
    if gates.get("pass"):
        fp32_mb = round((manifest.get("model_bytes_fp32") or 0) / (1024 * 1024), 1)
        ptq_mb = round((manifest.get("model_bytes_fp16_ptq") or 0) / (1024 * 1024), 1)
        return (
            f"**Status:** **Pass**{rep} — FP16 PTQ @ 50 Hz; "
            f"checkpoint ~{fp32_mb} MB → ~{ptq_mb} MB; Manifest `{manifest_rel}`."
        )
    failed = [k for k, v in gates.items() if v is False and k not in {"pass", "smoke_override"}]
    fail_note = ", ".join(failed[:4]) if failed else "see gates"
    return f"**Status:** Fail{rep} (`{fail_note}`). Manifest `{manifest_rel}`."


def promote_ravivarapu_6(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    if update_docs:
        _promote_ravivarapu_docs(
            "6",
            caption_marker="caption-6",
            caption=caption,
            manifest_path="artifacts/figures/papers/ravivarapu/6/manifest.json",
            replication_alt="Replication Fig 6",
            repo_rel=repo_rel_posix(png_path),
            status_line=_ravivarapu_fig6_status_line(manifest),
        )
    refresh_ravivarapu_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_RAVIVARAPU_6_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {"png": str(png_path), "caption": caption, "doc": str(PAPER_RAVIVARAPU_DOC)}


def _ravivarapu_fig7_status_line(manifest: dict[str, Any]) -> str:
    gates = manifest.get("gates") or {}
    rep = f" (rep v{manifest['png_version']})" if manifest.get("png_version") is not None else ""
    manifest_rel = "artifacts/figures/papers/ravivarapu/7/manifest.json"
    if bool(gates.get("pass", False)):
        return f"**Status:** **Pass**{rep} — ablation (Baseline / +PM / +GS / SEA-DBS); Manifest `{manifest_rel}`."
    failed = [k for k, v in gates.items() if isinstance(v, bool) and not v and not k.startswith("smoke_")]
    fail_note = ", ".join(failed[:4]) if failed else "see gates"
    return f"**Status:** Fail{rep} (`{fail_note}`). Manifest `{manifest_rel}`."


def promote_ravivarapu_7(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    if update_docs:
        _promote_ravivarapu_docs(
            "7",
            caption_marker="caption-7",
            caption=caption,
            manifest_path="artifacts/figures/papers/ravivarapu/7/manifest.json",
            replication_alt="Replication Fig 7",
            repo_rel=repo_rel_posix(png_path),
            status_line=_ravivarapu_fig7_status_line(manifest),
        )
    refresh_ravivarapu_gate_tables(update_docs=update_docs)
    materialize_ship_png(png_path, PAPER_RAVIVARAPU_7_PNG)
    _after_promote_publish(png_path, update_docs=update_docs)
    return {"png": str(png_path), "caption": caption, "doc": str(PAPER_RAVIVARAPU_DOC)}


