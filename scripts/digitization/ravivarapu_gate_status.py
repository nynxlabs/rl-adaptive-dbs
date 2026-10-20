"""Evaluate Ravivarapu panel gates and refresh gate tables in the figure pages + tracker (``docs/controllers/sea-dbs/``)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rl_adaptive_dbs.doc_paths import DocSet, panel_for_marker_id

_REPO = Path(__file__).resolve().parents[2]
_DIG = Path(__file__).resolve().parent
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))

RAVIVARAPU_GATE_ROWS: dict[str, list[tuple[str, str]]] = {
    "4a": [
        ("n_episodes_ok", "≥ 150 training episodes"),
        ("shared_start", "baseline and SEA-DBS agree at episode start (ep 0–15, ≤ 0.05)"),
        ("baseline_declines", "baseline PSD declines over training (≥ 0.02)"),
        ("sea_below_baseline_mid", "SEA-DBS ep 40–80 PSD below baseline"),
        ("sea_below_baseline_midlate", "SEA-DBS ep 70–110 PSD below baseline"),
        ("sea_below_baseline_late", "SEA-DBS ep 120–150 PSD below baseline"),
        ("sea_steeper_drop_than_baseline", "SEA-DBS drop steeper than baseline"),
        ("gap_widens_over_training", "baseline − SEA-DBS gap widens over training"),
        ("drop_timing_baseline", "baseline drop not front-loaded by ep 50 vs paper"),
        ("drop_timing_sea", "SEA-DBS drop not front-loaded by ep 50 vs paper"),
        ("gradual_decline_baseline", "baseline mid→late drop ≥ 30 % of paper's"),
        ("gradual_decline_sea", "SEA-DBS mid→late drop ≥ 30 % of paper's"),
        ("pearson_baseline_min", "baseline trajectory shape vs paper (Pearson r ≥ 0.58)"),
        ("pearson_sea_min", "SEA-DBS trajectory shape vs paper (Pearson r ≥ 0.58)"),
        ("shared_start_near_paper", "ep 0–15 level within 0.02 of paper (0.443 / 0.448)"),
        ("baseline_drop_vs_paper", "baseline early→late drop ≥ 40 % of paper's"),
        ("sea_drop_vs_paper", "SEA-DBS early→late drop ≥ 40 % of paper's"),
        ("baseline_mid_near_paper", "baseline ep 40–80 level within 12 % of paper"),
        ("baseline_midlate_near_paper", "baseline ep 70–110 level within 12 % of paper"),
        ("baseline_late_near_paper", "baseline ep 120–150 level within 12 % of paper"),
        ("sea_mid_near_paper", "SEA-DBS ep 40–80 level within 12 % of paper"),
        ("sea_midlate_near_paper", "SEA-DBS ep 70–110 level within 12 % of paper"),
        ("sea_late_near_paper", "SEA-DBS ep 120–150 level within 12 % of paper"),
        ("mid_gap_near_paper", "ep 40–80 gap in [0.40, 1.80] × paper"),
        ("midlate_gap_near_paper", "ep 70–110 gap in [0.40, 1.75] × paper"),
        ("late_gap_near_paper", "ep 120–150 gap in [0.50, 1.60] × paper (0.028)"),
        ("late_early_ratio_baseline_near_paper", "baseline late/early ratio within 8 % of paper (0.833)"),
        ("late_early_ratio_sea_near_paper", "SEA-DBS late/early ratio within 8 % of paper (0.762)"),
    ],
    "4b": [
        ("n_episodes_ok", "≥ 150 training episodes"),
        ("shared_start", "baseline and SEA-DBS agree at episode start (ep 0–15, ≤ 25 %)"),
        ("baseline_rises", "baseline reward rises ≥ 25 % of paper's rise"),
        ("sea_rises", "SEA-DBS reward rises ≥ 25 % of paper's rise"),
        ("sea_above_baseline_late", "SEA-DBS ep 90–150 reward above baseline"),
        ("sea_steeper_rise_than_baseline", "SEA-DBS rise to ep 90–150 steeper than baseline"),
        ("paper_pull_ahead_mid", "SEA-DBS ahead in ep 40–80"),
        ("sea_faster_initial_rise", "SEA-DBS covers more of its rise by ep 40–80 (paper 0.79 vs 0.66)"),
        ("rise_timing_baseline", "baseline rise not front-loaded by ep 50 vs paper"),
        ("rise_timing_sea", "SEA-DBS rise not front-loaded by ep 50 vs paper"),
        ("gradual_rise_baseline", "baseline mid→late rise ≥ 25 % of paper's"),
        ("gradual_rise_sea", "SEA-DBS mid→late rise ≥ 25 % of paper's"),
        ("pearson_baseline_min", "baseline trajectory shape vs paper (Pearson r ≥ 0.55)"),
        ("pearson_sea_min", "SEA-DBS trajectory shape vs paper (Pearson r ≥ 0.55)"),
        ("eq7_consistent_with_4a", "reward = Eq. (7) of the Fig 4a PSD, episode by episode (≤ 0.05; paper 0.038)"),
        ("shared_start_near_paper", "ep 0–15 reward within 30 % of paper (−0.88 / −0.94)"),
        ("baseline_rise_vs_paper", "baseline rise ≥ 40 % of paper's"),
        ("sea_rise_vs_paper", "SEA-DBS rise ≥ 40 % of paper's"),
        ("late_gap_near_paper", "ep 90–150 gap ≥ 35 % of paper's (0.081)"),
        ("midlate_gap_near_paper", "ep 70–110 gap ≥ 40 % of paper's (0.144)"),
    ],
    "5a": [
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("baseline_declines", "baseline PSD net drop step 0→10"),
        ("paper_declines", "SEA-DBS PSD net drop step 0→10"),
        ("paper_end_below_baseline", "SEA-DBS end PSD below baseline"),
        ("paper_steeper_drop", "SEA-DBS drop steeper than baseline"),
        ("carrier_hz_ok", "carrier frequency 50 Hz"),
        ("shared_start_near_paper", "step 0 within 0.02 of paper (~0.46)"),
        ("early_mae_baseline", "steps 0–5 MAE vs paper Baseline ≤ 20 % of its drop (0.021)"),
        ("early_mae_sea", "steps 0–5 MAE vs paper SEA-DBS ≤ 20 % of its drop (0.030)"),
        ("baseline_drop_near_paper", "Baseline step 0→10 drop in [0.5, 1.5] × paper (0.104)"),
        ("sea_drop_near_paper", "SEA-DBS step 0→10 drop in [0.5, 1.5] × paper (0.148)"),
        ("tail_gap_near_paper", "steps 6–10 Baseline − SEA-DBS gap in [0.5, 1.6] × paper (0.056)"),
        ("early_mae_sea_3_5", "steps 3–5 MAE vs paper SEA-DBS ≤ 10 % of its drop (0.015)"),
        ("early_sea_declines", "SEA-DBS drop steps 0→5 ≥ 40 % of paper's (0.048)"),
        ("early_baseline_declines", "Baseline drop steps 0→5"),
        ("early_sea_below_baseline", "SEA-DBS below Baseline at every step 1–5"),
        ("mid_mae_sea", "steps 4–10 MAE vs paper SEA-DBS ≤ 10 % of its drop (0.015)"),
        ("late_baseline_declines", "Baseline keeps declining steps 5→10"),
        ("late_sea_declines", "SEA-DBS keeps declining steps 5→10"),
    ],
    "5b": [
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("baseline_declines", "baseline PSD net drop step 0→10"),
        ("paper_declines", "SEA-DBS PSD net drop step 0→10"),
        ("paper_end_below_baseline", "SEA-DBS end PSD below baseline"),
        ("paper_steeper_drop", "SEA-DBS drop steeper than baseline"),
        ("carrier_hz_ok", "carrier frequency 30 Hz"),
        ("shared_start_near_paper", "step 0 within 0.02 of paper (~0.46)"),
        ("early_mae_baseline", "steps 0–5 MAE vs paper Baseline ≤ 20 % of its drop (0.012)"),
        ("early_mae_sea", "steps 0–5 MAE vs paper SEA-DBS ≤ 20 % of its drop (0.014)"),
        ("baseline_drop_near_paper", "Baseline step 0→10 drop in [0.5, 1.5] × paper (0.062)"),
        ("sea_drop_near_paper", "SEA-DBS step 0→10 drop in [0.5, 1.5] × paper (0.072)"),
        ("tail_gap_near_paper", "steps 6–10 Baseline − SEA-DBS gap in [0.5, 1.6] × paper (0.014)"),
        ("weaker_than_50hz_baseline", "Baseline 30 Hz drop positive but smaller than its 50 Hz drop"),
        ("weaker_than_50hz_sea", "SEA-DBS 30 Hz drop positive but smaller than its 50 Hz drop"),
        ("early_baseline_rises", "baseline PSD initial rise at step 1-2"),
        ("early_sea_plateau", "SEA-DBS delayed drop / plateau on steps 0-2"),
        ("early_sea_below_baseline", "SEA-DBS below baseline on steps 1-5 (mean; ≤ 0.006 crossing)"),
        ("pearson_baseline_min", "baseline trajectory correlation vs paper (r ≥ 0.70)"),
        ("pearson_sea_min", "SEA-DBS trajectory correlation vs paper (r ≥ 0.70)"),
        ("late_baseline_declines", "baseline keeps declining steps 5→10"),
        ("late_sea_declines", "SEA-DBS keeps declining steps 5→10"),
    ],
    "6": [
        ("four_series_present", "fp32 + PTQ for baseline and SEA-DBS"),
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "paired series share pre-stim level"),
        ("shared_start_near_paper", "step 0 within 0.02 of paper (~0.46)"),
        ("early_mae_baseline", "steps 0–5 MAE vs paper Baseline ≤ 20 % of its drop (0.021)"),
        ("early_mae_sea", "steps 0–5 MAE vs paper SEA-DBS ≤ 20 % of its drop (0.030)"),
        ("fp32_matches_fig5a", "fp32 traces equal the Fig 5a rollout (≤ 0.005; paper 0.0024)"),
        ("sea_below_baseline", "SEA-DBS fp32 tail below baseline fp32"),
        ("sea_ptq_below_baseline", "SEA-DBS PTQ tail below baseline fp32"),
        ("sea_ptq_tracks_fp32", "SEA-DBS PTQ within 0.03 of fp32 (tail) and 0.04 (any step)"),
        ("baseline_ptq_tracks_fp32", "Baseline PTQ within 0.03 of fp32 (tail) and 0.04 (any step)"),
        ("baseline_tail_near_paper", "Baseline fp32 steps 6–10 within 0.02 of paper (0.380)"),
        ("sea_tail_near_paper", "SEA-DBS fp32 steps 6–10 within 0.02 of paper (0.322)"),
        ("baseline_ptq_tail_near_paper", "Baseline PTQ steps 6–10 within 0.02 of paper (0.391)"),
        ("sea_ptq_tail_near_paper", "SEA-DBS PTQ steps 6–10 within 0.02 of paper (0.314)"),
    ],
    "7": [
        ("four_variants_present", "baseline / +PM / +GS / SEA-DBS"),
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("shared_start_near_paper", "step 0 within 0.02 of paper (~0.46)"),
        ("early_mae_baseline", "steps 0–5 MAE vs paper Baseline ≤ 20 % of its drop (0.021)"),
        ("early_mae_sea", "steps 0–5 MAE vs paper SEA-DBS ≤ 20 % of its drop (0.030)"),
        ("mae_pm", "steps 0–10 MAE vs paper +PM ≤ 20 % of its drop (0.020)"),
        ("mae_gs", "steps 0–10 MAE vs paper +GS ≤ 20 % of its drop (0.012)"),
        ("fp32_matches_fig5a", "Baseline / SEA-DBS equal the Fig 5a rollout (≤ 0.005)"),
        ("baseline_above_pm_early", "Baseline above Baseline+PM early (steps 0-4)"),
        ("sea_dbs_lowest_tail", "SEA-DBS tail ≥ 0.005 below every other variant"),
        ("gs_highest_tail", "Baseline+GS tail ≥ 0.005 above every other variant"),
        ("pm_not_sea", "PM closer to baseline than to SEA-DBS"),
        ("baseline_tail_near_paper", "Baseline steps 6–10 within 0.02 of paper (0.380)"),
        ("pm_tail_near_paper", "+PM steps 6–10 within 0.02 of paper (0.373)"),
        ("gs_tail_near_paper", "+GS steps 6–10 within 0.02 of paper (0.419)"),
        ("sea_tail_near_paper", "SEA-DBS steps 6–10 within 0.02 of paper (0.323)"),
    ],
    "t2": [
        ("all_intervals_present", "rows for n = 10, 20, 50, 75"),
        ("sea_below_baseline_every_n", "SEA-DBS avg PSD below Baseline at every n"),
        ("gap_near_paper_every_n", "Baseline − SEA-DBS gap in [0.5, 1.6] × paper (38 / 57 / 28 / 26)"),
        ("baseline_near_paper_every_n", "Baseline avg PSD within 20 of paper (364 / 366 / 332 / 328)"),
        ("sea_near_paper_every_n", "SEA-DBS avg PSD within 20 of paper (326 / 309 / 304 / 302)"),
        ("psd_falls_with_interval", "avg PSD lower at n = 75 than n = 10 (both)"),
    ],
}

RAVIVARAPU_MANIFEST_BY_PANEL: dict[str, str] = {
    "4a": "artifacts/figures/papers/ravivarapu/4/manifest_4a.json",
    "4b": "artifacts/figures/papers/ravivarapu/4/manifest_4b.json",
    "5a": "artifacts/figures/papers/ravivarapu/5a/manifest.json",
    "5b": "artifacts/figures/papers/ravivarapu/5b/manifest.json",
    "6": "artifacts/figures/papers/ravivarapu/6/manifest.json",
    "7": "artifacts/figures/papers/ravivarapu/7/manifest.json",
    "t2": "artifacts/figures/papers/ravivarapu/table2/manifest.json",
}

RAVIVARAPU_SUMMARY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("4a", "Fig 4a", "Training PSD vs episode"),
    ("4b", "Fig 4b", "Training reward vs episode"),
    ("5a", "Fig 5a", "Inference @ 50 Hz"),
    ("5b", "Fig 5b", "Inference @ 30 Hz"),
    ("6", "Fig 6", "FP16 PTQ @ 50 Hz"),
    ("7", "Fig 7", "Ablation (Baseline / +PM / +GS / SEA-DBS)"),
    ("t2", "Table II", "Avg PSD with the plant reseeded every n steps"),
)

def _fig_gate_tier(panel: str) -> dict[str, str]:
    if panel == "4a":
        from ravivarapu_gates import RAVIVARAPU_FIG4A_GATE_TIER

        return RAVIVARAPU_FIG4A_GATE_TIER
    if panel == "4b":
        from ravivarapu_gates import RAVIVARAPU_FIG4B_GATE_TIER

        return RAVIVARAPU_FIG4B_GATE_TIER
    return {}


def _fig_tier_pass(gate_values: dict[str, Any], *, panel: str, full: bool) -> bool:
    if panel == "4a":
        from ravivarapu_gates import ravivarapu_fig4a_tier_pass

        return ravivarapu_fig4a_tier_pass(gate_values, full=full)
    if panel == "4b":
        from ravivarapu_gates import ravivarapu_fig4b_tier_pass

        return ravivarapu_fig4b_tier_pass(gate_values, full=full)
    return False


@dataclass(frozen=True)
class GateRow:
    key: str
    description: str


@dataclass
class PanelGateStatus:
    panel: str
    pass_field: str
    overall: bool | None
    shape_overall: bool | None
    gates: dict[str, bool]
    source: str
    rows: tuple[GateRow, ...]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_rows_pass(gates: dict[str, Any], panel: str) -> bool | None:
    seen = False
    for key, _ in RAVIVARAPU_GATE_ROWS[panel]:
        if key not in gates:
            continue
        value = gates[key]
        if not isinstance(value, bool):
            continue
        seen = True
        if not value:
            return False
    return True if seen else None


def _pass_cell(value: bool | None) -> str:
    if value is None:
        return "—"
    return "yes" if value else "no"


def _fig_tier_cell(key: str, gate_values: dict[str, Any], *, panel: str, column: str) -> str:
    tier = _fig_gate_tier(panel).get(key)
    if tier is None:
        return "—"
    if column == "shape" and tier == "full":
        return "—"
    return _pass_cell(gate_values.get(key) if isinstance(gate_values.get(key), bool) else None)


def _fig_overall_pass(gate_values: dict[str, Any], *, panel: str, full: bool) -> bool | None:
    field = "pass" if full else "shape_pass"
    if isinstance(gate_values.get(field), bool):
        return bool(gate_values[field])
    return _fig_tier_pass(gate_values, panel=panel, full=full)


def _panel_gate_values(manifest: dict[str, Any], panel: str) -> dict[str, bool]:
    g = manifest.get("gates") or {}
    allowed = {key for key, _ in RAVIVARAPU_GATE_ROWS[panel]}
    return {k: bool(v) for k, v in g.items() if isinstance(v, bool) and k in allowed}


def evaluate_panel(panel: str) -> PanelGateStatus:
    manifest_rel = RAVIVARAPU_MANIFEST_BY_PANEL[panel]
    manifest_path = _REPO / manifest_rel
    rows = tuple(GateRow(key, desc) for key, desc in RAVIVARAPU_GATE_ROWS[panel])
    if not manifest_path.is_file():
        return PanelGateStatus(
            panel=panel,
            pass_field="pass",
            overall=None,
            shape_overall=None,
            gates={},
            source=f"no manifest at `{manifest_rel}`",
            rows=rows,
        )
    manifest = _load_json(manifest_path)
    gate_values = manifest.get("gates") or {}
    gates = _panel_gate_values(manifest, panel)
    if panel in ("4a", "4b"):
        shape_overall = _fig_overall_pass(gate_values, panel=panel, full=False)
        overall = _fig_overall_pass(gate_values, panel=panel, full=True)
    else:
        shape_overall = None
        overall = _all_rows_pass(gate_values, panel)
    return PanelGateStatus(
        panel=panel,
        pass_field="pass",
        overall=overall,
        shape_overall=shape_overall,
        gates=gates,
        source=manifest_rel,
        rows=rows,
    )


def _render_fig_tiered_gate_block(status: PanelGateStatus) -> str:
    manifest_path = _REPO / status.source
    gate_values: dict[str, Any] = {}
    if manifest_path.is_file():
        gate_values = _load_json(manifest_path).get("gates") or {}
    shape_cell = _pass_cell(status.shape_overall)
    full_cell = _pass_cell(status.overall)
    lines = [
        f"**Gates set** (`{status.source}`; **`shape_pass`**: {shape_cell}, **`pass`**: {full_cell}, "
        f"{_today()}). Phase 1: **`shape_pass`** (trajectory shape / ordering). "
        "Ship exit: **`pass`** (adds digitization level polish).",
        "",
        "| Key | Description | Shape | Full |",
        "|-----|-------------|-------|------|",
    ]
    for row in status.rows:
        lines.append(
            f"| `{row.key}` | {row.description} | "
            f"{_fig_tier_cell(row.key, gate_values, panel=status.panel, column='shape')} | "
            f"{_fig_tier_cell(row.key, gate_values, panel=status.panel, column='full')} |"
        )
    return "\n".join(lines)


def render_gate_block(status: PanelGateStatus) -> str:
    if status.panel in ("4a", "4b"):
        return _render_fig_tiered_gate_block(status)
    overall = _pass_cell(status.overall)
    lines = [
        f"**Gates set** (`{status.source}`; overall **`{status.pass_field}`**: {overall}, "
        f"{_today()}). Every row is required for exit.",
        "",
        "| Key | Description | Pass |",
        "|-----|-------------|------|",
    ]
    for row in status.rows:
        lines.append(
            f"| `{row.key}` | {row.description} | {_pass_cell(status.gates.get(row.key))} |"
        )
    return "\n".join(lines)


def _summary_status_line(status: PanelGateStatus, version: Any = None) -> str:
    """``Pass`` / ``Fail — 15/28 gates pass`` with the replication version."""
    rep = f" (rep v{version})" if version is not None else ""
    if not status.gates:
        return f"Open{rep}"
    n_pass = sum(1 for value in status.gates.values() if value)
    count = f"{n_pass}/{len(status.gates)} gates pass"
    if status.overall:
        return f"Pass — {count}{rep}"
    return f"Fail — {count}{rep}"


def render_summary_table(statuses: dict[str, PanelGateStatus]) -> str:
    lines = [
        "| Panel | Description | Status |",
        "|-------|-------------|--------|",
    ]
    for panel_key, label, description in RAVIVARAPU_SUMMARY_ROWS:
        status = statuses[panel_key]
        version = None
        manifest_path = _REPO / RAVIVARAPU_MANIFEST_BY_PANEL[panel_key]
        if manifest_path.is_file():
            version = _load_json(manifest_path).get("png_version")
        lines.append(f"| {label} | {description} | {_summary_status_line(status, version)} |")
    return "\n".join(lines)


def inject_summary_table(text: str, statuses: dict[str, PanelGateStatus]) -> str:
    block = render_summary_table(statuses)
    pattern = re.compile(
        r"(<!-- summary:start -->)(.*?)(<!-- summary:end -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError("missing summary markers in the Ravivarapu tracker")

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n{block}\n{match.group(3)}"

    return pattern.sub(_repl, text, count=1)


def refresh_gate_tables(doc_path: Path) -> dict[str, PanelGateStatus]:
    """Refresh ``<!-- gates-{panel}:start/end -->`` blocks and the tracker summary.

    ``doc_path`` is the paper tracker (``docs/controllers/<controller>/README.md``).
    Each panel's gate block is written in its figure page ``figures/<panel>.md``
    beside the tracker, or in the tracker when the page does not have it yet.
    """
    docs = DocSet([doc_path])
    statuses: dict[str, PanelGateStatus] = {}
    for panel in RAVIVARAPU_GATE_ROWS:
        status = evaluate_panel(panel)
        statuses[panel] = status
        view = docs.panel_view(doc_path, panel_for_marker_id("ravivarapu", panel))
        view.replace_marker(
            f"gates-{panel}",
            render_gate_block(status),
            error=f"missing gates markers for panel {panel} in {view.missing_label()}",
        )
    tracker_text = docs.text(doc_path)
    if tracker_text is None:
        raise FileNotFoundError(doc_path)
    docs.set_text(doc_path, inject_summary_table(tracker_text, statuses))
    docs.save()
    return statuses


def _today() -> str:
    return date.today().isoformat()


def main() -> int:
    promote_path = _REPO / "scripts" / "figures" / "papers" / "promote.py"
    spec = importlib.util.spec_from_file_location("figure_promote", promote_path)
    assert spec is not None and spec.loader is not None
    promote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(promote)
    doc = promote.resolve_ravivarapu_doc()
    statuses = refresh_gate_tables(doc)
    for panel, status in statuses.items():
        if panel == "4a":
            shape = status.shape_overall
            full = status.overall
            mark = (
                "PASS"
                if full
                else ("SHAPE" if shape and full is False else ("FAIL" if full is False else "OPEN"))
            )
        else:
            mark = "PASS" if status.overall else ("FAIL" if status.overall is False else "OPEN")
        failed = [k for k, v in status.gates.items() if not v]
        print(f"{panel}: {mark}  failed={failed or '—'}")
    print(f"updated {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
