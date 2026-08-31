"""Evaluate Ravivarapu panel gates and refresh gate tables in ``replications.md``."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_DIG = Path(__file__).resolve().parent
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))

RAVIVARAPU_GATE_ROWS: dict[str, list[tuple[str, str]]] = {
    "4a": [
        ("n_episodes_ok", "≥ 150 training episodes"),
        ("shared_start", "baseline and SEA-DBS agree at episode start"),
        ("baseline_declines", "baseline PSD declines over training"),
        ("sea_declines", "SEA-DBS PSD declines over training"),
        ("sea_below_baseline_mid", "SEA-DBS mid PSD below baseline"),
        ("sea_below_baseline_midlate", "SEA-DBS ep 70–110 PSD below baseline"),
        ("sea_below_baseline_late", "SEA-DBS late PSD below baseline"),
        ("sea_steeper_drop_than_baseline", "SEA-DBS drop steeper than baseline"),
        ("gap_widens_over_training", "baseline − SEA-DBS gap widens over training"),
        ("late_gap_substantial", "late baseline − SEA-DBS gap in [0.015, 0.050]"),
        ("drop_timing_baseline", "baseline drop not front-loaded by ep 50"),
        ("drop_timing_sea", "SEA-DBS drop not front-loaded by ep 50"),
        ("gradual_decline_baseline", "gradual baseline mid→late drop"),
        ("gradual_decline_sea", "gradual SEA-DBS mid→late drop"),
        ("pearson_baseline_min", "baseline trajectory shape (Pearson r)"),
        ("pearson_sea_min", "SEA-DBS trajectory shape (Pearson r)"),
        ("shared_start_near_paper", "shared start vs paper"),
        ("baseline_drop_vs_paper", "baseline drop vs paper"),
        ("sea_drop_vs_paper", "SEA-DBS drop vs paper"),
        ("sea_mid_near_paper", "SEA-DBS mid-training level vs paper"),
        ("sea_midlate_near_paper", "SEA-DBS ep 70–110 level vs paper"),
        ("sea_late_near_paper", "SEA-DBS late level vs paper"),
        ("mid_gap_near_paper", "mid-training gap vs paper (bounded)"),
        ("midlate_gap_near_paper", "ep 70–110 gap vs paper (bounded)"),
        ("late_gap_near_paper", "late gap vs paper (bounded)"),
        ("late_early_ratio_baseline_near_paper", "baseline late/early ratio vs paper"),
        ("late_early_ratio_sea_near_paper", "SEA-DBS late/early ratio vs paper"),
    ],
    "4b": [
        ("n_episodes_ok", "≥ 150 training episodes"),
        ("shared_start", "baseline and SEA-DBS agree at episode start"),
        ("baseline_rises", "baseline reward rises over training"),
        ("sea_rises", "SEA-DBS reward rises over training"),
        ("sea_above_baseline_late", "SEA-DBS late reward above baseline"),
        ("sea_steeper_rise_than_baseline", "SEA-DBS rise steeper than baseline"),
        ("paper_pull_ahead_mid", "SEA-DBS ahead in mid training window"),
        ("late_gap_substantial", "late SEA-DBS − baseline gap substantial"),
        ("rise_timing_baseline", "baseline rise not front-loaded by ep 50"),
        ("rise_timing_sea", "SEA-DBS rise not front-loaded by ep 50"),
        ("gradual_rise_baseline", "gradual baseline mid→late rise"),
        ("gradual_rise_sea", "gradual SEA-DBS mid→late rise"),
        ("pearson_baseline_min", "baseline trajectory shape (Pearson r)"),
        ("pearson_sea_min", "SEA-DBS trajectory shape (Pearson r)"),
        ("shared_start_near_paper", "shared start vs paper"),
        ("baseline_rise_vs_paper", "baseline rise vs paper"),
        ("sea_rise_vs_paper", "SEA-DBS rise vs paper"),
        ("late_gap_near_paper", "late gap vs paper (scaled)"),
        ("midlate_gap_near_paper", "ep 70–110 gap vs paper (scaled)"),
    ],
    "5a": [
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("baseline_declines", "baseline PSD net drop step 0→10"),
        ("paper_declines", "SEA-DBS PSD net drop step 0→10"),
        ("paper_end_below_baseline", "SEA-DBS end PSD below baseline"),
        ("paper_steeper_drop", "SEA-DBS drop steeper than baseline"),
        ("carrier_hz_ok", "carrier frequency 50 Hz"),
        ("shared_start_near_paper", "shared start vs paper ~462.5"),
        ("early_mae_baseline", "steps 0–5 MAE vs digitized Baseline ≤ 0.03"),
        ("early_mae_sea", "steps 0–5 MAE vs digitized SEA-DBS ≤ 0.03"),
        ("early_mae_sea_3_5", "steps 3–5 MAE vs digitized SEA-DBS ≤ 0.020"),
        ("early_sea_declines", "SEA-DBS drop steps 0→5 > 0.05"),
        ("early_baseline_declines", "Baseline drop steps 0→5"),
        ("early_sea_below_baseline", "SEA-DBS below Baseline at every step 1–5"),
        ("late_baseline_declines", "Baseline keeps declining steps 5→10"),
        ("late_sea_declines", "SEA-DBS keeps declining steps 5→10"),
        ("mid_mae_sea", "steps 4–10 MAE vs digitized SEA-DBS ≤ 0.012"),
    ],
    "5b": [
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("baseline_declines", "baseline PSD net drop step 0→10"),
        ("paper_declines", "SEA-DBS PSD net drop step 0→10"),
        ("paper_end_below_baseline", "SEA-DBS end PSD below baseline"),
        ("paper_steeper_drop", "SEA-DBS drop steeper than baseline"),
        ("carrier_hz_ok", "carrier frequency 30 Hz"),
        ("weaker_than_50hz_sea", "30 Hz SEA-DBS weaker suppression than 50 Hz panel"),
        ("weaker_than_50hz_baseline", "30 Hz baseline weaker suppression than 50 Hz panel"),
        ("early_baseline_rises", "baseline PSD initial rise at step 1-2"),
        ("early_sea_plateau", "SEA-DBS delayed drop / plateau on steps 0-2"),
        ("early_sea_below_baseline", "SEA-DBS below baseline on steps 1-5"),
        ("early_mae_baseline", "steps 0–5 MAE vs digitized baseline ≤ 0.025"),
        ("early_mae_sea", "steps 0–5 MAE vs digitized SEA-DBS ≤ 0.025"),
        ("late_baseline_declines", "baseline keeps declining steps 5→10"),
        ("late_sea_declines", "SEA-DBS keeps declining steps 5→10"),
        ("pearson_baseline_min", "baseline trajectory correlation vs paper (r ≥ 0.70)"),
        ("pearson_sea_min", "SEA-DBS trajectory correlation vs paper (r ≥ 0.70)"),
    ],
    "6": [
        ("four_series_present", "fp32 + PTQ for baseline and SEA-DBS"),
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "paired series share pre-stim level"),
        ("shared_start_near_paper", "shared start vs paper ~462.5"),
        ("early_mae_baseline", "steps 0–5 MAE vs digitized Baseline ≤ 0.03"),
        ("early_mae_sea", "steps 0–5 MAE vs digitized SEA-DBS ≤ 0.03"),
        ("sea_below_baseline", "SEA-DBS fp32 below baseline fp32 late"),
        ("sea_ptq_below_baseline", "SEA-DBS PTQ below baseline fp32 late"),
        ("sea_ptq_tracks_fp32", "SEA-DBS PTQ tracks fp32"),
        ("baseline_ptq_near_or_above_baseline", "baseline PTQ near/above baseline fp32"),
        ("ptq_traces_distinct", "PTQ traces not identical to paired fp32"),
    ],
    "7": [
        ("four_variants_present", "baseline / +PM / +GS / SEA-DBS"),
        ("n_steps_ok", "11 PSD samples (t=0 + 10 stim steps)"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("shared_start_near_paper", "shared start vs paper ~462.5"),
        ("early_mae_baseline", "steps 0–5 MAE vs digitized Baseline ≤ 0.03"),
        ("early_mae_sea", "steps 0–5 MAE vs digitized SEA-DBS ≤ 0.03"),
        ("sea_dbs_lowest_tail", "SEA-DBS lowest tail mean PSD"),
        ("gs_highest_or_near_highest_tail", "GS highest or near-highest tail"),
        ("pm_not_sea", "PM closer to baseline than to SEA-DBS"),
    ],
}

RAVIVARAPU_MANIFEST_BY_PANEL: dict[str, str] = {
    "4a": "artifacts/figures/papers/ravivarapu/4/manifest_4a.json",
    "4b": "artifacts/figures/papers/ravivarapu/4/manifest_4b.json",
    "5a": "artifacts/figures/papers/ravivarapu/5a/manifest.json",
    "5b": "artifacts/figures/papers/ravivarapu/5b/manifest.json",
    "6": "artifacts/figures/papers/ravivarapu/6/manifest.json",
    "7": "artifacts/figures/papers/ravivarapu/7/manifest.json",
}

RAVIVARAPU_SUMMARY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("4a", "Fig 4a", "Training PSD vs episode"),
    ("4b", "Fig 4b", "Training reward vs episode"),
    ("5a", "Fig 5a", "Inference @ 50 Hz"),
    ("5b", "Fig 5b", "Inference @ 30 Hz"),
    ("6", "Fig 6", "FP16 PTQ @ 50 Hz"),
    ("7", "Fig 7", "Ablation (Baseline / +PM / +GS / SEA-DBS)"),
)

RAVIVARAPU_STATUS_NOTES: dict[str, str] = {}


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


def _summary_status_line(status: PanelGateStatus, note: str = "") -> str:
    if status.panel in ("4a", "4b"):
        if status.overall:
            return f"Pass ({note})" if note else "Pass"
        if status.shape_overall and status.overall is False:
            return f"Shape OK (full open){f', {note}' if note else ''}"
        failed = [key for key, value in status.gates.items() if value is False]
        if failed:
            tier = _fig_gate_tier(status.panel).get(failed[0], "full")
            prefix = "shape" if tier == "shape" else "full"
            detail = f"`{failed[0]}` ({prefix})"
            if note:
                detail = f"{detail}, {note}"
            return f"Fail ({detail})"
        if status.overall is False:
            return f"Fail ({note})" if note else "Fail"
        return "Open"
    if status.overall:
        return f"Pass ({note})" if note else "Pass"
    failed = [key for key, value in status.gates.items() if value is False]
    if failed:
        detail = f"`{failed[0]}`"
        if note:
            detail = f"{detail}, {note}"
        return f"Fail ({detail})"
    if status.overall is False:
        return f"Fail ({note})" if note else "Fail"
    return "Open"


def render_summary_table(statuses: dict[str, PanelGateStatus]) -> str:
    lines = [
        "| Panel | Description | Status |",
        "|-------|-------------|--------|",
    ]
    for panel_key, label, description in RAVIVARAPU_SUMMARY_ROWS:
        status = statuses[panel_key]
        note = RAVIVARAPU_STATUS_NOTES.get(panel_key, "")
        if panel_key in ("4a", "4b"):
            manifest_path = _REPO / RAVIVARAPU_MANIFEST_BY_PANEL[panel_key]
            if manifest_path.is_file():
                manifest = _load_json(manifest_path)
                version = manifest.get("png_version") if manifest else None
                if version is not None:
                    note = f"v{version}"
        lines.append(
            f"| {label} | {description} | {_summary_status_line(status, note)} |"
        )
    return "\n".join(lines)


def inject_summary_table(text: str, statuses: dict[str, PanelGateStatus]) -> str:
    block = render_summary_table(statuses)
    pattern = re.compile(
        r"(<!-- summary:start -->)(.*?)(<!-- summary:end -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError("missing summary markers in Ravivarapu replications doc")

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n{block}\n{match.group(3)}"

    return pattern.sub(_repl, text, count=1)


def refresh_gate_tables(doc_path: Path) -> dict[str, PanelGateStatus]:
    """Replace ``<!-- gates-{panel}:start/end -->`` blocks in the tracker doc."""
    text = doc_path.read_text(encoding="utf-8")
    statuses: dict[str, PanelGateStatus] = {}
    for panel in RAVIVARAPU_GATE_ROWS:
        status = evaluate_panel(panel)
        statuses[panel] = status
        block = render_gate_block(status)
        pattern = re.compile(
            rf"(<!-- gates-{re.escape(panel)}:start -->)(.*?)(<!-- gates-{re.escape(panel)}:end -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise ValueError(f"missing gates markers for panel {panel} in {doc_path}")

        def _repl(match: re.Match[str], *, _block: str = block) -> str:
            return f"{match.group(1)}\n{_block}\n{match.group(3)}"

        text = pattern.sub(_repl, text, count=1)
    text = inject_summary_table(text, statuses)
    doc_path.write_text(text, encoding="utf-8")
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
