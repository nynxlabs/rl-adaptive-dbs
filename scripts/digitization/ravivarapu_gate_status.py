"""Evaluate Ravivarapu panel gates and refresh gate tables in ``replications.md``."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]

RAVIVARAPU_GATE_ROWS: dict[str, list[tuple[str, str]]] = {
    "4a": [
        ("shared_start", "baseline and SEA-DBS agree at episode start"),
        ("baseline_declines", "baseline PSD declines over training"),
        ("paper_declines", "SEA-DBS PSD declines over training"),
        ("paper_below_baseline_late", "SEA-DBS late PSD below baseline"),
        ("paper_steeper_drop", "SEA-DBS drop steeper than baseline"),
        ("late_gap_min", "late baseline − SEA-DBS gap > 0.01"),
        ("n_episodes_ok", "≥ 150 training episodes"),
        ("dig_enough_episodes", "digitization — enough episodes"),
        ("dig_shared_start_near_paper", "digitization — shared start vs paper"),
        ("dig_baseline_drop_vs_paper", "digitization — baseline drop vs paper"),
        ("dig_sea_drop_vs_paper", "digitization — SEA-DBS drop vs paper"),
        ("dig_sea_steeper_than_baseline_like_paper", "digitization — SEA steeper than baseline"),
        ("dig_sea_below_baseline_late_like_paper", "digitization — SEA below baseline late"),
        ("dig_late_gap_near_paper", "digitization — late gap vs paper"),
        ("dig_late_early_ratio_baseline_near_paper", "digitization — baseline late/early ratio"),
        ("dig_late_early_ratio_sea_near_paper", "digitization — SEA late/early ratio"),
        ("dig_gradual_decline_baseline", "digitization — gradual baseline mid→late drop"),
        ("dig_gradual_decline_sea", "digitization — gradual SEA mid→late drop"),
    ],
    "4b": [
        ("paper_above_baseline_late", "SEA-DBS late reward > baseline"),
        ("paper_pull_ahead_mid", "SEA-DBS ahead in mid training window"),
        ("both_rise", "both series rise from early to late"),
    ],
    "5a": [
        ("n_steps_ok", "10 inference steps"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("baseline_declines", "baseline PSD declines"),
        ("paper_declines", "SEA-DBS PSD declines"),
        ("paper_end_below_baseline", "SEA-DBS end PSD below baseline"),
        ("paper_steeper_drop", "SEA-DBS drop steeper than baseline"),
        ("carrier_hz_ok", "carrier frequency 50 Hz"),
    ],
    "5b": [
        ("n_steps_ok", "10 inference steps"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("baseline_declines", "baseline PSD declines"),
        ("paper_declines", "SEA-DBS PSD declines"),
        ("paper_end_below_baseline", "SEA-DBS end PSD below baseline"),
        ("paper_steeper_drop", "SEA-DBS drop steeper than baseline"),
        ("carrier_hz_ok", "carrier frequency 30 Hz"),
        ("weaker_than_50hz_sea", "30 Hz SEA-DBS weaker suppression than 50 Hz panel"),
        ("weaker_than_50hz_baseline", "30 Hz baseline weaker suppression than 50 Hz panel"),
    ],
    "6": [
        ("four_series_present", "fp32 + PTQ for baseline and SEA-DBS"),
        ("shared_start", "paired series share pre-stim level"),
        ("sea_below_baseline", "SEA-DBS fp32 below baseline fp32 late"),
        ("sea_ptq_below_baseline", "SEA-DBS PTQ below baseline fp32 late"),
        ("sea_ptq_tracks_fp32", "SEA-DBS PTQ tracks fp32"),
        ("baseline_ptq_near_or_above_baseline", "baseline PTQ near/above baseline fp32"),
    ],
    "7": [
        ("four_variants_present", "baseline / +PM / +GS / SEA-DBS"),
        ("sea_dbs_lowest_tail", "SEA-DBS lowest tail mean PSD"),
        ("gs_highest_or_near_highest_tail", "GS highest or near-highest tail"),
        ("pm_not_sea", "PM closer to baseline than to SEA-DBS"),
        ("shared_start", "baseline and SEA-DBS agree at step 0"),
        ("n_steps_ok", "10 inference steps"),
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

RAVIVARAPU_STATUS_NOTES: dict[str, str] = {
    "4a": "v8",
}


@dataclass(frozen=True)
class GateRow:
    key: str
    description: str


@dataclass
class PanelGateStatus:
    panel: str
    pass_field: str
    overall: bool | None
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
            gates={},
            source=f"no manifest at `{manifest_rel}`",
            rows=rows,
        )
    manifest = _load_json(manifest_path)
    gate_values = manifest.get("gates") or {}
    gates = _panel_gate_values(manifest, panel)
    overall = _all_rows_pass(gate_values, panel)
    return PanelGateStatus(
        panel=panel,
        pass_field="pass",
        overall=overall,
        gates=gates,
        source=manifest_rel,
        rows=rows,
    )


def render_gate_block(status: PanelGateStatus) -> str:
    overall = _pass_cell(status.overall)
    lines = [
        f"**Gates set** (`{status.source}`; overall **`{status.pass_field}`**: {overall}, "
        f"{_today()}). Every row is required for exit.",
        "",
        "| Key | Pass |",
        "|-----|------|",
    ]
    for row in status.rows:
        lines.append(
            f"| `{row.key}` — {row.description} | {_pass_cell(status.gates.get(row.key))} |"
        )
    return "\n".join(lines)


def _summary_status_line(status: PanelGateStatus, note: str = "") -> str:
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
        mark = "PASS" if status.overall else ("FAIL" if status.overall is False else "OPEN")
        failed = [k for k, v in status.gates.items() if not v]
        print(f"{panel}: {mark}  failed={failed or '—'}")
    print(f"updated {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
