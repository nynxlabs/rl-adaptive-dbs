"""Evaluate Mehregan panel gates and refresh gate tables in ``replications.md``."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_DIGITIZATION = Path(__file__).resolve().parent
if str(_DIGITIZATION) not in sys.path:
    sys.path.insert(0, str(_DIGITIZATION))

from paper_gates import (  # noqa: E402
    ARTIFACT_ROOT,
    fig1b_gates,
    fig2_time_gates,
    fig4a_gates,
    fig4b_gates,
    fig5_efficacy_gates,
)

_REPO = Path(__file__).resolve().parents[2]
_MEHREGAN_PLOTS = _REPO / "scripts" / "figures" / "papers" / "mehregan"

MEHREGAN_FIG6_PAPER_GATE_DESCRIPTIONS: dict[str, str] = {
    "paper_qat_elevated_vs_fp32": "QAT post-onset mean > fp32",
    "paper_fp32_level_ratio_near_paper": "fp32 post level ratio vs digitized paper",
    "paper_ptq_int8_level_ratio_near_paper": "PTQ int8 post level ratio vs digitized paper",
    "paper_ptq_fp16_level_ratio_near_paper": "PTQ fp16 post level ratio vs digitized paper",
    "paper_qat_level_ratio_near_paper": "QAT post level ratio vs digitized paper",
    "paper_ptq_fp16_near_fp32": "PTQ fp16 post mean within 15% of fp32",
    "paper_ptq_int8_near_fp32": "PTQ int8 post mean within 20% of fp32",
    "paper_not_open_loop_override": "eval uses trained/quantized policy, not open-loop lock",
    "paper_not_shared_constant_action_lock": "fp32+PTQ lack shared identical constant action",
    "paper_qat_late_sustained": "QAT stays elevated late (no end crash)",
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
    header: str
    rows: tuple[GateRow, ...]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fig2_replication(series: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    times = np.asarray(series["time_s"], dtype=float)
    traces = series["traces"]
    return {
        "pd": (times, np.asarray(traces["pd_no_treatment"], dtype=float)),
        "pd_130hz": (times, np.asarray(traces["pd_130hz"], dtype=float)),
    }


def _load_fig5a_pass():
    path = _MEHREGAN_PLOTS / "5a" / "plot.py"
    spec = importlib.util.spec_from_file_location("mehregan_fig5a_plot", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.fig5a_pass


def _load_fig5b_pass():
    path = _MEHREGAN_PLOTS / "5b" / "plot.py"
    spec = importlib.util.spec_from_file_location("mehregan_fig5b_plot", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.fig5b_pass


def _manifest_gates(manifest: dict[str, Any]) -> dict[str, bool]:
    for key in ("gates",):
        g = manifest.get(key)
        if isinstance(g, dict):
            return {k: bool(v) for k, v in g.items() if isinstance(v, bool) and k not in ("pass", "all_pass")}
    panel = manifest.get("panel") or {}
    g = panel.get("gates")
    if isinstance(g, dict):
        return {
            k: bool(v)
            for k, v in g.items()
            if isinstance(v, bool) and k not in ("pass", "all_pass", "gates_pass")
        }
    summary = manifest.get("summary") or {}
    g = summary.get("gates")
    if isinstance(g, dict):
        return {k: bool(v) for k, v in g.items() if isinstance(v, bool)}
    return {}


def _manifest_overall(manifest: dict[str, Any], pass_field: str) -> bool | None:
    for container in (manifest, manifest.get("panel") or {}, manifest.get("summary") or {}):
        if pass_field in container and isinstance(container[pass_field], bool):
            return bool(container[pass_field])
        g = container.get("gates")
        if isinstance(g, dict) and pass_field in g and isinstance(g[pass_field], bool):
            return bool(g[pass_field])
    gates = _manifest_gates(manifest)
    if not gates:
        return None
    return all(gates.values())


def evaluate_1b() -> PanelGateStatus:
    curves_path = ARTIFACT_ROOT / "1b" / "curves.json"
    payload = _load_json(curves_path)
    replication = {
        c["key"]: (
            np.asarray(c["freqs_hz"], dtype=float),
            np.asarray(c["psd"], dtype=float),
        )
        for c in payload["curves"]
    }
    dig = fig1b_gates(replication)
    rows = (
        GateRow("pd_gt_healthy", "beta-band PD > healthy"),
        GateRow("pd_130_lt_pd", "130 Hz cDBS < untreated PD"),
        GateRow("suppression_ratio_near_paper", "pd_130/pd ratio vs digitized paper"),
        GateRow("healthy_beta_near_paper", "healthy level within 30% of paper"),
        GateRow("pd_130hz_beta_near_paper", "treated level within 30% of paper"),
    )
    return PanelGateStatus(
        panel="1b",
        pass_field="gates_pass",
        overall=bool(dig["pass"]),
        gates=dict(dig["gates"]),
        source=str(curves_path),
        header="`fig1b_gates` → manifest `gates` / `gates_pass`",
        rows=rows,
    )


def evaluate_2a() -> PanelGateStatus:
    series_path = ARTIFACT_ROOT / "2a" / "series.json"
    dig = fig2_time_gates(_fig2_replication(_load_json(series_path)), panel="2a")
    rows = (
        GateRow("prestim_shared", "treated/untreated agree pre-onset (≤5% rel)"),
        GateRow("treated_below_untreated_late", "cDBS below no-treatment after t=2"),
        GateRow("late_ratio_near_paper", "late treated/untreated ratio vs digitization"),
        GateRow("suppression_drop_near_paper", "drop magnitude vs digitization"),
    )
    return PanelGateStatus(
        panel="2a",
        pass_field="gates_pass",
        overall=bool(dig["pass"]),
        gates=dict(dig["gates"]),
        source=str(series_path),
        header="`fig2_time_gates`, panel `2a`",
        rows=rows,
    )


def evaluate_2b() -> PanelGateStatus:
    series_path = ARTIFACT_ROOT / "2b" / "series.json"
    dig = fig2_time_gates(_fig2_replication(_load_json(series_path)), panel="2b")
    rows = (
        GateRow("prestim_shared", "treated/untreated agree pre-onset (≤5% rel)"),
        GateRow("treated_below_untreated_late", "cDBS below no-treatment after t=2"),
        GateRow("late_ratio_near_paper", "late treated/untreated ratio vs digitization"),
        GateRow("suppression_drop_near_paper", "drop magnitude vs digitization"),
    )
    return PanelGateStatus(
        panel="2b",
        pass_field="gates_pass",
        overall=bool(dig["pass"]),
        gates=dict(dig["gates"]),
        source=str(series_path),
        header="`fig2_time_gates`, panel `2b`",
        rows=rows,
    )


def evaluate_4a() -> PanelGateStatus:
    series_path = ARTIFACT_ROOT / "4a" / "series.json"
    trace = _load_json(series_path)["beta_norm_trace"]
    dig = fig4a_gates(trace)
    rows = (
        GateRow("plot_style", "300 training steps"),
        GateRow("overall_trend_down", "end window mean < start window mean"),
        GateRow("drop_vs_paper", "drop ≥ 70% of digitized paper drop"),
        GateRow("late_early_ratio_near_paper", "late/early ratio vs digitization"),
        GateRow("mid_fade_vs_paper", "mid [120,150] fade ≥ 50% of paper mid-drop"),
    )
    return PanelGateStatus(
        panel="4a",
        pass_field="gates_pass",
        overall=bool(dig["pass"]),
        gates=dict(dig["gates"]),
        source=str(series_path),
        header="`fig4a_gates` → live `series.json` (digitization revisit; was locked `series_v18.json`)",
        rows=rows,
    )


def evaluate_4b() -> PanelGateStatus:
    manifest_path = ARTIFACT_ROOT / "4b" / "manifest.json"
    manifest = _load_json(manifest_path)
    dig = fig4b_gates(manifest["episode_rewards"], manifest["episode_mean_beta"])
    gates = dict(dig["gates"])
    gates["plot_style"] = len(manifest["episode_rewards"]) >= 2
    summary_gates = (manifest.get("summary") or {}).get("gates") or {}
    if "automation" in summary_gates:
        gates["automation"] = bool(summary_gates["automation"])
    else:
        gates["automation"] = bool((manifest.get("summary") or {}).get("automation_pass", True))
    rows = (
        GateRow("early_negative", "mean reward ep 0–2 < 0"),
        GateRow("reward_rises", "late mean reward > early mean"),
        GateRow("late_plateau_improved", "late mean reward > −10"),
        GateRow("rise_timing", "reward exceeds ep0 + 10 by ep ≤ 6"),
        GateRow("beta_drops", "late episode-mean PSD < early"),
        GateRow("beta_drop_ratio_near_paper", "PSD late/early ratio vs digitization"),
        GateRow("reward_recovers_like_paper", "qualitative rise (not magnitude match)"),
        GateRow("late_beta_above_threshold", "late episode-mean PSD ≥ β_t=0.35"),
        GateRow("late_beta_near_paper", "late PSD within 15% of digitized paper"),
        GateRow("late_reward_near_zero", "late mean reward in (−10, 2] (paper ~−2)"),
        GateRow("plot_style", "≥ 2 episodes plotted"),
        GateRow("automation", "manifest summary.automation_pass mirrors fig4b bundle"),
    )
    return PanelGateStatus(
        panel="4b",
        pass_field="gates_pass",
        overall=all(gates.values()),
        gates=gates,
        source=str(manifest_path),
        header="`fig4b_gates` + legacy `_fig4b_pass` → manifest `summary.gates`",
        rows=rows,
    )


def evaluate_5a() -> PanelGateStatus:
    manifest_path = ARTIFACT_ROOT / "5a" / "manifest.json"
    manifest = _load_json(manifest_path)
    panel = manifest.get("panel") or {}
    fig5a_pass = _load_fig5a_pass()
    result = fig5a_pass(panel)
    gates = {k: bool(v) for k, v in result.items() if isinstance(v, bool) and k != "pass"}
    rows = (
        GateRow("shared_baseline", "no-stim vs periodic pre-onset Δ < 25"),
        GateRow("trained_below_no_stim", "trained post-onset mean < no stim"),
        GateRow("trained_above_periodic", "trained > periodic 45 Hz"),
        GateRow("cdbs_lowest", "130 Hz cDBS lowest of four series"),
        GateRow("trained_no_stim_ratio_near_paper", "late ratio vs digitized paper"),
        GateRow("periodic_no_stim_ratio_near_paper", "late ratio vs digitized paper"),
    )
    return PanelGateStatus(
        panel="5a",
        pass_field="pass",
        overall=bool(result.get("pass")),
        gates=gates,
        source=str(manifest_path),
        header="`fig5a_pass` / `fig5_efficacy_gates` → manifest `gates`",
        rows=rows,
    )


def evaluate_5b() -> PanelGateStatus:
    manifest_path = ARTIFACT_ROOT / "5b" / "manifest.json"
    manifest = _load_json(manifest_path)
    panel = manifest.get("panel") or {}
    fig5b_pass = _load_fig5b_pass()
    result = fig5b_pass(panel)
    gates = {k: bool(v) for k, v in result.items() if isinstance(v, bool) and k != "pass"}
    rows = (
        GateRow("shared_baseline", "no-stim vs periodic pre-onset Δ < 25"),
        GateRow("trained_below_no_stim", "trained post-onset mean < no stim"),
        GateRow("trained_below_periodic", "trained < periodic 30 Hz"),
        GateRow("periodic_above_no_stim", "periodic 30 Hz elevates beta vs no stim"),
        GateRow("trained_no_stim_ratio_near_paper", "late ratio vs digitized paper"),
        GateRow("periodic_no_stim_ratio_near_paper", "late ratio vs digitized paper"),
    )
    return PanelGateStatus(
        panel="5b",
        pass_field="pass",
        overall=bool(result.get("pass")),
        gates=gates,
        source=str(manifest_path),
        header="`fig5b_pass` / `fig5_efficacy_gates` → manifest `gates`",
        rows=rows,
    )


def _evaluate_6(panel: str) -> PanelGateStatus:
    manifest_path = ARTIFACT_ROOT / panel / "manifest.json"
    manifest = _load_json(manifest_path)
    gates = _manifest_gates(manifest)
    overall = _manifest_overall(manifest, "all_pass")
    rows = (
        GateRow("prestim_shared", "all series agree pre-onset (≤1 PSD unit vs fp32)"),
        GateRow("prestim_wiggly", "fp32 pre-onset std ≥ 5"),
        GateRow("fp32_suppresses_vs_baseline", "fp32 post-onset < pre-stim baseline"),
        GateRow("ptq-fp16_tracks_fp32", "PTQ fp16 post mean within tolerance of fp32"),
        GateRow("ptq-int8_tracks_fp32", "PTQ int8 post mean within tolerance of fp32"),
        GateRow("non_qat_traces_distinct", "fp32 / PTQ fp16 / PTQ int8 not identical post-onset"),
        GateRow("qat_elevated_vs_fp32", "QAT post-onset > fp32"),
        GateRow("qat_near_baseline_band", "QAT in elevated pre-stim band, not suppressed"),
        GateRow("not_shared_constant_action_lock", "fp32+PTQ do not share one constant action"),
    )
    rows = rows + tuple(
        GateRow(key, MEHREGAN_FIG6_PAPER_GATE_DESCRIPTIONS[key])
        for key in (
            "paper_qat_elevated_vs_fp32",
            "paper_fp32_level_ratio_near_paper",
            "paper_ptq_int8_level_ratio_near_paper",
            "paper_ptq_fp16_level_ratio_near_paper",
            "paper_qat_level_ratio_near_paper",
            "paper_ptq_fp16_near_fp32",
            "paper_ptq_int8_near_fp32",
            "paper_not_open_loop_override",
        )
    )
    if panel == "6a":
        rows = rows + (
            GateRow(
                "paper_not_shared_constant_action_lock",
                MEHREGAN_FIG6_PAPER_GATE_DESCRIPTIONS["paper_not_shared_constant_action_lock"],
            ),
            GateRow(
                "paper_qat_late_sustained",
                MEHREGAN_FIG6_PAPER_GATE_DESCRIPTIONS["paper_qat_late_sustained"],
            ),
        )
    return PanelGateStatus(
        panel=panel,
        pass_field="all_pass",
        overall=overall,
        gates=gates,
        source=str(manifest_path),
        header="`_gate_summary` → manifest `gates`",
        rows=rows,
    )


EVALUATORS = {
    "1b": evaluate_1b,
    "2a": evaluate_2a,
    "2b": evaluate_2b,
    "4a": evaluate_4a,
    "4b": evaluate_4b,
    "5a": evaluate_5a,
    "5b": evaluate_5b,
    "6a": lambda: _evaluate_6("6a"),
    "6b": lambda: _evaluate_6("6b"),
}

MEHREGAN_SUMMARY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("1b", "Fig 1b", "GPi PSD"),
    ("2a", "Fig 2a", "GPi $P_" + "\\beta$ time series"),
    ("2b", "Fig 2b", "Error Index time series"),
    ("4a", "Fig 4a", "Training $P_" + "\\beta$ vs step"),
    ("4b", "Fig 4b", "Training reward vs episode"),
    ("5a", "Fig 5a", "Post-train efficacy @ 45 Hz"),
    ("5b", "Fig 5b", "Post-train efficacy @ 30 Hz"),
    ("6a", "Fig 6a", "PTQ / QAT @ 45 Hz"),
    ("6b", "Fig 6b", "PTQ / QAT @ 30 Hz"),
)

MEHREGAN_STATUS_NOTES: dict[str, str] = {
    "4a": "τ 3→1.0, locked train v18",
    "4b": "paired train v18, v14",
    "5b": "burst alphabet, locked eval v3",
    "6a": "honest trailing eval",
    "6b": "tier PTQ",
}

MEHREGAN_MANIFEST_PATHS: dict[str, Path] = {
    panel: ARTIFACT_ROOT / panel / "manifest.json"
    for panel in ("1b", "2a", "2b", "4a", "4b", "5a", "5b", "6a", "6b")
}

_REP_VERSION_RE = re.compile(r"rep v\d+")


def _replication_png_version(panel: str) -> int | None:
    manifest_path = MEHREGAN_MANIFEST_PATHS.get(panel)
    if manifest_path is None or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = manifest.get("png_version")
    if isinstance(version, int):
        return version
    output_png = manifest.get("output_png")
    if not isinstance(output_png, str):
        return None
    match = re.search(r"_v(\d+)\.png$", output_png)
    return int(match.group(1)) if match else None


def _summary_note(panel: str) -> str:
    base = MEHREGAN_STATUS_NOTES.get(panel, "")
    version = _replication_png_version(panel)
    if version is None:
        return base
    rep = f"rep v{version}"
    if not base:
        return rep
    if _REP_VERSION_RE.search(base):
        return _REP_VERSION_RE.sub(rep, base)
    return f"{base}, {rep}"


def _pass_cell(value: bool | None) -> str:
    if value is None:
        return "—"
    return "yes" if value else "no"


def _summary_status_line(status: PanelGateStatus, note: str = "") -> str:
    if status.overall:
        return f"Pass ({note})" if note else "Pass"
    failed = [key for key, value in status.gates.items() if not value]
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
    for panel_key, label, description in MEHREGAN_SUMMARY_ROWS:
        status = statuses[panel_key]
        note = _summary_note(panel_key)
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
        raise ValueError("missing summary markers in Mehregan replications doc")

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n{block}\n{match.group(3)}"

    return pattern.sub(_repl, text, count=1)


def render_gate_block(status: PanelGateStatus) -> str:
    overall = _pass_cell(status.overall)
    lines = [
        f"**Gates set** ({status.header}). Overall **`{status.pass_field}`**: {overall} "
        f"(from `{status.source}`, {_today()}). Every row is required for exit.",
        "",
        "| Key | Description | Pass |",
        "|-----|-------------|------|",
    ]
    for row in status.rows:
        gate_val = status.gates.get(row.key)
        lines.append(f"| `{row.key}` | {row.description} | {_pass_cell(gate_val)} |")
    return "\n".join(lines)


def _today() -> str:
    return date.today().isoformat()


def refresh_gate_tables(doc_path: Path) -> dict[str, PanelGateStatus]:
    """Replace ``<!-- gates-{panel}:start/end -->`` blocks in the tracker doc."""
    text = doc_path.read_text()
    statuses: dict[str, PanelGateStatus] = {}
    for panel, evaluate in EVALUATORS.items():
        status = evaluate()
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


def main() -> int:
    promote_path = _REPO / "scripts" / "figures" / "papers" / "promote.py"
    spec = importlib.util.spec_from_file_location("figure_promote", promote_path)
    assert spec is not None and spec.loader is not None
    promote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(promote)
    doc = promote.resolve_paper_1_doc()
    statuses = refresh_gate_tables(doc)
    for panel, status in statuses.items():
        mark = "PASS" if status.overall else "FAIL"
        failed = [k for k, v in status.gates.items() if not v]
        print(f"{panel}: {mark}  failed={failed or '—'}")
    print(f"updated {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
