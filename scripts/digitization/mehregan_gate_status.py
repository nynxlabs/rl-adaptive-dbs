"""Evaluate Mehregan panel gates and refresh gate tables in ``replications.md``."""

from __future__ import annotations

import functools
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
    FIG4B_EPISODES,
    fig1b_gates,
    fig2_time_gates,
    fig4a_gates,
    fig4b_gates,
    fig5_efficacy_gates,
)

_REPO = Path(__file__).resolve().parents[2]
_MEHREGAN_PLOTS = _REPO / "scripts" / "figures" / "papers" / "mehregan"

MEHREGAN_FIG6_GATE_ROWS: tuple[tuple[str, str], ...] = (
    ("all_closed_loop", "fp32, PTQ and every QAT seed each run their own actor closed loop"),
    ("prestim_shared", "all series and QAT seeds share the pre-onset level (spread ≤ 1%)"),
    ("fp32_suppresses_vs_baseline", "fp32 late mean (t ≥ 4 s) < its pre-onset mean"),
    ("ptq_fp16_near_fp32", "PTQ fp16 late mean within 15% of fp32"),
    ("ptq_int8_near_fp32", "PTQ int8 late mean within 20% of fp32"),
    ("qat_elevated_vs_fp32", "median-seed QAT late mean > fp32"),
    ("fp32_level_near_paper", "fp32 late/pre within 20% of digitized paper"),
    ("ptq_fp16_level_near_paper", "PTQ fp16 late/pre within 20% of digitized paper"),
    ("ptq_int8_level_near_paper", "PTQ int8 late/pre within 20% of digitized paper"),
    ("qat_majority_not_below_paper", "QAT late/pre ≥ 80% of digitized paper in a strict majority of QAT training seeds"),
    ("qat_median_near_paper", "median-seed QAT late/pre within 20% of digitized paper (figure: same range)"),
    ("qat_late_sustained", "median QAT trace [10,12] s mean ≥ 90% of its [2,8] s mean (no late fade)"),
    ("fp32_onset_near_paper", "fp32 [2,4] s mean / pre-onset within 20% of digitized paper (onset speed)"),
    ("fp32_shape_r", "Pearson r fp32 trace vs digitized paper on a common [0,12] s grid ≥ 0.6"),
    ("prestim_level_near_paper", "fp32 pre-onset level within 10% of digitized paper"),
    ("qat_fp32_gap_vs_paper", "(median-seed QAT − fp32) late / pre ≥ 0.5× the digitized paper gap"),
    ("ptq_int8_alters_trace", "mean |PTQ int8 − fp32| over t ≥ 2 s ≥ 0.5× digitized paper (quantization alters outputs)"),
    ("ptq_fp16_alters_trace", "mean |PTQ fp16 − fp32| over t ≥ 2 s ≥ 0.5× digitized paper (quantization alters outputs)"),
)

MEHREGAN_FIG2_GATE_ROWS: tuple[tuple[str, str], ...] = (
    ("prestim_shared", "treated/untreated agree pre-onset (≤5% rel)"),
    ("treated_below_untreated_late", "cDBS below no-treatment after t=2"),
    ("late_ratio_near_paper", "late treated/untreated ratio vs digitization"),
    ("suppression_drop_near_paper", "drop magnitude vs digitization"),
    ("untreated_flat", "untreated late (t ≥ 6 s) / pre within ±0.15 (absolute) of digitized paper"),
    ("treated_late_over_pre_near_paper", "treated late / pre within 30% of digitized paper"),
    ("prestim_level_near_paper", "untreated pre-onset level within 15% of digitized paper"),
    ("treated_mid_ratio_near_paper", "treated/untreated mean on [4,7] s within 30% of digitized paper"),
    ("t50_near_paper", "treated trace reaches half its pre→late drop within ±1 s of digitized paper"),
    ("treated_shape_r", "Pearson r treated trace vs digitized paper on a common [0,12] s grid ≥ 0.7"),
    ("gap_shape_r", "Pearson r of the untreated − treated gap vs digitized paper on [0,12] s ≥ 0.7"),
)

MEHREGAN_FIG5_SHARED_ROWS_HEAD: tuple[tuple[str, str], ...] = (
    ("traces_present", "eval traces available for the trace-level gates"),
    ("shared_baseline", "pre-onset spread across all series (incl. trained) ≤ 3% of the level"),
    ("trained_closed_loop", "trained series runs the actor closed loop (no replayed actions)"),
    ("trained_below_no_stim", "trained mean (t ≥ 4 s) < no stim"),
)

MEHREGAN_FIG5_SHARED_ROWS_TAIL: tuple[tuple[str, str], ...] = (
    ("trained_no_stim_ratio_near_paper", "trained/no-stim late ratio (t ≥ 4 s) vs digitized paper"),
    ("periodic_no_stim_ratio_near_paper", "periodic/no-stim late ratio (t ≥ 4 s) vs digitized paper"),
    ("prestim_level_near_paper", "mean pre-onset level within 10% of digitized paper"),
    ("trained_onset_ratio_near_paper", "trained/no-stim on [2,4] s within 30% of digitized paper (onset speed)"),
    ("trained_shape_r", "Pearson r trained trace vs digitized paper on a common [0,12] s grid ≥ 0.6"),
    ("periodic_shape_r", "Pearson r periodic trace vs digitized paper on a common [0,12] s grid ≥ 0.6"),
)


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


@functools.cache
def _load_script(name: str, path: Path):
    from rl_adaptive_dbs.panel import load_script_module

    return load_script_module(name, path)


def _load_efficacy_panel():
    return _load_script("mehregan_efficacy_panel", _MEHREGAN_PLOTS / "efficacy_panel.py")


def _load_quant_panel():
    return _load_script("mehregan_quant_panel", _MEHREGAN_PLOTS / "quant_panel.py")


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
        GateRow("healthy_pd_ratio_near_paper", "healthy/PD beta ratio within 30% of digitized paper"),
        GateRow("pd_beta_near_paper", "PD beta level within 30% of digitized paper"),
        GateRow("pd_peak_freq_near_paper", "PD spectral peak (1.5 Hz smoothed) within ±4 Hz of digitized paper"),
        GateRow("low_over_beta_near_paper", "every series' [1,12]/[13,35] Hz power ratio within 35% of digitized paper"),
        GateRow("pd_shape_r", "Pearson r PD PSD vs digitized paper on a common 1–50 Hz grid ≥ 0.6"),
        GateRow("pd130_shape_r", "Pearson r 130 Hz PSD vs digitized paper on a common 1–50 Hz grid ≥ 0.3"),
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
    rows = tuple(GateRow(k, d) for k, d in MEHREGAN_FIG2_GATE_ROWS)
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
    rows = tuple(GateRow(k, d) for k, d in MEHREGAN_FIG2_GATE_ROWS)
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
        GateRow("mid_fade_vs_paper", "mid [120,150] fade (vs [0,100]) within 0.5–2× the digitized paper mid fade"),
        GateRow("ep0_near_paper", "steps 0–29 mean within 15% of digitized paper ep0 (seed spread)"),
        GateRow("episode_shape_r", "Pearson r of the 10 episode (30-step) means vs digitized paper ≥ 0.7"),
        GateRow("late_level_near_paper", "episodes 5–9 mean within 15% of digitized paper"),
        GateRow("episode_drop_near_paper", "drop eps 0–3 → eps 6–9 within 0.6–1.6× digitized paper"),
        GateRow("no_late_rebound", "eps 7–9 mean ≤ eps 5–6 mean + 0.02"),
        GateRow("half_drop_episode_near_paper", "first episode at half the eps 0–3 → 6–9 drop within ±1 of digitized paper"),
        GateRow("per_episode_near_paper", "every episode mean within 20% of digitized paper"),
    )
    return PanelGateStatus(
        panel="4a",
        pass_field="gates_pass",
        overall=bool(dig["pass"]),
        gates=dict(dig["gates"]),
        source=str(series_path),
        header="`fig4a_gates` → live `series.json`",
        rows=rows,
    )


def _fig4b_inputs(series: dict[str, Any], n_episodes: int) -> tuple[list[float], list[float]]:
    """Episode rewards + episode-mean PSD from the Fig 4a series cache."""
    training = series.get("training") or {}
    rewards = training.get("episode_rewards") or series.get("episode_rewards") or []
    trace = np.asarray(series.get("beta_norm_trace") or [], dtype=float)
    spe = int(series.get("steps_per_episode") or 30)
    n = min(n_episodes, len(rewards), trace.size // spe)
    beta = trace[: n * spe].reshape(n, spe).mean(axis=1) if n else np.array([])
    return [float(r) for r in rewards[:n]], beta.tolist()


def evaluate_4b() -> PanelGateStatus:
    # Fig 4b replots the Fig 4a run: gate the live 4a cache (all 10 episodes).
    series_path = ARTIFACT_ROOT / "4a" / "series.json"
    rewards, beta = _fig4b_inputs(_load_json(series_path), FIG4B_EPISODES)
    dig = fig4b_gates(rewards, beta)
    gates = dict(dig["gates"])
    rows = (
        GateRow("plot_style", "10 episodes plotted (paper: 10 vertices on its 0–8 axis, x = 8i/9)"),
        GateRow("early_negative", "mean reward ep 0–2 < 0"),
        GateRow("reward_rises", "late (eps 6–9) mean reward > early mean"),
        GateRow("late_plateau_improved", "late mean reward > −10"),
        GateRow("rise_timing", "reward exceeds ep0 + 10 by ep ≤ 6"),
        GateRow("beta_drops", "late episode-mean PSD < early"),
        GateRow("beta_drop_ratio_near_paper", "PSD late/early ratio vs digitization"),
        GateRow("reward_recovers_like_paper", "qualitative rise in ours and paper"),
        GateRow("late_beta_near_paper", "late PSD within 15% of digitized paper"),
        GateRow("late_reward_near_zero", "late mean reward in (−10, 2] (paper ~−3)"),
        GateRow("ep0_beta_near_paper", "episode 0 PSD within 15% of digitized paper (seed spread)"),
        GateRow("reward_shape_r", "Pearson r per-episode reward vs digitized paper ≥ 0.7"),
        GateRow("psd_shape_r", "Pearson r per-episode PSD vs digitized paper ≥ 0.7"),
        GateRow("early_reward_near_paper", "eps 0–2 mean reward within 50% of digitized paper"),
        GateRow("reward_floor_not_deeper", "min episode reward ≥ 1.5× digitized paper min"),
        GateRow("max_reward_le_cap", "max episode reward ≤ +5 (paper never above ≈ −2)"),
        GateRow("psd_floor_not_deeper", "min episode PSD ≥ 0.9× digitized paper min"),
        GateRow("per_episode_psd_near_paper", "every episode PSD within 20% of digitized paper"),
    )
    return PanelGateStatus(
        panel="4b",
        pass_field="gates_pass",
        overall=all(gates.values()),
        gates=gates,
        source=str(series_path),
        header="`fig4b_gates` → Fig 4a `series.json` (10 episodes)",
        rows=rows,
    )


def _evaluate_5(panel: str) -> PanelGateStatus:
    eff = _load_efficacy_panel()
    cfg = eff.FIG5A if panel == "5a" else eff.FIG5B
    manifest_path = ARTIFACT_ROOT / panel / "manifest.json"
    eval_path = ARTIFACT_ROOT / panel / "eval.json"
    if eval_path.exists():
        payload = _load_json(eval_path)
        means = eff.panel_means(payload)
        result = eff.efficacy_gates(cfg, means, eff.payload_traces(payload))
        source = eval_path
    else:
        result = eff.efficacy_gates(cfg, _load_json(manifest_path).get("panel") or {})
        source = manifest_path
    gates = {k: bool(v) for k, v in result.items() if isinstance(v, bool) and k != "pass"}
    if panel == "5a":
        mid = (
            ("trained_above_periodic", "trained > periodic 45 Hz"),
            ("cdbs_lowest", "130 Hz cDBS lowest of four series"),
        )
        extra = (
            ("trained_periodic_gap_near_paper", "(trained − periodic)/no-stim late within 0.4–2× digitized paper"),
            ("cdbs_no_stim_ratio_near_paper", "130 Hz/no-stim late ratio within 30% of digitized paper"),
        )
    else:
        mid = (
            ("trained_below_periodic", "trained < periodic 30 Hz"),
            ("periodic_above_no_stim", "periodic 30 Hz elevates beta vs no stim"),
        )
        extra = (
            ("trained_periodic_gap_near_paper", "trained-below-periodic gap (/no-stim, late) within 0.4–2× digitized paper"),
        )
    keys = MEHREGAN_FIG5_SHARED_ROWS_HEAD + mid + MEHREGAN_FIG5_SHARED_ROWS_TAIL + extra
    if gates.get("traces_present", True):
        keys = tuple(r for r in keys if r[0] != "traces_present")
    return PanelGateStatus(
        panel=panel,
        pass_field="pass",
        overall=bool(result.get("pass")),
        gates=gates,
        source=str(source),
        header="`efficacy_gates` / `fig5_efficacy_gates` → `eval.json`",
        rows=tuple(GateRow(k, d) for k, d in keys),
    )


def _evaluate_6(panel: str) -> PanelGateStatus:
    manifest_path = ARTIFACT_ROOT / panel / "manifest.json"
    eval_path = ARTIFACT_ROOT / panel / "eval.json"
    if eval_path.exists():
        # Recompute from the cached eval so new gates show without a re-run.
        quant = _load_quant_panel()
        cfg = quant.FIG6A if panel == "6a" else quant.FIG6B
        dig = quant.quant_gates(cfg, _load_json(eval_path))
        gates = dict(dig["gates"])
        overall: bool | None = bool(dig["pass"])
        source = eval_path
    else:
        manifest = _load_json(manifest_path)
        gates = _manifest_gates(manifest)
        overall = _manifest_overall(manifest, "all_pass")
        source = manifest_path
    rows = tuple(GateRow(key, desc) for key, desc in MEHREGAN_FIG6_GATE_ROWS)
    return PanelGateStatus(
        panel=panel,
        pass_field="all_pass",
        overall=overall,
        gates=gates,
        source=str(source),
        header="`fig6_quant_gates` → `eval.json`",
        rows=rows,
    )


EVALUATORS = {
    "1b": evaluate_1b,
    "2a": evaluate_2a,
    "2b": evaluate_2b,
    "4a": evaluate_4a,
    "4b": evaluate_4b,
    "5a": lambda: _evaluate_5("5a"),
    "5b": lambda: _evaluate_5("5b"),
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
    "4a": "paper Alg. 1",
    "4b": "paired to 4a",
    "5a": "closed loop, Fig 4a model",
    "5b": "closed loop, burst alphabet",
    "6a": "closed loop, int8 QAT",
    "6b": "closed loop, int8 QAT",
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
