"""Mehregan replication checklist (§IV qualitative claims)."""

from __future__ import annotations

from typing import Any


def assess_replication_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Score a ``ReplicationResult.summary()`` or compatible JSON payload."""
    ddpg = summary.get("ddpg_eval") or {}
    baselines = summary.get("baselines") or {}
    variant = str(summary.get("variant", "paper"))

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    p_beta_ddpg = float(ddpg.get("p_beta_mean", float("nan")))
    p_beta_none = float((baselines.get("none") or {}).get("p_beta_mean", float("nan")))
    p_beta_cdbs = float((baselines.get("cdbs-130hz") or {}).get("p_beta_mean", float("nan")))

    add(
        "ddpg_lowers_p_beta_vs_none",
        p_beta_ddpg < p_beta_none,
        f"ddpg p_beta_mean={p_beta_ddpg:.1f} vs none={p_beta_none:.1f}",
    )
    add(
        "ddpg_beats_or_matches_cdbs",
        p_beta_ddpg <= p_beta_cdbs * 1.05,
        f"ddpg p_beta_mean={p_beta_ddpg:.1f} vs cdbs={p_beta_cdbs:.1f} (5% slack)",
    )
    add(
        "mehregan_eval_protocol",
        ddpg.get("protocol") == "mehregan_eval",
        f"protocol={ddpg.get('protocol')!r}",
    )
    if variant in {"ptq-fp16", "ptq-int8"}:
        extra = ddpg.get("metrics_extra") or {}
        add(
            "quantization_tagged",
            extra.get("quantization") == variant,
            f"metrics_extra.quantization={extra.get('quantization')!r}",
        )

    passed = all(item["passed"] for item in checks)
    return {
        "variant": variant,
        "passed": passed,
        "checks": checks,
        "paper_notes": (
            "Qualitative §IV: learned policy should suppress beta vs unstimulated; "
            "PTQ should track full-precision; QAT may underperform at 10 episodes."
        ),
    }
