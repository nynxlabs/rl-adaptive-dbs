"""Replication checklist tests."""

from __future__ import annotations

from controllers.ddpg.checklist import assess_replication_summary


def test_checklist_passes_when_ddpg_beats_none() -> None:
    summary = {
        "variant": "paper",
        "ddpg_eval": {"protocol": "mehregan_eval", "p_beta_mean": 100.0},
        "baselines": {
            "none": {"p_beta_mean": 500.0},
            "cdbs-130hz": {"p_beta_mean": 150.0},
        },
    }
    report = assess_replication_summary(summary)
    assert report["passed"] is True


def test_checklist_fails_when_ddpg_worse_than_none() -> None:
    summary = {
        "variant": "paper",
        "ddpg_eval": {"protocol": "mehregan_eval", "p_beta_mean": 600.0},
        "baselines": {"none": {"p_beta_mean": 500.0}, "cdbs-130hz": {"p_beta_mean": 150.0}},
    }
    report = assess_replication_summary(summary)
    assert report["passed"] is False
