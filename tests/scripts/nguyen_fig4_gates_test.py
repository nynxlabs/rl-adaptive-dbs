"""Nguyen Fig 4 gates: paper self-consistency, scale freedom, failure cases, key completeness."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DIG = _ROOT / "scripts" / "digitization"
sys.path.insert(0, str(_DIG))

import nguyen_gates as ng  # noqa: E402

ARTIFACT = Path("artifacts/figures/papers/nguyen/paper_digitization")
pytestmark = pytest.mark.skipif(
    not (ARTIFACT / "curves_fig4_reward.json").exists(), reason="no digitization"
)

EPISODES = np.arange(ng.FIG4_EPISODES, dtype=float)


def _paper(shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Digitized paper Smoothed curves per episode, optionally shifted later by ``shift`` episodes."""
    ref = ng.fig4_paper_reference()
    rewards = np.interp(EPISODES - shift, EPISODES, ref["reward_grid"])
    lengths = np.interp(EPISODES - shift, EPISODES, ref["length_grid"])
    return rewards, lengths


def _required_values(report: dict) -> dict[str, bool]:
    return {
        f"{group}.{key}": report[group][key]
        for group, keys in ng.FIG4_REQUIRED_KEYS.items()
        for key in keys
    }


def test_paper_curve_passes_all_required_gates():
    rewards, lengths = _paper()
    report = ng.fig4_gates(rewards, lengths, max_episode_steps=25)
    assert all(_required_values(report).values()), _required_values(report)
    assert report["pass"] and report["shape_pass"]
    assert report["reward"]["tier2"] == {"reward_leads_length": True, "reward_scale_raw_d": True}


@pytest.mark.parametrize("shift", [-10.0, 10.0])
def test_paper_shifted_ten_episodes_passes(shift: float):
    rewards, lengths = _paper(shift)
    report = ng.fig4_gates(rewards, lengths, max_episode_steps=25)
    assert report["pass"], _required_values(report)
    assert not ng.fig4_abort_check(rewards[:150], lengths[:150])["abort"]


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_reward_gates_are_scale_free(scale: float):
    rewards, lengths = _paper()
    report = ng.fig4_gates(rewards * scale, lengths, max_episode_steps=25)
    assert report["pass"], _required_values(report)
    # The absolute-scale check is tier 2: it flags the rescale but never blocks pass.
    assert report["reward"]["tier2"]["reward_scale_raw_d"] is False
    assert not ng.fig4_abort_check(rewards[:150] * scale, lengths[:150])["abort"]


def test_flat_no_learning_fails_and_aborts():
    lengths = np.full(ng.FIG4_EPISODES, 25.0)
    rewards = np.full(ng.FIG4_EPISODES, -5.0e5)
    report = ng.fig4_gates(rewards, lengths, max_episode_steps=25)
    assert not report["pass"]
    assert not report["shape_pass"]
    assert not report["length"]["length_t50_timing"]
    assert not report["reward"]["reward_relative_gain"]
    probe = ng.fig4_abort_check(rewards[:150], lengths[:150])
    assert probe["decidable"] and probe["abort"]
    assert "abort_length_progress_120_150" in probe["failed"]


def test_late_collapse_with_timeouts_fails():
    rewards, lengths = _paper()
    rewards = rewards.copy()
    lengths = lengths.copy()
    lengths[300:] = 25.0
    rewards[300:] = float(np.mean(rewards[:50]))
    report = ng.fig4_gates(rewards, lengths, max_episode_steps=25)
    assert not report["pass"]
    for key in ("length_late_level", "late_timeouts", "length_late_plateau_hold"):
        assert not report["length"][key], key
    assert not report["reward"]["reward_late_plateau_hold"]


def test_abort_window_not_decidable_before_150():
    rewards, lengths = _paper()
    probe = ng.fig4_abort_check(rewards[:149], lengths[:149])
    assert not probe["decidable"] and not probe["abort"]


@pytest.mark.parametrize("n", [0, 5, 120, 200, 500])
def test_every_required_and_tier2_key_is_computed(n: int):
    rewards, lengths = _paper()
    report = ng.fig4_gates(rewards[:n], lengths[:n], max_episode_steps=25)
    for group, keys in ng.FIG4_REQUIRED_KEYS.items():
        for key in keys:
            assert isinstance(report[group][key], bool), (group, key)
        for key in ng.FIG4_TIER2_KEYS[group]:
            assert isinstance(report[group]["tier2"][key], bool), (group, key)
    if n < ng.FIG4_EPISODES:
        assert not report["pass"]


def test_shape_keys_are_required_keys():
    for group, keys in ng.FIG4_SHAPE_KEYS.items():
        assert set(keys) <= set(ng.FIG4_REQUIRED_KEYS[group])
        assert not set(ng.FIG4_TIER2_KEYS[group]) & set(ng.FIG4_REQUIRED_KEYS[group])


def test_missing_required_key_raises():
    gates = {key: True for key in ng.FIG4_LENGTH_REQUIRED_KEYS[:-1]}
    with pytest.raises(ng.Fig4GateKeyError):
        ng.gate_group_pass(gates, ng.FIG4_LENGTH_REQUIRED_KEYS, group="length")


def test_promote_tier_map_matches_gate_keys():
    spec = importlib.util.spec_from_file_location(
        "figure_promote_fig4_test", _ROOT / "scripts" / "figures" / "papers" / "promote.py"
    )
    promote = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = promote
    spec.loader.exec_module(promote)
    for group in ("reward", "length"):
        tiers = promote.NGUYEN_FIG4_GATE_TIER[group]
        assert {k for k, t in tiers.items() if t == "shape"} == set(ng.FIG4_SHAPE_KEYS[group])
        assert {k for k, t in tiers.items() if t in {"shape", "full"}} == set(ng.FIG4_REQUIRED_KEYS[group])
        assert {k for k, t in tiers.items() if t == "info"} == set(ng.FIG4_TIER2_KEYS[group])
        assert [k for k, _ in promote.NGUYEN_GATE_GROUPS["4"][group]] == list(tiers)


def test_raw_outline_split_into_sorted_edges():
    for stem in ("fig4_reward", "fig4_length"):
        edges = ng.raw_outline_edges(stem)
        for x, _ in edges.values():
            assert np.all(np.diff(x) >= 0)
        lo_x, lo_y = edges["lower"]
        hi_x, hi_y = edges["upper"]
        late = 300.0
        assert np.mean(lo_y[lo_x >= late]) < np.mean(hi_y[hi_x >= late])


def test_centered_smooth_is_centred():
    y = np.zeros(100)
    y[50] = 20.0
    s = ng.centered_smooth(y, 20)
    peak = np.flatnonzero(s > 0)
    assert peak[0] == 41 and peak[-1] == 60
    assert np.allclose(ng.centered_smooth(np.full(30, 3.0), 20), 3.0)
