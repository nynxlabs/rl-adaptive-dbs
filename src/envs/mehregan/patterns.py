"""Discrete STN pattern alphabet (Kumaravelu frequency grid)."""

from __future__ import annotations

from dataclasses import dataclass

from envs.plant.dbs import DbsSpec

# Kumaravelu simulate_network_model.m: freqs = 0:5:200 (41 entries); pick_dbs_freq in 1..41.
# pick_dbs_freq == 1 → no DBS; pick k>1 → freqs(k) Hz pulse train.


@dataclass(frozen=True)
class PatternAlphabet:
    """Gym action ``i`` maps to ``DbsSpec(pick_dbs_freq=i+1)`` (paper §IV.A.1 encoding TBD)."""

    n_patterns: int = 41

    @property
    def n_actions(self) -> int:
        return self.n_patterns

    def to_dbs_spec(self, action: int) -> DbsSpec:
        if action < 0 or action >= self.n_actions:
            msg = f"action {action} outside [0, {self.n_actions})"
            raise ValueError(msg)
        return DbsSpec(pick_dbs_freq=action + 1)

    def action_for_dbs_spec(self, spec: DbsSpec) -> int:
        pick = spec.pick_dbs_freq
        if pick < 1 or pick > self.n_patterns:
            msg = f"pick_dbs_freq {pick} outside [1, {self.n_patterns}]"
            raise ValueError(msg)
        return pick - 1

    def action_for_frequency_hz(self, hz: float) -> int:
        return self.action_for_dbs_spec(DbsSpec.from_frequency_hz(hz))
