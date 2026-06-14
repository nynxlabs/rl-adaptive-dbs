"""STN DBS drive specification for the Kumaravelu reference script."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DbsSpec:
    """Index into freqs = 0:5:200 Hz in simulate_network_model.m.

    pick_dbs_freq == 1 forces zero DBS current (reference convention).
    """

    pick_dbs_freq: int = 1

    @classmethod
    def none(cls) -> DbsSpec:
        return cls(pick_dbs_freq=1)

    @classmethod
    def from_frequency_hz(cls, hz: float) -> DbsSpec:
        """Map carrier frequency (Hz) to reference script index (1-based)."""
        if hz <= 0:
            return cls.none()
        index = int(round(hz / 5.0)) + 1
        return cls(pick_dbs_freq=index)

    @property
    def frequency_hz(self) -> float:
        if self.pick_dbs_freq <= 1:
            return 0.0
        return float((self.pick_dbs_freq - 1) * 5)
