"""Mock plant for fast Mehregan env tests."""

from __future__ import annotations

from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult


class MockPlant:
    """Deterministic $P_\\beta$ stub — monotonic in stimulation frequency."""

    def __init__(self, *, p_beta_unstimulated: float = 450.0) -> None:
        self.p_beta_unstimulated = p_beta_unstimulated
        self._seed: int | None = None
        self._calls = 0

    def reset(self, seed: int | None = None) -> MockPlant:
        self._seed = seed
        self._calls = 0
        return self

    def close(self) -> None:
        return None

    def integrate(
        self,
        duration_s: float,
        dbs_spec: DbsSpec | None = None,
        *,
        record_spikes: bool = True,
    ) -> IntegrateResult:
        self._calls += 1
        spec = dbs_spec if dbs_spec is not None else DbsSpec.none()
        effect = 0.0 if spec.pick_dbs_freq <= 1 else min(spec.frequency_hz, 200.0) * 0.6
        raw = max(50.0, self.p_beta_unstimulated - effect)
        return IntegrateResult(
            gpi_spikes=[],
            duration_s=duration_s,
            dt_ms=0.01,
            pd=1,
            dbs_spec=spec,
            seed=self._seed,
            p_beta=raw,
            info={"mock": True, "call": self._calls},
        )
