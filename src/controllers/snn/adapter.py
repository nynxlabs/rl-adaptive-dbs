"""Nguyen et al. environment adapter (100 ms steps, spike obs, α–β feedback).

Ternary DBS sensitivities (``SNNConfig`` defaults, replication.md §4.2):
amplitude **10** nA/cm², frequency **5** Hz, pulse width **0.05** ms per ``+1`` action.
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controllers.snn.config import SNNConfig
from controllers.snn.dbs_params import DBSParameterState
from controllers.snn.encoder import SpikeObservationEncoder
from controllers.snn.energy import dbs_energy_index
from controllers.snn.reward import alpha_beta_power, nguyen_reward
from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult


class PlantBackend(Protocol):
    def reset(self, seed: int | None = None) -> Any: ...

    def integrate(
        self,
        duration_s: float,
        dbs_spec: DbsSpec | None = None,
        *,
        record_spikes: bool = True,
    ) -> IntegrateResult: ...

    def close(self) -> None: ...


class NguyenEnvAdapter(gym.Env):
    """Wraps the shared Kumaravelu plant with Nguyen I/O (replication.md §3, §10)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        plant: PlantBackend | None = None,
        config: SNNConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = (config or SNNConfig()).with_variant_defaults()
        self.encoder = SpikeObservationEncoder(self.config)
        self._owns_plant = plant is None
        if plant is None:
            from envs.plant.python_backend import PythonPlant

            self._plant: PlantBackend = PythonPlant()
        else:
            self._plant = plant
        self.render_mode = render_mode

        obs_shape = self.config.observation_shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32,
        )
        if self.config.action_scheme == "joint":
            self.action_space = spaces.Discrete(self.config.n_action_outputs)
        else:
            self.action_space = spaces.MultiDiscrete([3, 3, 3])

        self._rng: np.random.Generator | None = None
        self._dbs = DBSParameterState.from_config(self.config)
        self._step_count = 0
        self._subthreshold_streak = 0

    def close(self) -> None:
        if self._owns_plant:
            self._plant.close()

    def _gpi_spike_trains(self, result: IntegrateResult) -> list[np.ndarray]:
        spikes = result.gpi_spikes
        n = self.config.neurons_per_region
        if len(spikes) >= n:
            return spikes[:n]
        padded = list(spikes)
        while len(padded) < n:
            padded.append(np.array([], dtype=float))
        return padded

    def _encode_observation(self, result: IntegrateResult) -> np.ndarray:
        return self.encoder.encode(
            self._gpi_spike_trains(result),
            duration_s=self.config.step_duration_s,
        )

    def _step_energy(self) -> float:
        return dbs_energy_index(
            frequency_hz=self._dbs.frequency_hz,
            amplitude=self._dbs.amplitude,
            pulse_width_ms=self._dbs.pulse_width_ms,
            step_duration_s=self.config.step_duration_s,
            stimulated_neurons=self.config.stimulated_neurons,
        )

    def _apply_action(self, action: np.ndarray | int) -> np.ndarray:
        if self.config.action_scheme == "joint":
            from controllers.snn.actions import decode_joint_action

            ternary = decode_joint_action(int(action))
        else:
            from controllers.snn.actions import decode_factored_action

            ternary = decode_factored_action(np.asarray(action, dtype=np.int64))
        self._dbs.apply_delta(ternary, self.config)
        return ternary

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._plant.reset(seed=seed)
        self._dbs = DBSParameterState.from_config(self.config)
        self._step_count = 0
        self._subthreshold_streak = 0

        result = self._plant.integrate(
            self.config.step_duration_s,
            self._dbs.to_dbs_spec(duration_s=self.config.step_duration_s),
            record_spikes=True,
            record_th_spikes=True,
            record_cor_spikes=True,
        )
        obs = self._encode_observation(result)
        alpha_beta = alpha_beta_power(
            self._gpi_spike_trains(result),
            duration_s=self.config.step_duration_s,
            dt_ms=result.dt_ms,
        )
        spike_count, step_energy = self._step_metrics(result)
        info = {
            "alpha_beta": alpha_beta,
            "dbs": self._dbs,
            "adapter": True,
            "step_duration_ms": self.config.step_duration_ms,
            "cbgt_spike_count": spike_count,
            "step_energy": step_energy,
        }
        return obs, info

    def _integrate_current_dbs(self) -> IntegrateResult:
        return self._plant.integrate(
            self.config.step_duration_s,
            self._dbs.to_dbs_spec(duration_s=self.config.step_duration_s),
            record_spikes=True,
            record_th_spikes=True,
            record_cor_spikes=True,
        )

    @staticmethod
    def _cbgt_spike_count(result: IntegrateResult) -> int:
        """Sum GPi + TH + cortical spike events in one RL step (paper Fig. 5a)."""
        info = result.info or {}
        total = 0
        for key in ("gpi_spike_counts", "th_spike_counts", "cor_spike_counts"):
            counts = info.get(key)
            if counts:
                total += int(np.sum(counts))
        if total > 0:
            return total
        return int(sum(len(times) for times in result.gpi_spikes))

    def _step_metrics(self, result: IntegrateResult) -> tuple[int, float]:
        return self._cbgt_spike_count(result), self._step_energy()

    def step(
        self,
        action: np.ndarray | int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_dbs = self._dbs.copy()
        ternary = self._apply_action(action)
        plant_guard = False
        try:
            result = self._integrate_current_dbs()
        except (ZeroDivisionError, ValueError, FloatingPointError):
            # Some DBS triples destabilize the Kumaravelu HH integrator; keep the
            # previous parameters and re-integrate so RL rollouts can continue.
            self._dbs = prev_dbs
            plant_guard = True
            try:
                result = self._integrate_current_dbs()
            except (ZeroDivisionError, ValueError, FloatingPointError):
                # Rollback triple also failed — reset to paper init and retry once.
                self._dbs = DBSParameterState.from_config(self.config)
                plant_guard = True
                result = self._integrate_current_dbs()
        obs = self._encode_observation(result)
        alpha_beta = alpha_beta_power(
            self._gpi_spike_trains(result),
            duration_s=self.config.step_duration_s,
            dt_ms=result.dt_ms,
        )

        if alpha_beta < self.config.alpha_beta_threshold:
            self._subthreshold_streak += 1
        else:
            self._subthreshold_streak = 0

        self._step_count += 1
        remaining = max(0, self.config.max_episode_steps - self._step_count)
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = self._subthreshold_streak >= self.config.subthreshold_steps_required
        reward = nguyen_reward(
            alpha_beta=alpha_beta,
            energy=self._step_energy(),
            terminated=terminated,
            remaining_steps=remaining,
            config=self.config,
        )
        spike_count, step_energy = self._step_metrics(result)
        info = {
            "alpha_beta": alpha_beta,
            "dbs": self._dbs,
            "ternary_action": ternary,
            "adapter": True,
            "step_duration_ms": self.config.step_duration_ms,
            "subthreshold_streak": self._subthreshold_streak,
            "plant_guard": plant_guard,
            "cbgt_spike_count": spike_count,
            "step_energy": step_energy,
        }
        return obs, reward, terminated, truncated, info
