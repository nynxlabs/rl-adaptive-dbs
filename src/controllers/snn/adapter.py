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
from envs.plant.dbs import ContinuousPulseTrain, DbsSpec
from envs.plant.matlab_backend import IntegrateResult


CBGT_POPULATIONS: tuple[str, ...] = (
    "cortex_exc",
    "cortex_inh",
    "str_dr",
    "str_indr",
    "stn",
    "gpe",
    "gpi",
    "th",
)
# DBS is delivered to the STN; Fig. 5a counts exclude it (docs/figures/nguyen/5.md).
STIMULATED_POPULATION = "stn"


def observed_populations(config: SNNConfig) -> tuple[str, ...]:
    """Observed populations in observation order: GPi only, or all eight CBGT populations."""
    return ("gpi",) if config.n_regions == 1 else CBGT_POPULATIONS


class PlantBackend(Protocol):
    def reset(self, seed: int | None = None) -> Any: ...

    def integrate(
        self,
        duration_s: float,
        dbs_spec: DbsSpec | None = None,
        *,
        record_spikes: bool = True,
        carry: bool = False,
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
        if self.config.n_regions not in (1, 8):
            msg = f"n_regions must be 1 (GPi) or 8 (all CBGT populations), got {self.config.n_regions}"
            raise ValueError(msg)
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
        self._prev_alpha_beta: float | None = None
        self._explore_epsilon: float | None = None
        self._episode_index: int | None = None
        self._pulses = ContinuousPulseTrain()

    def set_training_context(self, *, epsilon: float, episode: int) -> None:
        """Set ε and episode index for frequency_sensitivity curriculum."""
        self._explore_epsilon = float(epsilon)
        self._episode_index = int(episode)

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

    def _observed_spike_trains(self, result: IntegrateResult) -> list[np.ndarray]:
        """GPi only (``n_regions=1``) or all eight CBGT populations (``n_regions=8``)."""
        if self.config.n_regions == 1:
            return self._gpi_spike_trains(result)
        n = self.config.neurons_per_region
        info = result.info
        cor = list(info["cor_spikes"])
        trains = (
            cor[:n]  # cortex excitatory
            + cor[n : 2 * n]  # cortex inhibitory
            + list(info["str_dr_spikes"])
            + list(info["str_indr_spikes"])
            + list(info["stn_spikes"])
            + list(info["gpe_spikes"])
            + self._gpi_spike_trains(result)
            + list(info["th_spikes"])
        )
        return [np.asarray(t, dtype=float) for t in trains]

    def _spike_events(self, result: IntegrateResult) -> int:
        """Spike events in this 100 ms step across the observed populations (Fig. 5a)."""
        return int(sum(np.asarray(t).size for t in self._observed_spike_trains(result)))

    def _spike_events_by_population(self, result: IntegrateResult) -> list[int]:
        """Spike events in this step per observed population, in ``observed_populations`` order."""
        trains = self._observed_spike_trains(result)
        n = self.config.neurons_per_region
        return [
            int(sum(np.asarray(t).size for t in trains[i * n : (i + 1) * n]))
            for i in range(len(observed_populations(self.config)))
        ]

    def _encode_observation(self, result: IntegrateResult) -> np.ndarray:
        return self.encoder.encode(
            self._observed_spike_trains(result),
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
        self._dbs.apply_delta(
            ternary,
            self.config,
            epsilon=self._explore_epsilon,
            episode=self._episode_index,
        )
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
        self._prev_alpha_beta = None
        self._explore_epsilon = None
        self._episode_index = None

        self._pulses.reset()

        result = self._integrate_current_dbs()
        obs = self._encode_observation(result)
        alpha_beta = alpha_beta_power(
            self._gpi_spike_trains(result),
            duration_s=self.config.step_duration_s,
            dt_ms=result.dt_ms,
        )
        self._prev_alpha_beta = alpha_beta
        spike_count = int(np.sum(obs))
        step_energy = self._step_energy()
        info = {
            "alpha_beta": alpha_beta,
            "dbs": self._dbs,
            "adapter": True,
            "step_duration_ms": self.config.step_duration_ms,
            "cbgt_spike_count": spike_count,
            "cbgt_spike_events": self._spike_events(result),
            "cbgt_spike_events_by_population": self._spike_events_by_population(result),
            "step_energy": step_energy,
        }
        return obs, info

    def _bg_kwargs(self) -> dict[str, bool]:
        # Only ask the plant for STN/GPe/striatum spikes when they are observed, so
        # GPi-only runs keep calling plants that predate the flag.
        return {"record_bg_spikes": True} if self.config.n_regions == 8 else {}

    def _integrate_current_dbs(self) -> IntegrateResult:
        duration_s = self.config.step_duration_s
        if not self.config.plant_carry:
            return self._plant.integrate(
                duration_s,
                self._dbs.to_dbs_spec(duration_s=duration_s),
                record_spikes=True,
                record_th_spikes=True,
                record_cor_spikes=True,
                **self._bg_kwargs(),
            )
        idbs, pulse_state = self._pulses.segment(
            frequency_hz=self._dbs.frequency_hz,
            pulse_width_ms=self._dbs.pulse_width_ms,
            amplitude=self._dbs.amplitude,
            tmax_ms=duration_s * 1000.0,
        )
        spec = DbsSpec(pick_dbs_freq=2, idbs=idbs, mean_hz=self._dbs.frequency_hz)
        result = self._plant.integrate(
            duration_s,
            spec,
            record_spikes=True,
            record_th_spikes=True,
            record_cor_spikes=True,
            carry=True,
            **self._bg_kwargs(),
        )
        # Advance the pulse clock only after the plant accepted the segment.
        self._pulses.commit(pulse_state)
        return result

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
            prev_alpha_beta=self._prev_alpha_beta,
        )
        if truncated and not terminated and self.config.truncation_penalty > 0.0:
            reward -= self.config.truncation_penalty
        spike_count = int(np.sum(obs))
        step_energy = self._step_energy()
        self._prev_alpha_beta = alpha_beta
        info = {
            "alpha_beta": alpha_beta,
            "dbs": self._dbs,
            "ternary_action": ternary,
            "adapter": True,
            "step_duration_ms": self.config.step_duration_ms,
            "subthreshold_streak": self._subthreshold_streak,
            "plant_guard": plant_guard,
            "cbgt_spike_count": spike_count,
            "cbgt_spike_events": self._spike_events(result),
            "cbgt_spike_events_by_population": self._spike_events_by_population(result),
            "step_energy": step_energy,
        }
        return obs, reward, terminated, truncated, info
