"""Within-step Mehregan env with optional continuous plant integration (extension)."""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.mehregan.extensions.alphabet_diversity.config import (
    PLANT_INTEGRATION_MODES,
    REWARD_STATE_MODES,
    WithinStepEnvConfig,
    make_alphabet,
    resolve_state_mode,
)
from envs.mehregan.extensions.alphabet_diversity.observations import (
    within_step_p_beta_series,
)
from envs.mehregan.extensions.alphabet_diversity.plant_integration import (
    integrate_stitched_step,
    resolve_mean_hz,
)
from envs.mehregan.extensions.alphabet_diversity.reward import (
    mehregan_reward,
    mehregan_reward_from_s_sum,
)
from envs.mehregan.patterns import PatternAlphabet
from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult, MatlabPlant


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


class WithinStepMehreganEnv(gym.Env):
    """Parkinsonian CBGT env: 2 s steps, GPi $P_\\beta$, reward Eq. (8) ([environment.md](../../docs/environment.md))."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        plant: PlantBackend | None = None,
        config: WithinStepEnvConfig | None = None,
        alphabet: PatternAlphabet | None = None,  # or FixedMeanPatternAlphabet
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WithinStepEnvConfig()
        self._plant: PlantBackend = plant if plant is not None else MatlabPlant()
        self._owns_plant = plant is None
        self.render_mode = render_mode

        self._resolved_state_mode = resolve_state_mode(
            state_mode=self.config.state_mode,
            state_length=self.config.state_length,
        )
        if self.config.reward_state_mode not in REWARD_STATE_MODES:
            msg = (
                f"unknown reward_state_mode {self.config.reward_state_mode!r}; "
                f"expected one of {REWARD_STATE_MODES}"
            )
            raise ValueError(msg)
        if self._resolved_state_mode == "scalar" and self.config.state_length != 1:
            msg = (
                "state_mode='scalar' requires state_length=1; "
                "use state_mode='within_step' or 'multi_step_history' for L>1"
            )
            raise ValueError(msg)
        if self._resolved_state_mode == "within_step" and self.config.state_length < 2:
            msg = "state_mode='within_step' requires state_length >= 2"
            raise ValueError(msg)
        if self.config.plant_integration_mode not in PLANT_INTEGRATION_MODES:
            msg = (
                f"unknown plant_integration_mode {self.config.plant_integration_mode!r}; "
                f"expected one of {PLANT_INTEGRATION_MODES}"
            )
            raise ValueError(msg)
        if self.config.plant_integration_mode == "continuous" and isinstance(
            self._plant, MatlabPlant
        ):
            msg = (
                "plant_integration_mode='continuous' requires PythonPlant "
                "(stitched idbs waveforms)"
            )
            raise ValueError(msg)

        plant_dt_ms = self._plant_dt_ms()
        self.alphabet = (
            alphabet
            if alphabet is not None
            else make_alphabet(self.config, plant_dt_ms=plant_dt_ms)
        )

        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(self.config.state_length,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.alphabet.n_actions)

        self._rng: np.random.Generator | None = None
        self._episode_seed: int | None = None
        self._step_count = 0
        self._episode_actions: list[int] = []
        self._obs_window: deque[float] = deque(maxlen=self.config.state_length)

    def close(self) -> None:
        if self._owns_plant:
            self._plant.close()

    def _plant_dt_ms(self) -> float | None:
        if self.config.plant_dt_ms is not None:
            return self.config.plant_dt_ms
        cfg = getattr(self._plant, "config", None)
        if cfg is not None and hasattr(cfg, "dt_ms"):
            return float(cfg.dt_ms)
        return None

    def _normalize_p_beta(self, raw: float) -> float:
        return raw / self.config.observation_scale

    def _push_history_observation(self, raw_p_beta: float) -> np.ndarray:
        self._obs_window.append(self._normalize_p_beta(raw_p_beta))
        while len(self._obs_window) < self.config.state_length:
            self._obs_window.appendleft(self._obs_window[0])
        return np.asarray(self._obs_window, dtype=np.float32)

    def _observation_from_result(self, result: IntegrateResult) -> np.ndarray:
        if result.p_beta is None:
            msg = "plant integrate did not return p_beta"
            raise RuntimeError(msg)

        if self._resolved_state_mode == "within_step":
            if not result.gpi_spikes:
                msg = "within_step state_mode requires gpi_spikes from plant integrate"
                raise RuntimeError(msg)
            raw_series = within_step_p_beta_series(
                result.gpi_spikes,
                segment_duration_s=self.config.step_duration_s,
                state_length=self.config.state_length,
                dt_ms=result.dt_ms,
            )
            return (raw_series / self.config.observation_scale).astype(np.float32)

        if self._resolved_state_mode == "multi_step_history":
            return self._push_history_observation(result.p_beta)

        # scalar: single whole-segment Pβ per step.
        return np.array([self._normalize_p_beta(result.p_beta)], dtype=np.float32)

    def _reward_from_result(
        self, observation: np.ndarray, result: IntegrateResult
    ) -> float:
        if self.config.reward_state_mode == "full_segment":
            if result.p_beta is None:
                msg = "full_segment reward requires p_beta from plant integrate"
                raise RuntimeError(msg)
            s_sum = self._normalize_p_beta(result.p_beta)
        else:
            s_sum = float(np.mean(observation))
        return mehregan_reward_from_s_sum(
            s_sum,
            beta_threshold=self.config.beta_threshold,
            reward_scale=self.config.reward_scale,
        )

    def _pre_stim_duration_s(self) -> float:
        if self.config.pre_stim_duration_s is not None:
            return float(self.config.pre_stim_duration_s)
        return float(self.config.step_duration_s)

    def _is_continuous(self) -> bool:
        return self.config.plant_integration_mode == "continuous"

    def _integrate_segment(self, dbs_spec: DbsSpec) -> IntegrateResult:
        return self._plant.integrate(
            self.config.step_duration_s,
            dbs_spec,
            record_spikes=True,
        )

    def _integrate_continuous_step(self, action: int) -> IntegrateResult:
        self._episode_actions.append(int(action))
        dt_ms = self._plant_dt_ms()
        if dt_ms is None:
            msg = "continuous plant_integration_mode requires a known plant dt_ms"
            raise RuntimeError(msg)
        return integrate_stitched_step(
            self._plant,
            seed=self._episode_seed,
            pre_stim_s=self._pre_stim_duration_s(),
            step_duration_s=self.config.step_duration_s,
            actions=self._episode_actions,
            alphabet=self.alphabet,
            dt_ms=dt_ms,
            mean_hz=resolve_mean_hz(
                self.alphabet, fallback_hz=self.config.pattern_mean_hz
            ),
        )

    def _segment_info(
        self,
        result: IntegrateResult,
        dbs_spec: DbsSpec,
        *,
        observation: np.ndarray,
    ) -> dict[str, Any]:
        p_beta_raw = float(result.p_beta) if result.p_beta is not None else float("nan")
        info: dict[str, Any] = {
            "p_beta_raw": p_beta_raw,
            "p_beta_norm": self._normalize_p_beta(p_beta_raw),
            "dbs_freq_hz": dbs_spec.frequency_hz,
            "pick_dbs_freq": dbs_spec.pick_dbs_freq,
            "episode_step": self._step_count,
            "state_mode": self._resolved_state_mode,
            "reward_state_mode": self.config.reward_state_mode,
            "plant_integration_mode": self.config.plant_integration_mode,
            "observation": observation.tolist(),
        }
        if self._resolved_state_mode == "within_step":
            info["p_beta_subwindow_raw"] = [
                float(v) * self.config.observation_scale for v in observation
            ]
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = seed
        elif self._rng is not None:
            self._episode_seed = int(self._rng.integers(0, 2**31 - 1))
        else:
            self._episode_seed = None

        self._plant.reset(self._episode_seed)
        self._step_count = 0
        self._episode_actions.clear()
        self._obs_window.clear()

        reset_spec = DbsSpec.none()
        if options and "reset_dbs_spec" in options:
            reset_spec = options["reset_dbs_spec"]

        result = self._integrate_segment(reset_spec)
        observation = self._observation_from_result(result)
        reward = self._reward_from_result(observation, result)
        info = self._segment_info(result, reset_spec, observation=observation)
        info["reward"] = reward
        info["episode_seed"] = self._episode_seed
        info["dw"] = 0.0
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        dbs_spec = self.alphabet.to_dbs_spec(int(action))
        if self._is_continuous():
            result = self._integrate_continuous_step(int(action))
        else:
            result = self._integrate_segment(dbs_spec)
        observation = self._observation_from_result(result)
        reward = self._reward_from_result(observation, result)
        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = False
        info = self._segment_info(result, dbs_spec, observation=observation)
        info["action"] = int(action)
        info["dw"] = 1.0 if truncated else 0.0
        return observation, reward, terminated, truncated, info
