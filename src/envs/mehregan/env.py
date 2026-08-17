"""Mehregan et al. Gymnasium environment on the Kumaravelu plant."""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.config import make_alphabet
from envs.mehregan.patterns import PatternAlphabet
from envs.mehregan.reward import mehregan_reward
from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult, MatlabPlant

_PLANT_INTEGRATION_MODES = ("disconnected", "continuous")


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


class MehreganEnv(gym.Env):
    """Parkinsonian CBGT env: 2 s steps, GPi $P_\\beta$, reward Eq. (8) ([environment.md](../../docs/environment.md))."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        plant: PlantBackend | None = None,
        config: MehreganEnvConfig | None = None,
        alphabet: PatternAlphabet | None = None,  # or FixedMeanPatternAlphabet
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MehreganEnvConfig()
        mode = self.config.plant_integration_mode
        if mode not in _PLANT_INTEGRATION_MODES:
            msg = (
                f"unknown plant_integration_mode {mode!r}; "
                f"expected one of {_PLANT_INTEGRATION_MODES}"
            )
            raise ValueError(msg)
        self.alphabet = alphabet if alphabet is not None else make_alphabet(self.config)
        self._owns_plant = plant is None
        self._plant: PlantBackend = plant if plant is not None else MatlabPlant()
        if mode == "continuous" and isinstance(self._plant, MatlabPlant):
            msg = (
                "plant_integration_mode='continuous' requires PythonPlant "
                "(sequential 2 s state carry)"
            )
            raise ValueError(msg)
        self.render_mode = render_mode

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

    def _normalize_p_beta(self, raw: float) -> float:
        return raw / self.config.observation_scale

    def _push_observation(self, raw_p_beta: float) -> np.ndarray:
        self._obs_window.append(self._normalize_p_beta(raw_p_beta))
        while len(self._obs_window) < self.config.state_length:
            self._obs_window.appendleft(self._obs_window[0])
        return np.asarray(self._obs_window, dtype=np.float32)

    def _plant_dt_ms(self) -> float | None:
        cfg = getattr(self._plant, "config", None)
        if cfg is not None and hasattr(cfg, "dt_ms"):
            return float(cfg.dt_ms)
        dt = getattr(self.alphabet, "dt_ms", None)
        return float(dt) if dt is not None else None

    def _is_continuous(self) -> bool:
        return self.config.plant_integration_mode == "continuous"

    def _integrate_segment(self, dbs_spec: DbsSpec) -> IntegrateResult:
        kwargs: dict[str, Any] = {}
        if self._is_continuous():
            kwargs["carry"] = True
        result = self._plant.integrate(
            self.config.step_duration_s,
            dbs_spec,
            record_spikes=True,
            **kwargs,
        )
        if result.p_beta is None:
            msg = "plant integrate did not return p_beta"
            raise RuntimeError(msg)
        return result

    def _segment_info(self, result: IntegrateResult, dbs_spec: DbsSpec) -> dict[str, Any]:
        return {
            "p_beta_raw": result.p_beta,
            "p_beta_norm": self._normalize_p_beta(result.p_beta),
            "dbs_freq_hz": dbs_spec.frequency_hz,
            "pick_dbs_freq": dbs_spec.pick_dbs_freq,
            "episode_step": self._step_count,
            "plant_integration_mode": self.config.plant_integration_mode,
        }

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
        observation = self._push_observation(result.p_beta)
        reward = mehregan_reward(
            observation,
            beta_threshold=self.config.beta_threshold,
            reward_scale=self.config.reward_scale,
        )
        info = self._segment_info(result, reset_spec)
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
            self._episode_actions.append(int(action))
        result = self._integrate_segment(dbs_spec)
        observation = self._push_observation(result.p_beta)
        reward = mehregan_reward(
            observation,
            beta_threshold=self.config.beta_threshold,
            reward_scale=self.config.reward_scale,
        )
        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = False
        info = self._segment_info(result, dbs_spec)
        info["action"] = int(action)
        info["dw"] = 1.0 if truncated else 0.0
        return observation, reward, terminated, truncated, info
