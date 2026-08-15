"""Ravivarapu environment adapter (2 ms steps, binary pulse, Eq. (7) reward)."""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.reward import sea_dbs_reward
from envs.plant.dbs import DbsSpec, create_dbs_current
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


class SEA_DBSEnvAdapter(gym.Env):
    """Wraps Kumaravelu plant with Ravivarapu I/O (replication.md §3, §5–§6)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        plant: PlantBackend | None = None,
        config: SEADBSConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = (config or SEADBSConfig()).with_variant_defaults()
        self._owns_plant = plant is None
        if plant is None:
            from envs.plant.python_backend import PythonPlant

            self._plant: PlantBackend = PythonPlant()
        else:
            self._plant = plant
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(self.config.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.config.n_actions)

        self._rng: np.random.Generator | None = None
        self._step_count = 0
        self._obs_window: deque[float] = deque(maxlen=self.config.n_obs)
        # Eval may override via set_carrier_hz; reset must not clobber that.
        self._carrier_hz = float(self.config.carrier_hz)

    def _window_s_for_action(self, action: int) -> float:
        untreated = self.config.untreated_window_s
        if untreated is not None and int(action) == 0:
            return float(untreated)
        return float(self.config.integration_duration_s)

    def close(self) -> None:
        if self._owns_plant:
            self._plant.close()

    def set_carrier_hz(self, hz: float) -> None:
        """Fixed eval knob for inference (Fig 5a/5b); not an RL action.

        Survives ``reset()``. Fig 4 training uses 130 Hz; Fig 5 eval switches
        the same adapter to 50 Hz or 30 Hz without rebuilding the env.
        """
        self._carrier_hz = float(hz)

    def _normalize_p_beta(self, raw: float) -> float:
        return raw / self.config.observation_scale

    def _mean_p_beta(self) -> float:
        if not self._obs_window:
            return 0.0
        return float(np.mean(self._obs_window))

    def _observation_from_window(self) -> np.ndarray:
        mean_norm = self._mean_p_beta()
        return np.array([mean_norm], dtype=np.float32)

    def _raw_p_beta(self, result: IntegrateResult) -> float:
        if result.p_beta is not None:
            return float(result.p_beta)
        msg = "plant integrate did not return p_beta"
        raise RuntimeError(msg)

    def _dbs_spec_for_action(self, action: int, *, duration_s: float | None = None) -> DbsSpec:
        if int(action) == 0:
            return DbsSpec.none()
        cfg = self.config
        integration_ms = (
            self._window_s_for_action(action) if duration_s is None else float(duration_s)
        ) * 1000.0
        burst_ms = float(cfg.dbs_burst_ms)
        delay_ms = float(cfg.dbs_pulse_delay_ms)
        if burst_ms < integration_ms or delay_ms > 0.0:
            # Short-burst convention (paper Eq. (6)): apply the carrier train for
            # only ``burst_ms`` of the biomarker window, then leave the rest of the
            # window unstimulated. Optional ``dbs_pulse_delay_ms`` shifts the
            # train later in the window (Fig 5b: 5 ms, one 30 Hz pulse).
            full = create_dbs_current(
                self._carrier_hz,
                tmax_ms=integration_ms,
                dt_ms=cfg.plant_dt_ms,
            )
            if delay_ms > 0.0:
                shift = int(round(delay_ms / cfg.plant_dt_ms))
                delayed = np.zeros_like(full)
                if 0 < shift < full.size:
                    delayed[shift:] = full[: full.size - shift]
                    full = delayed
            burst = full.copy()
            if burst_ms < integration_ms:
                burst[int(round(burst_ms / cfg.plant_dt_ms)) :] = 0.0
            return DbsSpec(pick_dbs_freq=2, idbs=burst)
        return DbsSpec.from_frequency_hz(self._carrier_hz)

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
        self._step_count = 0
        self._obs_window.clear()

        duration_s = self._window_s_for_action(0)
        result = self._plant.integrate(
            duration_s,
            self._dbs_spec_for_action(0, duration_s=duration_s),
            record_spikes=True,
        )
        raw = self._raw_p_beta(result)
        self._obs_window.append(self._normalize_p_beta(raw))
        obs = self._observation_from_window()
        mean_norm = self._mean_p_beta()
        info = {
            "p_beta_raw": raw,
            "p_beta_norm": mean_norm,
            "mean_p_beta": mean_norm,
            "adapter": True,
            "step_duration_ms": self.config.step_duration_ms,
            "integration_duration_ms": duration_s * 1000.0,
            "carrier_hz": self._carrier_hz,
        }
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        duration_s = self._window_s_for_action(int(action))
        result = self._plant.integrate(
            duration_s,
            self._dbs_spec_for_action(int(action), duration_s=duration_s),
            record_spikes=True,
        )
        raw = self._raw_p_beta(result)
        self._obs_window.append(self._normalize_p_beta(raw))
        obs = self._observation_from_window()
        mean_norm = self._mean_p_beta()
        reward = sea_dbs_reward(
            mean_norm,
            beta_threshold=self.config.beta_threshold,
            reward_scale=self.config.reward_scale,
        )

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = False
        dw = 1.0 if truncated else 0.0
        info = {
            "p_beta_raw": raw,
            "p_beta_norm": mean_norm,
            "mean_p_beta": mean_norm,
            "action": int(action),
            "adapter": True,
            "step_duration_ms": self.config.step_duration_ms,
            "integration_duration_ms": duration_s * 1000.0,
            "carrier_hz": self._carrier_hz,
            "dw": dw,
        }
        return obs, reward, terminated, truncated, info
