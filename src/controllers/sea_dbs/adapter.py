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
            from envs.plant.config import PlantConfig
            from envs.plant.python_backend import PythonPlant

            self._plant: PlantBackend = PythonPlant(PlantConfig(dt_ms=self.config.plant_dt_ms))
        else:
            self._plant = plant
            plant_dt = getattr(getattr(plant, "config", None), "dt_ms", None)
            if plant_dt is not None and abs(float(plant_dt) - self.config.plant_dt_ms) > 1e-12:
                msg = (
                    f"plant dt_ms {plant_dt} != SEADBSConfig.plant_dt_ms "
                    f"{self.config.plant_dt_ms} (the stim trace is built on the config grid)"
                )
                raise ValueError(msg)
        if self.config.plant_integration_mode == "continuous" and isinstance(
            self._plant, MatlabPlant
        ):
            msg = (
                "plant_integration_mode='continuous' requires PythonPlant "
                "(carries plant state across segments)"
            )
            raise ValueError(msg)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(self.config.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.config.n_actions)

        self._rng: np.random.Generator | None = None
        self._episode_seed: int | None = None
        self._step_count = 0
        self._obs_window: deque[float] = deque(maxlen=self.config.n_obs)
        # Eval may override via set_carrier_hz; reset must not clobber that.
        self._carrier_hz = float(self.config.carrier_hz)

    def _window_s_for_action(self, action: int) -> float:  # noqa: ARG002 - same window for both actions
        """Plant shot length; identical for stim and no-stim so P_beta is one estimator."""
        return float(self.config.integration_duration_s)

    def close(self) -> None:
        if self._owns_plant:
            self._plant.close()

    def set_carrier_hz(self, hz: float) -> None:
        """Fixed eval knob for inference (Fig 5a/5b); not an RL action.

        Survives ``reset()``. Training and Figs 5a/6/7 use 50 Hz; Fig 5b eval
        switches the same adapter to 30 Hz without rebuilding the env.
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
        burst_ms = integration_ms if cfg.dbs_burst_ms is None else float(cfg.dbs_burst_ms)
        if burst_ms < integration_ms:
            # Short-burst convention (paper Eq. (6) "short bursts"): carrier train
            # for ``burst_ms`` of the window, then no stimulation.
            burst = create_dbs_current(
                self._carrier_hz,
                tmax_ms=integration_ms,
                dt_ms=cfg.plant_dt_ms,
            )
            burst[int(round(burst_ms / cfg.plant_dt_ms)) :] = 0.0
            return DbsSpec(pick_dbs_freq=2, idbs=burst)
        return DbsSpec.from_frequency_hz(self._carrier_hz)

    def _is_continuous(self) -> bool:
        return self.config.plant_integration_mode == "continuous"

    def _integrate(self, duration_s: float, spec: DbsSpec) -> IntegrateResult:
        """One plant segment. ``continuous`` carries the network state (and wiring)
        from the previous segment of the episode; ``disconnected`` restarts from
        the episode IC every step."""
        if self._is_continuous():
            return self._plant.integrate(  # type: ignore[call-arg]
                duration_s,
                spec,
                record_spikes=True,
                carry=True,
                fixed_network=True,
            )
        return self._plant.integrate(duration_s, spec, record_spikes=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._episode_seed = seed
        elif self._rng is not None:
            self._episode_seed = int(self._rng.integers(0, 2**31 - 1))
        else:
            self._episode_seed = None
        self._plant.reset(seed=seed)
        self._step_count = 0
        self._obs_window.clear()

        duration_s = self._window_s_for_action(0)
        result = self._integrate(duration_s, self._dbs_spec_for_action(0, duration_s=duration_s))
        raw = self._raw_p_beta(result)
        norm_val = self._normalize_p_beta(raw)
        if self.config.prefill_obs_window:
            self._obs_window.extend([norm_val] * self.config.n_obs)
        else:
            self._obs_window.append(norm_val)
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
            "plant_integration_mode": self.config.plant_integration_mode,
        }
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        duration_s = self._window_s_for_action(int(action))
        result = self._integrate(
            duration_s, self._dbs_spec_for_action(int(action), duration_s=duration_s)
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
            "plant_integration_mode": self.config.plant_integration_mode,
            "dw": dw,
        }
        return obs, reward, terminated, truncated, info
