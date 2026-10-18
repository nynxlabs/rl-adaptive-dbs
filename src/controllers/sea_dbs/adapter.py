"""Ravivarapu environment adapter (2 ms steps, binary pulse, Eq. (7) reward)."""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.reward import sea_dbs_reward
from envs.plant.biomarkers import p_beta as _p_beta
from envs.plant.dbs import DBS_AMPLITUDE_NA_PER_CM2, DBS_PULSE_WIDTH_MS, DbsSpec, create_dbs_current
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
            from envs.plant import PythonPlant

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
        if (
            self.config.plant_integration_mode == "continuous" or self.config.step_mode == "trailing"
        ) and isinstance(self._plant, MatlabPlant):
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
        # ``trailing`` step mode: GPi spike times (s, episode clock incl. warm-up),
        # the clock, the emitted STN drive on the global sample grid, and the
        # sample index where stimulation can start (end of the warm-up).
        self._spikes: list[list[np.ndarray]] = []
        self._t_s = 0.0
        self._sample = 0
        self._drive: np.ndarray = np.zeros(0)
        self._stim_origin = 0

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

    # ------------------------------------------------------------------
    # ``trailing`` step mode (paper §V.A timing; config.step_mode)
    # ------------------------------------------------------------------
    def _is_trailing(self) -> bool:
        return self.config.step_mode == "trailing"

    def _trailing_integrate(self, duration_s: float, idbs: np.ndarray | None) -> None:
        spec = DbsSpec.none() if idbs is None else DbsSpec(pick_dbs_freq=2, idbs=idbs, mean_hz=self._carrier_hz)
        result = self._plant.integrate(  # type: ignore[call-arg]
            duration_s, spec, record_spikes=True, carry=True, fixed_network=True
        )
        n = len(result.gpi_spikes)
        if not self._spikes:
            self._spikes = [[] for _ in range(n)]
        for j, times in enumerate(result.gpi_spikes):
            if np.asarray(times).size:
                self._spikes[j].append(np.asarray(times, dtype=np.float64) + self._t_s)
        self._t_s += duration_s
        # Keep only what a trailing window (plus the Eq. (4) history) can still reach.
        horizon = self._t_s - self.config.trailing_window_s - self.config.n_obs * self.config.step_duration_s
        for j, chunks in enumerate(self._spikes):
            self._spikes[j] = [c for c in chunks if c.size and c[-1] >= horizon]
        self._sample += int(round(duration_s * 1000.0 / self.config.plant_dt_ms))

    def _trailing_p_beta(self, t_end_s: float | None = None) -> float:
        """Eq. (1) P_beta of the GPi spikes in the trailing window ending at ``t_end_s``."""
        t1 = self._t_s if t_end_s is None else float(t_end_s)
        w = float(self.config.trailing_window_s)
        t0 = t1 - w
        trains = []
        for chunks in self._spikes:
            allt = np.concatenate(chunks) if chunks else np.zeros(0)
            trains.append(allt[(allt >= t0) & (allt < t1)] - t0)
        return float(_p_beta(trains, dt_ms=self.config.plant_dt_ms, segment_duration_s=w))

    def _trailing_step_drive(self, action: int) -> np.ndarray:
        """STN drive for one step on the segment grid (sample 0 = previous segment's last)."""
        cfg = self.config
        n = int(round(cfg.step_duration_ms / cfg.plant_dt_ms))
        g0 = self._sample
        need = g0 + 2 * n + int(round(DBS_PULSE_WIDTH_MS / cfg.plant_dt_ms)) + 2
        if self._drive.size < need:
            self._drive = np.concatenate([self._drive, np.zeros(need - self._drive.size)])
        if int(action) == 1:
            pulse_len = int(round(DBS_PULSE_WIDTH_MS / cfg.plant_dt_ms))
            if cfg.step_pulse_mode == "per_step":
                onsets = [g0 + 1]
            else:
                # Carrier clock starts with the episode: onsets at origin + 1 + k * ISI.
                isi = int(round((1000.0 / self._carrier_hz) / cfg.plant_dt_ms))
                first = self._stim_origin + 1
                k0 = max(0, -(-(g0 + 1 - first) // isi))
                onsets = list(range(first + k0 * isi, g0 + n + 1, isi))
            for onset in onsets:
                self._drive[onset : onset + pulse_len] = DBS_AMPLITUDE_NA_PER_CM2
        return self._drive[g0 : g0 + n + 1].copy()

    def _trailing_reset(self) -> list[float]:
        """Untreated warm-up; returns the raw P_beta estimates that fill the Eq. (4) window."""
        cfg = self.config
        self._spikes = []
        self._t_s = 0.0
        self._sample = 0
        self._drive = np.zeros(0)
        step_s = cfg.step_duration_s
        n_hist = cfg.n_obs if cfg.prefill_obs_window else 1
        self._trailing_integrate(cfg.trailing_window_s + (n_hist - 1) * step_s, None)
        self._stim_origin = self._sample
        return [self._trailing_p_beta(self._t_s - k * step_s) for k in range(n_hist - 1, -1, -1)]

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
        carry = self._is_trailing() and self.config.carry_plant_across_episodes and bool(self._spikes)
        self._step_count = 0
        if carry:
            # Same simulation, new episode: the state is the current Eq. (4) window.
            mean_norm = self._mean_p_beta()
            return self._observation_from_window(), {
                "p_beta_raw": float(self._obs_window[-1]) * self.config.observation_scale,
                "p_beta_norm": mean_norm,
                "mean_p_beta": mean_norm,
                "adapter": True,
                "step_duration_ms": self.config.step_duration_ms,
                "integration_duration_ms": self._t_s * 1000.0,
                "carrier_hz": self._carrier_hz,
                "plant_integration_mode": "trailing-carried",
            }
        self._plant.reset(seed=seed)
        self._obs_window.clear()

        if self._is_trailing():
            raws = self._trailing_reset()
            self._obs_window.extend(self._normalize_p_beta(r) for r in raws)
            mean_norm = self._mean_p_beta()
            info = {
                "p_beta_raw": raws[-1],
                "p_beta_norm": mean_norm,
                "mean_p_beta": mean_norm,
                "adapter": True,
                "step_duration_ms": self.config.step_duration_ms,
                "integration_duration_ms": self._t_s * 1000.0,
                "carrier_hz": self._carrier_hz,
                "plant_integration_mode": "trailing",
            }
            return self._observation_from_window(), info

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
        if self._is_trailing():
            duration_s = self.config.step_duration_s
            self._trailing_integrate(duration_s, self._trailing_step_drive(int(action)))
            raw = self._trailing_p_beta()
        else:
            duration_s = self._window_s_for_action(int(action))
            result = self._integrate(
                duration_s, self._dbs_spec_for_action(int(action), duration_s=duration_s)
            )
            raw = self._raw_p_beta(result)
        self._obs_window.append(self._normalize_p_beta(raw))
        obs = self._observation_from_window()
        mean_norm = self._mean_p_beta()
        reward_eq7 = sea_dbs_reward(
            mean_norm,
            beta_threshold=self.config.beta_threshold,
            reward_scale=self.config.reward_scale,
        )
        reward = reward_eq7 - float(self.config.stim_cost) * float(int(action) == 1)

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = False
        # Algorithm 1 stores no done flag; ``mask_episode_end`` (default) stops the
        # Q-target from bootstrapping past the 30-step time limit.
        dw = 1.0 if (truncated and self.config.mask_episode_end) else 0.0
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
            "reward_eq7": reward_eq7,
            "dw": dw,
        }
        return obs, reward, terminated, truncated, info
