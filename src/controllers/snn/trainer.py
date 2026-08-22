"""DSQN trainer — DQN updates on output membrane potentials (Nguyen §III.B)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import json

import numpy as np
import torch
import torch.nn.functional as F

from controllers.common.resume import (
    SNN_MATERIAL_FIELDS,
    config_to_dict,
    infer_completed_episodes,
    validate_resume_config_fields,
)

from controllers.snn.actions import select_action
from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer, Transition
from controllers.snn.config import (
    INIT_AMPLITUDE_NA_PER_CM2,
    INIT_FREQUENCY_HZ,
    INIT_PULSE_WIDTH_MS,
    SNNConfig,
)
from controllers.snn.networks import DSQN


@dataclass(frozen=True)
class TrainMetrics:
    episode: int
    episode_reward: float
    episode_length: int
    epsilon: float
    buffer_size: int
    loss: float | None = None


@dataclass
class TrainResult:
    """Outcome of a DSQN training run."""

    config: SNNConfig
    dsqn: DSQN
    metrics: list[TrainMetrics] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_spike_totals: list[int] = field(default_factory=list)
    episode_energies: list[float] = field(default_factory=list)
    episode_alpha_beta_means: list[float] = field(default_factory=list)
    episode_early_stops: list[bool] = field(default_factory=list)
    episode_amplitudes: list[float] = field(default_factory=list)
    episode_frequencies: list[float] = field(default_factory=list)
    episode_pulse_widths: list[float] = field(default_factory=list)
    update_count: int = 0


def _decode_factored_indices(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode base-3 factored action indices (0–26) into three group indices."""
    i0 = actions // 9
    rem = actions % 9
    i1 = rem // 3
    i2 = rem % 3
    return i0, i1, i2


def _q_from_membrane(
    membrane: torch.Tensor,
    actions: torch.Tensor,
    *,
    action_scheme: str,
) -> torch.Tensor:
    """Gather Q(s,a) from output membrane potentials."""
    if action_scheme == "joint":
        return membrane.gather(1, actions.view(-1, 1)).squeeze(1)

    if action_scheme == "factored":
        grouped = membrane.view(-1, 3, 3)
        i0, i1, i2 = _decode_factored_indices(actions)
        q0 = grouped[:, 0, :].gather(1, i0.view(-1, 1)).squeeze(1)
        q1 = grouped[:, 1, :].gather(1, i1.view(-1, 1)).squeeze(1)
        q2 = grouped[:, 2, :].gather(1, i2.view(-1, 1)).squeeze(1)
        return q0 + q1 + q2

    msg = f"unknown action_scheme {action_scheme!r}"
    raise ValueError(msg)


def _argmax_q_from_membrane(membrane: torch.Tensor, *, action_scheme: str) -> torch.Tensor:
    """argmax_a Q(s', a) from membrane potentials (batch of action indices)."""
    if action_scheme == "joint":
        return membrane.argmax(dim=1)

    if action_scheme == "factored":
        grouped = membrane.view(-1, 3, 3)
        parts = grouped.argmax(dim=-1)
        return parts[:, 0] * 9 + parts[:, 1] * 3 + parts[:, 2]

    msg = f"unknown action_scheme {action_scheme!r}"
    raise ValueError(msg)


def _max_q_from_membrane(membrane: torch.Tensor, *, action_scheme: str) -> torch.Tensor:
    """max_a Q(s', a) from membrane potentials."""
    if action_scheme == "joint":
        return membrane.max(dim=1).values

    if action_scheme == "factored":
        grouped = membrane.view(-1, 3, 3)
        return grouped.max(dim=-1).values.sum(dim=-1)

    msg = f"unknown action_scheme {action_scheme!r}"
    raise ValueError(msg)


class DSQNTrainer:
    """DQN-style trainer: control from spike counts, Q from membrane potentials."""

    def __init__(
        self,
        dsqn: DSQN,
        buffer: ReplayBuffer,
        config: SNNConfig | None = None,
        *,
        target_dsqn: DSQN | None = None,
    ) -> None:
        self.config = (config or SNNConfig()).with_variant_defaults()
        self.dsqn = dsqn
        self.buffer = buffer
        self.target_dsqn = target_dsqn if target_dsqn is not None else deepcopy(dsqn)
        self.target_dsqn.eval()
        for param in self.target_dsqn.parameters():
            param.requires_grad_(False)

        self.device = torch.device(self.config.device)
        self.dsqn.to(self.device)
        self.target_dsqn.to(self.device)
        self.optimizer = torch.optim.Adam(self.dsqn.parameters(), lr=self.config.learning_rate)
        self._rng = np.random.default_rng(self.config.seed)
        self._total_steps = 0
        self._update_count = 0
        self._last_loss: float | None = None

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def last_loss(self) -> float | None:
        return self._last_loss

    def current_epsilon(self) -> float:
        cfg = self.config
        delay = max(0, int(getattr(cfg, "epsilon_decay_delay_steps", 0) or 0))
        accel_after = max(0, int(getattr(cfg, "epsilon_accelerate_after_steps", 0) or 0))
        accel_decay = max(0, int(getattr(cfg, "epsilon_accelerate_decay_steps", 0) or 0))
        if self._total_steps < delay:
            return cfg.epsilon_start
        if cfg.epsilon_decay_steps <= 0:
            return cfg.epsilon_end
        start, end = cfg.epsilon_start, cfg.epsilon_end
        if accel_after > 0 and self._total_steps >= accel_after:
            slow_elapsed = max(0, accel_after - delay)
            slow_p = min(1.0, slow_elapsed / cfg.epsilon_decay_steps)
            eps_switch = start + slow_p * (end - start)
            dump = accel_decay if accel_decay > 0 else cfg.epsilon_decay_steps
            fast_p = min(1.0, (self._total_steps - accel_after) / dump)
            return eps_switch + fast_p * (end - eps_switch)
        elapsed = self._total_steps - delay
        progress = min(1.0, elapsed / cfg.epsilon_decay_steps)
        return start + progress * (end - start)

    def note_step(self) -> None:
        self._total_steps += 1

    def hard_update_target(self) -> None:
        self.target_dsqn.load_state_dict(self.dsqn.state_dict())

    def maybe_update(self) -> bool:
        """Run a gradient step when replay cadence is met."""
        if not self.buffer.ready_for_update():
            return False
        if len(self.buffer) < self.config.batch_size:
            return False
        self.train_step()
        self.buffer.mark_updated()
        if (
            self.config.target_update_period > 0
            and self._update_count % self.config.target_update_period == 0
        ):
            self.hard_update_target()
        return True

    def train_step(self) -> float:
        """One minibatch Bellman update on output membrane Q-values."""
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size)
        states = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_state, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)
        learn_scale = float(cfg.reward_learning_scale)
        if learn_scale != 1.0:
            rewards = rewards * learn_scale

        self.dsqn.train()
        q_out = self.dsqn(states)
        q_sa = _q_from_membrane(q_out.membrane, actions, action_scheme=cfg.action_scheme)

        with torch.no_grad():
            if cfg.double_dqn:
                online_next = self.dsqn(next_states)
                next_actions = _argmax_q_from_membrane(
                    online_next.membrane,
                    action_scheme=cfg.action_scheme,
                )
                target_next = self.target_dsqn(next_states)
                next_q = _q_from_membrane(
                    target_next.membrane,
                    next_actions,
                    action_scheme=cfg.action_scheme,
                )
            else:
                next_out = self.target_dsqn(next_states)
                next_q = _max_q_from_membrane(
                    next_out.membrane,
                    action_scheme=cfg.action_scheme,
                )
            target = rewards + cfg.gamma * (1.0 - done) * next_q

        if cfg.q_loss_fn == "huber":
            per_elem = F.smooth_l1_loss(q_sa, target, reduction="none")
        else:
            per_elem = F.mse_loss(q_sa, target, reduction="none")

        timeout_w = float(cfg.replay_timeout_weight)
        short_w = float(cfg.replay_short_stop_weight)
        if timeout_w != 1.0 or short_w != 1.0:
            sample_w = np.ones(batch.state.shape[0], dtype=np.float32)
            if timeout_w != 1.0:
                sample_w = np.where(batch.timeout_episode, sample_w * timeout_w, sample_w)
            if short_w != 1.0:
                sample_w = np.where(batch.short_stop_episode, sample_w * short_w, sample_w)
            weights = torch.as_tensor(sample_w, dtype=torch.float32, device=self.device)
            loss = (per_elem * weights).mean()
        else:
            loss = per_elem.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dsqn.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1
        self._last_loss = float(loss.detach().cpu())
        return self._last_loss

    def act(self, observation: np.ndarray, *, explore: bool = True) -> tuple[int, np.ndarray]:
        """ε-greedy action from spike counts; returns (replay_index, MultiDiscrete indices)."""
        flat = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        tensor = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
        self.dsqn.eval()
        with torch.no_grad():
            out = self.dsqn(tensor)
        counts = out.spike_counts.detach().cpu().numpy().reshape(-1)
        epsilon = self.current_epsilon() if explore else 0.0
        action_index, ternary = select_action(
            counts,
            config=self.config,
            epsilon=epsilon,
            rng=self._rng,
        )
        # Adapter MultiDiscrete expects {0,1,2}; ternary is {-1,0,1}.
        indices = (ternary + 1).astype(np.int64)
        return action_index, indices

    def train_episodes(
        self,
        env: NguyenEnvAdapter,
        *,
        start_episode: int = 0,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 50,
        on_checkpoint: Callable[[int, TrainResult], None] | None = None,
        initial_result: TrainResult | None = None,
        learn: bool = True,
    ) -> TrainResult:
        """Run ``num_episodes`` of interaction + DQN updates.

        When ``start_episode`` > 0, episodes ``start_episode .. num_episodes-1`` are
        trained and series from ``initial_result`` are appended (resume path).

        ``learn=False`` rolls out episodes only (no replay / optimizer updates) for
        metric relogging from a fixed checkpoint.
        """
        cfg = self.config
        if start_episode >= cfg.num_episodes:
            if initial_result is not None:
                return initial_result
            return TrainResult(config=cfg, dsqn=self.dsqn)

        if initial_result is not None:
            result = initial_result
            result.config = cfg
            result.dsqn = self.dsqn
        else:
            result = TrainResult(config=cfg, dsqn=self.dsqn)

        for episode in range(start_episode, cfg.num_episodes):
            obs, reset_info = env.reset(seed=cfg.seed + episode)
            episode_reward = 0.0
            # Fig. 5/6 per-episode spikes and energy sum env.step() transitions only
            # (same steps as episode_lengths); reset() initial integrate is not counted.
            episode_spikes = 0
            episode_energy = 0.0
            alpha_betas: list[float] = [float(reset_info.get("alpha_beta", float("nan")))]
            end_dbs = reset_info.get("dbs")
            step_info: dict[str, Any] | None = reset_info
            steps = 0
            terminated_early = False
            episode_slots: list[int] = []
            for _ in range(cfg.max_episode_steps):
                flat = np.asarray(obs, dtype=np.float32).reshape(-1)
                explore_eps = self.current_epsilon()
                env.set_training_context(epsilon=explore_eps, episode=episode)
                action_index, indices = self.act(obs, explore=True)
                next_obs, reward, terminated, truncated, step_info = env.step(indices)
                done = bool(terminated or truncated)
                episode_spikes += int(step_info.get("cbgt_spike_count", 0))
                episode_energy += float(step_info.get("step_energy", 0.0))
                alpha_betas.append(float(step_info.get("alpha_beta", float("nan"))))
                if learn:
                    slot = self.buffer.add(
                        Transition(
                            state=flat,
                            action=int(action_index),
                            reward=float(reward),
                            next_state=np.asarray(next_obs, dtype=np.float32).reshape(-1),
                            done=done,
                        )
                    )
                    episode_slots.append(slot)
                    self.note_step()
                    self.maybe_update()
                episode_reward += float(reward)
                steps += 1
                obs = next_obs
                if terminated:
                    terminated_early = True
                if done:
                    break

            if learn:
                if not terminated_early and episode_slots:
                    self.buffer.mark_timeout_episode(episode_slots)
                elif (
                    terminated_early
                    and episode_slots
                    and cfg.replay_short_stop_max_steps > 0
                    and steps <= cfg.replay_short_stop_max_steps
                ):
                    self.buffer.mark_short_stop_episode(episode_slots)

            if step_info is not None and step_info.get("dbs") is not None:
                end_dbs = step_info["dbs"]
            result.episode_amplitudes.append(float(getattr(end_dbs, "amplitude", INIT_AMPLITUDE_NA_PER_CM2)))
            result.episode_frequencies.append(float(getattr(end_dbs, "frequency_hz", INIT_FREQUENCY_HZ)))
            result.episode_pulse_widths.append(float(getattr(end_dbs, "pulse_width_ms", INIT_PULSE_WIDTH_MS)))

            metrics = TrainMetrics(
                episode=episode,
                episode_reward=episode_reward,
                episode_length=steps,
                epsilon=self.current_epsilon(),
                buffer_size=len(self.buffer),
                loss=self._last_loss,
            )
            result.metrics.append(metrics)
            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(steps)
            result.episode_spike_totals.append(episode_spikes)
            result.episode_energies.append(episode_energy)
            result.episode_alpha_beta_means.append(float(np.nanmean(alpha_betas)))
            result.episode_early_stops.append(terminated_early)
            if not learn:
                print(
                    f"rollout episode {episode + 1}/{cfg.num_episodes} "
                    f"spikes={episode_spikes} energy={episode_energy:.1f} len={steps}",
                    flush=True,
                )
            completed = episode + 1
            if checkpoint_path is not None and checkpoint_interval > 0:
                if completed % checkpoint_interval == 0 or completed == cfg.num_episodes:
                    save_checkpoint(
                        checkpoint_path,
                        dsqn=self.dsqn,
                        config=cfg,
                        optimizer=self.optimizer,
                        trainer=self,
                        extra=_episode_extra(result, completed_episodes=completed),
                    )
                    if on_checkpoint is not None:
                        on_checkpoint(completed, result)
            if cfg.log_episodes:
                print(
                    f"episode {episode + 1}/{cfg.num_episodes} "
                    f"reward={episode_reward:.3f} steps={steps} "
                    f"alpha_beta={result.episode_alpha_beta_means[-1]:.1f} "
                    f"energy={episode_energy:.1f} early_stop={terminated_early} "
                    f"eps={metrics.epsilon:.3f} loss={metrics.loss}",
                    flush=True,
                )

        result.update_count = self._update_count
        return result


def _episode_extra(result: TrainResult, *, completed_episodes: int) -> dict[str, Any]:
    return {
        "completed_episodes": int(completed_episodes),
        "episode_rewards": list(result.episode_rewards),
        "episode_lengths": list(result.episode_lengths),
        "episode_spike_totals": list(result.episode_spike_totals),
        "episode_energies": list(result.episode_energies),
        "episode_alpha_beta_means": list(result.episode_alpha_beta_means),
        "episode_early_stops": list(result.episode_early_stops),
        "update_count": result.update_count,
    }


def _series_from_payload(payload: dict[str, Any]) -> dict[str, list[Any]]:
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        extra = payload
    return {
        "episode_rewards": list(extra.get("episode_rewards", [])),
        "episode_lengths": list(extra.get("episode_lengths", [])),
        "episode_spike_totals": list(extra.get("episode_spike_totals", [])),
        "episode_energies": list(extra.get("episode_energies", [])),
        "episode_alpha_beta_means": list(extra.get("episode_alpha_beta_means", [])),
        "episode_early_stops": list(extra.get("episode_early_stops", [])),
    }


def train_result_from_payload(
    payload: dict[str, Any],
    *,
    dsqn: DSQN,
    config: SNNConfig,
) -> TrainResult:
    series = _series_from_payload(payload)
    result = TrainResult(config=config, dsqn=dsqn)
    result.episode_rewards = series["episode_rewards"]
    result.episode_lengths = series["episode_lengths"]
    result.episode_spike_totals = series["episode_spike_totals"]
    result.episode_energies = series["episode_energies"]
    result.episode_alpha_beta_means = series["episode_alpha_beta_means"]
    result.episode_early_stops = series["episode_early_stops"]
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        extra = payload
    result.update_count = int(extra.get("update_count", payload.get("update_count", 0)))
    for idx, reward in enumerate(result.episode_rewards):
        length = result.episode_lengths[idx] if idx < len(result.episode_lengths) else 0
        result.metrics.append(
            TrainMetrics(
                episode=idx,
                episode_reward=float(reward),
                episode_length=int(length),
                epsilon=0.0,
                buffer_size=0,
            )
        )
    return result


def train_metrics_to_dict(metrics: TrainMetrics) -> dict[str, Any]:
    return asdict(metrics)


def write_train_metrics(result: TrainResult, path: str | Path) -> Path:
    """Write per-episode training metrics as JSON beside the checkpoint."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "controller": "snn",
        "variant": result.config.variant,
        "seed": result.config.seed,
        "num_episodes": result.config.num_episodes,
        "max_episode_steps": result.config.max_episode_steps,
        "update_count": result.update_count,
        "episode_rewards": result.episode_rewards,
        "episode_lengths": result.episode_lengths,
        "episode_spike_totals": result.episode_spike_totals,
        "episode_energies": result.episode_energies,
        "episode_alpha_beta_means": result.episode_alpha_beta_means,
        "episode_early_stops": result.episode_early_stops,
        "episode_amplitudes": result.episode_amplitudes,
        "episode_frequencies": result.episode_frequencies,
        "episode_pulse_widths": result.episode_pulse_widths,
        "episodes": [train_metrics_to_dict(m) for m in result.metrics],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def save_checkpoint(
    path: str | Path,
    *,
    dsqn: DSQN,
    config: SNNConfig,
    optimizer: torch.optim.Optimizer | None = None,
    trainer: DSQNTrainer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dsqn_state_dict": dsqn.state_dict(),
        "config": config,
        "controller": "snn",
        "variant": config.variant,
    }
    if trainer is not None:
        payload["target_dsqn_state_dict"] = trainer.target_dsqn.state_dict()
        payload["buffer_state_dict"] = trainer.buffer.state_dict()
        payload["trainer_state"] = {
            "total_steps": trainer._total_steps,
            "update_count": trainer._update_count,
            "rng_state": trainer._rng.bit_generator.state,
        }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def validate_resume_config(
    saved_config: SNNConfig,
    active_config: SNNConfig,
    *,
    resume_start: int = 0,
) -> None:
    validate_resume_config_fields(
        config_to_dict(saved_config),
        config_to_dict(active_config),
        SNN_MATERIAL_FIELDS,
        label="SNNConfig",
        resume_start=resume_start,
    )


def resume_dsqn_trainer(
    payload: dict[str, Any],
    *,
    config: SNNConfig,
    dsqn: DSQN | None = None,
    buffer: ReplayBuffer | None = None,
    metrics_path: Path | None = None,
    start_episode: int | None = None,
) -> tuple[DSQNTrainer, int]:
    """Restore trainer weights, optimizer, replay buffer, and exploration state."""
    saved_cfg = payload["config"]
    if not isinstance(saved_cfg, SNNConfig):
        saved_cfg = SNNConfig(**saved_cfg)

    resume_start = infer_completed_episodes(
        payload,
        metrics_path=metrics_path,
        start_episode=start_episode,
    )
    validate_resume_config(saved_cfg, config, resume_start=resume_start)

    active_cfg = config.with_variant_defaults()
    model = dsqn or DSQN(active_cfg)
    model.load_state_dict(payload["dsqn_state_dict"])
    replay = buffer or ReplayBuffer(active_cfg, seed=active_cfg.seed)
    trainer = DSQNTrainer(model, replay, active_cfg)

    if "target_dsqn_state_dict" in payload:
        trainer.target_dsqn.load_state_dict(payload["target_dsqn_state_dict"])
    if "optimizer_state_dict" in payload:
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "buffer_state_dict" in payload:
        trainer.buffer.load_state_dict(payload["buffer_state_dict"])

    trainer_state = payload.get("trainer_state") or {}
    trainer._total_steps = int(trainer_state.get("total_steps", 0))
    trainer._update_count = int(trainer_state.get("update_count", 0))
    if "rng_state" in trainer_state:
        trainer._rng.bit_generator.state = trainer_state["rng_state"]

    return trainer, resume_start


def train_dsqn(
    env: NguyenEnvAdapter | None = None,
    config: SNNConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = 50,
    plant: Any | None = None,
) -> TrainResult:
    """Train DSQN on ``NguyenEnvAdapter`` and optionally save a checkpoint."""
    cfg = (config or SNNConfig()).with_variant_defaults()
    owns_env = env is None
    if env is None:
        env = NguyenEnvAdapter(plant=plant, config=cfg)
    try:
        initial_result: TrainResult | None = None
        trainer: DSQNTrainer
        if resume_path is not None:
            payload = load_checkpoint(resume_path, map_location=cfg.device)
            metrics_path = Path(resume_path).with_suffix(".metrics.json")
            trainer, resume_start = resume_dsqn_trainer(
                payload,
                config=cfg,
                metrics_path=metrics_path,
                start_episode=start_episode,
            )
            initial_result = train_result_from_payload(payload, dsqn=trainer.dsqn, config=cfg)
            result = trainer.train_episodes(
                env,
                start_episode=resume_start,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=checkpoint_interval,
                initial_result=initial_result,
            )
        else:
            dsqn = DSQN(cfg)
            buffer = ReplayBuffer(cfg, seed=cfg.seed)
            trainer = DSQNTrainer(dsqn, buffer, cfg)
            result = trainer.train_episodes(
                env,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=checkpoint_interval,
            )
        if checkpoint_path is not None:
            completed = len(result.episode_rewards)
            save_checkpoint(
                checkpoint_path,
                dsqn=result.dsqn,
                config=cfg,
                optimizer=trainer.optimizer,
                trainer=trainer,
                extra=_episode_extra(result, completed_episodes=completed),
            )
            metrics_path = Path(checkpoint_path).with_suffix(".metrics.json")
            write_train_metrics(result, metrics_path)
        return result
    finally:
        if owns_env:
            env.close()
