"""SEA-DBS trainer — Algorithm 1 (Ravivarapu §IV, §9)."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.buffer import ReplayBuffer, Transition
from controllers.sea_dbs.checkpoint import (
    infer_sea_dbs_start_episode,
    load_checkpoint,
    save_checkpoint,
    validate_resume_config,
)
from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.networks import (
    Actor,
    Critic,
    PredictiveModel,
    action_one_hot,
    clone_module,
    gumbel_softmax_sample,
    soft_update,
)


@dataclass
class TrainMetrics:
    episode_rewards: list[float] = field(default_factory=list)
    episode_psd: list[float] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    pred_losses: list[float] = field(default_factory=list)


@dataclass
class TrainResult:
    config: SEADBSConfig
    actor: Actor
    critic: Critic
    predictive_model: PredictiveModel | None
    metrics: TrainMetrics
    episode_rewards: list[float] = field(default_factory=list)
    episode_psd: list[float] = field(default_factory=list)
    update_count: int = 0


class SEA_DBSTrainer:
    """Online actor–critic with optional PM and GS ablations."""

    def __init__(
        self,
        env: SEA_DBSEnvAdapter,
        config: SEADBSConfig,
        *,
        actor: Actor | None = None,
        critic: Critic | None = None,
        predictive_model: PredictiveModel | None = None,
    ) -> None:
        self.env = env
        self.config = config.with_variant_defaults()
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        state_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)

        self.actor = actor or Actor(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_size=config.hidden_size,
            no_stim_bias=config.actor_no_stim_bias,
        )
        self.critic = critic or Critic(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_size=config.hidden_size,
        )
        self.predictive_model: PredictiveModel | None
        if self.config.use_predictive_model:
            self.predictive_model = predictive_model or PredictiveModel(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_size=config.hidden_size,
            )
        else:
            self.predictive_model = None

        self.actor_target = clone_module(self.actor)
        self.critic_target = clone_module(self.critic)
        self.actor_target.eval()
        self.critic_target.eval()

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)
        if self.predictive_model is not None:
            self.predictive_model.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.pred_optimizer: torch.optim.Optimizer | None = None
        if self.predictive_model is not None:
            self.pred_optimizer = torch.optim.Adam(
                self.predictive_model.parameters(),
                lr=config.pred_lr,
            )

        self.buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_shape=(state_dim,),
            n_actions=n_actions,
            seed=config.seed,
        )
        self._rng = np.random.default_rng(config.seed)
        self._total_steps = 0
        self._update_count = 0
        self._current_episode = 0
        self._metrics = TrainMetrics()

    def gs_temperature(self) -> float:
        cfg = self.config
        lam = cfg.gs_lambda
        if cfg.gs_early_lambda_episode_hi > 0 and self._current_episode < cfg.gs_early_lambda_episode_hi:
            lam *= cfg.gs_early_lambda_scale
        tau = max(cfg.gs_tau_min, cfg.gs_tau0 * np.exp(-lam * self._total_steps))
        if (
            cfg.gs_late_tau_floor_episode_lo > 0
            and cfg.gs_late_tau_floor > 0.0
            and self._current_episode >= cfg.gs_late_tau_floor_episode_lo
        ):
            tau = max(tau, cfg.gs_late_tau_floor)
        return float(tau)

    def pm_warmup_scale(self) -> float:
        steps = int(self.config.pm_warmup_steps)
        if steps <= 0:
            return 1.0
        return min(1.0, self._total_steps / steps)

    def epsilon(self) -> float:
        total = self.config.num_episodes * self.config.max_episode_steps
        if total <= 0:
            return self.config.epsilon_end
        frac = min(1.0, self._total_steps / total)
        return self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device, dtype=torch.float32)

    def _select_action(self, state: np.ndarray) -> tuple[int, np.ndarray]:
        state_t = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_t).squeeze(0)
            logits_np = logits.cpu().numpy().astype(np.float32)

        if self.config.use_gumbel_softmax:
            _, action_t = gumbel_softmax_sample(
                logits,
                tau=self.gs_temperature(),
                hard=True,
            )
            action = int(action_t.item() if action_t.ndim == 0 else action_t[0].item())
            return action, logits_np

        if self._rng.random() < self.epsilon():
            action = int(self._rng.integers(0, self.config.n_actions))
            return action, logits_np

        action = int(np.argmax(logits_np))
        return action, logits_np

    def _update_step(self) -> None:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size)

        states = self._to_tensor(batch.state)
        next_states = self._to_tensor(batch.next_state)
        actions = torch.as_tensor(batch.action, device=self.device, dtype=torch.long)
        rewards = self._to_tensor(batch.reward)
        r_hat_batch = self._to_tensor(batch.r_hat)
        dw = self._to_tensor(batch.dw)
        action_oh = action_one_hot(actions, cfg.n_actions).to(self.device)

        # Critic target (Eq. 9)
        with torch.no_grad():
            next_logits = self.actor_target(next_states)
            next_action = torch.argmax(next_logits, dim=-1)
            next_oh = action_one_hot(next_action, cfg.n_actions).to(self.device)
            next_q = self.critic_target(next_states, next_oh)
            pm_scale = self.pm_warmup_scale()
            pm_term = (
                pm_scale * r_hat_batch if cfg.use_predictive_model else torch.zeros_like(rewards)
            )
            target = rewards + pm_term + cfg.gamma * (1.0 - dw) * next_q

        q_pred = self.critic(states, action_oh)
        critic_loss = F.mse_loss(q_pred, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss (Algorithm 1 line 15)
        logits = self.actor(states)
        if cfg.use_gumbel_softmax:
            relaxed, _ = gumbel_softmax_sample(
                logits,
                tau=self.gs_temperature(),
                hard=False,
            )
            actor_action_oh = relaxed
        else:
            probs = F.softmax(logits, dim=-1)
            actor_action_oh = probs
        actor_q = self.critic(states, actor_action_oh)
        actor_loss = -actor_q.mean()
        for param in self.critic.parameters():
            param.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for param in self.critic.parameters():
            param.requires_grad_(True)

        pred_loss_val: float | None = None
        if self.predictive_model is not None and self.pred_optimizer is not None:
            r_hat_pred = self.predictive_model(states, action_oh)
            pred_loss = F.mse_loss(r_hat_pred, rewards)
            self.pred_optimizer.zero_grad(set_to_none=True)
            pred_loss.backward()
            self.pred_optimizer.step()
            pred_loss_val = float(pred_loss.detach().cpu())

        soft_update(self.critic_target, self.critic, cfg.polyak_tau)
        soft_update(self.actor_target, self.actor, cfg.polyak_tau)

        self._update_count += 1
        self.metrics.critic_losses.append(float(critic_loss.detach().cpu()))
        self.metrics.actor_losses.append(float(actor_loss.detach().cpu()))
        if pred_loss_val is not None:
            self.metrics.pred_losses.append(pred_loss_val)

    @property
    def metrics(self) -> TrainMetrics:
        return self._metrics

    def load_resume_state(self, payload: dict[str, Any]) -> None:
        if "actor_target_state_dict" in payload:
            self.actor_target.load_state_dict(payload["actor_target_state_dict"])
        if "critic_target_state_dict" in payload:
            self.critic_target.load_state_dict(payload["critic_target_state_dict"])
        if "buffer_state_dict" in payload:
            self.buffer.load_state_dict(payload["buffer_state_dict"])
        if "actor_optimizer_state_dict" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        if self.predictive_model is not None and "predictive_state_dict" in payload:
            self.predictive_model.load_state_dict(payload["predictive_state_dict"])
        if self.pred_optimizer is not None and "pred_optimizer_state_dict" in payload:
            self.pred_optimizer.load_state_dict(payload["pred_optimizer_state_dict"])

        trainer_state = payload.get("trainer_state") or {}
        self._total_steps = int(trainer_state.get("total_steps", 0))
        self._update_count = int(trainer_state.get("update_count", 0))
        if "rng_state" in trainer_state:
            self._rng.bit_generator.state = trainer_state["rng_state"]
        rewards = trainer_state.get("episode_rewards")
        if not isinstance(rewards, list):
            extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
            rewards = extra.get("episode_rewards")
        if isinstance(rewards, list):
            self._metrics.episode_rewards = list(rewards)
        psd = trainer_state.get("episode_psd")
        if not isinstance(psd, list):
            extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
            psd = extra.get("episode_psd")
        if isinstance(psd, list):
            self._metrics.episode_psd = list(psd)

    def train_episodes(
        self,
        *,
        start_episode: int = 0,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 50,
        on_checkpoint: Callable[[int, TrainResult], None] | None = None,
    ) -> TrainResult:
        cfg = self.config
        result = TrainResult(
            config=cfg,
            actor=self.actor,
            critic=self.critic,
            predictive_model=self.predictive_model,
            metrics=self.metrics,
        )
        if start_episode > 0:
            result.episode_rewards = list(self._metrics.episode_rewards)
            result.episode_psd = list(self._metrics.episode_psd)

        for episode in range(start_episode, cfg.num_episodes):
            self._current_episode = episode
            ep_seed = cfg.seed if cfg.fixed_episode_seed else cfg.seed + episode
            state, info = self.env.reset(seed=ep_seed)
            episode_reward = 0.0
            episode_p_beta: list[float] = [float(info.get("p_beta_norm", 0.0))]

            for _ in range(cfg.max_episode_steps):
                action, logits = self._select_action(state)
                next_state, reward, _terminated, truncated, step_info = self.env.step(action)
                self._total_steps += 1

                r_hat = 0.0
                if self.predictive_model is not None:
                    with torch.no_grad():
                        state_t = self._to_tensor(state).unsqueeze(0)
                        a_oh = action_one_hot(
                            torch.tensor([action], device=self.device),
                            cfg.n_actions,
                        )
                        r_hat = float(self.predictive_model(state_t, a_oh).item())

                dw = float(step_info.get("dw", 1.0 if truncated else 0.0))
                self.buffer.add(
                    Transition(
                        state=np.asarray(state, dtype=np.float32),
                        action=int(action),
                        action_logits=np.asarray(logits, dtype=np.float32),
                        reward=float(reward),
                        r_hat=r_hat,
                        next_state=np.asarray(next_state, dtype=np.float32),
                        dw=dw,
                    )
                )

                if len(self.buffer) >= cfg.min_buffer_size:
                    for _ in range(cfg.update_frequency):
                        self._update_step()

                episode_reward += float(reward)
                episode_p_beta.append(float(step_info.get("p_beta_norm", 0.0)))
                state = next_state
                if truncated:
                    break

            if cfg.episode_psd_metric == "last":
                episode_psd_val = float(episode_p_beta[-1]) if episode_p_beta else 0.0
            else:
                episode_psd_val = float(np.mean(episode_p_beta)) if episode_p_beta else 0.0
            result.episode_rewards.append(episode_reward)
            result.episode_psd.append(episode_psd_val)
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.episode_psd.append(episode_psd_val)
            completed = episode + 1
            if checkpoint_path is not None and checkpoint_interval > 0:
                if completed % checkpoint_interval == 0 or completed == cfg.num_episodes:
                    save_checkpoint(
                        checkpoint_path,
                        actor=self.actor,
                        critic=self.critic,
                        config=cfg,
                        predictive_model=self.predictive_model,
                        trainer=self,
                        extra={
                            "completed_episodes": completed,
                            "episode_rewards": list(result.episode_rewards),
                            "episode_psd": list(result.episode_psd),
                            "update_count": self._update_count,
                        },
                    )
                    if on_checkpoint is not None:
                        on_checkpoint(completed, result)

            if cfg.log_episodes:
                print(
                    f"episode {episode + 1}/{cfg.num_episodes} "
                    f"reward={episode_reward:.3f} "
                    f"{'last' if cfg.episode_psd_metric == 'last' else 'mean'}_p_beta={episode_psd_val:.4f}",
                    flush=True,
                )

        result.update_count = self._update_count
        return result


def write_train_metrics(result: TrainResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "controller": "sea_dbs",
        "variant": result.config.variant,
        "seed": result.config.seed,
        "num_episodes": result.config.num_episodes,
        "episode_rewards": result.episode_rewards,
        "episode_psd": result.episode_psd,
        "update_count": result.update_count,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def train_sea_dbs(
    env: SEA_DBSEnvAdapter | None = None,
    config: SEADBSConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = 50,
    plant: Any | None = None,
) -> TrainResult:
    """Train SEA-DBS on ``SEA_DBSEnvAdapter`` and optionally save checkpoint."""
    cfg = (config or SEADBSConfig()).with_variant_defaults()
    owns_env = env is None
    if env is None:
        env = SEA_DBSEnvAdapter(plant=plant, config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        resume_start = 0
        if resume_path is not None:
            payload = load_checkpoint(resume_path, device=cfg.device)
            saved_raw = payload["sea_dbs_config"]
            saved_cfg = SEADBSConfig(**saved_raw)
            metrics_path = Path(resume_path).with_suffix(".metrics.json")
            resume_start = infer_sea_dbs_start_episode(
                payload,
                metrics_path=metrics_path,
                start_episode=start_episode,
            )
            validate_resume_config(saved_cfg, cfg, resume_start=resume_start)
            trainer.actor.load_state_dict(payload["actor_state_dict"])
            trainer.critic.load_state_dict(payload["critic_state_dict"])
            trainer.load_resume_state(payload)

        ckpt_during_train = checkpoint_path if checkpoint_path is not None else None
        result = trainer.train_episodes(
            start_episode=resume_start,
            checkpoint_path=ckpt_during_train,
            checkpoint_interval=checkpoint_interval,
        )
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                actor=result.actor,
                critic=result.critic,
                config=cfg,
                predictive_model=result.predictive_model,
                trainer=trainer,
                extra={
                    "completed_episodes": len(result.episode_rewards),
                    "episode_rewards": result.episode_rewards,
                    "episode_psd": result.episode_psd,
                    "update_count": result.update_count,
                },
            )
            metrics_path = Path(checkpoint_path).with_suffix(".metrics.json")
            write_train_metrics(result, metrics_path)
        return result
    finally:
        if owns_env:
            env.close()
