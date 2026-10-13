"""Shared paper protocol for Mehregan et al. Figs 4–6 (one model per mean rate).

Mehregan et al. train **one** 45 Hz DDPG model (Fig 4a/4b training curves,
Fig 5a efficacy, Fig 6a fp32 + PTQ) and **one** 30 Hz model with "all other
parameters fixed" (Fig 5b, Fig 6b fp32 + PTQ). QAT is a separate 10-episode run
"with the same settings" (Fig 6a/6b). This module holds that single recipe.

Paper-stated (§III.B, §IV.A.1, Alg. 1, Fig 3) — implemented as written:

- Fig 3 networks (``controllers/ddpg/pulse_placement.py``): the actor places
  each of the f·l/1000 pulses of a step in one of l one-ms slots (logits
  f·l·l/1000 → reshape → softmax → argmax); the critic scores
  ``Q(s, a_logit)`` on the full logit vector. l = 2000 (2 s step).
- 2 s environment step, plant step 0.02 ms, 30 steps per episode, 10 episodes.
- Replay buffer 8192, batch 32, actor lr 5e-4, critic lr 1e-3, Adam, MSE
  critic loss; critic target from the target actor's logits (Eq. 3); actor
  maximizes ``Q(s, μ(s))`` (Eq. 5); soft target updates (Eqs. 6–7).
- Reward Eq. (8) on the mean of the state, β_t = 0.35 (state on $P_\\beta/1000$).
- Action space initialized to regular pulses at the mean rate.
- Eval: fixed seed, 2 s reset, five 2 s steps of the trained policy.
- PTQ: fp16 cast and PyTorch ``quantize_dynamic`` int8; QAT: fake-quant stub
  on the actor input, dequant stub on the logits, int8.

Paper-silent conventions (labelled; tune these, not the stated mechanism):

- **State** (Fig 3: a length-l vector "formed from $P_\\beta$", a CNN input "due
  to the temporal nature of the biomarker", whose mean is the reward's
  $s_{sum}$): trailing 2 s $P_\\beta / 1000$ sampled every ``state_sample_ms``
  across the step and held to l = 2000 samples. The reset state (no history
  yet) is the reset segment's $P_\\beta$, held.
- Exploration: Gaussian noise on the actor logits; the executed slots are the
  argmax of the noisy logits and the stored ``a_logit`` is that noisy vector.
- γ, τ, replay start size, init bias strength, seeds, and the number of
  minibatch updates per environment step (Alg. 1 line 15 "for update
  frequency do" — the paper gives no value).
- Plant state carries across the 2 s steps of an episode; each episode resets
  to new initial conditions (seed + episode).
- Eval display: 2 s no-stim pre-roll so the trailing 2 s $P_\\beta$ window is full
  at display t=0, then the reset segment, then five policy steps; the trace is
  sampled every 0.2 s (Fig 2a protocol).
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controllers.ddpg.networks import clone_module, hard_update, soft_update
from controllers.ddpg.pulse_placement import (
    PulseActor,
    PulseCritic,
    pulse_dbs_spec,
    regular_slots,
)
from controllers.ddpg.quantization import QATActor
from envs.mehregan.reward import mehregan_reward
from envs.plant.biomarkers import SpectrumParams, p_beta
from envs.plant.config import PlantConfig
from envs.plant.dbs import DbsSpec
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.panel import load_script_module
from rl_adaptive_dbs.user_config import resolve_config

_FIG2A = load_script_module(
    "mehregan_fig2a_for_protocol",
    Path(__file__).resolve().parent / "2a" / "plot.py",
)

# --- Paper-stated -----------------------------------------------------------
PAPER_DT_MS = 0.02
STEP_S = 2.0
STEP_MS = 2000  # l
STEPS_PER_EPISODE = 30
NUM_EPISODES = 10
ACTOR_LR = 5e-4
CRITIC_LR = 1e-3
BUFFER_CAPACITY = 8192
BATCH_SIZE = 32
EVAL_STEPS = 5
BETA_THRESHOLD = 0.35
REWARD_SCALE = 10.0
OBS_SCALE = 1000.0
STATE_LENGTH = STEP_MS

# --- Paper-silent defaults (conventions) ------------------------------------
DEFAULT_LOGIT_NOISE = 1.0
DEFAULT_INIT_BIAS = 0.0
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_STATE_SAMPLE_MS = 100
DEFAULT_UPDATES_PER_STEP = 1

# Eval display (Fig 2a protocol).
DISPLAY_S = 12.0
ONSET_DISPLAY_S = 2.0
SAMPLE_S = 0.2
WINDOW_S = 2.0


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class PaperKnobs:
    """Paper-silent knobs. Everything the paper states is fixed above."""

    logit_noise_std: float = DEFAULT_LOGIT_NOISE
    init_bias_scale: float = DEFAULT_INIT_BIAS
    gamma: float = DEFAULT_GAMMA
    tau: float = DEFAULT_TAU
    state_sample_ms: int = DEFAULT_STATE_SAMPLE_MS
    updates_per_step: int = DEFAULT_UPDATES_PER_STEP  # Alg. 1 line 15 "update frequency"
    # Linear anneal of the logit noise to this σ by the last episode (None = constant).
    logit_noise_end: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "logit_noise_std": self.logit_noise_std,
            "init_bias_scale": self.init_bias_scale,
            "gamma": self.gamma,
            "tau": self.tau,
            "state_sample_ms": self.state_sample_ms,
            "updates_per_step": self.updates_per_step,
            "logit_noise_end": self.logit_noise_end,
        }

    def noise_std(self, episode: int, num_episodes: int) -> float:
        if self.logit_noise_end is None or num_episodes < 2:
            return self.logit_noise_std
        frac = episode / (num_episodes - 1)
        return self.logit_noise_std + frac * (self.logit_noise_end - self.logit_noise_std)


_KNOB_TYPES = {
    "logit_noise_std": float,
    "init_bias_scale": float,
    "gamma": float,
    "tau": float,
    "state_sample_ms": int,
    "updates_per_step": int,
    "logit_noise_end": float,
}


def apply_overrides(knobs: PaperKnobs, pairs: list[str] | None) -> PaperKnobs:
    """Apply ``FIELD=VALUE`` overrides (``--set``) to paper-silent knobs only."""
    out = knobs
    for raw in pairs or []:
        key, sep, value = raw.partition("=")
        key = key.strip()
        if not sep or key not in _KNOB_TYPES:
            msg = f"--set expects one of {sorted(_KNOB_TYPES)} as FIELD=VALUE, got {raw!r}"
            raise ValueError(msg)
        out = replace(out, **{key: _KNOB_TYPES[key](value)})
    if STEP_MS % out.state_sample_ms:
        msg = f"state_sample_ms must divide {STEP_MS}"
        raise ValueError(msg)
    return out


@dataclass(frozen=True)
class PulseSpace:
    """The Fig 3 action space at one mean rate: f·l/1000 pulses × l slots."""

    mean_hz: float

    @property
    def n_pulses(self) -> int:
        return int(round(self.mean_hz * STEP_MS / 1000.0))

    @property
    def n_slots(self) -> int:
        return STEP_MS

    def to_dbs_spec(self, slots: np.ndarray) -> DbsSpec:
        return pulse_dbs_spec(slots, mean_hz=self.mean_hz, step_s=STEP_S, dt_ms=PAPER_DT_MS)

    def regular_slots(self) -> np.ndarray:
        return regular_slots(mean_hz=self.mean_hz, n_pulses=self.n_pulses, n_slots=self.n_slots)


def make_alphabet(mean_hz: float, knobs: PaperKnobs | None = None) -> PulseSpace:
    del knobs
    return PulseSpace(mean_hz=mean_hz)


def plant_config() -> PlantConfig:
    return replace(resolve_config().plant, dt_ms=PAPER_DT_MS)


def regular_spec(space: PulseSpace) -> DbsSpec:
    """Periodic train at the mean rate (the paper's periodic baseline)."""
    return space.to_dbs_spec(space.regular_slots())


# --- State ------------------------------------------------------------------

_SPECTRUM = SpectrumParams.from_dt_ms(PAPER_DT_MS)


class StateTracker:
    """Builds the length-l state from the GPi spikes of the last two steps."""

    def __init__(self, sample_ms: int) -> None:
        self.sample_ms = sample_ms
        self._segments: deque[list[np.ndarray]] = deque(maxlen=2)

    def reset(self) -> None:
        self._segments.clear()

    def push(self, gpi_spikes: list[np.ndarray], p_beta_raw: float) -> np.ndarray:
        self._segments.append([np.asarray(s, dtype=float).reshape(-1) for s in gpi_spikes])
        n = STEP_MS // self.sample_ms
        if len(self._segments) < 2:
            # Reset segment: no history for a trailing window — hold its P_beta.
            return np.full(STEP_MS, p_beta_raw / OBS_SCALE, dtype=np.float32)
        prev, cur = self._segments
        merged = [np.concatenate([a, b + STEP_S]) for a, b in zip(prev, cur, strict=True)]
        samples = np.empty(n, dtype=np.float32)
        for k in range(n):
            end = STEP_S + (k + 1) * self.sample_ms / 1000.0
            start = end - WINDOW_S
            sub = [s[(s >= start) & (s < end)] - start for s in merged]
            samples[k] = p_beta(sub, dt_ms=PAPER_DT_MS, segment_duration_s=WINDOW_S, params=_SPECTRUM)
        return np.repeat(samples / OBS_SCALE, self.sample_ms).astype(np.float32)


def reward_of(state: np.ndarray) -> float:
    return mehregan_reward(state, beta_threshold=BETA_THRESHOLD, reward_scale=REWARD_SCALE)


# --- Training (Alg. 1) ------------------------------------------------------


def make_actor(space: PulseSpace, knobs: PaperKnobs, variant: str) -> nn.Module:
    actor = PulseActor(state_len=STATE_LENGTH, n_pulses=space.n_pulses, n_slots=space.n_slots)
    actor.init_regular(mean_hz=space.mean_hz, bias_scale=knobs.init_bias_scale)
    return QATActor(actor) if variant == "qat" else actor


class Replay:
    """Alg. 1 replay buffer; grows lazily (10 × 30 transitions ≪ 8192)."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rows: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
        self.rng = np.random.default_rng(seed)

    def add(self, s: np.ndarray, logits: np.ndarray, r: float, s2: np.ndarray, dw: float) -> None:
        if len(self.rows) >= self.capacity:
            self.rows.pop(0)
        self.rows.append((s, logits.astype(np.float32), r, s2, dw))

    def __len__(self) -> int:
        return len(self.rows)

    def sample(self, n: int) -> tuple[torch.Tensor, ...]:
        idx = self.rng.integers(0, len(self.rows), size=n)
        cols = list(zip(*(self.rows[i] for i in idx), strict=True))
        return tuple(torch.as_tensor(np.stack(c), dtype=torch.float32) for c in cols)


def train_paper(
    *,
    mean_hz: float,
    seed: int,
    knobs: PaperKnobs,
    variant: str = "paper",
    checkpoint_path: Path | None = None,
    num_episodes: int = NUM_EPISODES,
    log: Callable[[str], None] = _log,
    extra: dict[str, Any] | None = None,
    qat_freeze_observer_after: int | None = None,
    episode_hook: Callable[[SimpleNamespace, int], None] | None = None,
    trainer_hook: Callable[[SimpleNamespace], None] | None = None,
) -> dict[str, Any]:
    """Run Alg. 1 and return the per-step training record.

    ``beta_trace`` is the per-step mean of the state ($s_{sum}$ in Eq. 8,
    $P_\\beta/1000$); ``segment_beta`` the full-step $P_\\beta/1000$.
    ``trainer_hook(agent)`` runs once before episode 0 and
    ``episode_hook(agent, episode)`` after each episode (probe diagnostics);
    ``agent`` holds ``actor``, ``actor_target``, ``critic``, ``critic_target``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    space = PulseSpace(mean_hz)
    actor = make_actor(space, knobs, variant)
    critic = PulseCritic(state_len=STATE_LENGTH, n_logits=space.n_pulses * space.n_slots)
    actor_t, critic_t = clone_module(actor), clone_module(critic)
    hard_update(actor_t, actor)
    hard_update(critic_t, critic)
    # Fused Adam: the same update as the default implementation, ~6x faster on
    # the 46M-weight logit head and action branch.
    opt_a = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR, fused=True)
    opt_c = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, fused=True)
    buf = Replay(BUFFER_CAPACITY, seed)
    plant = PythonPlant(config=plant_config())
    tracker = StateTracker(knobs.state_sample_ms)
    noise_rng = torch.Generator().manual_seed(seed)
    agent = SimpleNamespace(actor=actor, actor_target=actor_t, critic=critic, critic_target=critic_t)
    if trainer_hook is not None:
        trainer_hook(agent)

    beta_trace: list[float] = []
    segment_beta: list[float] = []
    step_rewards: list[float] = []
    episode_rewards: list[float] = []
    reset_beta: list[float] = []
    slot_spread: list[float] = []
    for episode in range(num_episodes):
        if qat_freeze_observer_after is not None and episode == qat_freeze_observer_after:
            for module in (actor, actor_t):
                module.apply(torch.ao.quantization.disable_observer)
            log(f"froze QAT observers at episode {episode}")
        plant.reset(seed=seed + episode)
        tracker.reset()
        res = plant.integrate(STEP_S, DbsSpec.none(), carry=True)
        state = tracker.push(res.gpi_spikes, float(res.p_beta))
        reset_beta.append(float(state.mean()))
        noise_std = knobs.noise_std(episode, num_episodes)
        ep_reward = reward_of(state)  # Alg. 1 line 7: reset reward is collected
        for step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                logits = actor(torch.as_tensor(state).unsqueeze(0)).squeeze(0)
                logits = logits + noise_std * torch.randn(
                    logits.shape, generator=noise_rng
                )
            slots = PulseActor.slots_from_logits(logits, space.n_pulses, space.n_slots)[0].numpy()
            res = plant.integrate(STEP_S, space.to_dbs_spec(slots), carry=True)
            next_state = tracker.push(res.gpi_spikes, float(res.p_beta))
            reward = reward_of(next_state)
            dw = 1.0 if step == STEPS_PER_EPISODE - 1 else 0.0
            buf.add(state, logits.numpy(), reward, next_state, dw)
            beta_trace.append(float(next_state.mean()))
            segment_beta.append(float(res.p_beta) / OBS_SCALE)
            step_rewards.append(reward)
            slot_spread.append(float(np.std(np.diff(np.sort(slots)))))
            ep_reward += reward
            state = next_state
            for _ in range(knobs.updates_per_step if len(buf) >= BATCH_SIZE else 0):
                s, a, r, s2, d = buf.sample(BATCH_SIZE)
                with torch.no_grad():
                    q_target = r + knobs.gamma * (1.0 - d) * critic_t(s2, actor_t(s2))
                critic_loss = F.mse_loss(critic(s, a), q_target)
                opt_c.zero_grad()
                critic_loss.backward()
                opt_c.step()
                for p in critic.parameters():
                    p.requires_grad_(False)
                actor_loss = -critic(s, actor(s)).mean()
                opt_a.zero_grad()
                actor_loss.backward()
                opt_a.step()
                for p in critic.parameters():
                    p.requires_grad_(True)
                soft_update(critic_t, critic, knobs.tau)
                soft_update(actor_t, actor, knobs.tau)
        episode_rewards.append(ep_reward)
        ep = slice(-STEPS_PER_EPISODE, None)
        log(
            f"episode {episode + 1}/{num_episodes} reward={ep_reward:.2f} "
            f"mean_beta={np.mean(beta_trace[ep]):.3f} isi_std_ms={np.mean(slot_spread[ep]):.1f}"
        )
        if episode_hook is not None:
            episode_hook(agent, episode)
    plant.close()

    if checkpoint_path is not None:
        save_actor(
            checkpoint_path,
            actor,
            space=space,
            knobs=knobs,
            variant=variant,
            extra={"completed_episodes": num_episodes, "seed": seed, **(extra or {})},
        )
        log(f"wrote checkpoint {checkpoint_path}")

    return {
        "mean_hz": mean_hz,
        "seed": seed,
        "variant": variant,
        "num_episodes": num_episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "knobs": knobs.as_dict(),
        "paper_stated": {
            "actor": "Fig 3a per-pulse placement",
            "n_pulses": space.n_pulses,
            "n_slots": space.n_slots,
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "buffer_capacity": BUFFER_CAPACITY,
            "batch_size": BATCH_SIZE,
            "step_s": STEP_S,
            "dt_ms": PAPER_DT_MS,
        },
        "beta_trace": beta_trace,
        "segment_beta": segment_beta,
        "step_rewards": step_rewards,
        "episode_rewards": episode_rewards,
        "reset_beta": reset_beta,
        "isi_std_ms": slot_spread,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }


# --- Checkpoints ------------------------------------------------------------


def save_actor(
    path: Path,
    actor: nn.Module,
    *,
    space: PulseSpace,
    knobs: PaperKnobs,
    variant: str,
    extra: dict[str, Any] | None = None,
) -> None:
    base = actor.actor if isinstance(actor, QATActor) else actor
    payload = {
        "format": "mehregan_fig3_actor",
        "mean_hz": space.mean_hz,
        "variant": variant,
        "knobs": knobs.as_dict(),
        "actor_state_dict": {
            k: v
            for k, v in base.state_dict().items()
            if ".activation_post_process" not in k and ".weight_fake_quant" not in k
        },
        "qat_state_dict": actor.state_dict() if isinstance(actor, QATActor) else None,
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load(path: Path) -> tuple[dict[str, Any], PulseActor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    space = PulseSpace(float(payload["mean_hz"]))
    actor = PulseActor(state_len=STATE_LENGTH, n_pulses=space.n_pulses, n_slots=space.n_slots)
    if payload.get("qat_state_dict") is None:
        actor.load_state_dict(payload["actor_state_dict"])
    return payload, actor


def load_fp32_actor(path: Path) -> nn.Module:
    payload, actor = _load(path)
    if payload.get("qat_state_dict") is not None:
        msg = f"{path} is a QAT checkpoint"
        raise ValueError(msg)
    return actor.eval()


def load_qat_actor(path: Path) -> nn.Module:
    """QAT actor with its trained fake-quant parameters, observers frozen."""
    payload, actor = _load(path)
    state = payload.get("qat_state_dict")
    if state is None:
        msg = f"{path} has no qat_state_dict — not a QAT checkpoint"
        raise ValueError(msg)
    qat = QATActor(actor)
    qat.load_state_dict(state)
    qat.apply(torch.ao.quantization.disable_observer)
    return qat.eval()


def ptq_actor(actor: nn.Module, kind: str) -> nn.Module:
    """§III.D PTQ: fp16 cast, or PyTorch ``quantize_dynamic`` int8."""
    src = copy.deepcopy(actor).cpu().eval()
    if kind == "fp16":
        return src.half()
    if kind == "int8":
        return torch.ao.quantization.quantize_dynamic(src, {nn.Linear}, dtype=torch.qint8)
    msg = f"unknown PTQ kind {kind!r}"
    raise ValueError(msg)


# --- Policies + closed-loop eval --------------------------------------------

Policy = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _module_dtype(module: nn.Module) -> torch.dtype:
    for p in module.parameters():
        if torch.is_floating_point(p):
            return p.dtype
    return torch.float32


def module_policy(module: nn.Module, space: PulseSpace) -> Policy:
    """Deployed policy: argmax slot per pulse (no exploration noise)."""
    module.eval()
    dtype = _module_dtype(module)

    def act(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(dtype)
            logits = module(x).float().squeeze(0)
        slots = PulseActor.slots_from_logits(logits, space.n_pulses, space.n_slots)[0]
        return slots.numpy().copy(), logits.numpy()

    return act


def sample_times() -> np.ndarray:
    return _FIG2A.sample_times(SAMPLE_S, duration_s=DISPLAY_S)


def run_condition(
    *,
    seed: int,
    alphabet: PulseSpace,
    policy: Policy | None = None,
    fixed_spec: DbsSpec | None = None,
    knobs: PaperKnobs | None = None,
) -> dict[str, Any]:
    """One eval condition on a fixed seed with plant state carried across steps.

    ``policy`` → closed loop: each 2 s step places the pulses by argmax of the
    actor on the current state (built exactly as in training). ``fixed_spec``
    → the same DBS every step. Neither → no stimulation.
    """
    knobs = knobs or PaperKnobs()
    plant = PythonPlant(config=plant_config())
    plant.reset(seed=seed)
    tracker = StateTracker(knobs.state_sample_ms)
    spikes: list[list[np.ndarray]] = []
    seg_beta: list[float] = []
    slots_rows: list[list[int]] = []
    states: list[np.ndarray] = []

    def _seg(spec: DbsSpec) -> tuple[list[np.ndarray], float]:
        res = plant.integrate(STEP_S, spec, carry=True)
        if res.p_beta is None:
            raise RuntimeError("plant integrate returned no p_beta")
        offset = len(spikes) * STEP_S
        spikes.append([np.asarray(s, dtype=float).reshape(-1) + offset for s in res.gpi_spikes])
        seg_beta.append(float(res.p_beta))
        return res.gpi_spikes, float(res.p_beta)

    _seg(DbsSpec.none())  # pre-roll (display −2..0)
    gpi, beta = _seg(DbsSpec.none())  # reset segment (display 0..2)
    state = tracker.push(gpi, beta)
    for _ in range(EVAL_STEPS):
        if policy is not None:
            states.append(state)
            slots, _logits = policy(state)
            slots_rows.append([int(v) for v in slots])
            spec = alphabet.to_dbs_spec(slots)
        elif fixed_spec is not None:
            spec = fixed_spec
        else:
            spec = DbsSpec.none()
        gpi, beta = _seg(spec)
        state = tracker.push(gpi, beta)
    plant.close()

    n_neurons = len(spikes[0])
    merged = [np.concatenate([seg[i] for seg in spikes]) for i in range(n_neurons)]
    times = sample_times()
    trace = _FIG2A.trailing_p_beta(
        merged, dt_ms=PAPER_DT_MS, times=times, window_s=WINDOW_S, verbose=False
    )
    return {
        "time_s": times.tolist(),
        "trace": [float(v) for v in trace],
        "segment_p_beta": seg_beta,
        "actions": slots_rows,
        "closed_loop": policy is not None,
        "_states": states,
    }


def post_mean(cond: dict[str, Any], *, lo: float = 4.0, hi: float | None = None) -> float:
    t = np.asarray(cond["time_s"], dtype=float)
    y = np.asarray(cond["trace"], dtype=float)
    m = t >= lo
    if hi is not None:
        m &= t <= hi
    return float(np.mean(y[m]))


def pre_mean(cond: dict[str, Any], *, onset: float = ONSET_DISPLAY_S) -> float:
    t = np.asarray(cond["time_s"], dtype=float)
    y = np.asarray(cond["trace"], dtype=float)
    return float(np.mean(y[t <= onset]))


def strip_private(cond: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cond.items() if not k.startswith("_")}
