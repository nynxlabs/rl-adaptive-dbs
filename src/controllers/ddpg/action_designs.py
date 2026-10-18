"""Structured per-pulse action designs for the Mehregan DDPG actor.

Only :class:`PerPulseLogits` is used by a panel (as a labelled candidate). Each design keeps the Fig 3a/3b state CNN (two conv +
avg-pool blocks, flatten, 2x Linear 256) and the Fig 3b critic layout (state
branch + ``Linear(ReLU(W a))`` action branch, added, ``Linear -> 1``) and swaps
only the actor head + action decoder. The step is ``l = 2000`` one-ms slots and
the mean rate fixes ``n_pulses = f * l / 1000`` (90 at 45 Hz).

Every design exposes the same hooks so one Alg. 1 loop can drive it:

- ``raw_dim``: actor head width; ``view_dim``: critic action-branch width.
- ``explore(raw, sigma, gen) -> stored``: the exploration action kept in replay.
- ``view_stored(stored)``: what the critic scores for a replayed transition.
- ``view_actor(raw, gen)``: the differentiable critic input for Eq. 5 and for
  the target actor in Eq. 3.
- ``decode_stored(stored)`` / ``decode_raw(raw)``: pulse onsets (ms) played.

Designs (Fig 3a deviations noted per class):

1. :class:`TopK` - one score per slot, play the ``n_pulses`` highest.
2. :class:`IsiDesign` - ``n_pulses`` tanh ISI deviations, cumulative-summed and
   rescaled to fill the step.
3. :class:`PerPulseRelaxed` - the Fig 3a ``n_pulses x l`` logits with a
   straight-through or Gumbel-softmax view for the critic.
4. :class:`LargeMenu` - a few hundred rate-matched patterns, argmax over logits.
5. :class:`PerPulseLogits` - the Fig 3a head with the Fig 3b critic on the raw
   logits (as drawn); trained by the panel recipe (``--set action_head=perpulse``)
   through :class:`AlgOneAgent`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controllers.ddpg.networks import StateEncoder, clone_module, hard_update, soft_update

STEP_MS = 2000


class StateCNN(StateEncoder):
    """Fig 3a/3b state branch: two conv + avg-pool blocks, flatten, 2x Linear 256.

    ``state_len = 2000`` is Figure 3a's ``1 x l`` input (``fc1`` fan-in 32,000);
    ``state_len = 1`` is the shipped menu's length-1 state (no pooling, fan-in 64).
    """

    def __init__(self, *, state_len: int, fc_hidden: int = 256) -> None:
        super().__init__(state_length=state_len, fc_hidden=fc_hidden)


class HeadActor(nn.Module):
    """State CNN -> Linear head (raw outputs; decoding happens in the design)."""

    def __init__(self, *, state_len: int, raw_dim: int, fc_hidden: int = 256) -> None:
        super().__init__()
        self.encoder = StateCNN(state_len=state_len, fc_hidden=fc_hidden)
        self.head = nn.Linear(fc_hidden, raw_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))


class ViewCritic(nn.Module):
    """Fig 3b critic with a configurable action-branch width."""

    def __init__(self, *, state_len: int, view_dim: int, fc_hidden: int = 256) -> None:
        super().__init__()
        self.encoder = StateCNN(state_len=state_len, fc_hidden=fc_hidden)
        self.action_fc = nn.Linear(view_dim, fc_hidden)
        self.out = nn.Linear(fc_hidden, 1)

    def forward(self, state: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
        return self.out(self.encoder(state) + F.relu(self.action_fc(view))).squeeze(-1)


def regular_onsets(n_pulses: int, n_slots: int = STEP_MS) -> np.ndarray:
    """Regular train at the mean rate: ``round(k * l / n)`` ms."""
    return np.minimum(np.round(np.arange(n_pulses) * n_slots / n_pulses).astype(np.int64), n_slots - 1)


def _gumbel(shape: torch.Size, gen: torch.Generator | None) -> torch.Tensor:
    u = torch.rand(shape, generator=gen).clamp_(1e-10, 1.0 - 1e-10)
    return -torch.log(-torch.log(u))


@dataclass
class TopK:
    """Design 1: one score per slot; the action is the ``n_pulses`` top slots.

    ``critic_view="raw"`` feeds the critic the 2000 scores (the Fig 3b
    critic-on-logits idea); ``"soft"`` feeds ``sigmoid((score - thr) / temp)``
    with ``thr`` the detached midpoint between the k-th and (k+1)-th score, a
    differentiable relaxation of the top-k mask.
    """

    n_pulses: int
    n_slots: int = STEP_MS
    critic_view: str = "raw"
    temp: float = 0.5
    init_bias: float = 1.0
    name: str = "topk"

    @property
    def raw_dim(self) -> int:
        return self.n_slots

    @property
    def view_dim(self) -> int:
        return self.n_slots

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.zero_()
            actor.head.bias[torch.as_tensor(regular_onsets(self.n_pulses, self.n_slots))] = self.init_bias

    def _view(self, scores: torch.Tensor) -> torch.Tensor:
        if self.critic_view == "raw":
            return scores
        top = torch.topk(scores, self.n_pulses + 1, dim=-1).values
        thr = (0.5 * (top[..., -1] + top[..., -2])).detach().unsqueeze(-1)
        return torch.sigmoid((scores - thr) / self.temp)

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        return raw + sigma * torch.randn(raw.shape, generator=gen)

    def view_stored(self, stored: torch.Tensor) -> torch.Tensor:
        return self._view(stored)

    def view_actor(self, raw: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        return self._view(raw)

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return self.decode_raw(stored)

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        idx = torch.topk(raw.reshape(-1).float(), self.n_pulses).indices
        return np.sort(idx.numpy().astype(np.int64))


@dataclass
class IsiDesign:
    """Design 2: ``n_pulses`` tanh ISI deviations around the regular period.

    ``d_k = tanh(raw_k)``; interval ``I_k = exp(beta * d_k)`` (regular period
    units); onsets ``t_k = l * sum_{j<k} I_j / sum_j I_j`` rounded to the ms grid,
    so the last interval wraps to the next step's first pulse and the count is
    exactly ``n_pulses``. ``d = 0`` is the regular train. Exploration: Gaussian
    noise on ``d``, clipped to [-1, 1] (standard DDPG on a bounded action).
    """

    n_pulses: int
    n_slots: int = STEP_MS
    beta: float = 1.5
    head_init_scale: float = 1.0
    name: str = "isi"

    @property
    def raw_dim(self) -> int:
        return self.n_pulses

    @property
    def view_dim(self) -> int:
        return self.n_pulses

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.zero_()
            actor.head.weight.mul_(self.head_init_scale)

    def onsets(self, d: torch.Tensor) -> np.ndarray:
        iv = np.exp(self.beta * d.reshape(-1).double().numpy())
        cum = np.concatenate([[0.0], np.cumsum(iv)[:-1]])
        t = np.round(self.n_slots * cum / iv.sum()).astype(np.int64)
        return np.minimum(t, self.n_slots - 1)

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        return (torch.tanh(raw) + sigma * torch.randn(raw.shape, generator=gen)).clamp(-1.0, 1.0)

    def view_stored(self, stored: torch.Tensor) -> torch.Tensor:
        return stored

    def view_actor(self, raw: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        return torch.tanh(raw)

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return self.onsets(stored)

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        return self.onsets(torch.tanh(raw.float()))


@dataclass
class PerPulseRelaxed:
    """Design 3: Fig 3a ``n_pulses x l`` logits, per-pulse argmax played.

    Exploration is Gumbel-max (``logits + sigma * Gumbel``), so the played slot
    of each pulse is a softmax sample at ``sigma = 1``. ``estimator="st"``:
    the critic scores the one-hot of the played slots, and Eq. 5 sees
    ``onehot(argmax) + p - p.detach()`` with ``p = softmax(logits / temp)``
    (straight-through). ``estimator="gs"``: the critic scores the Gumbel-softmax
    sample ``softmax((logits + sigma G) / temp)`` both in replay and in Eq. 5.
    ``critic_view="pulse"`` keeps the per-pulse ``n_pulses * l`` input (Fig 3b
    width); ``"train"`` sums over pulses (an ``l``-slot soft pulse train).
    """

    n_pulses: int
    n_slots: int = STEP_MS
    estimator: str = "st"
    critic_view: str = "pulse"
    temp: float = 1.0
    init_bias: float = 1.0
    name: str = "perpulse"

    @property
    def raw_dim(self) -> int:
        return self.n_pulses * self.n_slots

    @property
    def view_dim(self) -> int:
        return self.raw_dim if self.critic_view == "pulse" else self.n_slots

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.zero_()
            grid = actor.head.bias.view(self.n_pulses, self.n_slots)
            reg = regular_onsets(self.n_pulses, self.n_slots)
            grid[torch.arange(self.n_pulses), torch.as_tensor(reg)] = self.init_bias

    def _grid(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.n_pulses, self.n_slots)

    def _out(self, grid: torch.Tensor) -> torch.Tensor:
        if self.critic_view == "pulse":
            return grid.reshape(grid.shape[0], -1)
        return grid.sum(dim=1)

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        return raw + sigma * _gumbel(raw.shape, gen)

    def view_stored(self, stored: torch.Tensor) -> torch.Tensor:
        g = self._grid(stored)
        if self.estimator == "st":
            hard = F.one_hot(g.argmax(-1), self.n_slots).to(g.dtype)
            return self._out(hard)
        return self._out(F.softmax(g / self.temp, dim=-1))

    def view_actor(self, raw: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        g = self._grid(raw)
        if self.estimator == "st":
            p = F.softmax(g / self.temp, dim=-1)
            hard = F.one_hot(g.argmax(-1), self.n_slots).to(g.dtype)
            return self._out(hard + p - p.detach())
        return self._out(F.softmax((g + _gumbel(g.shape, gen)) / self.temp, dim=-1))

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return self.decode_raw(stored)

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        return np.sort(self._grid(raw.float())[0].argmax(-1).numpy().astype(np.int64))


@dataclass
class PerPulseLogits:
    """Design 5: Figure 3a / 3b as drawn — the critic scores the raw logits.

    The head emits ``n_pulses x l`` logits ("Linear (Action Logits)
    ``1 x f l l/1000`` -> Reshape -> Softmax(dim=1) -> Argmax(dim=-1)"); each
    pulse plays the argmax slot of its row. Exploration (paper-silent) adds
    noise to the logits, as the shipped menu does: ``"gauss"`` = Gaussian
    ``sigma``, ``"gumbel"`` = Gumbel-max (a softmax sample at ``sigma = 1``). The
    stored ``a_logit`` is the noisy vector, so the critic (Eq. 4) scores what was
    played, and Eq. 5 differentiates the critic through the raw logits.
    """

    n_pulses: int
    n_slots: int = STEP_MS
    init_bias: float = 1.0
    noise: str = "gauss"
    # Deviation (candidate): the critic scores the logits minus the actor's
    # regular-pulse init spike, so the spike (the same in every stored action)
    # is not a free bias input for the critic.
    center: bool = False
    name: str = "perpulse_logits"

    def __post_init__(self) -> None:
        spike = torch.zeros(self.n_pulses, self.n_slots)
        spike[torch.arange(self.n_pulses), torch.as_tensor(regular_onsets(self.n_pulses, self.n_slots))] = self.init_bias
        self._spike = spike.reshape(-1)

    @property
    def raw_dim(self) -> int:
        return self.n_pulses * self.n_slots

    @property
    def view_dim(self) -> int:
        return self.raw_dim

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.zero_()
            grid = actor.head.bias.view(self.n_pulses, self.n_slots)
            reg = regular_onsets(self.n_pulses, self.n_slots)
            grid[torch.arange(self.n_pulses), torch.as_tensor(reg)] = self.init_bias

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        if self.noise == "gumbel":
            return raw + sigma * _gumbel(raw.shape, gen)
        return raw + sigma * torch.randn(raw.shape, generator=gen)

    def view_stored(self, stored: torch.Tensor) -> torch.Tensor:
        return stored - self._spike if self.center else stored

    def view_actor(self, raw: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        return raw - self._spike if self.center else raw

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return self.decode_raw(stored)

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        return np.sort(raw.float().reshape(self.n_pulses, self.n_slots).argmax(-1).numpy().astype(np.int64))

    def regular_margin(self, raw: torch.Tensor) -> float:
        """Mean over pulses of (regular-slot logit - best other logit)."""
        g = raw.detach().float().reshape(self.n_pulses, self.n_slots).clone()
        rows = torch.arange(self.n_pulses)
        reg = torch.as_tensor(regular_onsets(self.n_pulses, self.n_slots))
        on = g[rows, reg].clone()
        g[rows, reg] = -float("inf")
        return float((on - g.max(-1).values).mean())


def fanin_param_groups(module: nn.Module, lr: float, ref: int) -> list[dict]:
    """Adam param groups with ``lr * min(1, ref / fan_in)`` per Linear / Conv1d.

    ``ref <= 0`` keeps one group at ``lr``. A fan-in-scaled step (as in muP-style
    parametrizations) keeps a layer's per-update change in pre-activation about
    ``lr * ref`` whatever its width; plain Adam moves a layer with fan-in ``n``
    and same-sign inputs by about ``lr * n`` per update.
    """
    if ref <= 0:
        return [{"params": list(module.parameters()), "lr": lr}]
    groups = []
    for sub in module.modules():
        if isinstance(sub, (nn.Linear, nn.Conv1d)):
            fan_in = sub.weight[0].numel()
            groups.append({"params": list(sub.parameters(recurse=False)), "lr": lr * min(1.0, ref / fan_in)})
    return groups


class AlgOneAgent:
    """Alg. 1 update (Eqs. 3-7) for a design: target nets, MSE critic, Eq. 5 actor.

    ``fanin_ref`` > 0 applies :func:`fanin_param_groups` to the networks named in
    ``fanin_scope`` (``"both"``, ``"actor"``, ``"critic"``) — a deviation from the
    single stated learning rate per network (candidate only).
    """

    def __init__(
        self,
        design,
        *,
        state_len: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        seed: int,
        fanin_ref: int = 0,
        fanin_scope: str = "both",
        actor_eps: float = 1e-8,
        critic_eps: float = 1e-8,
    ) -> None:
        torch.manual_seed(seed)
        self.d = design
        self.gamma, self.tau = gamma, tau
        self.actor = HeadActor(state_len=state_len, raw_dim=design.raw_dim)
        design.init_actor(self.actor)
        self.critic = ViewCritic(state_len=state_len, view_dim=design.view_dim)
        self.actor_t, self.critic_t = clone_module(self.actor), clone_module(self.critic)
        hard_update(self.actor_t, self.actor)
        hard_update(self.critic_t, self.critic)
        a_ref = fanin_ref if fanin_scope in ("both", "actor") else 0
        c_ref = fanin_ref if fanin_scope in ("both", "critic") else 0
        self.opt_a = torch.optim.Adam(fanin_param_groups(self.actor, actor_lr, a_ref), lr=actor_lr, eps=actor_eps, foreach=True)
        self.opt_c = torch.optim.Adam(fanin_param_groups(self.critic, critic_lr, c_ref), lr=critic_lr, eps=critic_eps, foreach=True)
        self.gen = torch.Generator().manual_seed(seed + 17)

    def update(self, s, a, r, s2, d) -> dict:
        with torch.no_grad():
            q_t = r + self.gamma * (1 - d) * self.critic_t(s2, self.d.view_actor(self.actor_t(s2), self.gen))
        loss_c = F.mse_loss(self.critic(s, self.d.view_stored(a)), q_t)
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        loss_a = -self.critic(s, self.d.view_actor(self.actor(s), self.gen)).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        soft_update(self.critic_t, self.critic, self.tau)
        soft_update(self.actor_t, self.actor, self.tau)
        return {"loss_c": loss_c.item(), "q": -loss_a.item()}


def build_menu(n_pulses: int, *, size: int = 300, n_slots: int = STEP_MS, seed: int = 0) -> list[np.ndarray]:
    """Rate-matched menu: regular, jittered, bursts, random placements.

    Every entry has exactly ``n_pulses`` distinct onsets in ``[0, l)``. Order:
    regular (index 0); bursts of 2-8 pulses at 60-200 Hz intra-burst rate with
    starts spread evenly over the step (last burst takes the remainder);
    jittered regular trains (uniform jitter 10-50 % of the period); then
    uniformly random placements until ``size`` entries.
    """
    rng = np.random.default_rng(seed)
    period = n_slots / n_pulses
    reg = regular_onsets(n_pulses, n_slots)
    menu: list[np.ndarray] = [reg]
    seen = {reg.tobytes()}

    def add(t: np.ndarray) -> None:
        t = np.sort(np.clip(np.asarray(t, dtype=np.int64), 0, n_slots - 1))
        if len(np.unique(t)) != n_pulses or t.tobytes() in seen or len(menu) >= size:
            return
        seen.add(t.tobytes())
        menu.append(t)

    for burst in (2, 3, 4, 5, 6, 8):
        for rate in (60.0, 80.0, 100.0, 120.0, 150.0, 200.0):
            intra = max(1, int(round(1000.0 / rate)))
            n_bursts = math.ceil(n_pulses / burst)
            starts = np.round(np.arange(n_bursts) * n_slots / n_bursts).astype(np.int64)
            t = [s + j * intra for i, s in enumerate(starts) for j in range(min(burst, n_pulses - i * burst))]
            add(np.asarray(t))
    n_jit = max(0, (size - len(menu)) // 3)
    for i in range(n_jit):
        frac = (0.1, 0.2, 0.33, 0.5)[i % 4]
        add(reg + rng.uniform(-frac, frac, n_pulses) * period)
    while len(menu) < size:
        add(rng.choice(n_slots, size=n_pulses, replace=False))
    return menu


@dataclass
class LargeMenu:
    """Design 4: argmax over a few hundred rate-matched patterns (as shipped)."""

    n_pulses: int
    size: int = 300
    n_slots: int = STEP_MS
    init_bias: float = 1.0
    menu_seed: int = 0
    name: str = "menu"

    def __post_init__(self) -> None:
        self.menu = build_menu(self.n_pulses, size=self.size, n_slots=self.n_slots, seed=self.menu_seed)

    @property
    def raw_dim(self) -> int:
        return len(self.menu)

    @property
    def view_dim(self) -> int:
        return len(self.menu)

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.zero_()
            actor.head.bias[0] = self.init_bias

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        return raw + sigma * torch.randn(raw.shape, generator=gen)

    def view_stored(self, stored: torch.Tensor) -> torch.Tensor:
        return stored

    def view_actor(self, raw: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        return raw

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return self.decode_raw(stored)

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        return self.menu[int(raw.reshape(-1).float().argmax())]


DESIGNS = {
    "topk": TopK,
    "isi": IsiDesign,
    "perpulse": PerPulseRelaxed,
    "perpulse_logits": PerPulseLogits,
    "menu": LargeMenu,
}


def make_design(name: str, n_pulses: int, **kwargs: object):
    return DESIGNS[name](n_pulses=n_pulses, **kwargs)


def pulse_dbs_spec(slots_ms: np.ndarray, *, mean_hz: float, step_s: float = 2.0, dt_ms: float = 0.02):
    """STN drive for one step: a standard DBS pulse at each onset (ms).

    Pulses that share a slot coincide (one pulse), as with the Fig 3a argmax.
    """
    from envs.plant.dbs import DBS_AMPLITUDE_NA_PER_CM2, DBS_PULSE_WIDTH_MS, DbsSpec

    n_steps = int(round(step_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    pulse_len = int(round(DBS_PULSE_WIDTH_MS / dt_ms))
    for slot in np.unique(np.asarray(slots_ms, dtype=np.int64)):
        start = int(round(float(slot) / dt_ms))
        idbs[start : min(start + pulse_len, n_steps)] = DBS_AMPLITUDE_NA_PER_CM2
    return DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(mean_hz).pick_dbs_freq,
        idbs=idbs,
        mean_hz=mean_hz,
    )


class SlotAlphabet:
    """Adapter so ``MehreganEnv`` / ``run_condition`` can play a slot array.

    ``to_dbs_spec(action)`` takes the onsets (ms) directly; ``env.step`` calls
    it with ``int(action)``, so the training loop sets :attr:`pending` first
    and steps with action 0.
    """

    def __init__(self, mean_hz: float, *, step_s: float = 2.0, dt_ms: float = 0.02) -> None:
        self.mean_hz = mean_hz
        self.step_duration_s = step_s
        self.dt_ms = dt_ms
        self.n_actions = 1
        self.pending: np.ndarray | None = None

    def to_dbs_spec(self, action):
        slots = self.pending if np.ndim(action) == 0 else np.asarray(action)
        return pulse_dbs_spec(slots, mean_hz=self.mean_hz, step_s=self.step_duration_s, dt_ms=self.dt_ms)
