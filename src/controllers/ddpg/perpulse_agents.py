"""Per-pulse action candidates for the Mehregan Fig 4a study (deviations, labelled).

Every class here deviates from the paper's stated mechanism (Figure 3a/3b,
Eqs. 4-5, Adam at 5e-4) in at least one way; they exist to test what could
make a per-pulse head learn in 10 x 30 steps, as "our best guess of what the
authors may have done" candidates. The faithful head is
:class:`controllers.ddpg.action_designs.PerPulseLogits` with
:class:`~controllers.ddpg.action_designs.AlgOneAgent`.

One interface drives them all from ``paper_protocol._train_perpulse``:

- ``act(s, sigma) -> (stored, slots)``: exploration action kept in replay and
  the pulse onsets (ms) to play.
- ``update(s, a, r, s2, d) -> dict``: one update on a batch (off-policy agents
  get a replay sample, on-policy ones the most recent transitions).
- ``noise_free(s) -> slots`` and ``diag(s) -> dict`` for per-episode logging.
- ``actor``: the module saved in the checkpoint.

Designs (what the actor outputs and how it becomes pulses):

- ``logits``: Figure 3a's ``n_pulses x l`` logits, argmax per pulse.
- ``residual``: fixed regular-pulse init logits plus a bounded learned delta
  ``delta_max * tanh(raw)`` — the init cannot be erased, only reshaped.
- ``offset``: one timing offset per pulse relative to its regular slot,
  ``tanh`` in units of half a period (90 numbers at 45 Hz).
- ``mixture``: one number ``w``: each pulse sits on its regular slot with
  probability ``w``, otherwise on a uniformly random slot.

Critic views (what the critic scores) for the logits / residual designs:

- ``raw``: the logits (Eq. 4); ``delta``: the logits minus the init spike.
- ``regprob``: per-pulse probability of the regular slot (90 numbers).
- ``offhist``: pulse-averaged histogram of the offset from the regular slot
  (8 bins); ``offhist4``: 4 coarse bins (within a quarter period, the two
  outer quarters, elsewhere).
- ``ipi``: expected histogram of the intervals between adjacent pulses (11
  bins) — sees how pulses group (bursts, pairs), which the offset views
  pool away; ``ipioff4``: ``ipi`` plus the 4 offset bins. Both compressed views score the played one-hot for replayed
  actions and the exact Gumbel-max choice probabilities
  ``softmax(logits / sigma)`` for Eq. 5, so the actor's view is the expected
  value of the stored one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controllers.ddpg.action_designs import STEP_MS, HeadActor, ViewCritic, fanin_param_groups, regular_onsets
from controllers.ddpg.networks import clone_module, hard_update, soft_update

def _gumbel(shape: torch.Size, gen: torch.Generator | None) -> torch.Tensor:
    u = torch.rand(shape, generator=gen).clamp_(1e-10, 1.0 - 1e-10)
    return -torch.log(-torch.log(u))


@dataclass
class PulseGrid:
    """Geometry shared by the per-pulse designs."""

    n_pulses: int
    n_slots: int = STEP_MS

    def __post_init__(self) -> None:
        self.reg = regular_onsets(self.n_pulses, self.n_slots)
        self.period = self.n_slots / self.n_pulses
        slots = torch.arange(self.n_slots).unsqueeze(0)
        off = (slots - torch.as_tensor(self.reg).unsqueeze(1)).float() / self.period  # (n_pulses, n_slots)
        self.is_reg = off == 0
        # bins: [-0.5,-0.25) [-0.25,-0.1) [-0.1,0) {0} (0,0.1] (0.1,0.25] (0.25,0.5] far
        bins = torch.full(off.shape, 7, dtype=torch.long)
        bins[(off >= -0.5) & (off < -0.25)] = 0
        bins[(off >= -0.25) & (off < -0.1)] = 1
        bins[(off >= -0.1) & (off < 0)] = 2
        bins[off == 0] = 3
        bins[(off > 0) & (off <= 0.1)] = 4
        bins[(off > 0.1) & (off <= 0.25)] = 5
        bins[(off > 0.25) & (off <= 0.5)] = 6
        self.bins = bins
        self.n_bins = 8
        # Coarse 4-bin view: |offset| <= 0.25 period (incl. exact), (-0.5, -0.25), (0.25, 0.5], far.
        coarse = torch.full(off.shape, 3, dtype=torch.long)
        coarse[off.abs() <= 0.25] = 0
        coarse[(off >= -0.5) & (off < -0.25)] = 1
        coarse[(off > 0.25) & (off <= 0.5)] = 2
        self.bins4 = coarse
        # Inter-pulse intervals between adjacent pulse rows (ms): bin 0 = the
        # pair crossed order (negative lag), then 0-2, 3-5, 6-9, 10-14, 15-20,
        # 21-27, 28-35, 36-45, 46-60, > 60 ms.
        self.fft_n = 4096
        lags = torch.arange(self.fft_n)
        lags = torch.where(lags < self.fft_n // 2, lags, lags - self.fft_n)
        edges = torch.tensor([2, 5, 9, 14, 20, 27, 35, 45, 60])
        self.lag_bins = torch.where(lags < 0, torch.zeros_like(lags), 1 + torch.bucketize(lags, edges, right=False))
        self.n_ipi = 11

    def spike(self, height: float, width_ms: int = 0) -> torch.Tensor:
        """Init logits: ``height`` on each pulse's regular slot (and the slots within ``width_ms``)."""
        s = torch.zeros(self.n_pulses, self.n_slots)
        if width_ms > 0:
            slots = torch.arange(self.n_slots).unsqueeze(0)
            near = (slots - torch.as_tensor(self.reg).unsqueeze(1)).abs() <= width_ms
            s[near] = height
        else:
            s[self.is_reg] = height
        return s.reshape(-1)

    def compress(self, probs: torch.Tensor, view: str) -> torch.Tensor:
        """``probs``: (B, n_pulses, n_slots) slot distribution per pulse."""
        if view == "regprob":
            return probs[:, self.is_reg]  # (B, n_pulses)
        if view == "ipi":
            return self.ipi_hist(probs)
        if view == "ipioff4":
            return torch.cat([self.ipi_hist(probs), self.compress(probs, "offhist4")], dim=-1)
        bins, n = (self.bins4, 4) if view == "offhist4" else (self.bins, self.n_bins)
        out = torch.zeros(probs.shape[0], n, dtype=probs.dtype)
        idx = bins.reshape(1, -1).expand(probs.shape[0], -1)
        out.scatter_add_(1, idx, probs.reshape(probs.shape[0], -1))
        return out / self.n_pulses

    def ipi_hist(self, probs: torch.Tensor) -> torch.Tensor:
        """Expected inter-pulse-interval histogram of adjacent pulse rows.

        ``corr_k[d] = sum_i p_k(i) p_{k+1}(i + d)`` (FFT cross-correlation) is
        the distribution of the lag between pulse k and pulse k+1 when each
        pulse's slot is drawn independently from its row; for played one-hots
        it is the actual interval. Averaged over adjacent pairs.
        """
        b = probs.shape[0]
        spec = torch.fft.rfft(probs, n=self.fft_n, dim=-1)
        corr = torch.fft.irfft(torch.conj(spec[:, :-1]) * spec[:, 1:], n=self.fft_n, dim=-1).clamp_min(0.0)
        out = torch.zeros(b, self.n_ipi, dtype=probs.dtype)
        out.scatter_add_(1, self.lag_bins.unsqueeze(0).expand(b, -1), corr.sum(1))
        return out / (self.n_pulses - 1)

    def view_dim(self, view: str) -> int:
        return {"regprob": self.n_pulses, "offhist4": 4, "ipi": self.n_ipi, "ipioff4": self.n_ipi + 4}.get(view, self.n_bins)

    def on_regular(self, slots: np.ndarray) -> float:
        return float(np.isin(self.reg, slots).mean())


# --- designs ------------------------------------------------------------------


@dataclass
class LogitsDesign:
    """Per-pulse logits (``residual=False``: Figure 3a head; ``True``: init + bounded delta)."""

    grid: PulseGrid
    init_bias: float = 1.0
    residual: bool = False
    delta_max: float = 3.0
    view: str = "raw"  # raw | delta | regprob | offhist | offhist4
    noise: str = "gauss"
    init_width: int = 0  # ms either side of the regular slot that share the init bias

    def __post_init__(self) -> None:
        self._spike = self.grid.spike(self.init_bias, self.init_width)
        if self.view in ("regprob", "offhist", "offhist4", "ipi", "ipioff4") and self.noise != "gumbel":
            msg = "compressed critic views need pp_noise=gumbel (exact choice probabilities)"
            raise ValueError(msg)

    @property
    def raw_dim(self) -> int:
        return self.grid.n_pulses * self.grid.n_slots

    @property
    def view_dim(self) -> int:
        return self.raw_dim if self.view in ("raw", "delta") else self.grid.view_dim(self.view)

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.zero_()
            if self.residual:
                actor.head.weight.mul_(0.01)  # tanh(raw) ~ 0: start on the init logits
            else:
                actor.head.bias.copy_(self._spike)

    def logits(self, raw: torch.Tensor) -> torch.Tensor:
        return self._spike + self.delta_max * torch.tanh(raw) if self.residual else raw

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        lg = self.logits(raw)
        noise = _gumbel(lg.shape, gen) if self.noise == "gumbel" else torch.randn(lg.shape, generator=gen)
        return lg + sigma * noise  # stored = played noisy logits

    def _grid3(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.grid.n_pulses, self.grid.n_slots)

    def view_stored(self, stored: torch.Tensor, sigma: float) -> torch.Tensor:
        if self.view == "raw":
            return stored
        if self.view == "delta":
            return stored - self._spike
        g = self._grid3(stored)
        hard = F.one_hot(g.argmax(-1), self.grid.n_slots).to(g.dtype)
        return self.grid.compress(hard, self.view)

    def view_actor(self, raw: torch.Tensor, sigma: float) -> torch.Tensor:
        lg = self.logits(raw)
        if self.view == "raw":
            return lg
        if self.view == "delta":
            return lg - self._spike
        return self.grid.compress(F.softmax(self._grid3(lg) / max(sigma, 1e-3), dim=-1), self.view)

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return np.sort(self._grid3(stored.detach().float())[0].argmax(-1).numpy().astype(np.int64))

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        return self.decode_stored(self.logits(raw.detach()))

    def diag(self, raw: torch.Tensor) -> dict:
        lg = self._grid3(self.logits(raw.detach()).float())[0].clone()
        on = lg[self.grid.is_reg].clone()
        lg[self.grid.is_reg] = -float("inf")
        return {"margin": float((on - lg.max(-1).values).mean()), "logit_std": float(self.logits(raw.detach()).std())}


@dataclass
class OffsetDesign:
    """One offset per pulse from its regular slot: ``tanh`` x half a period.

    Exploration adds Gaussian noise before the ``tanh`` so offsets stay inside
    the period without piling up at the clip. ``raw = 0`` is the regular train.
    """

    grid: PulseGrid
    name: str = "offset"

    @property
    def raw_dim(self) -> int:
        return self.grid.n_pulses

    @property
    def view_dim(self) -> int:
        return self.grid.n_pulses

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.zero_()
            actor.head.weight.mul_(0.01)

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        return torch.tanh(raw + sigma * torch.randn(raw.shape, generator=gen))

    def view_stored(self, stored: torch.Tensor, sigma: float) -> torch.Tensor:
        return stored

    def view_actor(self, raw: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.tanh(raw)

    def _slots(self, off: torch.Tensor) -> np.ndarray:
        o = off.detach().reshape(-1).double().numpy() * self.grid.period / 2
        return np.sort(np.clip(np.round(self.grid.reg + o), 0, self.grid.n_slots - 1).astype(np.int64))

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return self._slots(stored)

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        return self._slots(torch.tanh(raw))

    def diag(self, raw: torch.Tensor) -> dict:
        return {"mean_abs_offset": float(torch.tanh(raw.detach()).abs().mean())}


@dataclass
class MixtureDesign:
    """One number ``w = sigmoid(raw)``: each pulse regular with probability ``w``, else random.

    The stochastic placement is part of the environment; the stored action is
    ``w``. ``init_logit`` sets the starting ``w`` (``-3`` -> 0.05).
    """

    grid: PulseGrid
    init_logit: float = -3.0
    seed: int = 0
    name: str = "mixture"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(10_000 + self.seed)

    @property
    def raw_dim(self) -> int:
        return 1

    @property
    def view_dim(self) -> int:
        return 1

    def init_actor(self, actor: HeadActor) -> None:
        with torch.no_grad():
            actor.head.bias.fill_(self.init_logit)
            actor.head.weight.mul_(0.01)

    def explore(self, raw: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
        return torch.sigmoid(raw + sigma * torch.randn(raw.shape, generator=gen))

    def view_stored(self, stored: torch.Tensor, sigma: float) -> torch.Tensor:
        return stored

    def view_actor(self, raw: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.sigmoid(raw)

    def _slots(self, w: float) -> np.ndarray:
        keep = self.rng.random(self.grid.n_pulses) < w
        rand = self.rng.integers(0, self.grid.n_slots, self.grid.n_pulses)
        return np.sort(np.where(keep, self.grid.reg, rand).astype(np.int64))

    def decode_stored(self, stored: torch.Tensor) -> np.ndarray:
        return self._slots(float(stored.reshape(-1)[0]))

    def decode_raw(self, raw: torch.Tensor) -> np.ndarray:
        return self._slots(float(torch.sigmoid(raw.detach()).reshape(-1)[0]))

    def diag(self, raw: torch.Tensor) -> dict:
        return {"w": float(torch.sigmoid(raw.detach()).reshape(-1)[0])}


# --- agents ---------------------------------------------------------------------


@dataclass
class AgentKnobs:
    state_len: int = 1
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    seed: int = 0
    sigma: float = 1.0
    twin: bool = False  # TD3: two critics, min target
    target_noise: float = 0.0  # TD3: noise on the target actor's raw output
    policy_delay: int = 1  # TD3: actor update every k critic updates
    grad_clip: float = 0.0  # global grad-norm clip (0 = off)
    pull_l2: float = 0.0  # actor loss + pull_l2 * mean(raw^2) (residual / offset: pull toward the init)
    bc_steps: int = 0  # behaviour-clone the head to the regular pattern before training
    critic_warmup: int = 0  # critic-only updates before the first actor update
    qat: bool = False  # wrap the actor in the paper's fake-quant QATActor (Fig 6 QAT)
    qat_avg_const: float = 0.0  # QAT activation observer averaging constant (0 = PyTorch default)
    pg_baseline: float = 0.9  # REINFORCE: EMA factor of the reward baseline


class DDPGPulseAgent:
    """Alg. 1 (Eqs. 3-7) on any design, with optional TD3 stabilisers."""

    on_policy = False

    def __init__(self, design, k: AgentKnobs) -> None:
        torch.manual_seed(k.seed)
        self.d, self.k = design, k
        self.actor = HeadActor(state_len=k.state_len, raw_dim=design.raw_dim)
        design.init_actor(self.actor)
        if k.bc_steps:
            behaviour_clone(self.actor, design, k)
        if k.qat:
            from controllers.ddpg.quantization import QATActor

            self.actor = QATActor(self.actor, averaging_constant=k.qat_avg_const or None)
        n_critics = 2 if k.twin else 1
        self.critics = [ViewCritic(state_len=k.state_len, view_dim=design.view_dim) for _ in range(n_critics)]
        self.actor_t = clone_module(self.actor)
        hard_update(self.actor_t, self.actor)
        self.critics_t = [clone_module(c) for c in self.critics]
        for ct, c in zip(self.critics_t, self.critics, strict=True):
            hard_update(ct, c)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=k.actor_lr, foreach=True)
        self.opt_c = torch.optim.Adam([p for c in self.critics for p in c.parameters()], lr=k.critic_lr, foreach=True)
        self.gen = torch.Generator().manual_seed(k.seed + 17)
        self.n_updates = 0

    def act(self, s: torch.Tensor, sigma: float) -> tuple[torch.Tensor, np.ndarray]:
        with torch.no_grad():
            stored = self.d.explore(self.actor(s), sigma, self.gen)[0]
        return stored, self.d.decode_stored(stored)

    def noise_free(self, s: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.d.decode_raw(self.actor(s))

    def diag(self, s: torch.Tensor) -> dict:
        with torch.no_grad():
            return self.d.diag(self.actor(s))

    def _clip(self, params) -> None:
        if self.k.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, self.k.grad_clip)

    def update(self, s, a, r, s2, d) -> dict:
        sig = self.k.sigma
        with torch.no_grad():
            raw2 = self.actor_t(s2)
            if self.k.target_noise > 0:
                raw2 = raw2 + self.k.target_noise * torch.randn(raw2.shape, generator=self.gen)
            v2 = self.d.view_actor(raw2, sig)
            q2 = torch.min(torch.stack([ct(s2, v2) for ct in self.critics_t]), dim=0).values
            q_t = r + self.k.gamma * (1 - d) * q2
        va = self.d.view_stored(a, sig)
        loss_c = sum(F.mse_loss(c(s, va), q_t) for c in self.critics)
        self.opt_c.zero_grad()
        loss_c.backward()
        self._clip([p for c in self.critics for p in c.parameters()])
        self.opt_c.step()
        self.n_updates += 1
        out = {"loss_c": loss_c.item(), "q": float("nan")}
        if self.n_updates > self.k.critic_warmup and self.n_updates % self.k.policy_delay == 0:
            c0 = self.critics[0]
            for p in c0.parameters():
                p.requires_grad_(False)
            raw = self.actor(s)
            q = c0(s, self.d.view_actor(raw, sig)).mean()
            loss_a = -q + (self.k.pull_l2 * raw.pow(2).mean() if self.k.pull_l2 else 0.0)
            self.opt_a.zero_grad()
            loss_a.backward()
            self._clip(self.actor.parameters())
            self.opt_a.step()
            for p in c0.parameters():
                p.requires_grad_(True)
            soft_update(self.actor_t, self.actor, self.k.tau)
            out["q"] = q.item()
        for ct, c in zip(self.critics_t, self.critics, strict=True):
            soft_update(ct, c, self.k.tau)
        return out


class ReinforcePulseAgent:
    """Score-function (REINFORCE) update on the per-pulse softmax policy, no critic.

    The state barely changes between steps, so the task is close to a bandit:
    the gradient of ``E[r]`` is ``E[(r - b) * grad log pi(slots)]`` with
    ``pi = prod_k softmax(logits_k / sigma)`` (exact under Gumbel-max sampling)
    and ``b`` an EMA of past rewards. Only played slots get credit, so the
    update concentrates on the ~90 slots each sample used.
    """

    on_policy = True

    def __init__(self, design: LogitsDesign, k: AgentKnobs) -> None:
        if design.noise != "gumbel":
            msg = "REINFORCE needs pp_noise=gumbel (exact log-probabilities)"
            raise ValueError(msg)
        torch.manual_seed(k.seed)
        self.d, self.k = design, k
        self.actor = HeadActor(state_len=k.state_len, raw_dim=design.raw_dim)
        design.init_actor(self.actor)
        if k.bc_steps:
            behaviour_clone(self.actor, design, k)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=k.actor_lr, foreach=True)
        self.gen = torch.Generator().manual_seed(k.seed + 17)
        self.baseline: float | None = None

    act = DDPGPulseAgent.act
    noise_free = DDPGPulseAgent.noise_free
    diag = DDPGPulseAgent.diag

    def update(self, s, a, r, s2, d) -> dict:
        g = self.d._grid3(a)
        played = g.argmax(-1)  # (B, n_pulses)
        if self.baseline is None:
            self.baseline = float(r.mean())
        adv = r - self.baseline
        self.baseline = self.k.pg_baseline * self.baseline + (1 - self.k.pg_baseline) * float(r.mean())
        raw = self.actor(s)
        logp = F.log_softmax(self.d._grid3(self.d.logits(raw)) / max(self.k.sigma, 1e-3), dim=-1)
        lp = logp.gather(-1, played.unsqueeze(-1)).squeeze(-1).sum(-1)  # (B,)
        loss = -(adv * lp).mean() + (self.k.pull_l2 * raw.pow(2).mean() if self.k.pull_l2 else 0.0)
        self.opt_a.zero_grad()
        loss.backward()
        if self.k.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.k.grad_clip)
        self.opt_a.step()
        return {"loss_c": float("nan"), "q": float(adv.mean())}


def behaviour_clone(actor: HeadActor, design, k: AgentKnobs, *, lr: float = 1e-3) -> None:
    """Fit the head's weights to the regular pattern before training (bias stays at its init).

    Logits designs: cross-entropy of each pulse's slot distribution toward its
    regular slot, until the mean regular-slot margin reaches the design's
    ``init_bias`` (or ``k.bc_steps`` steps). The init then lives in the
    weights instead of a bias spike.
    """
    if not isinstance(design, LogitsDesign) or design.residual:
        return
    with torch.no_grad():
        actor.head.bias.zero_()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    states = torch.linspace(0.3, 0.6, 16).unsqueeze(1).expand(16, k.state_len).contiguous()
    target = torch.as_tensor(design.grid.reg).unsqueeze(0).expand(16, -1)
    for _ in range(k.bc_steps):
        lg = design._grid3(actor(states))
        loss = F.cross_entropy(lg.reshape(-1, design.grid.n_slots), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if design.diag(actor(states[:1]))["margin"] >= design.init_bias:
                break


def make_agent(*, design: str, algo: str, n_pulses: int, k: AgentKnobs, init_bias: float, noise: str,
               view: str, delta_max: float, mix_init: float, init_width: int = 0):
    grid = PulseGrid(n_pulses)
    if design in ("logits", "residual"):
        d = LogitsDesign(grid, init_bias=init_bias, residual=design == "residual", delta_max=delta_max,
                         view=view, noise=noise, init_width=init_width)
    elif design == "offset":
        d = OffsetDesign(grid)
    elif design == "mixture":
        d = MixtureDesign(grid, init_logit=mix_init, seed=k.seed)
    else:
        msg = f"unknown pp_design {design!r}"
        raise ValueError(msg)
    if algo == "reinforce":
        return ReinforcePulseAgent(d, k), grid
    if algo == "td3":  # twin critics, target policy smoothing, delayed actor updates
        k = replace(k, twin=True, target_noise=k.target_noise or 0.2, policy_delay=max(k.policy_delay, 2))
        return DDPGPulseAgent(d, k), grid
    if algo == "ddpg":
        return DDPGPulseAgent(d, k), grid
    msg = f"unknown pp_algo {algo!r}"
    raise ValueError(msg)


__all__ = [
    "AgentKnobs",
    "DDPGPulseAgent",
    "LogitsDesign",
    "MixtureDesign",
    "OffsetDesign",
    "PulseGrid",
    "ReinforcePulseAgent",
    "make_agent",
]
