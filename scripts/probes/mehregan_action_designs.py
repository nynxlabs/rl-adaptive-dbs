#!/usr/bin/env python3
"""Offline screen of structured per-pulse action designs under Alg. 1 (no plant).

Designs live in ``controllers.ddpg.action_designs`` (top-k slots, ISI
parametrization, per-pulse logits with straight-through / Gumbel-softmax, large
rate-matched menu). Three modes per ``--run`` spec:

- ``screen``: the Fig 3 runaway screen - a fixed 300-transition buffer filled by
  the init actor (states near the untreated level, rewards Eq. 8 on synthetic
  next states, independent of the action), then ``--updates`` Alg. 1 updates.
  Logs head-output scale and whether the noise-free decoded action moves.
- ``learn``: online Alg. 1 for 10 episodes x 30 steps with a synthetic reward
  that depends on the played pulse train in a known way (``reward=phase``: pulses
  at a hidden 20 Hz phase; ``reward=target``: distance to a hidden burst train).
  States are synthetic and action-independent (a contextual bandit with
  Alg. 1's bootstrapped target). Logs the noise-free reward per episode.
- Both modes end with a quantization report: fp16 / int8 ``quantize_dynamic``
  PTQ of the final actor and a QAT fake-quant wrap, decoded against fp32.

Spec: ``label:design=NAME,k=v,...``; run knobs ``mode, sigma, update_every,
updates, reward, reward_noise, variant, seed, gamma, tau, n_trans`` - all other
keys go to the design constructor.

  nice -n 10 uv run python scripts/probes/mehregan_action_designs.py \\
      --run "topk:design=topk,mode=learn,reward=phase" --out artifacts/probes/mehregan_action_designs/x.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controllers.ddpg.action_designs import HeadActor, ViewCritic, make_design, regular_onsets
from controllers.ddpg.networks import clone_module, hard_update, soft_update
from controllers.ddpg.quantization import QATActor
from envs.mehregan.reward import mehregan_reward

STATE_LEN = 2000
N_SLOTS = 2000
ACTOR_LR = 5e-4
CRITIC_LR = 1e-3
BATCH = 32
EPISODES = 10
STEPS = 30
BETA_HZ = 20.0


@dataclass(frozen=True)
class RunKnobs:
    mode: str = "screen"
    sigma: float = 1.0
    update_every: int = 2
    updates: int = 300
    n_trans: int = 300
    reward: str = "phase"
    reward_noise: float = 0.0
    variant: str = "fp32"
    seed: int = 0
    gamma: float = 0.99
    tau: float = 0.005
    mean_hz: float = 45.0


def _coerce(v: str):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def parse_spec(spec: str) -> tuple[str, str, dict, RunKnobs]:
    label, _, rest = spec.partition(":")
    kv = dict(p.split("=", 1) for p in rest.split(",") if p)
    design = kv.pop("design")
    run_fields = {f.name: f.type for f in fields(RunKnobs)}
    knobs = RunKnobs()
    dkw = {}
    for k, v in kv.items():
        if k in run_fields:
            cur = getattr(knobs, k)
            knobs = replace(knobs, **{k: type(cur)(v) if not isinstance(cur, str) else v})
        else:
            dkw[k] = _coerce(v)
    return label, design, dkw, knobs


# --- synthetic rewards ------------------------------------------------------


class Rewards:
    """Known functions of the pulse train (onsets in ms, length n_pulses)."""

    def __init__(self, n_pulses: int, seed: int) -> None:
        rng = np.random.default_rng(1000 + seed)
        self.phi = float(rng.uniform(0, 2 * np.pi))
        self.n = n_pulses
        # hidden target: bursts of 3 pulses at 90 Hz intra-burst, bursts evenly spaced
        nb = int(np.ceil(n_pulses / 3))
        starts = np.round(np.arange(nb) * N_SLOTS / nb).astype(np.int64)
        tgt = [s + j * 11 for i, s in enumerate(starts) for j in range(min(3, n_pulses - 3 * i))]
        self.target = np.sort(np.asarray(tgt, dtype=np.int64))
        reg = regular_onsets(n_pulses, N_SLOTS)
        self.d_reg = float(np.mean(np.abs(reg - self.target)))
        cosv = np.cos(2 * np.pi * BETA_HZ * np.arange(N_SLOTS) / 1000.0 + self.phi)
        self._cos = cosv
        self.phase_best = 2.0 * float(np.sort(cosv)[-n_pulses:].mean())

    def phase(self, t: np.ndarray) -> float:
        return 2.0 * float(self._cos[np.asarray(t)].mean())

    def target_r(self, t: np.ndarray) -> float:
        return -float(np.mean(np.abs(np.sort(t) - self.target))) / self.d_reg

    def fn(self, name: str):
        return self.phase if name == "phase" else self.target_r

    def refs(self, name: str, design) -> dict:
        f = self.fn(name)
        rng = np.random.default_rng(7)
        rand = [f(np.sort(rng.choice(N_SLOTS, self.n, replace=False))) for _ in range(200)]
        best = self.phase_best if name == "phase" else 0.0
        out = {"regular": f(regular_onsets(self.n, N_SLOTS)), "random_mean": float(np.mean(rand)),
               "random_max": float(np.max(rand)), "best": best}
        if hasattr(design, "menu"):
            out["menu_best"] = float(max(f(m) for m in design.menu))
        return out


# --- metrics ----------------------------------------------------------------


def isi_cv(t: np.ndarray) -> float:
    u = np.unique(t)
    d = np.diff(u)
    return float(d.std() / d.mean()) if len(d) > 1 else float("nan")


def compare(t: np.ndarray, ref: np.ndarray) -> dict:
    a, b = np.sort(t), np.sort(ref)
    moved = 1.0 - len(np.intersect1d(a, b)) / len(b)
    return {"moved": float(moved), "mad_ms": float(np.mean(np.abs(a - b))) if len(a) == len(b) else float("nan")}


def make_states(rng: np.random.Generator, n: int) -> np.ndarray:
    lv = rng.normal(0.45, 0.06, size=n).clip(0.2, 0.7)
    return np.stack([np.full(STATE_LEN, v, np.float32) + rng.normal(0, 0.01, STATE_LEN).astype(np.float32) for v in lv])


def eq8(state: np.ndarray) -> float:
    return mehregan_reward(state, below_threshold="square")


# --- Alg. 1 update ----------------------------------------------------------


class Agent:
    def __init__(self, design, knobs: RunKnobs) -> None:
        torch.manual_seed(knobs.seed)
        self.d = design
        self.k = knobs
        core = HeadActor(state_len=STATE_LEN, raw_dim=design.raw_dim)
        design.init_actor(core)
        self.actor = QATActor(core) if knobs.variant == "qat" else core
        self.critic = ViewCritic(state_len=STATE_LEN, view_dim=design.view_dim)
        self.actor_t, self.critic_t = clone_module(self.actor), clone_module(self.critic)
        hard_update(self.actor_t, self.actor)
        hard_update(self.critic_t, self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.gen = torch.Generator().manual_seed(knobs.seed + 17)

    def update(self, s, a, r, s2, d) -> dict:
        with torch.no_grad():
            q_t = r + self.k.gamma * (1 - d) * self.critic_t(s2, self.d.view_actor(self.actor_t(s2), self.gen))
        loss_c = F.mse_loss(self.critic(s, self.d.view_stored(a)), q_t)
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        raw = self.actor(s)
        loss_a = -self.critic(s, self.d.view_actor(raw, self.gen)).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        gnorm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in self.actor.parameters() if p.grad is not None)))
        self.opt_a.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        soft_update(self.critic_t, self.critic, self.k.tau)
        soft_update(self.actor_t, self.actor, self.k.tau)
        return {"loss_c": float(loss_c), "q": float(-loss_a), "actor_gnorm": gnorm}


def probe_stats(agent: Agent, probe: torch.Tensor, ref_slots: list[np.ndarray], raw0: torch.Tensor, rfn) -> dict:
    with torch.no_grad():
        raw = agent.actor(probe).float()
        view = agent.d.view_actor(raw, None) if not hasattr(agent.d, "estimator") or agent.d.estimator == "st" else agent.d.view_stored(raw)
    slots = [agent.d.decode_raw(raw[i : i + 1]) for i in range(raw.shape[0])]
    cmp = [compare(t, r) for t, r in zip(slots, ref_slots, strict=True)]
    out = {
        "raw_std": float(raw.std()),
        "raw_absmax": float(raw.abs().max()),
        "raw_drift": float((raw - raw0).norm() / (raw0.norm() + 1e-12)),
        "view_std": float(view.float().std()),
        "moved_vs_init": float(np.mean([c["moved"] for c in cmp])),
        "mad_ms_vs_init": float(np.nanmean([c["mad_ms"] for c in cmp])),
        "isi_cv": float(np.mean([isi_cv(t) for t in slots])),
        "distinct_across_probe": len({t.tobytes() for t in slots}),
    }
    if rfn is not None:
        out["reward_nf"] = float(np.mean([rfn(t) for t in slots]))
    return out


# --- quantization -----------------------------------------------------------


def quant_report(agent: Agent, probe: torch.Tensor, rfn, calib: torch.Tensor) -> dict:
    """Decode the noise-free action under each precision and compare to fp32.

    fp32 actor: fp16 cast, int8 ``quantize_dynamic`` (Linear), and a fresh QAT
    fake-quant wrap calibrated on 32 synthetic states. QAT-trained actor: its
    deployed (fake-quant, observers frozen) action vs the same weights with fake
    quantization disabled.
    """
    src = agent.actor

    def decode(module: nn.Module, dtype=torch.float32) -> list[np.ndarray]:
        module.eval()
        with torch.no_grad():
            raw = module(probe.to(dtype)).float()
        return [agent.d.decode_raw(raw[i : i + 1]) for i in range(raw.shape[0])]

    if isinstance(src, QATActor):
        dep = copy.deepcopy(src)
        dep.apply(torch.ao.quantization.disable_observer)
        nofq = copy.deepcopy(src)
        nofq.apply(torch.ao.quantization.disable_fake_quant)
        ref = decode(nofq)
        kinds = {"fp32_weights": ref, "qat_deployed": decode(dep)}
    else:
        core = copy.deepcopy(src).cpu().eval()
        ref = decode(core)
        kinds = {"fp32": ref}
        kinds["fp16"] = decode(copy.deepcopy(core).half(), torch.float16)
        kinds["int8_dynamic"] = decode(
            torch.ao.quantization.quantize_dynamic(copy.deepcopy(core), {nn.Linear}, dtype=torch.qint8)
        )
        qat = QATActor(copy.deepcopy(core))
        qat.train()
        with torch.no_grad():
            for i in range(0, calib.shape[0], 8):
                qat(calib[i : i + 8])
        qat.apply(torch.ao.quantization.disable_observer)
        kinds["qat_wrap"] = decode(qat)
    out = {}
    for name, sl in kinds.items():
        cm = [compare(t, r) for t, r in zip(sl, ref, strict=True)]
        row = {
            "moved_vs_fp32": float(np.mean([c["moved"] for c in cm])),
            "mad_ms_vs_fp32": float(np.nanmean([c["mad_ms"] for c in cm])),
            "isi_cv": float(np.mean([isi_cv(t) for t in sl])),
            "frac_on_regular": float(
                np.mean([1 - compare(t, regular_onsets(len(t), N_SLOTS))["moved"] for t in sl])
            ),
        }
        if rfn is not None:
            row["reward"] = float(np.mean([rfn(t) for t in sl]))
        out[name] = row
    return out


# --- modes ------------------------------------------------------------------


def run(design_name: str, dkw: dict, knobs: RunKnobs) -> dict:
    n_pulses = int(round(knobs.mean_hz * N_SLOTS / 1000.0))
    design = make_design(design_name, n_pulses, **dkw)
    agent = Agent(design, knobs)
    rng = np.random.default_rng(knobs.seed)
    rewards = Rewards(n_pulses, knobs.seed)
    rfn = rewards.fn(knobs.reward) if knobs.mode == "learn" else None
    probe = torch.as_tensor(make_states(np.random.default_rng(99), 8))
    calib = torch.as_tensor(make_states(np.random.default_rng(98), 32))
    with torch.no_grad():
        raw0 = agent.actor(probe).float()
    ref = [design.decode_raw(raw0[i : i + 1]) for i in range(8)]
    init_q = quant_report(agent, probe, rfn, calib)
    trace = [{"step": 0, "update": 0, **probe_stats(agent, probe, ref, raw0, rfn)}]
    t0 = time.time()
    played = []
    if knobs.mode == "screen":
        states = make_states(rng, knobs.n_trans + 1)
        with torch.no_grad():
            raw = agent.actor(torch.as_tensor(states[:-1]))
        stored = torch.stack([design.explore(raw[i : i + 1], knobs.sigma, agent.gen)[0] for i in range(knobs.n_trans)])
        played = [design.decode_stored(stored[i : i + 1]) for i in range(knobs.n_trans)]
        rew = torch.as_tensor([eq8(states[i + 1]) for i in range(knobs.n_trans)], dtype=torch.float32)
        dw = torch.as_tensor([(i % STEPS) == STEPS - 1 for i in range(knobs.n_trans)], dtype=torch.float32)
        S = torch.as_tensor(states)
        marks = {5, 15, 30, 60, 120, 200, knobs.updates}
        for u in range(1, knobs.updates + 1):
            idx = torch.as_tensor(rng.integers(0, knobs.n_trans, size=BATCH))
            info = agent.update(S[idx], stored[idx], rew[idx], S[idx + 1], dw[idx])
            if u in marks:
                trace.append({"update": u, **info, **probe_stats(agent, probe, ref, raw0, None)})
    else:
        buf_s, buf_a, buf_r, buf_s2, buf_d = [], [], [], [], []
        n_up = 0
        step = 0
        for ep in range(EPISODES):
            s = make_states(rng, 1)[0]
            ep_r = []
            for t in range(STEPS):
                with torch.no_grad():
                    raw = agent.actor(torch.as_tensor(s).unsqueeze(0))
                a = design.explore(raw, knobs.sigma, agent.gen)
                slots = design.decode_stored(a)
                r = rfn(slots) + knobs.reward_noise * float(rng.standard_normal())
                s2 = make_states(rng, 1)[0]
                buf_s.append(s); buf_a.append(a[0]); buf_r.append(r); buf_s2.append(s2)
                buf_d.append(1.0 if t == STEPS - 1 else 0.0)
                ep_r.append(rfn(slots))
                played.append(slots)
                s = s2
                step += 1
                if len(buf_s) >= BATCH and step % knobs.update_every == 0:
                    idx = rng.integers(0, len(buf_s), size=BATCH)
                    info = agent.update(
                        torch.as_tensor(np.stack([buf_s[i] for i in idx])),
                        torch.stack([buf_a[i] for i in idx]),
                        torch.as_tensor([buf_r[i] for i in idx], dtype=torch.float32),
                        torch.as_tensor(np.stack([buf_s2[i] for i in idx])),
                        torch.as_tensor([buf_d[i] for i in idx], dtype=torch.float32),
                    )
                    n_up += 1
            trace.append({"episode": ep + 1, "step": step, "update": n_up, "played_reward": float(np.mean(ep_r)),
                          **({k: v for k, v in info.items()} if n_up else {}),
                          **probe_stats(agent, probe, ref, raw0, rfn)})
    elapsed = time.time() - t0
    out = {
        "design": design_name, "design_kwargs": dkw, "knobs": knobs.__dict__, "n_pulses": n_pulses,
        "raw_dim": design.raw_dim, "view_dim": design.view_dim, "elapsed_s": elapsed,
        "played_isi_cv_mean": float(np.mean([isi_cv(t) for t in played])),
        "played_distinct": len({t.tobytes() for t in played}),
        "trace": trace, "quant_init": init_q, "quant_final": quant_report(agent, probe, rfn, calib),
    }
    if knobs.mode == "learn":
        out["refs"] = rewards.refs(knobs.reward, design)
        r0, r1 = trace[0]["reward_nf"], trace[-1]["reward_nf"]
        best = out["refs"].get("menu_best", out["refs"]["best"])
        out["progress"] = float((r1 - r0) / (best - r0)) if best != r0 else float("nan")
    return out


def summary(label: str, res: dict) -> str:
    tr = res["trace"]
    parts = [f"{label} [{res['elapsed_s']:.0f}s]"]
    for row in tr:
        key = f"u{row['update']}" if res["knobs"]["mode"] == "screen" else f"e{row.get('episode', 0)}"
        bit = f"{key}:std={row['raw_std']:.3g}/max={row['raw_absmax']:.3g}/mv={row['moved_vs_init']:.2f}/cv={row['isi_cv']:.2f}"
        if "reward_nf" in row:
            bit += f"/r={row['reward_nf']:.3f}"
        parts.append(bit)
    if "progress" in res:
        parts.append(f"refs={ {k: round(v, 3) for k, v in res['refs'].items()} } progress={res['progress']:.3f}")
    q = res["quant_final"]
    parts.append("quant_final " + " ".join(f"{k}:mv={v['moved_vs_fp32']:.3f}/mad={v['mad_ms_vs_fp32']:.2f}/cv={v['isi_cv']:.2f}/reg={v['frac_on_regular']:.2f}" + (f"/r={v['reward']:.3f}" if "reward" in v else "") for k, v in q.items()))
    return "\n  ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    for spec in args.run:
        label, design, dkw, knobs = parse_spec(spec)
        res = run(design, dkw, knobs)
        out[label] = res
        print(summary(label, res), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
