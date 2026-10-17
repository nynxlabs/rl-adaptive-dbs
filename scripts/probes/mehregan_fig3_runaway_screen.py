#!/usr/bin/env python3
"""Offline screen of Fig 3 actor logit runaway under Alg. 1 updates (no plant).

Fills a replay buffer with 300 synthetic transitions — states near the
untreated level ($P_\\beta/1000 \\approx 0.45$ with small spread), stored logits =
the regular-init actor's logits plus σ·N(0, 1), rewards from Eq. (8) (Fig 3c
reading) on the next state — then runs ``--updates`` Alg. 1 updates exactly as
``paper_protocol.train_paper`` does (critic MSE on the stored logits, Eq. 5
actor loss, soft targets) for each knob set, logging the noise-free logit std,
the max logit, and how many pulses still sit on their regular slot.

  uv run python -m rl_adaptive_dbs.run --max-threads 2 scripts/probes/mehregan_fig3_runaway_screen.py \\
      --variant base --variant "eps1:adam_eps=1.0" --variant "l2:logit_l2=1.0"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from controllers.ddpg.networks import clone_module, hard_update, soft_update
from controllers.ddpg.pulse_placement import PulseActor, PulseCritic
from rl_adaptive_dbs.panel import load_script_module

_PP = load_script_module(
    "mehregan_paper_protocol_screen",
    Path(__file__).resolve().parents[1] / "figures/papers/mehregan/paper_protocol.py",
)


def run(knobs, *, updates: int, n_trans: int, seed: int, mean_hz: float) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    space = _PP.PulseSpace(mean_hz)
    actor = _PP.make_actor(space, knobs, "paper")
    critic = PulseCritic(state_len=_PP.STATE_LENGTH, n_logits=space.n_pulses * space.n_slots)
    actor_t, critic_t = clone_module(actor), clone_module(critic)
    hard_update(actor_t, actor)
    hard_update(critic_t, critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=_PP.ACTOR_LR, eps=knobs.adam_eps,
                             betas=(0.9, knobs.adam_beta2), fused=True)
    opt_c = torch.optim.Adam(critic.parameters(), lr=_PP.CRITIC_LR, fused=True)
    L = _PP.STATE_LENGTH
    lv = rng.normal(0.45, 0.06, size=n_trans + 1).clip(0.2, 0.7)
    states = np.stack([np.full(L, v, np.float32) + rng.normal(0, 0.01, L).astype(np.float32) for v in lv])
    with torch.no_grad():
        base = actor(torch.as_tensor(states[:n_trans])).numpy()
    logits = base + knobs.logit_noise_std * rng.standard_normal(base.shape).astype(np.float32)
    rewards = np.array([_PP.reward_of(states[i + 1]) for i in range(n_trans)], np.float32)
    dw = np.array([(i % 30) == 29 for i in range(n_trans)], np.float32)
    reg = torch.as_tensor(space.regular_slots())
    probe = torch.as_tensor(states[:8])
    rows = []
    for u in range(updates + 1):
        if u in (0, 5, 15, 30, 60, 120, 300, updates):
            with torch.no_grad():
                a = actor(probe)
                slots = PulseActor.slots_from_logits(a, space.n_pulses, space.n_slots)
                on_reg = float((slots == reg).float().mean())
            rows.append({"update": u, "logit_std": float(a.std()), "logit_max": float(a.abs().max()),
                         "frac_on_regular_slot": on_reg})
        if u == updates:
            break
        idx = rng.integers(0, n_trans, size=_PP.BATCH_SIZE)
        s = torch.as_tensor(states[idx]); s2 = torch.as_tensor(states[idx + 1])
        a_b = torch.as_tensor(logits[idx]); r = torch.as_tensor(rewards[idx]); d = torch.as_tensor(dw[idx])
        with torch.no_grad():
            q_t = r + knobs.gamma * (1 - d) * critic_t(s2, actor_t(s2))
        loss_c = F.mse_loss(critic(s, a_b), q_t)
        opt_c.zero_grad(); loss_c.backward(); opt_c.step()
        for p in critic.parameters():
            p.requires_grad_(False)
        a_now = actor(s)
        loss_a = -critic(s, a_now).mean()
        if knobs.logit_l2 > 0:
            loss_a = loss_a + knobs.logit_l2 * a_now.pow(2).mean()
        opt_a.zero_grad(); loss_a.backward()
        if knobs.actor_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), knobs.actor_grad_clip)
        opt_a.step()
        for p in critic.parameters():
            p.requires_grad_(True)
        soft_update(critic_t, critic, knobs.tau)
        soft_update(actor_t, actor, knobs.tau)
    return {"knobs": knobs.as_dict(), "trace": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", action="append", required=True, help="name[:k=v,k=v]")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--transitions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mean-hz", type=float, default=45.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}
    for spec in args.variant:
        name, _, rest = spec.partition(":")
        pairs = [p for p in rest.split(",") if p]
        knobs = _PP.apply_overrides(_PP.PaperKnobs(logit_noise_std=0.3), pairs)
        res = run(knobs, updates=args.updates, n_trans=args.transitions, seed=args.seed, mean_hz=args.mean_hz)
        out[name] = res
        print(name, " ".join(f"u{r['update']}:std={r['logit_std']:.3g}/max={r['logit_max']:.3g}/reg={r['frac_on_regular_slot']:.2f}" for r in res["trace"]), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
