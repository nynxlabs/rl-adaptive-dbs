#!/usr/bin/env python3
"""Plant train + eval of a structured action design (probe-only candidate deviation).

Trains one design from ``controllers.ddpg.action_designs`` (``perpulse`` with the
straight-through critic view, or ``isi``) on the Kumaravelu plant with main's
Mehregan conventions: ``MehreganEnv`` continuous mode (``carry=True,
fixed_network=True``), Fig 3c reward, 10 episodes x 30 steps of 2 s, dt 0.02 ms,
45 Hz, one Alg. 1 update on every 2nd step once the buffer holds a batch, reset
reward collected (Alg. 1 line 7), new initial conditions per episode
(``seed + episode``). The Alg. 1 update is the screen's ``Agent``
(``scripts/probes/mehregan_action_designs.py``).

Deviations from the paper (labelled; candidate only): the action head/decoder
(design), the critic's action input (one-hot of the played slots or the
continuous ISI action instead of Eq. 4's logits), and the exploration noise.
The Fig 3a ``1 x l`` state input holds the current step's $P_\\beta/1000$
(main's ``state_length=1`` state, broadcast to ``l = 2000``).

Outputs in ``--out-dir``: ``series.json`` (Fig 4a series shape:
``beta_norm_trace``, ``episode_rewards``, ``actions``), ``checkpoint.pt``,
``gates.json`` (Fig 4a / 4b), and with ``--eval`` ``eval.json`` (Fig 5a
conditions + PTQ fp16/int8 + random placements + init actor) with Fig 5a gates.

  uv run --no-sync python -m rl_adaptive_dbs.run --max-threads 1 scripts/probes/mehregan_action_design_train.py \\
      --design perpulse --set estimator=st --out-dir artifacts/probes/mehregan_action_designs/plant/pp_st --eval
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from controllers.ddpg.action_designs import HeadActor, SlotAlphabet, make_design, regular_onsets
from controllers.ddpg.quantization import QATActor
from rl_adaptive_dbs.panel import load_script_module

_ROOT = Path(__file__).resolve().parents[2]
_MEH = _ROOT / "scripts/figures/papers/mehregan"
_PP = load_script_module("mehregan_paper_protocol_ad", _MEH / "paper_protocol.py")
_EFF = load_script_module("mehregan_efficacy_panel_ad", _MEH / "efficacy_panel.py")
_SCREEN = load_script_module("mehregan_action_designs_screen", Path(__file__).resolve().parent / "mehregan_action_designs.py")
sys.path.insert(0, str(_ROOT / "scripts/digitization"))
from paper_gates import fig4a_gates, fig4b_gates, fig6_quant_gates  # noqa: E402

STATE_LEN = 2000
MEAN_HZ = 45.0
N_PULSES = int(round(MEAN_HZ * _PP.STEP_S))


def _coerce(v: str):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def expand(obs: np.ndarray) -> np.ndarray:
    return np.full(STATE_LEN, float(np.asarray(obs).reshape(-1)[-1]), dtype=np.float32)


def train(design, knobs, *, episodes: int, steps: int, log) -> tuple[dict, "object"]:
    agent = _SCREEN.Agent(design, knobs)
    env = _PP.make_env(MEAN_HZ, _PP.PaperKnobs())
    slot_alpha = SlotAlphabet(MEAN_HZ, step_s=_PP.STEP_S, dt_ms=_PP.PAPER_DT_MS)
    env.alphabet = slot_alpha  # env.step(0) plays slot_alpha.pending
    env.config = type(env.config)(**{**env.config.__dict__, "max_episode_steps": steps})
    buf: list[tuple] = []
    rng = np.random.default_rng(knobs.seed)
    beta_trace, actions, step_rewards, episode_rewards, reset_beta = [], [], [], [], []
    env_step = n_up = 0
    t0 = time.time()
    try:
        for ep in range(episodes):
            obs, info0 = env.reset(seed=knobs.seed + ep)
            reset_beta.append(float(info0["p_beta_norm"]))
            ep_r = float(info0.get("reward", 0.0))
            s = expand(obs)
            done = False
            while not done:
                with torch.no_grad():
                    raw = agent.actor(torch.as_tensor(s).unsqueeze(0))
                a = design.explore(raw, knobs.sigma, agent.gen)
                slots = design.decode_stored(a)
                slot_alpha.pending = slots
                obs2, r, term, trunc, info = env.step(0)
                done = bool(term or trunc)
                s2 = expand(obs2)
                buf.append((s, a[0], float(r), s2, float(info.get("dw", 0.0))))
                beta_trace.append(float(info["p_beta_norm"]))
                actions.append([int(v) for v in slots])
                step_rewards.append(float(r))
                ep_r += float(r)
                s = s2
                env_step += 1
                if len(buf) >= _PP.BATCH_SIZE and env_step % knobs.update_every == 0:
                    idx = rng.integers(0, len(buf), size=_PP.BATCH_SIZE)
                    agent.update(
                        torch.as_tensor(np.stack([buf[i][0] for i in idx])),
                        torch.stack([buf[i][1] for i in idx]),
                        torch.as_tensor([buf[i][2] for i in idx], dtype=torch.float32),
                        torch.as_tensor(np.stack([buf[i][3] for i in idx])),
                        torch.as_tensor([buf[i][4] for i in idx], dtype=torch.float32),
                    )
                    n_up += 1
            episode_rewards.append(ep_r)
            ep_sl = actions[-steps:]
            with torch.no_grad():
                nf = design.decode_raw(agent.actor(torch.as_tensor(expand(np.array([0.45]))).unsqueeze(0)))
            log(
                f"episode {ep + 1}/{episodes} reward={ep_r:.2f} mean_beta={np.mean(beta_trace[-steps:]):.3f} "
                f"isi_cv={np.mean([_SCREEN.isi_cv(np.array(x)) for x in ep_sl]):.2f} "
                f"nf_vs_regular moved={_SCREEN.compare(nf, regular_onsets(N_PULSES))['moved']:.2f}/"
                f"mad_ms={_SCREEN.compare(nf, regular_onsets(N_PULSES))['mad_ms']:.1f} "
                f"updates={n_up} t={time.time() - t0:.0f}s"
            )
    finally:
        env.close()
    series = {
        "beta_norm_trace": beta_trace,
        "episode_rewards": episode_rewards,
        "actions": actions,
        "step_rewards": step_rewards,
        "reset_beta": reset_beta,
        "updates": n_up,
        "elapsed_s": time.time() - t0,
    }
    return series, agent


def core_actor(module: nn.Module) -> nn.Module:
    return module.actor if isinstance(module, QATActor) else module


def make_policy(module: nn.Module, design, *, dtype=torch.float32):
    module.eval()

    def act(state: np.ndarray):
        with torch.no_grad():
            raw = module(torch.as_tensor(expand(state)).unsqueeze(0).to(dtype)).float()
        return [int(v) for v in design.decode_raw(raw)], []

    return act


def deployed(module: nn.Module) -> nn.Module:
    m = copy.deepcopy(module)
    if isinstance(m, QATActor):
        m.apply(torch.ao.quantization.disable_observer)
    return m.eval()


def run_eval(agent, design, *, seed: int, log) -> dict:
    alpha = SlotAlphabet(MEAN_HZ, step_s=_PP.STEP_S, dt_ms=_PP.PAPER_DT_MS)
    menu = _PP.make_alphabet(MEAN_HZ, _PP.PaperKnobs())
    actor = deployed(agent.actor)
    rng = np.random.default_rng(1234)

    def random_policy(_state):
        return [int(v) for v in np.sort(rng.choice(2000, N_PULSES, replace=False))], []

    torch.manual_seed(agent.k.seed)
    init = HeadActor(state_len=STATE_LEN, raw_dim=design.raw_dim)
    design.init_actor(init)
    conds = {
        "no_stim": _PP.run_condition(seed=seed, alphabet=alpha),
        "periodic": _PP.run_condition(seed=seed, alphabet=alpha, fixed_spec=_PP.regular_spec(menu)),
        "trained": _PP.run_condition(seed=seed, alphabet=alpha, policy=make_policy(actor, design)),
        "cdbs_130": _PP.run_condition(seed=seed, alphabet=alpha, fixed_spec=_PP.DbsSpec.from_frequency_hz(130.0)),
        "init_actor": _PP.run_condition(seed=seed, alphabet=alpha, policy=make_policy(init, design)),
        "random_placement": _PP.run_condition(seed=seed, alphabet=alpha, policy=random_policy),
    }
    if not isinstance(agent.actor, QATActor):
        core = copy.deepcopy(core_actor(agent.actor)).eval()
        conds["ptq_fp16"] = _PP.run_condition(
            seed=seed, alphabet=alpha, policy=make_policy(copy.deepcopy(core).half(), design, dtype=torch.float16)
        )
        conds["ptq_int8"] = _PP.run_condition(
            seed=seed,
            alphabet=alpha,
            policy=make_policy(
                torch.ao.quantization.quantize_dynamic(copy.deepcopy(core), {nn.Linear}, dtype=torch.qint8), design
            ),
        )
    for k, c in conds.items():
        log(f"eval {k}: pre={_PP.pre_mean(c):.1f} late={_PP.post_mean(c, lo=_EFF.LATE_LO_S):.1f}")
    payload = {"conditions": conds, "seed": seed}
    eff_conds = {k: conds[k] for k in ("no_stim", "periodic", "trained", "cdbs_130")}
    means = _EFF.panel_means({"conditions": eff_conds})
    traces = _EFF.payload_traces({"conditions": eff_conds})
    payload["fig5a"] = _jsonable(_EFF.efficacy_gates(_EFF.FIG5A, means, traces))
    payload["means"] = {k: _PP.post_mean(c, lo=_EFF.LATE_LO_S) for k, c in conds.items()}
    payload["pre"] = {k: _PP.pre_mean(c) for k, c in conds.items()}
    tr = [np.array(a) for a in conds["trained"]["actions"]]
    cmp = {}
    for k in ("init_actor", "random_placement", "ptq_fp16", "ptq_int8"):
        if k not in conds:
            continue
        other = [np.array(a) for a in conds[k]["actions"]]
        cmp[k] = {
            "moved_vs_trained_per_step": [_SCREEN.compare(o, t)["moved"] for o, t in zip(other, tr, strict=True)],
            "mad_ms_per_step": [_SCREEN.compare(o, t)["mad_ms"] for o, t in zip(other, tr, strict=True)],
        }
        if k.startswith("ptq"):
            y0, y1 = np.asarray(conds["trained"]["trace"]), np.asarray(conds[k]["trace"])
            cmp[k]["trace_max_abs_diff"] = float(np.max(np.abs(y0 - y1)))
            cmp[k]["late_mean_diff"] = payload["means"][k] - payload["means"]["trained"]
    payload["action_compare"] = cmp
    payload["trained_isi_cv"] = [_SCREEN.isi_cv(t) for t in tr]
    payload["trained_n_distinct"] = [int(len(np.unique(t))) for t in tr]
    if "ptq_int8" in conds:
        q = {"fp32": conds["trained"], "ptq_int8": conds["ptq_int8"], "ptq_fp16": conds["ptq_fp16"]}
        try:
            g = fig6_quant_gates(
                {k: (np.asarray(v["time_s"]), np.asarray(v["trace"])) for k, v in q.items()},
                qat_seeds=None,
                panel="6a",
                closed_loop={k: True for k in q},
            )
            payload["fig6a_ptq_only"] = _jsonable(g.get("gates", g))
        except Exception as exc:  # gate needs QAT seeds for some keys
            payload["fig6a_ptq_only"] = {"error": repr(exc)}
    return payload


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True, choices=("perpulse", "isi"))
    ap.add_argument("--set", action="append", default=[], help="design or run knob k=v")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="fp32", choices=("fp32", "qat"))
    ap.add_argument("--episodes", type=int, default=_PP.NUM_EPISODES)
    ap.add_argument("--steps", type=int, default=_PP.STEPS_PER_EPISODE)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    run_fields = {f for f in _SCREEN.RunKnobs.__dataclass_fields__}
    knobs = _SCREEN.RunKnobs(seed=args.seed, variant=args.variant, update_every=2,
                             sigma=1.0 if args.design == "perpulse" else 0.3)
    dkw = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k in run_fields:
            cur = getattr(knobs, k)
            knobs = _SCREEN.replace(knobs, **{k: type(cur)(v)})
        else:
            dkw[k] = _coerce(v)
    design = make_design(args.design, N_PULSES, **dkw)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    meta = {"design": args.design, "design_kwargs": dkw, "knobs": knobs.__dict__, "mean_hz": MEAN_HZ,
            "state": "current-step P_beta/1000 broadcast to l=2000", "deviation": "candidate (probe-only)"}
    log(f"meta {json.dumps(meta)} torch_threads={torch.get_num_threads()}")
    series, agent = train(design, knobs, episodes=args.episodes, steps=args.steps, log=log)
    series.update(meta)
    (out / "series.json").write_text(json.dumps(series) + "\n")
    torch.save({"meta": meta, "actor_state_dict": agent.actor.state_dict(),
                "qat": isinstance(agent.actor, QATActor)}, out / "checkpoint.pt")
    gates = {}
    if args.episodes == _PP.NUM_EPISODES and args.steps == _PP.STEPS_PER_EPISODE:
        y = np.asarray(series["beta_norm_trace"]).reshape(args.episodes, args.steps)
        gates["fig4a"] = _jsonable(fig4a_gates(series["beta_norm_trace"], n_expected=300))
        gates["fig4b"] = _jsonable(fig4b_gates(series["episode_rewards"], y.mean(axis=1).tolist(), n_expected=10))
        for k in ("fig4a", "fig4b"):
            g = gates[k]["gates"]
            log(f"{k} pass={all(g.values())} fails={[n for n, ok in g.items() if not ok]}")
    (out / "gates.json").write_text(json.dumps(gates, indent=1) + "\n")
    if args.eval:
        payload = run_eval(agent, design, seed=args.eval_seed, log=log)
        g = {k: v for k, v in payload["fig5a"].items() if isinstance(v, bool)}
        log(f"fig5a pass={payload['fig5a'].get('pass')} fails={[n for n, ok in g.items() if not ok and n != 'pass']}")
        log(f"action_compare { {k: (round(float(np.mean(v['moved_vs_trained_per_step'])), 3), v.get('late_mean_diff'), v.get('trace_max_abs_diff')) for k, v in payload['action_compare'].items()} }")
        log(f"fig6a_ptq_only {payload.get('fig6a_ptq_only')}")
        (out / "eval.json").write_text(json.dumps(_jsonable(payload)) + "\n")
    log("# done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
