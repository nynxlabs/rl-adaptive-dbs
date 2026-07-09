# TASK-172 — Fig 5b failure: paper vs implementation gap analysis

**Date:** 2026-07-09  
**Context:** TASK-166 (greedy+logits, L=16, decoupled reward) and TASK-170 (softmax τ=1, one_hot critic) both pass Fig 4a/4b but fail Fig 5b at 30 Hz. Trained Pβ ≈ 542–545 > no-stim ≈ 503.

This document is the TASK-172 deliverable: a systematic paper-spec vs implementation comparison, with impact assessment and minimum paper-aligned changes.

---

## Executive summary

Fig 5b fails for **two stacked reasons**:

1. **Training finds a suboptimal collapsed constant policy** (actions 8 or 30) that is worse than no-stim under the eval protocol — even though reward improves during training (Fig 4 pass).

2. **At paper-aligned `plant.dt_ms=0.02`, no pattern in our 41-pattern alphabet beats no-stim on open-loop single-step Pβ** (best action 19 → Pβ ≈ 520 > no-stim ≈ 503). The earlier claim “2/41 patterns beat no-stim (29, 38)” came from a **dt=0.01** landscape (`pattern_reward_landscape_30hz.json`), not the paper grid. This means Fig 5b may be **structurally unreachable** with any *constant* policy under the current plant + alphabet at 30 Hz — unless Mehregan's trained policy uses **state-dependent** pattern selection across steps (as the paper describes) or their pattern construction / plant differs from ours.

**Minimum paper-aligned next steps** (no entropy bonus, logit noise, or reward shaping):

| Priority | Change | Paper basis |
|----------|--------|-------------|
| P0 | Fix actor init: bias logits toward pattern 0 **without zeroing CNN weights** | §IV.A.1 “initialize … using regular pulses”; §III.B init — **done (TASK-173)** |
| P0 | Re-run 30 Hz with `exploration_mode=greedy`, `critic_action_input=logits` (TASK-166 stack) after init fix | Algorithm 1 tuple uses `a_logit`; no exploration specified |
| P1 | A/B `state_length=1` vs `L=16` under same stack | §IV.A.1 “window size = length of simulation” (ambiguous: scalar vs sub-windows) |
| P1 | Document pattern alphabet as **implementation hypothesis**; flag plant/alphabet parity as blocker if open-loop oracle still loses to no-stim | Paper silent on alphabet construction |
| P2 | Confirm γ, τ only after P0–P1 (paper silent) | Algorithm 1 lines 1–2 |

---

## Evidence from failed runs

| Run | Exploration | Critic input | Dominant action | Trained Pβ | No-stim Pβ | Fig 5b |
|-----|-------------|--------------|-----------------|------------|------------|--------|
| TASK-166 | greedy | logits | 8 | 542.1 | 503.4 | FAIL |
| TASK-170 | softmax τ=1 | one_hot | 30 | 544.7 | 503.4 | FAIL |

Open-loop (TASK-169): action 8 → Pβ 568.6 (rank 30/41); action 8 does **not** beat no-stim. Constant action-8 eval stim-only mean 549.9 > no-stim — failure is real policy quality, not eval bleed alone.

**30 Hz landscape (`q6_landscape_1step_30hz.json`, dt=0.02, seed 0):**

| Metric | Value |
|--------|-------|
| No-stim Pβ | 503.4 |
| Best 1-step Pβ (action 19) | 520.3 |
| Patterns with Pβ < no-stim | **0 / 41** |
| Pattern 0 (regular 30 Hz) rank | 41 / 41 (worst) |

At **45 Hz** with the same grid, **40/41** patterns beat no-stim and pattern 0 is best — consistent with paper Fig 5a.

---

## Comparison table: paper spec vs our implementation

| # | Topic | Paper (Mehregan et al.) | Our implementation | Match? | Could explain Fig 5b? |
|---|-------|-------------------------|-------------------|--------|----------------------|
| 1 | **Algorithm 1 — action selection** | Line 10: `u ← μ(s\|θ_μ)`; §III.B: softmax + argmax for discrete pattern | Greedy argmax (TASK-166) or softmax sample (TASK-170) | Partial | **Medium** — TASK-170 softmax is non-paper; greedy still collapses |
| 2 | **Algorithm 1 — replay tuple** | Store `(s', a, a_logit, R, s, dw)` | Same layout in `ReplayBuffer` | Yes | No |
| 3 | **Critic loss Eq. (4)** | `MSE(Q_target, Q(s, a_logit))` — **stored logits** | `logits` mode (TASK-166) or `one_hot` (TASK-170) | Partial | **Low–medium** — one_hot diverges from paper diagram |
| 4 | **Actor loss Eq. (5)** | `(1/\|B̃\|) Σ Q(s(t), a_logit(t))` — batch logits | Fresh `μ(s)` forward pass on batch states | Ambiguous | **Unclear** — stored detached logits would block actor gradients; fresh pass is standard DDPG |
| 5 | **Critic freeze during actor step** | Lines 21–24: freeze → actor Adam → unfreeze | Implemented in `DDPGTrainer._update_step` | Yes | No |
| 6 | **Target soft update Eqs. (6)–(7)** | Polyak on critic and actor targets | `soft_update` after both optimizers | Yes | No |
| 7 | **Replay buffer** | “adopted to ensure reliable training convergence”; §IV.A.1: capacity **8192**, batch **32** | `buffer_capacity=8192`, `batch_size=32` | Yes | No |
| 8 | **Learning rates** | Actor 5×10⁻⁴, critic 10⁻³ | `DDPGConfig` defaults | Yes | No |
| 9 | **γ, τ, update frequency** | Initialized in Alg. 1; **not numerically specified** in §IV.A.1 | γ=0.99, τ=0.005, `update_frequency=1` | Open | Unknown — paper silent |
| 10 | **Episode structure** | 10 episodes × 30 steps × 2 s; reset with new ICs each episode | `num_episodes=10`, `max_episode_steps=30`; `reset(seed+episode)` | Yes | No |
| 11 | **Reward Eq. (8)** | s_sum = mean of obs; β_t=0.35; scale ×10 | `mehregan_reward`; linear branch negated (TASK-78) | Yes | No (decoupling fix helped Fig 4) |
| 12 | **Reward s_sum source** | §IV.A.1: biomarker window = **full simulation segment** | `reward_state_mode=full_segment` (TASK-162) | Yes | No — fixed prior collapse |
| 13 | **State / CNN input** | Temporal biomarker window = segment length; “shrink dimension” in Brain class | `within_step`, `state_length=16` (16 sub-window Pβ samples) | Hypothesis | **Medium** — L=1 (single scalar per 2 s) is alternate paper reading |
| 14 | **Action space** | Discrete stimulation **patterns** at fixed mean rate f (Alg. 1 input) | `FixedMeanPatternAlphabet`, 41 jittered patterns | Hypothesis | **High** — paper silent; alphabet may not contain Mehregan-effective patterns |
| 15 | **Network init** | “action space is initialized using **regular pulses**” | `init_toward_action`: head bias[0]=2.0; weights preserved (TASK-173) | **Partial** | **Medium** — retrain needed to validate Fig 4/5b |
| 16 | **Plant dt** | §IV.A.1: **0.02 ms** | Paper runs use 0.02; repo default 0.01 | Partial | **High** — at 0.02, **zero** patterns beat no-stim open-loop at 30 Hz |
| 17 | **45 Hz vs 30 Hz** | Separate training runs; other settings unchanged | Same — no pre-train / fine-tune chain | Yes | N/A — 45 Hz landscape is easy; 30 Hz is hard |
| 18 | **QAT vs full precision** | Fig 5 = full-precision; QAT is §IV.A.3 separate experiment | TASK-166/170 are full-precision | Yes | No |
| 19 | **Eval protocol Fig 5** | Fixed seed; 10 s sim; 2 s reset + 5 stim applications | `mehregan_eval`: 2 s reset + 5×2 s steps; mean Pβ | Close | **Low** — reset dilutes mean but stim-only Pβ still > no-stim |
| 20 | **Online exploration** | Not specified; deploy = greedy | ε-greedy / softmax / logit noise available | Extension | TASK-170 softmax is explicitly non-paper |

---

## What the paper does that we are not doing (actionable)

### 1. Actor initialization (highest-impact paper gap)

**Paper:** “a simple but efficient initialization approach, where the action space is initialized using regular pulses” (§I, §IV.A.1 init at 45/30 Hz mean).

**Us (pre–TASK-173):** `Actor.init_toward_action` zeroed **head weights** and set `bias[pattern_0] = 2.0`, decoupling logits from CNN features and causing instant argmax collapse (TASK-169).

**Fix (TASK-173):** Keep PyTorch default encoder and head weights; zero head biases and set `bias[pattern_0] = init_bias_scale` (default 2.0). Retrain under experimenter gate to validate Fig 4/5b.

### 2. Greedy training + critic on stored logits

**Paper:** No ε-greedy, no softmax temperature, no logit noise. Critic trained on `a_logit` from replay (Eq. 4).

**Us:** TASK-166 already uses `greedy` + `logits` — closest to paper. TASK-170’s softmax + `one_hot` critic is a deliberate non-paper experiment.

**Paper-aligned fix:** Standardize on TASK-166 config after init fix; do not add exploration knobs.

### 3. State length interpretation

**Paper:** “window size was set to the length of the simulation” (§IV.A.1) — for a 2 s step, this can mean one Pβ scalar over 2 s **or** a temporal decomposition of that segment.

**Us:** Default `state_length=16` within-step sub-windows (TASK-158 interpretation).

**Paper-aligned fix:** Run paired retrains at `state_length=1` (scalar window) and `L=16`; compare Fig 5b. Paper does not pick between them.

### 4. Pattern alphabet (paper silent — structural risk)

**Paper:** Discrete patterns at fixed mean rate; irregular trains beat periodic at 30 Hz (Fig 5b prose). **No** alphabet size, jitter model, or encoding.

**Us:** 41 patterns, ±1/3 ISI jitter, deterministic PRNG — documented hypothesis ([environment.md](../environment.md) §4.2).

**Impact:** At dt=0.02, **no** open-loop pattern beats no-stim. Mehregan's Fig 5b implies either (a) state-dependent policies across eval steps, (b) different pattern construction, or (c) plant/biomarker differences. This cannot be closed by training hyperparameters alone if the oracle constant-policy bound fails.

---

## What we ruled out

| Hypothesis | Verdict |
|------------|---------|
| Reward sign bug | Fixed TASK-78; Fig 4 now passes |
| Reward coupled to obs mean (TASK-159) | Fixed TASK-162 decoupling; Fig 4 passes |
| Eval reset bleed inflating mean Pβ | TASK-169: stim-only mean still > no-stim |
| Need non-paper exploration (softmax) | TASK-170 still fails Fig 5b; collapses to action 30 |
| 45 Hz pre-train required | Paper §IV.A.2: 30 Hz is a **separate** run, same hyperparameters |
| Wrong experiment (QAT) | TASK-166/170 are full-precision |

---

## Recommended child work (paper-bounded)

1. ~~**Programmer:** Implement `init_toward_action` v2 — preserve encoder weights, bias-only toward pattern 0.~~ **Done (TASK-173).**
2. **Experimenter:** 10-ep 30 Hz retrain (greedy, logits, full_segment, dt=0.02) with init v2; gate Fig 4/5b. Tmux: `task173-train-30hz-initv2`.
3. **Experimenter:** Paired `state_length=1` vs `16` ablation (same stack).
4. **Reviewer:** If Fig 5b still fails after P0–P1, escalate **pattern alphabet / plant parity** to COO — paper-silent but blocks replication claim.

---

## References

- Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning* — paper markdown notes.
- Artifacts: `task166_*_final.json`, `task170_*_final.json`, `task169_fig5b_postmortem.json`, `q6_landscape_1step_30hz.json`.
- Specs: [environment.md](../environment.md), [controllers/ddpg/replication.md](../controllers/ddpg/replication.md).
