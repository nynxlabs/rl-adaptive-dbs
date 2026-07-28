# Alphabet diversity and within-step extensions

**Not the default paper path.** Parked code and probes for Fig **6a** honest PTQ diversity, near-hub burst alphabets, and within-step L=16 ablations. Default Mehregan training/eval on `main` uses scalar `state_length=1` and disconnected plant steps via `envs.mehregan.env.MehreganEnv`.

---

## Layout

| Path | Role |
|------|------|
| `src/envs/mehregan/extensions/alphabet_diversity/` | `WithinStepMehreganEnv`, `WithinStepEnvConfig`, continuous plant stitching, near-hub alphabet |
| `scripts/probes/alphabet_diversity/` | Open-loop sweeps, within-step train, plant continuity, near-hub pipeline |
| `tests/envs/extensions/alphabet_diversity/` | Unit tests for extension env |
| `artifacts/ddpg/` | Probe JSON outputs (e.g. `alphabet_diversity_sweep.json`, `within_step_L16_burst_train.json`) |

Import pattern for probes:

```python
from envs.mehregan.extensions.alphabet_diversity import (
    NearHubBurstAlphabet,
    WithinStepEnvConfig,
    WithinStepMehreganEnv,
)
```

---

## Problem statement (Fig 6a)

Honest PTQ/QAT panels need **more than one greedy action** and PTQ noise-0 argmax flips. Open-loop landscape sweeps and larger burst menus were insufficient: policies still locked to a single action with huge logit margins.

---

## Within-step L=16 burst ablation (2026-07-27)

Configuration: burst `skip_regular` @ 45 Hz, `dt_ms=0.02`, `state_mode=within_step`, `state_length=16`, `reward_state_mode=full_segment`, 10 episodes.

| `plant_integration_mode` | Greedy action | Mean logit margin | Artifact |
|--------------------------|---------------|-------------------|----------|
| `disconnected` | **6** only | ≈681 | `artifacts/ddpg/within_step_L16_burst_train.json` |
| `continuous` | **0** only | ≈363 | `artifacts/ddpg/within_step_L16_burst_train_continuous.json` |

Continuous integration fixes train/eval plant mismatch (`plant_continuity_probe.json`) and **lowers** margins, but does **not** yield multi-action greedy rollouts or PTQ argmax flips at noise 0. Fig 6a honest PTQ still relies on accepted panel extensions (v9-style stylization) or a different lever — not more alphabet sweeps alone.

**Train probe:**

```bash
tmux new-session -d -s within-step-l16-train \
 "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \
   scripts/probes/alphabet_diversity/run_within_step_L16_burst_train.py \
   >> logs/within-step-l16-train.log 2>&1 < /dev/null"
```

Optional continuous mode: add `--plant-integration continuous`. Chain continuity check:

```bash
bash scripts/probes/alphabet_diversity/run_after_within_step_pipeline.sh
```

---

## Open-loop diversity sweeps

`run_alphabet_diversity_sweep.py` — 1-step $P_\beta$ landscapes at 30 Hz and 45 Hz across TASK-177 constructions (burst, jitter, random, alternating) plus **near-hub** menus (`NearHubBurstAlphabet`).

```bash
tmux new-session -d -s alphabet-diversity \
 "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \
   scripts/probes/alphabet_diversity/run_alphabet_diversity_sweep.py \
   >> logs/alphabet-diversity.log 2>&1 < /dev/null"
```

Near-hub gated pipeline (skip Fig 6a train if diversity gates fail):

```bash
bash scripts/probes/alphabet_diversity/run_near_hub_pipeline.sh
```

`plot_large_alphabet_landscape.py` — side-by-side ranking plot from `alphabet_diversity_large_n.json` (diagnostic PNG under `figures/mehregan/images/6a/`).

---

## Near-hub alphabet

`NearHubBurstAlphabet` builds **local phase/gap perturbations** of known strong burst hub indices (default 8 hubs × 32 variants → 257 patterns; `skip_regular` → 256 agent actions). Purpose: dense menu near open-loop optima without a wider random pattern space. **Probe-only** — not wired into `make_alphabet` or panel ship scripts.

---

## Extension env API (summary)

`WithinStepEnvConfig` adds fields on top of the paper defaults:

- `state_mode`: `scalar` | `within_step` | `multi_step_history`
- `reward_state_mode`: `observation_mean` | `full_segment`
- `plant_integration_mode`: `disconnected` | `continuous` (Python plant only)
- `pre_stim_duration_s`, `plant_dt_ms` — continuous / biomarker alignment

See module docstrings in `src/envs/mehregan/extensions/alphabet_diversity/` for validation rules.

---

## Status

**Parked / failed for Fig 6a honest diversity.** Keep this tree for reproducibility and future levers; do not treat extension defaults as paper replication without explicit promotion into `environment.md` and panel scripts.

---

## See also

- [figures/mehregan/replications.md](../../../figures/mehregan/replications.md) § Fig 6a — shipped panel status
- [environment.md](../../environment.md) — default Mehregan API
- [controllers/ddpg/extensions.md](../../controllers/ddpg/extensions.md) — quantization deeper dives
