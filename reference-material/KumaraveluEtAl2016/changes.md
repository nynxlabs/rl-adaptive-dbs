# Vendor changes (Kumaravelu et al., 2016 bundle)

Upstream: [ModelDB 206232](https://github.com/ModelDBRepository/206232). Citation and provenance: [`readme.txt`](readme.txt).

**rl-adaptive-dbs** keeps upstream dynamics; these edits support **base MATLAB** and the Phase 2 plant bridge. Do not merge back to ModelDB without review.

---

## 1. `randsample` → `randperm` (`CTX_BG_TH_network`)

**Where:** wiring shuffle for cortical connectivity (`all`, `bll`, … `oll`).

**Upstream:** `randsample(n,n)` — requires **Statistics and Machine Learning Toolbox**.

**Here:** `randperm(n)` — base MATLAB. For fixed `n`, a uniform random permutation of `1:n` (same role as sampling all indices without replacement).

---

## 2. Optional `dynamics_only` (`simulate_network_model`)

**Where:** fifth argument to `simulate_network_model(IT, pd, corstim, pick_dbs_freq, dynamics_only, seed, tmax_ms)`.

**Behavior:** when `dynamics_only` is true, return immediately after `CTX_BG_TH_network`, skipping:

- `make_Spectrum` / inlined `mtspectrumpt` (needs **Signal Processing Toolbox** for `dpss`)
- `.mat` save and `quit`

**Optional args (plant bridge):**

- **`seed`** — when provided, `rng(seed)` for reproducible ICs; otherwise `rng('shuffle')` (upstream default).
- **`tmax_ms`** — segment duration in ms; default **2000**.

**Return values** when `dynamics_only` is true (for `matlab.engine`):

| `nargout` | Value |
|-----------|--------|
| 1 | `gpi_spike_times` — 1×10 cell of GPi spike time vectors (seconds) |
| 2–6 | `dt_ms`, `tmax_ms`, `pd`, `pick_dbs_freq`, `dbs_freq_hz` |

`spikes_to_cell` packs `find_spike_times` output because nested struct arrays cannot cross the Python engine.

**Why:** Phase 2 calls network dynamics only; Mehregan $P_\beta$ (13–35 Hz) is computed in Python ([`docs/plant.md`](../../docs/plant.md)). `bash scripts/matlab/verify.sh` uses `simulate_network_model(1,1,0,1,true)` (no return values).

**Default:** `dynamics_only = false` — preserves upstream script shape when the fifth argument is omitted.

---

## 3. Not changed

- Network equations, integration (`dt = 0.01` ms), DBS waveform construction, inlined multitaper code path (still runs when `dynamics_only` is false).
- Full upstream run still needs Statistics + Signal Processing toolboxes unless further patches are added.

---

## 4. `plant_band_power.m` (Python $P_\beta$ validation)

**Where:** [`plant_band_power.m`](plant_band_power.m) — exported `plant_band_power` / `plant_p_beta` plus inlined Chronux `mtspectrumpt` helpers (copied from `simulate_network_model.m`).

**Why:** Local functions in `simulate_network_model.m` are not callable from `matlab.engine`; this file validates Python `envs.plant.biomarkers.p_beta` in `@pytest.mark.matlab` tests.

**Requires:** **Signal Processing Toolbox** (`dpss`). Skipped when unavailable; Phase 2 biomarkers for training use **Python** by default.
