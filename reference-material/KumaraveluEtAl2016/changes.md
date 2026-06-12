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

**Where:** fifth argument to `simulate_network_model(IT, pd, corstim, pick_dbs_freq, dynamics_only)`.

**Behavior:** when `dynamics_only` is true, return immediately after `CTX_BG_TH_network`, skipping:

- `make_Spectrum` / inlined `mtspectrumpt` (needs **Signal Processing Toolbox** for `dpss`)
- `.mat` save and `quit`

**Why:** Phase 2 calls network dynamics only; Mehregan $P_\beta$ (13–35 Hz) is computed in Python ([`docs/plant.md`](../../docs/plant.md)). `bash scripts/matlab/verify.sh` uses `simulate_network_model(1,1,0,1,true)`.

**Default:** `dynamics_only = false` — preserves upstream script shape when the fifth argument is omitted.

---

## 3. Not changed

- Network equations, integration (`dt = 0.01` ms), DBS waveform construction, inlined multitaper code path (still runs when `dynamics_only` is false).
- Full upstream run still needs Statistics + Signal Processing toolboxes unless further patches are added.
