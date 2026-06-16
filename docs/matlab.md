# MATLAB setup (Kumaravelu plant bridge)

MATLAB is required for **Phase 2** plant work: running the bundled Kumaravelu et al. (2016) network (`reference-material/KumaraveluEtAl2016/`) and validating the Python bridge. CI and day-to-day Python-only work do **not** need MATLAB ([development/testing.md](development/testing.md)).

**Platforms:** **Windows, macOS, and Linux** (including WSL) — same repo contract everywhere ([AGENTS.md](../AGENTS.md)). Shell helpers (`scripts/matlab/env.sh`, `scripts/matlab/verify.sh`) are **bash**; on Windows use **Git Bash**, **WSL**, or the manual steps in §2.3 / §5.

**Quick start (interactive):** from the repo root,

```bash
bash scripts/setup.sh              # Python + optional MATLAB
# or MATLAB only:
bash scripts/matlab/setup.sh
```

Walks through install (optional), `MATLAB_ROOT`, license, `uv sync`, shell persistence, and `scripts/matlab/verify.sh`. Manual steps below if you prefer.

**Phase 4:** harden `scripts/matlab/` on Linux, macOS, and Windows (Git Bash / WSL) and keep prompts aligned with this doc. Top-level **`scripts/setup.sh`** runs Python setup and can delegate here — [setup.md](setup.md), [development/roadmap.md](development/roadmap.md).

| You have… | Start at |
|-----------|----------|
| MATLAB already installed and licensed | **§2 Connect** — or run `bash scripts/matlab/setup.sh` |
| No MATLAB on this machine | **§3 Fresh install** — or run `bash scripts/matlab/setup.sh` |

---

## 1. What this repo needs from MATLAB

| Piece | Purpose |
|-------|---------|
| **MATLAB** (R2024b+; repo tested on **R2025b**) | Run `simulate_network_model.m` and the plant bridge |
| **Valid license** | Batch mode and `matlab.engine` both require it |
| **`matlabengine` (Python)** | `import matlab.engine` from the project venv — version must match your MATLAB release |
| **Toolboxes** | **Base MATLAB** for Phase 2 bridge (`CTX_BG_TH_network`; vendored patches in [kumaravelu_vendor_patches.md](reference-material/kumaravelu_vendor_patches.md)). Full upstream `simulate_network_model` also needs **Signal Processing Toolbox** (`dpss` in `make_Spectrum`) — biomarkers for Mehregan are computed in Python per [plant.md](plant.md) |
| **OS extras** | Headless **Linux/WSL** only: `xvfb` + GTK libs (§3.1) |

Default `MATLAB_ROOT` assumed by scripts: **`~/MATLAB`** (Linux/macOS) or your MathWorks install path (§2). Override when your install lives elsewhere.

---

## 2. Connect an existing install

Skip §3 if MATLAB is already on this machine. On every OS the flow is the same:

1. Set **`MATLAB_ROOT`** and **`MATLAB_RELEASE`** to your install.
2. `uv sync --all-groups` (or `--group matlab`).
3. Source **`scripts/matlab/env.sh`** (bash) or set the equivalent env vars (§5).
4. Run **`bash scripts/matlab/verify.sh`** — all checks should pass.

### Universal checklist (bash)

```bash
cd rl-adaptive-dbs

export MATLAB_ROOT=/path/to/MATLAB    # see §2.1–§2.3
export MATLAB_RELEASE=R2025b          # must match your install

uv sync --all-groups
source scripts/matlab/env.sh          # Git Bash / macOS / Linux / WSL

bash scripts/matlab/verify.sh
```

Shared topics (all OS): **license file copy** (§2.4), **`matlabengine` version** (§2.5), **persist settings** (§2.6).

### 2.1 Linux and WSL

**Typical `MATLAB_ROOT`:**

| Layout | Example |
|--------|---------|
| User install (MPM) | `~/MATLAB` |
| System install | `/usr/local/MATLAB/R2025b` |

**Find the root** (directory that contains `bin/matlab`):

```bash
dirname "$(dirname "$(readlink -f "$(which matlab)")")"
```

**License file from Windows → WSL** (common):

```bash
mkdir -p ~/MATLAB/licenses
cp "/mnt/c/Users/<you>/Downloads/license.lic" ~/MATLAB/licenses/
```

**Manual activation (License Center):** OS = **Linux**; **Host ID** = `cat /sys/class/net/eth0/address`; **login name** = `whoami`.

**Headless WSL:** `matlab-env.sh` wraps `xvfb-run` when `DISPLAY` is unset. Install §3.1 packages if batch mode fails.

### 2.2 macOS

**Typical `MATLAB_ROOT`:**

| Layout | Example |
|--------|---------|
| Default user install | `/Applications/MATLAB_R2025b.app` |
| Alternate | `~/MATLAB` |

The scripts expect the **app bundle root** (the folder that contains `bin/matlab`), not `Contents/MacOS`.

**Find the root:**

```bash
# If matlab is on PATH:
dirname "$(dirname "$(which matlab)")"

# Or list installs:
ls /Applications/MATLAB*.app
```

**Activate:** open MATLAB from **Applications** and sign in, or:

```bash
/Applications/MATLAB_R2025b.app/bin/matlab
```

(Product Authorizer ships under `bin/maca64/` or `bin/maci64/` if you need the standalone activation app.)

License files usually land under `$MATLAB_ROOT/licenses/` or `~/.matlab/<release>_licenses/`.

**Shell:** use **Terminal** or **iTerm** (bash/zsh). After verify, add §2.6 to `~/.zshrc` or `~/.bash_profile`.

**Note:** `matlab-env.sh` sets `DYLD_LIBRARY_PATH` for the engine arch folder (`maca64` / `maci64`). If macOS blocks it, ensure MATLAB’s `bin` is on `PATH` and run verify from the same shell where you sourced `matlab-env.sh`.

### 2.3 Windows

**Typical `MATLAB_ROOT`:**

| Layout | Example |
|--------|---------|
| Per-user | `C:\Users\<you>\AppData\Local\Programs\MATLAB\R2025b` |
| System | `C:\Program Files\MATLAB\R2025b` |

**Option A — Git Bash** (recommended for repo scripts):

```bash
export MATLAB_ROOT="/c/Program Files/MATLAB/R2025b"
export MATLAB_RELEASE=R2025b
source scripts/matlab/env.sh
bash scripts/matlab/verify.sh
```

Use forward slashes and the `/c/...` drive prefix in Git Bash.

**Option B — WSL:** treat as Linux (§2.1). MATLAB may live in WSL (`~/MATLAB`) or license may be copied from Windows Downloads.

**Option C — PowerShell** (no bash): set user environment variables, then run verify steps manually:

```powershell
$env:MATLAB_ROOT = "C:\Program Files\MATLAB\R2025b"
$env:MATLAB_RELEASE = "R2025b"
$env:PATH = "$env:MATLAB_ROOT\bin;$env:PATH"
uv sync --all-groups
& "$env:MATLAB_ROOT\bin\matlab.exe" -batch "license('test','MATLAB'); exit"
uv run python -c "import matlab.engine; eng = matlab.engine.start_matlab(); eng.exit()"
```

**Activate:** run MATLAB from the Start menu once, or `matlab -batch "exit"` after install. License file → `%MATLAB_ROOT%\licenses\`.

**Manual activation (License Center):** OS = **Windows**; Host ID from MATLAB **Help → About** or `!hostname` in MATLAB.

### 2.4 License already working elsewhere

**Same machine** — if `matlab -batch "license('test','MATLAB'); exit"` already succeeds outside this repo, set `MATLAB_ROOT` and source `matlab-env.sh`; no extra license step.

**License file** — copy `.lic` / `license.dat` into `$MATLAB_ROOT/licenses/` (see §2.1 for WSL copy path).

**Not yet activated** — §4.

**Campus / network license** — set `MLM_LICENSE_FILE=port@host` (from IT). `matlab-env.sh` leaves an explicit, valid value alone.

### 2.5 `matlabengine` version must match MATLAB

Repo default: **`matlabengine==25.2.2`** for **R2025b**. Other releases: pick the matching version from [PyPI `matlabengine`](https://pypi.org/project/matlabengine/) (major.minor aligns with MATLAB, e.g. R2024b → `24.2.x`):

```bash
uv add --group matlab "matlabengine==<version-for-your-release>"
uv sync --group matlab
```

### 2.6 Persist settings (optional)

**bash** (`~/.bashrc`, Git Bash `~/.bash_profile`):

```bash
export MATLAB_ROOT=/path/to/MATLAB
export MATLAB_RELEASE=R2025b
source /path/to/rl-adaptive-dbs/scripts/matlab/env.sh
```

**zsh** (`~/.zshrc`): same as above.

**Windows (system env):** set `MATLAB_ROOT`, `MATLAB_RELEASE`, and prepend `%MATLAB_ROOT%\bin` to `PATH` in Settings → Environment Variables.

---

## 3. Fresh install

Install MATLAB from [MathWorks](https://www.mathworks.com/downloads/) if you do not have it. Then continue at **§2** (connect + verify).

### 3.1 Linux and WSL

**Recommended:** `bash scripts/matlab/setup.sh` (prompts for paths and license).

**Install only** (no prompts):

```bash
bash scripts/matlab/install.sh
```

Downloads [MPM](https://www.mathworks.com/help/install/ug/get-mpm-os-command-line.html) to **`$MPM_DIR`** (default `~/.cache/rl-adaptive-dbs/mpm` — a download cache, not the MATLAB install) and installs MATLAB to **`$MATLAB_ROOT`** (default `~/MATLAB`).

**Manual** (same as `install-matlab.sh`):

```bash
export MPM_DIR="${MPM_DIR:-$HOME/.cache/rl-adaptive-dbs/mpm}"
export MATLAB_ROOT="${MATLAB_ROOT:-$HOME/MATLAB}"
mkdir -p "$MPM_DIR" && cd "$MPM_DIR"
wget -O mpm https://www.mathworks.com/mpm/glnxa64/mpm
chmod +x mpm
./mpm install --release=R2025b --destination="$MATLAB_ROOT" --products=MATLAB
```

**WSL system dependencies** (headless batch):

```bash
sudo apt-get install -y xvfb libnss3 libxss1 libasound2t64 libgbm1 libxrandr2 \
  libatk-bridge2.0-0 libgtk-3-0 libdrm2 libxdamage1 libxcomposite1 libxfixes3 libxi6 libxtst6
```

### 3.2 macOS

1. Download the macOS installer from MathWorks (Apple Silicon or Intel as appropriate).
2. Run the installer GUI, or silent/text mode per [MathWorks install docs](https://www.mathworks.com/help/install/).
3. Default location: `/Applications/MATLAB_R2025b.app`.
4. Activate when prompted, then **§2.2**.

### 3.3 Windows

1. Download the Windows installer from MathWorks.
2. Run `setup.exe` (GUI) or `setup.exe -mode silent` with an installer input file per MathWorks docs.
3. Note the install path (often `C:\Program Files\MATLAB\R2025b`).
4. Activate when prompted, then **§2.3**.

---

## 4. Activate your license

Batch mode fails until MATLAB is licensed:

```text
MathWorks Licensing Error 1 — Unable to find a license for MATLAB.
```

| OS | Easiest path |
|----|----------------|
| **Linux / WSL** | `bash scripts/matlab/activate.sh` (Product Authorizer under `xvfb`), or place `license.lic` in `$MATLAB_ROOT/licenses/` |
| **macOS** | Open MATLAB → sign in; or Product Authorizer under `/Applications/MATLAB_*.app/bin/` |
| **Windows** | Open MATLAB from Start menu → sign in; or place license in `%MATLAB_ROOT%\licenses\` |

**License file locations** (any one is fine):

- `$MATLAB_ROOT/licenses/license.lic` (or `license.dat`)
- `~/.matlab/<RELEASE>_licenses/` (folder of `.lic` files)

**Network license:**

```bash
export MLM_LICENSE_FILE=27000@license.example.edu
```

**Confirm:**

```bash
source scripts/matlab/env.sh
matlab -batch "license('inuse'); disp('licensed'); exit"
```

---

## 5. Shell environment

**bash (Linux, macOS, WSL, Git Bash):**

```bash
source /path/to/rl-adaptive-dbs/scripts/matlab/env.sh
```

| Variable | Purpose |
|----------|---------|
| `MATLAB_ROOT` | Install root (contains `bin/matlab` or `bin/matlab.exe`) |
| `MATLAB_RELEASE` | e.g. `R2025b` — drives prefdir and license search |
| `MATLAB_PREFDIR` | Default `~/.matlab/<RELEASE>` |
| `RL_ADAPTIVE_DBS_MATLAB_MODEL` | `reference-material/KumaraveluEtAl2016` |
| `PATH` | Prepends `$MATLAB_ROOT/bin` |
| `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` | Platform arch `bin` folder (for `matlab.engine`) |

**Linux/WSL only:** `matlab` shell function uses **`xvfb-run`** when `DISPLAY` is unset; Ubuntu 24.04 may set `LD_PRELOAD` for system `libstdc++`.

**Windows (PowerShell):** set `MATLAB_ROOT`, `MATLAB_RELEASE`, and `PATH` as in §2.3; no `matlab-env.sh` required if vars are correct.

---

## 6. Connect Python (`matlab.engine`)

```bash
uv sync --group matlab    # R2025b → matlabengine==25.2.2
source scripts/matlab/env.sh
```

Quick check:

```python
import matlab.engine

eng = matlab.engine.start_matlab()
eng.cd("/path/to/rl-adaptive-dbs/reference-material/KumaraveluEtAl2016")
print(eng.which("simulate_network_model"))
eng.exit()
```

Repo helpers:

```bash
bash scripts/setup.sh             # Python + optional MATLAB (see docs/setup.md)
bash scripts/matlab/setup.sh      # interactive MATLAB-only setup
bash scripts/matlab/install.sh    # Linux/WSL install only
bash scripts/matlab/activate.sh   # license activation (Linux/WSL)
bash scripts/matlab/verify.sh     # install + license + Python engine (all OS via bash)
```

---

## 7. Reference model smoke test

**Phase 2 needs:** `CTX_BG_TH_network` dynamics (GPi spike trains). `scripts/matlab/verify.sh` runs a short segment automatically.

```bash
source scripts/matlab/env.sh
cd reference-material/KumaraveluEtAl2016
matlab -batch "simulate_network_model(1,1,0,1,true); exit"
```

Fifth argument `dynamics_only=true` returns after `CTX_BG_TH_network` (vendored patch; see [kumaravelu_vendor_patches.md](reference-material/kumaravelu_vendor_patches.md)).

**Full upstream script** (optional, for `.mat` reference dumps):

```bash
matlab -batch "simulate_network_model(1,1,0,1); exit"
```

Requires **Statistics** (upstream `randsample`, patched here) and **Signal Processing** (`dpss` in inlined `mtspectrumpt`) toolboxes. Mehregan $P_\beta$ (13–35 Hz) is **not** taken from this script’s 7–35 Hz integral — see [plant.md](plant.md) §6.

---

## 8. Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `matlab-env: MATLAB not found` | Set `MATLAB_ROOT` (§2.1–§2.3) |
| `Licensing Error 1` | §4; existing install: §2.4 |
| **Linux/WSL** segfault on first launch | `source matlab-env.sh`; writable `~/.matlab/<RELEASE>` |
| **Linux/WSL** `Unable to launch MVM server` | Install `xvfb` (§3.1); use `matlab` from `matlab-env.sh` |
| **Linux/WSL** `libstdc++` / GLIBCXX on Ubuntu 24.04 | [MathWorks Ubuntu 24.04 guidance](https://www.mathworks.com/matlabcentral/answers/2150489); `matlab-env.sh` sets `LD_PRELOAD` when needed |
| **macOS** `import matlab.engine` fails | Source `matlab-env.sh`; match `matlabengine` to release; ensure same Terminal session |
| **Windows** scripts not found | Use Git Bash or WSL; or PowerShell steps in §2.3 |
| **Windows** path with spaces | Quote `MATLAB_ROOT` in Git Bash: `"/c/Program Files/MATLAB/R2025b"` |
| Engine version mismatch | `matlabengine` must match release (R2025b → `25.2.2`) |

---

## 9. pytest integration

Tests that need MATLAB use `@pytest.mark.matlab`. They are **skipped** when MATLAB is not installed or not licensed. Equivalence tests compare Python plant output to reference MATLAB runs ([plant.md](plant.md) §8).

```bash
uv run pytest -m "not matlab"    # default fast CI
uv run pytest -m matlab          # requires licensed MATLAB
```

---

## 10. Related docs

- Cross-platform policy: [AGENTS.md](../AGENTS.md)
- Plant spec and bridge API: [plant.md](plant.md)
- Equivalence testing: [development/testing.md](development/testing.md)
- Day-to-day Python setup: [setup.md](setup.md), [development/venv.md](development/venv.md)
