# Fresh machine validation (Phase 4)

Portability gate: confirm the repo **installs and passes basic checks** on clean Linux and Windows hosts — not only your daily WSL checkout. Day-to-day install and training workflows live in [setup.md](../setup.md). Pytest layout: [testing.md](testing.md).

---

## Not for training

**Multipass** and **Windows Sandbox** are for **testing and validation only**. Do **not** use them for real DDPG training, full Mehregan replication runs, or MATLAB plant work.

| Use Multipass / Sandbox for | Use your persistent dev machine (WSL) for |
|-----------------------------|-------------------------------------------|
| “Can a stranger clone and set up?” | `rl-dbs train`, `replicate_mehregan_ddpg.py` |
| `bash scripts/validate-fresh.sh` (pytest + CLI smoke) | Checkpoints under `artifacts/` |
| Occasional portability sign-off (Phase 4) | MATLAB + Kumaravelu plant (`scripts/matlab/`) |
| Tiny training smokes **via pytest** (mock env, 1 episode) | Long rollouts, benchmarks, TUI over `results/` |

**Why:** Sandbox is **disposable** (state wiped on close), explicitly **no WSL**, and the validation path is **Python-only**. Multipass defaults to **3G RAM / 2 CPUs** and clones from GitHub (not your live WSL tree). Both environments prove **setup portability**; they are not sized or intended as training workstations.

---

## Overview

**Recommended on Windows hosts:** **Multipass** (fresh **Linux**) + **Windows Sandbox** (fresh **Windows**, Git Bash, **no WSL**). **macOS** is deferred until hardware is available.

**Where commands run:** launchers (`run-multipass-linux-validation.ps1`, `launch-windows-sandbox-validation.ps1`, `check-windows-host.ps1`) run on the **Windows desktop** in **PowerShell** — not inside WSL. Validation itself runs **`bash scripts/validate-fresh.sh`** (Git Bash in Sandbox; bash in the Multipass guest).

### One-time host setup

Install on the **Windows desktop** (Administrator PowerShell), not inside WSL:

```powershell
# Both (default)
pwsh -ExecutionPolicy Bypass -File scripts/install-fresh-validation-host.ps1

# One environment only
pwsh -ExecutionPolicy Bypass -File scripts/install-fresh-validation-host.ps1 -Sandbox
pwsh -ExecutionPolicy Bypass -File scripts/install-fresh-validation-host.ps1 -Multipass
```

From **WSL** (raises UAC on Windows):

```bash
bash scripts/install-fresh-validation-host.sh
bash scripts/install-fresh-validation-host.sh --sandbox
bash scripts/install-fresh-validation-host.sh --multipass
bash scripts/install-fresh-validation-host.sh --check
```

Requires **Windows Pro/Enterprise**, firmware virtualization, and **Hyper-V**. Reboot if the installer exits with code **3010**, then rerun or install the remaining component.

Legacy alias: `scripts/prepare-desktop-host.ps1` (installs both, no flags).

Check readiness any time:

```powershell
pwsh -File scripts/check-windows-host.ps1
```

### Scripts

| Script | Role |
|--------|------|
| `scripts/setup.sh` | Install deps, import check, pytest; optional MATLAB; `--validate` for report |
| `scripts/validate-fresh.sh` | Fresh-host run: setup + CLI smoke + **report block** (Multipass / Sandbox) |
| `scripts/check-windows-host.ps1` | Hyper-V, Sandbox, Multipass readiness on Windows |
| `scripts/install-fresh-validation-host.ps1` | **Admin:** install `-Sandbox` and/or `-Multipass` (default both) |
| `scripts/install-fresh-validation-host.sh` | **WSL:** elevated wrapper for the `.ps1` installer |
| `scripts/prepare-desktop-host.ps1` | **Admin:** legacy alias (both; use install script for flags) |
| `scripts/refresh-multipass-catalog.ps1` | **Admin:** fix stale Multipass catalog (see Troubleshooting) |
| `scripts/repair-multipass.ps1` | **Admin:** unstick hung Multipass CLI / service (`repair-multipass.sh` from WSL) |
| `scripts/run-multipass-linux-validation.ps1` | **Desktop:** Multipass VM, **git clone**, validation |
| `scripts/launch-windows-sandbox-validation.ps1` | **Desktop:** Sandbox validate (default: WSL map; `-Clone`: git clone; 4:3 window resize) |
| `scripts/sandbox-window.ps1` | **Desktop:** resize running Sandbox window (no relaunch) |
| `scripts/run-parallel-fresh-validation.ps1` | **Desktop:** Multipass then Sandbox in parallel (staggered launch) |
| `scripts/validation-repo.ps1` | WSL repo path for host logs + Sandbox folder map |
| `scripts/bootstrap-fresh-linux.sh` | Inside Multipass: apt + uv + clone + validate |
| `scripts/bootstrap-fresh-windows.ps1` | Inside Sandbox: Git (GitHub release installer) + uv + validate |

### Host logs (WSL repo, gitignored)

Launchers write validation output under **`/home/nynxbox/neuroengineering/rl-adaptive-dbs/.validation-logs/`** (gitignored). Logs persist after the Multipass VM is deleted or Sandbox closes.

| File | Source |
|------|--------|
| `.validation-logs/multipass.log` | Multipass (clone inside VM; host captures stdout) |
| `.validation-logs/sandbox.log` | Windows Sandbox (map or `-Clone`; logs via mapped `.validation-logs/`) |
| `.validation-logs/multipass-launcher.log` | Parallel launcher PowerShell wrapper |

Tail in WSL:

```bash
tail -f .validation-logs/sandbox.log
tail -f .validation-logs/multipass.log
```

Manual: `bash scripts/validate-fresh.sh --log-file .validation-logs/manual.log`

On another machine, edit the default WSL path in `scripts/validation-repo.ps1`.

**Pass (Python-only):** `bash scripts/validate-fresh.sh` exits 0 and prints a report block. Shorthand after setup on an existing clone: `bash scripts/setup.sh --python-only --non-interactive --validate`. **With MATLAB (optional, on WSL — not in Sandbox):** `bash scripts/matlab/verify.sh` after `source scripts/matlab/env.sh` — [matlab.md](../matlab.md).

**What `validate-fresh.sh` runs:** `setup.sh --python-only --non-interactive --skip-tests`, then import check, `pytest -m "not matlab"`, and CLI smoke (`rl-dbs info`, `rl-dbs benchmark --dry-run`). It does **not** run full training or MATLAB suites.

### How long it takes (when healthy)

| Phase | Multipass | Sandbox |
|-------|-----------|---------|
| VM / environment boot | ~1–3 min | ~1–2 min |
| Git + `uv sync` + deps | ~3–10 min | ~4–12 min (GitHub installer download + `uv sync`) |
| pytest + CLI smoke | ~2–8 min | ~2–8 min |
| **Typical total** | **~10–20 min** | **~12–22 min** |

Launch has **no artificial timeout** in our scripts — Multipass runs until the VM is up or the hypervisor reports failure. A VM in `Starting` for a long time may still be booting under load; see Troubleshooting if it never reaches **Running**.

Record OS version, blockers, and doc fixes in [roadmap.md](roadmap.md) or [matlab.md](../matlab.md) when something fails only on one platform.

---

## Linux — Multipass (Ubuntu guest)

**One command (Windows desktop PowerShell)** — clones from GitHub inside a fresh Ubuntu VM (tests published repo, not uncommitted WSL changes):

```powershell
pwsh -ExecutionPolicy Bypass -File scripts/run-multipass-linux-validation.ps1
```

Options: `-Memory 3G` (default; raise to `4G` if the host has plenty of free RAM), `-KeepVm` (leave VM for debugging). **Host log:** `.validation-logs/multipass.log`.

**Manual** — from **PowerShell on the Windows host**:

```powershell
multipass launch 24.04 --name rl-dbs-linux --cpus 2 --memory 3G --disk 20G
multipass shell rl-dbs-linux
```

Inside the Ubuntu VM (only **git** + **uv** before clone):

```bash
sudo apt update && sudo apt install -y git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

git clone https://github.com/wilzen3476/rl-adaptive-dbs.git
cd rl-adaptive-dbs
bash scripts/validate-fresh.sh
```

When finished: `exit`, then on the host `multipass delete rl-dbs-linux --purge` for a clean slate next time.

---

## Windows — Sandbox (Git Bash, no WSL)

**One command (Windows desktop PowerShell)** — default maps this repo into Sandbox and runs validation:

```powershell
pwsh -File scripts/launch-windows-sandbox-validation.ps1
```

Default launcher window size is **1280×960 (4:3)** — applied after boot by resizing the host `WindowsSandboxClient` window (not a `.wsb` setting). Override with `-WindowWidth` / `-WindowHeight`, or `-NoWindowResize`. To resize a **running** Sandbox without relaunching: `pwsh -File scripts/sandbox-window.ps1`.

**Clone from GitHub** (parity with Multipass; tests published `main`, not uncommitted WSL changes):

```powershell
pwsh -File scripts/launch-windows-sandbox-validation.ps1 -Clone
```

`-Clone` maps only `scripts/` (bootstrap) and `.validation-logs/` (host log + cache) — not the full repo.

**`-Clone` flow:** launcher prefetches Git installer + **shallow `git clone`** + **`uv sync` wheel cache** into `.validation-logs/cache/` on the host. Inside Sandbox, bootstrap **tries `git clone --depth 1 --progress` first** (15 min timeout). If that fails, it copies the host repo cache. **`validate-fresh.sh`** tees to the host log (`--log-file`) and uses **`UV_CACHE_DIR`** so `uv sync` reuses host-downloaded wheels. **Mapped mode** — avoid for full validation; prefer **`-Clone`**.

**Background launch (WSL):** `bash scripts/run-sandbox-validation-background.sh` — kills any prior Sandbox, runs launcher via `nohup`, tails `sandbox.log` when ready.

Inside Sandbox, `bootstrap-fresh-windows.ps1` installs **Git for Windows** from a **host-prefetched** pinned `.exe` under `.validation-logs/cache/` (mapped into Sandbox). Then **uv** and `validate-fresh.sh` on native Sandbox disk (`-Clone`) or mapped WSL tree (default). Bump Git pin in `scripts/validation-repo.ps1`. **Host log:** `.validation-logs/sandbox.log`.

**Manual** — do **not** install or enable WSL in Sandbox. Each Sandbox session starts empty unless you use the launcher above.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `launch failed: Remote "" is unknown or unreachable` | **Admin:** `pwsh -File scripts/refresh-multipass-catalog.ps1` |
| Multipass CLI hangs / service won't stop | **Admin:** `pwsh -File scripts/repair-multipass.ps1` (from WSL: `bash scripts/repair-multipass.sh`) |
| Multipass VM stuck `Starting`, no IP | Boot can be slow under RAM load — wait, or close heavy apps; `repair-multipass.ps1`; retry after reboot; one validation at a time if RAM is tight |
| `Running` but IPv4 `N/A`, SSH timeout to `*.mshome.net` | `multipass delete rl-dbs-linux --purge`, then **Admin:** `repair-multipass.ps1` (clears stale `hosts.ics` lease), then relaunch |
| `Not enough memory` starting VM | Default `-Memory 3G`; free RAM on the host (close apps). Do **not** run `wsl --shutdown` from automation — only if **you** choose to |
| `WindowsSandbox.exe` missing after enable | Full **Windows reboot** after `install-fresh-validation-host.ps1`; rerun `-Sandbox` if needed |
| Multipass not on PATH | Use `C:\Program Files\Multipass\bin\multipass.exe` or re-open PowerShell after winget install |
| Sandbox window easy to miss | Check taskbar; tail **`.validation-logs/sandbox.log`** on the host |
| `git clone` silent / hung in Sandbox | Bootstrap uses shallow `--progress` clone with timeout; host cache fallback under `.validation-logs/cache/rl-adaptive-dbs-shallow/`. Log line `git clone succeeded inside Sandbox` = live clone worked |
| Mapped mode stuck on `validate-fresh.sh` | Use **`-Clone`** — `uv sync` over `\\wsl.localhost\...` is unreliable |

---

## Results template

| Date | Environment | Python-only | MATLAB (optional) | Notes |
|------|-------------|-------------|-------------------|-------|
| | Multipass Ubuntu 24.04 | | | |
| | Windows Sandbox + Git Bash | | | |
| | macOS | deferred | deferred | no hardware |

Paste the `=== rl-adaptive-dbs fresh validation ===` block from `validate-fresh.sh` into the Notes column or an issue.
