# Fresh machine validation (Phase 4)

Portability gate: confirm the repo **installs and passes basic checks** on clean Linux and Windows hosts — not only your daily WSL checkout. Day-to-day install and training workflows live in [setup.md](../setup.md). Pytest layout: [testing.md](testing.md).

**Verification status:** Fresh-host scripts and the `validate-fresh.sh` CLI smoke block were added during Phase 4 alongside `rl-dbs`, `rl-dbs-tui`, and expanded dependency groups. **Maintainer WSL pass (2026-07-18):** `setup.sh --python-only --non-interactive --validate` succeeded after syncing `dev` + `figures` and fixing fixture-path / MATLAB-engine skip logic — see [setup.md](../setup.md) § Setup script verification status. **Multipass / Windows Sandbox** have not been re-run since that expansion (last Sandbox pass: 2026-06-30).

---

## Not for training

**Multipass** and **Windows Sandbox** are for **testing and validation only**. Do **not** use them for real DDPG training, full Mehregan replication runs, or MATLAB plant work.

| Use Multipass / Sandbox for | Use your persistent dev machine (WSL) for |
|-----------------------------|-------------------------------------------|
| “Can a stranger clone and set up?” | `rl-dbs train`, `replicate_mehregan_ddpg.py` |
| `bash scripts/validation/validate-fresh.sh` (pytest + CLI smoke) | Checkpoints under `artifacts/` |
| Occasional portability sign-off (Phase 4) | MATLAB + Kumaravelu plant (`scripts/matlab/`) |
| Tiny training smokes **via pytest** (mock env, 1 episode) | Long rollouts, benchmarks, TUI over `results/` |

**Why:** Sandbox is **disposable** (state wiped on close), explicitly **no WSL**, and the validation path is **Python-only**. Multipass defaults to **3G RAM / 2 CPUs** and clones from GitHub (not your live WSL tree). Both environments prove **setup portability**; they are not sized or intended as training workstations.

---

## Overview

**Recommended on Windows hosts:** **Multipass** (fresh **Linux**) + **Windows Sandbox** (fresh **Windows**, Git Bash, **no WSL**). **macOS** is deferred until hardware is available.

**Where commands run:** launchers (`run-multipass-linux-validation.ps1`, `launch-windows-sandbox-validation.ps1`, `check-windows-host.ps1`) run on the **Windows desktop** in **PowerShell** — not inside WSL. Validation itself runs **`bash scripts/validation/validate-fresh.sh`** (Git Bash in Sandbox; bash in the Multipass guest).

### One-time host setup

Install on the **Windows desktop** (Administrator PowerShell), not inside WSL:

```powershell
# Both (default)
pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1

# One environment only
pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1 -Sandbox
pwsh -ExecutionPolicy Bypass -File scripts/validation/install-fresh-validation-host.ps1 -Multipass
```

From **WSL** (raises UAC on Windows):

```bash
bash scripts/validation/install-fresh-validation-host.sh
bash scripts/validation/install-fresh-validation-host.sh --sandbox
bash scripts/validation/install-fresh-validation-host.sh --multipass
bash scripts/validation/install-fresh-validation-host.sh --check
```

Requires **Windows Pro/Enterprise**, firmware virtualization, and **Hyper-V**. Reboot if the installer exits with code **3010**, then rerun or install the remaining component.

Legacy alias: `scripts/validation/prepare-desktop-host.ps1` (installs both, no flags).

Check readiness any time:

```powershell
pwsh -File scripts/validation/check-windows-host.ps1
```

### Scripts

| Script | Role |
|--------|------|
| `scripts/setup.sh` | Install deps, import check, pytest; optional MATLAB; `--validate` for report |
| `scripts/validation/validate-fresh.sh` | Fresh-host run: setup + CLI smoke + **report block** (Multipass / Sandbox) |
| `scripts/validation/check-windows-host.ps1` | Hyper-V, Sandbox, Multipass readiness on Windows |
| `scripts/validation/install-fresh-validation-host.ps1` | **Admin:** install `-Sandbox` and/or `-Multipass` (default both) |
| `scripts/validation/install-fresh-validation-host.sh` | **WSL:** elevated wrapper for the `.ps1` installer |
| `scripts/validation/prepare-desktop-host.ps1` | **Admin:** legacy alias (both; use install script for flags) |
| `scripts/validation/refresh-multipass-catalog.ps1` | **Admin:** fix stale Multipass catalog (see Troubleshooting) |
| `scripts/validation/repair-multipass.ps1` | **Admin:** unstick hung Multipass CLI / service (`repair-multipass.sh` from WSL) |
| `scripts/validation/run-multipass-linux-validation.ps1` | **Desktop:** Multipass VM, **git clone**, validation |
| `scripts/validation/launch-windows-sandbox-validation.ps1` | **Desktop:** Sandbox validate (`-Clone` recommended; 4:3 window resize) |
| `scripts/validation/run-sandbox-validation-background.sh` | **WSL:** background `-Clone` launch + log sync (`nohup`) |
| `scripts/validation/sandbox-window.ps1` | **Desktop:** resize running Sandbox window (no relaunch) |
| `scripts/validation/run-parallel-fresh-validation.ps1` | **Desktop:** Multipass then Sandbox in parallel (staggered launch) |
| `scripts/validation/validation-repo.ps1` | WSL repo path for host logs + Sandbox folder map |
| `scripts/validation/bootstrap-fresh-linux.sh` | Inside Multipass: apt + uv + clone + validate |
| `scripts/validation/bootstrap-fresh-windows.ps1` | Inside Sandbox: Git (GitHub release installer) + uv + validate |

### Host logs (WSL repo, gitignored)

Launchers write validation output under **`/home/nynxbox/bme/rl-adaptive-dbs/.validation-logs/`** (gitignored). Logs persist after the Multipass VM is deleted or Sandbox closes.

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

Manual: `bash scripts/validation/validate-fresh.sh --log-file .validation-logs/manual.log`

On another machine, edit the default WSL path in `scripts/validation/validation-repo.ps1`.

**Pass (Python-only):** `bash scripts/validation/validate-fresh.sh` exits 0 and prints a report block. Shorthand after setup on an existing clone: `bash scripts/setup.sh --python-only --non-interactive --validate`. **With MATLAB (optional, on WSL — not in Sandbox):** `bash scripts/matlab/verify.sh` after `source scripts/matlab/env.sh` — [matlab.md](../matlab.md).

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
pwsh -ExecutionPolicy Bypass -File scripts/validation/run-multipass-linux-validation.ps1
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
bash scripts/validation/validate-fresh.sh
```

When finished: `exit`, then on the host `multipass delete rl-dbs-linux --purge` for a clean slate next time.

---

## Windows — Sandbox (Git Bash, no WSL)

**Recommended:** **`-Clone`** mode (git clone inside Sandbox on native `C:\` disk). Mapped WSL tree mode is for quick dev only — `\\wsl.localhost\...` maps are flaky and `uv sync` over 9p is slow.

### Quick start (WSL)

```bash
bash scripts/validation/run-sandbox-validation-background.sh
tail -f .validation-logs/sandbox.log
```

This kills any prior Sandbox, stages bootstrap scripts onto **NTFS** (`%LOCALAPPDATA%\rl-adaptive-dbs-validation` — reliable folder maps), launches **`-Clone`**, and mirrors `sandbox.log` back into the repo until Sandbox exits.

### Desktop PowerShell

```powershell
pwsh -ExecutionPolicy Bypass -File scripts/validation/launch-windows-sandbox-validation.ps1 -Clone
```

Default launcher window size is **1280×960 (4:3)** — applied after boot by resizing the host `WindowsSandboxClient` window. Override with `-WindowWidth` / `-WindowHeight`, or `-NoWindowResize`. Running instance: `pwsh -File scripts/validation/sandbox-window.ps1`.

**Mapped WSL tree (dev only):**

```powershell
pwsh -File scripts/validation/launch-windows-sandbox-validation.ps1
```

From WSL mapped mode: `bash scripts/validation/run-sandbox-validation-background.sh --mapped`. Legacy UNC paths: `--no-stage`.

### `-Clone` flow

| Step | Where | What |
|------|--------|------|
| Prefetch | Host (PowerShell) | Pinned **Git** installer, shallow **git clone** cache, **Windows** `uv` wheel cache, optional **VC++ redist** under `.validation-logs/cache/` |
| Stage (background launcher) | `%LOCALAPPDATA%\rl-adaptive-dbs-validation` | `scripts/` + Git installer copied to NTFS when WSL UNC is unreliable |
| Bootstrap | Inside Sandbox | Git + uv + **VC++ Redistributable** (PyTorch DLLs); shallow `git clone` (15 min timeout) or host repo-cache fallback |
| Validate | Git Bash in Sandbox | `validate-fresh.sh` with `UV_PYTHON=3.12`, `UV_CACHE_DIR` → copied Sandbox temp cache, output teed to host log |

**Host cache layout** (gitignored):

```
.validation-logs/
  sandbox.log
  sandbox-launcher.log
  cache/
    Git-2.55.0-64-bit.exe
    vc_redist.x64.exe          # optional prefetch; bootstrap downloads if missing
    rl-adaptive-dbs-shallow/   # shallow clone fallback
    uv/                        # Windows wheels (prefetch via native host uv, not WSL)
```

Bump the pinned Git release in `scripts/validation/validation-repo.ps1` when validating against a newer Git for Windows.

**Inside Sandbox:** `bootstrap-fresh-windows.ps1` installs Git from the host cache, **uv**, and **Microsoft VC++ Redistributable** (required for `torch` DLLs). Then `validate-fresh.sh` on `C:\rl-adaptive-dbs` (`-Clone`). **Host log:** `.validation-logs/sandbox.log` (or NTFS stage path when using the background launcher).

**Manual** — do **not** install or enable WSL in Sandbox. Each session starts empty unless you use the launcher above.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `launch failed: Remote "" is unknown or unreachable` | **Admin:** `pwsh -File scripts/validation/refresh-multipass-catalog.ps1` |
| Multipass CLI hangs / service won't stop | **Admin:** `pwsh -File scripts/validation/repair-multipass.ps1` (from WSL: `bash scripts/validation/repair-multipass.sh`) |
| Multipass VM stuck `Starting`, no IP | Boot can be slow under RAM load — wait, or close heavy apps; `repair-multipass.ps1`; retry after reboot; one validation at a time if RAM is tight |
| `Running` but IPv4 `N/A`, SSH timeout to `*.mshome.net` | `multipass delete rl-dbs-linux --purge`, then **Admin:** `repair-multipass.ps1` (clears stale `hosts.ics` lease), then relaunch |
| `Not enough memory` starting VM | Default `-Memory 3G`; free RAM on the host (close apps). Do **not** run `wsl --shutdown` from automation — only if **you** choose to |
| `WindowsSandbox.exe` missing after enable | Full **Windows reboot** after `install-fresh-validation-host.ps1`; rerun `-Sandbox` if needed |
| Multipass not on PATH | Use `C:\Program Files\Multipass\bin\multipass.exe` or re-open PowerShell after winget install |
| Sandbox window easy to miss | Check taskbar; tail **`.validation-logs/sandbox.log`** on the host |
| `git clone` silent / hung in Sandbox | Shallow `--progress` clone with 15 min timeout; host fallback `.validation-logs/cache/rl-adaptive-dbs-shallow/`. Log `git clone succeeded inside Sandbox` = live clone worked |
| Mapped mode stuck / no `sandbox.log` | Use **`-Clone`** + background launcher (NTFS stage). `\\wsl.localhost\...` maps are intermittent |
| `validate-fresh` instant exit 2, no output | Fixed in bootstrap: PowerShell must not expand `$PATH` in the bash `-lc` string (use backtick-escaped `$`) |
| `uv sync` access denied on `C:\host-logs\cache\uv` | Host cache must be **Windows** wheels (native `uv` prefetch). Bootstrap copies cache to Sandbox `%TEMP%` before sync |
| `matlabengine` / Python 3.14 in Sandbox | Repo `.python-version` pins 3.12; bootstrap sets `UV_PYTHON=3.12`. `--python-only` uses `uv sync --group dev --group figures` (no MATLAB group) |
| PyTorch `c10.dll` / WinError 126 | Install **VC++ Redistributable** in bootstrap; prefetch `vc_redist.x64.exe` into `.validation-logs/cache/` |
| `uv` / GitHub download hung in Sandbox | Host prefetch into `.validation-logs/cache/` before launch (launcher does this for `-Clone`) |

---

## Results template

| Date | Environment | Python-only | MATLAB (optional) | Notes |
|------|-------------|-------------|-------------------|-------|
| 2026-07-18 | Maintainer WSL (existing checkout) | pass (291 pytest + CLI smoke) | not run | `setup.sh --python-only --non-interactive --validate`; post Phase 4 re-verify |
| | Multipass Ubuntu 24.04 | | | |
| 2026-06-30 | Windows Sandbox + Git Bash (`-Clone`) | pass (89 pytest) | n/a | `git_sha` from shallow clone; VC++ redist + Windows uv cache |
| | macOS | deferred | deferred | no hardware |

Paste the `=== rl-adaptive-dbs fresh validation ===` block from `validate-fresh.sh` into the Notes column or an issue.
