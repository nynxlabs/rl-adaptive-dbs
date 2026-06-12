# Vault markdown (symlinks + githooks)

Many `*.md` files in this repo are **symlinks** into the Insync knowledge base (`~/knowledge-base/…`). Git stores **vault bytes** in commits via pre-commit hooks; the working tree stays symlinks.

**This is maintainer / machine workflow** — not required to build or run the project on a generic clone.

---

## What `vault-md-scan.sh` does

Script: `~/setup/knowledge-base/vault-md-scan.sh`  
Config: `~/setup/knowledge-base/kb-repo-link.toml`

1. Reads **`[link.*]`** — explicit repo path → vault file (e.g. `docs/matlab.md`, hookless `AGENTS.md`).
2. Reads **`[dir.*]`** — walks repo trees for `*.md` and symlinks each file to `vault_prefix` + relative path (this repo: `effort/…`).
3. **Bootstraps** empty vault files from existing repo files once (does not overwrite non-empty vault content).
4. **Idempotent** — safe to re-run after pull, clone, or broken symlinks.

It does **not** commit and does **not** run githooks.

---

## When to run `vault-md-scan.sh`

| Situation | Run scan? | Also run `install.sh`? |
|-----------|-----------|-------------------------|
| Normal edit to an existing vault-linked `*.md` | **No** — edit and commit; pre-commit indexes vault bytes | No |
| Edit **`kb-repo-link.toml`** (new `[link.*]` / `[dir.*]`, hookless path) | **Yes** | **Yes** — `install.sh --repo rl-adaptive-dbs` (or `--all`) |
| New `*.md` under a hooked tree (e.g. `docs/development/`) | **Yes** (creates symlink + vault path) | **Yes** if hooks not yet installed for that repo |
| Symlink **missing** or points at wrong vault file | **Yes** | **Yes** if `core.hooksPath` is wrong |
| After **`git reset --hard`**, **rebase**, **stash pop**, **`git restore`** | Usually **no** — hooks may not have run | **Yes** — `install.sh --repo <id>` restores symlinks (scan if still broken) |
| After **rebase / ship** when symlinks look wrong | Scan if `install.sh` alone did not fix | **Yes** (see **`~/AGENTS.md`** § Shipping) |
| Standing **`T` typechange** in `git status`, content matches `HEAD` | **No** — expected (index blob vs symlink) | No |

**Typical repair / bootstrap** (after `kb-repo-link.toml` changes or new hooked markdown):

```bash
bash ~/setup/knowledge-base/vault-md-scan.sh
bash ~/setup/knowledge-base/githooks/install.sh --repo rl-adaptive-dbs
```

Verify: `bash ~/setup/nynxbox/check-bind-mounts.sh`

---

## Authoritative docs (machine-wide)

| Topic | Where |
|-------|--------|
| Vault-md policy, `T` typechange, ship + re-run hooks | **`~/AGENTS.md`** § Vault markdown symlinks + git hooks |
| Setup index, SCM wrapper, cheat sheet | **`~/setup/AGENTS.md`** |
| Hook coverage, recovery | **`~/setup/knowledge-base/githooks/COVERAGE.md`** |

**Repo config** for this effort: `[dir.rl-adaptive-dbs]` and `[link.effort-*]` entries in `kb-repo-link.toml`. Hookless / gitignored paths: `AGENTS.md`, `matlab-license.md`.

---

## rl-adaptive-dbs markdown surfaces

| Kind | Examples | In git? |
|------|----------|---------|
| Vault symlink + hook | `README.md`, `docs/**`, `reference-material/.../changes.md` | Yes (materialized blob) |
| Vault symlink, hookless | `AGENTS.md`, `matlab-license.md` | No (gitignored) |
| Plain tracked | `reference-material/.../readme.txt`, `.m` sources | Yes |

Edits: open the repo symlink or the same path under `~/Insync/knowledge-base/neuroengineering/brain-stimulation-engineering/effort/`.
