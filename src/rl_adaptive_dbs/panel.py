"""Shared plumbing for paper-figure panel scripts (``scripts/figures/papers/<paper>/<panel>/plot.py``).

One place for: loading sibling script modules, reading a manifest's overall gate verdict,
stamping run provenance (git commit, argv, input hashes) into manifests, and the exit-code
convention watchers and launchers rely on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_MISSING_INPUT = 2
EXIT_ABORTED = 3

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script_module(name: str, path: Path | str) -> ModuleType:
    """Import a script file as a fresh module ``name`` (registered in ``sys.modules`` first).

    Registration before ``exec_module`` is required for dataclasses and pickling; skipping it
    crashes on Python 3.12 when the loaded file defines a dataclass.
    """
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        msg = f"cannot load module {name!r} from {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _as_bool(value: Any) -> bool | None:
    return bool(value) if isinstance(value, bool) else None


def gates_pass(manifest: Mapping[str, Any] | None) -> bool | None:
    """Overall gate verdict from any panel manifest layout (``None`` if the manifest has none)."""
    if not manifest:
        return None
    candidates: list[Any] = [manifest.get("gates_pass")]
    for key in ("gates", "panel", "summary"):
        block = manifest.get(key)
        if not isinstance(block, Mapping):
            continue
        candidates += [block.get("pass"), block.get("all_pass"), block.get("gates_pass")]
        inner = block.get("gates")
        if isinstance(inner, Mapping):
            candidates += [inner.get("pass"), inner.get("all_pass")]
    for value in candidates:
        verdict = _as_bool(value)
        if verdict is not None:
            return verdict
    return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_state() -> dict[str, Any]:
    """Commit + dirty flag + hash of the uncommitted diff (tracked files), for reproducibility."""
    sha = _git("rev-parse", "HEAD")
    diff = _git("diff", "HEAD", "--", "src", "scripts")
    return {
        "commit": sha.strip() if sha else None,
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(
    *,
    argv: list[str] | None = None,
    inputs: Mapping[str, Path | str | None] | None = None,
) -> dict[str, Any]:
    """Run identity: id, time, git state, command line, Python, and sha256 of named inputs."""
    return {
        "run_id": uuid.uuid4().hex[:12],
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "argv": list(sys.argv if argv is None else argv),
        "python": platform.python_version(),
        "inputs": {
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in (inputs or {}).items()
            if path is not None
        },
    }


def stamp_manifest(
    manifest: dict[str, Any],
    *,
    argv: list[str] | None = None,
    inputs: Mapping[str, Path | str | None] | None = None,
) -> dict[str, Any]:
    """Add ``provenance`` and a normalized top-level ``gates_pass`` (in place; also returned)."""
    manifest["provenance"] = provenance(argv=argv, inputs=inputs)
    verdict = gates_pass(manifest)
    if verdict is not None:
        manifest["gates_pass"] = verdict
    return manifest


def exit_code(manifest: Mapping[str, Any] | None, *, smoke: bool = False) -> int:
    """0 = gates pass (or smoke plumbing check), 1 = gate fail or no verdict."""
    if smoke:
        return EXIT_PASS
    return EXIT_PASS if gates_pass(manifest) else EXIT_GATE_FAIL
