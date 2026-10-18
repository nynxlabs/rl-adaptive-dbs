#!/usr/bin/env python3
"""Fetch the Kumaravelu et al. (2016) MATLAB model and apply our integration edits.

The model is not redistributed in this repository (no license found; see
LICENSING.md). This script downloads the pinned upstream revision from the
ModelDB GitHub mirror, checks every file against a recorded SHA-256, and writes
it into the gitignored files of ``reference-material/KumaraveluEtAl2016/``.

Our patches to the model (zero-context diffs that quote the upstream lines they
replace) live in the private ``kumaravelu_plant`` package while the model's
license is pending. When that package is installed, the script applies them and
checks the results against their ``manifest.json``; without it, it writes the
unmodified upstream files and says so (the plant bridge needs the patched copy).

Usage (standard library only; run it with ``uv run`` so the private package is found)::

    uv run python scripts/matlab/fetch_kumaravelu.py            # download (+ patch)
    uv run python scripts/matlab/fetch_kumaravelu.py --check    # verify an existing copy
    uv run python scripts/matlab/fetch_kumaravelu.py --source DIR   # offline: files from DIR
    uv run python scripts/matlab/fetch_kumaravelu.py --update-patches   # after editing the .m files

``plant_band_power.m`` has no upstream counterpart: its base is the upstream
``make_Spectrum`` function plus the Chronux routines that follow it in
``simulate_network_model.m`` (everything before ``specerr``); its patch turns
that base into our helper.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "ModelDBRepository/206232"
UPSTREAM_COMMIT = "0addcc04d7e6f16213a937591e4d2952e3a911ed"
UPSTREAM_URL = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{{name}}"

# SHA-256 of the files at UPSTREAM_COMMIT, as downloaded (CRLF kept).
UPSTREAM_SHA256 = {
    "simulate_network_model.m": "b436bc18b06ae2940be54a9e6a9b2e535b6863fb9962bc32cd00165c553a2d49",
    "run_simulate_network_model.q": "90461eebc46876856bd5d9c9257eccbaf0a48710ccc4f74c26ec59af52583a33",
    "readme.txt": "ca91ac4227c0af07b5388ff69ffafbc76ec273d45a7fb153a3a452ce0e02bb97",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "reference-material" / "KumaraveluEtAl2016"
SEAM = REPO_ROOT / "src" / "envs" / "plant" / "_kumaravelu.py"

# Output file -> how its unpatched base is built from the upstream files.
DERIVED_BASE = "plant_band_power.m"
OUTPUTS = ("simulate_network_model.m", "run_simulate_network_model.q", "readme.txt", DERIVED_BASE)

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FetchError(RuntimeError):
    pass


def default_patch_dir() -> Path | None:
    """Our patches from the private ``kumaravelu_plant`` package, or ``None`` without it.

    Loads the plant seam by path so this script keeps to the standard library.
    """
    spec = importlib.util.spec_from_file_location("_kumaravelu_seam", SEAM)
    if spec is None or spec.loader is None:
        return None
    seam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seam)
    return seam.matlab_patches_dir()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split(data: bytes) -> tuple[list[str], str]:
    """Split into lines without terminators; return (lines, eol). Rejects mixed EOLs."""
    text = data.decode("utf-8")
    crlf = text.count("\r\n")
    lf = text.count("\n")
    if crlf and crlf != lf:
        raise FetchError("mixed line endings")
    eol = "\r\n" if crlf else "\n"
    if not text.endswith(eol):
        raise FetchError("missing final newline")
    return text[: -len(eol)].split(eol), eol


def _join(lines: list[str], eol: str) -> bytes:
    return (eol.join(lines) + eol).encode("utf-8")


def _derived_base(upstream: dict[str, bytes]) -> bytes:
    """Upstream ``make_Spectrum`` through the Chronux routines before ``specerr``."""
    lines, eol = _split(upstream["simulate_network_model.m"])
    start = lines.index("function [area S f] = make_Spectrum(raw,params)")
    stop = lines.index("function Serr=specerr(S,J,err,trialave,numsp)")
    return _join(lines[start:stop], eol)


def _bases(upstream: dict[str, bytes]) -> dict[str, bytes]:
    bases = {name: upstream[name] for name in UPSTREAM_SHA256}
    bases[DERIVED_BASE] = _derived_base(upstream)
    return bases


def _patch_path(patch_dir: Path, name: str) -> Path:
    return patch_dir / f"{name}.patch"


def apply_patch(base: bytes, patch: str) -> bytes:
    """Apply a zero-context unified diff strictly (every removed line must match)."""
    lines, eol = _split(base)
    out: list[str] = []
    pos = 0  # next unconsumed base line (0-based)
    patch_lines = patch.splitlines()
    i = 0
    while i < len(patch_lines):
        m = HUNK_RE.match(patch_lines[i])
        i += 1
        if not m:
            continue
        old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
        # For pure insertions (old_len == 0) the diff names the line *after* which to insert.
        begin = old_start if old_len == 0 else old_start - 1
        if begin < pos:
            raise FetchError("overlapping hunks")
        out.extend(lines[pos:begin])
        pos = begin
        while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
            row = patch_lines[i]
            i += 1
            tag, body = row[:1], row[1:]
            if tag == "-":
                if pos >= len(lines) or lines[pos] != body:
                    raise FetchError(f"patch does not match base at line {pos + 1}")
                pos += 1
            elif tag == "+":
                out.append(body)
            elif tag == " ":
                if pos >= len(lines) or lines[pos] != body:
                    raise FetchError(f"patch context does not match at line {pos + 1}")
                out.append(body)
                pos += 1
            elif row.startswith("\\"):
                raise FetchError("'No newline at end of file' is not supported")
    out.extend(lines[pos:])
    return _join(out, eol)


def make_patch(name: str, base: bytes, result: bytes) -> str:
    a, eol_a = _split(base)
    b, eol_b = _split(result)
    if eol_a != eol_b:
        raise FetchError(f"{name}: line endings differ from upstream")
    diff = difflib.unified_diff(a, b, f"a/{name}", f"b/{name}", n=0, lineterm="")
    return "".join(f"{row}\n" for row in diff)


def _download(name: str) -> bytes:
    url = UPSTREAM_URL.format(name=name)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        return resp.read()


def load_upstream(source: Path | None) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for name, digest in UPSTREAM_SHA256.items():
        data = (source / name).read_bytes() if source else _download(name)
        if _sha256(data) != digest:
            where = source / name if source else UPSTREAM_URL.format(name=name)
            raise FetchError(f"{where}: SHA-256 mismatch (expected upstream commit {UPSTREAM_COMMIT[:12]})")
        files[name] = data
    return files


def _manifest(patch_dir: Path) -> dict[str, str]:
    return json.loads((patch_dir / "manifest.json").read_text(encoding="utf-8"))["result_sha256"]


def expected_sha256(patch_dir: Path | None) -> dict[str, str]:
    """SHA-256 of every output: patched results, or the plain upstream files without patches."""
    return _manifest(patch_dir) if patch_dir is not None else dict(UPSTREAM_SHA256)


def build(upstream: dict[str, bytes], patch_dir: Path | None) -> dict[str, bytes]:
    """Patched outputs, or the unmodified upstream files when ``patch_dir`` is ``None``."""
    if patch_dir is None:
        return {name: upstream[name] for name in UPSTREAM_SHA256}
    expected = _manifest(patch_dir)
    results: dict[str, bytes] = {}
    for name, base in _bases(upstream).items():
        patch = _patch_path(patch_dir, name)
        data = apply_patch(base, patch.read_text(encoding="utf-8")) if patch.exists() else base
        if _sha256(data) != expected[name]:
            raise FetchError(f"{name}: patched result does not match manifest.json")
        results[name] = data
    return results


def check(model_dir: Path, patch_dir: Path | None) -> list[str]:
    """Return problems with the fetched copy (empty when it is complete and unmodified)."""
    problems = []
    for name, digest in expected_sha256(patch_dir).items():
        path = model_dir / name
        if not path.is_file():
            problems.append(f"missing {path}")
        elif _sha256(path.read_bytes()) != digest:
            problems.append(f"modified {path} (run --update-patches if the edit is intended)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, help="read upstream files from this directory instead of downloading")
    parser.add_argument("--dest", type=Path, default=MODEL_DIR, help="output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="overwrite locally modified files")
    parser.add_argument(
        "--patches",
        type=Path,
        help="directory with our patches + manifest.json (default: the private kumaravelu_plant package, if installed)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the fetched copy; exit 1 if missing or modified")
    mode.add_argument(
        "--update-patches", action="store_true", help="rewrite patches/ and manifest.json from the files in --dest"
    )
    args = parser.parse_args(argv)
    dest: Path = args.dest
    patch_dir: Path | None = args.patches or default_patch_dir()
    flavour = "+ our patches" if patch_dir is not None else "unpatched"

    try:
        if patch_dir is None and not args.check:
            print(
                "fetch_kumaravelu: kumaravelu_plant is not installed, so our patches are not available "
                "(private while the model's license is pending; LICENSING.md). Writing the unmodified "
                "upstream files; the MATLAB plant bridge needs the patched copy.",
                file=sys.stderr,
            )
        if args.check:
            problems = check(dest, patch_dir)
            for p in problems:
                print(f"fetch_kumaravelu: {p}", file=sys.stderr)
            if not problems:
                print(f"fetch_kumaravelu: {dest} matches upstream {UPSTREAM_COMMIT[:12]} ({flavour})")
            return 1 if problems else 0

        upstream = load_upstream(args.source)

        if args.update_patches:
            if patch_dir is None:
                raise FetchError("--update-patches needs our patches: install kumaravelu_plant or pass --patches")
            digests = {}
            for name, base in _bases(upstream).items():
                result = (dest / name).read_bytes()
                patch = make_patch(name, base, result)
                path = _patch_path(patch_dir, name)
                if patch:
                    path.write_text(patch, encoding="utf-8", newline="\n")
                else:
                    path.unlink(missing_ok=True)
                if apply_patch(base, patch) != result:
                    raise FetchError(f"{name}: regenerated patch does not round-trip")
                digests[name] = _sha256(result)
            (patch_dir / "manifest.json").write_text(
                json.dumps({"upstream_commit": UPSTREAM_COMMIT, "result_sha256": digests}, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"fetch_kumaravelu: updated {patch_dir}")
            return 0

        results = build(upstream, patch_dir)
        if not args.force:
            expected = expected_sha256(patch_dir)
            dirty = [
                n for n in results if (dest / n).is_file() and _sha256((dest / n).read_bytes()) != expected[n]
            ]
            if dirty:
                raise FetchError(f"local edits in {', '.join(dirty)}; keep them with --update-patches or pass --force")
        dest.mkdir(parents=True, exist_ok=True)
        for name, data in results.items():
            with tempfile.NamedTemporaryFile(dir=dest, delete=False) as tmp:
                tmp.write(data)
            os.chmod(tmp.name, 0o644)  # NamedTemporaryFile creates 0600
            Path(tmp.name).replace(dest / name)
        print(f"fetch_kumaravelu: wrote {', '.join(results)} to {dest} (upstream {UPSTREAM_COMMIT[:12]}, {flavour})")
        return 0
    except (FetchError, OSError, ValueError) as exc:
        print(f"fetch_kumaravelu: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
