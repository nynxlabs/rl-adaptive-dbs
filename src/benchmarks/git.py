"""Optional git metadata for benchmark manifests."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit_short(repo_root: Path | None = None) -> str | None:
    """Return short git commit hash when ``repo_root`` is inside a git work tree."""
    root = repo_root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None
