"""Shared pytest fixtures for rl-adaptive-dbs."""

from __future__ import annotations

import os
import subprocess

import pytest


def _matlab_available() -> bool:
    matlab_root = os.environ.get("MATLAB_ROOT", os.path.expanduser("~/MATLAB"))
    matlab_bin = os.path.join(matlab_root, "bin", "matlab")
    if not os.path.isfile(matlab_bin):
        return False
    try:
        result = subprocess.run(
            [matlab_bin, "-batch", "license('test','MATLAB'); exit"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "matlab: requires MATLAB and the Kumaravelu reference bridge",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _matlab_available():
        return
    skip = pytest.mark.skip(reason="MATLAB not installed or not licensed")
    for item in items:
        if "matlab" in item.keywords:
            item.add_marker(skip)
