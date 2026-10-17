"""Shared pytest fixtures for rl-adaptive-dbs."""

from __future__ import annotations

import os
import subprocess

import pytest


def matlab_available() -> bool:
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


def matlab_engine_available() -> bool:
    """Licensed MATLAB **and** ``matlabengine`` installed (``uv sync --group matlab``)."""
    if not matlab_available():
        return False
    try:
        import matlab.engine  # noqa: F401
    except ImportError:
        return False
    return True


def kumaravelu_model_fetched() -> bool:
    """The MATLAB model is fetched, not bundled (``scripts/matlab/fetch_kumaravelu.py``)."""
    from envs.plant.matlab_backend import model_dir

    return (model_dir(require=False) / "simulate_network_model.m").is_file()


def matlab_skip_reason() -> str | None:
    """Why MATLAB-backed tests cannot run here, or ``None`` when they can."""
    if not kumaravelu_model_fetched():
        return "Kumaravelu MATLAB model not fetched (python scripts/matlab/fetch_kumaravelu.py)"
    if not matlab_engine_available():
        return "MATLAB engine not available (install/license or uv sync --group matlab)"
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "matlab: requires MATLAB and the Kumaravelu reference bridge",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not any("matlab" in item.keywords for item in items):
        return
    reason = matlab_skip_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "matlab" in item.keywords:
            item.add_marker(skip)
