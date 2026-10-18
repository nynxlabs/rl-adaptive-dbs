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
    """The MATLAB model is fetched with our patches (``scripts/matlab/fetch_kumaravelu.py``).

    ``plant_band_power.m`` exists only when the patches were applied, which needs
    the private ``kumaravelu_plant`` package; the plain upstream copy can't run
    the plant bridge.
    """
    from envs.plant.matlab_backend import model_dir

    return (model_dir(require=False) / "plant_band_power.m").is_file()


def kumaravelu_plant_available() -> bool:
    """The private Kumaravelu plant package is installed (LICENSING.md)."""
    from envs.plant._kumaravelu import available

    return available()


PLANT_SKIP_REASON = "kumaravelu_plant not installed (private while the model's license is pending; LICENSING.md)"


def matlab_skip_reason() -> str | None:
    """Why MATLAB-backed tests cannot run here, or ``None`` when they can."""
    if not kumaravelu_plant_available():
        return PLANT_SKIP_REASON
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
    config.addinivalue_line(
        "markers",
        "plant: requires the private kumaravelu_plant package (LICENSING.md)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not kumaravelu_plant_available():
        skip_plant = pytest.mark.skip(reason=PLANT_SKIP_REASON)
        for item in items:
            if item.get_closest_marker("plant") is not None:
                item.add_marker(skip_plant)
    if not any("matlab" in item.keywords for item in items):
        return
    reason = matlab_skip_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "matlab" in item.keywords:
            item.add_marker(skip)
