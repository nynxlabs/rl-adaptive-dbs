"""MATLAB-compatible RNG helpers for Kumaravelu plant initialization.

MATLAB ``rng(seed)`` uses the legacy Mersenne Twister (``mt19937ar``). Uniform
``rand`` draws match :class:`numpy.random.RandomState` bit-for-bit (verified vs
Engine for seeds 1–200). ``randperm(n)`` is ``argsort(rand(1,n))`` (0-based in Python).
``randperm(n, k)`` (MATLAB's k-subset draw) is **not** ``randperm(n)[:k]`` — use
:func:`load_cached_init_draws` / ``plant_init_export`` instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from envs.plant.network.integrator import NetworkInitDraws

__all__ = [
    "MATLAB_DEFAULT_SEED",
    "MatlabRandomState",
    "init_draws_cache_dir",
    "load_cached_init_draws",
    "resolve_init_draws",
]

MATLAB_DEFAULT_SEED: int = 5489  # rng(0) in modern MATLAB


def _normalize_seed(seed: int) -> int:
    return MATLAB_DEFAULT_SEED if seed == 0 else int(seed)


class MatlabRandomState:
    """Subset of MATLAB ``rng(seed)`` for Kumaravelu IC draw order."""

    def __init__(self, seed: int) -> None:
        self.seed = _normalize_seed(seed)
        self._rs = np.random.RandomState(self.seed)

    def rand(self, n: int) -> np.ndarray:
        return self._rs.rand(int(n))

    def randperm(self, n: int) -> np.ndarray:
        """0-based permutation matching MATLAB ``randperm(n)``."""
        size = int(n)
        return np.argsort(self._rs.rand(size))

    def randn(self, n: int) -> np.ndarray:
        """Not MATLAB-bit-exact — prefer cached :class:`NetworkInitDraws`."""
        return self._rs.randn(int(n))


def init_draws_cache_dir() -> Path:
    env = os.environ.get("RL_ADAPTIVE_DBS_PLANT_INIT_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "rl-adaptive-dbs" / "plant_init"


def load_cached_init_draws(seed: int, *, search_dirs: tuple[Path, ...] | None = None) -> NetworkInitDraws | None:
    """Load ``plant_init_seed{seed}.npz`` from repo fixtures or user cache."""
    candidates: list[Path] = []
    if search_dirs is not None:
        candidates.extend(search_dirs)
    repo_fixtures = Path(__file__).resolve().parents[4] / "tests" / "fixtures"
    candidates.append(repo_fixtures)
    candidates.append(init_draws_cache_dir())

    for directory in candidates:
        path = directory / f"plant_init_seed{seed}.npz"
        if path.is_file():
            return NetworkInitDraws.from_npz(path)
    return None


def resolve_init_draws(
    seed: int | None,
    *,
    export_if_missing: bool = False,
) -> NetworkInitDraws | None:
    """Return cached MATLAB IC draws for ``seed``, optionally exporting via Engine."""
    if seed is None:
        return None

    cached = load_cached_init_draws(seed)
    if cached is not None:
        return cached

    if not export_if_missing:
        return None

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "export_plant_init_draws",
            Path(__file__).resolve().parents[2] / "scripts" / "export_plant_init_draws.py",
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        export_fn = module.export_matlab_init_draws
    except (ImportError, OSError, FileNotFoundError):
        return None

    cache_dir = init_draws_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"plant_init_seed{seed}.npz"
    payload = export_fn(seed=seed)
    np.savez_compressed(out, **payload)
    return NetworkInitDraws.from_npz(out)
