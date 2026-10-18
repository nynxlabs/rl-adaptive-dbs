"""The one place this repository reaches the Kumaravelu plant package.

The native Python port of the Kumaravelu et al. (2016) model, our patches to its
MATLAB source, and the data it generated live in a separate package,
``kumaravelu_plant``, that is not public while permission from the model's
authors is pending (LICENSING.md). Everything here imports without it; using
the plant without it raises :class:`PlantUnavailableError`.

Public modules expose the package's names through :func:`deferred`, which
resolves on first use, so the package can import this repository's own types
(``PlantConfig``, ``DbsSpec``, ``IntegrateResult``) without an import cycle.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

PACKAGE = "kumaravelu_plant"


class PlantUnavailableError(ImportError):
    """The Kumaravelu plant package is not installed."""

    def __init__(self, what: str = "The Kumaravelu plant") -> None:
        super().__init__(
            f"{what} is in the kumaravelu_plant package, which is not installed. That package "
            "holds the code and data derived from the Kumaravelu et al. (2016) model and stays "
            "private until the model's authors grant permission; see LICENSING.md "
            "§ Kumaravelu et al. (2016) model."
        )


def available() -> bool:
    """True when the package is installed."""
    return importlib.util.find_spec(PACKAGE) is not None


def require(module: str = "", *, what: str | None = None) -> ModuleType:
    """Import ``kumaravelu_plant`` (or its submodule ``module``), or raise a clear error."""
    if not available():
        raise PlantUnavailableError(what or (f"{PACKAGE}.{module}" if module else "The Kumaravelu plant"))
    return importlib.import_module(f"{PACKAGE}.{module}" if module else PACKAGE)


def data_dir() -> Path | None:
    """Model-generated data (plant init draws), or ``None`` without the package."""
    return require().DATA_DIR if available() else None


def matlab_patches_dir() -> Path | None:
    """Our patches to the MATLAB model, or ``None`` without the package."""
    return require().MATLAB_PATCHES_DIR if available() else None


class _Deferred:
    """Stands in for ``kumaravelu_plant.<module>.<name>`` until first use."""

    __slots__ = ("_module", "_name", "_target")

    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name
        self._target: Any = None

    def _resolve(self) -> Any:
        if self._target is None:
            self._target = getattr(require(self._module, what=self._name), self._name)
        return self._target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._resolve(), attr)

    def __instancecheck__(self, obj: object) -> bool:
        return isinstance(obj, self._resolve())

    def __reduce__(self) -> tuple[Any, tuple[str, str]]:
        return (deferred, (self._module, self._name))

    def __repr__(self) -> str:
        return f"<deferred {PACKAGE}.{self._module}.{self._name}>"


def deferred(module: str, name: str) -> Any:
    """``kumaravelu_plant.<module>.<name>``, resolved on first call or attribute access."""
    return _Deferred(module, name)
