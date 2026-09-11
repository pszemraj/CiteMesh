"""Shared PEP 562 lazy-export plumbing for CiteMesh packages.

Several ``__init__`` modules expose a stable import surface whose backing
modules are expensive to import (matplotlib, h5py, torch). Each one used to
hand-roll the same ``__getattr__``/``__dir__`` pair over a name-to-target
mapping; :func:`install_lazy_exports` is that plumbing in one place.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any

__all__ = ["install_lazy_exports"]


def install_lazy_exports(
    module_globals: dict[str, Any],
    mapping: Mapping[str, tuple[str, str]],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build (and install) module-level ``__getattr__``/``__dir__`` for lazy exports.

    The returned ``__getattr__`` resolves each mapped name on first access and
    caches it in ``module_globals`` so later lookups skip the import machinery.
    Module targets may be absolute (``"citemesh.strategies.base"``) or relative
    to the calling package (``".base"``).

    :param dict[str, Any] module_globals: Calling module's ``globals()``.
    :param Mapping[str, tuple[str, str]] mapping: Export name to
        ``(module, attribute)`` target.
    :return tuple[Callable[[str], Any], Callable[[], list[str]]]: The
        ``__getattr__`` and ``__dir__`` pair, also installed into
        ``module_globals``.
    """
    package = module_globals["__name__"]
    targets = dict(mapping)

    def lazy_getattr(name: str) -> Any:
        """Resolve a lazy export on first attribute access.

        :param str name: Requested module attribute.
        :return Any: Export resolved from its defining module.
        :raises AttributeError: If ``name`` is not a supported export.
        """
        target = targets.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")

        module_name, attr_name = target
        value = getattr(import_module(module_name, package), attr_name)
        module_globals[name] = value
        return value

    def lazy_dir() -> list[str]:
        """Return sorted module attribute names for interactive inspection.

        :return list[str]: Sorted module attribute names plus lazy exports.
        """
        return sorted(set(module_globals) | set(module_globals.get("__all__", ())))

    module_globals["__getattr__"] = lazy_getattr
    module_globals["__dir__"] = lazy_dir
    return lazy_getattr, lazy_dir
