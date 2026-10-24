"""Install lazy package exports without importing their backing modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import cast

__all__ = ["install_lazy_exports"]


def install_lazy_exports(
    module_globals: dict[str, object],
    mapping: Mapping[str, tuple[str, str]],
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Build (and install) module-level ``__getattr__``/``__dir__`` for lazy exports.

    The returned ``__getattr__`` resolves each mapped name on first access and
    caches it in ``module_globals`` so later lookups skip the import machinery.
    Module targets may be absolute (``"citemesh.strategies.base"``) or relative
    to the calling package (``".base"``).

    :param dict[str, object] module_globals: Calling module's ``globals()``.
    :param Mapping[str, tuple[str, str]] mapping: Export name to
        ``(module, attribute)`` target.
    :return tuple[Callable[[str], object], Callable[[], list[str]]]: The
        ``__getattr__`` and ``__dir__`` pair, also installed into
        ``module_globals``.
    """
    package = cast(str, module_globals["__name__"])
    targets = dict(mapping)

    def lazy_getattr(name: str) -> object:
        """Resolve a lazy export on first attribute access.

        :param str name: Requested module attribute.
        :return object: Export resolved from its defining module.
        :raises AttributeError: If ``name`` is not a supported export.
        """
        target = targets.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")

        module_name, attr_name = target
        value: object = getattr(import_module(module_name, package), attr_name)
        module_globals[name] = value
        return value

    def lazy_dir() -> list[str]:
        """Return sorted module attribute names for interactive inspection.

        :return list[str]: Sorted module attribute names plus lazy exports.
        """
        all_exports = cast(Sequence[str], module_globals.get("__all__", ()))
        return sorted(set(module_globals) | set(all_exports))

    module_globals["__getattr__"] = lazy_getattr
    module_globals["__dir__"] = lazy_dir
    return lazy_getattr, lazy_dir
