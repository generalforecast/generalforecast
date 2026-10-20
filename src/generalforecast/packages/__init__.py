"""Immutable declarative packages, with lazy exports to keep table conversion independent."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from generalforecast.packages.compiler import Package as Package
    from generalforecast.packages.lock import PackageLock as PackageLock
    from generalforecast.packages.models import CatalogResolver as CatalogResolver
    from generalforecast.packages.models import ModelPin as ModelPin
    from generalforecast.packages.models import ModelResolver as ModelResolver
    from generalforecast.packages.python_lock import PythonLock as PythonLock
    from generalforecast.packages.python_spec import (
        PythonPackageSpec as PythonPackageSpec,
    )
    from generalforecast.packages.python_spec import PythonSchema as PythonSchema
    from generalforecast.packages.schema import ColumnSchema as ColumnSchema
    from generalforecast.packages.schema import TableSchema as TableSchema
    from generalforecast.packages.spec import PackageSpec as PackageSpec
    from generalforecast.packages.spec import parse_spec as parse_spec
    from generalforecast.packages.validation import PackageError as PackageError

__all__ = [
    "CatalogResolver",
    "ColumnSchema",
    "ModelPin",
    "ModelResolver",
    "Package",
    "PackageError",
    "PackageLock",
    "PackageSpec",
    "PythonLock",
    "PythonPackageSpec",
    "PythonSchema",
    "TableSchema",
    "parse_spec",
]


def __getattr__(name: str) -> Any:
    """Loads compiler types only when requested, avoiding a tabular import cycle."""
    if name == "Package":
        from generalforecast.packages.compiler import Package

        return Package
    if name == "PackageLock":
        from generalforecast.packages.lock import PackageLock

        return PackageLock
    if name == "PythonLock":
        from generalforecast.packages.python_lock import PythonLock

        return PythonLock
    if name in ("PythonPackageSpec", "PythonSchema"):
        from generalforecast.packages import python_spec

        return getattr(python_spec, name)
    if name in ("ModelPin", "ModelResolver", "CatalogResolver"):
        from generalforecast.packages import models

        return getattr(models, name)
    if name in ("ColumnSchema", "TableSchema"):
        from generalforecast.packages import schema

        return getattr(schema, name)
    if name in ("PackageSpec", "parse_spec"):
        from generalforecast.packages import spec

        return getattr(spec, name)
    if name == "PackageError":
        from generalforecast.packages.validation import PackageError

        return PackageError
    raise AttributeError(name)
