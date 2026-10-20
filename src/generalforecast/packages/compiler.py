"""Functional package load/check/resolve/compile operations shared by all surfaces."""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa

from generalforecast.packages.capabilities import admit
from generalforecast.packages.files import MAX_DATA_BYTES, parse_yaml, read
from generalforecast.packages.lock import PackageLock
from generalforecast.packages.models import CatalogResolver, ModelPin, ModelResolver
from generalforecast.packages.python_lock import PythonLock
from generalforecast.packages.python_spec import PythonPackageSpec
from generalforecast.packages.schema import infer_schema, select_context
from generalforecast.packages.spec import PackageSpec, parse_spec
from generalforecast.packages.validation import (
    PackageError,
    bytes_digest,
    digest,
    ensure,
)
from generalforecast.python_packages.api import Table as PythonTable
from generalforecast.python_packages.compiler import PythonRevision, compile_python
from generalforecast.python_packages.tables import from_data, to_data
from generalforecast.tabular import (
    file_format,
    physical_type,
    read_table,
    table_digest,
    to_arrow,
)


@dataclass(frozen=True)
class Package:
    """Immutable authored package. check returns a new object, never mutates this one."""

    root: Path
    spec: PackageSpec | PythonPackageSpec
    lock: PackageLock | PythonLock | None = None

    @classmethod
    def load(cls, path: str | os.PathLike = ".") -> "Package":
        """Loads gf.yaml and optional lock without imports, downloads or directory scans."""
        root = Path(os.path.abspath(path))
        spec = parse_spec(parse_yaml(read(root, "gf.yaml")))
        lock = cls._lock(root)
        return cls(root, spec, lock)

    @staticmethod
    def _lock(root: Path) -> PackageLock | PythonLock | None:
        try:
            return PackageLock.load(root)
        except PackageError as error:
            if error.code != "MissingFile":
                raise
            return None

    def _source(self, data: Any | None) -> tuple[str, str | None, pa.Table | None]:
        assert isinstance(self.spec, PackageSpec)
        current = PackageSpec.from_dict(parse_yaml(read(self.root, "gf.yaml")))
        ensure(
            current == self.spec,
            "gf.yaml changed after load; load the package again",
            "SourceChanged",
        )
        files = {}
        table = to_arrow(data) if data is not None else None
        if self.spec.context.mode == "snapshot":
            ensure(
                data is None,
                "snapshot context is read only from its declared local file",
            )
            assert self.spec.context.source is not None
            raw = read(self.root, self.spec.context.source, limit=MAX_DATA_BYTES)
            files[self.spec.context.source] = bytes_digest(raw)
            table = read_table(raw, file_format(self.spec.context.source))
        source_digest = digest({"spec": self.spec.to_dict(), "files": files})
        # A live dataset changes independently; only its schema is revision content.
        data_digest = (
            table_digest(table)
            if table is not None and self.spec.context.mode == "snapshot"
            else None
        )
        return source_digest, data_digest, table

    def check(
        self,
        *,
        data: Any | None = None,
        offline: bool = True,
        resolver: ModelResolver | None = None,
        locked: bool = False,
        frozen: bool = False,
        write: bool = True,
        upgrade: bool = False,
    ) -> "Package":
        """Compiles a revision, optionally writing gf.lock atomically.

        locked rechecks and refuses changes without writing. frozen verifies
        source identity and lock integrity without model resolution or inference.
        Neither mode accepts upgrades or network resolution. Offline is default.
        """
        ensure(not (locked and frozen), "locked and frozen are mutually exclusive")
        ensure(
            not (locked or frozen) or (offline and not upgrade),
            "locked/frozen cannot resolve online or upgrade",
        )
        if isinstance(self.spec, PythonPackageSpec):
            ensure(
                data is None,
                "Python packages do not accept context data",
                "UnsupportedCapability",
            )
            return self._check_python(
                locked=locked, frozen=frozen, write=write, upgrade=upgrade
            )
        source_digest, data_digest, table = self._source(data)
        previous = self.lock
        if isinstance(previous, PythonLock):
            ensure(
                upgrade and not (locked or frozen),
                "package kind changed; explicit upgrade is required",
                "SchemaChanged",
            )
            previous = None
        if locked or frozen:
            ensure(previous is not None, "gf.lock is required", "LockRequired")
        if frozen:
            assert previous is not None
            previous = PackageLock.from_dict(previous.to_dict())
            assert isinstance(previous, PackageLock)
            ensure(
                previous.source_digest == source_digest,
                "source differs from gf.lock",
                "LockOutdated",
            )
            if data_digest is not None or self.spec.context.mode == "snapshot":
                ensure(
                    previous.data_digest == data_digest,
                    "data differs from gf.lock",
                    "LockOutdated",
                )
            return replace(self, lock=previous)
        if (
            previous is not None
            and previous.spec.model == self.spec.model
            and not upgrade
        ):
            pin = previous.model
            if not offline:
                resolved = (resolver or CatalogResolver()).resolve(
                    pin.resolved, offline=False
                )
                ensure(
                    resolved == pin,
                    "online model capabilities differ from locked metadata; explicit upgrade required",
                    "ModelMismatch",
                )
        else:
            pin = (resolver or CatalogResolver()).resolve(
                self.spec.model, offline=offline
            )
        pin = ModelPin.from_dict(pin.to_dict())
        task = self.spec.task
        if task is None:
            ensure(
                len(pin.tasks) == 1,
                "model supports multiple tasks; declare task explicitly",
                "AmbiguousTask",
            )
            task = pin.tasks[0]
        outputs = self.spec.outputs or (
            ("label", "probabilities") if task == "classification" else ("value",)
        )
        admit(self.spec, pin, task, outputs)
        schema = infer_schema(self.spec, pin, task, table)
        if previous is not None and previous.spec == self.spec and not upgrade:
            ensure(
                (
                    previous.schema.columns,
                    previous.schema.classes,
                    previous.schema.frequency,
                )
                == (schema.columns, schema.classes, schema.frequency),
                "schema differs from gf.lock; explicit upgrade is required",
                "SchemaChanged",
            )
        selected = select_context(self.spec, table, pin) if table is not None else None
        context_rows = (
            selected.num_rows
            if selected is not None and self.spec.context.mode == "snapshot"
            else None
        )
        lock = PackageLock.build(
            spec=self.spec,
            model=pin,
            task=task,
            schema=schema,
            source_digest=source_digest,
            data_digest=data_digest,
            outputs=outputs,
            context_rows=context_rows,
        )
        if locked:
            ensure(
                previous == lock,
                "compiled package differs from gf.lock",
                "LockOutdated",
            )
        elif write:
            # Avoid publishing a lock if the source changed during compilation.
            latest_source, latest_data, _ = self._source(data)
            ensure(
                (source_digest, data_digest) == (latest_source, latest_data),
                "source changed during compilation",
                "SourceChanged",
            )
            lock.write(self.root)
        return replace(self, lock=lock)

    def _python_source(self) -> PythonRevision:
        assert isinstance(self.spec, PythonPackageSpec)
        current = parse_spec(parse_yaml(read(self.root, "gf.yaml")))
        ensure(
            current == self.spec,
            "gf.yaml changed after load; load the package again",
            "SourceChanged",
        )
        return compile_python(self.root, self.spec.python)

    def _check_python(
        self, *, locked: bool, frozen: bool, write: bool, upgrade: bool
    ) -> "Package":
        assert isinstance(self.spec, PythonPackageSpec)
        previous = self.lock
        if locked or frozen:
            ensure(previous is not None, "gf.lock is required", "LockRequired")
        if previous is not None:
            previous = PackageLock.from_dict(previous.to_dict())
        candidate = PythonLock.build(self.spec, self._python_source())
        if locked or frozen:
            ensure(
                previous == candidate,
                "compiled Python package differs from gf.lock",
                "LockOutdated",
            )
            return replace(self, lock=candidate)
        if previous is not None and not upgrade:
            ensure(
                isinstance(previous, PythonLock)
                and previous.schema == candidate.schema
                and previous.task == candidate.task,
                "Python schema or package kind changed; explicit upgrade is required",
                "SchemaChanged",
            )
        if write:
            latest = PythonLock.build(self.spec, self._python_source())
            ensure(
                candidate == latest,
                "source changed during compilation",
                "SourceChanged",
            )
            candidate.write(self.root)
        return replace(self, lock=candidate)

    def source_archive(self) -> bytes:
        """Export pinned Python source bytes for a separately authorized upload."""
        ensure(
            isinstance(self.spec, PythonPackageSpec)
            and isinstance(self.lock, PythonLock),
            "only checked Python packages contain a source archive",
            "LockRequired",
        )
        assert isinstance(self.spec, PythonPackageSpec) and isinstance(
            self.lock, PythonLock
        )
        ensure(
            self.spec == self.lock.spec,
            "package intent differs from gf.lock",
            "LockOutdated",
        )
        revision = self._python_source()
        self.lock.verify_source(revision.archive)
        return revision.archive

    def resolve(
        self,
        *,
        resolver: ModelResolver | None = None,
        offline: bool = False,
        data: Any | None = None,
        upgrade: bool = False,
        write: bool = True,
    ) -> "Package":
        """Explicitly qualifies model metadata online, separately from artifact loading."""
        ensure(
            not isinstance(self.spec, PythonPackageSpec),
            "Python runtime qualification belongs to isolated publishing, not catalog resolution",
            "UnsupportedCapability",
        )
        return self.check(
            resolver=resolver, offline=offline, data=data, upgrade=upgrade, write=write
        )

    def compile(
        self, *, data: Any | None = None, frozen: bool = True
    ) -> PackageLock | PythonLock:
        """Returns portable immutable compiler output, requiring the checked lock by default."""
        checked = self.check(data=data, frozen=frozen, write=False)
        ensure(checked.lock is not None, "package has no compiled lock", "LockRequired")
        assert checked.lock is not None
        return checked.lock

    def snapshot_table(self) -> pa.Table:
        """Returns full normalized revision content for content-addressed upload."""
        ensure(self.lock is not None, "check the package first", "LockRequired")
        assert self.lock is not None
        self.check(frozen=True, write=False)
        ensure(
            self.spec.context.mode == "snapshot",
            "only snapshot packages contain local context",
        )
        _, _, table = self._source(None)
        assert table is not None
        ensure(
            table_digest(table) == self.lock.data_digest,
            "snapshot differs from gf.lock",
            "LockOutdated",
        )
        return table

    def context_table(self) -> pa.Table:
        """Returns selected snapshot context; live/request context belongs to runtime."""
        table = self.snapshot_table()
        assert self.lock is not None
        assert isinstance(self.spec, PackageSpec) and isinstance(self.lock, PackageLock)
        return select_context(self.spec, table, self.lock.model)

    def validate_queries(self, data: Any) -> pa.Table:
        """Validates a locked tabular request without silently redefining the schema."""
        ensure(
            self.lock is None or self.spec == self.lock.spec,
            "package intent differs from gf.lock",
            "LockOutdated",
        )
        if isinstance(self.lock, PythonLock):
            lock = PythonLock.from_dict(self.lock.to_dict())
            if isinstance(data, PythonTable):
                ensure(
                    data.schema == lock.schema.inputs,
                    "Python input schema differs from lock",
                    "SchemaMismatch",
                )
                return to_data(data)
            try:
                return to_data(from_data(data, lock.schema.inputs))
            except PackageError:
                raise
            except (ValueError, TypeError, OverflowError):
                raise PackageError(
                    "SchemaMismatch",
                    "Python query values differ from declared scalar schema",
                ) from None
        ensure(
            self.lock is not None and bool(self.lock.schema.columns),
            "request-context packages validate their schema per request",
            "SchemaRequired",
        )
        assert self.lock is not None
        assert isinstance(self.spec, PackageSpec)
        ensure(
            self.lock.task != "forecast",
            "forecast request alignment requires the series runtime validator",
            "UnsupportedCapability",
        )
        table = to_arrow(data)
        expected = {
            column.name: column
            for column in self.lock.schema.columns
            if column.role in ("input", "id")
        }
        required = {
            column.name for column in expected.values() if column.role == "input"
        }
        ensure(
            required <= set(table.column_names) <= expected.keys(),
            "query columns differ from locked inputs",
        )
        for field, array in zip(table.schema, table.columns, strict=True):
            column = expected[field.name]
            ensure(
                physical_type(field.type) == column.type,
                "query physical type differs from lock",
            )
            if column.role == "input":
                ensure(
                    not array.null_count
                    or (
                        self.lock.model.missing_values
                        and self.spec.inputs.missing_values == "allow"
                    ),
                    "query has unsupported missing values",
                )
        return table.select(
            [column for column in expected if column in table.column_names]
        )
