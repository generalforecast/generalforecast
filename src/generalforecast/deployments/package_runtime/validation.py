# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Admission and bounded Arrow validation for the in-process package executor."""

from pathlib import Path

import pyarrow as pa

from generalforecast.packages import CatalogResolver, Package, PackageError, PackageLock
from generalforecast.packages.python_lock import PythonLock
from generalforecast.packages.schema import TableSchema, infer_schema, select_context
from generalforecast.packages.validation import ensure
from generalforecast.tabular import arrow_bytes, physical_type, table_digest, to_arrow

MAX_QUERY_ROWS = 10_000


def isolated(data: pa.Table, *, max_rows: int = 1_000_000) -> pa.Table:
    """Copies caller-owned buffers into immutable, normalized Arrow IPC bytes."""
    return pa.ipc.open_stream(arrow_bytes(to_arrow(data, max_rows=max_rows))).read_all()


def admit(lock: PackageLock | PythonLock) -> PackageLock:
    """Authenticates pinned capabilities independently of client-provided hashes."""
    parsed = PackageLock.from_dict(lock.to_dict())
    if isinstance(parsed, PythonLock):
        raise PackageError(
            "UnsupportedCapability", "Python packages require an isolated executor"
        )
    lock = parsed
    lock.admit(offline=True)
    ensure(
        CatalogResolver().resolve(lock.spec.model, offline=True) == lock.model,
        "authored model differs from locked model",
        "ModelMismatch",
    )
    ensure(
        lock.spec.target is not None
        or (lock.spec.context.mode == "request" and not lock.schema.columns),
        "runtime requires an explicit target outside schema-free request mode",
    )
    expected = lock.spec.outputs or (
        ("label", "probabilities") if lock.task == "classification" else ("value",)
    )
    ensure(
        lock.outputs == expected, "outputs differ from authored intent", "LockMismatch"
    )
    ensure(
        lock.spec.id not in lock.outputs,
        "ID collides with a prediction output",
    )
    spec = lock.spec
    if lock.task == "forecast":
        ensure(
            spec.time is not None and spec.horizon is not None,
            "forecast requires time and horizon",
        )
    return lock


def context_table(lock: PackageLock, context: pa.Table) -> tuple[pa.Table, TableSchema]:
    """Verifies full snapshots, locked schema and explicit selection policies."""
    table = isolated(context)
    if lock.spec.context.mode == "snapshot":
        ensure(
            table_digest(table) == lock.data_digest,
            "snapshot digest differs from lock",
            "SnapshotMismatch",
        )
    schema = infer_schema(lock.spec, lock.model, lock.task, table)
    if lock.schema.columns:
        expected = {c.name: c for c in lock.schema.columns}
        actual = {c.name: c for c in schema.columns}
        ensure(
            expected.keys() == actual.keys(),
            "context columns differ from lock",
            "SchemaChanged",
        )
        for name, column in actual.items():
            held = expected[name]
            ensure(
                (column.type, column.role, column.semantic)
                == (held.type, held.role, held.semantic)
                and (held.nullable or not column.nullable),
                "context physical schema differs from lock",
                "SchemaChanged",
            )
        ensure(
            not lock.schema.classes or schema.classes == lock.schema.classes,
            "context classes differ from lock",
            "SchemaChanged",
        )
        ensure(
            lock.schema.frequency == schema.frequency,
            "context frequency differs from lock",
            "SchemaChanged",
        )
    selected = select_context(lock.spec, table, lock.model)
    ensure(selected.num_rows > 0, "context is empty", "EmptyContext")
    return selected, infer_schema(lock.spec, lock.model, lock.task, selected)


def query_table(lock: PackageLock, schema: TableSchema, queries: pa.Table) -> pa.Table:
    """Rejects leaked targets, unknown columns, physical drift and missing roles."""
    table = isolated(queries, max_rows=MAX_QUERY_ROWS)
    ensure(table.num_rows > 0, "queries are empty")
    if lock.task != "forecast":
        from dataclasses import replace

        return Package(
            Path("."), lock.spec, replace(lock, schema=schema)
        ).validate_queries(table)
    expected = {
        c.name: c
        for c in schema.columns
        if c.role in ("time", "series", "future", "id")
    }
    required = {c.name for c in expected.values() if c.role != "id"}
    ensure(
        required <= set(table.column_names) <= expected.keys(),
        "forecast query columns differ from locked roles",
    )
    for field, values in zip(table.schema, table.columns, strict=True):
        ensure(
            physical_type(field.type) == expected[field.name].type,
            "query physical type differs from lock",
        )
        ensure(not values.null_count, "forecast queries cannot contain missing values")
    return table
