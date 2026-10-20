"""Canonical SDK/runtime data adapters without extra prediction or HTTP clients."""

from typing import Any

import pyarrow as pa

from generalforecast.packages.validation import ensure
from generalforecast.python_packages.api import MAX_ROWS, MAX_WIRE, Table, TableSchema
from generalforecast.tabular import physical_type, to_arrow


def from_data(data: Any, schema: TableSchema) -> Table:
    """Convert existing SDK table formats and enforce exact Python predictor types."""
    table = to_arrow(data, max_rows=MAX_ROWS, max_bytes=MAX_WIRE)
    ensure(
        table.column_names == [f.name for f in schema.fields],
        "Python input column order/names differ from schema",
        "SchemaMismatch",
    )
    ensure(
        all(
            physical_type(column.type) == field.kind
            for column, field in zip(table.columns, schema.fields, strict=True)
        ),
        "Python input physical types differ from schema",
        "SchemaMismatch",
    )
    columns = [column.to_pylist() for column in table.columns]
    return Table(schema, tuple(zip(*columns, strict=True)))


def to_data(table: Table) -> pa.Table:
    """Return typed Arrow results through the existing transport's data path."""
    kinds = {
        "int64": pa.int64(),
        "float64": pa.float64(),
        "boolean": pa.bool_(),
        "string": pa.string(),
    }
    schema = pa.schema(
        [
            pa.field(field.name, kinds[field.kind], nullable=field.nullable)
            for field in table.schema.fields
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([row[i] for row in table.rows], type=kinds[field.kind])
            for i, field in enumerate(table.schema.fields)
        ],
        schema=schema,
    )
