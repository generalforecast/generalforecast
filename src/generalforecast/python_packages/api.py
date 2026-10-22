# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small stdlib-only predictor contract shared with isolated workers."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

MAX_ROWS = 4096
MAX_WIRE = 60_000
Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class Field:
    """An ordered scalar column; nullability is explicit."""

    name: str
    kind: str
    nullable: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported schemas."""
        if (
            type(self.name) is not str
            or not 0 < len(self.name) <= 128
            or any(ord(c) < 32 for c in self.name)
            or self.kind not in ("string", "int64", "float64", "boolean")
            or type(self.nullable) is not bool
        ):
            raise ValueError("invalid table field")


@dataclass(frozen=True)
class TableSchema:
    """Exact ordered input/output fields, without coercion or dropped columns."""

    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        """Keep the contract immutable and bounded."""
        if (
            type(self.fields) is not tuple
            or not 0 < len(self.fields) <= 128
            or any(type(f) is not Field for f in self.fields)
            or len({f.name for f in self.fields}) != len(self.fields)
        ):
            raise ValueError("invalid table schema")

    def to_dict(self) -> list[dict]:
        """Return portable declarations."""
        return [
            {"name": f.name, "kind": f.kind, "nullable": f.nullable}
            for f in self.fields
        ]

    @classmethod
    def from_dict(cls, value: Any) -> "TableSchema":
        """Parse a strict scalar schema."""
        if type(value) is not list or not 0 < len(value) <= 128:
            raise ValueError("invalid table schema")
        fields = []
        for item in value:
            if type(item) is not dict or set(item) != {"name", "kind", "nullable"}:
                raise ValueError("invalid table field")
            fields.append(Field(**item))
        return cls(tuple(fields))


@dataclass(frozen=True)
class Table:
    """Bounded typed rows. Predictors return this type with the declared output schema."""

    schema: TableSchema
    rows: tuple[tuple[Scalar, ...], ...]

    def __post_init__(self) -> None:
        """Check values, including finite floats and signed 64-bit integer bounds."""
        if type(self.schema) is not TableSchema or type(self.rows) is not tuple:
            raise ValueError("invalid table")
        if len(self.rows) > MAX_ROWS:
            raise ValueError("table exceeds row bound")
        size = len(json.dumps(self.schema.to_dict(), ensure_ascii=False).encode()) + 32
        types = {"string": str, "int64": int, "float64": float, "boolean": bool}
        for row in self.rows:
            if type(row) is not tuple or len(row) != len(self.schema.fields):
                raise ValueError("table width differs from schema")
            size += len(row) + 3
            for field, value in zip(self.schema.fields, row, strict=True):
                if value is None:
                    if not field.nullable:
                        raise ValueError("null in nonnullable column")
                elif type(value) is not types[field.kind]:
                    raise ValueError("table value differs from declared type")
                elif type(value) is int and not -(2**63) <= value < 2**63:
                    raise ValueError("integer exceeds int64")
                elif type(value) is float and not math.isfinite(value):
                    raise ValueError("nonfinite table value")
                elif type(value) is str and len(value.encode()) > MAX_WIRE:
                    raise ValueError("table cell exceeds byte bound")
                size += len(
                    json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                )
                if size > MAX_WIRE:
                    raise ValueError("table exceeds aggregate byte bound")

    def to_dict(self) -> dict:
        """Copy table values into a JSON-compatible envelope."""
        return {"schema": self.schema.to_dict(), "rows": [list(r) for r in self.rows]}

    @classmethod
    def from_dict(cls, value: Any) -> "Table":
        """Reject extra fields and malformed row containers."""
        if type(value) is not dict or set(value) != {"schema", "rows"}:
            raise ValueError("invalid table envelope")
        rows = value["rows"]
        if type(rows) is not list or len(rows) > MAX_ROWS:
            raise ValueError("invalid table rows")
        if any(type(row) is not list for row in rows):
            raise ValueError("invalid table row")
        return cls(
            TableSchema.from_dict(value["schema"]), tuple(tuple(r) for r in rows)
        )


class Predictor(Protocol):
    """Implement class Model in predictor.py; it is instantiated once per worker.

    load receives a private artifacts directory containing only declared files.
    predict receives Table and must return Table with matching row count. Model
    execution is arbitrary Python, never safe merely because it implements this API.
    """

    def load(self, artifacts: Path) -> None:
        """Load immutable artifacts and initialize reusable predictor state."""
        ...

    def predict(self, table: Table) -> Table:
        """Return one output row per input row without implicit schema coercion."""
        ...
