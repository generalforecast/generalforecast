# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Physical schema inspection and explicit target, feature and time contracts."""

import datetime
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast.columns import DATETIME_PARTS
from generalforecast.packages.models import ModelPin
from generalforecast.packages.spec import PHYSICAL_TYPES, PackageSpec
from generalforecast.packages.validation import ensure, fields, text
from generalforecast.tabular import physical_type, to_arrow

pc: Any = _compute


@dataclass(frozen=True)
class ColumnSchema:
    """Physical type and missingness, with only user-declared semantic overrides."""

    name: str
    type: str
    nullable: bool
    role: str
    semantic: str | None = None

    def to_dict(self) -> dict:
        """Serializes a schema field without example row values."""
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "role": self.role,
            "semantic": self.semantic,
        }


@dataclass(frozen=True)
class TableSchema:
    """Inspectable locked schema, not a mutable dataframe or sampled approximation."""

    columns: tuple[ColumnSchema, ...] = ()
    rows: int | None = None
    classes: tuple[str | bool | int, ...] = ()
    frequency: float | None = None

    @property
    def features(self) -> tuple[str, ...]:
        """Lists model inputs in their locked order, excluding ID, time and target."""
        return tuple(column.name for column in self.columns if column.role == "input")

    def to_dict(self) -> dict:
        """Returns a fresh serialization of observed schema and task facts."""
        return {
            "columns": [column.to_dict() for column in self.columns],
            "rows": self.rows,
            "classes": list(self.classes),
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TableSchema":
        """Parses a serialized schema; compiler revalidation checks semantic coherence."""
        fields(value, {"columns", "rows", "classes", "frequency"})
        ensure(
            type(value["columns"]) is list and type(value["classes"]) is list,
            "invalid schema collections",
        )
        columns = []
        for column in value["columns"]:
            fields(column, {"name", "type", "nullable", "role", "semantic"})
            text(column["name"])
            ensure(
                type(column["type"]) is str and column["type"] in PHYSICAL_TYPES,
                "invalid physical type",
            )
            ensure(
                type(column["nullable"]) is bool
                and column["role"]
                in (
                    "input",
                    "target",
                    "id",
                    "time",
                    "series",
                    "past",
                    "future",
                    "excluded",
                ),
                "invalid column schema",
            )
            ensure(
                column["semantic"] is None
                or column["semantic"] in ("category", "numeric", "datetime"),
                "invalid semantic type",
            )
            columns.append(ColumnSchema(**column))
        ensure(
            len({column.name for column in columns}) == len(columns),
            "duplicate schema columns",
        )
        ensure(
            value["rows"] is None
            or (type(value["rows"]) is int and value["rows"] >= 0),
            "invalid schema row count",
        )
        ensure(
            all(type(item) in (str, bool, int) for item in value["classes"]),
            "invalid class labels",
        )
        ensure(
            value["frequency"] is None
            or (
                type(value["frequency"]) in (int, float)
                and 0 < value["frequency"] < 1e12
            ),
            "invalid schema frequency",
        )
        return cls(
            tuple(columns), value["rows"], tuple(value["classes"]), value["frequency"]
        )


def _compatible(actual: str, declared: str) -> bool:
    return (
        actual == declared
        or (declared == "category" and actual in ("string", "int64", "boolean"))
        or (declared == "numeric" and actual in ("int64", "float64"))
        or (declared == "datetime" and actual in ("date", "timestamp"))
    )


def _frequency(spec: PackageSpec, table: pa.Table) -> float:
    buckets: dict[tuple, list] = {}
    for row in table.select([*spec.series, spec.time]).to_pylist():
        key = tuple(row[column] for column in spec.series)
        instant = row[spec.time]
        ensure(
            instant is not None and all(value is not None for value in key),
            "forecast time and series cannot be missing",
        )
        ensure(
            type(instant) in (int, float, datetime.date, datetime.datetime),
            "forecast time must be numeric, date or timestamp",
        )
        if isinstance(instant, datetime.datetime):
            instant = instant.timestamp()
        elif isinstance(instant, datetime.date):
            instant = (instant - datetime.date(1970, 1, 1)).days * 86400
        buckets.setdefault(key, []).append(instant)
    steps = set()
    for values in buckets.values():
        ensure(
            len(values) >= 2, "each forecast series needs two times to verify frequency"
        )
        for left, right in pairwise(sorted(values)):
            ensure(right > left, "duplicate forecast time within a series")
            steps.add(float(right - left))
    ensure(
        len(steps) == 1,
        "forecast frequency is irregular; resample explicitly",
        "AmbiguousFrequency",
    )
    observed = steps.pop()
    if spec.frequency is not None:
        declared = spec.frequency
        if type(declared) is str:
            declared = (
                int(declared[:-1])
                * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[declared[-1]]
            )
        ensure(observed == declared, "declared frequency differs from observed times")
    return observed


def infer_schema(
    spec: PackageSpec, pin: ModelPin, task: str, data: Any | None
) -> TableSchema:
    """Infers physical structure from all bounded rows and refuses semantic ambiguity."""
    if data is None:
        ensure(
            spec.context.mode == "request" or bool(spec.inputs.types),
            "live datasets require declared physical types or explicit schema data",
            "SchemaRequired",
        )
        if not spec.inputs.types:
            return TableSchema()
        ensure(
            spec.target is not None and spec.target_type is not None,
            "typed schema requires target column and type",
        )
        assert spec.target is not None and spec.target_type is not None
        types = dict(spec.inputs.types)
        ensure(
            all(
                type(value) is str and value in PHYSICAL_TYPES
                for value in types.values()
            ),
            "schema without data requires physical types",
        )
        types[spec.target] = spec.target_type
        ensure(
            spec.target_type in ("boolean", "int64", "float64", "string"),
            "invalid target physical type",
        )
        source = [(column, kind, True) for column, kind in types.items()]
        table = None
    else:
        table = to_arrow(data)
        ensure(table.num_rows > 0, "context is empty", "EmptyContext")
        source = [
            (field.name, physical_type(field.type), column.null_count > 0)
            for field, column in zip(table.schema, table.columns, strict=True)
        ]
    available = {column for column, _, _ in source}
    roles = {
        spec.target: "target",
        spec.id: "id",
        spec.time: "time",
        **dict.fromkeys(spec.series, "series"),
        **dict.fromkeys(spec.past_inputs, "past"),
        **dict.fromkeys(spec.future_inputs, "future"),
    }
    roles.pop(None, None)
    ensure(roles.keys() <= available, "declared role column is missing")
    ensure(
        set(spec.inputs.exclude) <= available
        and set(dict(spec.inputs.types)) <= available,
        "input override column is missing",
    )
    ensure(
        not set(spec.inputs.exclude) & roles.keys(),
        "cannot exclude a required role column",
    )
    included = spec.inputs.include
    if included is not None:
        ensure(
            set(included) <= available and not set(included) & roles.keys(),
            "inputs include unknown or reserved role columns",
        )
    columns = []
    overrides = dict(spec.inputs.types)
    for column, kind, nullable in source:
        text(column)
        role = roles.get(
            column,
            "input"
            if (included is None or column in included)
            and column not in spec.inputs.exclude
            else "excluded",
        )
        declared = spec.target_type if role == "target" else overrides.get(column)
        if declared is not None:
            ensure(
                _compatible(kind, declared), "declared type differs from physical data"
            )
        if role in ("input", "past", "future"):
            ensure(
                kind in pin.input_types,
                "model does not support an input physical type",
                "UnsupportedCapability",
            )
            ensure(
                not nullable
                or (pin.missing_values and spec.inputs.missing_values == "allow"),
                "missing inputs are not supported by this contract",
                "MissingValues",
            )
        columns.append(
            ColumnSchema(
                column,
                kind,
                nullable,
                role,
                declared if declared in ("category", "numeric", "datetime") else None,
            )
        )
    features = [column for column in columns if column.role == "input"]
    feature_count = sum(
        DATETIME_PARTS
        if column.type in ("date", "timestamp")
        and column.semantic != "category"
        and task != "forecast"
        else 1
        for column in columns
        if column.role in ("input", "past", "future")
    )
    ensure(
        feature_count <= pin.max_features,
        "feature count exceeds model capability",
        "CapacityExceeded",
    )
    if task != "forecast":
        ensure(bool(features), "tabular context requires feature columns")
    classes = ()
    frequency = None
    if spec.target is not None:
        target = next(column for column in columns if column.role == "target")
        ensure(
            task == "classification" or target.type in ("int64", "float64"),
            "regression and forecast targets must be numeric",
        )
        if task == "classification":
            ensure(
                target.type in ("string", "int64", "boolean"),
                "classification target requires string, integer or boolean labels",
            )
            if table is not None:
                labels = table[spec.target].drop_null()
                ensure(len(labels) > 0, "context has no labeled rows", "EmptyContext")
                count = pc.count_distinct(labels).as_py()
                ensure(count >= 2, "classification context needs at least two classes")
                ensure(
                    pin.max_classes is None or count <= pin.max_classes,
                    "class count exceeds model capability",
                    "UnsupportedClassCount",
                )
                classes = tuple(sorted(pc.unique(labels).to_pylist()))
                for label in classes:
                    if type(label) is str:
                        text(label)
        elif table is not None:
            ensure(
                len(table[spec.target].drop_null()) > 0,
                "context has no labeled rows",
                "EmptyContext",
            )
    if task == "forecast":
        ensure(
            spec.time is not None and spec.horizon is not None,
            "forecast requires time and horizon",
        )
        assert spec.horizon is not None
        ensure(
            pin.max_horizon is not None and spec.horizon <= pin.max_horizon,
            "forecast horizon exceeds model capability",
            "UnsupportedCapability",
        )
        ensure(
            not spec.past_inputs or pin.past_inputs,
            "model does not support past inputs",
            "UnsupportedCapability",
        )
        ensure(
            not spec.future_inputs or pin.future_inputs,
            "model does not support known-future inputs",
            "UnsupportedCapability",
        )
        ensure(
            not features,
            "forecast covariates must be explicitly declared as past_inputs or future_inputs",
        )
        if table is not None:
            frequency = _frequency(spec, table)
        else:
            ensure(
                spec.frequency is not None,
                "live forecast requires explicit frequency without schema data",
            )
            value = spec.frequency
            assert value is not None
            if isinstance(value, str):
                frequency = float(
                    int(value[:-1])
                    * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[value[-1]]
                )
            else:
                frequency = float(value)
    else:
        ensure(
            not any(
                (
                    spec.time,
                    spec.series,
                    spec.horizon,
                    spec.frequency,
                    spec.past_inputs,
                    spec.future_inputs,
                )
            ),
            "time and series semantics require a forecast task",
        )
    return TableSchema(
        tuple(columns),
        table.num_rows
        if table is not None and spec.context.mode == "snapshot"
        else None,
        classes,
        frequency,
    )


def select_context(spec: PackageSpec, data: Any, pin: ModelPin) -> pa.Table:
    """Applies an explicit ordering policy, never arbitrary first-N truncation."""
    table = to_arrow(data)
    context = spec.context
    if context.selection == "most_recent":
        ensure(
            context.order_by in table.column_names, "context ordering column is missing"
        )
        order = table[context.order_by]
        ensure(order.null_count == 0, "context ordering cannot contain missing values")
        keys = [(context.order_by, "ascending")]
        if pc.count_distinct(order).as_py() != table.num_rows:
            ensure(
                spec.id is not None and spec.id in table.column_names,
                "tied context ordering requires an explicit unique ID",
            )
            ensure(
                table[spec.id].null_count == 0
                and pc.count_distinct(table[spec.id]).as_py() == table.num_rows,
                "context tie-breaker ID must be unique",
            )
            keys.append((spec.id, "ascending"))
        table = table.take(pc.sort_indices(table, sort_keys=keys))
        table = table.slice(max(0, table.num_rows - context.max_rows))
    ensure(
        table.num_rows <= pin.max_context_rows,
        "context exceeds model limit; declare an explicit ordered selection",
        "CapacityExceeded",
    )
    return table
