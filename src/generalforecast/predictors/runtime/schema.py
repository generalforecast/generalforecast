# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Physical schema inspection and explicit target, feature and time contracts."""

from dataclasses import dataclass, replace
from typing import Any

import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast.columns import DATETIME_PARTS
from generalforecast.models.catalog import V1_POLICY
from generalforecast.predictors import frequency as _frequency_module
from generalforecast.predictors.models import ModelPin
from generalforecast.predictors.runtime.plan import (
    ExecutionPlan,
)
from generalforecast.predictors.validation import PredictorError, ensure, fields
from generalforecast.predictors.validation import column as column_name
from generalforecast.tabular import PHYSICAL_TYPES, physical_type, to_arrow

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
            column_name(column["name"])
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


def _failure(error: Any) -> PredictorError:
    """A frequency or kind refusal as the runtime raises it."""
    because = getattr(error, "because", None)
    message = getattr(error, "message", str(error))
    return PredictorError(
        getattr(error, "code", "InvalidPredictor"),
        f"{message}: {because}" if because else message,
    )


def series_times(spec: ExecutionPlan, table: pa.Table) -> dict[tuple, list]:
    """Each series' times, by the values of the plan's series columns."""
    assert spec.time is not None
    buckets: dict[tuple, list] = {}
    for row in table.select([*spec.series, spec.time]).to_pylist():
        key = tuple(row[column] for column in spec.series)
        ensure(
            all(value is not None for value in key),
            "forecast time and series cannot be missing",
        )
        buckets.setdefault(key, []).append(row[spec.time])
    return buckets


def series_groups(
    spec: Any, identities: list[tuple]
) -> tuple[list[int], list[tuple[int, ...]]]:
    """Each series' group, and each group's series, by the plan's `related`.

    With no `related`, each series is its own group. `related: all` makes
    one group of every series; related series columns make a group of the
    series that share their values. Groups and their series are in the
    order the series first appear.

    Raises:
        PredictorError: `CapacityExceeded` for a group of more series than
            `ServingPolicy.group_series`.
    """
    if spec.related is None:
        return list(range(len(identities))), [
            (index,) for index in range(len(identities))
        ]
    positions = [spec.series.index(column) for column in spec.related]
    keys: dict[tuple, int] = {}
    group_of = [
        keys.setdefault(tuple(identity[at] for at in positions), len(keys))
        for identity in identities
    ]
    groups: list[list[int]] = [[] for _ in keys]
    for series, group in enumerate(group_of):
        groups[group].append(series)
    largest = max((len(members) for members in groups), default=0)
    ensure(
        largest <= V1_POLICY.group_series,
        f"a related group holds {largest} series; at most "
        f"{V1_POLICY.group_series} forecast together",
        "CapacityExceeded",
    )
    return group_of, [tuple(members) for members in groups]


def check_groups(spec: ExecutionPlan, table: pa.Table) -> None:
    """Holds a forecaster's declared groups to their bounds: a size, one origin each.

    Raises:
        PredictorError: `CapacityExceeded` for a group past
            `ServingPolicy.group_series`, or for related series whose last
            times differ.
    """
    if spec.related is None:
        return
    assert spec.frequency is not None
    times = series_times(spec, table)
    identities = list(times)
    _, groups = series_groups(spec, identities)
    ends = [
        max(_frequency_module.coordinate(value, spec.frequency) for value in times[key])
        for key in identities
    ]
    for members in groups:
        ensure(
            len({ends[series] for series in members}) == 1,
            "related series must share the same forecast origin",
        )


def observed_frequency(spec: ExecutionPlan, table: pa.Table) -> str | int | float:
    """The frequency a forecaster's rows follow, held to the plan's when it declares one.

    One rule for every path, `frequency.check` and `frequency.infer`: within
    each series, times are distinct and one step apart.
    """
    times = series_times(spec, table)
    try:
        if spec.frequency is not None:
            _frequency_module.check(times, spec.frequency)
            return spec.frequency
        return _frequency_module.infer(times)
    except _frequency_module.FrequencyError as error:
        raise _failure(error) from None


def _declared(kind: str | None, physical: str) -> str | None:
    """The binder's override for a column of `kind` stored as `physical`."""
    from generalforecast.errors import InvalidInput
    from generalforecast.predictors.tables import semantic

    try:
        return semantic(kind, physical)
    except InvalidInput:
        raise PredictorError(
            "SchemaChanged", "declared type differs from physical data"
        ) from None


def infer_schema(
    spec: ExecutionPlan, pin: ModelPin, task: str, data: Any | None
) -> TableSchema:
    """Infers physical structure from all bounded rows and refuses semantic ambiguity.

    Without rows, a request or live context's schema is the lock's kinds in
    their `tables.PHYSICAL` storage; one whose lock knows no kinds is
    schema-free, and the rows it binds give its schema.
    """
    from generalforecast.predictors.tables import PHYSICAL

    kinds = dict(spec.inputs.types)
    role_kinds = dict(spec.role_types)
    roles = {
        spec.target: "target",
        spec.id: "id",
        spec.time: "time",
        **dict.fromkeys(spec.series, "series"),
        **dict.fromkeys(spec.past_inputs, "past"),
        **dict.fromkeys(spec.future_inputs, "future"),
    }
    roles.pop(None, None)
    if data is None:
        ensure(
            spec.context.mode != "snapshot",
            "a snapshot plan compiles against its rows",
            "SchemaRequired",
        )
        typed = (
            spec.target is not None
            and spec.target_type is not None
            and all(
                column in kinds
                for column in (*(spec.inputs.include or ()), *spec.forecast_inputs)
            )
            and all(
                column in role_kinds
                for column in (spec.id, spec.time, *spec.series)
                if column is not None
            )
        )
        if not typed:
            # A request or live context's rows give its schema when they bind.
            return TableSchema()
        assert spec.target is not None and spec.target_type is not None
        names = [
            *(column for column in (spec.id, spec.time) if column is not None),
            *spec.series,
            *(spec.inputs.include or ()),
            *spec.forecast_inputs,
            spec.target,
        ]
        source = [
            (
                column,
                PHYSICAL[
                    spec.target_type
                    if column == spec.target
                    else role_kinds.get(column) or kinds[column]
                ],
                True,
            )
            for column in names
        ]
        table = None
    else:
        table = to_arrow(data)
        ensure(table.num_rows > 0, "context is empty", "EmptyContext")
        source = [
            (field.name, physical_type(field.type), column.null_count > 0)
            for field, column in zip(table.schema, table.columns, strict=True)
        ]
    available = {column for column, _, _ in source}
    ensure(roles.keys() <= available, "declared role column is missing")
    ensure(
        set(spec.inputs.exclude) <= available,
        "input override column is missing",
    )
    included = spec.inputs.include
    if included is not None:
        ensure(
            set(included) <= available and not set(included) & roles.keys(),
            "inputs include unknown or reserved role columns",
        )
    columns = []
    for column, physical, nullable in source:
        column_name(column)
        role = roles.get(
            column,
            "input"
            if (included is None or column in included)
            and column not in spec.inputs.exclude
            else "excluded",
        )
        declared = (
            _declared(kinds.get(column), physical)
            if role in ("input", "past", "future")
            else None
        )
        if role in ("input", "past", "future"):
            accepted = ", ".join(sorted(pin.input_types))
            ensure(
                physical in pin.input_types,
                f"input {column!r} is {physical}; the model reads {accepted}"
                + (
                    " (a forecaster's covariates are numbers or booleans)"
                    if role in ("past", "future")
                    else ""
                ),
                "UnsupportedCapability",
            )
            ensure(
                not nullable
                or (pin.missing_values and spec.inputs.missing_values == "allow"),
                f"input {column!r} has missing values, which this model does not read",
                "MissingValues",
            )
        columns.append(ColumnSchema(column, physical, nullable, role, declared))
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
                        column_name(label)
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
            cadence = observed_frequency(spec, table)
            check_groups(replace(spec, frequency=cadence), table)
        else:
            ensure(
                spec.frequency is not None,
                "live forecast requires explicit frequency without schema data",
            )
            cadence = spec.frequency
        assert cadence is not None
        frequency = float(_frequency_module.step(cadence))
    else:
        ensure(
            not any(
                (
                    spec.series,
                    spec.horizon,
                    spec.frequency,
                    spec.past_inputs,
                    spec.future_inputs,
                )
            ),
            "series semantics require a forecast task",
        )
    return TableSchema(
        tuple(columns),
        table.num_rows
        if table is not None and spec.context.mode == "snapshot"
        else None,
        classes,
        frequency,
    )


def select_context(spec: ExecutionPlan, data: Any, pin: ModelPin) -> pa.Table:
    """The rows the plan binds, held to its selection's bounds.

    An exact selection binds every row and so stays within the model's row
    cap; a retrieved or adaptive one leaves each query's rows to the engine.
    A request context also stays within its declared `max_rows`.
    """
    table = to_arrow(data)
    context = spec.context
    if context.max_rows is not None:
        ensure(
            table.num_rows <= context.max_rows,
            "context exceeds the predictor's max_rows",
            "CapacityExceeded",
        )
    if spec.task == "forecast":
        # A forecaster reads each series' own history, so the model's row
        # cap bounds a series, not the context; any number of series bind.
        if spec.series and table.num_rows > pin.max_context_rows:
            counts = table.group_by(list(spec.series)).aggregate([([], "count_all")])
            longest = max(counts["count_all"].to_pylist(), default=0)
        else:
            longest = table.num_rows
        ensure(
            longest <= pin.max_context_rows,
            f"a series has {longest:,} context rows; the model reads at most "
            f"{pin.max_context_rows:,} per series, so keep each series' most recent ones",
            "CapacityExceeded",
        )
        return table
    strategy = spec.selection.strategy if spec.selection is not None else "exact"
    if strategy == "exact":
        ensure(
            table.num_rows <= pin.max_context_rows,
            "context exceeds model limit; declare an explicit selection",
            "CapacityExceeded",
        )
    return table
