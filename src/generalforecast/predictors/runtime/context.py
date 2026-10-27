# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The served context: the rows a predictor binds, computed one way from a lock and a table.

`gf check` records the context rows' digest in `gf.lock`. `gf.Predictor`,
`gf serve`, `gf push` and hosted serving all read the rows they bind through
`served_context`, so every path binds the same rows: the query split, the
column projection and the classifier's label normalization happen here and
nowhere else. Selection, the rows each query reads from these, is the lock's
`selection`, which the binder applies.
"""

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from generalforecast.predictors.validation import PredictorError, ensure
from generalforecast.tabular import physical_type, table_digest, to_arrow


def context_columns(predictor: dict[str, Any]) -> list[str]:
    """The columns a lock's context rows keep, in order: id, time, series, inputs, target."""
    names = [
        *(predictor[key] for key in ("id", "time") if predictor.get(key)),
        *(predictor.get("series") or ()),
        *(column["name"] for column in predictor["inputs"]),
        predictor["target"],
    ]
    return list(dict.fromkeys(names))


def project(
    table: pa.Table, columns: list[str], target: str, *, classifier: bool
) -> pa.Table:
    """The rows' locked columns in order, with a classifier's whole-number labels as integers.

    A CSV with blanks in an integer target spells its labels `1.0`; the
    classes a lock records and every binder reads stay integers.
    """
    from generalforecast.predictors.tables import whole_numbers

    missing = [column for column in columns if column not in table.column_names]
    ensure(not missing, f"context has no column {missing[0]!r}" if missing else "")
    table = table.select(columns)
    if classifier:
        whole = whole_numbers(table[target])
        if whole is not None:
            table = table.set_column(
                table.schema.get_field_index(target), target, whole
            )
    return table


def snapshot_context(
    table: pa.Table,
    predictor: dict[str, Any],
    *,
    error: Any = None,
) -> tuple[pa.Table, pa.Table]:
    """A snapshot's context rows and query rows, by a lock's (or a spec's) split.

    Args:
        table: Every row of the snapshot file.
        predictor: A lock's `predictor` mapping, or the same keys from a
            spec being resolved.
        error: Builds the manifest-pointing error for a missing time column.
    """
    from generalforecast.predictors.tables import split_by

    context = predictor["context"]
    labeled, queries = split_by(
        table,
        predictor["target"],
        time=predictor.get("time"),
        until=context.get("until"),
        holdout=context.get("holdout"),
        error=error,
    )
    projected = project(
        labeled,
        context_columns(predictor),
        predictor["target"],
        classifier=predictor["task"] == "classification",
    )
    return typed(projected, predictor), queries


def served_context(lock: Any, table: pa.Table | None) -> pa.Table | None:
    """The rows a predictor binds, from its lock and a table of its context.

    Args:
        lock: A `contracts.PredictorLock`, or a version 2 lock document.
        table: For a snapshot, the snapshot file's rows, or the context rows
            `gf push` uploaded, which are those rows already; for a live
            context, its captured rows; for a request context, a call's rows.

    Returns:
        The rows to compile and bind; None for no table.

    Raises:
        PredictorError: `SnapshotMismatch` when a snapshot's context rows
            differ from the ones `gf check` locked.
    """
    if table is None:
        return None
    document = lock.to_dict() if hasattr(lock, "to_dict") else lock
    predictor = document["predictor"]
    context = predictor["context"]
    table = to_arrow(table)
    if context["kind"] == "snapshot":
        if table_digest(table) == context.get("digest"):
            return table
        rows, _ = snapshot_context(table, predictor)
        ensure(
            table_digest(rows) == context.get("digest"),
            "snapshot rows differ from gf.lock; run `gf check` again",
            "SnapshotMismatch",
        )
        return rows
    present = [
        column for column in context_columns(predictor) if column in table.column_names
    ]
    return typed(
        project(
            table,
            present,
            predictor["target"],
            classifier=predictor["task"] == "classification",
        ),
        predictor,
    )


def typed(table: pa.Table, predictor: dict[str, Any]) -> pa.Table:
    """Live or request rows in the storage a lock's kinds take.

    JSON carries times as ISO text and whole numbers as integers; a column
    the lock calls a timestamp is read as one, in UTC, and a `float` column
    of integers widens to float64.
    """
    kinds = {column["name"]: column.get("type") for column in predictor["inputs"]}
    for key, kind in (("id", "id_type"), ("time", "time_type")):
        if predictor.get(key):
            kinds[predictor[key]] = predictor.get(kind)
    for column, kind in zip(
        predictor.get("series") or (), predictor.get("series_types") or (), strict=False
    ):
        kinds[column] = kind
    kinds[predictor["target"]] = predictor.get("target_type")
    for index, name in enumerate(table.column_names):
        kind, values = kinds.get(name), table.column(index)
        storage = physical_type(values.type) if values.type != pa.null() else None
        try:
            if kind == "timestamp" and storage == "string":
                values = _timestamps(values)
            elif kind == "float" and storage == "int64":
                values = pc.cast(values, pa.float64())
            else:
                continue
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            raise PredictorError(
                "SchemaChanged", f"column {name!r} differs from the lock's {kind}"
            ) from None
        table = table.set_column(index, name, values)
    return table


def _timestamps(values: Any) -> Any:
    """ISO text as naive UTC timestamps; text with an offset is converted to UTC."""
    try:
        return pc.cast(values, pa.timestamp("us"))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        zoned = pc.cast(values, pa.timestamp("us", tz="UTC"))
        return pc.cast(zoned, pa.timestamp("us"))


__all__ = [
    "context_columns",
    "project",
    "served_context",
    "snapshot_context",
    "typed",
]
