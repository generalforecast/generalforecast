# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reading context tables: column kinds, role candidates and the query split.

A column has one kind, the vocabulary `gf.yaml`'s `inputs.types` and `gf.lock`
use: `int`, `float`, `0/1`, `category`, `text`, `timestamp` or `bool`.
`kind_of` reads it from a column's values. Two documented translations
leave it:

- `PHYSICAL` is the Arrow storage a kind takes when no rows say otherwise,
  as a request or live context's schema is compiled from the lock alone.
  With rows, the storage is the rows' own (`tabular.physical_type`).
- `semantic` is the override the predictor binder reads when a kind encodes
  otherwise than its storage does: an integer column of `category` kind is
  encoded as categories, not numbers. `columns.ENCODINGS` gives each
  storage's encoding, `columns.KIND_ENCODINGS` each kind's.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast import tabular
from generalforecast.errors import InvalidInput
from generalforecast.predictors import files
from generalforecast.predictors.spec import KINDS, PredictorSpec, until_timestamp
from generalforecast.predictors.validation import PredictorError

pc: Any = _compute  # ty cannot see Arrow's generated compute functions
FORMATS = (".csv", ".tsv", ".parquet", ".arrow", ".feather", ".json")
CATEGORY_DISTINCT = 20
CATEGORY_SHARE = 0.05
ID_NAMES = ("id", "key", "uuid", "guid", "pk", "code", "number")
TIME_NAMES = ("time", "date", "timestamp", "ts", "at", "when", "day", "month")
TARGET_NAMES = ("target", "label", "outcome", "class", "churn", "fraud", "y")


@dataclass(frozen=True)
class ColumnInfo:
    """One column of an inspected table.

    Attributes:
        name: The column name.
        kind: One of `int`, `float`, `0/1`, `category`, `text`, `timestamp`, `bool`.
        unique: Whether every row has a distinct, present value.
        empty: How many rows have no value.
        distinct: How many distinct present values there are.
    """

    name: str
    kind: str
    unique: bool
    empty: int
    distinct: int


@dataclass(frozen=True)
class TableInfo:
    """What `inspect_table` learns about a context file.

    Attributes:
        path: The file inspected, as given.
        rows: The row count.
        columns: Each column with its inferred kind.
        targets: Target candidates, best first.
        ids: Id candidates, best first.
        times: Time candidates, best first.
    """

    path: str
    rows: int
    columns: tuple[ColumnInfo, ...]
    targets: tuple[str, ...]
    ids: tuple[str, ...]
    times: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """The column names in file order."""
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnInfo:
        """The column with a name."""
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def kinds(self) -> dict[str, str]:
        """Each column's kind by name."""
        return {column.name: column.kind for column in self.columns}


def _invalid(error: PredictorError, path: str) -> InvalidInput:
    messages = {
        "MissingFile": (f"`{path}` does not exist", "check the path"),
        "CapacityExceeded": (
            f"`{path}` is too large to read",
            f"snapshots are at most {files.MAX_DATA_BYTES // (1024 * 1024)} MB",
        ),
        "UnsafePath": (f"`{path}` is not a readable file", None),
    }
    message, hint = messages.get(
        error.code, (f"cannot read `{path}`: {error}".rstrip("."), None)
    )
    return InvalidInput(message=message, param="context.snapshot", hint=hint)


def read_frame(root: str | Path, path: str) -> pa.Table:
    """Reads a .csv, .parquet, .arrow or .json file inside a predictor directory.

    Raises:
        InvalidInput: If the path leaves the directory, the format is unknown,
            or the file is missing, too large or malformed.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in FORMATS:
        raise InvalidInput(
            message=f"`{path}` is not a .csv, .parquet, .arrow or .json file",
            param="context.snapshot",
            hint="save the context as one of those formats",
        )
    try:
        data = files.read(Path(root), path, limit=files.MAX_DATA_BYTES)
        return tabular.read_table(data, suffix)
    except PredictorError as error:
        raise _invalid(error, path) from None


def inspect_table(path: str | Path) -> TableInfo:
    """Reads a table file and infers each column's kind and role candidates.

    Args:
        path: A .csv, .parquet, .arrow or .json file.

    Returns:
        The columns with kinds, uniqueness and empty counts, plus ranked target,
        id and time candidates.

    Raises:
        InvalidInput: If the file is missing, unreadable or empty.
    """
    file = Path(path)
    frame = read_frame(file.parent if str(file.parent) else ".", file.name)
    return inspect_frame(frame, path=str(path))


def inspect_frame(frame: Any, *, path: str = "") -> TableInfo:
    """Infers kinds and role candidates for rows already in memory."""
    table = to_table(frame)
    if table.num_rows == 0:
        raise InvalidInput(
            message=f"`{Path(path).name if path else 'context'}` has no rows",
            param="context.snapshot",
            hint="the context needs rows to learn from",
        )
    columns = tuple(
        _column(name, table[name], table.num_rows) for name in table.column_names
    )
    return TableInfo(
        path=path,
        rows=table.num_rows,
        columns=columns,
        targets=_targets(columns),
        ids=_ids(columns),
        times=_times(columns),
    )


def to_table(frame: Any) -> pa.Table:
    """Any supported rows as an Arrow table."""
    try:
        return tabular.to_arrow(frame)
    except PredictorError as error:
        raise InvalidInput(
            message=f"cannot read the rows: {error}".rstrip("."),
            param="context",
        ) from None


def _column(name: str, values: pa.ChunkedArray, rows: int) -> ColumnInfo:
    empty = _empty_count(values)
    present = pc.drop_null(_blank_to_null(values))
    distinct = pc.count_distinct(present).as_py() if len(present) else 0
    kind = kind_of(values, present, distinct)
    return ColumnInfo(name, kind, empty == 0 and distinct == rows, empty, distinct)


def _blank_to_null(values: pa.ChunkedArray) -> pa.ChunkedArray:
    kind = values.type
    if pa.types.is_dictionary(kind):
        values = pc.cast(values, kind.value_type)
        kind = values.type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        stripped = pc.utf8_trim_whitespace(values)
        return pc.if_else(pc.equal(stripped, ""), pa.scalar(None, kind), stripped)
    if pa.types.is_floating(kind):
        return pc.if_else(pc.is_nan(values), pa.scalar(None, kind), values)
    return values


def _empty_count(values: pa.ChunkedArray) -> int:
    return _blank_to_null(values).null_count


def kind_of(values: pa.ChunkedArray, present: pa.ChunkedArray, distinct: int) -> str:
    """The inferred kind of a column from its present values."""
    kind = present.type
    if pa.types.is_boolean(kind):
        return "bool"
    if pa.types.is_date(kind) or pa.types.is_timestamp(kind):
        return "timestamp"
    if pa.types.is_integer(kind):
        return "0/1" if _binary(present, distinct) else "int"
    if pa.types.is_floating(kind):
        return "0/1" if _binary(present, distinct) else "float"
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        if distinct and distinct <= 2:
            lowered = set(pc.utf8_lower(pc.unique(present)).to_pylist())
            if lowered <= {"0", "1"}:
                return "0/1"
            if lowered <= {"true", "false"} or lowered <= {"yes", "no"}:
                return "bool"
        if _timestamps(present):
            return "timestamp"
        if distinct <= CATEGORY_DISTINCT or distinct <= CATEGORY_SHARE * len(present):
            return "category"
        return "text"
    raise InvalidInput(
        message=f"column type {kind} is not supported",
        param="context",
        hint="use numbers, text, booleans or timestamps",
    )


# The Arrow storage a kind takes in a schema compiled without rows.
PHYSICAL = {
    "int": "int64",
    "float": "float64",
    "0/1": "int64",
    "category": "string",
    "text": "string",
    "timestamp": "timestamp",
    "bool": "boolean",
}


def semantic(kind: str | None, physical: str) -> str | None:
    """The binder's override for a kind stored as `physical`, or None when the storage encodes it.

    Raises:
        InvalidInput: For a kind the storage cannot hold, such as a `float`
            column stored as text.
    """
    from generalforecast.columns import ENCODINGS, KIND_ENCODINGS

    if kind is None or KIND_ENCODINGS[kind] == ENCODINGS[physical]:
        return None
    encoding = KIND_ENCODINGS[kind]
    if encoding == "categorical" and physical in ("int64", "boolean"):
        return "category"
    if encoding == "numeric" and physical == "boolean":
        return None
    if encoding == "boolean" and physical in ("string", "int64"):
        return "category"
    raise InvalidInput(
        message=f"a column stored as {physical} cannot be read as {kind}",
        param="inputs.types",
        hint="declare a type its values hold, or convert the column",
    )


def _binary(present: pa.ChunkedArray, distinct: int) -> bool:
    if not 1 <= distinct <= 2:
        return False
    return set(pc.unique(present).to_pylist()) <= {0, 1}


def _timestamps(present: pa.ChunkedArray) -> bool:
    sample = present.slice(0, 1000)
    try:
        pc.cast(sample, pa.timestamp("us"))
        return True
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return False


def _named_like(name: str, words: Sequence[str]) -> bool:
    lowered = name.lower()
    parts = [part for part in lowered.replace("-", "_").split("_") if part]
    return any(
        part in words or any(part.endswith(word) for word in words if len(word) > 2)
        for part in parts
    )


def _ids(columns: Sequence[ColumnInfo]) -> tuple[str, ...]:
    unique = [
        column
        for column in columns
        if column.unique and column.kind in ("int", "text", "category")
    ]
    named = [column.name for column in unique if _named_like(column.name, ID_NAMES)]
    rest = [column.name for column in unique if column.name not in named]
    return (*named, *rest)


def _times(columns: Sequence[ColumnInfo]) -> tuple[str, ...]:
    stamps = [column for column in columns if column.kind == "timestamp"]
    named = [column.name for column in stamps if _named_like(column.name, TIME_NAMES)]
    rest = [column.name for column in stamps if column.name not in named]
    return (*named, *rest)


def _targets(columns: Sequence[ColumnInfo]) -> tuple[str, ...]:
    tiers = {"0/1": 0, "bool": 1, "category": 2, "int": 3, "float": 3}
    ranked = []
    for index, column in enumerate(columns):
        tier = tiers.get(column.kind)
        if tier is None or column.unique or column.distinct < 2:
            continue
        if column.kind == "category" and column.distinct > CATEGORY_DISTINCT:
            continue
        bonus = 0 if _named_like(column.name, TARGET_NAMES) else 1
        ranked.append((tier, bonus, index, column.name))
    return tuple(name for _, _, _, name in sorted(ranked))


def whole_numbers(values: pa.ChunkedArray) -> pa.ChunkedArray | None:
    """A float column cast to int64 when every present value is a whole number.

    A CSV with blanks in an integer column often spells its values `1.0`; the
    classes a lock records and a server checks labels against stay integers.
    """
    if not pa.types.is_floating(values.type):
        return None
    present = pc.drop_null(values)
    present = present.filter(pc.invert(pc.is_nan(present)))
    if present.length() == 0:
        return None
    if not pc.all(pc.equal(present, pc.floor(present))).as_py():
        return None
    return pc.cast(values, pa.int64(), safe=False)


def labels(values: pa.ChunkedArray) -> pa.ChunkedArray:
    """A target column's class labels: its present values, whole floats as ints."""
    values = whole_numbers(values) or values
    return pc.drop_null(_blank_to_null(values))


def task_for(column: ColumnInfo, rows: int) -> str:
    """The task a target column calls for.

    Booleans, 0/1 flags, categories and text are classes. So is an int with few
    distinct values, by the rule that makes a text column a category: at most
    `CATEGORY_DISTINCT` of them, or at most `CATEGORY_SHARE` of the rows. Every
    other number is regressed.
    """
    if column.kind == "float":
        return "regression"
    if column.kind == "int":
        few = (
            column.distinct <= CATEGORY_DISTINCT
            or column.distinct <= CATEGORY_SHARE * rows
        )
        return "classification" if few else "regression"
    return "classification"


def split(frame: Any, spec: PredictorSpec) -> tuple[pa.Table, pa.Table]:
    """Divides snapshot rows into context and queries, by the one rule every path applies.

    Rows whose target is empty are always queries. A declared split then
    partitions the labeled rows: with `context.until`, labeled rows at or
    after that time are queries too; with `context.holdout`, a seeded random
    `fraction` of the labeled rows is. The context is every other labeled
    row. `gf check`, `gf.Predictor`, `gf eval`, `gf serve`, `gf push` and
    hosted serving all split through here, so they agree on every row.

    The holdout draw is `numpy.random.default_rng(seed).permutation` over the
    labeled rows in file order; its first `round(fraction * labeled)` rows
    are the queries.

    Raises:
        InvalidInput: If the target or time column is missing.
    """
    table = to_table(frame)
    assert spec.target is not None
    if spec.target not in table.column_names:
        raise spec.error(
            f"target column `{spec.target}` is not in the context",
            param="target",
            hint=f"the context has {', '.join(table.column_names)}",
        )
    return split_by(
        table,
        spec.target,
        time=spec.time,
        until=spec.context.until,
        holdout=spec.context.holdout,
        error=spec.error,
    )


def split_by(
    table: pa.Table,
    target: str,
    *,
    time: str | None = None,
    until: str | None = None,
    holdout: Any = None,
    error: Any = None,
) -> tuple[pa.Table, pa.Table]:
    """`split` from its settings alone, as a lock records them.

    Args:
        table: The snapshot rows.
        target: The target column.
        time: The time column `until` compares against.
        until: An ISO date or timestamp; labeled rows at or after it are queries.
        holdout: A `Holdout`, or a mapping with `fraction` and `seed`.
        error: Builds the `InvalidInput` for a missing time column, pointing
            at the manifest when there is one.
    """
    empty = pc.is_null(_blank_to_null(table[target]))
    labeled = pc.invert(empty)
    queried = empty
    if until is not None:
        queried = pc.or_(empty, _after(table, time, until, error))
    elif holdout is not None:
        fraction, seed = _holdout_of(holdout)
        rows = np.flatnonzero(labeled.to_numpy(zero_copy_only=False))
        order = np.random.default_rng(seed).permutation(len(rows))
        held = np.zeros(table.num_rows, dtype=bool)
        held[rows[order[: round(fraction * len(rows))]]] = True
        queried = pc.or_(empty, pa.array(held))
    queried = pc.fill_null(queried, False)
    return table.filter(pc.invert(queried)), table.filter(queried)


def _holdout_of(holdout: Any) -> tuple[float, int]:
    if isinstance(holdout, dict):
        return float(holdout["fraction"]), int(holdout.get("seed", 0))
    return float(holdout.fraction), int(holdout.seed)


def _after(
    table: pa.Table, time: str | None, value: str, error: Any
) -> pa.ChunkedArray:
    """Whether each row's time is at or after `until`, in UTC."""
    if time is None or time not in table.column_names:
        message = (
            f"time column `{time}` is not in the context"
            if time
            else "`until` needs a `time` column to compare against"
        )
        param = "time" if time else "context.until"
        hint = f"the context has {', '.join(table.column_names)}"
        if error is not None:
            raise error(message, param=param, hint=hint)
        raise InvalidInput(message=message, param=param, hint=hint)
    until = until_timestamp(value)
    assert until is not None
    if until.tzinfo is not None:
        until = until.astimezone(dt.UTC).replace(tzinfo=None)
    column = table[time]
    if pa.types.is_timestamp(column.type) and column.type.tz is not None:
        column = pc.cast(column, pa.timestamp("us", tz="UTC"))
        column = pc.cast(column, pa.timestamp("us"))
    try:
        stamps = pc.cast(column, pa.timestamp("us"))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        message = f"time column `{time}` has no timestamps"
        hint = "store times as ISO dates or timestamps"
        if error is not None:
            raise error(message, param="time", hint=hint) from None
        raise InvalidInput(message=message, param="time", hint=hint) from None
    return pc.fill_null(
        pc.greater_equal(stamps, pa.scalar(until, pa.timestamp("us"))), False
    )


__all__ = [
    "KINDS",
    "PHYSICAL",
    "ColumnInfo",
    "TableInfo",
    "inspect_frame",
    "inspect_table",
    "semantic",
    "split",
    "split_by",
]
