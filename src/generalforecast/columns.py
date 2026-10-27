# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Feature columns: their kinds, and how a frame's values become the numbers models read.

Each of a table's columns has a kind. Numeric columns are read as float32, with
NaN for missing values. Boolean columns become 0 or 1. Datetime columns take
dates and times, as datetime64, pandas or Arrow timestamps, Python dates or
ISO 8601 strings, with time zones read as UTC, kept to the millisecond; plain
numbers are refused, since nothing says whether they count seconds or days.
Each becomes five model features, as tabular libraries split dates: days since
1970-01-01 for the trend, then the hour of the day with its fraction, the day of
the week from Monday as 0, the day of the month and the month. One float32 of
days alone cannot tell apart times minutes apart today; the hour of the day
keeps them to well under a second. Categorical columns hold strings, which the
table's vocabulary codes in the order it first saw them, or numbers that are
codes already; a string a query brings that the table has not seen is missing.
A column holds at most `MAX_WORDS` categories, each word at most
`MAX_WORD_BYTES` long, so codes stay exact in float32 and a vocabulary cannot
grow without bound. Retrieval columns are numbers that place a row for
retrieval, such as an embedding, and that the model never reads; they follow
every other column. Servers encode the same way, so a table answers the same
wherever it lives.

Rows may be a NumPy array, a pandas or polars frame, an Arrow table, or a
mapping of column names to values. Frames are read by column name; one without
the table's names but with as many columns is read in order. `infer` reads a
frame's column kinds from its types, so a table can be made like a frame.
"""

import datetime
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

import numpy as np

ColumnKind = Literal["numeric", "categorical", "boolean", "datetime", "retrieval"]
KINDS: tuple[ColumnKind, ...] = get_args(ColumnKind)

# Each Arrow storage's encoding, as `tabular.physical_type` names storages.
ENCODINGS: dict[str, ColumnKind] = {
    "int64": "numeric",
    "float64": "numeric",
    "boolean": "boolean",
    "string": "categorical",
    "date": "datetime",
    "timestamp": "datetime",
}
# Each column kind's encoding, in the kinds `gf.yaml` and `gf.lock` name
# (`generalforecast.predictors.tables`).
KIND_ENCODINGS: dict[str, ColumnKind] = {
    "int": "numeric",
    "float": "numeric",
    "0/1": "numeric",
    "category": "categorical",
    "text": "categorical",
    "timestamp": "datetime",
    "bool": "boolean",
}

# Names the fixed columns of writes and queries take, so no feature column may.
RESERVED = frozenset({"key", "target", "time", "before"})

_MILLIS_PER_DAY = 86_400_000
_MILLIS_PER_HOUR = 3_600_000.0

# The model features a datetime column becomes: days since 1970-01-01, hour of
# the day, day of the week, day of the month, and month.
DATETIME_PARTS = 5

# The most categories one column holds, and the most UTF-8 bytes in one word.
# Float32 holds every whole number up to 2**24 exactly, so codes stay exact.
MAX_WORDS = 1 << 24
MAX_WORD_BYTES = 1024


@dataclass(frozen=True)
class Column:
    """A feature column.

    Attributes:
        name: The column's name.
        kind: One of `numeric`, `categorical`, `boolean`, `datetime` or
            `retrieval`.
    """

    name: str
    kind: ColumnKind = "numeric"

    def __post_init__(self) -> None:
        """Refuses a kind that is not one of `KINDS`, or an empty name.

        Raises:
            ValueError: If the kind or the name is not one.
        """
        column_kind(self.name, self.kind)
        if not self.name:
            raise ValueError("every column needs a name")


def column_kind(name: str, kind: object) -> ColumnKind:
    """A column's kind, checked.

    Args:
        name: The column's name, for the error.
        kind: The kind given, as a string.

    Returns:
        The kind.

    Raises:
        ValueError: If it is not one of `KINDS`.
    """
    if kind not in KINDS:
        raise ValueError(
            f"column {name!r} has kind {kind!r}; kinds are " + ", ".join(KINDS)
        )
    return cast(ColumnKind, kind)


def columns_of(features: int | None, columns: Any) -> tuple[Column, ...]:
    """A table's columns from a feature count, names with their kinds, or a frame.

    Args:
        features: Number of numeric columns, named `f0 ..`.
        columns: Column names with their kinds, names alone for numeric
            columns, or rows whose column kinds `infer` reads: a pandas or
            polars frame, an Arrow table, or a mapping of names to values.

    Returns:
        The columns, checked.

    Raises:
        ValueError: If neither or both are given, a kind is unknown, a name
            is empty, repeated, or that of a fixed column, or a retrieval
            column comes before another or the columns are all retrieval ones.
    """
    if (features is None) == (columns is None):
        raise ValueError("give a table its number of features, or its columns")
    if columns is not None and not _names_or_kinds(columns):
        columns = infer(columns)
    if columns is None:
        if features is None or features < 1:
            raise ValueError("a table needs at least one feature column")
        return tuple(Column(f"f{index}") for index in range(features))
    if isinstance(columns, Mapping):
        made = tuple(
            Column(str(name), column_kind(str(name), kind))
            for name, kind in columns.items()
        )
    else:
        made = tuple(Column(str(name)) for name in columns)
    if not made:
        raise ValueError("a table needs at least one feature column")
    for column in made:
        if column.name in RESERVED:
            raise ValueError(
                f"{column.name!r} names a fixed column; feature columns need other names"
            )
    names = [column.name for column in made]
    if len(set(names)) != len(names):
        raise ValueError("every column needs a name of its own")
    read = len(made) - retrieval(made)
    if not read or any(column.kind != "retrieval" for column in made[read:]):
        raise ValueError(
            "retrieval columns come after every other column, and a table needs one other"
        )
    return made


def retrieval(columns: Sequence[Column]) -> int:
    """How many retrieval columns there are, which end the columns."""
    return sum(column.kind == "retrieval" for column in columns)


def _names_or_kinds(columns: Any) -> bool:
    """Whether `columns` names columns, with or without kinds, rather than holding rows."""
    if isinstance(columns, Mapping):
        return all(isinstance(kind, str) for kind in columns.values())
    return isinstance(columns, Sequence) and all(
        isinstance(name, str) for name in columns
    )


def infer(rows: Any) -> dict[str, ColumnKind]:
    """Each column's kind, read from a frame's column types.

    Strings and categories are categorical, booleans boolean, dates and times
    datetime, and numbers numeric. A column of Python objects takes the kind
    of its first value that is not missing.

    Args:
        rows: A pandas or polars frame, an Arrow table, or a mapping of column
            names to values.

    Returns:
        Each column's kind, by name, in the frame's order.

    Raises:
        ValueError: If a column's type is none of these, such as durations.
    """
    if hasattr(rows, "schema") and hasattr(rows, "column_names"):
        return {field.name: _arrow_kind(field) for field in rows.schema}
    if hasattr(rows, "schema") and hasattr(rows, "get_column"):
        return {
            str(name): _polars_kind(str(name), dtype)
            for name, dtype in rows.schema.items()
        }
    if hasattr(rows, "dtypes") and hasattr(rows, "columns"):
        return {
            str(name): _values_kind(str(name), rows[name].dtype, rows[name])
            for name in rows.columns
        }
    if isinstance(rows, Mapping):
        return {
            str(name): _values_kind(str(name), np.asarray(values).dtype, values)
            for name, values in rows.items()
        }
    raise ValueError(
        "column kinds come from a frame, an Arrow table or a mapping of columns; "
        f"give a {type(rows).__name__} its columns by name"
    )


def _arrow_kind(field: Any) -> ColumnKind:
    import pyarrow as pa

    kind = field.type
    if pa.types.is_dictionary(kind):
        return "categorical"
    if pa.types.is_large_string(kind):
        kind = pa.string()
    storage = {
        pa.types.is_string: "string",
        pa.types.is_boolean: "boolean",
        pa.types.is_timestamp: "timestamp",
        pa.types.is_date: "date",
        pa.types.is_integer: "int64",
        pa.types.is_floating: "float64",
    }
    for test, physical in storage.items():
        if test(kind):
            return ENCODINGS[physical]
    raise ValueError(f"column {field.name!r} is {kind}, which no column kind holds")


def _polars_kind(name: str, dtype: Any) -> ColumnKind:
    text = str(dtype)
    if text.startswith(("String", "Utf8", "Categorical", "Enum")):
        return "categorical"
    if text == "Boolean":
        return "boolean"
    if text.startswith(("Datetime", "Date")):
        return "datetime"
    if dtype.is_numeric():
        return "numeric"
    raise ValueError(f"column {name!r} is {text}, which no column kind holds")


def _values_kind(name: str, dtype: Any, values: Any) -> ColumnKind:
    """A pandas or NumPy column's kind, from its dtype or its first value."""
    if getattr(dtype, "name", "") == "category":
        return "categorical"
    if dtype.kind == "b":
        return "boolean"
    if dtype.kind == "M":
        return "datetime"
    if dtype.kind in "iuf":
        return "numeric"
    if dtype.kind in "US":
        return "categorical"
    for value in np.asarray(values, dtype=object).ravel():
        if _missing(value):
            continue
        if isinstance(value, str | bytes):
            return "categorical"
        if isinstance(value, bool | np.bool_):
            return "boolean"
        if isinstance(value, datetime.date | np.datetime64):
            return "datetime"
        if isinstance(value, int | float | np.number):
            return "numeric"
        break
    raise ValueError(f"column {name!r} holds values no column kind holds")


def _missing(value: Any) -> bool:
    """Whether a value stands for a missing one."""
    if value is None or type(value).__name__ == "NAType":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, np.datetime64) and np.isnat(value)


def _values(rows: Any, columns: Sequence[Column]) -> tuple[list[Any], int]:
    """Each column's values and the row count, from a frame or an array."""
    names = [column.name for column in columns]
    if hasattr(rows, "column_names") and hasattr(rows, "column"):
        held, pick = list(rows.column_names), _arrow_column
    elif hasattr(rows, "get_column") and hasattr(rows, "columns"):
        held, pick = list(rows.columns), lambda frame, name: frame.get_column(name)
    elif hasattr(rows, "iloc") and hasattr(rows, "columns"):
        held, pick = list(rows.columns), lambda frame, name: frame[name].to_numpy()
    elif isinstance(rows, Mapping):
        held, pick = list(rows), lambda frame, name: frame[name]
    else:
        matrix = rows if isinstance(rows, np.ndarray) else np.array(rows)
        if matrix.dtype.kind in "US" and not isinstance(rows, np.ndarray):
            # NumPy turns rows mixing numbers and strings into strings alone.
            matrix = np.array(rows, dtype=object)
        if matrix.ndim != 2:
            raise ValueError(
                f"rows are a 2-D array, one row each, not shape {matrix.shape}; "
                "for one row, use reshape(1, -1)"
            )
        if matrix.shape[1] != len(columns):
            placing = [column.name for column in columns if column.kind == "retrieval"]
            lack = (
                f"; rows and queries carry its retrieval columns {placing} too"
                if placing and matrix.shape[1] == len(columns) - len(placing)
                else ""
            )
            raise ValueError(
                f"rows have {matrix.shape[1]} columns, but the table has "
                f"{len(columns)}{lack}"
            )
        return [matrix[:, index] for index in range(len(columns))], len(matrix)
    missing = [name for name in names if name not in held]
    # Labels that are not names, such as a pandas frame's default 0, 1, ...,
    # are read in order; names that differ from the table's are refused.
    unnamed = not any(isinstance(label, str) for label in held)
    if missing and unnamed and len(held) == len(columns):
        values = [pick(rows, label) for label in held]
    elif missing:
        raise ValueError(f"rows lack the columns {missing}; they have {held}")
    else:
        values = [pick(rows, name) for name in names]
    counts = {len(column) for column in values}
    if len(counts) > 1:
        raise ValueError("every column needs one value per row")
    return values, counts.pop() if counts else 0


def pick(rows: Any, names: Sequence[str]) -> tuple[list[Any], int]:
    """Named columns of a frame, as given, and the row count.

    Args:
        rows: A frame, Arrow table or mapping of columns.
        names: The columns to pick.

    Returns:
        Each column's values, and the number of rows.

    Raises:
        ValueError: If a column is missing.
    """
    return _values(rows, [Column(name) for name in names])


def _arrow_column(table: Any, name: str) -> np.ndarray:
    """An Arrow column as a NumPy array, with categories as their values."""
    import pyarrow as pa

    column = table.column(name)
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if pa.types.is_dictionary(column.type):
        column = column.dictionary_decode()
    return column.to_numpy(zero_copy_only=False)


def _objects(name: str, values: np.ndarray, allow: str) -> np.ndarray:
    """An object column as float32: missing values are NaN, and strings an error."""
    numbers = np.empty(len(values), dtype=np.float32)
    for index, value in enumerate(values):
        if _missing(value):
            numbers[index] = np.nan
        elif isinstance(value, str | bytes):
            raise ValueError(f"column {name!r} is {allow}; it cannot hold strings")
        else:
            numbers[index] = value
    return numbers


def _numbers(name: str, values: Any, kind: str) -> np.ndarray:
    """A numeric or boolean column as float32."""
    array = np.asarray(values)
    if array.dtype.kind in "biuf":
        return array.astype(np.float32)
    if array.dtype.kind == "M":
        raise ValueError(f"column {name!r} is {kind}; it holds times")
    if array.dtype.kind in "US":
        raise ValueError(f"column {name!r} is {kind}; it cannot hold strings")
    return _objects(name, array, kind)


# The datetime64 units dates and times are held in.
Unit = Literal["D", "s", "ms", "us", "ns"]


def _moment(value: Any, unit: Unit) -> np.datetime64:
    """One date, time or ISO 8601 string as a datetime64 of `unit`, in UTC."""
    if isinstance(value, datetime.datetime) and value.tzinfo is not None:
        value = value.astimezone(datetime.UTC).replace(tzinfo=None)
    return np.datetime64(value, unit)


def moments(name: str, values: Any, unit: Unit = "ms") -> np.ndarray:
    """Dates and times as datetime64 of `unit`, NaT for missing.

    Args:
        name: What the values are, for errors.
        values: datetime64 of any unit, pandas or Arrow timestamps, Python
            dates and datetimes, or ISO 8601 strings.
        unit: The datetime64 unit to return, such as `ms` or `us`.

    Returns:
        The values as datetime64 of `unit`.

    Raises:
        ValueError: If a value is a plain number, or not a date or time.
    """
    array = np.asarray(values)
    if array.dtype.kind == "M" or array.size == 0:
        return array.astype(f"datetime64[{unit}]")
    if array.dtype.kind in "biuf":
        raise ValueError(_numbers_refused(name, array.ravel()[:1].tolist()))
    moments = np.empty(array.shape, dtype=f"datetime64[{unit}]")
    for index, value in enumerate(array.ravel()):
        if _missing(value):
            moments.flat[index] = np.datetime64("NaT", unit)
            continue
        if isinstance(value, int | float | np.number) and not isinstance(value, bool):
            # A wrong value for the column, as other columns' errors are.
            raise ValueError(_numbers_refused(name, [value]))  # noqa: TRY004
        try:
            text = value.decode() if isinstance(value, bytes) else value
            moments.flat[index] = _moment(text, unit)
        except (ValueError, TypeError):
            raise ValueError(
                f"{name} takes dates, times or ISO 8601 strings, not {value!r}"
            ) from None
    return moments


def _numbers_refused(name: str, values: list[Any]) -> str:
    """Why plain numbers are not dates or times."""
    shown = f" such as {values[0]!r}" if values else ""
    return (
        f"{name} takes dates and times, not plain numbers{shown}, which could "
        "count days, seconds or milliseconds; convert them first, for example "
        "with np.datetime64(seconds, 's') or pd.to_datetime(seconds, unit='s')"
    )


def datetime_parts(moments: np.ndarray) -> np.ndarray:
    """Millisecond datetime64 as their model features, NaN for NaT.

    Args:
        moments: Times as datetime64[ms], in UTC.

    Returns:
        A float32 matrix with a row per time: days since 1970-01-01, hour of the
        day with its fraction, day of the week from Monday as 0, day of the
        month, and month.
    """
    moments = moments.astype("datetime64[ms]")
    millis = moments.astype(np.int64)
    day = np.floor_divide(millis, _MILLIS_PER_DAY)
    month = moments.astype("datetime64[M]")
    parts = np.column_stack(
        [
            millis.astype(np.float64) / _MILLIS_PER_DAY,
            np.mod(millis, _MILLIS_PER_DAY).astype(np.float64) / _MILLIS_PER_HOUR,
            # 1970-01-01 was a Thursday.
            np.mod(day + 3, 7),
            (moments.astype("datetime64[D]") - month).astype(np.int64) + 1,
            np.mod(month.astype(np.int64), 12) + 1,
        ]
    ).astype(np.float32)
    parts[np.isnat(moments)] = np.nan
    return parts.reshape(len(moments), DATETIME_PARTS)


def width(columns: Sequence[Column]) -> int:
    """How many values of a row the columns become: more than one per datetime."""
    return sum(_width(column) for column in columns)


def offsets(columns: Sequence[Column]) -> list[int]:
    """The model feature each column's features start at."""
    starts, start = [], 0
    for column in columns:
        starts.append(start)
        start += _width(column)
    return starts


def categorical(columns: Sequence[Column]) -> list[int]:
    """The model features that hold category codes."""
    return [
        start
        for column, start in zip(columns, offsets(columns), strict=True)
        if column.kind == "categorical"
    ]


def _width(column: Column) -> int:
    return DATETIME_PARTS if column.kind == "datetime" else 1


def _codes(name: str, values: Any) -> np.ndarray | list[str | None]:
    """A categorical column: strings, with None for missing, or codes as float32."""
    array = np.asarray(values)
    if array.dtype.kind in "biuf":
        codes = array.astype(np.float32)
    elif array.dtype.kind == "U":
        return [str(value) for value in array]
    elif array.dtype.kind == "S":
        return [value.decode() for value in array.tolist()]
    elif array.dtype.kind == "M":
        raise ValueError(f"column {name!r} is categorical; it holds times")
    else:
        words: list[str | None] = []
        strings = False
        for value in array:
            if _missing(value):
                words.append(None)
            elif isinstance(value, str):
                words.append(value)
                strings = True
            elif isinstance(value, bytes):
                words.append(value.decode())
                strings = True
            else:
                words.append(value)
        if strings:
            if any(word is not None and not isinstance(word, str) for word in words):
                raise ValueError(f"column {name!r} mixes strings and numbers")
            return words
        codes = _objects(name, array, "categorical")
    finite = codes[np.isfinite(codes)]
    if (
        (finite < 0).any()
        or (finite >= MAX_WORDS).any()
        or (finite != np.round(finite)).any()
    ):
        raise ValueError(
            f"column {name!r} is categorical; its numbers are codes, whole numbers "
            f"from 0 below {MAX_WORDS}, or strings"
        )
    return codes


Prepared = list[np.ndarray | list[str | None]]


def prepare(rows: Any, columns: Sequence[Column]) -> tuple[Prepared, int]:
    """Each column's values: float32, strings if categorical, or datetime64 if datetime.

    Args:
        rows: A NumPy array, frame, Arrow table or mapping of columns.
        columns: The table's columns.

    Returns:
        One entry per column, and the number of rows.

    Raises:
        ValueError: If a column is missing, or holds values of another kind.
    """
    values, count = _values(rows, columns)
    prepared: Prepared = []
    for column, held in zip(columns, values, strict=True):
        if column.kind == "categorical":
            prepared.append(_codes(column.name, held))
        elif column.kind == "datetime":
            prepared.append(moments(f"column {column.name!r}", held))
        else:
            prepared.append(_numbers(column.name, held, column.kind))
    return prepared, count


def encode(
    prepared: Prepared,
    columns: Sequence[Column],
    vocabularies: Sequence[dict[str, int]],
    assign: bool,
) -> tuple[np.ndarray, dict[int, list[str]]]:
    """Prepared columns as one float32 matrix of model features, coding strings.

    Args:
        prepared: Columns from `prepare`.
        columns: The table's columns.
        vocabularies: Each column's codes by word; empty but for categorical columns.
        assign: Give words not held yet the next codes, rather than NaN.

    Returns:
        The matrix, and the words that took new codes, by column, in code order.
        They are not in the vocabularies yet; add them once the rows are in.
    """
    count = len(prepared[0]) if prepared else 0
    matrix = np.empty((count, width(columns)), dtype=np.float32)
    added: dict[int, list[str]] = {}
    for index, (values, start) in enumerate(
        zip(prepared, offsets(columns), strict=True)
    ):
        if isinstance(values, np.ndarray) and values.dtype.kind == "M":
            matrix[:, start : start + DATETIME_PARTS] = datetime_parts(values)
            continue
        if isinstance(values, np.ndarray):
            matrix[:, start] = values
            continue
        vocabulary, new = vocabularies[index], {}
        codes: list[float] = []
        for word in values:
            if word is None:
                codes.append(np.nan)
                continue
            code = vocabulary.get(word, new.get(word))
            if code is None and assign:
                _check_word(columns[index].name, word, len(vocabulary) + len(new))
                code = new[word] = len(vocabulary) + len(new)
            codes.append(np.nan if code is None else code)
        matrix[:, start] = codes
        if new:
            added[index] = list(new)
    return matrix, added


def _check_word(name: str, word: str, code: int) -> None:
    """Refuses a new word past the length or the count a column holds."""
    if len(word.encode()) > MAX_WORD_BYTES:
        raise ValueError(
            f"column {name!r}: a category is at most {MAX_WORD_BYTES} bytes, "
            f"not {len(word.encode())}"
        )
    if code >= MAX_WORDS:
        raise ValueError(
            f"column {name!r}: a column holds at most {MAX_WORDS} categories"
        )


def arrow_columns(prepared: Prepared, columns: Sequence[Column]) -> dict[str, Any]:
    """Prepared columns as Arrow arrays, with strings and times for a server to code."""
    import pyarrow as pa

    arrays = {}
    for column, values in zip(columns, prepared, strict=True):
        if isinstance(values, np.ndarray) and values.dtype.kind == "M":
            arrays[column.name] = pa.array(values, pa.timestamp("ms"))
        elif isinstance(values, np.ndarray):
            arrays[column.name] = pa.array(values, pa.float32())
        else:
            arrays[column.name] = pa.array(values, pa.string())
    return arrays
