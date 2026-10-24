# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded local tabular conversion without semantic guesses or remote I/O."""

import io
import json
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as _compute
import pyarrow.json as arrow_json
from pyarrow import csv, parquet

from generalforecast.predictors.validation import PredictorError, bytes_digest, ensure

PHYSICAL_TYPES = frozenset(
    {"boolean", "int64", "float64", "string", "binary", "date", "timestamp"}
)
DECLARED_TYPES = PHYSICAL_TYPES | {"category", "numeric", "datetime"}

MAX_ROWS = 1_000_000
MAX_BYTES = 128 * 1024 * 1024
pc: Any = _compute
_ARROW_TYPES = {
    "boolean": pa.bool_(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "string": pa.string(),
    "binary": pa.binary(),
    "date": pa.date32(),
    "timestamp": pa.timestamp("us", tz="UTC"),
}


def _limits(max_rows: int, max_bytes: int) -> None:
    ensure(
        type(max_rows) is int
        and max_rows > 0
        and type(max_bytes) is int
        and max_bytes > 0,
        "row and byte limits must be positive integers",
    )


def _batch_table(
    batches: Iterable[pa.RecordBatch],
    schema: pa.Schema,
    *,
    max_rows: int,
    max_bytes: int,
) -> pa.Table:
    rows = size = 0
    accepted: list[pa.RecordBatch] = []
    for batch in batches:
        rows += batch.num_rows
        size += batch.nbytes
        ensure(
            rows <= max_rows and size <= max_bytes,
            "decoded source exceeds row or byte limit",
            "CapacityExceeded",
        )
        accepted.append(batch)
    return pa.Table.from_batches(accepted, schema=schema)


def _json_records(data: bytes, *, max_rows: int) -> list[dict[str, Any]]:
    text = data.decode("utf-8")
    decoder = json.JSONDecoder()
    position = 0

    def skip_space(index: int) -> int:
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    position = skip_space(position)
    ensure(
        position < len(text) and text[position] == "[",
        "JSON table must be an array of records",
    )
    position += 1
    records: list[dict[str, Any]] = []
    position = skip_space(position)
    if position < len(text) and text[position] == "]":
        position += 1
    else:
        while True:
            record, position = decoder.raw_decode(text, position)
            ensure(type(record) is dict, "JSON table must contain records")
            records.append(record)
            ensure(
                len(records) <= max_rows,
                "source exceeds row limit",
                "CapacityExceeded",
            )
            position = skip_space(position)
            ensure(position < len(text), "unterminated JSON array")
            if text[position] == "]":
                position += 1
                break
            ensure(text[position] == ",", "invalid JSON array separator")
            position = skip_space(position + 1)
    ensure(skip_space(position) == len(text), "trailing JSON content")
    return records


def physical_type(kind: pa.DataType) -> str:
    """Infers physical storage only; string is never guessed to mean category or ID."""
    if pa.types.is_boolean(kind):
        return "boolean"
    if pa.types.is_integer(kind):
        return "int64"
    if pa.types.is_floating(kind):
        return "float64"
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return "string"
    if pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
        return "binary"
    if pa.types.is_date(kind):
        return "date"
    if pa.types.is_timestamp(kind):
        return "timestamp"
    if pa.types.is_dictionary(kind):
        return physical_type(kind.value_type)
    raise PredictorError(
        "UnsupportedType",
        "nested, null, decimal and other unsupported physical types need explicit conversion",
    )


def coerce(table: pa.Table, physical_types: Mapping[str, str]) -> pa.Table:
    """Safely cast only explicitly declared columns without reordering or inference."""
    ensure(isinstance(table, pa.Table), "expected Arrow table", "InvalidTable")
    for name, kind in physical_types.items():
        ensure(
            name in table.column_names,
            "declared column is missing",
            "SchemaMismatch",
        )
        ensure(
            type(kind) is str and kind in PHYSICAL_TYPES,
            "unsupported physical type",
            "UnsupportedType",
        )
        index = table.column_names.index(name)
        try:
            values = pc.cast(table.column(index), _ARROW_TYPES[kind], safe=True)
        except (pa.ArrowException, TypeError, ValueError, OverflowError):
            raise PredictorError(
                "SchemaMismatch",
                "values cannot be represented by declared physical types",
            ) from None
        table = table.set_column(
            index, table.schema.field(index).with_type(_ARROW_TYPES[kind]), values
        )
    return table


def to_arrow(
    data: Any,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    max_rows: int = MAX_ROWS,
    max_bytes: int = MAX_BYTES,
) -> pa.Table:
    """Converts Arrow, pandas, Polars, NumPy or records with an explicit size ceiling.

    A Polars LazyFrame is limited before collection. Exceeding the bound fails;
    the bound is never a context selection policy.
    """
    _limits(max_rows, max_bytes)
    try:
        module = type(data).__module__.split(".")[0]
        if isinstance(data, pa.Table):
            table = data
        elif isinstance(data, pa.RecordBatch):
            table = pa.Table.from_batches([data])
        elif module == "polars":
            if type(data).__name__ == "LazyFrame":
                data.collect_schema()
                data = data.limit(max_rows + 1).collect(engine="streaming")
            ensure(
                hasattr(data, "height"),
                "Polars input must be a DataFrame or LazyFrame",
                "UnsupportedFormat",
            )
            ensure(
                data.height <= max_rows, "table exceeds row limit", "CapacityExceeded"
            )
            table = data.to_arrow()
        elif module == "pandas":
            ensure(
                getattr(data, "ndim", None) == 2,
                "pandas input must be a DataFrame",
                "UnsupportedFormat",
            )
            ensure(len(data) <= max_rows, "table exceeds row limit", "CapacityExceeded")
            table = pa.Table.from_pandas(data, preserve_index=False)
        elif isinstance(data, np.ndarray):
            ensure(
                data.ndim == 2 and len(data) <= max_rows,
                "NumPy input must be a bounded two-dimensional array",
            )
            labels = (
                list(columns)
                if columns is not None
                else [f"f{i}" for i in range(data.shape[1])]
            )
            ensure(
                len(labels) == data.shape[1], "column count differs from NumPy width"
            )
            ensure(len(set(labels)) == len(labels), "NumPy columns must be unique")
            table = pa.table({column: data[:, i] for i, column in enumerate(labels)})
        elif type(data) is list:
            ensure(
                len(data) <= max_rows and all(isinstance(row, Mapping) for row in data),
                "records must be a bounded list of objects",
            )
            keys = list(dict.fromkeys(key for row in data for key in row))
            table = pa.Table.from_pylist(
                [dict(row) for row in data],
                schema=pa.schema(
                    [
                        pa.field(key, pa.array([row.get(key) for row in data]).type)
                        for key in keys
                    ]
                ),
            )
        elif isinstance(data, Mapping):
            ensure(
                all(
                    hasattr(value, "__len__") and len(value) <= max_rows
                    for value in data.values()
                ),
                "columns exceed row limit",
            )
            table = pa.table(dict(data))
        else:
            raise PredictorError(
                "UnsupportedFormat", "expected Arrow, pandas, Polars, NumPy or records"
            )
        ensure(
            table.num_rows <= max_rows and table.nbytes <= max_bytes,
            "table exceeds row or byte limit",
            "CapacityExceeded",
        )
        ensure(
            table.num_columns > 0 and len(set(table.column_names)) == table.num_columns,
            "columns must be nonempty and unique",
        )
        ensure(
            all(
                type(column) is str and column and len(column) <= 512
                for column in table.column_names
            ),
            "invalid column names",
        )
        arrays = []
        for field, array in zip(table.schema, table.columns, strict=True):
            kind = physical_type(field.type)
            if pa.types.is_dictionary(field.type):
                array = pc.cast(array, field.type.value_type)
            array = pc.cast(array, _ARROW_TYPES[kind], safe=True)
            if kind == "float64":
                ensure(
                    not pc.any(pc.is_inf(array)).as_py(),
                    "infinite values are not supported",
                )
                array = pc.if_else(pc.is_nan(array), None, array)
            arrays.append(array)
        return pa.table(arrays, names=table.column_names).combine_chunks()
    except PredictorError:
        raise
    except (pa.ArrowException, TypeError, ValueError, OverflowError):
        raise PredictorError(
            "InvalidTable",
            "tabular conversion failed; convert unsupported or mixed columns explicitly",
        ) from None


def read_table(
    data: bytes, format: str, *, max_rows: int = MAX_ROWS, max_bytes: int = MAX_BYTES
) -> pa.Table:
    """Parses local bytes with incremental row and decoded-byte checks.

    A single Arrow batch (or JSON record) can still allocate before its size is
    checked. This is a budget, not isolation of the underlying decoder.
    """
    _limits(max_rows, max_bytes)
    ensure(len(data) <= max_bytes, "source exceeds byte limit", "CapacityExceeded")
    format = format.lower().lstrip(".")
    stream = pa.BufferReader(data)
    try:
        if format in ("csv", "tsv"):
            source = csv.open_csv(
                stream,
                read_options=csv.ReadOptions(block_size=min(max_bytes, 1 << 20)),
                parse_options=csv.ParseOptions(
                    delimiter="\t" if format == "tsv" else ","
                ),
            )
            table = _batch_table(
                source, source.schema, max_rows=max_rows, max_bytes=max_bytes
            )
        elif format in ("parquet", "pq"):
            source = parquet.ParquetFile(stream)
            ensure(
                source.metadata.num_rows <= max_rows,
                "source exceeds row limit",
                "CapacityExceeded",
            )
            ensure(
                sum(
                    source.metadata.row_group(i).total_byte_size
                    for i in range(source.metadata.num_row_groups)
                )
                <= max_bytes,
                "uncompressed Parquet source exceeds byte limit",
                "CapacityExceeded",
            )
            table = _batch_table(
                source.iter_batches(batch_size=min(max_rows, 65536), use_threads=False),
                source.schema_arrow,
                max_rows=max_rows,
                max_bytes=max_bytes,
            )
        elif format in ("arrow", "ipc", "feather"):
            if format == "feather":
                ensure(
                    not data.startswith(b"FEA1"),
                    "Feather V1 cannot be read in bounded batches",
                    "UnsupportedFormat",
                )
            try:
                source = pa.ipc.open_stream(stream) if format != "feather" else None
            except pa.ArrowInvalid:
                source = None
            if source is not None:
                batches = iter(source)
            else:
                stream.seek(0)
                source = pa.ipc.open_file(stream)
                batches = (
                    source.get_batch(i) for i in range(source.num_record_batches)
                )
            table = _batch_table(
                batches, source.schema, max_rows=max_rows, max_bytes=max_bytes
            )
        elif format == "json":
            records = _json_records(data, max_rows=max_rows)
            return to_arrow(records, max_rows=max_rows, max_bytes=max_bytes)
        elif format in ("jsonl", "ndjson"):
            source = arrow_json.open_json(
                stream,
                read_options=arrow_json.ReadOptions(block_size=min(max_bytes, 1 << 20)),
            )
            table = _batch_table(
                source, source.schema, max_rows=max_rows, max_bytes=max_bytes
            )
        else:
            raise PredictorError(
                "UnsupportedFormat",
                "supported files are CSV, TSV, Parquet, Arrow, Feather, JSON and JSONL",
            )
        return to_arrow(table, max_rows=max_rows, max_bytes=max_bytes)
    except PredictorError:
        raise
    except (pa.ArrowException, ValueError, UnicodeError):
        raise PredictorError(
            "InvalidTable", "source bytes do not match a supported table format"
        ) from None


def file_format(path: str) -> str:
    """Gets a format from the local filename, never probes a remote URL."""
    return PurePosixPath(path).suffix.lstrip(".").lower()


def arrow_bytes(data: Any) -> bytes:
    """Produces deterministic normalized IPC bytes, excluding dataframe metadata."""
    table = to_arrow(data)
    output = io.BytesIO()
    with pa.ipc.new_stream(output, table.schema.remove_metadata()) as writer:
        writer.write_table(table, max_chunksize=MAX_ROWS)
    return output.getvalue()


def table_digest(data: Any) -> str:
    """Hashes normalized table content separately from source container bytes."""
    return bytes_digest(arrow_bytes(data))


_FRAME_FORMATS = {
    "csv": "csv",
    "tsv": "tsv",
    "parquet": "parquet",
    "pq": "parquet",
    "arrow": "arrow",
    "feather": "feather",
    "json": "json",
    "jsonl": "jsonl",
    "ndjson": "ndjson",
}


def frame(obj: Any, *, columns: list[str] | tuple[str, ...] | None = None) -> pa.Table:
    """Rows as an Arrow table, however the caller holds them.

    Accepts a pandas or Polars frame, a list of dicts, a dict of columns, an
    Arrow table or record batch, a NumPy array (columns `columns`, else
    `f0`, `f1`, ...), or a path or URL to a `.csv`, `.parquet`, `.json`,
    `.jsonl`, `.arrow` or `.feather` file. Every public method that takes
    `queries` or `context` reads them through this function.

    Raises:
        InvalidInput: For a value that is not rows, a file that cannot be read,
            or rows past the size limits.
    """
    from generalforecast.errors import InvalidInput, boundary

    if isinstance(obj, (str, PurePosixPath)) or hasattr(obj, "__fspath__"):
        return _frame_from_location(str(obj))
    if obj is None:
        raise InvalidInput("rows are required: a frame, a list of dicts or a file")
    if isinstance(obj, Mapping):
        with boundary():
            return pa.Table.from_pydict(dict(obj))
    if isinstance(obj, (list, tuple)) and len(obj) == 0:
        return pa.table({})
    with boundary():
        return to_arrow(obj, columns=columns)


def _frame_from_location(location: str) -> pa.Table:
    from urllib.parse import urlsplit

    from generalforecast.errors import InvalidInput, boundary

    parts = urlsplit(location)
    if parts.scheme in ("http", "https"):
        import urllib.request

        try:
            with urllib.request.urlopen(location, timeout=60) as response:
                data = response.read(MAX_BYTES + 1)
        except OSError as exc:
            raise InvalidInput(f"cannot read {location}: {exc}".rstrip(".")) from None
        name = parts.path
    else:
        import os

        path = os.path.expanduser(location)
        if not os.path.isfile(path):
            raise InvalidInput(f"no file at `{location}`")
        with open(path, "rb") as handle:
            data = handle.read(MAX_BYTES + 1)
        name = path
    suffix = PurePosixPath(name).suffix.lower().lstrip(".")
    if suffix not in _FRAME_FORMATS:
        raise InvalidInput(
            f"`{location}` is not a file this reads",
            hint="pass a .csv, .parquet, .json, .jsonl, .arrow or .feather file",
        )
    if len(data) > MAX_BYTES:
        raise InvalidInput(
            f"`{location}` is larger than {MAX_BYTES // (1024 * 1024)} MiB",
            limit=MAX_BYTES,
            actual=len(data),
        )
    with boundary():
        return read_table(data, _FRAME_FORMATS[suffix])
