# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import datetime
import io
import json
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import feather, parquet

from generalforecast.packages import PackageError
from generalforecast.packages.schema import TableSchema
from generalforecast.packages.spec import PHYSICAL_TYPES
from generalforecast.tabular import (
    arrow_bytes,
    coerce,
    physical_type,
    read_table,
    table_digest,
    to_arrow,
)


@pytest.mark.parametrize(
    "kind,value",
    [
        ("boolean", True),
        ("int64", 1),
        ("float64", 1.0),
        ("string", "unchanged"),
        ("binary", b"unchanged"),
        ("date", "2024-02-29"),
        ("timestamp", "2024-02-29T01:30:12.123456+02:00"),
    ],
)
def test_explicit_physical_coercion_uses_the_portable_vocabulary(kind, value):
    source = pa.table({"x": [value], "unchanged": ["1"]})
    result = coerce(source, {"x": kind})
    assert kind in PHYSICAL_TYPES
    assert physical_type(result["x"].type) == kind
    assert result.column_names == source.column_names
    assert result["unchanged"].equals(source["unchanged"])
    if kind == "timestamp":
        assert result["x"][0].as_py() == datetime.datetime(
            2024, 2, 28, 23, 30, 12, 123456, tzinfo=datetime.UTC
        )


def test_coercion_accepts_chunks_and_preserves_nulls_and_field_metadata():
    field = pa.field("date", pa.string(), metadata={b"source": b"test"})
    source = pa.Table.from_arrays(
        [pa.chunked_array([["2024-02-29"], [None]])], schema=pa.schema([field])
    )
    result = coerce(source, {"date": "date"})
    assert result.schema.field("date").metadata == field.metadata
    assert result["date"].to_pylist() == [datetime.date(2024, 2, 29), None]
    assert source.schema.field("date").type == pa.string()


@pytest.mark.parametrize(
    "source,types,code",
    [
        ({"x": ["private-invalid-date"]}, {"x": "date"}, "SchemaMismatch"),
        ({"x": ["2024-01-01T00:00:00"]}, {"x": "timestamp"}, "SchemaMismatch"),
        ({"x": [1.5]}, {"x": "int64"}, "SchemaMismatch"),
        ({"x": [1]}, {"missing": "int64"}, "SchemaMismatch"),
        ({"x": [1]}, {"x": "category"}, "UnsupportedType"),
    ],
)
def test_explicit_coercion_rejects_lossy_or_unsupported_values(source, types, code):
    with pytest.raises(PackageError) as caught:
        coerce(pa.table(source), types)
    assert caught.value.code == code
    assert "private-invalid-date" not in str(caught.value)


@pytest.mark.parametrize("kind", [["int64"], {"type": "int64"}, None, 1, True])
def test_locked_physical_types_reject_malformed_values_as_package_errors(kind):
    with pytest.raises(PackageError):
        TableSchema.from_dict(
            {
                "columns": [
                    {
                        "name": "x",
                        "type": kind,
                        "role": "input",
                        "nullable": False,
                        "semantic": None,
                    }
                ],
                "rows": 1,
                "classes": [],
                "frequency": None,
            }
        )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_limits_are_positive_integer_budgets(limit):
    for convert in (
        lambda **options: read_table(b"x\n1\n", "csv", **options),
        lambda **options: to_arrow(pa.table({"x": [1]}), **options),
    ):
        for name in ("max_rows", "max_bytes"):
            with pytest.raises(PackageError) as error:
                convert(**{name: limit})
            assert error.value.code == "InvalidPackage"


@pytest.mark.parametrize(
    "format", ["csv", "jsonl", "arrow", "ipc", "feather", "parquet"]
)
def test_batch_readers_stop_without_collecting_whole_source(monkeypatch, format):
    oversized = pa.record_batch({"x": ["a" * 300]})

    class Streaming:
        schema = oversized.schema

        def __iter__(self):
            yield oversized
            pytest.fail("read past the first over-budget batch")

        def read_all(self):
            pytest.fail("collected the complete source")

    class Indexed(Streaming):
        num_record_batches = 2

        def get_batch(self, index):
            assert index == 0, "read past the first over-budget batch"
            return oversized

    if format == "csv":
        monkeypatch.setattr(
            "generalforecast.tabular.csv.open_csv", lambda *a, **kw: Streaming()
        )
    elif format == "jsonl":
        monkeypatch.setattr(
            "generalforecast.tabular.arrow_json.open_json", lambda *a, **kw: Streaming()
        )
    elif format == "arrow":
        monkeypatch.setattr(
            "generalforecast.tabular.pa.ipc.open_stream", lambda *a, **kw: Streaming()
        )
    elif format in ("ipc", "feather"):
        if format == "ipc":

            def not_a_stream(*args, **kwargs):
                raise pa.ArrowInvalid("not a stream")

            monkeypatch.setattr(
                "generalforecast.tabular.pa.ipc.open_stream", not_a_stream
            )
        monkeypatch.setattr(
            "generalforecast.tabular.pa.ipc.open_file", lambda *a, **kw: Indexed()
        )
    else:

        class Metadata:
            num_rows = 1
            num_row_groups = 1

            def row_group(self, index):
                assert index == 0
                return type("Group", (), {"total_byte_size": 100})()

        class ParquetSource(Streaming):
            metadata = Metadata()
            schema_arrow = oversized.schema

            def iter_batches(self, **kwargs):
                return iter(self)

        monkeypatch.setattr(
            "generalforecast.tabular.parquet.ParquetFile",
            lambda *a, **kw: ParquetSource(),
        )

    with pytest.raises(PackageError) as error:
        read_table(b"small", format, max_bytes=256)
    assert error.value.code == "CapacityExceeded"


def test_parquet_rejects_uncompressed_metadata_before_batches(monkeypatch):
    class Metadata:
        num_rows = 1
        num_row_groups = 1

        def row_group(self, index):
            assert index == 0
            return type("Group", (), {"total_byte_size": 2048})()

    class ParquetSource:
        metadata = Metadata()

        def iter_batches(self, **kwargs):
            pytest.fail("Parquet batches must not be decoded after metadata rejection")

    monkeypatch.setattr(
        "generalforecast.tabular.parquet.ParquetFile", lambda *a, **kw: ParquetSource()
    )
    with pytest.raises(PackageError) as error:
        read_table(b"small", "parquet", max_bytes=1024)
    assert error.value.code == "CapacityExceeded"


def test_compressed_parquet_and_ipc_exceed_decoded_budget():
    table = pa.table({"x": ["z" * 64] * 32})
    parquet_data = io.BytesIO()
    parquet.write_table(table, parquet_data, compression="zstd", use_dictionary=False)
    ipc_data = io.BytesIO()
    with pa.ipc.new_stream(
        ipc_data, table.schema, options=pa.ipc.IpcWriteOptions(compression="zstd")
    ) as writer:
        writer.write_table(table)
    feather_data = io.BytesIO()
    feather.write_feather(table, feather_data, compression="zstd")
    for format, data in (
        ("parquet", parquet_data.getvalue()),
        ("arrow", ipc_data.getvalue()),
        ("feather", feather_data.getvalue()),
    ):
        assert len(data) < 1024 < table.nbytes
        with pytest.raises(PackageError) as error:
            read_table(data, format, max_bytes=1024)
        assert error.value.code == "CapacityExceeded"


def test_multiple_ipc_batches_retain_normalized_digest():
    table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "a"]})
    data = io.BytesIO()
    with pa.ipc.new_stream(data, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=1):
            writer.write_batch(batch)
    assert read_table(data.getvalue(), "arrow").equals(table)
    assert table_digest(read_table(data.getvalue(), "arrow")) == table_digest(table)


def test_json_array_stops_decoding_after_row_limit(monkeypatch):
    from generalforecast import tabular

    original = tabular.json.JSONDecoder
    count = 0

    class CountingDecoder(original):
        def raw_decode(self, s, idx=0):
            nonlocal count
            count += 1
            assert count <= 2, "decoded records beyond row limit"
            return super().raw_decode(s, idx)

    monkeypatch.setattr(tabular.json, "JSONDecoder", CountingDecoder)
    with pytest.raises(PackageError) as error:
        read_table(b'[{"x":1},{"x":2},{"x":3}]', "json", max_rows=1)
    assert error.value.code == "CapacityExceeded"
    assert count == 2


def test_legacy_feather_rejects_unbounded_reader():
    with pytest.raises(PackageError) as error:
        read_table(b"FEA1" + b"x" * 5, "feather")
    assert error.value.code == "UnsupportedFormat"


@pytest.mark.parametrize(
    "format", ["csv", "tsv", "parquet", "arrow", "ipc", "feather", "json", "jsonl"]
)
def test_local_formats_have_same_physical_content(format):
    table = pa.table({"x": [1, 2], "label": [False, True]})
    output = io.BytesIO()
    if format in ("csv", "tsv"):
        delimiter = "\t" if format == "tsv" else ","
        output.write(
            f"x{delimiter}label\n1{delimiter}false\n2{delimiter}true\n".encode()
        )
    elif format == "parquet":
        parquet.write_table(table, output)
    elif format == "feather":
        feather.write_feather(table, output)
    elif format == "arrow":
        output.write(arrow_bytes(table))
    elif format == "ipc":
        with pa.ipc.new_file(output, table.schema) as writer:
            writer.write_table(table)
    elif format == "json":
        output.write(json.dumps(table.to_pylist()).encode())
    else:
        output.write(b"\n".join(json.dumps(row).encode() for row in table.to_pylist()))
    parsed = read_table(output.getvalue(), format)
    assert parsed.equals(table)
    assert table_digest(parsed) == table_digest(table)


def test_pandas_and_polars_eager_lazy_numpy_are_supported():
    pandas = pytest.importorskip("pandas")
    polars = pytest.importorskip("polars")
    expected = pa.table({"x": [1, 2], "y": [3, 4]})
    for data in (
        pandas.DataFrame({"x": [1, 2], "y": [3, 4]}),
        polars.DataFrame({"x": [1, 2], "y": [3, 4]}),
        polars.DataFrame({"x": [1, 2], "y": [3, 4]}).lazy(),
        np.array([[1, 3], [2, 4]]),
    ):
        assert to_arrow(data, columns=["x", "y"]).equals(expected)


def test_lazy_frame_is_bounded_before_collection(monkeypatch):
    polars = pytest.importorskip("polars")
    lazy = polars.DataFrame({"x": range(8)}).lazy()
    original = polars.LazyFrame.limit
    limits = []

    def limit(self, count):
        limits.append(count)
        return original(self, count)

    monkeypatch.setattr(polars.LazyFrame, "limit", limit)
    with pytest.raises(PackageError) as error:
        to_arrow(lazy, max_rows=4)
    assert error.value.code == "CapacityExceeded"
    assert limits == [5]


def test_normalized_hash_ignores_chunks_metadata_and_dictionary_codes():
    plain = pa.table({"x": ["a", "b", "a"]})
    chunks = pa.Table.from_batches(
        plain.to_batches(max_chunksize=1)
    ).replace_schema_metadata({b"pandas": b"ignored"})
    encoded = pa.table({"x": plain["x"].dictionary_encode()})
    assert table_digest(plain) == table_digest(chunks) == table_digest(encoded)


def test_missing_and_datetime_are_physical_not_semantic():
    data = pa.table(
        {
            "x": [1.0, float("nan")],
            "t": [datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC), None],
        }
    )
    result = to_arrow(data)
    assert result["x"].null_count == 1
    assert result["t"].type == pa.timestamp("us", tz="UTC")
    assert result["t"].null_count == 1


def test_record_conversion_does_not_drop_later_columns():
    table = to_arrow([{"x": 1}, {"x": 2, "y": 3}])
    assert table.column_names == ["x", "y"]
    assert table["y"].to_pylist() == [None, 3]


def test_read_only_mappings_and_records_are_supported():
    assert to_arrow(MappingProxyType({"x": [1, 2]})).equals(pa.table({"x": [1, 2]}))
    assert to_arrow([MappingProxyType({"x": 1}), MappingProxyType({"x": 2})]).equals(
        pa.table({"x": [1, 2]})
    )


def test_duplicate_numpy_names_and_series_fail_explicitly():
    with pytest.raises(PackageError):
        to_arrow(np.array([[1, 2]]), columns=["x", "x"])
    pandas = pytest.importorskip("pandas")
    polars = pytest.importorskip("polars")
    for value in (pandas.Series([1, 2]), polars.Series([1, 2])):
        with pytest.raises(PackageError) as error:
            to_arrow(value)
        assert error.value.code == "UnsupportedFormat"


@pytest.mark.parametrize(
    "data",
    [
        np.array([1, 2]),
        pa.table({"x": [[1], [2]]}),
        pa.table({"x": [float("inf")]}),
        pa.table({"x": [None]}),
        {"x": [1, "mixed"]},
    ],
)
def test_unsupported_tables_fail(data):
    with pytest.raises(PackageError):
        to_arrow(data)


def test_formats_limits_and_invalid_bytes_are_explicit():
    with pytest.raises(PackageError) as error:
        read_table(b"x\n1\n2\n", "csv", max_rows=1)
    assert error.value.code == "CapacityExceeded"
    with pytest.raises(PackageError):
        read_table(b"not parquet", "parquet")
    with pytest.raises(PackageError) as error:
        read_table(b"x\n1\n", "xlsx")
    assert error.value.code == "UnsupportedFormat"
