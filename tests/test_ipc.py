# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Arrow IPC uploads are bounded from their metadata before anything decodes."""

import io
import json
import random
import subprocess
import sys
import textwrap

import numpy as np
import pyarrow as pa
import pytest

from generalforecast import ipc
from generalforecast.predictors.validation import PredictorError
from generalforecast.tabular import MAX_BYTES, read_table

ZSTD = pa.ipc.IpcWriteOptions(compression="zstd")
GIB = 1 << 30
ROWS = 1_000_000


def stream(table: pa.Table, options: pa.ipc.IpcWriteOptions = ZSTD) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    return sink.getvalue()


def file(table: pa.Table, options: pa.ipc.IpcWriteOptions = ZSTD) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_file(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    return sink.getvalue()


def bomb() -> pa.Table:
    """A million rows of 128 int64 zero columns: 1 GB that zstd packs into kilobytes.

    Every column shares one 8 MB array, so building it allocates only that.
    """
    zeros = pa.array(np.zeros(ROWS, dtype=np.int64))
    return pa.table([zeros] * 128, names=[f"c{i}" for i in range(128)])


def normal() -> pa.Table:
    return pa.table(
        {
            "id": pa.array(range(1000), pa.int64()),
            "x": pa.array(np.linspace(0, 1, 1000)),
            "name": pa.array([f"row-{i % 7}" for i in range(1000)]),
        }
    )


def refused(data: bytes, format: str = "arrow", **limits) -> str:
    with pytest.raises(PredictorError) as error:
        read_table(data, format, **limits)
    return error.value.code


# Reads the bomb in a fresh process, so its Arrow pool and peak RSS are its own.
_PEAK = textwrap.dedent(
    """
    import json, resource, sys
    import pyarrow as pa
    from generalforecast.predictors.validation import PredictorError
    from generalforecast.tabular import read_table

    data = open(sys.argv[1], "rb").read()
    scale = 1 if sys.platform == "darwin" else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    try:
        read_table(data, sys.argv[2])
        code = None
    except PredictorError as error:
        code = error.code
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    pool = pa.default_memory_pool().max_memory()
    print(json.dumps({"code": code, "rss": after - before, "pool": pool}))
    """
)


@pytest.mark.parametrize(("write", "format"), [(stream, "arrow"), (file, "feather")])
def test_a_compression_bomb_is_refused_without_decompressing(tmp_path, write, format):
    data = write(bomb())
    assert len(data) < 256 * 1024
    path = tmp_path / "bomb"
    path.write_bytes(data)
    run = subprocess.run(
        [sys.executable, "-c", _PEAK, str(path), format],
        capture_output=True,
        text=True,
        check=True,
    )
    peak = json.loads(run.stdout)
    assert peak["code"] == "CapacityExceeded"
    # The bomb decodes to 1 GB; refusing it allocates nothing of that order.
    assert peak["pool"] < 1 << 20, peak
    assert peak["rss"] < 32 << 20, peak


def test_the_bound_is_read_from_metadata_and_length_prefixes():
    table = bomb().select(["c0", "c1"])
    for write in (stream, file):
        for options in (ZSTD, pa.ipc.IpcWriteOptions(compression="lz4"), None):
            data = write(table, options or pa.ipc.IpcWriteOptions())
            # Two 8 MB columns: exactly 16 MB of decoded buffers.
            ipc.check(data, max_rows=ROWS, max_bytes=16 * ROWS)
            with pytest.raises(PredictorError) as error:
                ipc.check(data, max_rows=ROWS, max_bytes=16 * ROWS - 1)
            assert error.value.code == "CapacityExceeded"
            with pytest.raises(PredictorError) as error:
                ipc.check(data, max_rows=ROWS - 1, max_bytes=GIB)
            assert error.value.code == "CapacityExceeded"


def test_a_dictionary_that_expands_past_the_limit_is_refused():
    # One 1 MiB value, indexed by a million rows, stands for a terabyte.
    values = pa.array(["x" * (1 << 20)])
    indices = pa.array(np.zeros(ROWS, dtype=np.int8))
    table = pa.table({"d": pa.DictionaryArray.from_arrays(indices, values)})
    for data in (stream(table), file(table)):
        assert len(data) < 64 * 1024
        assert refused(data) == "CapacityExceeded"
        with pytest.raises(PredictorError) as error:
            ipc.check(data, max_rows=ROWS, max_bytes=MAX_BYTES)
        assert error.value.code == "UnsupportedType"


def test_a_small_dictionary_column_still_reads_as_its_values():
    categories = pa.array(["a", "bb", None, "a"]).dictionary_encode()
    table = pa.table({"d": categories, "n": pa.array([1, 2, 3, 4])})
    for data in (stream(table), file(table)):
        read = read_table(data, "arrow")
        assert read.column("d").type == pa.string()
        assert read.column("d").to_pylist() == ["a", "bb", None, "a"]


def test_truncated_and_corrupted_bytes_raise_the_typed_input_error():
    rng = random.Random(0)
    for data in (stream(normal()), file(normal())):
        cuts = sorted({*range(64), *rng.sample(range(len(data)), 200)})
        for cut in cuts:
            try:
                read_table(data[:cut], "arrow")
            except PredictorError as error:
                assert error.code in ("InvalidTable", "CapacityExceeded")
        for _ in range(300):
            corrupt = bytearray(data)
            for _ in range(rng.randint(1, 4)):
                corrupt[rng.randrange(len(corrupt))] = rng.randrange(256)
            try:
                read_table(bytes(corrupt), "arrow")
            except PredictorError:
                pass
    assert refused(stream(normal())[:-20]) == "InvalidTable"
    assert refused(b"ARROW1\0\0" + b"\0" * 32 + b"ARROW1") == "InvalidTable"
    assert refused(b"\xff\xff\xff\xff" + b"\x10\0\0\0" + b"\0" * 8) == "InvalidTable"
    assert refused(b"not arrow at all") == "InvalidTable"


def test_a_file_is_bounded_by_the_messages_its_footer_names():
    data = file(bomb().select(["c0", "c1", "c2", "c3"]))
    assert refused(data, "feather", max_bytes=16 << 20) == "CapacityExceeded"
    assert read_table(data, "feather", max_bytes=64 << 20).num_rows == ROWS
    # Feather is the file format; a stream named .feather is not one.
    assert refused(stream(normal()), "feather") == "InvalidTable"


@pytest.mark.parametrize("write", [stream, file])
@pytest.mark.parametrize("compression", [None, "zstd", "lz4"])
def test_normal_data_reads_unchanged(write, compression):
    table = normal()
    data = write(table, pa.ipc.IpcWriteOptions(compression=compression))
    read = read_table(data, "arrow")
    assert read.equals(table)
    assert read_table(data, "ipc").equals(table)
    # Several batches, each checked as it comes.
    sink = io.BytesIO()
    opener = pa.ipc.new_stream if write is stream else pa.ipc.new_file
    options = pa.ipc.IpcWriteOptions(compression=compression)
    with opener(sink, table.schema, options=options) as writer:
        writer.write_table(table, max_chunksize=64)
    assert read_table(sink.getvalue(), "arrow").equals(table)
