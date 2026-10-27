# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounds an Arrow IPC stream's or file's decoded size from its metadata alone.

Arrow IPC buffers may be LZ4 or ZSTD compressed, so a few kilobytes can decode
to gigabytes. Each record batch's metadata lists its buffers and, when the body
is compressed, every buffer starts with its uncompressed length as an 8-byte
little-endian integer (-1 for a buffer left uncompressed). Summing those
bounds the decoded size before any buffer is decompressed or allocated.

A file is checked through its footer, the way `pyarrow.ipc.open_file` reads
it: each footer block names a message that will be decoded, whatever else the
file holds between them.

Dictionary batches count toward the byte bound like record batches, but the
data they stand for is only known once decoded: normalization replaces each
index with its value, so a few bytes of dictionary can stand for any amount
of data. `check` refuses them unless the caller passes `dictionaries=True`
and bounds that expansion itself, as `tabular.to_arrow` does.
"""

import struct
from collections.abc import Iterator

import pyarrow as pa

from generalforecast.predictors.validation import PredictorError

MAGIC = b"ARROW1"

_SCHEMA, _DICTIONARY, _RECORD_BATCH = 1, 2, 3
# A footer block: offset (int64), metadata length (int32), padding, body length.
_BLOCK = struct.Struct("<qi4xq")
# A record batch buffer: offset and length within the message body (int64s).
_BUFFER = struct.Struct("<qq")


def _invalid() -> PredictorError:
    return PredictorError(
        "InvalidTable", "source bytes do not match a supported table format"
    )


def _exceeded() -> PredictorError:
    return PredictorError(
        "CapacityExceeded", "decoded source exceeds row or byte limit"
    )


def _read(data: bytes | memoryview | pa.Buffer, kind: str, at: int) -> int:
    # struct counts a negative offset from the end; a flatbuffer never does.
    if at < 0:
        raise _invalid()
    return struct.unpack_from(kind, data, at)[0]


class _Table:
    """One flatbuffer table, read with bounds-checked offsets."""

    def __init__(self, data: bytes | memoryview, position: int) -> None:
        self.data, self.position = data, position
        self.vtable = position - _read(data, "<i", position)
        self.size = _read(data, "<H", self.vtable)

    def _slot(self, index: int) -> int | None:
        slot = 4 + 2 * index
        if slot + 2 > self.size:
            return None
        offset = _read(self.data, "<H", self.vtable + slot)
        return self.position + offset if offset else None

    def present(self, index: int) -> bool:
        return self._slot(index) is not None

    def scalar(self, index: int, kind: str, default: int = 0) -> int:
        slot = self._slot(index)
        return default if slot is None else _read(self.data, kind, slot)

    def child(self, index: int) -> "_Table | None":
        slot = self._slot(index)
        if slot is None:
            return None
        return _Table(self.data, slot + _read(self.data, "<I", slot))

    def structs(self, index: int, layout: struct.Struct) -> list[tuple[int, ...]]:
        slot = self._slot(index)
        if slot is None:
            return []
        start = slot + _read(self.data, "<I", slot)
        count = _read(self.data, "<I", start)
        if start + 4 + count * layout.size > len(self.data):
            raise _invalid()
        return [
            layout.unpack_from(self.data, start + 4 + layout.size * item)
            for item in range(count)
        ]


def _batch_bytes(batch: _Table, body: pa.Buffer) -> int:
    """A record batch's decoded buffer bytes, read from metadata and prefixes."""
    compressed = batch.present(3)
    total = 0
    for offset, length in batch.structs(2, _BUFFER):
        if offset < 0 or length < 0 or offset + length > body.size:
            raise _invalid()
        if not compressed or length == 0:
            total += length
            continue
        # Arrow refuses a compressed buffer too short to hold its prefix.
        if length < 8:
            raise _invalid()
        raw = _read(body, "<q", offset)
        if raw < -1:
            raise _invalid()
        total += length - 8 if raw == -1 else raw
    return total


def _stream_messages(data: bytes) -> Iterator[pa.ipc.Message]:
    yield from pa.ipc.MessageReader.open_stream(pa.py_buffer(data))


def _file_messages(data: bytes) -> Iterator[pa.ipc.Message]:
    """The message of each footer block, dictionaries first, as a reader reads them.

    The file is magic, padding, messages, the footer, its int32 length, magic.
    """
    view = memoryview(data)
    trailer = len(data) - len(MAGIC) - 4
    if not data.endswith(MAGIC) or trailer < 8:
        raise _invalid()
    length = _read(view, "<i", trailer)
    start = trailer - length
    if length <= 0 or start < 8:
        raise _invalid()
    footer = view[start:trailer]
    root = _Table(footer, _read(footer, "<I", 0))
    buffer = pa.py_buffer(data)
    for index in (2, 3):
        for offset, _metadata, _body in root.structs(index, _BLOCK):
            if not 8 <= offset < start:
                raise _invalid()
            yield pa.ipc.read_message(buffer.slice(offset, start - offset))


def check(
    data: bytes, *, max_rows: int, max_bytes: int, dictionaries: bool = False
) -> None:
    """Refuses IPC bytes whose batches would decode past `max_rows` or `max_bytes`.

    `data` is a stream, or a file when it starts with the file magic. Only
    metadata and the 8-byte length prefix of each compressed buffer are read;
    nothing is decompressed.

    Raises:
        PredictorError: `InvalidTable` for bytes that are not an IPC stream or
            file, `CapacityExceeded` for bytes that decode past a bound, or
            `UnsupportedType` for a dictionary batch unless `dictionaries`.
    """
    rows = size = 0
    file = data.startswith(MAGIC)
    try:
        # A file's schema is in its footer, not a message of its own.
        seen_schema = file
        for message in (_file_messages if file else _stream_messages)(data):
            metadata = message.metadata.to_pybytes()
            root = _Table(metadata, _read(metadata, "<I", 0))
            kind = root.scalar(1, "<B")
            header = root.child(2)
            if kind == _SCHEMA and not seen_schema:
                seen_schema = True
                continue
            if not seen_schema or header is None:
                raise _invalid()
            if kind == _DICTIONARY:
                if not dictionaries:
                    raise PredictorError(
                        "UnsupportedType",
                        "dictionary-encoded columns are not supported; decode them first",
                    )
                batch = header.child(1)
                if batch is None:
                    raise _invalid()
            elif kind == _RECORD_BATCH:
                batch = header
                length = batch.scalar(0, "<q")
                if length < 0:
                    raise _invalid()
                rows += length
            else:
                raise _invalid()
            size += _batch_bytes(batch, message.body or pa.py_buffer(b""))
            if rows > max_rows or size > max_bytes:
                raise _exceeded()
        if not seen_schema:
            raise _invalid()
    except PredictorError:
        raise
    except (
        pa.ArrowException,
        OSError,
        struct.error,
        ValueError,
        IndexError,
        OverflowError,
    ):
        raise _invalid() from None
