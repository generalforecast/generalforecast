"""Row keys: integers, strings, bytes or UUIDs, as the 128 bits tables hold.

An integer key keeps its value, a negative one in two's complement, and a UUID
keeps its 128 bits, as do the 16-byte values of an Arrow column of Arrow's UUID
type or of 16-byte binaries, which store UUIDs. A string key takes the 16-byte
BLAKE2b digest of its UTF-8 bytes, read little-endian, and a bytes key that of
its bytes. Numbers given as floats must be whole and below 2**53, past which
float64 cannot tell neighboring integers apart. Tables take keys as an (n, 2)
array of each key's low and high 64 bits, and servers as 16 little-endian bytes
each.
"""

import contextlib
import hashlib
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

Key = int | str | bytes | uuid.UUID

_LOW = (1 << 64) - 1
_DIGEST = re.compile(r"0x([0-9a-f]{32})")
# Integers past this differ from their neighbors by more than one as float64.
_EXACT_FLOAT = 1 << 53


@dataclass(frozen=True)
class Keys:
    """A batch's keys, as tables hold them and as given.

    Attributes:
        halves: Each key's low and high 64 bits, an (n, 2) uint64 array.
        given: The keys as the caller gave them, to name them in errors.
    """

    halves: np.ndarray
    given: np.ndarray

    def __len__(self) -> int:
        """Number of keys."""
        return len(self.halves)

    def arrow(self) -> Any:
        """The keys as an Arrow column of 16 little-endian bytes each."""
        import pyarrow as pa

        data = np.ascontiguousarray(self.halves, dtype="<u8").tobytes()
        return pa.FixedSizeBinaryArray.from_buffers(
            pa.binary(16), len(self), [None, pa.py_buffer(data)]
        )

    def name(self, message: str) -> str:
        """A message with each key digest it holds replaced by the key given."""
        wide = self.halves[:, 0].astype(object) | (
            self.halves[:, 1].astype(object) << 64
        )

        def given(match: re.Match[str]) -> str:
            found = np.flatnonzero(wide == int(match.group(1), 16))
            return repr(self.given[found[0]]) if len(found) else match.group(0)

        return _DIGEST.sub(given, message)

    @contextlib.contextmanager
    def naming(self) -> Iterator[None]:
        """Re-raises a key or value error naming keys as the caller gave them."""
        try:
            yield
        except (KeyError, ValueError) as error:
            message = str(error.args[0]) if error.args else str(error)
            named = self.name(message)
            if named == message:
                raise
            kind = KeyError if isinstance(error, KeyError) else ValueError
            raise kind(named) from error


def series_keys(series: Any, times: Any) -> tuple[np.ndarray, np.ndarray]:
    """Row keys of observations, as tables of series hold them.

    A series is known by the low 64 bits of its key's digest, and an
    observation's row key packs that digest in its high half and the time in
    its low half, so the node reads the series and the time from the key.

    Args:
        series: Each observation's series key, as `keys_of` takes keys.
        times: Each observation's time, int64, one per observation.

    Returns:
        The series' 64-bit digests, and the row keys as an (n, 2) uint64 array.

    Raises:
        ValueError: If there is not one time per series key.
    """
    keys = keys_of(series)
    stamps = np.ascontiguousarray(times, dtype=np.int64)
    if stamps.shape != (len(keys),):
        raise ValueError(f"expected {len(keys)} times, got shape {stamps.shape}")
    digests = np.ascontiguousarray(keys.halves[:, 0])
    return digests, np.column_stack([stamps.view(np.uint64), digests])


def _wide(value: Any) -> int:
    """One key's 128 bits, as a non-negative integer."""
    if isinstance(value, bool | np.bool_):
        raise TypeError("a key is an integer, string, bytes or UUID, not a boolean")
    if isinstance(value, int | np.integer):
        number = int(value)
        if not -(1 << 127) <= number < 1 << 128:
            raise ValueError(f"integer key {number} does not fit 128 bits")
        return number % (1 << 128)
    if isinstance(value, uuid.UUID):
        return value.int
    if isinstance(value, str):
        value = value.encode()
    if isinstance(value, bytes):
        digest = hashlib.blake2b(value, digest_size=16).digest()
        return int.from_bytes(digest, "little")
    raise TypeError(
        f"a key is an integer, string, bytes or UUID, not {type(value).__name__}"
    )


def keys_of(values: Any) -> Keys:
    """A batch's keys, from any array-like, Arrow array, or iterable.

    Args:
        values: One key per row: integers, strings, bytes or UUIDs.

    Returns:
        The keys as tables hold them.

    Raises:
        ValueError: If the keys are not one-dimensional, or a number is not whole.
        TypeError: If a key is of another type, such as a boolean.
    """
    if _arrow(values) and (uuids := _arrow_uuids(values)) is not None:
        return uuids
    if isinstance(values, range):
        values = np.arange(values.start, values.stop, values.step)
    elif hasattr(values, "to_numpy") and not isinstance(values, np.ndarray):
        values = (
            values.to_numpy(zero_copy_only=False)
            if _arrow(values)
            else values.to_numpy()
        )
    elif not hasattr(values, "__len__") and hasattr(values, "__iter__"):
        values = list(values)
    if isinstance(values, list | tuple) and any(
        isinstance(value, bool | np.bool_) for value in values
    ):
        # NumPy would read True as 1 among integers.
        raise TypeError("a key is an integer, string, bytes or UUID, not a boolean")
    array = np.asarray(values)
    # NumPy turns a list mixing numbers and strings into strings alone, and
    # integers too large for int64 into floats.
    if (
        array.dtype.kind in "USf"
        and isinstance(values, list | tuple)
        and (array.dtype.kind != "f" or not any(isinstance(v, float) for v in values))
    ):
        array = np.array(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"keys are one per row, not shape {array.shape}")
    if array.dtype.kind == "f":
        if not np.isfinite(array).all() or (array != np.round(array)).any():
            raise ValueError("keys that are numbers must be whole")
        if (np.abs(array) >= _EXACT_FLOAT).any():
            raise ValueError(
                "keys given as floats must be below 2**53, past which float64 "
                "rounds them; give such keys as integers"
            )
        array = array.astype(np.int64)
    halves = np.zeros((len(array), 2), dtype=np.uint64)
    if array.dtype.kind == "u":
        halves[:, 0] = array
    elif array.dtype.kind == "i":
        signed = array.astype(np.int64)
        halves[:, 0] = signed.view(np.uint64)
        halves[:, 1] = np.where(signed < 0, np.uint64(_LOW), np.uint64(0))
    elif array.dtype.kind == "b":
        raise TypeError("a key is an integer, string, bytes or UUID, not a boolean")
    else:
        for index, value in enumerate(array.tolist()):
            wide = _wide(value)
            halves[index] = (wide & _LOW, wide >> 64)
    return Keys(halves, array)


def _arrow(values: Any) -> bool:
    """Whether values are an Arrow array, whose `to_numpy` copies on request."""
    return type(values).__module__.startswith("pyarrow")


def _arrow_uuids(values: Any) -> Keys | None:
    """An Arrow column of UUIDs or 16-byte binaries as keys keeping their bits.

    Returns None for a column of another type.
    """
    import pyarrow as pa

    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    storage = values.storage if isinstance(values, pa.ExtensionArray) else values
    if not (
        pa.types.is_fixed_size_binary(storage.type) and storage.type.byte_width == 16
    ):
        return None
    if storage.null_count:
        raise ValueError("keys cannot be missing")
    data = storage.buffers()[1]
    start = storage.offset * 16
    # A UUID's bytes are big-endian: its first 8 hold the high half.
    big = np.frombuffer(data, dtype=">u8", count=2 * len(storage), offset=start)
    big = big.reshape(len(storage), 2)
    halves = np.ascontiguousarray(big[:, ::-1], dtype=np.uint64)
    given = np.array(
        [uuid.UUID(bytes=value) for value in storage.to_pylist()], dtype=object
    )
    return Keys(halves, given)
