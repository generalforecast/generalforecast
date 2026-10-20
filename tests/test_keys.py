import hashlib
import uuid

import numpy as np
import pytest

from generalforecast.keys import keys_of


def _wide(halves: np.ndarray) -> list[int]:
    return [int(low) | (int(high) << 64) for low, high in halves]


def test_keys_map_to_128_bits_by_type() -> None:
    token = uuid.UUID(int=(1 << 127) + 5)
    digest = int.from_bytes(
        hashlib.blake2b(b"acct-9", digest_size=16).digest(), "little"
    )

    assert _wide(keys_of(np.array([0, 2**64 - 1], dtype=np.uint64)).halves) == [
        0,
        2**64 - 1,
    ]
    assert _wide(keys_of(np.array([-1, 3])).halves) == [2**128 - 1, 3]
    assert _wide(keys_of(["acct-9", b"acct-9", token]).halves) == [
        digest,
        digest,
        token.int,
    ]
    # A list mixing numbers and strings keeps its numbers, as NumPy alone would not.
    assert _wide(keys_of([7, "acct-9"]).halves) == [7, digest]
    assert _wide(keys_of(range(2)).halves) == [0, 1]
    assert _wide(keys_of(np.array([4.0])).halves) == [4]
    # Integers past int64 stay exact rather than pass through float64.
    assert _wide(keys_of([2**63, -1]).halves) == [2**63, 2**128 - 1]


def test_arrow_uuid_columns_keep_their_bits() -> None:
    pa = pytest.importorskip("pyarrow")
    tokens = [uuid.UUID(int=(1 << 127) + 5), uuid.UUID(int=7)]
    raw = [token.bytes for token in tokens]

    for column in (
        pa.array(raw, pa.binary(16)),
        pa.chunked_array([raw[:1], raw[1:]], pa.binary(16)),
    ):
        assert _wide(keys_of(column).halves) == [token.int for token in tokens]
    assert _wide(keys_of(pa.array(raw, pa.binary(16)).slice(1)).halves) == [7]
    if hasattr(pa, "uuid"):
        assert _wide(keys_of(pa.array(raw, pa.uuid())).halves) == [
            token.int for token in tokens
        ]


def test_keys_of_other_types_are_refused() -> None:
    with pytest.raises(TypeError):
        keys_of([True])
    with pytest.raises(TypeError, match="boolean"):
        keys_of([True, 1])
    with pytest.raises(ValueError, match="2\\*\\*53"):
        keys_of(np.array([2.0**53]))
    with pytest.raises(TypeError):
        keys_of([1.5j])
    with pytest.raises(ValueError):
        keys_of(np.array([0.5]))
    with pytest.raises(ValueError):
        keys_of(np.zeros((2, 2), dtype=np.int64))


def test_errors_name_keys_as_given() -> None:
    keys = keys_of(["acct-9", 3])
    digest = f"0x{_wide(keys.halves)[0]:032x}"

    with (
        pytest.raises(KeyError, match="row key 'acct-9' does not exist"),
        keys.naming(),
    ):
        raise KeyError(f"row key {digest} does not exist")


def test_series_keys_pack_the_series_digest_over_the_time() -> None:
    from generalforecast.keys import series_keys

    digests, keys = series_keys(["shoes", 7], np.array([100, -10]))

    assert digests.dtype == np.uint64 and keys.shape == (2, 2)
    assert digests[1] == 7
    assert keys[1, 0] == np.int64(-10).view(np.uint64) and keys[1, 1] == 7
    assert digests[0] == keys_of(["shoes"]).halves[0, 0]
    with pytest.raises(ValueError, match="expected 2 times"):
        series_keys(["a", "b"], [1])
