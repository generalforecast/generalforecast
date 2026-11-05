# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Admission charges each context as its calls will hold it.

The numbers are a Mitra-2 regressor's on an A10G, as a probe of its
programs measured them: 1,032,192 cache bytes a row, an encode of 5,120 rows
holding 210 MiB of inputs and 1,890 MiB of temporaries beside its 5,040 MiB
cache, and 7,452 MiB beside the pool.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from generalforecast import protocol
from generalforecast.constants import AUTOTUNE_SHARE, WORKING_MARGIN
from generalforecast.worker.admission import Admission
from generalforecast.worker.encoding import Encoding, _reads

MIB = 1 << 20
GIB = 1 << 30
# Mitra-2's cache bytes per row, and its encode's inputs and temporaries a row.
MITRA_ROW = 1_032_192
MITRA_INPUTS = 210 * MIB // 5120
MITRA_TEMPORARIES = 1890 * MIB // 5120
# Decoding the smallest chunk, 8 queries, against 5,120 rows: 27 MiB.
MITRA_DECODE = 27 * MIB // 5120
# An A10G's room beside its pool, as a pool worker allocates it.
A10G_ROOM = 7452 * MIB


def _size(rows: int) -> int:
    """Buckets as a worker of bucket minimum 8 pads to: powers of two."""
    return max(8, 1 << (rows - 1).bit_length())


def _worker(
    per_row: int = MITRA_ROW,
    encode: Any = None,
    decode: Any = None,
    room: int = A10G_ROOM,
    cpu: bool = False,
    row_bytes: int | None = None,
) -> tuple[Any, list[tuple[int, bool]]]:
    """A worker whose sizes are given, recording each encode it measures: its rows and whether exactly."""
    measured: list[tuple[int, bool]] = []

    def encode_line(rows: int) -> int:
        held = rows * (per_row + MITRA_INPUTS)
        return int(WORKING_MARGIN * (held + rows * MITRA_TEMPORARIES))

    encode = encode or encode_line
    decode = decode or (lambda rows: int(WORKING_MARGIN * rows * MITRA_DECODE))

    def encode_bytes(model, features, rows, fit=False, exact=False):
        measured.append((rows, exact))
        return encode(rows)

    pool = 10 * GIB
    worker: Any = SimpleNamespace(
        _tables={0: SimpleNamespace(model=0, features=20, own=None)},
        profile=lambda model, features: SimpleNamespace(cache_bytes_per_row=per_row),
        layout=lambda model, features: SimpleNamespace(
            row_bytes=per_row if row_bytes is None else row_bytes
        ),
        _size=_size,
        _slot_bytes=MIB,
        decode_profile=lambda model: SimpleNamespace(
            queries=8, working_bytes=lambda per_row, size: decode(size)
        ),
        encode_profile=lambda model, features, fit=False: object(),
        budget=SimpleNamespace(device=pool + room, cpu=cpu),
        _pool_bytes=pool,
        encode_bytes=encode_bytes,
    )
    worker._admitted_rows = lambda unit: Admission._admitted_rows(worker, unit)
    worker._gather_bytes = lambda model, features, size: Admission._gather_bytes(
        worker, model, features, size
    )
    worker._transient_size = lambda rows: Encoding._transient_size(worker, rows)
    return worker, measured


def _unit(rows: int, action: str = "transient") -> Any:
    return SimpleNamespace(table=0, rows=range(rows), action=action)


def test_a_transient_context_is_charged_at_the_size_its_encode_pads_to() -> None:
    worker, measured = _worker()

    with pytest.raises(ValueError) as refused:
        Admission._fits_device(worker, _unit(5120))

    # 5,120 rows, padded to 5,120 rather than its bucket of 8,192: its cache
    # is 5,040 MiB, not 8,064, and its encode is read at that size.
    assert "padded to 5120" in str(refused.value)
    assert "takes 5040 MiB" in str(refused.value)
    assert measured == [(5120, False)]


def test_a_mitra_slice_decoded_in_its_fused_program_is_admitted() -> None:
    # Probed: the one program that gathers, encodes and decodes it held
    # 2.4 GiB, which the charge holds with room for the autotuner.
    worker, _ = _worker()
    encode = int(WORKING_MARGIN * 5120 * (MITRA_ROW + MITRA_INPUTS + MITRA_TEMPORARIES))
    working = int(WORKING_MARGIN * 5120 * MITRA_DECODE)
    charged = encode - 5120 * MITRA_ROW + 2 * working + encode // AUTOTUNE_SHARE

    Admission._fits_device(worker, _unit(5120), fused=True)

    assert 2400 * MIB < charged < 3 * GIB
    assert charged < A10G_ROOM


def test_a_mitra_slice_decoded_apart_from_its_encode_is_refused() -> None:
    # Probed: its encode, then its decode beside a copy of its cache, held
    # 10.3 GiB, past the 7.3 GiB beside the pool.
    worker, measured = _worker()

    with pytest.raises(ValueError, match="use a strategy that reads fewer rows"):
        Admission._fits_device(worker, _unit(5120), fused=False)

    # Its cache twice is past the room whatever its encode, so nothing is
    # compiled to measure it.
    assert measured == [(5120, False)]


def test_a_cpu_charges_every_context_at_its_bucket_and_never_as_fused() -> None:
    worker, measured = _worker(cpu=True)

    with pytest.raises(ValueError) as refused:
        Admission._fits_device(worker, _unit(5120), fused=True)

    assert "padded to 8192" in str(refused.value)
    assert measured[0] == (8192, False)


def test_a_small_transient_context_reads_its_buckets_encode_measure() -> None:
    # 1,500 rows pad to 1,536 as they encode, below where the encode profile's
    # line holds: the bucket's exact measure, which a kept context of these
    # rows compiles anyway, bounds it without compiling a new size.
    worker, measured = _worker()

    Admission._fits_device(worker, _unit(1500), fused=True)

    assert measured == [(2048, False)]


def test_a_kept_context_is_charged_its_decodes_gather() -> None:
    # A kept context of 4 GiB whose encode takes 4.2 GiB and whose decode
    # works in 1 GiB, on a device with 5 GiB beside its pool: its encode fits,
    # but its paged decode's copy of the context and that decode do not.
    rows = 4096
    worker, _ = _worker(
        per_row=4 * GIB // rows,
        encode=lambda rows: int(4.2 * GIB),
        decode=lambda rows: GIB,
        room=5 * GIB,
    )

    with pytest.raises(ValueError, match="needs 5"):
        Admission._fits_device(worker, _unit(rows, action="encode"))

    # Without its gather, as on a CPU, whose budget is its host's, it fits.
    worker.budget.cpu = True
    Admission._fits_device(worker, _unit(rows, action="encode"))


def test_a_kept_context_whose_gather_fits_is_admitted() -> None:
    rows = 4096
    worker, _ = _worker(
        per_row=4 * GIB // rows,
        encode=lambda rows: int(4.2 * GIB),
        decode=lambda rows: GIB // 4,
        room=5 * GIB,
    )

    Admission._fits_device(worker, _unit(rows, action="encode"))


def test_reads_counts_each_units_queries() -> None:
    batches = [
        protocol.QueryBatch(0, 0, 0, np.zeros((3, 1), np.float32), [0, 2, 2]),
        protocol.QueryBatch(1, 0, 0, np.zeros((1, 1), np.float32), [2]),
    ]

    assert _reads(batches, 4).tolist() == [1, 0, 3, 0]
    assert _reads([], 2).tolist() == [0, 0]


def _encoding(on_device: bool) -> Any:
    worker: Any = SimpleNamespace(
        _tables={0: SimpleNamespace(own=None), 1: SimpleNamespace(own=object())},
        _minimum=8,
        _joins_on_device=lambda values, targets, padded: on_device,
    )
    return worker


@pytest.mark.parametrize(
    ("reads", "on_device", "table", "action", "fused"),
    [
        (8, True, 0, "transient", True),
        (9, True, 0, "transient", False),
        (1, False, 0, "transient", False),
        (1, True, 1, "transient", False),
        (1, True, 0, "encode", False),
    ],
)
def test_a_transient_unit_is_fused_as_its_decode_will_be(
    reads: int, on_device: bool, table: int, action: str, fused: bool
) -> None:
    worker = _encoding(on_device)
    rows = np.arange(16)
    joined = {table: (rows, np.zeros((16, 2)), np.zeros(16))}
    unit: Any = SimpleNamespace(table=table, rows=range(16), action=action)

    found = Encoding._fused_transient(
        worker, unit, 0, joined, np.asarray([reads], np.int64)
    )

    assert found is fused
    assert Encoding._fused_transient(worker, unit, 0, joined, None) is False
