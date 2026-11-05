# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sizes a context with no room to keep pads to, and that its calls and admission take the same one."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from generalforecast._core import bucket_size
from generalforecast.constants import (
    TRANSIENT_ROWS,
    TRANSIENT_STEPPED_ROWS,
    TRANSIENT_STEPS,
)
from generalforecast.worker.admission import Admission
from generalforecast.worker.encoding import (
    TRANSIENT_STEPS_ENV,
    Encoding,
    transient_size,
)


def _padded(rows: int) -> int:
    """`rows`' transient size for a worker of bucket minimum 8."""
    return transient_size(rows, bucket_size(rows, 8))


def _sizes(most: int) -> set[int]:
    """Every transient size of contexts of up to `most` rows."""
    return {_padded(rows) for rows in range(1, most + 1)}


@pytest.fixture(autouse=True)
def _stepped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRANSIENT_STEPS_ENV, raising=False)


def test_the_constants_are_the_agreed_ones() -> None:
    assert (TRANSIENT_ROWS, TRANSIENT_STEPPED_ROWS, TRANSIENT_STEPS) == (512, 8192, 4)


@pytest.mark.parametrize(
    ("rows", "padded"),
    [
        # Up to 512 rows, the bucket.
        (1, 8),
        (9, 16),
        (300, 512),
        (512, 512),
        # Past it, steps of 512 rows up to 8,192.
        (513, 1024),
        (1024, 1024),
        (1025, 1536),
        (5000, 5120),
        (5120, 5120),
        (8191, 8192),
        (8192, 8192),
        # Past 8,192, four sizes an octave: 2,048 rows apart up to 16,384.
        (8193, 10240),
        (10240, 10240),
        (10241, 12288),
        (12288, 12288),
        (12289, 14336),
        (14336, 14336),
        (14337, 16384),
        (16384, 16384),
        # 4,096 apart up to 32,768.
        (16385, 20480),
        (20480, 20480),
        (20481, 24576),
        (24577, 28672),
        (28673, 32768),
        # 8,192 apart up to 65,536: TabICL v2's 60,000 rows pad to its bucket.
        (32769, 40960),
        (40961, 49152),
        (49153, 57344),
        (57345, 65536),
        (60000, 65536),
        # 16,384 apart past it, to the catalog's 100,000.
        (65537, 81920),
        (81921, 98304),
        (98305, 114688),
        (100000, 114688),
    ],
)
def test_a_context_pads_to_its_step(rows: int, padded: int) -> None:
    assert _padded(rows) == padded


def test_no_context_pads_past_its_bucket_or_below_its_rows() -> None:
    for rows in range(1, 100_001, 7):
        bucket = bucket_size(rows, 8)
        assert rows <= _padded(rows) <= bucket
    # A bucket smaller than the step caps it.
    assert transient_size(8193, 9000) == 9000


def test_past_the_stepped_rows_a_context_pads_by_at_most_a_quarter() -> None:
    for rows in range(TRANSIENT_STEPPED_ROWS + 1, 100_001, 13):
        assert _padded(rows) <= 1.25 * rows


def test_sizes_per_model() -> None:
    # Mitra-2 to its 5,120 rows: as before, all within the steps of 512.
    assert len(_sizes(5120)) == 16
    # TabICL v2 to its 60,000, and any model to the catalog's 100,000: 124
    # and 202 sizes with steps of 512 all the way.
    assert len(_sizes(60_000)) == 34
    assert len(_sizes(100_000)) == 37
    assert sorted(_sizes(100_000) - _sizes(8192)) == [
        10240,
        12288,
        14336,
        16384,
        20480,
        24576,
        28672,
        32768,
        40960,
        49152,
        57344,
        65536,
        81920,
        98304,
        114688,
    ]


def test_the_environment_restores_steps_of_512(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRANSIENT_STEPS_ENV, "0")

    assert _padded(8193) == 8704
    assert _padded(60000) == 60416
    assert len(_sizes(5120)) == 16
    assert len(_sizes(60_000)) == 124
    assert len(_sizes(100_000)) == 202


def _worker() -> Any:
    worker: Any = SimpleNamespace(
        _tables={0: SimpleNamespace(own=None, state=None)},
        _minimum=8,
        _size=lambda rows: bucket_size(rows, 8),
        budget=SimpleNamespace(cpu=False),
        _joins_on_device=lambda values, targets, padded: False,
        stats=None,
    )
    worker._transient_size = lambda rows: Encoding._transient_size(worker, rows)
    return worker


@pytest.mark.parametrize("rows", [5120, 9000, 60000])
def test_admission_charges_the_size_the_gather_pads_to(rows: int) -> None:
    # The encode, a decode apart from it and the fused encode and decode
    # all take the size from the gather's index; admission must charge it.
    worker = _worker()
    unit: Any = SimpleNamespace(table=0, rows=list(range(rows)), action="transient")
    step = np.arange(rows, dtype=np.int64)
    joined = {0: (step, np.zeros((rows, 1), np.float32), np.zeros(rows, np.float32))}

    _, gather, length = Encoding._gathered_later(worker, unit, joined, {})

    assert length == rows
    assert len(gather.index) == _padded(rows)
    assert Admission._admitted_rows(worker, unit) == len(gather.index)
