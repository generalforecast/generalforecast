# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

from generalforecast.constants import MEASURE_OVERSTATE
from generalforecast.worker.admission import Admission

GIB = 1 << 30


def _worker(context: int, line: int, exact: int) -> tuple[Any, list[bool]]:
    """A worker of 30 GiB beside its pool whose encode profile gives `line` and an exact measure `exact`."""
    measured: list[bool] = []

    def encode_bytes(model, features, size, fit=False, exact_measure=False, **named):
        exact_measure = named.get("exact", exact_measure)
        measured.append(exact_measure)
        return exact if exact_measure else line

    worker: Any = SimpleNamespace(
        _tables={0: SimpleNamespace(model=0, features=8, own=None)},
        # The context's cache over its 1,000 rows, padded to themselves.
        profile=lambda model, features: SimpleNamespace(
            cache_bytes_per_row=context // 1000
        ),
        _size=lambda rows: rows,
        decode_profile=lambda model: SimpleNamespace(
            queries=8, working_bytes=lambda per_row, size: 0
        ),
        budget=SimpleNamespace(device=40 * GIB, cpu=True),
        _pool_bytes=10 * GIB,
        encode_bytes=encode_bytes,
    )
    worker._admitted_rows = lambda unit: Admission._admitted_rows(worker, unit)
    return worker, measured


def _unit() -> Any:
    return SimpleNamespace(table=0, rows=list(range(1000)), action="transient")


def test_a_context_the_profile_overstates_is_measured_before_refusal() -> None:
    worker, measured = _worker(context=1 * GIB, line=50 * GIB, exact=20 * GIB)

    Admission._fits_device(worker, _unit())

    assert measured == [False, True]


def test_a_context_whose_cache_alone_cannot_fit_is_refused_without_a_measure() -> None:
    # Its cache twice is past the device whatever its encode measures.
    worker, measured = _worker(context=20 * GIB, line=50 * GIB, exact=1 * GIB)

    with pytest.raises(ValueError, match="use a strategy that reads fewer rows"):
        Admission._fits_device(worker, _unit())

    assert measured == [False]


def test_a_context_far_past_the_device_is_refused_without_a_measure() -> None:
    line = (MEASURE_OVERSTATE * 30 + 1) * GIB
    worker, measured = _worker(context=1 * GIB, line=line, exact=1 * GIB)

    with pytest.raises(ValueError, match="use a strategy that reads fewer rows"):
        Admission._fits_device(worker, _unit())

    assert measured == [False]
