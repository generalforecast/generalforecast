# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A plane's push-time estimate of one step's device memory is the worker's own admission.

The numbers are a Mitra-2 regressor's, as a probe of its programs measured
them on an A10G: 1,032,192 cache bytes a row, an encode of 5,120 rows
holding 210 MiB of inputs and 1,890 MiB of temporaries beside its 5,040 MiB
cache. A 24 GB pool GPU has 7,452 MiB beside its pool, an L40S 17,756 MiB.
"""

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest

from generalforecast import hosting, protocol
from generalforecast.constants import AUTOTUNE_SHARE, WORKING_MARGIN
from generalforecast.runtime.memory import EncodeProfile, WorkingProfile, encode_widths
from generalforecast.runtime.step_memory import (
    device_room,
    fit_of,
    step_memory,
    step_sheet,
)
from generalforecast.worker.admission import (
    Admission,
    DeviceMemoryRefused,
    step_bytes,
)
from generalforecast.worker.runner import DEVICE_MEMORY, Worker, _kind

MIB = 1 << 20
GIB = 1 << 30
MITRA_ROW = 1_032_192
MITRA_INPUTS = 210 * MIB // 5120
MITRA_TEMPORARIES = 1890 * MIB // 5120
MITRA_DECODE = 27 * MIB // 5120
# JAX's limit on each card at 90%, and every catalog model's parameters.
A10G_LIMIT = 20330 * MIB
L40S_LIMIT = 40938 * MIB
PARAMETERS = 5426 * MIB
SETTINGS = "DeviceSettings(gpu)"
WIDTH = 20


def _measures(settings: str = SETTINGS) -> dict[Any, Any]:
    """Mitra-2's measures at width 20 as a pool worker keeps them."""
    encode = EncodeProfile(
        held=WorkingProfile(0, row_scale=1.0, row_bytes=MITRA_INPUTS),
        temporaries=WorkingProfile(0, row_scale=0.0, row_bytes=MITRA_TEMPORARIES),
        outputs=WorkingProfile(0, row_scale=1.0),
    )
    return {
        ("decode", 8, settings): WorkingProfile(
            8, row_scale=0.0, row_bytes=MITRA_DECODE
        ),
        ("profile", WIDTH, settings): SimpleNamespace(cache_bytes_per_row=MITRA_ROW),
        ("layout", WIDTH, settings): SimpleNamespace(
            row_bytes=MITRA_ROW, rows_per_slot=1, slots_per_row=1
        ),
        ("encode_profile", False, encode_widths(WIDTH), settings): encode,
        ("encode", WIDTH, 512, False, settings): 600 * MIB,
        ("fit_profile", WIDTH, "step", settings): object(),
    }


def _sheet() -> dict[str, Any]:
    # As the plane keeps it: through JSON.
    return json.loads(json.dumps(step_sheet(_measures())))


def _fit(limit: int) -> dict[str, Any]:
    room, pool = device_room(limit, PARAMETERS)
    return {
        "version": 1,
        "room": room,
        "pool": pool,
        "minimum": 8,
        "slot_bytes": MIB,
        "models": {"general-forecast/mitra-2-regressor": _sheet()},
    }


def test_a_pool_gpu_has_what_its_workers_leave_beside_their_pool() -> None:
    assert device_room(A10G_LIMIT, PARAMETERS) == (7452 * MIB, 7452 * MIB)
    assert device_room(L40S_LIMIT, PARAMETERS) == (17756 * MIB, 17756 * MIB)


def test_a_sheet_keeps_only_the_pool_workers_settings_and_what_admission_reads() -> (
    None
):
    measures = {**_measures("DeviceSettings(other)"), **_measures()}
    measures[("decode", 8, "DeviceSettings(other)")] = WorkingProfile(
        8, fixed_bytes=1.0
    )
    sheet = step_sheet(measures, SETTINGS)

    assert sheet["settings"] == SETTINGS
    assert sheet["decode"]["row_bytes"] == MITRA_DECODE
    assert list(sheet["widths"]) == ["20"]
    assert sheet["widths"]["20"]["per_row"] == MITRA_ROW
    assert sheet["widths"]["20"]["encode"] == {"512": 600 * MIB}
    assert step_sheet({}) == {}


def test_the_mitra_slice_needs_10_gib_apart_and_under_3_gib_fused() -> None:
    # Retrieved: core 1,280 and 3,840 neighbors a query, past Mitra-2's
    # trained cap of 5,120 rows in a 10,000-row context.
    step = step_memory(_sheet(), WIDTH, 1280 + 3840)
    assert step is not None

    encode = int(WORKING_MARGIN * 5120 * (MITRA_ROW + MITRA_INPUTS + MITRA_TEMPORARIES))
    working = int(WORKING_MARGIN * 5120 * MITRA_DECODE)
    autotune = encode // AUTOTUNE_SHARE
    assert step.padded == 5120
    assert step.apart == pytest.approx(2 * 5120 * MITRA_ROW + working + autotune, abs=4)
    assert step.fused == pytest.approx(
        encode - 5120 * MITRA_ROW + 2 * working + autotune, abs=4
    )
    # Probed: 10.3 GiB decoded apart, 2.4 GiB fused, which the charge holds.
    assert 10.2 * GIB < step.apart < 10.5 * GIB
    assert step.fused is not None
    assert 2.4 * GIB < step.fused < 3 * GIB
    assert step.measured


def test_a_24_gb_gpu_runs_the_mitra_slice_only_fused_and_an_l40s_either_way() -> None:
    spec = "general-forecast/mitra-2-regressor"
    small = fit_of(_fit(A10G_LIMIT), spec, WIDTH, 5120)
    large = fit_of(_fit(L40S_LIMIT), spec, WIDTH, 5120)
    assert small is not None and large is not None

    assert small.room == 7452 * MIB
    assert (small.fits(fused=True), small.fits(fused=False)) == (True, False)
    assert large.room == 17756 * MIB
    assert (large.fits(fused=True), large.fits(fused=False)) == (True, True)
    assert (
        fit_of(_fit(A10G_LIMIT), "general-forecast/tabicl-v2-regressor", 20, 5120)
        is None
    )


def test_the_estimate_is_what_the_worker_admits() -> None:
    # Admission's rule, on the sizes the sheet holds, at each path; the
    # estimate runs `_fits_device` itself, so a room one byte short refuses.
    sheet = _sheet()
    for rows in (600, 2048, 5120, 7000):
        step = step_memory(sheet, WIDTH, rows)
        assert step is not None
        refused = step_memory(sheet, WIDTH, rows, room=step.apart - 1, pool=GIB)
        admitted = step_memory(sheet, WIDTH, rows, room=step.apart, pool=GIB)
        assert refused is not None and admitted is not None
        assert refused.fits(fused=False) is False
        assert admitted.fits(fused=False) is True


def test_one_rule_charges_each_path() -> None:
    def charge(**path: Any) -> int:
        sizes: dict[str, Any] = {"encode": 900, "context": 400, "working": 50}
        return step_bytes(**{**sizes, **path})

    # Put off and decoded apart: its encode, or its cache twice and a decode.
    assert charge(transient=True) == 900 + 900 // AUTOTUNE_SHARE
    assert charge(transient=True, encode=None) - 50 // AUTOTUNE_SHARE == 850
    # Fused: the encode without its cache as an output, and the decode twice.
    assert charge(transient=True, fused=True) == 500 + 100 + 900 // AUTOTUNE_SHARE
    # Kept: its encode, or its gather, state and decode.
    assert charge(transient=False, gather=1000, state=10) == 1060 + 900 // 16
    assert charge(transient=True, accelerator=False) == 900


def test_a_kept_context_its_pool_cannot_hold_is_charged_put_off() -> None:
    sheet = _sheet()

    kept = step_memory(sheet, WIDTH, 4096, kept=True, room=20 * GIB, pool=8 * GIB)
    put_off = step_memory(sheet, WIDTH, 4096, kept=True, room=20 * GIB, pool=GIB)

    assert kept is not None and put_off is not None
    assert kept.kept and kept.fused is None
    assert not put_off.kept and put_off.fused is not None
    assert kept.apart < put_off.apart


def test_a_width_the_pool_did_not_measure_reads_the_nearest_below() -> None:
    step = step_memory(_sheet(), 100, 5120)

    assert step is not None
    assert step.width == WIDTH
    assert not step.measured


def test_the_slice_a_plan_reads() -> None:
    def plan(selection: Any, context_rows: int | None, task: str = "regression") -> Any:
        columns = [
            SimpleNamespace(name="a", type="double", semantic=None, role="input"),
            SimpleNamespace(name="t", type="timestamp", semantic=None, role="input"),
            SimpleNamespace(name="y", type="double", semantic=None, role="target"),
        ]
        return SimpleNamespace(
            task=task,
            spec=SimpleNamespace(selection=selection),
            context_rows=context_rows,
            schema=SimpleNamespace(columns=columns),
            model=SimpleNamespace(
                resolved=hosting.pin("general-forecast/mitra-2-regressor").resolved
            ),
        )

    retrieved = SimpleNamespace(strategy="retrieved", core=1280, neighbors=3840)
    adaptive = SimpleNamespace(strategy="adaptive", core=None, neighbors=0)

    step = hosting.plan_step(plan(retrieved, 10_000))
    assert step is not None
    assert step["rows"] == 5120 and not step["kept"]
    # A timestamp input reads as its five parts.
    assert step["width"] == 6
    assert (step["core"], step["neighbors"]) == (1280, 3840)
    assert step["spec"] == "general-forecast/mitra-2-regressor"
    exact = hosting.plan_step(plan(None, 900))
    assert exact is not None and exact["rows"] == 900 and exact["kept"]
    assert hosting.plan_step(plan(None, None)) is None
    assert hosting.plan_step(plan(adaptive, 10_000)) is None
    assert hosting.plan_step(plan(retrieved, 10_000, task="forecast")) is None


def test_a_device_memory_refusal_carries_its_sizes() -> None:
    refused = DeviceMemoryRefused("too big", needed=3, room=2)

    assert isinstance(refused, ValueError)
    assert (refused.needed, refused.room) == (3, 2)
    assert Admission._fits_device.__annotations__["return"] is int


def test_a_step_the_device_cannot_hold_fails_with_its_kind(monkeypatch) -> None:
    worker: Any = Worker.__new__(Worker)
    worker.deferred, worker.failure_kinds = {}, {}
    worker.stats = SimpleNamespace(timed=lambda name: _Null())
    worker.apply = lambda movement: None
    worker._widened = lambda step: step
    errors = {
        0: DeviceMemoryRefused("the slice needs 10 GiB", 10, 7),
        1: RuntimeError("bad rows"),
    }

    def answered(step: Any) -> dict:
        tables = {unit.table for unit in step.units}
        if len(tables) > 1:
            raise RuntimeError("one part failed")
        raise errors[tables.pop()]

    worker._answered = answered
    monkeypatch.setattr(
        "generalforecast.worker.runner._part",
        lambda step, table: SimpleNamespace(units=[SimpleNamespace(table=table)]),
    )
    step: Any = SimpleNamespace(
        movements=[],
        fits=[],
        pages=[],
        units=[SimpleNamespace(table=0), SimpleNamespace(table=1)],
        queries=[],
        series_units=[],
    )

    _, failed = Worker.run_apart(worker, step)
    header, _ = protocol.predictions({}, failed, kinds=worker.failure_kinds)

    assert worker.failure_kinds == {0: DEVICE_MEMORY}
    assert header["failures"] == [
        {
            "table": 0,
            "message": "DeviceMemoryRefused: the slice needs 10 GiB",
            "kind": "device_memory",
        },
        {"table": 1, "message": "RuntimeError: bad rows"},
    ]


def test_only_a_device_memory_failure_has_a_kind() -> None:
    import jax

    out = jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating")

    assert _kind(out) == DEVICE_MEMORY
    assert _kind(DeviceMemoryRefused("x", 2, 1)) == DEVICE_MEMORY
    assert _kind(ValueError("a context of 5 rows")) is None
    assert _kind(jax.errors.JaxRuntimeError("INTERNAL: bad kernel")) is None


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def test_the_plane_asks_a_process_of_its_own_for_the_estimate() -> None:
    # The hosting surface imports no worker code: it asks `shipped`, as it
    # takes every worker-side measure, and gets the worker's own numbers.
    fits = {"A10G": _fit(A10G_LIMIT), "L40S": _fit(L40S_LIMIT)}
    spec = "general-forecast/mitra-2-regressor"

    found = hosting.step_fits(fits, spec, WIDTH, [5120, 2048])
    other = hosting.step_fits(fits, "general-forecast/tabicl-v2-regressor", 20, [8])

    direct = fit_of(fits["A10G"], spec, WIDTH, 5120)
    assert direct is not None
    assert found["A10G"][0] == dataclasses.asdict(direct)
    large, smaller = found["L40S"][0], found["A10G"][1]
    assert large is not None and large["room"] == 17756 * MIB
    assert smaller is not None and smaller["padded"] == 2048
    assert other == {"A10G": [None], "L40S": [None]}
    assert hosting.device_room(A10G_LIMIT, PARAMETERS) == (7452 * MIB, 7452 * MIB)
