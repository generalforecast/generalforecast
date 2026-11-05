# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device memory one step of a locked strategy needs on a pool's GPU, from its kept measures, without compiling.

A hosted plane holds a pushed predictor's strategy to what its pool's GPUs
can run before it accepts the push: a lock is the customer's contract, which
the engine serves as locked or refuses. A worker refuses a step past its
device in `Admission._fits_device`; this runs that same check on a stand-in
worker whose sizes are a GPU type's kept measures (`step_sheet`) and whose
room beside the pool is that GPU type's (`device_room`), so the plane's
estimate and the worker's admission never disagree.

A pool takes, once per GPU type, `pool_fit`: its room and a compact sheet of
each served model's measures, plain JSON the plane keeps. The plane reads it
with `step_memory`, for each slice its strategy reads: a retrieved
strategy's per-query context of core and neighbors, put off and encoded by
the decode that reads it, on the fused path where at most `FUSED_QUERIES`
queries read it and decoded apart from its encode otherwise; an exact
strategy's whole context, kept in the pool.
"""

import dataclasses
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from generalforecast.constants import MEASURED_ROWS

# Importing this module loads no JAX, as importing `generalforecast.hosting`
# does not: what reads measures imports what it needs as it runs.

# The sheet's format; a plane refuses to read another.
SHEET_VERSION = 1
# Bytes a pool slot holds, and the bucket minimum a gf-server worker pads to,
# by default.
SLOT_BYTES = 1 << 20
BUCKET_MINIMUM = 8


def device_room(
    device: int, parameters: int, slot_bytes: int = SLOT_BYTES
) -> tuple[int, int]:
    """Bytes beside the pool, and the pool's, on an accelerator a gf-server worker counts `device` bytes of.

    As a worker sizes them: the loaded models' `parameters` come off the
    device, half of the rest is the pool, in whole slots, and the step's
    calls have what the pool leaves.
    """
    from generalforecast.runtime.budget import MemoryBudget

    budget = MemoryBudget(device=device, host=0, cpu=False).holding(parameters)
    pool = budget.device_tier // slot_bytes * slot_bytes
    return budget.device - pool, pool


def step_sheet(
    measures: Mapping[Hashable, Any], settings: str | None = None
) -> dict[str, Any]:
    """What `step_memory` reads of one model's kept measures, as plain JSON; empty for none.

    Args:
        measures: A model's measures as a worker keeps them
            (`Manifest.measured`), keyed with the worker's device settings.
        settings: The `repr` of the settings a pool's workers run with; the
            ones its decode was measured under most often if None.

    Returns:
        The decode's profile, and per padded width the cache's bytes a row,
        its layout's, the encode's profile and its exact encodes by rows.
    """
    from generalforecast.runtime.memory import encode_widths

    keys = [key for key in measures if isinstance(key, tuple) and len(key) > 1]
    if settings is None:
        seen = Counter(key[-1] for key in keys if key[0] == "decode")
        if not seen:
            return {}
        settings = seen.most_common(1)[0][0]
    ours = [key for key in keys if key[-1] == settings]
    decodes = [measures[key] for key in ours if key[0] == "decode"]
    if not decodes:
        return {}
    widths: dict[int, dict[str, Any]] = {}
    profiles: dict[tuple, Any] = {}
    for key in ours:
        value = measures[key]
        if key[0] == "profile":
            widths.setdefault(key[1], {})["per_row"] = value.cache_bytes_per_row
        elif key[0] == "layout":
            widths.setdefault(key[1], {}).update(
                row_bytes=value.row_bytes,
                rows_per_slot=value.rows_per_slot,
                slots_per_row=value.slots_per_row,
            )
        elif key[0] == "encode_profile" and key[1] is False:
            profiles[tuple(key[2])] = value
        elif key[0] == "encode" and key[3] is False and value is not None:
            widths.setdefault(key[1], {}).setdefault("encode", {})[str(key[2])] = value
    sheet: dict[str, Any] = {}
    for width, entry in sorted(widths.items()):
        if "per_row" not in entry or "row_bytes" not in entry:
            continue
        profile = profiles.get(encode_widths(width))
        entry["encode_profile"] = (
            None if profile is None else dataclasses.asdict(profile)
        )
        entry.setdefault("encode", {})
        sheet[str(width)] = entry
    return {
        "version": SHEET_VERSION,
        "settings": settings,
        "decode": dataclasses.asdict(decodes[0]),
        "widths": sheet,
    }


@dataclass(frozen=True)
class StepMemory:
    """Device memory one step of a slice needs beside the pool, as a worker admits it.

    Attributes:
        rows: The slice's context rows.
        padded: The rows its context is charged at.
        width: The padded width the measures were read at.
        kept: Whether the context is kept in the pool; a put-off one is
            encoded by the decode that reads it.
        apart: Bytes with the context decoded apart from its encode: a kept
            context's encode or its paged decode, or a put-off context's
            encode or its cache twice and its decode.
        fused: Bytes where at most `FUSED_QUERIES` queries read a put-off
            context, which then encodes and decodes in one program; None
            for a kept one.
        room: Bytes beside the pool it was checked against, if any.
        measured: Whether the model's measures cover the width; if not, it
            is read at the nearest measured width.
    """

    rows: int
    padded: int
    width: int
    kept: bool
    apart: int
    fused: int | None
    room: int | None
    measured: bool

    def fits(self, fused: bool) -> bool | None:
        """Whether the step fits the room on the fused path or apart; None without a room."""
        if self.room is None:
            return None
        needed = self.fused if fused and self.fused is not None else self.apart
        return needed <= self.room


def _width(sheet: Mapping[str, Any], width: int) -> tuple[int, bool] | None:
    """The width a sheet holds measures of for a table this wide, and whether they are its own."""
    from generalforecast.runtime.programs import padded_width

    widths = sorted(int(key) for key in sheet["widths"])
    if not widths:
        return None
    for candidate in (width, padded_width(width)):
        if candidate in widths:
            return candidate, True
    # Measured widths stop at the pool's `MEASURED_COLUMNS`; a wider table is
    # read at the widest below it, whose cache a row is no larger.
    below = [candidate for candidate in widths if candidate <= width]
    return (below[-1] if below else widths[0]), False


def _stand_in(
    sheet: Mapping[str, Any],
    width: int,
    room: int,
    pool: int,
    minimum: int,
    slot_bytes: int,
) -> Any:
    """A worker whose sizes are the sheet's and whose device has `room` beside a pool of `pool` bytes.

    Only what `Admission._fits_device` reads, so it admits as a pool worker
    would before it measures a context exactly; a context's exact encode the
    pool kept stands in for that measure.
    """
    from generalforecast._core import bucket_size
    from generalforecast.runtime.memory import EncodeProfile, WorkingProfile
    from generalforecast.worker.admission import Admission
    from generalforecast.worker.encoding import transient_size

    entry = sheet["widths"][str(width)]
    per_row = entry["per_row"]
    decode = WorkingProfile(**sheet["decode"])
    stored = entry["encode_profile"]
    profile = (
        None
        if stored is None
        else EncodeProfile(
            **{part: WorkingProfile(**values) for part, values in stored.items()}
        )
    )
    measures = {int(rows): value for rows, value in entry["encode"].items()}

    class Measured(Admission):
        def __init__(self) -> None:
            self._tables = {0: SimpleNamespace(model=0, features=width, own=None)}
            self.budget = SimpleNamespace(device=room + pool, cpu=False)
            self._pool_bytes = pool
            self._slot_bytes = slot_bytes
            self._minimum = minimum

        def profile(self, model: int, features: int) -> Any:
            return SimpleNamespace(cache_bytes_per_row=per_row)

        def layout(self, model: int, features: int) -> Any:
            return SimpleNamespace(row_bytes=entry["row_bytes"])

        def decode_profile(self, model: int) -> Any:
            return decode

        def encode_profile(self, model: int, features: int, fit: bool = False) -> Any:
            return profile

        def encode_bytes(
            self,
            model: int,
            features: int,
            rows: int,
            fit: bool = False,
            exact: bool = False,
        ) -> int | None:
            # As `Admission.encode_bytes` reads it: by the profile's line from
            # `MEASURED_ROWS` up, and otherwise, or exactly, the measure at
            # its own size, which the pool kept or the line stands in for.
            if rows >= MEASURED_ROWS and not exact and profile is not None:
                return profile.working_bytes(per_row, rows)
            if rows in measures:
                return measures[rows]
            return None if profile is None else profile.working_bytes(per_row, rows)

        def _size(self, rows: int) -> int:
            return bucket_size(rows, minimum)

        def _transient_size(self, rows: int) -> int:
            return transient_size(rows, self._size(rows))

    return Measured()


def step_memory(
    sheet: Mapping[str, Any],
    width: int,
    rows: int,
    *,
    kept: bool = False,
    room: int | None = None,
    pool: int | None = None,
    minimum: int = BUCKET_MINIMUM,
    slot_bytes: int = SLOT_BYTES,
) -> StepMemory | None:
    """Device memory one step of a slice of `rows` context rows needs beside the pool, as a worker admits it.

    Runs `Admission._fits_device` itself on a stand-in worker with the
    sheet's sizes. With `room` and `pool`, a GPU type's (`device_room`), it
    admits as that GPU's worker would, and a kept context its pool cannot
    hold is put off; without, it charges as before any exact measure.

    Args:
        sheet: A model's `step_sheet` on the GPU type.
        width: The table's width: its feature columns.
        rows: The slice's context rows.
        kept: Whether the context is kept in the pool, as an exact
            strategy's whole context is; a retrieved strategy's per-query
            slices are put off.
        room: Bytes beside the pool, if checked against a GPU type.
        pool: The pool's bytes there.
        minimum: The bucket minimum the pool's workers pad to.
        slot_bytes: Bytes a pool slot holds.

    Returns:
        The step's bytes on each path, or None for a sheet without measures.

    Raises:
        ValueError: For a sheet of another version.
    """
    from generalforecast.worker.admission import DeviceMemoryRefused

    if not sheet:
        return None
    if sheet.get("version") != SHEET_VERSION:
        raise ValueError(f"a step sheet of version {sheet.get('version')!r}")
    found = _width(sheet, width)
    if found is None:
        return None
    at, measured = found
    # Past what a check can refuse, so the charge itself comes back.
    unbounded = 1 << 62
    checked = unbounded if room is None else room
    held = unbounded if pool is None else pool
    worker = _stand_in(sheet, at, checked, held, minimum, slot_bytes)
    entry = sheet["widths"][str(at)]
    if kept and pool is not None:
        # The node keeps a context only where its slots fit the pool.
        slots = (
            -(-worker._size(rows) // entry["rows_per_slot"]) * entry["slots_per_row"]
        )
        kept = slots * slot_bytes <= pool

    def charged(action: str, fused: bool) -> int:
        unit = SimpleNamespace(table=0, rows=range(rows), action=action)
        try:
            return worker._fits_device(unit, fused=fused)
        except DeviceMemoryRefused as refused:
            return refused.needed

    put_off = SimpleNamespace(table=0, rows=range(rows), action="transient")
    padded = worker._size(rows) if kept else worker._admitted_rows(put_off)
    return StepMemory(
        rows=rows,
        padded=padded,
        width=at,
        kept=kept,
        apart=charged("encode" if kept else "transient", fused=False),
        fused=None if kept else charged("transient", fused=True),
        room=room,
        measured=measured,
    )


def pool_fit(
    directory: Path,
    specs: Iterable[str],
    slot_bytes: int = SLOT_BYTES,
    minimum: int = BUCKET_MINIMUM,
) -> dict[str, Any]:
    """This GPU's room beside a pool worker's pool, and a sheet of each served model's measures it keeps, as JSON.

    Run on the pool's GPU type, with nothing else on the device, as a pool
    worker would start: the served models' parameters are loaded to count
    them, and the device's memory read as JAX allows it.

    Args:
        directory: Where the pool keeps measures (`GF_MEASURES`), one
            directory per identity.
        specs: Every model the pool's workers load, as they load them.
        slot_bytes: Bytes a pool slot holds.
        minimum: The bucket minimum the pool's workers pad to.
    """
    from generalforecast.models import registry
    from generalforecast.runtime.budget import MemoryBudget
    from generalforecast.runtime.device import default_settings
    from generalforecast.runtime.memory import tree_bytes
    from generalforecast.runtime.programs import Manifest
    from generalforecast.runtime.shipped import _digest, identity

    specs = list(specs)
    kept_under = identity(minimum, slot_bytes)
    settings = repr(default_settings())
    device = MemoryBudget.measure().device
    parameters = sum(tree_bytes(registry.load_engine_model(spec)[1]) for spec in specs)
    room, pool = device_room(device, parameters, slot_bytes)
    models = {}
    for spec in specs:
        manifest = Manifest(directory / kept_under / f"{_digest(spec)}.json")
        sheet = step_sheet(manifest.measured(), settings)
        if sheet:
            models[spec] = sheet
    return {
        "version": SHEET_VERSION,
        "identity": kept_under,
        "device": device,
        "parameters": parameters,
        "room": room,
        "pool": pool,
        "slot_bytes": slot_bytes,
        "minimum": minimum,
        "models": models,
    }


def fit_of(
    fit: Mapping[str, Any], spec: str, width: int, rows: int, *, kept: bool = False
) -> StepMemory | None:
    """`step_memory` of a slice on the GPU type a `pool_fit` describes; None where it keeps no measures of the model."""
    if fit.get("version") != SHEET_VERSION:
        raise ValueError(f"a pool fit of version {fit.get('version')!r}")
    sheet = fit["models"].get(spec)
    if not sheet:
        return None
    return step_memory(
        sheet,
        width,
        rows,
        kept=kept,
        room=fit["room"],
        pool=fit["pool"],
        minimum=fit["minimum"],
        slot_bytes=fit["slot_bytes"],
    )
