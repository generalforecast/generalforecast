# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Encoding: contexts encoded from pages or fit on their own rows, kept or put off until a decode needs them."""

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast._core import bucket_size
from generalforecast.constants import FUSED_QUERIES, TRANSIENT_ROWS
from generalforecast.worker.calls import _host_padded
from generalforecast.worker.pages import Representations
from generalforecast.worker.records import Page, QueryBatch, Unit

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


@dataclass
class _Deferred:
    """A unit there is no room to keep, encoded only when its decode needs it.

    Attributes:
        table: The unit's table.
        arguments: Its encode's arguments after the model parameters.
        rows: Its real rows.
        stage: `encode`, or `fit_encode` for a unit fit on its own rows.
        encoded: Whether it was encoded once already, for the counts.
    """

    table: int
    arguments: tuple
    rows: int
    stage: str = "encode"
    encoded: bool = False


# Units without slots, put off until decoded.
Transient = dict[int, _Deferred]


@dataclass(frozen=True)
class _Gather:
    """A put-off unit's rows, gathered from its step's rows once it is encoded.

    Attributes:
        values: The step's rows' representations for the unit's table, on
            the device, or on the host where they would take more than half
            a call's budget there.
        targets: Their targets, where `values` are.
        index: Each of the unit's padded rows' place in `values`; padding
            points past them, and reads as zeros.
    """

    values: jax.Array | np.ndarray
    targets: jax.Array | np.ndarray
    index: np.ndarray

    @property
    def on_device(self) -> bool:
        """Whether the step's rows are on the device, for a gather there."""
        return isinstance(self.values, jax.Array)


@jax.jit
def _taken(values: jax.Array, index: jax.Array) -> jax.Array:
    """`values`' rows at `index`, zeros where it points past them."""
    return jnp.take(values, index, axis=0, mode="fill", fill_value=0)


def _host_taken(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """`values`' rows at `index`, zeros where it points past them, on the host."""
    taken = np.zeros((len(index), *values.shape[1:]), values.dtype)
    inside = index < len(values)
    taken[inside] = values[index[inside]]
    return taken


def _expanded(arguments: tuple) -> tuple:
    """A put-off unit's encode arguments, its rows gathered where the step's rows are."""
    if len(arguments) != 3 or not isinstance(arguments[1], _Gather):
        return arguments
    state, gather, rows = arguments
    if not gather.on_device:
        return (
            state,
            _host_taken(np.asarray(gather.values), gather.index),
            _host_taken(np.asarray(gather.targets), gather.index),
            rows,
        )
    index = jnp.asarray(gather.index)
    return (state, _taken(gather.values, index), _taken(gather.targets, index), rows)


def _reads(queries: Sequence[QueryBatch], units: int) -> np.ndarray:
    """How many of a step's queries read each of its `units` units."""
    read = [np.asarray(batch.units, dtype=np.int64) for batch in queries]
    if not read:
        return np.zeros(units, dtype=np.int64)
    return np.bincount(np.concatenate(read), minlength=units)


class Encoding:
    """How a worker encodes a step's units."""

    def _encode(
        self: "Worker",
        units: Sequence[Unit],
        pages: Sequence[Page],
        computed: dict[int, Representations],
        subsets: dict[int, np.ndarray] | None = None,
        queries: Sequence[QueryBatch] | None = None,
    ) -> Transient:
        """Encodes the units that are not cached; returns those there was no room to keep.

        A unit with no room to keep is encoded only by the decode that reads
        it, so a step never holds every such context at once. A page in
        `subsets` holds the representations of those rows alone. `queries`,
        the step's batches, say how many queries read each such unit, so it
        is admitted as its decode will run (`_fused_transient`); without
        them, each is admitted as decoded apart from its encode.
        """
        transient: Transient = {}
        waiting = [
            index
            for index, unit in enumerate(units)
            if unit.action not in ("reuse", "fuse")
        ]
        self.stats.reused += sum(unit.action == "reuse" for unit in units)
        self.stats.rows_reused += sum(
            len(unit.rows) for unit in units if unit.action == "reuse"
        )
        joined = self._tables_rows(pages, computed, subsets) if waiting else {}
        # The step's rows of each table that put-off units gather from, on
        # the device once: such units are many, each a context of its own,
        # and gathering each on the host and copying it over took most of
        # their encode. Padded to a bucket, so few sizes compile a gather.
        # Where they would take more than half a call's budget, they stay
        # on the host, and each unit gathers its own there.
        gathered: dict[int, tuple[Any, Any]] = {}
        work: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
        group: dict[str, list[int]] = defaultdict(list)
        reads = (
            _reads(queries, len(units))
            if queries is not None
            and any(units[index].action == "transient" for index in waiting)
            else None
        )
        for index in waiting:
            unit = units[index]
            self._fits_device(
                unit, fused=self._fused_transient(unit, index, joined, reads)
            )
            if self._tables[unit.table].own is not None:
                stage, arguments = "fit_encode", self._own_arguments(unit)
            elif unit.action == "transient":
                stage = "encode"
                arguments = self._gathered_later(unit, joined, gathered)
            else:
                stage, arguments = "encode", self._gathered(unit, joined)
            if unit.action == "transient":
                transient[index] = _Deferred(
                    unit.table, arguments, len(unit.rows), stage
                )
                continue
            work[stage].append((unit.table, arguments))
            group[stage].append(index)

        # Each context goes to its slots as its call finishes, so a step never
        # holds all it encodes beside the pool.
        def keeper(stage: str) -> Callable[[int, Any], None]:
            def keep(position: int, outputs: Any) -> None:
                unit = units[group[stage][position]]
                state, cache = outputs if stage == "fit_encode" else (None, outputs)
                if state is not None:
                    self._warn_crowded(unit.table, state)
                self.stats.encoded += 1
                self.stats.rows_encoded += len(unit.rows)
                table = self._tables[unit.table]
                layout = self._layouts[table.model, table.features]
                self._store(
                    unit.key,
                    unit.slots,
                    layout,
                    len(unit.rows),
                    cache,
                    table.own,
                    state,
                )

            return keep

        for stage, items in work.items():
            self._call(stage, items, consume=keeper(stage))
        return transient

    def _fused_transient(
        self: "Worker",
        unit: Unit,
        index: int,
        joined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
        reads: np.ndarray | None,
    ) -> bool:
        """Whether a put-off unit will encode and decode in one program, as `_decode_window` picks.

        One that gathers its rows from the step's rows on the device, read by
        at most `FUSED_QUERIES` of the step's queries, as `_gathered_later`
        and `_decode_window` decide it.
        """
        if (
            reads is None
            or unit.action != "transient"
            or self._tables[unit.table].own is not None
            or reads[index] > FUSED_QUERIES
        ):
            return False
        rows, values, targets = joined[unit.table]
        return self._joins_on_device(
            values, targets, bucket_size(len(rows), self._minimum)
        )

    def _gathered(
        self: "Worker",
        unit: Unit,
        joined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple:
        """A unit's encode arguments, its rows' representations gathered from pages."""
        rows, values, targets = joined[unit.table]
        wanted = np.asarray(unit.rows)
        positions = np.searchsorted(rows, wanted)
        if np.any(positions >= len(rows)) or np.any(rows[positions] != wanted):
            raise RuntimeError(
                f"a unit of table {unit.table} reads rows that are in none "
                "of the step's pages"
            )
        size = self._size(len(unit.rows))
        with self.stats.timed("pad"):
            return (
                self._tables[unit.table].state,
                _host_padded(values[positions], size),
                _host_padded(targets[positions], size),
                len(unit.rows),
            )

    def _gathered_later(
        self: "Worker",
        unit: Unit,
        joined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
        gathered: dict[int, tuple[Any, Any]],
    ) -> tuple:
        """A put-off unit's encode arguments, its rows to gather once encoded.

        Only the places of its rows are kept meanwhile, so a step of many
        put-off units holds one copy of its rows rather than one per unit.
        """
        rows, values, targets = joined[unit.table]
        wanted = np.asarray(unit.rows)
        positions = np.searchsorted(rows, wanted)
        if np.any(positions >= len(rows)) or np.any(rows[positions] != wanted):
            raise RuntimeError(
                f"a unit of table {unit.table} reads rows that are in none "
                "of the step's pages"
            )
        if unit.table not in gathered:
            padded = bucket_size(len(rows), self._minimum)
            if self._joins_on_device(values, targets, padded):
                with self.stats.timed("pad"):
                    gathered[unit.table] = (
                        jnp.asarray(_host_padded(values, padded)),
                        jnp.asarray(_host_padded(targets, padded)),
                    )
            else:
                # More than half a call's budget, as a wide model's
                # representations of many rows are: each unit
                # gathers its own rows on the host as it is encoded.
                gathered[unit.table] = (values, targets)
        step_values, step_targets = gathered[unit.table]
        index = np.full(
            self._transient_size(len(unit.rows)), len(step_values), np.int32
        )
        index[: len(positions)] = positions
        return (
            self._tables[unit.table].state,
            _Gather(step_values, step_targets, index),
            len(unit.rows),
        )

    def _joins_on_device(
        self: "Worker", values: np.ndarray, targets: np.ndarray, padded: int
    ) -> bool:
        """Whether a step's rows, padded, take at most half a call's budget on the device.

        Put-off units gather from them through the step's fused calls, whose
        budget they take from, so the step holds them and a call within the
        device beside its pool. On a CPU, whose memory is the host's, they
        always go to the device.
        """
        if self.budget.cpu:
            return True
        row = (values.nbytes + targets.nbytes) // max(1, len(values))
        return padded * row <= self._call_bytes // 2

    def _transient_size(self: "Worker", rows: int) -> int:
        """Rows a put-off context of `rows` rows pads to: its bucket, or past `TRANSIENT_ROWS`, the next multiple of it."""
        bucket = self._size(rows)
        if bucket <= TRANSIENT_ROWS:
            return bucket
        return min(bucket, -(-rows // TRANSIENT_ROWS) * TRANSIENT_ROWS)

    def _own_arguments(self: "Worker", unit: Unit) -> tuple:
        """A unit's fit arguments, from the rows it came with."""
        if unit.features is None or unit.targets is None:
            raise RuntimeError(
                f"a unit of table {unit.table} to fit came without its rows"
            )
        size = self._size(len(unit.rows))
        with self.stats.timed("pad"):
            return (
                _host_padded(unit.features, size),
                _host_padded(unit.targets, size),
                len(unit.rows),
                self._tables[unit.table].categorical,
            )

    def _encode_deferred(
        self: "Worker", indices: list[int], transient: Transient
    ) -> dict[int, tuple[Any, Any]]:
        """Encodes the put-off units among `indices`, once each, in shared calls.

        Returns:
            Each unit's own state, or None for one that gathered its rows from
            pages, and its cache.
        """
        wanted = list(dict.fromkeys(indices))
        deferred = [transient[unit] for unit in wanted]
        outputs = self._staged(
            [(d.stage, d.table, _expanded(d.arguments)) for d in deferred]
        )
        encoded: dict[int, tuple[Any, Any]] = {}
        for unit, put_off, output in zip(wanted, deferred, outputs, strict=True):
            state, cache = output if put_off.stage == "fit_encode" else (None, output)
            if state is not None:
                self._warn_crowded(put_off.table, state)
            if not put_off.encoded:
                put_off.encoded = True
                self.stats.encoded += 1
                self.stats.transient += 1
                self.stats.rows_encoded += put_off.rows
            encoded[unit] = (state, cache)
        return encoded
