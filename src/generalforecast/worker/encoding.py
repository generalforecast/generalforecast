# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Encoding: contexts encoded from pages or fit on their own rows, kept or put off until a decode needs them."""

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from generalforecast.worker.calls import _host_padded
from generalforecast.worker.pages import Representations
from generalforecast.worker.records import Page, Unit

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


class Encoding:
    """How a worker encodes a step's units."""

    def _encode(
        self: "Worker",
        units: Sequence[Unit],
        pages: Sequence[Page],
        computed: dict[int, Representations],
    ) -> Transient:
        """Encodes the units that are not cached; returns those there was no room to keep.

        A unit with no room to keep is encoded only by the decode that reads
        it, so a step never holds every such context at once.
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
        joined = self._tables_rows(pages, computed) if waiting else {}
        work: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
        group: dict[str, list[int]] = defaultdict(list)
        for index in waiting:
            unit = units[index]
            self._fits_device(unit)
            if self._tables[unit.table].own is not None:
                stage, arguments = "fit_encode", self._own_arguments(unit)
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
        outputs = self._staged([(d.stage, d.table, d.arguments) for d in deferred])
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
