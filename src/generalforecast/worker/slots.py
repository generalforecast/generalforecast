# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Slots: where the node puts each encoded context, as the worker packs and moves it between tiers."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from generalforecast import protocol
from generalforecast._core import CacheKey
from generalforecast.runtime.decoding import (
    Context,
)
from generalforecast.runtime.paging import (
    CacheLayout,
    HostPool,
    StateLayout,
    pack_state,
)
from generalforecast.worker.records import Movement

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


class Slots:
    """How a worker stores contexts in the slots the node assigns, and moves them between tiers."""

    def apply(self: "Worker", movement: Movement) -> None:
        """Applies one copy or drop from the node.

        A drop may name an entry a failed step never stored, and a copy one
        the node dropped since it planned the copy, when the step that was to
        encode it failed meanwhile; both are skipped.
        """
        # Pages are only ever dropped.
        if movement.key.kind == "rows":
            self._pages.pop(movement.key, None)
            return
        if movement.target is None:
            self._fixed.pop(movement.key, None)
            if self._contexts.pop(movement.key, None) is not None:
                self.stats.dropped += 1
            return
        context = self._contexts.get(movement.key)
        if context is None:
            return
        slots = list(movement.slots)
        if movement.source == "device":
            data = self._device.read_host(context.slots)
        else:
            data = self._below(movement.source).read(context.slots)
        if movement.target == "device":
            self._device.write(slots, data)
            self._settled(self._device.data)
            self.stats.promoted += 1
        else:
            self._below(movement.target).write(slots, data)
            if movement.target == "host":
                self.stats.demoted += 1
            else:
                self.stats.spilled += 1
        context.slots = slots

    def _below(self: "Worker", tier: protocol.Tier) -> HostPool:
        """The host pool, or the disk pool."""
        if tier == "host":
            return self._host
        if self._disk is None:
            raise RuntimeError(
                "the node moved a context to disk, but the worker has no disk tier"
            )
        return self._disk

    def _store(
        self: "Worker",
        key: CacheKey,
        slots: Sequence[int],
        layout: CacheLayout,
        rows: int,
        cache: Any,
        own: StateLayout | None = None,
        state: Any = None,
    ) -> None:
        """Packs an encoded context of `rows` layout rows, and any own state after it, into the device slots the node assigned it."""
        slots = list(slots)
        packed = layout.packed_slots(cache)
        own_data = None if own is None else pack_state(own, state)
        used = packed + (0 if own_data is None else own_data.shape[0])
        # The node keeps room for the table's state, with or without
        # categorical columns; the state may take fewer whole slots.
        written = slots if own is None else slots[:used]
        if used != len(written):
            raise RuntimeError(
                f"a context packs into {used} slots, but the node gave {len(slots)}"
            )
        with self.stats.timed("store"):
            self._device.write_cache(slots[:packed], layout, cache)
            if own_data is not None:
                self._device.write(slots[packed:used], own_data)
            self._settled(self._device.data)
        self._contexts[key] = Context(layout, rows, slots)
