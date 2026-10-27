# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a worker counts across its steps, and the parts each step's time splits into."""

import contextlib
import dataclasses
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class WorkerStats:
    """Counters across every step.

    Attributes:
        encoded: Units encoded, whether kept or used once.
        reused: Units read from the cache instead of encoded, an extended
            unit's prefix among them.
        extended: Series units grown from a cached encoding by new patches.
        transient: Encoded units there was no room to keep.
        promoted: Cached units moved back from host to device memory.
        demoted: Cached units moved from device to host memory.
        spilled: Cached contexts moved from host memory to disk.
        dropped: Cached contexts dropped, evicted from the host or from a past epoch.
        rows_encoded: Context rows encoded across all units.
        rows_embedded: Rows embedded outside a fit, for representation pages
            that were not cached. Rows a page copies from its earlier entry
            count under rows_copied, not here.
        rows_copied: Page rows copied from the page's earlier entry instead
            of embedded, after other rows of the page were written.
        rows_reused: Context rows in units read from the cache.
        steps: Steps computed.
        queries: Query rows decoded.
        calls: Compiled calls per stage; tables batched together share one.
        compiles: Calls per stage whose shape their program had not seen,
            which compiled, or came from the compile cache.
        seconds: Time in compiled calls per stage, compiling included, since
            each call waits for its results.
        flops: Floating-point operations in compiled calls per stage, by
            XLA's estimate for each program, read once as the program
            compiles or loads for a call; padding to its shape counts.
        parts: Times each part of a step ran, by name. The parts split a
            step's time without overlap: `frame` decodes a server's step
            frame and `reply` converts predictions into its reply, `move`
            applies the step's copies and drops, `pad` pads rows to their
            bucket on the host, `stack` stacks a call's items, `compile`
            traces and compiles a call whose shape its program had not seen
            and hands it to the device, `dispatch` hands a call its program
            has seen and its host arrays to the device, `wait` waits for the
            device to finish it, `unstack` takes each item's outputs from
            the call's, `fetch` copies results to the host, `store` packs
            contexts and pages into their slots, and `other` is the rest of
            the step, the Python between the parts and XLA's measures of a
            call's memory. Timed, `store`, `move` and the decode's
            `dispatch` wait for the device work they start, which would
            otherwise count in the next part that waits. A step that fails keeps no part but `other`,
            which takes its whole time. The compiled stages' `seconds`
            overlap the parts: a call's seconds are its `compile` or
            `dispatch`, its `wait` and its measure.
        part_seconds: Time in each part.
        timers: Whether parts are timed. Untimed, `part` and `timed` count
            nothing and read no clock.
    """

    encoded: int = 0
    reused: int = 0
    extended: int = 0
    transient: int = 0
    promoted: int = 0
    demoted: int = 0
    spilled: int = 0
    dropped: int = 0
    rows_encoded: int = 0
    rows_embedded: int = 0
    rows_copied: int = 0
    rows_reused: int = 0
    steps: int = 0
    queries: int = 0
    calls: dict[str, int] = field(default_factory=dict)
    compiles: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    flops: dict[str, float] = field(default_factory=dict)
    parts: dict[str, int] = field(default_factory=dict)
    part_seconds: dict[str, float] = field(default_factory=dict)
    timers: bool = False

    def part(self, name: str, seconds: float) -> None:
        """Counts one run of a part of a step, and the seconds it took."""
        if not self.timers:
            return
        with _COUNTING:
            self.parts[name] = self.parts.get(name, 0) + 1
            self.part_seconds[name] = self.part_seconds.get(name, 0.0) + seconds

    def timed(self, name: str) -> contextlib.AbstractContextManager[None]:
        """Counts the block's time as one run of a part of a step."""
        return self._timing(name) if self.timers else _UNTIMED

    @contextlib.contextmanager
    def _timing(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.part(name, time.perf_counter() - started)

    def snapshot(self) -> "WorkerStats":
        """A copy to count from, or to go back to."""
        return dataclasses.replace(
            self,
            calls=dict(self.calls),
            compiles=dict(self.compiles),
            seconds=dict(self.seconds),
            flops=dict(self.flops),
            parts=dict(self.parts),
            part_seconds=dict(self.part_seconds),
        )

    def stages_since(
        self, before: "WorkerStats"
    ) -> dict[str, tuple[int, float, float]]:
        """Each stage's compiled calls, their seconds and their FLOPs since `before`."""
        return {
            stage: (
                calls,
                seconds,
                self.flops.get(stage, 0.0) - before.flops.get(stage, 0.0),
            )
            for stage, (calls, seconds) in _since(
                before.calls, before.seconds, self.calls, self.seconds
            ).items()
        }

    def parts_since(self, before: "WorkerStats") -> dict[str, tuple[int, float]]:
        """Each part's runs and their seconds since `before`."""
        return _since(before.parts, before.part_seconds, self.parts, self.part_seconds)


# Guards the counts of calls that run side by side; see `Worker._alone`.
_COUNTING = threading.Lock()


# What `WorkerStats.timed` gives untimed: a block that counts nothing.
_UNTIMED = contextlib.nullcontext()


def _since(
    counts_before: dict[str, int],
    seconds_before: dict[str, float],
    counts: dict[str, int],
    seconds: dict[str, float],
) -> dict[str, tuple[int, float]]:
    """Each name's count and seconds that rose between two readings."""
    return {
        name: (
            count - counts_before.get(name, 0),
            seconds[name] - seconds_before.get(name, 0.0),
        )
        for name, count in counts.items()
        if count > counts_before.get(name, 0)
    }
