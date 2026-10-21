import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, cast

import jax

from generalforecast.runtime.device import available_memory, device_memory, host_memory
from generalforecast.runtime.memory import tree_bytes
from generalforecast.runtime.shapes import bucket_size

# The most device memory a local client keeps contexts in by default.
LOCAL_DEVICE_TIER = 4 << 30


@dataclass(frozen=True)
class MemoryBudget:
    """How a worker's memory divides between its caches and its calls.

    Every default share of memory comes from here: a local client's and a
    server's cache tiers, a server's row memory, and what one compiled call
    may take beside the device pool. A server's workers report theirs when
    they start, and the server takes its tiers from them unless its
    configuration sets them.

    Attributes:
        device: Bytes of memory the worker counts as its device's.
        host: Bytes of memory the process may use, its container's limit included.
        cpu: Whether the device is the CPU, so its memory is the host's.
        workers: Workers sharing the host's memory.
        local: Whether a local client shares the machine with the programs
            around it, rather than a server having it to itself.
    """

    device: int
    host: int
    cpu: bool
    workers: int = 1
    local: bool = False

    @classmethod
    def measure(cls, workers: int = 1, local: bool = False) -> "MemoryBudget":
        """This machine's budget.

        Args:
            workers: Workers sharing the host's memory.
            local: Whether it is for a local client.

        Returns:
            The budget, from this process's device and host memory.
        """
        cpu = jax.default_backend() == "cpu"
        return cls(device_memory(), host_memory(), cpu, max(1, workers), local)

    @property
    def device_tier(self) -> int:
        """Device bytes for encoded contexts.

        Half the device on a server; on a local client a quarter, at most
        `LOCAL_DEVICE_TIER`, which leaves room for the programs around it.
        """
        if self.local:
            return min(self.device // 4, LOCAL_DEVICE_TIER)
        return self.device // 2

    @property
    def host_tier(self) -> int:
        """Host bytes for a server's contexts evicted from the device.

        A quarter of the host's memory, split between its workers. A local
        client's host tier is as large as its device tier.
        """
        return self.host // 4 // self.workers

    @property
    def row_memory(self) -> int:
        """Host bytes for a server's table rows: a quarter of the host's, split between workers."""
        return self.host // 4 // self.workers

    def call_bytes(self, pool: int) -> int:
        """Bytes one compiled call's inputs, outputs and temporaries may take.

        Half of what a device pool of `pool` bytes leaves, so a call never needs
        memory the pool holds; at most a quarter of the device, or an eighth on
        a CPU, whose memory the host's caches and other programs share; and at
        least a sixteenth.

        Args:
            pool: Bytes of the device pool.

        Returns:
            The call budget in bytes.
        """
        share = 8 if self.cpu else 4
        return max(
            self.device // 16, min(self.device // share, (self.device - pool) // 2)
        )

    def window_bytes(self, pool: int) -> int:
        """Bytes of contexts a decode gathers before its call: half a call's, for the call's copy.

        Args:
            pool: Bytes of the device pool.

        Returns:
            The window in bytes.
        """
        return self.call_bytes(pool) // 2

    def short_of_memory(self) -> bool:
        """Whether less than a quarter of the host's memory is free.

        Where free memory cannot be told it is not short, so programs and
        caches are kept rather than dropped on every check.
        """
        free = available_memory()
        return free is not None and free < self.host // 4


def out_of_memory(error: BaseException) -> bool:
    """Whether a compiled call failed for want of device memory."""
    if not isinstance(error, jax.errors.JaxRuntimeError):
        return False
    message = str(error)
    return "RESOURCE_EXHAUSTED" in message or "out of memory" in message.lower()


@dataclass(frozen=True)
class Call:
    """One compiled call over a chunk of items.

    Attributes:
        function: The compiled function.
        shared: Arguments every item shares, such as parameters and the pool.
        items: Arguments stacked over the chunk's items.
        padded: Items the call takes, the chunk padded as `Chunks.run` asks.
        program: The `Programs` key `function` is compiled under, for a
            kind of call a later process can compile again; None otherwise.
    """

    function: Callable[..., Any]
    shared: tuple
    items: tuple
    padded: int
    program: tuple | None = None


def _traces(function: Callable[..., Any]) -> int:
    """Shapes a compiled function has been called with, as JAX counts them; 0 for one it cannot say."""
    cache_size = getattr(function, "_cache_size", None)
    return int(cache_size()) if callable(cache_size) else 0


def measured_bytes(call: Call) -> int:
    """Bytes per item a call takes, by XLA's own analysis, or 0 when XLA cannot say.

    Its temporaries and outputs, with the items' own inputs, spread over the
    items; arguments every item shares are held anyway. Compiling here is the
    compile the call would do anyway.
    """
    try:
        jitted = cast(jax.stages.Wrapped, call.function)
        analysis = jitted.lower(*call.shared, *call.items).compile().memory_analysis()
    except Exception:  # noqa: BLE001
        return 0
    if analysis is None:
        return 0
    total = (
        analysis.temp_size_in_bytes
        + analysis.output_size_in_bytes
        + tree_bytes(call.items)
    )
    return total // max(1, call.padded)


class Chunks:
    """Runs many items of one kind of call in as few calls as fit a call budget.

    A kind of call takes the bytes per item its estimate gives until XLA
    measures its first compiled call, which counts what the call holds in
    between, and twice as many after each call that runs out of memory. A
    chunk is a power of two, so one padded to its bucket stays in the budget.
    A run of more items than a chunk pads its last call to the chunk, and a
    run of fewer pads its one call to their bucket, so a kind of call
    compiles for the chunk and for the buckets below it that single calls
    take, not for every remainder.
    """

    def __init__(
        self,
        counted: Callable[[str, float], None],
        compiled: Callable[[str], None] | None = None,
        timed: Callable[[str, float], None] | None = None,
        seen: Callable[[Call], None] | None = None,
    ) -> None:
        """Sets up with nothing measured.

        Args:
            counted: Takes each call's stage and its seconds.
            compiled: Takes the stage of each call whose shape its program had
                not seen, which compiled, or came from the compile cache.
            timed: Takes each call's seconds until it is handed to the
                device, as `compile` for a shape its program had not seen
                and `dispatch` otherwise, and its `wait` seconds, until the
                device finishes it. Without it, no call is split so.
            seen: Takes each call whose shape its program had not seen, once
                it ran.
        """
        self._counted = counted
        self._compiled = compiled
        self.timed = timed
        self._seen = seen
        # Bytes per item of each kind, as estimated or measured, whichever is more.
        self._item_bytes: dict[Hashable, int] = {}
        self._measured: set[Hashable] = set()
        # How many times over a kind's bytes calls that ran out of memory showed.
        self._backoff: dict[Hashable, int] = {}
        # Parts of one item a call takes, once a single item ran out of memory.
        self._parts: dict[Hashable, int] = {}
        self._lock = threading.Lock()

    def size(self, key: Hashable, budget: int, estimate: Callable[[], int]) -> int:
        """Items of a kind one call of at most `budget` bytes takes.

        Args:
            key: The kind of call.
            budget: Bytes the call may take.
            estimate: Bytes per item, before anything is measured.

        Returns:
            A power of two, at least 1.
        """
        with self._lock:
            if key not in self._item_bytes:
                self._item_bytes[key] = estimate()
            bytes = self._item_bytes[key] * self._backoff.get(key, 1)
        size = max(1, budget // max(1, bytes))
        return 1 << (size.bit_length() - 1)

    def run(
        self,
        stage: str,
        key: Hashable,
        count: int,
        budget: int,
        estimate: Callable[[], int],
        prepare: Callable[[int, int, int], Call],
        consume: Callable[[int, int, Any], None],
    ) -> None:
        """Runs `count` items of a kind of call, as many to a call as fit `budget`.

        Args:
            stage: The stage each call counts under.
            key: The kind of call: what it compiles to.
            count: Items to run.
            budget: Bytes one call may take.
            estimate: Bytes per item, before anything is measured.
            prepare: Makes the call for items `first` to `stop`, exclusive,
                padded to the third argument's count.
            consume: Takes each call's first and last item, exclusive, and its outputs.

        Raises:
            JaxRuntimeError: If a call fails other than for want of memory, or
                one item alone runs out of it.
        """
        first = 0
        # Seconds measuring a kind took, counted with the call that reuses
        # the program the measure compiled.
        measuring = 0.0
        while first < count:
            size = self.size(key, budget, estimate)
            stop = min(count, first + size)
            call = prepare(first, stop, size if count > size else bucket_size(count, 1))
            if key not in self._measured:
                self._measured.add(key)
                started = time.perf_counter()
                measured = measured_bytes(call)
                measuring += time.perf_counter() - started
                with self._lock:
                    larger = measured > self._item_bytes[key]
                    if larger:
                        self._item_bytes[key] = measured
                if larger and self.size(key, budget, estimate) < stop - first:
                    continue
            started = time.perf_counter()
            traces = _traces(call.function)
            try:
                outputs = call.function(*call.shared, *call.items)
                dispatched = time.perf_counter() if self.timed is not None else 0.0
                jax.block_until_ready(outputs)
            except jax.errors.JaxRuntimeError as error:
                if not out_of_memory(error) or stop - first == 1:
                    raise
                # XLA's measure fell short of what the call needed: halve the
                # call, from here on.
                with self._lock:
                    self._backoff[key] = 2 * self._backoff.get(key, 1)
                continue
            finished = time.perf_counter()
            self._counted(stage, measuring + finished - started)
            compiled = _traces(call.function) > traces
            if self.timed is not None:
                self.timed("compile" if compiled else "dispatch", dispatched - started)
                self.timed("wait", finished - dispatched)
            measuring = 0.0
            if self._compiled is not None and compiled:
                self._compiled(stage)
            if self._seen is not None and compiled:
                self._seen(call)
            consume(first, stop, outputs)
            # Freed before the next call, so the device never holds two calls' outputs.
            del outputs
            first = stop

    def parts(self, key: Hashable) -> int | None:
        """The most parts of one item a call of `key` takes, or None while any number fit."""
        return self._parts.get(key)

    def split(self, key: Hashable, parts: int) -> None:
        """Limits a kind of call to `parts` parts of an item, after one item ran out of memory."""
        self._parts[key] = parts

    def relax(self) -> None:
        """Forgets what calls that ran out of memory taught, as when the pool is sized again."""
        with self._lock:
            self._backoff.clear()
            self._parts.clear()
