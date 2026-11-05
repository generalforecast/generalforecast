# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax

from generalforecast.constants import LOCAL_DEVICE_TIER
from generalforecast.runtime.device import (
    UNTUNED_OPTIONS,
    available_memory,
    device_memory,
    host_memory,
)
from generalforecast.runtime.memory import tree_bytes
from generalforecast.runtime.shapes import bucket_size

logger = logging.getLogger(__name__)


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

    def holding(self, held: int) -> "MemoryBudget":
        """This budget with `held` bytes of an accelerator's memory taken for the worker's life, as loaded models' parameters are.

        The pool, calls and everything sized from `device` then divide only
        what is left. A CPU's device is a share of host memory, which holds
        the parameters beside it, so its budget stays as it is.
        """
        if self.cpu:
            return self
        return dataclasses.replace(self, device=max(0, self.device - held))

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


# A kind's first call measures its memory compiled without autotuning unless
# the device has this many times its estimated bytes free. The estimate can
# fall many times short of the program (TabPFN-3.5's per-query decodes on
# Internet-Ads: 26x), and an autotuner out of memory keeps some of what it took.
MEASURE_SHARE = 4


def free_device_bytes() -> int | None:
    """Bytes the first local device's allocator has left, or None where it does not say, as on a CPU."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        return None
    if not stats or "bytes_limit" not in stats:
        return None
    return int(stats["bytes_limit"]) - int(stats.get("bytes_in_use", 0))


class StepExpired(Exception):
    """A step stopped between its calls because every request it answers is past its deadline.

    Its callers were answered as timed out and have gone, so the calls left
    would compute what nobody reads while every later step waits behind them.
    """


def autotune_ran_out(error: BaseException) -> bool:
    """Whether a call failed compiling, for want of device memory to autotune its kernels on.

    XLA's GPU autotuner times each candidate kernel on buffers of its own,
    beside what the device holds, and fails the compile when it cannot
    allocate them.
    """
    message = str(error)
    return out_of_memory(error) and (
        "Failed to get configs" in message or "autotun" in message.lower()
    )


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


def measured_bytes(call: Call, untuned: bool = False) -> int:
    """Bytes per item a call takes, by XLA's own analysis, or 0 when XLA cannot say.

    Its temporaries and outputs, with the items' own inputs, spread over the
    items; arguments every item shares are held anyway. Compiling here is the
    compile the call would do anyway, which JAX then finds in its caches;
    with `untuned`, for a call too large to autotune beside what the device
    holds, it compiles without autotuning instead, whose kernels time on
    buffers of their own.
    """
    try:
        jitted = cast(jax.stages.Wrapped, call.function)
        lowered = jitted.lower(*call.shared, *call.items)
        options = UNTUNED_OPTIONS if untuned else None
        analysis = lowered.compile(compiler_options=options).memory_analysis()
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


# FLOPs of the programs compiled or loaded outside a call, by function and shapes.
_compiled_flops: dict[Hashable, float] = {}


def estimated_flops(call: Call) -> float:
    """Floating-point operations one run of a call's program takes, by XLA's cost analysis; 0 when XLA cannot say.

    The program is the one the call compiled or loaded, which compiling here
    finds in JAX's caches, as `measured_bytes` does. The count is for the
    call's padded shapes.
    """
    try:
        jitted = cast(jax.stages.Wrapped, call.function)
        analysis = jitted.lower(*call.shared, *call.items).compile().cost_analysis()
    except Exception:  # noqa: BLE001
        return 0.0
    return flops_of(analysis)


def note_compiled(function: Any, arguments: tuple, compiled: Any) -> None:
    """Keeps the FLOPs of a program compiled or loaded outside a call, for `Chunks` to count at its calls."""
    try:
        flops = flops_of(compiled.cost_analysis())
    except Exception:  # noqa: BLE001
        flops = 0.0
    _compiled_flops[_signature(function, arguments)] = flops


def forget_compiled() -> None:
    """Drops what `note_compiled` kept, as programs are dropped."""
    _compiled_flops.clear()


def _noted_flops(call: Call) -> float:
    """A call's program's FLOPs as `note_compiled` kept them, or else as `estimated_flops` reads them."""
    noted = _compiled_flops.get(_signature(call.function, (*call.shared, *call.items)))
    return estimated_flops(call) if noted is None else noted


def _signature(function: Any, arguments: tuple) -> Hashable:
    """A function with its arguments' shapes and dtypes, which pick its program."""
    return (
        function,
        tuple(
            (getattr(leaf, "shape", None), str(getattr(leaf, "dtype", type(leaf))))
            for leaf in jax.tree.leaves(arguments)
        ),
    )


def flops_of(analysis: Any) -> float:
    """The `flops` of an XLA cost analysis, a dict or, in older JAX, a list of one per program; 0 when it has none."""
    if isinstance(analysis, (list, tuple)):
        return sum(flops_of(part) for part in analysis)
    if not isinstance(analysis, dict):
        return 0.0
    flops = analysis.get("flops")
    if not isinstance(flops, (int, float)) or not flops > 0:
        return 0.0
    return float(flops)


def _shapes(tree: Any) -> Any:
    """A tree's leaves as their shapes and dtypes alone."""
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, weak_type=getattr(leaf, "weak_type", False)
        ),
        tree,
    )


# Stages each of whose items is a whole context's encode, and the most of
# them a call may hold for its items to count as large: a padded item then
# costs about what a real one does, and a call more costs little, so such
# calls are not padded. Small contexts, of which a call holds many, still pad
# to their bucket, in one call and one program.
WHOLE_CONTEXT_STAGES = frozenset({"encode", "fit_encode"})
LARGE_ITEMS = 128


def _unpadded(stage: str, padded: int, left: int, size: int) -> int:
    """`padded`, or for `stage`'s large contexts, the most powers of two `left` items fill."""
    if stage not in WHOLE_CONTEXT_STAGES or size > LARGE_ITEMS or left >= padded:
        return padded
    return 1 << (left.bit_length() - 1)


def _bucket_of(buckets: Sequence[int], items: int, size: int) -> int:
    """The least of `buckets` within `size` that takes `items`, or else the most within it."""
    within = [bucket for bucket in buckets if bucket <= size] or [1]
    return min((bucket for bucket in within if bucket >= items), default=max(within))


class Chunks:
    """Runs many items of one kind of call in as few calls as fit a call budget.

    A kind of call takes the bytes per item its estimate gives until XLA
    measures its first compiled call, which counts what the call holds in
    between, and twice as many after each call that runs out of memory. A
    chunk is a power of two, so one padded to its bucket stays in the budget.
    A run of more items than a chunk pads its last call to the chunk, and a
    run of fewer pads its one call to their bucket, so a kind of call
    compiles for the chunk and for the buckets below it that single calls
    take, not for every remainder. Encodes of contexts so large that a call
    holds at most `LARGE_ITEMS` are not padded: they go in calls of the
    powers of two they fill, nineteen as sixteen, two and one.
    """

    def __init__(
        self,
        counted: Callable[[str, float, float], None],
        compiled: Callable[[str], None] | None = None,
        timed: Callable[[str, float], None] | None = None,
        seen: Callable[[Call], None] | None = None,
        noted: Callable[[str, Hashable, Call, float], None] | None = None,
    ) -> None:
        """Sets up with nothing measured.

        Args:
            counted: Takes each call's stage, its seconds and its program's
                FLOPs, as `estimated_flops` gives them once per program.
            compiled: Takes the stage of each call whose shape its program had
                not seen, which compiled, or came from the compile cache.
            timed: Takes each call's seconds until it is handed to the
                device, as `compile` for a shape its program had not seen
                and `dispatch` otherwise, and its `wait` seconds, until the
                device finishes it. Without it, no call is split so.
            seen: Takes each call whose shape its program had not seen, once
                it ran, and a kind's first call at each count in this
                process, which a program compiled at start had not.
            noted: Takes the stage, kind, call and seconds of each call whose
                shape its program had not seen, to say what compiled.
        """
        self._counted = counted
        self._compiled = compiled
        self.timed = timed
        self._seen = seen
        self._noted = noted
        # Bytes per item of each kind, as estimated or measured, whichever is more.
        self._item_bytes: dict[Hashable, int] = {}
        self._measured: set[Hashable] = set()
        # How many times over a kind's bytes calls that ran out of memory showed.
        self._backoff: dict[Hashable, int] = {}
        # Programs compiled without autotuning, by function and argument
        # shapes, for calls whose autotuning ran out of device memory.
        self._untuned: dict[Hashable, Any] = {}
        # Signatures in `_untuned` whose program has since compiled with
        # autotuning, and those waiting for a quiet gap to.
        self._retuned: set[Hashable] = set()
        self._retuning: set[Hashable] = set()
        # Each untuned program's lowering and its call's bytes, until it retunes.
        self._lowered: dict[Hashable, tuple[Any, int]] = {}
        # Parts of one item a call takes, once a single item ran out of memory.
        self._parts: dict[Hashable, int] = {}
        # Padded counts each kind has run at, and those compiling ahead.
        self._counts: dict[Hashable, set[int]] = {}
        self._compiling: set[tuple[Hashable, int]] = set()
        # FLOPs of each kind's program at each padded count, read once.
        self._flops: dict[tuple[Hashable, int], float] = {}
        self._lock = threading.Lock()
        # Waits for compiles running beside calls, returning whether any were:
        # a call that runs out of memory beside one runs again once it is done.
        self.settle: Callable[[], bool] | None = None
        # Runs a task in a quiet gap between steps, such as `Programs.later`:
        # a program compiled without autotuning compiles with it there, once
        # the device has room for it.
        self.later: Callable[[Callable[[], None]], None] | None = None
        # Whether the running step's requests are all past their deadline,
        # checked before each call: such a step stops (`StepExpired`).
        self.expired: Callable[[], bool] | None = None

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
        ahead: Callable[[Call, Callable[[bool], None]], None] | None = None,
        buckets: Sequence[int] | None = None,
    ) -> None:
        """Runs `count` items of a kind of call, as many to a call as fit `budget`.

        With `ahead`, a kind that has run before runs only at padded counts it
        has run at, and a count it has not compiles on the side: see
        `_compiled_count`. A call that runs out of memory at such a count,
        larger than its items' own, runs again at their own, compiled in the
        call.

        Args:
            stage: The stage each call counts under.
            key: The kind of call: what it compiles to.
            count: Items to run.
            budget: Bytes one call may take.
            estimate: Bytes per item, before anything is measured.
            prepare: Makes the call for items `first` to `stop`, exclusive,
                padded to the third argument's count.
            consume: Takes each call's first and last item, exclusive, and its outputs.
            ahead: Compiles a call without running it, then calls its second
                argument, on a thread of its own.
            buckets: The only padded counts calls take, with 1 among them, in
                place of every power of two: each pads to the least that
                takes the items left, or else takes the most within the chunk.

        Raises:
            JaxRuntimeError: If a call fails other than for want of memory, or
                one item alone runs out of it.
            StepExpired: If `expired` says the step's requests are all past
                their deadline before a call.
        """
        # Whether the next call runs at its items' own count, compiled in the
        # call, rather than at a larger one `_compiled_count` reuses.
        exact = False
        first = 0
        # Seconds measuring a kind took, counted with the call that reuses
        # the program the measure compiled.
        measuring = 0.0
        # Whether the call about to run already waited for compiles beside it.
        settled = False
        while first < count:
            if self.expired is not None and self.expired():
                raise StepExpired(
                    f"stopped a step's {stage} calls with {count - first} of "
                    f"{count} items left: every request it answers is past its "
                    "deadline"
                )
            size = self.size(key, budget, estimate)
            if buckets is None:
                padded = size if count > size else bucket_size(count, 1)
                padded = _unpadded(stage, padded, count - first, size)
            else:
                padded = _bucket_of(buckets, count - first, size)
            own = padded
            if ahead is not None and not exact:
                padded = self._compiled_count(
                    key, padded, size, count, first, prepare, ahead
                )
            exact = False
            stop = min(count, first + padded)
            call = prepare(first, stop, padded)
            if self._first_measure(key):
                started = time.perf_counter()
                # Before its measure a kind's bytes are an estimate, which
                # can fall far short of what its program holds, so its
                # measure autotunes only with four times the estimate free.
                cautious = self._crowded(key, padded, MEASURE_SHARE)
                measured = measured_bytes(call, untuned=cautious)
                measuring += time.perf_counter() - started
                with self._lock:
                    larger = measured > self._item_bytes[key]
                    if larger:
                        self._item_bytes[key] = measured
                if larger and self.size(key, budget, estimate) < stop - first:
                    continue
            started = time.perf_counter()
            traces = _traces(call.function)
            untuned = self._untuned_of(call)
            # Autotuning its kernels would take buffers of their own the size
            # of its own beside what the device holds now.
            crowded = untuned is None and self._crowded(key, padded)
            needed = self._call_bytes(key, padded)
            if crowded and self._untune(
                call, needed, "is too large to autotune beside steps"
            ):
                untuned = self._untuned_of(call)
            if untuned is not None:
                self._retune(call)
            try:
                if untuned is None:
                    outputs = call.function(*call.shared, *call.items)
                else:
                    outputs = untuned(*call.shared, *call.items)
                dispatched = time.perf_counter() if self.timed is not None else 0.0
                jax.block_until_ready(outputs)
            except jax.errors.JaxRuntimeError as error:
                if (
                    untuned is None
                    and autotune_ran_out(error)
                    and self._untune(call, self._call_bytes(key, padded))
                ):
                    # Its kernels autotuned on buffers the device could not
                    # hold: the same call again, compiled without autotuning.
                    continue
                if (
                    out_of_memory(error)
                    and not settled
                    and self.settle is not None
                    and self.settle()
                ):
                    # A compile beside the call held the device's memory as it
                    # autotuned: the same call again, now that it is done.
                    settled = True
                    continue
                if out_of_memory(error) and padded > own:
                    # A larger count's program, reused while the items' own
                    # compiles on the side, needs more than the device has
                    # free: the items' own count, compiled in the call.
                    logger.warning(
                        "a %s call of %d items ran out of device memory at a "
                        "compiled count of %d; it runs at its own count of %d",
                        stage,
                        stop - first,
                        padded,
                        own,
                    )
                    exact = True
                    continue
                if not out_of_memory(error) or stop - first == 1:
                    raise
                # XLA's measure fell short of what the call needed: halve the
                # call, from here on.
                with self._lock:
                    self._backoff[key] = 2 * self._backoff.get(key, 1)
                continue
            finished = time.perf_counter()
            settled = False
            compiled = _traces(call.function) > traces
            with self._lock:
                flops = self._flops.get((key, padded))
            if flops is None:
                # Read once per program: here if this call compiled it, else
                # where it was compiled ahead or at start.
                if untuned is not None:
                    # Read from the program the call ran, which compiling the
                    # call's own would autotune.
                    try:
                        flops = flops_of(untuned.cost_analysis())
                    except Exception:  # noqa: BLE001
                        flops = 0.0
                else:
                    flops = estimated_flops(call) if compiled else _noted_flops(call)
                with self._lock:
                    self._flops[(key, padded)] = flops
            self._counted(stage, measuring + finished - started, flops)
            if compiled and self._noted is not None:
                self._noted(stage, key, call, finished - started)
            if self.timed is not None:
                self.timed("compile" if compiled else "dispatch", dispatched - started)
                self.timed("wait", finished - dispatched)
            measuring = 0.0
            with self._lock:
                first_use = padded not in self._counts.get(key, ())
            self._ran(key, padded)
            if ahead is not None and buckets is not None:
                # The first of a kind's calls compiles in it; its other
                # buckets compile on the side from then on, before steps ask.
                for bucket in buckets:
                    if bucket <= size:
                        self._start(key, bucket, count, first, prepare, ahead)
            if self._compiled is not None and compiled:
                self._compiled(stage)
            if self._seen is not None and (compiled or first_use):
                self._seen(call)
            consume(first, stop, outputs)
            # Freed before the next call, so the device never holds two calls' outputs.
            del outputs
            first = stop

    def _untuned_of(self, call: Call) -> Any:
        """The program a call runs in place of its own: compiled without autotuning for want of room, or since with it; else None."""
        if not self._untuned:
            return None
        with self._lock:
            return self._untuned.get(
                _signature(call.function, (*call.shared, *call.items))
            )

    def _crowded(self, key: Hashable, padded: int, share: int = 2) -> bool:
        """Whether a count a kind has not run at would take more than one `share`th of the device's free memory, by its items' bytes.

        XLA's autotuner times each fused kernel on buffers of its own, as
        large as the kernel's, beside what the device holds. Past half the
        free memory it runs out, and what it took is not all given back.
        """
        with self._lock:
            if padded in self._counts.get(key, ()):
                return False
        needed = self._call_bytes(key, padded)
        free = free_device_bytes()
        return bool(needed) and free is not None and share * needed > free

    def _call_bytes(self, key: Hashable, padded: int) -> int:
        """Bytes a kind's call of `padded` items takes, by its items' measure; 0 before any."""
        with self._lock:
            return self._item_bytes.get(key, 0) * self._backoff.get(key, 1) * padded

    def _untune(
        self,
        call: Call,
        needed: int,
        why: str = "ran out of device memory to autotune its kernels",
    ) -> bool:
        """Compiles a call's program without autotuning, for it and later calls of its shapes; False if it cannot be.

        `needed` is the call's bytes: the program compiles with autotuning
        again in a quiet gap once the device has twice that free, and
        serves its later calls from then on (`_retune`).
        """
        arguments = (*call.shared, *call.items)
        try:
            lowered = cast(jax.stages.Wrapped, call.function).lower(*arguments)
            program = lowered.compile(compiler_options=UNTUNED_OPTIONS)
        except Exception:  # noqa: BLE001
            return False
        # A compiled program takes its call's arguments as they are, so only
        # one called with every argument, none of them static, serves it.
        if program.in_tree != jax.tree.structure((arguments, {})):
            return False
        logger.warning("a call's program %s; it runs compiled without autotuning", why)
        signature = _signature(call.function, arguments)
        with self._lock:
            self._untuned[signature] = program
            self._lowered[signature] = (lowered, needed)
        return True

    def _retune(self, call: Call) -> None:
        """Compiles an untuned call's program with autotuning in a quiet gap, once the device has room; its later calls run it.

        Untuned kernels run a decode up to about twice as slow, so a program
        compiled without autotuning for want of room stays so only while
        the device lacks it. Without a quiet gap to compile in, as with no
        compiling ahead, it stays untuned.
        """
        if self.later is None:
            return
        signature = _signature(call.function, (*call.shared, *call.items))
        with self._lock:
            if signature in self._retuned or signature in self._retuning:
                return
            pending = self._lowered.get(signature)
            if pending is None:
                return
            self._retuning.add(signature)
        lowered, needed = pending

        def retune() -> None:
            try:
                free = free_device_bytes()
                if free is not None and 2 * needed > free:
                    # Still no room: the next call of it asks again.
                    return
                tuned = lowered.compile()
            except Exception as error:  # noqa: BLE001
                logger.warning("autotuning an untuned program failed: %s", error)
                return
            finally:
                with self._lock:
                    self._retuning.discard(signature)
            with self._lock:
                self._untuned[signature] = tuned
                self._retuned.add(signature)
                self._lowered.pop(signature, None)

        self.later(retune)

    def _compiled_count(
        self,
        key: Hashable,
        padded: int,
        size: int,
        count: int,
        first: int,
        prepare: Callable[[int, int, int], Call],
        ahead: Callable[[Call, Callable[[bool], None]], None],
    ) -> int:
        """The padded count to run items from `first` at, of a kind that compiles on the side.

        A kind's first call compiles for `padded` in the call. After it, a
        count the kind has not run at compiles on the side, through `ahead`,
        while the items run at the least count it has run at that takes them
        all, or else in calls of the most, within `size`. On an accelerator
        a program takes many seconds to compile, and padding a few items
        more takes far less than every call behind waiting for it.
        """
        with self._lock:
            counts = [known for known in self._counts.get(key, ()) if known <= size]
            if not counts or padded in counts:
                return padded
        self._start(key, padded, count, first, prepare, ahead)
        left = count - first
        return min((known for known in counts if known >= left), default=max(counts))

    def _start(
        self,
        key: Hashable,
        padded: int,
        count: int,
        first: int,
        prepare: Callable[[int, int, int], Call],
        ahead: Callable[[Call, Callable[[bool], None]], None],
    ) -> None:
        """Compiles a kind for a padded count on the side, unless it has it or is compiling it."""
        with self._lock:
            if (key, padded) in self._compiling or padded in self._counts.get(key, ()):
                return
            self._compiling.add((key, padded))
        made = prepare(first, min(count, first + padded), padded)
        # Compiled from shapes, so the side holds no items' memory.
        call = Call(
            made.function, made.shared, _shapes(made.items), padded, made.program
        )
        del made

        def done(compiled: bool) -> None:
            if not compiled:
                # A later call starts it again.
                with self._lock:
                    self._compiling.discard((key, padded))
                return
            self._ran(key, padded)
            if self._seen is not None:
                self._seen(call)

        ahead(call, done)

    def _first_measure(self, key: Hashable) -> bool:
        """Whether a kind's first call is to be measured; only one call of each is."""
        with self._lock:
            if key in self._measured:
                return False
            self._measured.add(key)
            return True

    def ran(self, key: Hashable) -> bool:
        """Whether a kind has a program for any padded count."""
        with self._lock:
            return bool(self._counts.get(key))

    def mark(self, key: Hashable, padded: int) -> None:
        """Counts a padded count a kind was compiled for elsewhere."""
        self._ran(key, padded)

    def _ran(self, key: Hashable, padded: int) -> None:
        """Counts a padded count a kind has a program for."""
        with self._lock:
            self._counts.setdefault(key, set()).add(padded)
            self._compiling.discard((key, padded))

    def forget_counts(self) -> None:
        """Forgets the padded counts each kind has a program for, as when every program is dropped.

        Each kind's next call then compiles in it, as its first did, and its
        other counts compile on the side again.
        """
        with self._lock:
            self._counts.clear()
            self._compiling.clear()

    def parts(self, key: Hashable) -> int | None:
        """The most parts of one item a call of `key` takes, or None while any number fit."""
        with self._lock:
            return self._parts.get(key)

    def split(self, key: Hashable, parts: int) -> None:
        """Limits a kind of call to `parts` parts of an item, after one item ran out of memory."""
        with self._lock:
            self._parts[key] = parts

    def relax(self) -> None:
        """Forgets what calls that ran out of memory taught, as when the pool is sized again."""
        with self._lock:
            self._backoff.clear()
            self._parts.clear()
