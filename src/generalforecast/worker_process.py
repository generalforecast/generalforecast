# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A worker process for `gf-server`: `python -m generalforecast.worker_process --socket PATH --model NAME=SPEC`.

The server starts one per accelerator. The worker loads its models, connects
to the server's socket, and answers each frame until the server goes away.
"""

import argparse
import contextlib
import importlib
import itertools
import json
import os
import queue
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import jax

from generalforecast import protocol
from generalforecast.constants import SAMPLE_INTERVAL
from generalforecast.models import registry
from generalforecast.profiling import memory
from generalforecast.runtime.budget import MemoryBudget
from generalforecast.runtime.device import (
    default_settings,
    portable_cpu_programs,
    without_command_buffers,
)
from generalforecast.system import pin_cpus
from generalforecast.worker import Worker

Frame = tuple[dict[str, Any], list[bytes]]
# The event JAX records as XLA finishes compiling a program.
BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"


def main(arguments: Sequence[str] | None = None) -> int:
    """Runs a worker until its server disconnects; returns its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", action="append", help="NAME=SPEC, repeated")
    parser.add_argument("--bucket-minimum", type=int, default=8)
    parser.add_argument("--slot-bytes", type=int, default=1 << 20)
    parser.add_argument("--disk-path", type=Path, help="the disk tier's file, if any")
    parser.add_argument(
        "--workers", type=int, default=1, help="workers sharing this machine's memory"
    )
    parser.add_argument(
        "--timers", action="store_true", help="time each part of a step"
    )
    parser.add_argument(
        "--cpus",
        help="the cores to run on, comma-separated, if this worker has its own",
    )
    options = parser.parse_args(arguments)
    if options.cpus:
        # Before JAX starts its backend, which sizes XLA's thread pools from
        # the cores the process may run on.
        pin_cpus(int(cpu) for cpu in options.cpus.split(","))

    # Before `default_settings` starts JAX's backend, which reads XLA's flags.
    without_command_buffers()
    portable_cpu_programs()
    budget = MemoryBudget.measure(workers=options.workers)
    worker = Worker(
        default_settings(),
        options.bucket_minimum,
        options.slot_bytes,
        options.disk_path,
        budget=budget,
        timers=options.timers,
    )
    for module in filter(None, os.environ.get("GF_WORKER_MODULES", "").split(",")):
        importlib.import_module(module)
    names: dict[str, int] = {}
    for entry in options.model or []:
        name, spec = entry.split("=", 1)
        names[name] = len(names)
        worker.add_model(names[name], *registry.load_engine_model(spec), spec=spec)

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(options.socket)
    ready = protocol.Ready(
        protocol=protocol.PROTOCOL_VERSION,
        models=list(names),
        device_memory=budget.device,
        device_bytes=budget.device_tier,
        host_bytes=budget.host_tier,
        row_memory_bytes=budget.row_memory,
    )
    protocol.write_frame(connection, protocol.reply(ready))
    # The server pipelines frames, so reading the next frame and sending the last
    # reply overlap the current step's compute.
    frames: queue.Queue[tuple[dict[str, Any], list[memoryview]] | None] = queue.Queue(2)
    replies: queue.Queue[Callable[[], Frame] | None] = queue.Queue()
    failed = threading.Event()

    def read() -> None:
        try:
            while True:
                frames.put(protocol.read_frame(connection))
        except ConnectionError:
            pass
        except Exception:  # noqa: BLE001
            # The stream is out of step after a bad frame, so the worker stops.
            traceback.print_exc()
            failed.set()
        frames.put(None)

    def write() -> None:
        try:
            while (reply := replies.get()) is not None:
                protocol.write_frame(connection, *reply())
        except Exception as error:  # noqa: BLE001
            if not isinstance(error, ConnectionError):
                traceback.print_exc()
            sys.stderr.flush()
            # The server waits for this reply; exiting lets it restart the worker.
            os._exit(1)

    progress = protocol.reply(protocol.Progress())

    def compiled(event: str, duration_secs: float, **details: Any) -> None:
        # A step that compiles programs for many new shapes runs long, so each
        # compile tells the server the step is moving rather than hung.
        if event == BACKEND_COMPILE:
            replies.put(lambda: (progress, []))

    jax.monitoring.register_event_duration_secs_listener(compiled)
    threading.Thread(target=read, daemon=True).start()
    writer = threading.Thread(target=write, daemon=True)
    writer.start()

    def progressed() -> None:
        replies.put(lambda: (progress, []))

    def announced(keys: list[str]) -> None:
        frame = protocol.reply(protocol.Compiled(keys))
        replies.put(lambda: (frame, []))

    worker.announce = announced

    while (frame := frames.get()) is not None:
        replies.put(_answer(worker, names, progressed, *frame))
        if worker.lost:
            # The server starts the worker again, forgetting every cached context.
            failed.set()
            break
    replies.put(None)
    writer.join()
    return 1 if failed.is_set() else 0


def _answer(
    worker: Worker,
    names: dict[str, int],
    progressed: Callable[[], None],
    header: dict[str, Any],
    buffers: list[memoryview],
) -> Callable[[], Frame]:
    """Handles one frame; returns what formats its reply, for the writer to call.

    `progressed` tells the server a long answer is moving, before its reply.
    """
    tracing = _tracing
    if tracing is None or header.get("kind") != "step":
        return _handled(worker, names, progressed, header, buffers)
    with tracing.step(worker):
        return _handled(worker, names, progressed, header, buffers)


def _handled(
    worker: Worker,
    names: dict[str, int],
    progressed: Callable[[], None],
    header: dict[str, Any],
    buffers: list[memoryview],
) -> Callable[[], Frame]:
    """Answers one frame, outside any trace."""
    answer: protocol.Reply
    try:
        before = worker.stats.snapshot()
        started = time.perf_counter()
        with worker.stats.timed("frame"):
            request = protocol.request(header, buffers)
            if worker.timers and isinstance(request, protocol.Step):
                memory.reset()
        match request:
            case protocol.Step() as step:
                # The step's seconds run from its frame's decode, above, to
                # its reply's conversion, so its parts add up to them.
                predictions, failed = worker.run_apart(step)
                with worker.stats.timed("reply"):
                    frame = protocol.predictions(
                        predictions, failed, deferred=worker.deferred
                    )
                    held = memory.reading() if worker.timers else None
                frame[0].update(
                    protocol.timing(
                        time.perf_counter() - started,
                        worker.stats.stages_since(before),
                        worker.stats.parts_since(before),
                        held,
                    )
                )
                return lambda: frame
            case protocol.Record(seconds=seconds, path=path, trace=trace):
                answer = _recorded(seconds, path, trace)
            case protocol.Profile(model=model, features=features):
                answer = _profiled(worker, names[model], features)
            case protocol.FitLimits(model=model, features=features):
                state_rows, fused_rows = worker.fit_rows(names[model], features)
                answer = protocol.Limited(_limit(state_rows), _limit(fused_rows))
            case protocol.Check(
                model=model, classes=classes, features=features, series=series
            ):
                try:
                    if worker.forecaster(names[model]):
                        worker.check_series(names[model], features, rows=series)
                    else:
                        worker.check(names[model], classes, features)
                    answer = protocol.Done()
                except ValueError as error:
                    answer = protocol.Invalid(str(error))
            case protocol.Allocate(device_slots, host_slots, disk_slots):
                worker.allocate(device_slots, host_slots, disk_slots)
                # The server takes steps only after this reply, so the
                # programs earlier workers compiled are ready before them.
                before = worker.programs.warmed, worker.programs.warm_seconds
                worker.warm_start(progressed)
                answer = protocol.Warmed(
                    worker.programs.warmed - before[0],
                    worker.programs.warm_seconds - before[1],
                )
            case protocol.Controls(requests):
                failures = []
                for index, control in enumerate(requests):
                    try:
                        _control(worker, names, control, progressed)
                    except Exception as error:  # noqa: BLE001
                        message = f"{type(error).__name__}: {error}"
                        failures.append(protocol.ControlFailure(index, message))
                answer = protocol.Controlled(failures) if failures else protocol.Done()
            case protocol.AddTable() | protocol.RemoveTable() | protocol.Apply():
                _control(worker, names, request, progressed)
                answer = protocol.Done()
    except Exception as error:  # noqa: BLE001
        answer = protocol.Failed(f"{type(error).__name__}: {error}")
    header = protocol.reply(answer)
    return lambda: (header, [])


def _control(
    worker: Worker,
    names: dict[str, int],
    control: protocol.AddModel
    | protocol.AddTable
    | protocol.RemoveTable
    | protocol.Apply,
    progressed: Callable[[], None],
) -> None:
    """Loads a model, adds or removes a table, or applies tiers' copies and drops.

    A model loads under its name unless the worker serves one by it, as at
    launch, and compiles what earlier workers recorded for it before any step.
    """
    match control:
        case protocol.AddModel(name, spec):
            if name not in names:
                worker.add_model(
                    len(names), *registry.load_engine_model(spec), spec=spec
                )
                names[name] = len(names)
                if worker.allocated:
                    worker.warm_start(progressed, models=[names[name]])
        case protocol.AddTable(
            table, model, classes, features, categorical, page_size, summary, series
        ):
            worker.add_table(
                table,
                names[model],
                classes,
                features,
                categorical,
                page_size,
                summary,
                series,
            )
        case protocol.RemoveTable(table):
            worker.remove_table(table)
        case protocol.Apply(movements):
            for movement in movements:
                worker.apply(movement)


def _profiled(worker: Worker, model: int, features: int) -> protocol.Profiled:
    """What a model needs per table."""
    if worker.forecaster(model):
        properties = worker.series_properties(model)
        sizes = worker.series_profile(model)
        return protocol.Profiled(
            bytes_per_row=sizes.per_row,
            representation_bytes=0,
            regression=True,
            class_slots=0,
            outputs=len(properties.quantiles),
            context_rows=None,
            state_bytes=0,
            column_summary=False,
            series=True,
            patch_bytes=sizes.per_patch,
            patch=properties.patch,
            output_patch=properties.output_patch,
            context_limit=properties.context_limit,
            pass_limit=properties.pass_limit,
            quantiles=list(properties.quantiles),
            reusable=properties.reusable,
            max_rows=properties.max_rows,
            extendable=worker.series_extendable(model),
        )
    # A table of series may have no columns; its shard refuses a tabular
    # model from this profile, which needs a column to size.
    features = max(features, 1)
    profile = worker.profile(model, features)
    properties = worker.properties(model)
    return protocol.Profiled(
        bytes_per_row=profile.cache_bytes_per_row,
        representation_bytes=profile.row_bytes,
        regression=properties.regression,
        class_slots=properties.class_slots,
        outputs=properties.num_outputs,
        context_rows=properties.context_rows,
        state_bytes=profile.state_bytes,
        categorical_state_bytes=profile.categorical_state_bytes,
        column_summary=properties.column_summary,
        unlabeled_context=properties.unlabeled_context,
        fit_bound=_limit(worker.fit_bound(model, features)),
    )


def _limit(rows: int) -> int | None:
    """A limit on a fit's rows as a server reads it: None where there is none."""
    return None if rows >= sys.maxsize else rows


class _Tracing:
    """JAX's profiler tracing into a directory, with each step's boundaries kept.

    A step is marked in the trace by a `protocol.STEP_EVENT` numbered from
    zero, and the steps that end before the trace stops are written to
    `protocol.STEPS_FILE` in the directory, each with its number, its start
    and end on `perf_counter` and the timing its reply carried.
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.numbers = itertools.count()
        self.steps: list[dict[str, Any]] = []
        self.open = True
        options = jax.profiler.ProfileOptions()
        # The sampler covers Python; tracing every Python call costs more.
        options.python_tracer_level = 0
        os.makedirs(directory, exist_ok=True)
        jax.profiler.start_trace(directory, profiler_options=options)

    @contextlib.contextmanager
    def step(self, worker: Worker) -> Iterator[None]:
        """Marks a step in the trace and keeps its boundaries and parts."""
        number = next(self.numbers)
        before = worker.stats.snapshot()
        started = time.perf_counter()
        with jax.profiler.StepTraceAnnotation(protocol.STEP_EVENT, step_num=number):
            yield
        ended = time.perf_counter()
        if self.open:
            self.steps.append(
                {
                    "step": number,
                    "started": started,
                    "ended": ended,
                    **protocol.timing(
                        ended - started,
                        worker.stats.stages_since(before),
                        worker.stats.parts_since(before),
                        memory.reading() if worker.timers else None,
                    ),
                }
            )

    def stop(self) -> None:
        """Stops the trace and writes the steps it covers."""
        self.open = False
        jax.profiler.stop_trace()
        path = os.path.join(self.directory, protocol.STEPS_FILE)
        with open(f"{path}.part", "w") as file:
            json.dump({"steps": self.steps}, file)
        os.replace(f"{path}.part", path)


# Whether a record is running, set by `_recorded` and cleared once its
# sampling and trace have stopped, so one runs at a time.
_recording = False
# The trace running now, which steps mark themselves in.
_tracing: _Tracing | None = None


def _recorded(seconds: float, path: str, trace: str | None = None) -> protocol.Reply:
    """Starts sampling this process's threads for `seconds`, their collapsed stacks written to `path`.

    The sampling runs on a thread of its own beside the steps, so the reply
    does not wait for it, and the window starts once the step in flight, if
    any, has answered. The file appears when the window ends. With `trace`,
    JAX's profiler traces the same window into that directory, and the steps
    that end inside it are listed beside the trace; see
    `generalforecast.profiling.trace`. A record asked for while one runs is refused.

    Raises:
        RuntimeError: If a record is running.
        OSError: If the file's directory cannot be made.
    """
    global _recording, _tracing
    if _recording:
        raise RuntimeError("a record is already running")
    os.makedirs(os.path.dirname(path) or os.curdir, exist_ok=True)
    tracing = None
    if trace is not None:
        tracing = _Tracing(trace)
        _tracing = tracing
    _recording = True
    threading.Thread(
        target=_captured, args=(seconds, path, tracing), name="gf-record", daemon=True
    ).start()
    return protocol.Done()


def _captured(seconds: float, path: str, tracing: _Tracing | None) -> None:
    """Samples for `seconds`, then ends the trace of the same window, if one."""
    global _recording, _tracing
    try:
        _sampled(seconds, path)
    finally:
        try:
            if tracing is not None:
                _tracing = None
                tracing.stop()
        finally:
            _recording = False


def _sampled(seconds: float, path: str, interval: float = SAMPLE_INTERVAL) -> None:
    """Samples every other thread's stack each `interval` for `seconds`, then writes them to `path`.

    Each line of the file is a stack, root first, then its samples, as
    py-spy's collapsed format has them, with the thread's name as the root
    frame and each frame `function (file:line)`. A sample reads every
    thread's frames through `sys._current_frames()` and counts the stack in a
    dict, so it needs no ptrace and holds the GIL for little. A thread
    blocked in a call that released the GIL samples as that call.
    """
    me = threading.get_ident()
    # A stack is its thread and each frame's code id and line, innermost
    # first. Ids hash in a third of the time code objects take, so the code
    # objects are kept apart, once per stack seen.
    counts: dict[tuple[int, tuple[int, ...]], int] = {}
    codes: dict[int, Any] = {}
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for ident, frame in sys._current_frames().items():
            if ident == me:
                continue
            stack: list[int] = []
            trail = []
            while frame is not None:
                code = frame.f_code
                stack.append(id(code))
                stack.append(frame.f_lineno)
                trail.append(code)
                frame = frame.f_back
            key = (ident, tuple(stack))
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1
                codes.update(zip(stack[::2], trail, strict=True))
        time.sleep(interval)
    names.update({thread.ident: thread.name for thread in threading.enumerate()})
    lines = []
    for (ident, seen), count in sorted(counts.items(), key=lambda item: -item[1]):
        frames = [names.get(ident, f"thread-{ident}")]
        for at in range(len(seen) - 2, -1, -2):
            code, line = codes[seen[at]], seen[at + 1]
            frames.append(f"{code.co_name} ({_short(code.co_filename)}:{line})")
        lines.append(f"{';'.join(frames)} {count}\n")
    # Whole once it exists, for a reader polling for it.
    with open(f"{path}.part", "w") as file:
        file.writelines(lines)
    os.replace(f"{path}.part", path)


def _short(filename: str) -> str:
    """A source file's path from the longest `sys.path` entry that holds it."""
    for root in sorted(sys.path, key=len, reverse=True):
        if root and filename.startswith(root.rstrip(os.sep) + os.sep):
            return filename[len(root.rstrip(os.sep)) + 1 :]
    return filename


if __name__ == "__main__":
    sys.exit(main())
