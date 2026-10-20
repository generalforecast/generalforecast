"""Summarises a worker's device trace per step: `python -m generalforecast.profiling.trace DIRECTORY`.

A `profile_workers` action with `trace` set has each worker write JAX's
profiler trace of its window under `DIRECTORY/worker-N/plugins/profile/<run>/`
and, beside that, `steps.json` listing the steps that ended inside the window
with their boundaries and parts. The summary reads the trace's Chrome JSON
(`*.trace.json.gz`, complete events with `ts` and `dur` in microseconds from
the trace's start) and joins its device events to the steps, which the worker
marks in the trace with a `gf step` event numbered as `steps.json` numbers
them.

On a GPU the device events are the kernels and copies on the streams of a
`/device:GPU:N` process, the copies named `MemcpyH2D`, `MemcpyD2H` and
`MemcpyD2D` with their bytes in `memcpy_details`, as XLA's CUPTI tracer names
them. On a CPU there is no device process. XLA's thunks run on the PjRt
client thread as events carrying `hlo_module` and `hlo_op`, and nothing is
copied, so those thunks stand in for the kernels.

The summary writes `steps`, `kernels` and `programs` tables beside
`steps.json` and prints them. `summarize` returns them for a notebook.
"""

import argparse
import gzip
import itertools
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv
import pyarrow.parquet

from generalforecast.protocol import STEP_EVENT, STEPS_FILE

# Where JAX's profiler writes the Chrome JSON under the directory it was given.
TRACE_GLOB = "plugins/profile/*/*.trace.json.gz"

_COPY = re.compile(r"Memcpy\s*(H2D|HtoD|D2H|DtoH|D2D|DtoD|P2P|Other)", re.IGNORECASE)
_SIZE = re.compile(r"\b(?:size|bytes)[:=]\s*(\d+)")


@dataclass(frozen=True)
class Event:
    """One complete event, its times in seconds from the trace's start."""

    pid: int
    tid: int
    name: str
    start: float
    end: float
    args: dict[str, Any]

    @property
    def seconds(self) -> float:
        """How long the event lasted."""
        return self.end - self.start

    @property
    def copy(self) -> str | None:
        """`H2D`, `D2H` or `D2D` for a copy between memories, else None."""
        found = _COPY.search(self.name)
        if found is None:
            return None
        return {"HTOD": "H2D", "DTOH": "D2H", "DTOD": "D2D"}.get(
            found.group(1).upper(), found.group(1).upper()
        )

    @property
    def bytes(self) -> int:
        """The bytes a copy moved, or 0 where the trace does not say."""
        for key in ("bytes", "num_bytes", "size"):
            value = self.args.get(key)
            if isinstance(value, int | float) or (
                isinstance(value, str) and value.isdigit()
            ):
                return int(value)
        for value in self.args.values():
            if isinstance(value, str) and (found := _SIZE.search(value)):
                return int(found.group(1))
        return 0


@dataclass(frozen=True)
class Trace:
    """A Chrome trace: its complete events and the names of their processes and threads."""

    events: list[Event]
    processes: dict[int, str]
    threads: dict[tuple[int, int], str]

    @classmethod
    def read(cls, path: Path) -> "Trace":
        """Reads a trace's JSON, gzipped or not."""
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as file:
            data = json.load(file)
        return cls.parse(data["traceEvents"] if isinstance(data, dict) else data)

    @classmethod
    def parse(cls, records: Iterable[dict[str, Any]]) -> "Trace":
        """The trace a list of Chrome trace events holds."""
        events: list[Event] = []
        processes: dict[int, str] = {}
        threads: dict[tuple[int, int], str] = {}
        for record in records:
            match record.get("ph"):
                case "M" if record.get("name") == "process_name":
                    processes[record["pid"]] = record["args"]["name"]
                case "M" if record.get("name") == "thread_name":
                    threads[record["pid"], record["tid"]] = record["args"]["name"]
                case "X":
                    start = record["ts"] / 1e6
                    events.append(
                        Event(
                            record.get("pid", 0),
                            record.get("tid", 0),
                            record.get("name", ""),
                            start,
                            start + record.get("dur", 0) / 1e6,
                            record.get("args") or {},
                        )
                    )
        events.sort(key=lambda event: (event.start, event.end))
        return cls(events, processes, threads)

    def device_events(self) -> list[Event]:
        """The kernels and copies, in start order.

        They are every event of a `/device:` process, or where there is none,
        the host's events naming an `hlo_module`, which is how XLA's CPU
        thunks appear.
        """
        devices = {
            pid for pid, name in self.processes.items() if name.startswith("/device:")
        }
        if devices:
            return [event for event in self.events if event.pid in devices]
        return [event for event in self.events if "hlo_module" in event.args]

    def steps(self) -> dict[int, Event]:
        """The window of each step a worker marked, by the step's number."""
        return {
            int(event.args["step_num"]): event
            for event in self.events
            if event.name == STEP_EVENT and "step_num" in event.args
        }


@dataclass(frozen=True)
class Summary:
    """The tables `summarize` builds; see the module docstring."""

    steps: pa.Table
    kernels: pa.Table
    programs: pa.Table

    def write(self, directory: Path, csv: bool = False) -> list[Path]:
        """Writes each table under `directory` as parquet, or as CSV."""
        written = []
        for name in ("steps", "kernels", "programs"):
            table = getattr(self, name)
            if csv:
                path = directory / f"{name}.csv"
                pa.csv.write_csv(table, path)
            else:
                path = directory / f"{name}.parquet"
                pa.parquet.write_table(table, path)
            written.append(path)
        return written

    def __str__(self) -> str:
        """The tables as aligned text."""
        return "\n\n".join(
            f"{name}\n{_formatted(getattr(self, name))}"
            for name in ("steps", "kernels", "programs")
        )


def summarize(directory: Path) -> Summary:
    """Summarises the trace and `steps.json` a worker wrote under `directory`.

    Raises:
        FileNotFoundError: Without a trace or a `steps.json` there.
    """
    traces = sorted(directory.glob(TRACE_GLOB))
    if not traces:
        raise FileNotFoundError(f"no {TRACE_GLOB} under {directory}")
    with open(directory / STEPS_FILE) as file:
        steps = json.load(file)["steps"]
    trace = Trace.read(traces[-1])
    return summarize_trace(trace, steps)


def summarize_trace(trace: Trace, steps: Sequence[dict[str, Any]]) -> Summary:
    """The tables for a parsed trace and the rows of its `steps.json`."""
    device = trace.device_events()
    windows = trace.steps()
    rows = [_step_row(step, windows.get(step["step"]), device) for step in steps]
    parts = sorted({part for step in steps for part in step.get("parts", {})})
    for row, step in zip(rows, steps, strict=True):
        for part in parts:
            row[f"{part}_s"] = step.get("parts", {}).get(part, {}).get("seconds", 0.0)
    return Summary(
        steps=_table(rows, _STEP_COLUMNS + [f"{part}_s" for part in parts]),
        kernels=_kernels(device),
        programs=_programs(device),
    )


_STEP_COLUMNS = [
    "step",
    "started",
    "ended",
    "wall_s",
    "trace_s",
    "device_s",
    "idle_s",
    "lead_s",
    "tail_s",
    "kernels",
    "h2d",
    "h2d_bytes",
    "h2d_s",
    "d2h",
    "d2h_bytes",
    "d2h_s",
    "gap_s",
    "gap_at_s",
    "gap_after",
]


def _step_row(
    step: dict[str, Any], window: Event | None, device: Sequence[Event]
) -> dict[str, Any]:
    """One step's row: its wall from `steps.json` and its device time from the trace.

    Device time is the union of the device events clipped to the step's
    window, so it is at most the window. An event counts under the step it
    starts in. `gap_s` is the longest the device waited between two of the
    step's events, `gap_at_s` seconds after the step started and `gap_after`
    the event before it. `lead_s` and `tail_s` are the waits before the first
    event and after the last.
    """
    row: dict[str, Any] = {
        "step": step["step"],
        "started": step["started"],
        "ended": step["ended"],
        "wall_s": step["ended"] - step["started"],
    }
    if window is None:
        return row
    inside = [
        event
        for event in device
        if event.end > window.start and event.start < window.end
    ]
    busy = _merged(
        (max(event.start, window.start), min(event.end, window.end), event.name)
        for event in inside
    )
    gap, gap_at, gap_after = 0.0, 0.0, ""
    for (_, end, name), (start, _, _) in itertools.pairwise(busy):
        if start - end > gap:
            gap, gap_at, gap_after = start - end, end - window.start, name
    copies = {"H2D": [0, 0, 0.0], "D2H": [0, 0, 0.0]}
    kernels = 0
    for event in inside:
        if event.start < window.start:
            continue
        direction = event.copy
        if direction is None:
            kernels += 1
        elif direction in copies:
            count = copies[direction]
            count[0] += 1
            count[1] += event.bytes
            count[2] += min(event.end, window.end) - event.start
    device_s = sum(end - start for start, end, _ in busy)
    row.update(
        trace_s=window.seconds,
        device_s=device_s,
        idle_s=window.seconds - device_s,
        lead_s=busy[0][0] - window.start if busy else window.seconds,
        tail_s=window.end - busy[-1][1] if busy else 0.0,
        kernels=kernels,
        h2d=copies["H2D"][0],
        h2d_bytes=copies["H2D"][1],
        h2d_s=copies["H2D"][2],
        d2h=copies["D2H"][0],
        d2h_bytes=copies["D2H"][1],
        d2h_s=copies["D2H"][2],
        gap_s=gap,
        gap_at_s=gap_at,
        gap_after=gap_after,
    )
    return row


def _merged(
    intervals: Iterable[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Overlapping intervals joined, each named for the event that ends it."""
    merged: list[tuple[float, float, str]] = []
    for start, end, name in sorted(intervals):
        if merged and start <= merged[-1][1]:
            last = merged[-1]
            merged[-1] = (last[0], end, name) if end > last[1] else last
        else:
            merged.append((start, end, name))
    return merged


def _kernels(device: Sequence[Event]) -> pa.Table:
    """Each kernel name's count, seconds and share of the device's time."""
    totals: dict[str, list[float]] = {}
    for event in device:
        if event.copy is None:
            count = totals.setdefault(event.name, [0, 0.0])
            count[0] += 1
            count[1] += event.seconds
    all_seconds = sum(seconds for _, seconds in totals.values()) or 1.0
    rows: list[dict[str, Any]] = [
        {
            "kernel": name,
            "count": int(count),
            "seconds": seconds,
            "mean_s": seconds / count,
            "share": seconds / all_seconds,
        }
        for name, (count, seconds) in sorted(
            totals.items(), key=lambda item: -item[1][1]
        )
    ]
    return _table(rows, ["kernel", "count", "seconds", "mean_s", "share"])


def _programs(device: Sequence[Event]) -> pa.Table:
    """Each compiled program's runs, kernels, seconds and share of the device's time.

    A program is its `hlo_module`. Its runs are its distinct `run_id`s where
    the events carry one, else its kernels.
    """
    kernels: dict[str, int] = {}
    seconds: dict[str, float] = {}
    runs: dict[str, set[str]] = {}
    for event in device:
        if event.copy is not None:
            continue
        program = str(event.args.get("hlo_module", ""))
        kernels[program] = kernels.get(program, 0) + 1
        seconds[program] = seconds.get(program, 0.0) + event.seconds
        if "run_id" in event.args:
            runs.setdefault(program, set()).add(str(event.args["run_id"]))
    all_seconds = sum(seconds.values()) or 1.0
    rows: list[dict[str, Any]] = [
        {
            "program": program,
            "runs": len(runs[program]) if program in runs else kernels[program],
            "kernels": kernels[program],
            "seconds": seconds[program],
            "share": seconds[program] / all_seconds,
        }
        for program in sorted(kernels, key=lambda program: -seconds[program])
    ]
    return _table(rows, ["program", "runs", "kernels", "seconds", "share"])


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> pa.Table:
    """Rows as a table with `columns` in order, missing values null."""
    return pa.table({column: [row.get(column) for row in rows] for column in columns})


def _formatted(table: pa.Table, limit: int = 30) -> str:
    """A table as aligned text, its first `limit` rows."""
    columns = table.column_names
    cells = [columns]
    for row in table.slice(0, limit).to_pylist():
        cells.append(
            [
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in (row[column] for column in columns)
            ]
        )
    widths = [max(len(line[at]) for line in cells) for at in range(len(columns))]
    lines = [
        "  ".join(cell.rjust(width) for cell, width in zip(line, widths, strict=True))
        for line in cells
    ]
    if table.num_rows > limit:
        lines.append(f"... {table.num_rows - limit} more rows")
    return "\n".join(lines)


def main(arguments: Sequence[str] | None = None) -> int:
    """Summarises one worker's trace directory; returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("directory", type=Path, help="a worker's trace directory")
    parser.add_argument("--csv", action="store_true", help="write CSV, not parquet")
    options = parser.parse_args(arguments)
    try:
        summary = summarize(options.directory)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    for path in summary.write(options.directory, csv=options.csv):
        print(f"wrote {path}")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
