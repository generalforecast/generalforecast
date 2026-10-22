# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a worker's step held in memory, read when its parts are timed.

`host_peak` is the process's peak resident memory since the last `reset`,
Linux's `VmHWM`, which writing 5 to `/proc/self/clear_refs` resets.
`device_in_use` is the bytes JAX holds on the worker's device as the step
ends: JAX's own device peak cannot be reset, so it would cover the process's
life rather than the step.
"""

from pathlib import Path

import jax

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


def reset() -> None:
    """Starts the host peak again from what the process holds now, where Linux lets it."""
    try:
        _CLEAR_REFS.write_text("5")
    except OSError:
        pass


def reading() -> dict[str, int]:
    """The step's memory in bytes, by kind; a kind the platform does not report is left out."""
    found: dict[str, int] = {}
    try:
        status = _STATUS.read_text()
    except OSError:
        status = ""
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            found["host_peak"] = int(line.split()[1]) * 1024
    stats = jax.devices()[0].memory_stats() or {}
    if "bytes_in_use" in stats:
        found["device_in_use"] = int(stats["bytes_in_use"])
    return found
