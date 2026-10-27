# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a worker's step held in memory, read when its parts are timed.

`host_peak` is the most memory the process held since the last `reset`, as
the kernel tracks it: on Linux the peak resident set, `VmHWM`, which writing
5 to `/proc/self/clear_refs` resets; on macOS the peak physical footprint
over the process's current interval, which `proc_reset_footprint_interval`
starts again. A platform that tracks neither, or one that refused the last
reset, reports no `host_peak`, since its peak would cover more than the step.
`device_in_use` is the bytes JAX holds on the worker's device as the step
ends: JAX's own device peak cannot be reset, so it would cover the process's
life rather than the step.
"""

import ctypes
import functools
import os
import sys
from pathlib import Path

import jax

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")
# `proc_pid_rusage`'s flavor for `rusage_info_v4`, the first with the
# footprint's interval peak.
_RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    """`struct rusage_info_v4` from macOS's `<sys/resource.h>`, in its order."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *(
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
            )
        ),
    ]


@functools.cache
def _libproc() -> ctypes.CDLL | None:
    """The process calls of macOS, or None where they are missing."""
    if sys.platform != "darwin":
        return None
    try:
        system = ctypes.CDLL(None)
        system.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_RusageInfoV4),
        ]
        system.proc_reset_footprint_interval.argtypes = [ctypes.c_int]
    except (OSError, AttributeError):
        return None
    return system


# Whether the last `reset` took, so a peak read after one that failed is not
# taken for the step's.
_reset_refused = False


def reset() -> None:
    """Starts the host peak again from what the process holds now, where the platform lets it."""
    global _reset_refused
    system = _libproc()
    if system is not None:
        _reset_refused = system.proc_reset_footprint_interval(os.getpid()) != 0
        return
    try:
        _CLEAR_REFS.write_text("5")
    except OSError:
        _reset_refused = True
    else:
        _reset_refused = False


def _host_peak() -> int | None:
    """Bytes of the process's peak since the last reset, or None where it cannot be read."""
    if _reset_refused:
        return None
    system = _libproc()
    if system is not None:
        info = _RusageInfoV4()
        if system.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)):
            return None
        return int(info.ri_interval_max_phys_footprint)
    try:
        status = _STATUS.read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def reading() -> dict[str, int]:
    """The step's memory in bytes, by kind; a kind the platform does not report is left out."""
    found: dict[str, int] = {}
    if (peak := _host_peak()) is not None:
        found["host_peak"] = peak
    stats = jax.devices()[0].memory_stats() or {}
    if "bytes_in_use" in stats:
        found["device_in_use"] = int(stats["bytes_in_use"])
    return found
