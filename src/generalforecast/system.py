# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the operating system lets this process use, the same on Linux and macOS.

Calls only some platforms have, like Linux's CPU affinity, go through here,
so the code that needs them runs everywhere.
"""

import os
from collections.abc import Iterable

from generalforecast.constants import MAPPINGS_SHORT


def usable_cpus() -> int:
    """The cores this process may run on: its affinity on Linux, every core elsewhere."""
    affinity = getattr(os, "sched_getaffinity", None)
    count = len(affinity(0)) if affinity else os.cpu_count()
    return max(1, count or 1)


def mappings_short() -> bool:
    """Whether this process holds most of the memory mappings Linux allows it.

    Each compiled CPU program maps its code apart, so past `vm.max_map_count`
    a compile fails though memory is free. Elsewhere there is no such cap.
    """
    try:
        with open("/proc/sys/vm/max_map_count") as file:
            limit = int(file.read())
        with open("/proc/self/maps", "rb") as file:
            used = sum(1 for _ in file)
    except (OSError, ValueError):
        return False
    return used > limit * MAPPINGS_SHORT


def pin_cpus(cpus: Iterable[int]) -> None:
    """Keeps this process on `cpus` where the platform allows it, and does nothing elsewhere."""
    pin = getattr(os, "sched_setaffinity", None)
    if pin is not None:
        pin(0, list(cpus))
