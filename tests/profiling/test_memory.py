# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from generalforecast.profiling import memory


def test_a_reset_starts_the_host_peak_from_what_the_process_holds() -> None:
    held = np.ones(256 << 20, dtype=np.uint8)
    before = memory.reading()["host_peak"]
    del held

    memory.reset()

    assert memory.reading()["host_peak"] < before


def test_a_reading_leaves_out_what_the_platform_does_not_report() -> None:
    assert set(memory.reading()) <= {"host_peak", "device_in_use"}
