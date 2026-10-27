# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The worker: runs the steps a node schedules, on one accelerator.

`Worker` is one object made of its parts, each a module here: `runner`
executes steps, `calls` sizes and runs the compiled calls, `admission`
measures models and refuses what would not fit, `ahead` compiles before
steps need it and defers tables waiting on a program, `contexts` keeps
pages and contexts in the tiers' slots and decodes from them, and `series`
forecasts series units.
"""

from generalforecast.worker.ahead import _manifests
from generalforecast.worker.calls import _stacked
from generalforecast.worker.runner import Worker, check_summary
from generalforecast.worker.stats import WorkerStats

__all__ = ["Worker", "WorkerStats", "_manifests", "_stacked", "check_summary"]
