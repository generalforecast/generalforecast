"""Requests per second of `_core.Engine` at N concurrent callers, on distinct contexts.

The workload the engine port is gated on: tabicl-classifier, 256 labeled
context rows of 8 features and 32 queries per request, every request a
context no other request repeats. Each argument is a caller count run as one
pass; repeat a count to see it with the shapes its first pass compiled:

    THREADS=4 python -m tests.benchmarks.engine_throughput 1 8 64 8 64

`THREADS` is `cpu_threads`, the cores each CPU worker has to itself. With
`TIMING=path` the shards write gf-server's per-request timing log there, and
the pass ends by summarizing it: requests per step and worker seconds.
"""

import collections
import json
import os
import sys
import tempfile
import threading
import time

import numpy as np

from generalforecast import _core
from generalforecast.engine import configuration


def payload(seed: int, rows: int = 256, features: int = 8, queries: int = 32) -> bytes:
    """A classification request whose context no other seed repeats."""
    generator = np.random.default_rng(seed)
    context = generator.normal(size=(rows, features))
    labels = context[:, 0] + context[:, 1] > 0
    names = [f"f{index}" for index in range(features)]
    request = {
        "model": "tabicl-classifier",
        "task": "classification",
        "target": "y",
        "features": names,
        "context": [
            {
                "key": str(index),
                "features": {
                    name: float(value) for name, value in zip(names, row, strict=True)
                },
                "label": bool(label),
            }
            for index, (row, label) in enumerate(
                zip(context.tolist(), labels.tolist(), strict=True)
            )
        ],
        "queries": [
            {
                "key": f"q{index}",
                "features": {
                    name: float(value) for name, value in zip(names, row, strict=True)
                },
            }
            for index, row in enumerate(
                generator.normal(size=(queries, features)).tolist()
            )
        ],
        "options": {},
    }
    return json.dumps(request).encode()


def summarize(path: str) -> None:
    """Requests per step and worker seconds per request, from a timing log."""
    with open(path) as log:
        rows = [json.loads(line) for line in log]
    steps = collections.Counter(row["finished_us"] for row in rows)
    sizes = collections.Counter(steps.values())
    worker = sum(row["gpu_seconds"] for row in rows)
    print(
        f"timing: requests={len(rows)} steps={len(steps)} "
        f"requests_per_step={dict(sorted(sizes.items()))} "
        f"worker_seconds_per_request={worker / max(len(rows), 1):.3f}",
        flush=True,
    )


def main() -> None:
    data = tempfile.mkdtemp(prefix="gf-bench-")
    config = configuration(data, int(os.environ.get("THREADS", "4")))
    if timing := os.environ.get("TIMING"):
        config = f"timing_log = {json.dumps(timing)}\n" + config
    started = time.perf_counter()
    engine = _core.Engine(config)
    engine.add_model("tabicl-classifier", "tabicl-classifier")
    print(f"shards={engine.shards} started in {time.perf_counter() - started:.1f}s")

    def parse(seed: int) -> _core.TabularRequest:
        request = _core.parse_inference(payload(seed), "local")
        assert isinstance(request, _core.TabularRequest)
        return request

    engine.tabular("tabicl-classifier", parse(10_000_000))
    for index, callers in enumerate(map(int, sys.argv[1:] or ["1", "8", "64"])):
        per = max(4, 128 // callers)
        work = [
            [parse(index * 1_000_000 + caller * 1000 + k) for k in range(per)]
            for caller in range(callers)
        ]

        def run(caller: int, work: list = work) -> None:
            for request in work[caller]:
                engine.tabular("tabicl-classifier", request)

        threads = [threading.Thread(target=run, args=(c,)) for c in range(callers)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        seconds = time.perf_counter() - started
        print(
            f"callers={callers} requests={callers * per} seconds={seconds:.2f} "
            f"req/s={callers * per / seconds:.2f}",
            flush=True,
        )
    engine.close()
    if timing:
        summarize(timing)


if __name__ == "__main__":
    main()
