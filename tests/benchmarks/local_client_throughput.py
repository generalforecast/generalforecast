# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Predictions per second of `LocalClient` at N concurrent callers on one table.

tabicl-classifier on a table of 256 labeled rows of 8 features; every call
predicts 32 queries no other call repeats. Each argument is a caller count run
as one pass, after a warm-up prediction:

    python -m tests.benchmarks.local_client_throughput 1 8 64
"""

import sys
import threading
import time

import numpy as np

import generalforecast


def main() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(256, 8)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] > 0).astype(np.float32)
    started = time.perf_counter()
    client = generalforecast.LocalClient()
    table = client.create_table(
        "bench", model="tabicl-classifier", features=8, classes=2
    )
    table.insert(np.arange(256), features, labels)
    table.predict(generator.normal(size=(32, 8)).astype(np.float32))
    print(f"ready in {time.perf_counter() - started:.1f}s", flush=True)
    for callers in map(int, sys.argv[1:] or ["1", "8", "64"]):
        per = max(4, 128 // callers)
        work = [
            [generator.normal(size=(32, 8)).astype(np.float32) for _ in range(per)]
            for _ in range(callers)
        ]

        def run(caller: int, work: list = work) -> None:
            for queries in work[caller]:
                table.predict(queries)

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
    client.close()


if __name__ == "__main__":
    main()
