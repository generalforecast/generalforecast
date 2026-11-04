# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How the remote client reads a Flight reply on a thread of its own, as the async client's calls run."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import pytest

pa = pytest.importorskip("pyarrow")

from pyarrow import flight

from generalforecast.native import remote

REPLY = pa.table({"p": pa.array([0.25, 0.5, 0.75], pa.float32())})


class _Stub(flight.FlightServerBase):
    """Answers an exchange whole in two batches, or, for `cancelled`, ends it CANCELLED."""

    def do_exchange(self, context, descriptor, reader, writer) -> None:
        reader.read_all()
        if descriptor.path[0] == b"cancelled":
            raise flight.FlightCancelledError("Timeout expired")
        writer.begin(REPLY.schema)
        for batch in REPLY.to_batches(max_chunksize=2):
            writer.write_batch(batch)

    def do_action(self, context, action) -> Iterator[flight.Result]:
        # A call after the failed one, as dropping its table is, so the main
        # thread runs again while an interrupt could be pending.
        time.sleep(0.2)
        yield flight.Result(b"null")


@pytest.fixture
def client() -> Iterator[flight.FlightClient]:
    server = _Stub("grpc://127.0.0.1:0")
    connected = flight.FlightClient(f"grpc://127.0.0.1:{server.port}")
    yield connected
    connected.close()
    server.shutdown()


def _exchange(client: flight.FlightClient, command: str) -> pa.Table:
    writer, reader = client.do_exchange(flight.FlightDescriptor.for_path(command))
    batch = pa.RecordBatch.from_pydict({"x": pa.array([1.0])})
    writer.begin(batch.schema)
    writer.write_batch(batch)
    writer.done_writing()
    try:
        return remote._read_whole(reader)
    finally:
        list(client.do_action(flight.Action("after", b"")))


def _on_a_thread(client: flight.FlightClient, command: str) -> pa.Table:
    async def call() -> pa.Table:
        return await asyncio.get_running_loop().run_in_executor(
            None, _exchange, client, command
        )

    return asyncio.run(call())


def test_a_reply_read_on_a_thread_comes_back_whole(
    client: flight.FlightClient,
) -> None:
    assert _on_a_thread(client, "predict").equals(REPLY)


def test_a_reply_cancelled_on_a_thread_fails_that_call_and_never_interrupts(
    client: flight.FlightClient,
) -> None:
    try:
        with pytest.raises(flight.FlightCancelledError, match="Timeout expired"):
            _on_a_thread(client, "cancelled")
        # A pending interrupt is raised once the main thread runs bytecode again.
        for _ in range(1000):
            pass
    except KeyboardInterrupt:
        pytest.fail("a reply cancelled on a thread interrupted the main thread")
