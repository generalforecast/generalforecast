# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Snapshots are listed, read and deleted by name like every other collection."""

import pytest

from tests.product_sdk.test_client import run

SNAPSHOT = {
    "name": "credit-2026-01",
    "state": "ready",
    "digest": "sha256:" + "5" * 64,
    "bytes": 10,
    "rows": 2,
    "columns": [{"name": "y", "type": "int64"}],
    "metadata": {"target": "y", "task": "classification"},
    "error": None,
    "expires": None,
    "created": 1.0,
}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_snapshots_are_listed_read_and_deleted(wire, asynchronous):
    snapshot = SNAPSHOT["name"]
    wire.expect("GET", "/snapshots", {"snapshots": [SNAPSHOT]})
    wire.expect("GET", f"/snapshots/{snapshot}", SNAPSHOT)
    deleted = {**SNAPSHOT, "state": "deleted"}
    wire.expect("POST", f"/snapshots/{snapshot}/delete", deleted, payload={})

    async def body(client):
        found = []
        for call in (
            lambda: client.snapshots.list(),
            lambda: client.snapshots.get(snapshot),
            lambda: client.snapshots.delete(snapshot),
        ):
            value = call()
            found.append(await value if asynchronous else value)
        return found

    listed, got, gone = run(asynchronous, wire.url, body)
    assert listed == [SNAPSHOT]
    assert got == SNAPSHOT
    assert gone == deleted
