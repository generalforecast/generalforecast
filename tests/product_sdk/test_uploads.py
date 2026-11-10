# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Training uploads are listed, read and deleted like every other collection."""

import pytest

from tests.product_sdk.test_client import run

UPLOAD = {
    "id": "up-3f1c2a9b8e7d6c5b4a392817",
    "state": "verified",
    "bytes": 10,
    "digest": "sha256:" + "5" * 64,
    "error": None,
    "metadata": {"target": "y", "task": "classification"},
    "expires": None,
    "created": 1.0,
}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_uploads_are_listed_read_and_deleted(wire, asynchronous):
    upload = UPLOAD["id"]
    wire.expect("GET", "/uploads", {"uploads": [UPLOAD]})
    wire.expect("GET", f"/uploads/{upload}", {**UPLOAD, "snapshot": {"rows": 2}})
    deleted = {
        "id": upload,
        "state": "deleted",
        "digest": UPLOAD["digest"],
        "uploads": [upload],
    }
    wire.expect("POST", f"/uploads/{upload}/delete", deleted, payload={})

    async def body(client):
        found = []
        for call in (
            lambda: client.uploads.list(),
            lambda: client.uploads.get(upload),
            lambda: client.uploads.delete(upload),
        ):
            value = call()
            found.append(await value if asynchronous else value)
        return found

    listed, got, gone = run(asynchronous, wire.url, body)
    assert listed == [UPLOAD]
    assert got["snapshot"] == {"rows": 2}
    assert gone == deleted
