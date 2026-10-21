# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Python SDK upload/lifecycle contracts; fixture results are not isolated inference."""

import asyncio
import dataclasses
import json
import threading
import time

import httpx
import pytest
import yaml

import generalforecast as gf
import generalforecast.async_client as async_sdk
import generalforecast.deployments.sdk as shared
from tests.product_sdk.test_lifecycle import invoke


def python_package(root, bias=1):
    root.mkdir()
    (root / "gf.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "risk",
                "task": "regression",
                "context": {"source": "none"},
                "model": {
                    "kind": "python",
                    "code": ["predictor.py"],
                    "dependencies": [],
                    "artifacts": ["probe.json", "bias.json"],
                    "inputs": [{"name": "x", "kind": "float64", "nullable": False}],
                    "outputs": [
                        {"name": "value", "kind": "float64", "nullable": False}
                    ],
                },
            }
        )
    )
    (root / "predictor.py").write_text(
        f"raise RuntimeError('must not import predictor {bias}')\n"
    )
    (root / "bias.json").write_text(json.dumps({"bias": bias}))
    (root / "probe.json").write_text(
        '{"schema":[{"name":"x","kind":"float64","nullable":false}],"rows":[[1.0]]}'
    )
    return gf.Package.load(root).check()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_public_scoped_source_upload_returns_immutable_receipt(
    wire, tmp_path, asynchronous
):
    package = python_package(tmp_path / "package")
    lock = package.compile()
    assert isinstance(lock, gf.PythonLock)
    archive = package.source_archive()
    wire.expect("POST", "/deployments/risk/python-sources", payload=archive)

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            receipt = await invoke(
                client.upload_python_source, "risk", archive, source=lock.python.source
            )
            assert isinstance(receipt, gf.Snapshot)
            assert receipt.to_dict() == shared.python_source_upload(
                "test", "risk", archive, lock.python.source
            )
            changed = receipt.to_dict()
            changed["source"] = "changed"
            assert receipt["source"] == lock.python.source
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        {"project": "foreign"},
        {"deployment": "other"},
        {"source": "sha256:" + "0" * 64},
        {"receipt_version": True},
        {"receipt_version": 2},
        {"size": True},
        {"size": 1.0},
        {"kind": "image"},
        {"image": "untrusted"},
        {"size": 0},
    ],
)
def test_source_receipt_requires_exact_scoped_content(change):
    expected = shared.python_source_upload(
        "test", "risk", b"source", shared.bytes_digest(b"source")
    )
    with pytest.raises(gf.ServiceError) as error:
        shared.python_source_receipt(expected | change, expected)
    assert error.value.code == "ProtocolError"
    for field in expected:
        with pytest.raises(gf.ServiceError):
            shared.python_source_receipt(
                {key: value for key, value in expected.items() if key != field},
                expected,
            )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_source_upload_rejects_invalid_bytes_digest_and_route_before_network(
    wire, monkeypatch, asynchronous
):
    monkeypatch.setattr(shared, "MAX_ARCHIVE", 5)

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            for deployment, archive, source in [
                ("risk", b"", shared.bytes_digest(b"")),
                ("risk", b"123456", shared.bytes_digest(b"123456")),
                ("risk", bytearray(b"123"), shared.bytes_digest(b"123")),
                ("risk", b"123", "sha256:" + "0" * 64),
                ("../risk", b"123", shared.bytes_digest(b"123")),
            ]:
                with pytest.raises(gf.ServiceError):
                    await invoke(
                        client.upload_python_source, deployment, archive, source=source
                    )
            assert not wire.requests
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_python_push_revision_predict_and_dev_patch_reuse_canonical_lifecycle(
    wire, tmp_path, monkeypatch, asynchronous
):
    first = python_package(tmp_path / "first")
    second = python_package(tmp_path / "second", 2)
    lock, replacement = first.compile(), second.compile()
    assert isinstance(lock, gf.PythonLock) and isinstance(replacement, gf.PythonLock)
    revision = gf.Revision("risk", 1, lock.package_digest, lock.identity, time.time())
    initial = gf.Environment("risk", "production", None, 0, "local")
    promoted = initial.promote(gf.PromoteRequest(1, 0))
    ready = dataclasses.replace(promoted, status="ready")
    base = "/deployments/risk"
    wire.expect("GET", base, "NotFound", status=404)
    wire.expect("POST", "/deployments", {"name": "risk"}, payload={"name": "risk"})
    wire.expect("POST", base + "/python-sources", payload=first.source_archive())
    wire.expect(
        "POST",
        base + "/revisions",
        revision.to_dict(),
        payload={"lock": lock.to_dict(), "snapshot": None},
    )
    wire.expect("GET", base + "/environments/production", "NotFound", status=404)
    wire.expect(
        "POST",
        base + "/environments",
        initial.to_dict(),
        payload={"name": "production", "runtime": "local"},
    )
    wire.expect(
        "POST",
        base + "/environments/production/promote",
        promoted.to_dict(),
        payload={"revision": 1, "expected_generation": 0},
    )
    wire.expect("GET", base + "/environments/production", ready.to_dict())
    wire.expect(
        "GET",
        base + "/revisions/1",
        {"revision": revision.to_dict(), "lock": lock.to_dict()},
    )
    prediction = {
        "model": lock.identity,
        "task": "regression",
        "context": None,
        "deployment": "risk",
        "revision": 1,
        "environment": "production",
        "predictions": [{"value": 3.0}],
    }
    wire.expect(
        "POST",
        base + "/environments/production/predict",
        prediction,
        payload={"queries": [{"x": 1.0}]},
    )
    dev = gf.DevSession(
        "dev-test", "risk", 1, 0, lock.package_digest, time.time() + 600, "ready"
    )
    wire.expect(
        "POST",
        base + "/dev-sessions",
        dev.to_dict(),
        payload={"base_revision": 1, "runtime": "local", "ttl_seconds": 3600},
    )
    wire.expect("POST", base + "/python-sources", payload=second.source_archive())
    patch = {
        "expected_generation": 0,
        "base_revision": 1,
        "lock": replacement.to_dict(),
        "snapshot": None,
    }
    patched = dataclasses.replace(
        dev, generation=1, package_digest=replacement.package_digest
    )
    wire.expect(
        "POST", base + "/dev-sessions/dev-test/patch", patched.to_dict(), payload=patch
    )
    observed = []
    if asynchronous:
        for name in ("checked_python_source", "python_source_upload"):
            function = getattr(async_sdk, name)

            def record(*args, _function=function, **kwargs):
                observed.append(threading.get_ident())
                return _function(*args, **kwargs)

            monkeypatch.setattr(async_sdk, name, record)

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            assert (
                await invoke(client.deployments.push, first, runtime="local", wait=True)
                == revision
            )
            details = await invoke(client.deployments.revision, "risk", 1)
            assert (
                isinstance(details.lock, gf.PythonLock)
                and details.lock.identity == lock.identity
            )
            assert (
                await invoke(client.deployments.predict, "risk", {"x": [1.0]})
            ).model == lock.identity
            development = await invoke(
                client.dev_sessions.start, "risk", base_revision=1, runtime="local"
            )
            updated = await invoke(development.patch, second)
            assert updated.generation == 1 and development.generation == 0
            assert updated.state.package_digest == replacement.package_digest
            if asynchronous:
                assert len(observed) == 4 and threading.get_ident() not in observed
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_source_tamper_and_forged_receipt_never_register_revision(
    wire, tmp_path, asynchronous
):
    package = python_package(tmp_path / "package")
    lock = package.compile()
    assert isinstance(lock, gf.PythonLock)
    archive = package.source_archive()
    wire.expect("GET", "/deployments/risk", {"name": "risk"})
    receipt = shared.python_source_upload("other", "risk", archive, lock.python.source)
    wire.expect("POST", "/deployments/risk/python-sources", receipt)

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            with pytest.raises(gf.ServiceError) as error:
                await invoke(client.deployments.push, package, runtime="local")
            assert error.value.code == "ProtocolError"
            count = len(wire.requests)
            (package.root / "predictor.py").write_text(
                "raise RuntimeError('changed')\n"
            )
            with pytest.raises(gf.PackageError):
                await invoke(client.deployments.push, package, runtime="local")
            assert len(wire.requests) == count
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_async_source_upload_cancellation_and_timeout_close_response_without_retry(
    cancel,
):
    async def run():
        entered = asyncio.Event()
        closed = []
        requests = []

        class Pending(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b""

            async def aclose(self):
                closed.append(True)

        async def handle(request):
            requests.append(request)
            assert request.content == b"source"
            assert request.headers["content-type"] == shared.PYTHON_SOURCE_MEDIA_TYPE
            return httpx.Response(200, stream=Pending())

        client = gf.AsyncClient(
            "http://127.0.0.1:8000",
            project="test",
            api_key="fixture-key",
            http_transport=httpx.MockTransport(handle),
            retries=3,
            timeout=1,
        )
        try:
            task = asyncio.create_task(
                client.upload_python_source(
                    "risk", b"source", source=shared.bytes_digest(b"source")
                )
            )
            await entered.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(gf.ServiceError) as error:
                    await task
                assert error.value.code == "Timeout"
            assert closed == [True] and len(requests) == 1
        finally:
            await client.aclose()

    asyncio.run(run())
