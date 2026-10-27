# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`Client` and `AsyncClient` expose the same operations and send the same requests."""

import asyncio
import copy
import hashlib
import inspect
import re
import time
from typing import Any

import pytest

from generalforecast import _operations
from generalforecast.async_client import AsyncClient
from generalforecast.client import Client
from generalforecast.contracts import ContextImportRef, Environment, Revision
from generalforecast.service.sdk import ContextSpec
from generalforecast.tabular import table_digest
from generalforecast.transport import Profile
from tests.product_sdk.test_lifecycle import checked, checked_lock

URL = "https://service.example"
RESOURCES = (
    "contexts",
    "sources",
    "models",
    "predictors",
    "identity",
    "keys",
    "secrets",
)


def public(value: Any) -> dict[str, Any]:
    return {
        name: getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and callable(getattr(value, name))
    }


def clients() -> tuple[Client, AsyncClient]:
    profile = Profile(URL, "test", environment="staging", runtime="local")
    return Client(profile=profile, api_key="fixture-key"), AsyncClient(
        profile=profile, api_key="fixture-key"
    )


def parameters(method: Any) -> list[inspect.Parameter]:
    """A method's parameters; the return differs only by sync or async handle type."""
    return list(inspect.signature(method, eval_str=True).parameters.values())


SYNC_ONLY = {"close", "__enter__", "__exit__"}
ASYNC_ONLY = {"aclose", "__aenter__", "__aexit__"}


def test_every_public_operation_exists_in_both_clients_with_one_signature():
    sync, asynchronous = clients()
    pairs = [(sync, asynchronous)] + [
        (getattr(sync, name), getattr(asynchronous, name)) for name in RESOURCES
    ]
    for left, right in pairs:
        names = set(public(left)) - SYNC_ONLY
        assert names == set(public(right)) - ASYNC_ONLY, type(left).__name__
        for name in names:
            method, awaited = getattr(left, name), getattr(right, name)
            assert not inspect.iscoroutinefunction(method), name
            assert inspect.iscoroutinefunction(awaited), name
            assert parameters(method) == parameters(awaited), name
    assert not hasattr(sync.keys, "get") and not hasattr(asynchronous.keys, "get")


def handle_methods(cls: type) -> set[str]:
    return {
        name
        for name, value in vars(cls).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }


def test_async_handles_share_the_sync_handles_facts_and_operations():
    from generalforecast.async_client import AsyncHostedPredictor, AsyncLiveContext
    from generalforecast.service.sdk import (
        ContextHandle,
        HostedPredictor,
        LiveContext,
        PredictorHandle,
    )

    for sync, awaited, base in (
        (HostedPredictor, AsyncHostedPredictor, PredictorHandle),
        (LiveContext, AsyncLiveContext, ContextHandle),
    ):
        assert issubclass(sync, base) and issubclass(awaited, base)
        assert handle_methods(sync) == handle_methods(awaited)
        for name in handle_methods(sync):
            assert inspect.signature(getattr(sync, name)).parameters.keys() == (
                inspect.signature(getattr(awaited, name)).parameters.keys()
            )
            assert inspect.iscoroutinefunction(getattr(awaited, name))
    assert {"rows", "sequence"} <= set(dir(AsyncLiveContext))


def results(lock: Any, revision: Revision) -> list[tuple[str, str, Any]]:
    environment = Environment("risk", "staging", 1, 1, "local", "ready").to_dict()
    prediction = {
        "model": "general-forecast/tabicl-v2-classifier",
        "task": "classification",
        "predictions": [{"label": True, "probabilities": {"false": 0.1, "true": 0.9}}],
    }
    imported = {"context": "history", "digest": "sha256:" + "a" * 64, "rows": 1}
    return [
        ("GET", r"/(contexts|sources|models|predictors|keys|secrets)", "listing"),
        ("POST", r"/snapshots", None),
        ("POST", r"/predict", prediction),
        ("POST", r"/predictors/risk/predict\?env=\w+", prediction),
        ("GET", r"/predictors/risk/revisions", {"revisions": [revision.to_dict()]}),
        (
            "GET",
            r"/predictors/risk/revisions/1",
            {"revision": revision.to_dict(), "lock": lock.to_dict()},
        ),
        ("POST", r"/predictors/risk/revisions", revision.to_dict()),
        ("GET", r"/predictors/risk/environments", {"environments": [environment]}),
        ("GET", r"/predictors/risk/environments/\w+", environment),
        (
            "POST",
            r"/predictors/risk/environments(/\w+/(promote|rollback))?",
            environment,
        ),
        ("GET", r"/predictors/risk/environments/\w+/(metrics|logs)", {"requests": 0}),
        ("POST", r"/predictors/risk/delete", {"name": "risk", "deleted": True}),
        ("GET", r"/predictors/risk", {"name": "risk", "latest_revision": 1}),
        ("POST", r"/predictors", {"name": "risk"}),
        ("POST", r"/contexts/history/(append|label)", {"sequence": 1, "write_id": "w"}),
        ("POST", r"/contexts/history/(imports|refresh)", imported | {"sequence": 2}),
        ("POST", r"/contexts/history/(mutations|snapshot)", {"sequence": 3}),
        ("GET|POST", r"/contexts(/history)?", {"name": "history", "sequence": 0}),
        ("GET|POST", r".*", {"ok": True}),
    ]


def calls(authored: Any) -> dict[str, tuple[tuple, dict]]:
    """Representative arguments for every public operation, by `resource.method`."""
    rows = [{"id": "row1", "x": 2, "y": 1}]
    return {
        "predict": (
            (
                "general-forecast/tabicl-v2-classifier",
                "y",
                {"x": [1, 2], "y": [0, 1]},
                {"x": [3]},
            ),
            {"inputs": ["x"]},
        ),
        "upload_snapshot": (({"x": [1, 2]},), {}),
        "contexts.list": ((), {}),
        "contexts.get": (("history",), {}),
        "contexts.create": ((ContextSpec("history", ("x",), {"y": "number"}),), {}),
        "contexts.append": (("history", rows), {"id": "id", "idempotency_key": "w"}),
        "contexts.label": (
            ("history", rows),
            {"id": "id", "column": "y", "idempotency_key": "w"},
        ),
        "contexts.import_source": (
            (ContextImportRef("history", "s3://bucket/events", "parquet"),),
            {},
        ),
        "contexts.import_file": (("history", {"x": [1]}), {}),
        "contexts.refresh": (("history",), {}),
        "contexts.mutate": (
            ("history", {"write_id": "w", "operation": "insert", "records": []}),
            {},
        ),
        "contexts.snapshot": (("history",), {"target": "y", "knowledge_at": 1.0}),
        "sources.list": ((), {}),
        "sources.get": (("source-1",), {}),
        "sources.pin": (({"path": "a.csv", "format": "csv", "mapping": {}},), {}),
        "models.list": ((), {}),
        "models.get": (("tabicl-v2-classifier",), {}),
        "predictors.list": ((), {}),
        "predictors.get": (("risk",), {"env": "staging"}),
        "predictors.create": (("risk",), {}),
        "predictors.revisions": (("risk",), {}),
        "predictors.revision": (("risk", 1), {}),
        "predictors.environments": (("risk",), {}),
        "predictors.environment": (("risk",), {}),
        "predictors.create_environment": (
            ("risk",),
            {"env": "dev", "runtime": "local"},
        ),
        "predictors.promote": (("risk",), {"revision": 1}),
        "predictors.rollback": (("risk",), {"revision": 1, "expected_generation": 1}),
        "predictors.predict": (("risk", {"x": [3]}), {}),
        "predictors.push": ((authored,), {"wait": True}),
        "predictors.wait": (("risk",), {"revision": 1}),
        "predictors.metrics": (("risk",), {}),
        "predictors.logs": (("risk",), {"env": "production"}),
        "predictors.delete": (("risk",), {}),
        "identity.manage": (("audit.list",), {}),
        "keys.list": ((), {}),
        "keys.create": ((), {"name": "ci", "permissions": ["models:read"]}),
        "keys.revoke": (("key-1",), {}),
        "keys.rotate": (("key-1",), {}),
        "secrets.list": ((), {}),
        "secrets.get": (("token",), {}),
        "secrets.create": ((), {"name": "token", "value": "v"}),
        "secrets.rotate": ((), {"name": "token", "value": "v", "expected_version": 1}),
        "secrets.delete": (("token",), {}),
    }


class Recorder:
    """Answers every request from `results` and records it without its request id."""

    def __init__(self, table: list[tuple[str, str, Any]]) -> None:
        self.table, self.sent = table, []

    def reply(self, method, path, body, headers, deadline=None):
        route = path.removeprefix("/v1/projects/test").removeprefix("/v1")
        if isinstance(body, bytes):
            recorded: Any = hashlib.sha256(body).hexdigest()
        else:
            recorded = copy.deepcopy(body)
            if isinstance(recorded, dict):
                recorded.pop("request_id", None)
        kept = {k: v for k, v in headers.items() if k != "X-GF-Request-ID"}
        self.sent.append((method, route, recorded, kept, deadline is not None))
        for verbs, pattern, result in self.table:
            target = route if "?" in pattern else route.split("?")[0]
            if method in verbs.split("|") and re.fullmatch(pattern, target):
                if result is None:
                    import pyarrow as pa

                    table = pa.ipc.open_stream(body).read_all()
                    return {"digest": table_digest(table), "rows": table.num_rows}
                if result == "listing":
                    return {route.strip("/"): []}
                return result
        raise AssertionError(route)


def record(client: Any, recorder: Recorder, call: str, args: tuple, kwargs: dict):
    if isinstance(client, AsyncClient):

        async def send(*a, **k):
            return recorder.reply(*a, **k)

    else:

        def send(*a, **k):
            return recorder.reply(*a, **k)

    client._transport.send = send
    owner, _, method = call.rpartition(".")
    target = getattr(client, owner) if owner else client
    value = getattr(target, method)(*args, **kwargs)
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def test_identical_calls_send_identical_requests(tmp_path, monkeypatch):
    authored = checked(tmp_path)
    lock = checked_lock(authored)
    revision = Revision("risk", 1, lock.digest, lock.model, time.time())
    monkeypatch.setattr(_operations, "answer", lambda _prepared, value: value)
    sync, asynchronous = clients()
    table = calls(authored)
    surface = {
        f"{owner}.{name}" if owner else name
        for owner, value in [("", sync)] + [(r, getattr(sync, r)) for r in RESOURCES]
        for name in public(value)
        if name not in SYNC_ONLY
    }
    assert surface == set(table), "add the new operation to the parity table"
    for call, (args, kwargs) in table.items():
        sent = []
        for client in (sync, asynchronous):
            recorder = Recorder(results(lock, revision))
            record(client, recorder, call, args, kwargs)
            sent.append(recorder.sent)
        assert sent[0] == sent[1], call
        assert sent[0], call


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_closed_client_raises_the_same_typed_error(asynchronous):
    import generalforecast as gf
    from generalforecast.service.contracts import ServiceError

    sync, awaited = clients()

    async def run():
        if asynchronous:
            await awaited.aclose()
            return await awaited.models.list()
        sync.close()
        return sync.models.list()

    with pytest.raises(ServiceError) as error:
        asyncio.run(run())
    assert error.value.code == "ClientClosed"
    assert isinstance(error.value, gf.InvalidInput)
