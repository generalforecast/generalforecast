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
            (
                "history",
                {
                    "write_id": "w",
                    "operation": "create_features",
                    "records": [{"key": "row1", "features": {"x": 2.0}}],
                },
            ),
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
        "keys.revoke": (("key-" + "0" * 24,), {}),
        "keys.rotate": (("key-" + "0" * 24,), {}),
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


def spec_routes() -> list[tuple[str, re.Pattern[str], Any]]:
    from generalforecast import _api

    return [
        (
            route.method,
            re.compile(re.sub(r"\{[^}]+\}", "[^/]+", route.path)),
            route.request,
        )
        for route in _api.ROUTES
    ]


def identity_calls() -> list[tuple[str, dict]]:
    """Every identity action, with the identities its path takes."""
    from generalforecast._operations import WORKSPACE_ACTIONS

    ids = {"workspace": "acme", "id": "item-1"}
    calls = [
        (action, {k: v for k, v in ids.items() if "{" + k + "}" in path})
        for action, (_method, path, _resource) in WORKSPACE_ACTIONS.items()
    ]
    return [
        *calls,
        ("workspace.get", {}),
        ("auth.status", {}),
        ("auth.logout", {}),
        ("organizations.list", {}),
        ("organizations.create", {"name": "acme", "account": "a"}),
        ("projects.list", {}),
        ("projects.create", {"name": "web", "organization": "acme"}),
        ("memberships.list", {}),
        ("memberships.add", {"issuer": "i", "subject": "s", "role": "admin"}),
        ("memberships.remove", {"issuer": "i", "subject": "s"}),
        ("sso.get", {}),
        ("sso.configure", {"enabled": True}),
        ("audit.list", {"after": 0, "limit": 10}),
    ]


# A revision's lock uses gf.yaml's type names (`bool`, `int`), which the spec's
# Lock schema does not list; the lock and the spec are reconciled elsewhere.
_LOCK_ROUTE = re.compile(r"/v1/projects/[^/]+/predictors/[^/]+/revisions")


def sent_requests(tmp_path, monkeypatch) -> list[tuple[str, str, Any, bool]]:
    """Every request a client sends for the parity table and every identity action.

    Each is marked with whether the SDK built its payload; an identity action
    forwards the caller's.
    """
    authored = checked(tmp_path)
    lock = checked_lock(authored)
    revision = Revision("risk", 1, lock.digest, lock.model, time.time())
    monkeypatch.setattr(_operations, "answer", lambda _prepared, value: value)
    sync, _asynchronous = clients()
    sent: list[tuple[str, str, Any, bool]] = []
    built = True

    class Spec(Recorder):
        def reply(self, method, path, body, headers, deadline=None):
            sent.append((method, path.split("?")[0], body, built))
            return super().reply(method, path, body, headers, deadline)

    for call, (args, kwargs) in calls(authored).items():
        record(sync, Spec(results(lock, revision)), call, args, kwargs)
    built = False
    for action, payload in identity_calls():
        record(
            sync,
            Spec(results(lock, revision)),
            "identity.manage",
            (action, payload),
            {},
        )
    return sent


def spec_request(routes, method: str, path: str) -> Any:
    matching = [
        request
        for verb, pattern, request in routes
        if verb == method and pattern.fullmatch(path)
    ]
    assert matching, (method, path)
    return matching[0]


def test_every_request_either_client_sends_is_a_route_of_the_spec(
    tmp_path, monkeypatch
):
    routes = spec_routes()
    for method, path, body, built in sent_requests(tmp_path, monkeypatch):
        request = spec_request(routes, method, path)
        if (
            built
            and method == "POST"
            and isinstance(body, dict)
            and request is not None
            and not _LOCK_ROUTE.fullmatch(path)
        ):
            request.from_payload(body)


def test_a_pushed_lock_matches_the_spec(tmp_path, monkeypatch):
    routes = spec_routes()
    for method, path, body, _built in sent_requests(tmp_path, monkeypatch):
        if method == "POST" and _LOCK_ROUTE.fullmatch(path):
            spec_request(routes, method, path).from_payload(body)


class Refusing(Recorder):
    """Answers as `Recorder`, refusing the first request to each named route."""

    def __init__(self, table, refused: dict[tuple[str, str], Any]) -> None:
        super().__init__(table)
        self.refused = dict(refused)

    def reply(self, method, path, body, headers, deadline=None):
        route = path.removeprefix("/v1/projects/test").split("?")[0]
        refusal = self.refused.pop((method, route), None)
        if refusal is not None:
            super().reply(method, path, body, headers, deadline)
            raise refusal
        return super().reply(method, path, body, headers, deadline)


def test_failures_inside_operations_are_handled_alike(tmp_path, monkeypatch):
    from generalforecast.service.contracts import ServiceError

    authored = checked(tmp_path)
    lock = checked_lock(authored)
    revision = Revision("risk", 1, lock.digest, lock.model, time.time())
    monkeypatch.setattr(_operations, "answer", lambda _prepared, value: value)
    missing = ServiceError("NotFound", "gone", 404)
    cases = [
        # A new predictor and environment are created before the promotion.
        (
            "predictors.push",
            (authored,),
            {},
            {
                ("GET", "/predictors/risk"): missing,
                ("GET", "/predictors/risk/environments/staging"): missing,
            },
            None,
        ),
        # Running out of time inside a poll is a WaitTimeout.
        (
            "predictors.wait",
            ("risk",),
            {"timeout": 0},
            {
                ("GET", "/predictors/risk/environments/staging"): ServiceError(
                    "Timeout", "request deadline exceeded", 504
                )
            },
            "WaitTimeout",
        ),
    ]
    for call, args, kwargs, refused, raised in cases:
        outcomes = []
        for client in clients():
            recorder = Refusing(results(lock, revision), refused)
            try:
                record(client, recorder, call, args, kwargs)
                outcomes.append((recorder.sent, None))
            except ServiceError as error:
                outcomes.append((recorder.sent, error.code))
        assert outcomes[0] == outcomes[1], call
        assert outcomes[0][1] == raised, call


def test_every_versioned_request_goes_through_call(monkeypatch):
    seen = []

    def sync_call(self, method, path, *args, **kwargs):
        seen.append(("sync", method, path))
        return {"keys": []}

    async def async_call(self, method, path, *args, **kwargs):
        seen.append(("async", method, path))
        return {"keys": []}

    monkeypatch.setattr(Client, "_call", sync_call)
    monkeypatch.setattr(AsyncClient, "_call", async_call)
    sync, asynchronous = clients()
    assert sync.keys.list() == []
    assert asyncio.run(asynchronous.keys.list()) == []
    assert seen == [("sync", "GET", "/keys"), ("async", "GET", "/keys")]
