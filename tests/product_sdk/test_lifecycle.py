# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP SDK contract coverage; fixture predictions are not inference evidence."""

import asyncio
import dataclasses
import inspect
import time

import pytest
import yaml

from generalforecast import predictors
from generalforecast.async_client import AsyncClient
from generalforecast.client import Client
from generalforecast.contracts import (
    ContextImportRef,
    Environment,
    PredictionResult,
    PredictorLock,
    PromoteRequest,
    Revision,
)
from generalforecast.predictors.lock import read_lock
from generalforecast.service.contracts import ServiceError
from generalforecast.service.sdk import ContextSpec, ImportReceipt
from generalforecast.tabular import table_digest


async def invoke(function, *args, **kwargs):
    value = function(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


def checked(tmp_path, *, selection=False):
    """A checked predictor directory: gf.yaml, its snapshot and the gf.lock it resolved to."""
    spec = {
        "name": "risk",
        "model": "general-forecast/tabicl-v2-classifier",
        "target": "y",
        "context": {"snapshot": "history.csv"},
        "id": "id",
        "inputs": ["x"],
    }
    if selection:
        spec["selection"] = {"core": 0.5, "neighbors": 0.5, "kind": "stratified"}
    (tmp_path / "gf.yaml").write_text(yaml.safe_dump(spec))
    (tmp_path / "history.csv").write_text("id,x,y\na,1,false\nb,2,true\nc,3,false\n")
    predictors.write_lock(
        predictors.resolve(predictors.load(tmp_path), tmp_path), tmp_path
    )
    return tmp_path


def checked_lock(root) -> PredictorLock:
    """The lock a checked predictor directory holds."""
    document = read_lock(root)
    assert document is not None
    return PredictorLock(document)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_empty_environment_never_falls_back_to_production(
    wire, tmp_path, asynchronous
):
    authored = checked(tmp_path)

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            for operation, args, kwargs in (
                (client.predictors.environment, ("risk", ""), {}),
                (client.predictors.metrics, ("risk",), {"env": ""}),
                (client.predictors.logs, ("risk",), {"env": ""}),
                (client.predictors.predict, ("risk", {"x": [1]}), {"env": ""}),
                (
                    client.predictors.promote,
                    ("risk",),
                    {"env": "", "revision": 1, "expected_generation": 0},
                ),
                (
                    client.predictors.rollback,
                    ("risk",),
                    {"env": "", "revision": 1, "expected_generation": 0},
                ),
                (client.predictors.push, (authored,), {"env": "", "runtime": "local"}),
            ):
                with pytest.raises(ServiceError):
                    await invoke(operation, *args, **kwargs)
            assert not wire.requests
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_push_cas_readiness_and_rollback(wire, tmp_path, asynchronous):
    authored = checked(tmp_path, selection=True)
    lock = checked_lock(authored)
    assert lock.context_kind == "snapshot" and lock.context["rows"] == 3
    revision = Revision("risk", 1, lock.digest, lock.model, time.time())
    initial = Environment("risk", "production", None, 0, "local")
    pending = initial.promote(PromoteRequest(1, 0))
    ready = dataclasses.replace(pending, status="ready")
    base = "/predictors/risk"
    wire.expect("GET", base, "NotFound", status=404)
    wire.expect("POST", "/predictors", {"name": "risk"}, payload={"name": "risk"})
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        base + "/revisions",
        revision.to_dict(),
        payload={"lock": lock.to_dict(), "snapshot": lock.data_digest},
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
        pending.to_dict(),
        payload={"revision": 1, "expected_generation": 0},
    )
    wire.expect("GET", base + "/environments/production", pending.to_dict())
    wire.expect("GET", base + "/environments/production", ready.to_dict())
    wire.expect("GET", base, {"name": "risk"})
    wire.expect("GET", base + "/environments/production", ready.to_dict())
    wire.expect("GET", base + "/revisions", {"revisions": [revision.to_dict()]})
    wire.expect(
        "GET",
        base + "/revisions/1",
        {"revision": revision.to_dict(), "lock": lock.to_dict()},
    )
    wire.expect("GET", base + "/environments", {"environments": [ready.to_dict()]})
    wire.expect(
        "GET",
        base + "/environments/production/metrics",
        {"context": {"source": "snapshot", "rows": 2}, "requests": 0},
    )
    wire.expect("GET", base + "/environments/production/logs", {"logs": []})
    wire.expect(
        "POST",
        base + "/environments/production/rollback",
        "Conflict",
        payload={"revision": 1, "expected_generation": 1},
        status=409,
    )
    wire.expect(
        "POST", base + "/delete", {"name": "risk", "status": "stopped"}, payload={}
    )

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            pushed = await invoke(
                client.predictors.push, authored, runtime="local", wait=True
            )
            assert pushed == revision
            uploaded = next(
                value for path, value in wire.requests if path.endswith("/snapshots")
            )
            assert uploaded.num_rows == 3 and table_digest(uploaded) == lock.data_digest
            predictor = await invoke(client.predictors.get, "risk", env="production")
            assert predictor.revision == 1 and predictor.status == "ready"
            with pytest.raises(dataclasses.FrozenInstanceError):
                predictor.name = "changed"
            assert await invoke(client.predictors.revisions, "risk") == (revision,)
            assert (await invoke(client.predictors.revision, "risk", 1)).lock == lock
            assert await invoke(client.predictors.environments, "risk") == (ready,)
            assert (await invoke(predictor.metrics))["requests"] == 0
            assert (await invoke(predictor.logs))["logs"] == []
            with pytest.raises(ServiceError) as error:
                await invoke(predictor.rollback, revision=1)
            assert error.value.code == "Conflict" and "fixture-key" not in str(
                error.value
            )
            assert (await invoke(predictor.delete))["status"] == "stopped"
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_predictions_and_context_transactions(wire, asynchronous):
    context = {"x": [1, 2], "y": [False, True]}
    result = {
        "model": "general-forecast/tabicl-v2-classifier",
        "task": "classification",
        "context": {"source": "request", "version": None, "rows": 2, "as_of": None},
        "predictor": None,
        "revision": None,
        "environment": None,
        "predictions": [{"label": True, "probabilities": {"false": 0.1, "true": 0.9}}],
    }
    wire.expect(
        "POST",
        "/predict",
        result,
        payload={
            "model": "general-forecast/tabicl-v2-classifier",
            "target": "y",
            "context": [{"x": 1, "y": False}, {"x": 2, "y": True}],
            "queries": [{"x": 3}],
            "inputs": ["x"],
        },
    )
    deployed_result = result | {
        "predictor": "risk",
        "revision": 1,
        "environment": "production",
    }
    # Production, the services' default, is named by omission.
    wire.expect(
        "POST",
        "/predictors/risk/predict",
        deployed_result,
        payload={"queries": [{"x": 3}]},
    )
    schema = ContextSpec("history", ("x",), {"y": "number"})
    wire.expect(
        "POST",
        "/contexts",
        {"name": "history", "sequence": 0},
        payload=schema.to_dict(),
    )
    wire.expect(
        "POST",
        "/contexts/history/append",
        {"context": "history", "sequence": 1, "write_id": "append-1"},
        payload={"write_id": "append-1", "rows": [{"id": "row1", "x": 2}]},
    )
    wire.expect(
        "POST",
        "/contexts/history/label",
        {"context": "history", "sequence": 2, "write_id": "label-1"},
        payload={
            "write_id": "label-1",
            "target": "y",
            "labels": [{"id": "row1", "value": 1}],
        },
    )
    reference = ContextImportRef(
        "history", "s3://fixture-bucket/events", "parquet", "secret/reader"
    )
    wire.expect(
        "POST",
        "/contexts/history/imports",
        {"digest": "sha256:" + "a" * 64, "sequence": 3},
        payload=reference.to_dict(),
    )
    # A refresh answers with the import it ran, as hosted's imports.refresh does.
    wire.expect(
        "POST",
        "/contexts/history/refresh",
        {
            "context": "history",
            "digest": "sha256:" + "b" * 64,
            "rows": 4,
            "sequence": 3,
            "accepted_at": 1.0,
            "receipt": "receipt-1",
            "mode": "replace",
            "continuous": False,
        },
        payload={},
    )
    wire.expect("GET", "/contexts/history", {"name": "history", "sequence": 3})

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            prediction = await invoke(
                client.predict,
                "general-forecast/tabicl-v2-classifier",
                "y",
                context,
                {"x": [3]},
                inputs=["x"],
            )
            assert isinstance(prediction, PredictionResult)
            assert prediction.to_numpy().shape == (1, 2)
            assert prediction.context.rows == 2
            assert (
                await invoke(client.predictors.predict, "risk", {"x": [3]})
            ).revision == 1
            live = await invoke(client.contexts.create, schema)
            assert live.sequence == 0
            receipt = await invoke(
                live.append,
                [{"id": "row1", "x": 2}],
                id="id",
                idempotency_key="append-1",
            )
            assert receipt.sequence == 1 and receipt.write_id == "append-1"
            assert (
                await invoke(
                    live.label,
                    [{"id": "row1", "y": 1}],
                    id="id",
                    column="y",
                    idempotency_key="label-1",
                )
            ).sequence == 2
            assert (await invoke(live.import_source, reference)).digest.startswith(
                "sha256:"
            )
            refreshed = await invoke(live.refresh)
            assert isinstance(refreshed, ImportReceipt)
            assert (refreshed.sequence, refreshed.rows) == (3, 4) and live.sequence == 0
            assert (await invoke(live.status)).sequence == 3
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "pending", "ready"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_never_accepts_failure_or_a_superseded_pointer(wire, status, asynchronous):
    state = Environment(
        "risk", "production", 2 if status == "ready" else 1, 2, "local", status
    )
    wire.expect("GET", "/predictors/risk/environments/production", state.to_dict())

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            with pytest.raises(ServiceError) as error:
                await invoke(
                    client.predictors.wait,
                    "risk",
                    revision=1,
                    generation=2,
                    timeout=0.3,
                    poll_interval=1,
                )
            assert error.value.code in {
                "EnvironmentFailed",
                "Conflict",
                "WaitTimeout",
                "Timeout",
            }
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("corruption", [True, "boolean"])
def test_snapshot_receipt_integrity(wire, asynchronous, corruption):
    wire.corrupt = corruption
    wire.expect("POST", "/snapshots")

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            with pytest.raises(ServiceError) as error:
                await invoke(client.upload_snapshot, {"x": [1]})
            assert error.value.code == "ProtocolError"
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_file_import_uploads_content_before_logical_transaction(
    wire, tmp_path, asynchronous
):
    from generalforecast.tabular import table_digest, to_arrow

    source = tmp_path / "data.csv"
    source.write_text("id,x,y\na,1,2\nb,2,3\n")
    digest = table_digest(to_arrow({"id": ["a", "b"], "x": [1, 2], "y": [2, 3]}))
    reference = ContextImportRef(
        "history", "upload://" + digest, "arrow", digest=digest
    )
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/contexts/history/imports",
        {"context": "history", "digest": digest, "rows": 2, "sequence": 1},
        payload=reference.to_dict(),
    )

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            receipt = await invoke(client.contexts.import_file, "history", source)
            assert receipt.digest == digest and receipt.sequence == 1
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())
