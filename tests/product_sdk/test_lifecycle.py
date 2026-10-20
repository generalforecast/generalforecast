"""Real HTTP SDK contract coverage; fixture predictions are not inference evidence."""

import asyncio
import dataclasses
import inspect
import time

import pytest
import yaml

import generalforecast as gf
from generalforecast.tabular import table_digest


async def invoke(function, *args, **kwargs):
    value = function(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


def checked(tmp_path, *, selection=False):
    spec = {
        "name": "risk",
        "model": "tabicl-classifier",
        "target": "y",
        "context": "history.csv",
        "id": "id",
        "inputs": {"include": ["x"]},
    }
    if selection:
        spec["context"] = {
            "source": "history.csv",
            "selection": "most_recent",
            "order_by": "id",
            "max_rows": 2,
        }
    (tmp_path / "gf.yaml").write_text(yaml.safe_dump(spec))
    (tmp_path / "history.csv").write_text("id,x,y\na,1,false\nb,2,true\nc,3,false\n")
    return gf.Package.load(tmp_path).check()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_empty_environment_never_falls_back_to_production(
    wire, tmp_path, asynchronous
):
    package = checked(tmp_path)

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            for operation, args, kwargs in (
                (client.deployments.environment, ("risk", ""), {}),
                (client.deployments.metrics, ("risk",), {"env": ""}),
                (client.deployments.logs, ("risk",), {"env": ""}),
                (client.deployments.predict, ("risk", {"x": [1]}), {"env": ""}),
                (
                    client.deployments.promote,
                    ("risk",),
                    {"env": "", "revision": 1, "expected_generation": 0},
                ),
                (
                    client.deployments.rollback,
                    ("risk",),
                    {"env": "", "revision": 1, "expected_generation": 0},
                ),
                (client.deployments.push, (package,), {"env": "", "runtime": "local"}),
            ):
                with pytest.raises(gf.ServiceError):
                    await invoke(operation, *args, **kwargs)
            assert not wire.requests
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_push_cas_readiness_rollback_and_dev_patch(wire, tmp_path, asynchronous):
    package = checked(tmp_path, selection=True)
    lock = package.compile()
    assert package.context_table().num_rows == 2
    assert package.snapshot_table().num_rows == 3
    revision = gf.Revision("risk", 1, lock.package_digest, lock.model.name, time.time())
    initial = gf.Environment("risk", "production", None, 0, "local")
    pending = initial.promote(gf.PromoteRequest(1, 0))
    ready = dataclasses.replace(pending, status="ready")
    base = "/deployments/risk"
    wire.expect("GET", base, "NotFound", status=404)
    wire.expect("POST", "/deployments", {"name": "risk"}, payload={"name": "risk"})
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
    dev = gf.DevSession(
        "dev-test", "risk", 1, 0, lock.package_digest, time.time() + 600, "ready"
    )
    wire.expect(
        "POST",
        base + "/dev-sessions",
        dev.to_dict(),
        payload={"base_revision": 1, "runtime": "local", "ttl_seconds": 3600},
    )
    wire.expect("GET", base + "/dev-sessions/dev-test", dev.to_dict())
    wire.expect("POST", "/snapshots")
    patch = gf.PatchRequest(0, 1, lock, lock.data_digest)
    patched = dev.apply_patch(patch, now=time.time())
    wire.expect(
        "POST",
        base + "/dev-sessions/dev-test/patch",
        patched.to_dict(),
        payload=patch.to_dict(),
    )
    wire.expect("GET", base + "/environments/production", ready.to_dict())
    wire.expect(
        "POST",
        base + "/dev-sessions/dev-test/stop",
        dataclasses.replace(patched, status="stopped").to_dict(),
        payload={},
    )
    wire.expect(
        "POST", base + "/delete", {"name": "risk", "status": "stopped"}, payload={}
    )

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            pushed = await invoke(
                client.deployments.push, package, runtime="local", wait=True
            )
            assert pushed == revision
            uploaded = next(
                value for path, value in wire.requests if path.endswith("/snapshots")
            )
            assert uploaded.num_rows == 3 and table_digest(uploaded) == lock.data_digest
            deployment = await invoke(client.deployments.get, "risk", env="production")
            assert deployment.revision == 1 and deployment.status == "ready"
            with pytest.raises(dataclasses.FrozenInstanceError):
                deployment.name = "changed"
            assert await invoke(client.deployments.revisions, "risk") == (revision,)
            assert (await invoke(client.deployments.revision, "risk", 1)).lock == lock
            assert await invoke(client.deployments.environments, "risk") == (ready,)
            assert (await invoke(deployment.metrics))["requests"] == 0
            assert (await invoke(deployment.logs))["logs"] == []
            with pytest.raises(gf.ServiceError) as error:
                await invoke(deployment.rollback, revision=1)
            assert error.value.code == "Conflict" and "fixture-key" not in str(
                error.value
            )
            development = await invoke(
                client.dev_sessions.start, "risk", base_revision=1, runtime="local"
            )
            assert (await invoke(development.refresh)).state == dev
            updated = await invoke(development.patch, package)
            assert updated.generation == 1 and development.generation == 0
            assert await invoke(client.deployments.environment, "risk") == ready
            assert (await invoke(updated.stop)).state.status == "stopped"
            assert (await invoke(deployment.delete))["status"] == "stopped"
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_predictions_dataset_transactions_jobs_and_runtimes(wire, asynchronous):
    context = {"x": [1, 2], "y": [False, True]}
    result = {
        "model": "catalog/tabicl-v2-classifier",
        "task": "classification",
        "context": {"source": "request", "version": None, "rows": 2, "as_of": None},
        "deployment": None,
        "revision": None,
        "environment": None,
        "predictions": [{"label": True, "probabilities": {"false": 0.1, "true": 0.9}}],
    }
    wire.expect(
        "POST",
        "/predictions",
        result,
        payload={
            "model": "tabicl-classifier",
            "task": "classification",
            "options": {},
            "target": "y",
            "features": ["x"],
            "context": [{"x": 1, "y": False}, {"x": 2, "y": True}],
            "queries": [{"x": 3}],
        },
    )
    deployed_result = result | {
        "deployment": "risk",
        "revision": 1,
        "environment": "production",
    }
    wire.expect(
        "POST",
        "/deployments/risk/environments/production/predict",
        deployed_result,
        payload={"queries": [{"x": 3}]},
    )
    wire.expect(
        "POST",
        "/deployments/risk/dev-sessions/dev-test/predict",
        deployed_result,
        payload={"queries": [{"x": 4}]},
    )
    schema = gf.DatasetSpec("history", ("x",), (("y", "numeric"),))
    wire.expect(
        "POST",
        "/datasets",
        {"name": "history", "sequence": 0},
        payload=schema.to_dict(),
    )
    wire.expect(
        "POST",
        "/datasets/history/mutations",
        {"sequence": 1, "write_id": "append-1"},
        payload={
            "write_id": "append-1",
            "operation": "create_features",
            "records": [{"key": "row1", "features": {"x": 2}}],
        },
    )
    wire.expect(
        "POST",
        "/datasets/history/mutations",
        {"sequence": 2, "write_id": "label-1"},
        payload={
            "write_id": "label-1",
            "operation": "attach_labels",
            "records": [{"key": "row1", "labels": {"y": 1}}],
        },
    )
    reference = gf.DatasetImportRef(
        "history", "s3://fixture-bucket/events", "parquet", "secret/reader"
    )
    wire.expect(
        "POST",
        "/datasets/history/imports",
        {"digest": "sha256:" + "a" * 64, "sequence": 3},
        payload=reference.to_dict(),
    )
    wire.expect(
        "POST",
        "/datasets/history/refresh",
        {"name": "history", "sequence": 3},
        payload={},
    )
    wire.expect("GET", "/datasets/history", {"name": "history", "sequence": 3})
    spec = gf.JobSpec(
        "score",
        "submit-1",
        "dataset.snapshot",
        gf.Snapshot.from_dict({"dataset": "history"}),
    )
    wire.expect(
        "POST", "/jobs", {"name": "score", "state": "queued"}, payload=spec.to_dict()
    )
    wire.expect("GET", "/jobs/score", {"name": "score", "state": "running"})
    wire.expect("GET", "/jobs/score", {"name": "score", "state": "completed"})
    wire.expect(
        "GET",
        "/jobs/score/results?offset=0&limit=100",
        {"results": [{"row": 1}], "next_offset": None},
    )
    wire.expect(
        "POST",
        "/jobs/score/cancel",
        {"name": "score", "state": "cancelled"},
        payload={},
    )
    wire.expect(
        "GET", "/runtimes", {"runtimes": [{"name": "local", "status": "ready"}]}
    )
    wire.expect("GET", "/runtimes/local", {"name": "local", "status": "ready"})
    for action in ("preflight", "connect", "cleanup"):
        wire.expect(
            "POST",
            "/runtimes/local/" + action,
            "UnsupportedOperation",
            payload={},
            status=501,
        )

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            prediction = await invoke(
                client.predict,
                model="tabicl-classifier",
                target="y",
                context=context,
                queries={"x": [3]},
                features=["x"],
            )
            assert isinstance(prediction, gf.PredictionResult)
            assert prediction.to_numpy().shape == (1, 2)
            assert prediction.context.rows == 2
            assert (
                await invoke(client.deployments.predict, "risk", {"x": [3]})
            ).revision == 1
            assert (
                await invoke(
                    client.dev_sessions.predict, "risk", "dev-test", {"x": [4]}
                )
            ).revision == 1
            dataset = await invoke(client.datasets.create, schema)
            assert dataset.sequence == 0
            receipt = await invoke(
                dataset.append,
                [{"id": "row1", "x": 2}],
                id="id",
                idempotency_key="append-1",
            )
            assert receipt.sequence == 1 and receipt.write_id == "append-1"
            assert (
                await invoke(
                    dataset.label,
                    [{"id": "row1", "y": 1}],
                    id="id",
                    column="y",
                    idempotency_key="label-1",
                )
            ).sequence == 2
            assert (await invoke(dataset.import_source, reference)).digest.startswith(
                "sha256:"
            )
            assert (
                await invoke(dataset.refresh)
            ).sequence == 3 and dataset.sequence == 0
            assert (await invoke(dataset.status)).sequence == 3
            job = await invoke(client.jobs.submit, spec)
            assert (await invoke(job.poll)).state == "running" and job.state == "queued"
            assert (await invoke(job.wait)).state == "completed"
            assert (await invoke(job.results))["results"] == [{"row": 1}]
            assert (await invoke(job.cancel)).state == "cancelled"
            assert (await invoke(client.runtimes.list))[0].name == "local"
            assert (await invoke(client.runtimes.get, "local"))["status"] == "ready"
            for action in ("preflight", "connect", "cleanup"):
                with pytest.raises(gf.ServiceError) as error:
                    await invoke(getattr(client.runtimes, action), "local")
                assert error.value.status == 501
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "pending", "ready"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_never_accepts_failure_or_a_superseded_pointer(wire, status, asynchronous):
    state = gf.Environment(
        "risk", "production", 2 if status == "ready" else 1, 2, "local", status
    )
    wire.expect("GET", "/deployments/risk/environments/production", state.to_dict())

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            with pytest.raises(gf.ServiceError) as error:
                await invoke(
                    client.deployments.wait,
                    "risk",
                    revision=1,
                    generation=2,
                    timeout=0.01,
                    poll_interval=0.02,
                )
            assert error.value.code in {
                "DeploymentFailed",
                "Conflict",
                "WaitTimeout",
                "Timeout",
            }
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("corruption", [True, "boolean"])
def test_snapshot_receipt_integrity(wire, asynchronous, corruption):
    wire.corrupt = corruption
    wire.expect("POST", "/snapshots")

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            with pytest.raises(gf.ServiceError) as error:
                await invoke(client.upload_snapshot, {"x": [1]})
            assert error.value.code == "ProtocolError"
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
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
    reference = gf.DatasetImportRef(
        "history", "upload://" + digest, "arrow", digest=digest
    )
    wire.expect("POST", "/snapshots")
    wire.expect(
        "POST",
        "/datasets/history/imports",
        {"dataset": "history", "digest": digest, "rows": 2, "sequence": 1},
        payload=reference.to_dict(),
    )

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            receipt = await invoke(client.datasets.import_file, "history", source)
            assert receipt.digest == digest and receipt.sequence == 1
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())
