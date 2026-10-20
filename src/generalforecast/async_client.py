"""Native asynchronous SDK with the same scoped resources and immutable values."""

from __future__ import annotations

import asyncio
import builtins
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, cast
from urllib.parse import urlencode

from generalforecast.client import WORKSPACE_ACTIONS, workspace_request
from generalforecast.contracts import (
    DatasetImportRef,
    DevSession,
    Environment,
    PatchRequest,
    PredictionResult,
    PromoteRequest,
    Revision,
)
from generalforecast.deployments.boundary import (
    JsonObject,
    Request,
    Resource,
    ResourceKind,
    Scope,
    json_object,
)
from generalforecast.deployments.contracts import ServiceError, name, number, require
from generalforecast.deployments.interfaces import (
    AcceptedWrite,
    ContextSnapshot,
    InferenceRequest,
    InferenceResult,
    Mutation,
    SourcePin,
)
from generalforecast.deployments.sdk import (
    PYTHON_SOURCE_MEDIA_TYPE,
    BoundSnapshot,
    DatasetSpec,
    ImportReceipt,
    JobSpec,
    RevisionDetails,
    Runtime,
    Snapshot,
    WriteReceipt,
    captured_environment,
    checked_lock,
    checked_python_source,
    environment_name,
    inference_result,
    mutation,
    one_shot_payload,
    prediction_payload,
    python_source_receipt,
    python_source_upload,
    table,
    unwrap,
)
from generalforecast.models.catalog import resource_name
from generalforecast.packages import Package, PackageLock, PythonLock
from generalforecast.tabular import arrow_bytes, table_digest
from generalforecast.transport import AsyncTransport, Profile, connection


class AsyncClient:
    """One explicit endpoint/project and independent application/provider keys.

    Uses only versioned routes. Request IDs correlate; callers must preserve
    write_id/submission_id on retries. No automatic retries or cookie login.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        project: str | None = None,
        token: str | None = None,
        api_key: str | None = None,
        profile: str | Profile | None = None,
        config_path: str | Path | None = None,
        **transport_options: Any,
    ) -> None:
        """Resolves scoped GF_API_KEY/profile auth without persisting credentials."""
        self.profile, key = connection(
            url, project, token, api_key, profile, config_path
        )
        self.project = self.profile.project
        self._transport = AsyncTransport(
            self.profile.url, token=key, **transport_options
        )
        self.datasets = Datasets(self)
        self.sources = Sources(self)
        self.models = Models(self)
        self.jobs = Jobs(self)
        self.packages = Packages(self)
        self.deployments = Deployments(self)
        self.identity = Identity(self)
        self.keys = Keys(self)
        self.secrets = Secrets(self)
        self.operators = Operators(self)
        self.runtimes = Runtimes(self)
        self.dev_sessions = DevSessions(self)

    async def __aenter__(self) -> Self:
        """Returns this asynchronous client."""
        return self

    async def __aexit__(self, *_args) -> None:
        """Closes the client on scope exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Closes the transport; snapshots remain immutable and readable."""
        await self._transport.aclose()

    async def predict(
        self,
        *,
        model: str,
        target: str,
        context: Any,
        queries: Any,
        context_columns: Sequence[str] | None = None,
        task: str | None = None,
        features: Sequence[str] | None = None,
        options: JsonObject | None = None,
        outputs: Sequence[str] | None = None,
    ) -> PredictionResult:
        """Runs one-shot remote inference without creating or mutating a dataset."""
        payload = await asyncio.to_thread(
            one_shot_payload,
            queries,
            model=model,
            task=task,
            context=context,
            context_columns=context_columns,
            target=target,
            features=features,
            options=options,
            outputs=outputs,
        )
        return inference_result(
            await self._call("POST", "/predictions", "model", "predictions", payload)
        )

    async def upload_snapshot(self, data: Any) -> JsonObject:
        """Uploads the full normalized Arrow snapshot and verifies the returned digest."""
        normalized = await asyncio.to_thread(table, data)
        encoded = await asyncio.to_thread(arrow_bytes, normalized)
        digest = await asyncio.to_thread(table_digest, normalized)
        request_id = "r-" + uuid.uuid4().hex
        value = await self._transport.send(
            "POST",
            f"/v1alpha1/projects/{self.project}/snapshots",
            encoded,
            {
                "Content-Type": "application/vnd.apache.arrow.stream",
                "X-GF-Project": self.project,
                "X-GF-Request-ID": request_id,
            },
        )
        request = Request(
            Scope(self.project, request_id), Resource("dataset", "snapshots"), {}
        )
        receipt = unwrap(value, request)
        if type(receipt.get("rows")) is not int or receipt != {
            "digest": digest,
            "rows": normalized.num_rows,
        }:
            raise ServiceError(
                "ProtocolError", "snapshot receipt does not match uploaded content", 502
            )
        return receipt

    async def upload_python_source(
        self, deployment: str, archive: bytes, *, source: str
    ) -> Snapshot:
        """Uploads bounded source with native async transport and off-loop hashing."""
        expected = await asyncio.to_thread(
            python_source_upload, self.project, deployment, archive, source
        )
        request_id = "r-" + uuid.uuid4().hex
        value = await self._transport.send(
            "POST",
            f"/v1alpha1/projects/{self.project}/deployments/{deployment}/python-sources",
            archive,
            {
                "Content-Type": PYTHON_SOURCE_MEDIA_TYPE,
                "X-GF-Project": self.project,
                "X-GF-Request-ID": request_id,
            },
        )
        request = Request(
            Scope(self.project, request_id), Resource("deployment", deployment), {}
        )
        return python_source_receipt(unwrap(value, request), expected)

    async def _call(
        self,
        method: str,
        path: str,
        kind: ResourceKind,
        resource: str,
        payload: JsonObject | None = None,
        *,
        global_route: bool = False,
        params: JsonObject | None = None,
        deadline: float | None = None,
    ) -> JsonObject:
        request = Request.parse(
            Request(
                Scope(self.project, "r-" + uuid.uuid4().hex),
                Resource(kind, resource),
                {} if payload is None else payload,
            ).to_dict()
        )
        prefix = "/v1alpha1" if global_route else f"/v1alpha1/projects/{self.project}"
        query = "?" + urlencode(params) if params else ""
        options = {} if deadline is None else {"deadline": deadline}
        value = await self._transport.send(
            method,
            prefix + path + query,
            request.to_dict() if method == "POST" else None,
            {
                "X-GF-Request-ID": request.scope.request_id,
                "X-GF-Project": self.project,
            },
            **options,
        )
        return unwrap(value, request)


class Collection:
    """Common bounded list/get operations, never arbitrary remote dispatch."""

    kind: ResourceKind
    collection: str

    def __init__(self, client: AsyncClient) -> None:
        """Uses the parent's authenticated, project-scoped transport."""
        self.client = client

    async def list(self) -> builtins.list[JsonObject]:
        """Returns this project's actual resource metadata."""
        result = await self.client._call(
            "GET", f"/{self.collection}", self.kind, self.collection
        )
        value = result.get(self.collection)
        if type(value) is not list or any(type(item) is not dict for item in value):
            raise ServiceError("ProtocolError", "invalid collection response", 502)
        return value

    async def get(self, resource: str) -> Mapping[str, Any]:
        """Gets one project-qualified resource."""
        return await self.client._call(
            "GET", f"/{self.collection}/{name(resource)}", self.kind, resource
        )

    async def _post(
        self, resource: str, action: str, payload: JsonObject
    ) -> JsonObject:
        suffix = f"/{name(resource)}" + (f"/{action}" if action else "")
        return await self.client._call(
            "POST", f"/{self.collection}{suffix}", self.kind, resource, payload
        )

    async def _create(self, resource: str, payload: JsonObject) -> JsonObject:
        return await self.client._call(
            "POST", f"/{self.collection}", self.kind, resource, payload
        )


class Datasets(Collection):
    """Create-only observations and guarded component corrections."""

    kind: ResourceKind = "dataset"
    collection = "datasets"

    async def create(self, config: JsonObject | DatasetSpec) -> AsyncDataset:
        """Creates a bounded immutable schema using the direct config payload."""
        value = config.to_dict() if isinstance(config, DatasetSpec) else config
        return AsyncDataset.from_response(
            self.client, value["name"], await self._create(value["name"], value)
        )

    async def get(self, resource: str) -> AsyncDataset:
        """Returns a detached dataset snapshot with typed mutation methods."""
        return AsyncDataset.from_response(
            self.client, resource, await super().get(resource)
        )

    async def append(
        self, dataset: str, data: Any, *, id: str, idempotency_key: str
    ) -> WriteReceipt:
        """Appends observations using the existing durable write-ID namespace."""
        return WriteReceipt.from_dict(
            await self._post(
                dataset,
                "mutations",
                await asyncio.to_thread(
                    mutation,
                    data,
                    id=id,
                    operation="create_features",
                    write_id=idempotency_key,
                ),
            )
        )

    async def label(
        self, dataset: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches outcome components independently of prediction requests."""
        return WriteReceipt.from_dict(
            await self._post(
                dataset,
                "mutations",
                await asyncio.to_thread(
                    mutation,
                    data,
                    id=id,
                    column=column,
                    operation="attach_labels",
                    write_id=idempotency_key,
                ),
            )
        )

    async def import_source(self, reference: DatasetImportRef) -> ImportReceipt:
        """Binds a cloud or uploaded table to an existing logical dataset."""
        return ImportReceipt.from_dict(
            await self._post(reference.dataset, "imports", reference.to_dict())
        )

    async def import_file(self, dataset: str, data: Any) -> ImportReceipt:
        """Uploads a local bounded table before binding it to a logical dataset."""
        receipt = await self.client.upload_snapshot(data)
        return await self.import_source(
            DatasetImportRef(
                dataset,
                "upload://" + receipt["digest"],
                "arrow",
                digest=receipt["digest"],
            )
        )

    async def refresh(self, dataset: str) -> AsyncDataset:
        """Requests an explicit provider refresh; unsupported adapters fail visibly."""
        return AsyncDataset.from_response(
            self.client, dataset, await self._post(dataset, "refresh", {})
        )

    async def mutate(self, dataset: str, mutation: Mutation) -> AcceptedWrite:
        """Accepts an atomic write, not serving readiness. Preserve write_id."""
        return cast(
            AcceptedWrite, await self._post(dataset, "mutations", dict(mutation))
        )

    async def snapshot(
        self,
        dataset: str,
        *,
        target: str,
        knowledge_at: float | None = None,
        event_before: float | None = None,
    ) -> ContextSnapshot:
        """Keeps knowledge/acceptance time independent from event time."""
        return cast(
            ContextSnapshot,
            await self._post(
                dataset,
                "snapshot",
                {
                    "target": target,
                    "knowledge_at": knowledge_at,
                    "event_before": event_before,
                },
            ),
        )


class Sources(Collection):
    """Immutable pins of files under the server's approved import root."""

    kind: ResourceKind = "source"
    collection = "sources"

    async def pin(self, spec: JsonObject) -> SourcePin:
        """Pins server-relative bytes; this does not upload a client-side path."""
        return cast(SourcePin, await self._create("new", spec))


class Models(Collection):
    """Curated model metadata and pure caller-context inference."""

    kind: ResourceKind = "model"
    collection = "models"

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Never ingests queries; returns genuine outputs or a typed gate."""
        return cast(
            InferenceResult,
            await self._post(resource_name(request["model"]), "infer", dict(request)),
        )


class Jobs(Collection):
    """Durable jobs with linked retries and paginated per-occurrence reports."""

    kind: ResourceKind = "job"
    collection = "jobs"

    async def submit(self, spec: JsonObject | JobSpec) -> AsyncJob:
        """Preserves submission_id; acceptance is not job completion."""
        value = spec.to_dict() if isinstance(spec, JobSpec) else spec
        return AsyncJob.from_response(
            self.client, value["name"], await self._create(value["name"], value)
        )

    async def get(self, resource: str) -> AsyncJob:
        """Returns a durable job snapshot; polling creates new snapshots."""
        return AsyncJob.from_response(
            self.client, resource, await super().get(resource)
        )

    async def wait(
        self, job: str, *, timeout: float = 120, poll_interval: float = 0.2
    ) -> AsyncJob:
        """Polls a durable job with one deadline; failures are never successful results."""
        number(timeout, 0.001, 86400, "timeout")
        number(poll_interval, 0.001, 30, "poll_interval")
        deadline = time.monotonic() + timeout
        while True:
            state = AsyncJob.from_response(
                self.client,
                job,
                await self.client._call(
                    "GET", f"/jobs/{name(job)}", self.kind, job, deadline=deadline
                ),
            )
            if state.state in {"completed", "completed_with_issues"}:
                return state
            if state.state in {"failed", "cancelled"}:
                raise ServiceError(
                    "JobFailed", "job did not complete successfully", 409
                )
            if time.monotonic() >= deadline:
                raise ServiceError("WaitTimeout", "job readiness timed out", 504)
            await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    async def control(
        self, job: str, action: Literal["pause", "resume", "cancel", "retry"]
    ) -> JsonObject:
        """Retry links a NEW job; it does not rewrite a completed report."""
        require(action in ("pause", "resume", "cancel", "retry"), "invalid job action")
        return await self._post(job, action, {})

    async def results(
        self, job: str, *, offset: int = 0, limit: int = 100
    ) -> JsonObject:
        """Pages every occurrence, including skipped and rejected records."""
        require(type(offset) is int and 0 <= offset <= 2**53 - 1, "invalid offset")
        require(type(limit) is int and 1 <= limit <= 100, "invalid limit")
        return await self.client._call(
            "GET",
            f"/jobs/{name(job)}/results",
            self.kind,
            job,
            params={"offset": offset, "limit": limit},
        )

    async def iter_results(self, job: str) -> AsyncIterator[JsonObject]:
        """Iterates all report pages, rejecting a non-progressing cursor."""
        offset = 0
        while True:
            page = await self.results(job, offset=offset)
            rows, next_offset = (page.get("results"), page.get("next_offset"))
            if type(rows) is not list or any(type(row) is not dict for row in rows):
                raise ServiceError("ProtocolError", "invalid job results", 502)
            for row in rows:
                yield row
            if next_offset is None:
                return
            if type(next_offset) is not int or next_offset <= offset:
                raise ServiceError("ProtocolError", "job results did not advance", 502)
            offset = next_offset


class Packages(Collection):
    """Immutable packages. A test succeeds only with a proven isolation boundary."""

    kind: ResourceKind = "package"
    collection = "packages"

    @staticmethod
    async def load(path: str | Path = ".") -> Package:
        """Loads a declarative package without network access."""
        return await asyncio.to_thread(Package.load, path)

    @staticmethod
    async def check(package: Package, **options: Any) -> Package:
        """Returns a new checked package using the canonical compiler."""
        return await asyncio.to_thread(package.check, **options)

    async def inspect(self, manifest: JsonObject) -> JsonObject:
        """Validates metadata without executing untrusted code."""
        return await self.client._call(
            "POST", "/packages/inspect", self.kind, "new", manifest
        )

    async def register(self, manifest: JsonObject) -> JsonObject:
        """Registers a digest-verified manifest, not a running deployment."""
        return await self._create(manifest["name"], manifest)

    async def test(self, package: str) -> JsonObject:
        """Preserves typed IsolationUnavailable when the boundary is absent."""
        return await self._post(package, "test", {})


class Deployments(Collection):
    """Immutable package revisions and runtime-bound environment operations."""

    kind: ResourceKind = "deployment"
    collection = "deployments"

    async def get(self, resource: str, *, env: str | None = None) -> AsyncDeployment:
        """Returns immutable deployment metadata and optional environment state."""
        state = await super().get(resource)
        environment = await self.environment(resource, env) if env is not None else None
        return AsyncDeployment.from_response(self.client, resource, state, environment)

    async def create(self, deployment: str) -> AsyncDeployment:
        """Creates only the deployment identity, not serving readiness."""
        return AsyncDeployment.from_response(
            self.client,
            deployment,
            await self._create(deployment, {"name": name(deployment)}),
        )

    async def revisions(self, deployment: str) -> tuple[Revision, ...]:
        """Lists immutable revisions without inferring environment readiness."""
        value = await self.client._call(
            "GET", f"/deployments/{name(deployment)}/revisions", self.kind, deployment
        )
        return tuple(Revision.from_dict(row) for row in value["revisions"])

    async def revision(self, deployment: str, number: int) -> RevisionDetails:
        """Reads the exact immutable compiled lock for a revision."""
        require(type(number) is int and number > 0, "invalid revision")
        value = await self.client._call(
            "GET",
            f"/deployments/{name(deployment)}/revisions/{number}",
            self.kind,
            deployment,
        )
        return RevisionDetails(
            Revision.from_dict(value["revision"]), PackageLock.from_dict(value["lock"])
        )

    async def environments(self, deployment: str) -> tuple[Environment, ...]:
        """Lists runtime-bound environment pointers and their actual states."""
        value = await self.client._call(
            "GET",
            f"/deployments/{name(deployment)}/environments",
            self.kind,
            deployment,
        )
        return tuple(Environment.from_dict(row) for row in value["environments"])

    async def environment(
        self, deployment: str, env: str | None = None, *, deadline: float | None = None
    ) -> Environment:
        """Gets one environment with revision and CAS generation."""
        selected = environment_name(env, self.client.profile.environment)
        return Environment.from_dict(
            await self.client._call(
                "GET",
                f"/deployments/{name(deployment)}/environments/{selected}",
                self.kind,
                deployment,
                deadline=deadline,
            )
        )

    async def create_environment(
        self, deployment: str, *, env: str, runtime: str
    ) -> Environment:
        """Binds a runtime separately from package content and authentication."""
        return Environment.from_dict(
            await self._post(
                deployment,
                "environments",
                {"name": name(env), "runtime": name(runtime)},
            )
        )

    async def promote(
        self,
        deployment: str,
        *,
        revision: int,
        env: str | None = None,
        expected_generation: int | None = None,
    ) -> Environment:
        """CAS-promotes an explicit revision; races remain visible conflicts."""
        selected = environment_name(env, self.client.profile.environment)
        generation = (
            (await self.environment(deployment, selected)).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            await self._post(
                deployment,
                f"environments/{name(selected)}/promote",
                PromoteRequest(revision, generation).to_dict(),
            )
        )

    async def rollback(
        self,
        deployment: str,
        *,
        revision: int,
        env: str | None = None,
        expected_generation: int | None = None,
    ) -> Environment:
        """Rolls back to an explicit immutable revision under the same CAS guard."""
        selected = environment_name(env, self.client.profile.environment)
        generation = (
            (await self.environment(deployment, selected)).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            await self._post(
                deployment,
                f"environments/{name(selected)}/rollback",
                PromoteRequest(revision, generation).to_dict(),
            )
        )

    async def predict(
        self, deployment: str, queries: Any, *, env: str | None = None, **options: Any
    ) -> PredictionResult:
        """Predicts through the runtime-bound environment with actual context metadata."""
        return inference_result(
            await self._post(
                deployment,
                f"environments/{environment_name(env, self.client.profile.environment)}/predict",
                await asyncio.to_thread(prediction_payload, queries, **options),
            )
        )

    async def push(
        self,
        package: Package,
        *,
        env: str | None = None,
        runtime: str | None = None,
        wait: bool = False,
        timeout: float = 120,
    ) -> Revision:
        """Uploads a checked package, registers an immutable revision and CAS-promotes."""
        lock = await asyncio.to_thread(checked_lock, package)
        deployment, selected = (
            lock.spec.name,
            environment_name(env, self.client.profile.environment),
        )
        try:
            await self.get(deployment)
        except ServiceError as error:
            if error.status != 404:
                raise
            await self.create(deployment)
        snapshot = None
        if isinstance(lock, PythonLock):
            await self.client.upload_python_source(
                deployment,
                await asyncio.to_thread(checked_python_source, package, lock),
                source=lock.python.source,
            )
        elif lock.spec.context.mode == "snapshot":
            snapshot = (
                await self.client.upload_snapshot(
                    await asyncio.to_thread(package.snapshot_table)
                )
            )["digest"]
            require(
                snapshot == lock.data_digest, "snapshot differs from checked package"
            )
        revision = Revision.from_dict(
            await self._post(
                deployment, "revisions", {"lock": lock.to_dict(), "snapshot": snapshot}
            )
        )
        try:
            environment = await self.environment(deployment, selected)
            if runtime is not None:
                require(
                    environment.runtime == runtime,
                    "environment is bound to a different runtime",
                )
        except ServiceError as error:
            if error.status != 404:
                raise
            chosen = self.client.profile.runtime if runtime is None else runtime
            require(chosen is not None, "runtime is required for a new environment")
            assert chosen is not None
            environment = await self.create_environment(
                deployment, env=selected, runtime=chosen
            )
        promoted = await self.promote(
            deployment,
            revision=revision.number,
            env=selected,
            expected_generation=environment.generation,
        )
        if wait:
            await self.wait(
                deployment,
                env=selected,
                revision=revision.number,
                generation=promoted.generation,
                timeout=timeout,
            )
        return revision

    async def wait(
        self,
        deployment: str,
        *,
        env: str | None = None,
        revision: int | None = None,
        generation: int | None = None,
        timeout: float = 120,
        poll_interval: float = 0.2,
    ) -> Environment:
        """Polls an exact environment pointer; superseded and failed states never succeed."""
        number(timeout, 0, 86400, "timeout")
        number(poll_interval, 0.001, 30, "poll_interval")
        deadline = time.monotonic() + timeout
        while True:
            state = await self.environment(deployment, env, deadline=deadline)
            if (revision is not None and state.revision != revision) or (
                generation is not None and state.generation != generation
            ):
                raise ServiceError(
                    "Conflict", "environment pointer changed while waiting", 409
                )
            revision, generation = (state.revision, state.generation)
            if state.status == "ready":
                return state
            if state.status in {"failed", "stopped"}:
                raise ServiceError(
                    "DeploymentFailed", "environment did not become ready", 409
                )
            if time.monotonic() >= deadline:
                raise ServiceError(
                    "WaitTimeout", "environment readiness timed out", 504
                )
            await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    async def metrics(self, deployment: str, *, env: str | None = None) -> Snapshot:
        """Gets reported metrics without fabricating missing samples."""
        return Snapshot.from_dict(
            await self.client._call(
                "GET",
                f"/deployments/{name(deployment)}/environments/{environment_name(env, self.client.profile.environment)}/metrics",
                self.kind,
                deployment,
            )
        )

    async def logs(self, deployment: str, *, env: str | None = None) -> Snapshot:
        """Gets bounded safe runtime logs."""
        return Snapshot.from_dict(
            await self.client._call(
                "GET",
                f"/deployments/{name(deployment)}/environments/{environment_name(env, self.client.profile.environment)}/logs",
                self.kind,
                deployment,
            )
        )

    async def delete(self, deployment: str) -> Snapshot:
        """Requests explicit stop/delete, never treating unsupported cleanup as success."""
        return Snapshot.from_dict(await self._post(deployment, "delete", {}))


class DevSessions:
    """Remote dev-only worker sessions with full compiled replacement patches."""

    def __init__(self, client: AsyncClient) -> None:
        """Uses the same authenticated scoped transport as deployment operations."""
        self.client = client

    async def _call(
        self, deployment, session, action="", payload=None, method="POST"
    ) -> JsonObject:
        path = f"/deployments/{name(deployment)}/dev-sessions"
        if session is not None:
            path += "/" + name(session)
        if action:
            path += "/" + action
        return await self.client._call(method, path, "deployment", deployment, payload)

    async def start(
        self,
        deployment: str,
        *,
        base_revision: int,
        runtime: str,
        ttl_seconds: int = 3600,
    ) -> AsyncDevelopment:
        """Starts from an immutable base without selecting any production environment."""
        from generalforecast.contracts import DevSession

        value = await self._call(
            deployment,
            None,
            payload={
                "base_revision": base_revision,
                "runtime": name(runtime),
                "ttl_seconds": ttl_seconds,
            },
        )
        return AsyncDevelopment(DevSession.from_dict(value), self.client)

    async def get(self, deployment: str, session: str) -> AsyncDevelopment:
        """Reads actual development lifecycle state."""
        from generalforecast.contracts import DevSession

        return AsyncDevelopment(
            DevSession.from_dict(await self._call(deployment, session, method="GET")),
            self.client,
        )

    async def patch(
        self, development: AsyncDevelopment, package: Package
    ) -> AsyncDevelopment:
        """CAS-patches a dev worker; stale handles fail without automatic rebase."""
        from generalforecast.contracts import DevSession

        require(
            development._client is self.client,
            "development session belongs to another client",
        )
        lock = await asyncio.to_thread(checked_lock, package)
        require(
            lock.spec.name == development.state.deployment,
            "patch deployment differs from development session",
        )
        if isinstance(lock, PythonLock):
            await self.client.upload_python_source(
                development.state.deployment,
                await asyncio.to_thread(checked_python_source, package, lock),
                source=lock.python.source,
            )
        snapshot = (
            (
                await self.client.upload_snapshot(
                    await asyncio.to_thread(package.snapshot_table)
                )
            )["digest"]
            if lock.spec.context.mode == "snapshot"
            else None
        )
        request = PatchRequest(
            development.generation, development.state.base_revision, lock, snapshot
        )
        return AsyncDevelopment(
            DevSession.from_dict(
                await self._call(
                    development.state.deployment,
                    development.id,
                    "patch",
                    request.to_dict(),
                )
            ),
            self.client,
        )

    async def predict(
        self, deployment: str, session: str, queries: Any, **options: Any
    ) -> PredictionResult:
        """Runs the isolated remote development package, not production."""
        return inference_result(
            await self._call(
                deployment,
                session,
                "predict",
                await asyncio.to_thread(prediction_payload, queries, **options),
            )
        )

    async def stop(self, deployment: str, session: str) -> AsyncDevelopment:
        """Stops only the named remote development session."""
        from generalforecast.contracts import DevSession

        return AsyncDevelopment(
            DevSession.from_dict(await self._call(deployment, session, "stop", {})),
            self.client,
        )


class Runtimes:
    """Authenticated runtime inventory; provisioning is never inferred from a plan."""

    def __init__(self, client: AsyncClient) -> None:
        """Shares project scope and transport, not provider credentials in config."""
        self.client = client

    async def list(self) -> tuple[Runtime, ...]:
        """Lists configured runtime metadata."""
        value = await self.client._call("GET", "/runtimes", "operator", "runtimes")
        return tuple(
            Runtime.from_response(self.client, row["name"], row)
            for row in value["runtimes"]
        )

    async def get(self, runtime: str) -> Runtime:
        """Gets actual runtime state or a typed unsupported/provider error."""
        return Runtime.from_response(
            self.client,
            runtime,
            await self.client._call(
                "GET", f"/runtimes/{name(runtime)}", "operator", runtime
            ),
        )

    async def preflight(self, runtime: str) -> Snapshot:
        """Runs explicit provider preflight separately from authentication."""
        return Snapshot.from_dict(
            await self.client._call(
                "POST", f"/runtimes/{name(runtime)}/preflight", "operator", runtime, {}
            )
        )

    async def connect(self, runtime: str) -> Snapshot:
        """Explicitly connects a configured provider, never changing authentication."""
        return await self._action(runtime, "connect")

    async def cleanup(self, runtime: str) -> Snapshot:
        """Cleans up only the named configured runtime through provider policy."""
        return await self._action(runtime, "cleanup")

    async def _action(self, runtime: str, action: str) -> Snapshot:
        return Snapshot.from_dict(
            await self.client._call(
                "POST", f"/runtimes/{name(runtime)}/{action}", "operator", runtime, {}
            )
        )


class Identity:
    """Allowlisted management routes; authentication never implies membership."""

    def __init__(self, client: AsyncClient) -> None:
        """Uses one scoped client; Google cookie login belongs to the console."""
        self.client = client

    async def manage(
        self, action: str, payload: JsonObject | None = None
    ) -> JsonObject:
        """Calls concrete identity actions, not a user-selected server method."""
        payload = {} if payload is None else json_object(payload)
        if action in WORKSPACE_ACTIONS:
            method, path, resource, body, params = workspace_request(action, payload)
            return await self.client._call(
                method,
                path,
                "workspace",
                resource,
                body,
                global_route=True,
                params=params,
            )
        globals_ = {
            "workspace.get": ("workspace", "/workspace", "workspace"),
            "auth.status": ("workspace", "/auth/status", "auth"),
            "organizations.list": ("workspace", "/organizations", "organizations"),
            "organizations.create": (
                "workspace",
                "/organizations",
                payload.get("name", "new"),
            ),
            "projects.list": ("project", "/projects", "projects"),
            "projects.create": ("project", "/projects", payload.get("name", "new")),
        }
        if action in globals_:
            require(
                action.endswith("create") or not payload,
                "GET operation requires empty payload",
            )
            kind, path, resource = globals_[action]
            return await self.client._call(
                "POST" if action.endswith("create") else "GET",
                path,
                cast(ResourceKind, kind),
                resource,
                payload,
                global_route=True,
            )
        domain, _, operation = action.partition(".")
        if domain in ("keys", "secrets") and operation in (
            {"list", "create", "rotate", "revoke"}
            if domain == "keys"
            else {"list", "create", "get", "rotate", "delete"}
        ):
            if operation == "list":
                require(not payload, "GET operation requires empty payload")
            resource = (
                domain
                if operation == "list"
                else "new"
                if domain == "keys" and operation == "create"
                else payload.get("id" if domain == "keys" else "name", "")
            )
            path = f"/{domain}"
            if operation not in ("list", "create"):
                path += f"/{name(resource)}" + (
                    "" if operation == "get" else f"/{operation}"
                )
            return await self.client._call(
                "GET" if operation in ("list", "get") else "POST",
                path,
                "api-key" if domain == "keys" else "secret",
                resource,
                payload,
            )
        if action in (
            "memberships.list",
            "memberships.add",
            "memberships.remove",
            "sso.get",
            "sso.configure",
            "audit.list",
        ):
            params = None
            if action == "audit.list":
                require(set(payload) <= {"after", "limit"}, "invalid audit parameters")
                after, limit = (payload.get("after", 0), payload.get("limit", 100))
                require(
                    type(after) is int and 0 <= after <= 2**53 - 1,
                    "invalid audit cursor",
                )
                require(type(limit) is int and 1 <= limit <= 100, "invalid audit limit")
                params = {"after": after, "limit": limit}
            elif operation in ("get", "list"):
                require(not payload, "GET operation requires empty payload")
            path = "/" + domain + (f"/{operation}" if operation == "remove" else "")
            return await self.client._call(
                "GET" if operation in ("get", "list") else "POST",
                path,
                "workspace",
                domain,
                payload if operation not in ("get", "list") else None,
                params=params,
            )
        raise ServiceError(
            "UnsupportedOperation", "identity action is not supported", 501
        )


class Keys(Collection):
    """Revocable API keys; create/rotate return the token once."""

    kind: ResourceKind = "api-key"
    collection = "keys"

    async def create(
        self, *, name: str, permissions: list[str], expires_in: float | None = None
    ) -> JsonObject:
        """Creates only the permissions the server allows the caller to grant."""
        payload: JsonObject = {"name": name, "permissions": permissions}
        if expires_in is not None:
            payload["expires_in"] = expires_in
        return await self.client.identity.manage("keys.create", payload)

    async def revoke(self, key: str) -> JsonObject:
        """Revokes a key by public identifier, not by supplying its token."""
        return await self.client.identity.manage("keys.revoke", {"id": key})

    async def rotate(self, key: str) -> JsonObject:
        """Returns a replacement token once; does not persist it locally."""
        return await self.client.identity.manage("keys.rotate", {"id": key})


class Secrets(Collection):
    """Encrypted secret writes and metadata-only reads/listings."""

    kind: ResourceKind = "secret"
    collection = "secrets"

    async def create(self, *, name: str, value: str) -> JsonObject:
        """Sends secret material only in an authenticated TLS request body."""
        return await self.client.identity.manage(
            "secrets.create", {"name": name, "value": value}
        )

    async def rotate(
        self, *, name: str, value: str, expected_version: int
    ) -> JsonObject:
        """Uses an explicit version guard instead of blindly overwriting."""
        return await self.client.identity.manage(
            "secrets.rotate",
            {"name": name, "value": value, "expected_version": expected_version},
        )

    async def delete(self, secret: str) -> JsonObject:
        """Deletes by public name; it never returns the previous value."""
        return await self.client.identity.manage("secrets.delete", {"name": secret})


class Operators:
    """Project-safe doctor only. Full-store operations require host authority."""

    def __init__(self, client: AsyncClient) -> None:
        """Keeps operator authority separate from ordinary project credentials."""
        self.client = client

    async def doctor(self) -> JsonObject:
        """Returns redacted capability metadata, never host backup contents."""
        return await self.client._call("GET", "/operators/doctor", "operator", "doctor")

    async def backup(self, _spec: JsonObject) -> JsonObject:
        """Fails closed until an independent host authority exists."""
        raise ServiceError(
            "HostAuthorityUnavailable",
            "full-store backup is not a project operation",
            501,
        )

    async def restore_check(self, _manifest: JsonObject) -> JsonObject:
        """Fails closed; a project administrator is not a host administrator."""
        raise ServiceError(
            "HostAuthorityUnavailable",
            "full-store restore check requires independent host authority",
            501,
        )


@dataclass(frozen=True, eq=False)
class AsyncDeployment(BoundSnapshot):
    """Immutable deployment metadata with explicit environment operations."""

    environment: Environment | None = None

    @classmethod
    def from_response(
        cls,
        client,
        resource: str,
        value: Mapping[str, Any],
        environment: Environment | None = None,
    ):
        """Captures state without mutating older deployment handles."""
        return cls(
            Snapshot.from_dict(value)._encoded, client, name(resource), environment
        )

    @property
    def revision(self) -> int | None:
        """Returns the selected environment revision, never desired-as-serving state."""
        return self.environment.revision if self.environment else self.get("revision")

    @property
    def status(self) -> str | None:
        """Returns reported lifecycle status, preserving pending/error states."""
        return self.environment.status if self.environment else self.get("status")

    async def refresh(self) -> AsyncDeployment:
        """Fetches a new snapshot without modifying this handle."""
        return await self._client.deployments.get(
            self.name, env=self.environment.name if self.environment else None
        )

    async def predict(self, queries: Any, **options: Any) -> PredictionResult:
        """Uses the captured environment selection, not a mutable global default."""
        captured_environment(self.environment, options)
        return await self._client.deployments.predict(self.name, queries, **options)

    async def wait(self, **options: Any) -> Environment:
        """Waits for the captured pointer by default."""
        captured_environment(self.environment, options, pointer=True)
        return await self._client.deployments.wait(self.name, **options)

    async def promote(self, *, revision: int, **options: Any) -> Environment:
        """Promotes under this snapshot's generation when an environment was selected."""
        self._guard(options)
        return await self._client.deployments.promote(
            self.name, revision=revision, **options
        )

    async def rollback(self, *, revision: int, **options: Any) -> Environment:
        """Rolls back under this snapshot's generation."""
        self._guard(options)
        return await self._client.deployments.rollback(
            self.name, revision=revision, **options
        )

    def _guard(self, options):
        captured_environment(self.environment, options, guard=True)

    async def metrics(self, **options: Any) -> Snapshot:
        """Fetches actual metrics for the selected environment."""
        captured_environment(self.environment, options)
        return await self._client.deployments.metrics(self.name, **options)

    async def logs(self, **options: Any) -> Snapshot:
        """Fetches bounded runtime logs for the selected environment."""
        captured_environment(self.environment, options)
        return await self._client.deployments.logs(self.name, **options)

    async def delete(self) -> Snapshot:
        """Requests deletion; the server owns authorization and cleanup."""
        return await self._client.deployments.delete(self.name)


@dataclass(frozen=True, eq=False)
class AsyncDataset(BoundSnapshot):
    """Logical dataset snapshot and explicit append/label/import/refresh actions."""

    @property
    def sequence(self) -> int | None:
        """Returns the actual accepted transaction sequence."""
        return self.get("sequence")

    async def append(self, data: Any, *, id: str, idempotency_key: str) -> WriteReceipt:
        """Appends tabular observations without mutating this state snapshot."""
        return await self._client.datasets.append(
            self.name, data, id=id, idempotency_key=idempotency_key
        )

    async def label(
        self, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches labels using an explicit stable write identity."""
        return await self._client.datasets.label(
            self.name, data, id=id, column=column, idempotency_key=idempotency_key
        )

    async def import_source(self, reference) -> ImportReceipt:
        """Binds an import only when its logical dataset matches this handle."""
        require(reference.dataset == self.name, "import dataset differs from handle")
        return await self._client.datasets.import_source(reference)

    async def refresh(self) -> AsyncDataset:
        """Explicitly requests provider refresh rather than polling source silently."""
        return await self._client.datasets.refresh(self.name)

    async def status(self) -> AsyncDataset:
        """Fetches metadata without triggering provider ingestion."""
        return await self._client.datasets.get(self.name)


@dataclass(frozen=True, eq=False)
class AsyncJob(BoundSnapshot):
    """Durable job snapshot; acceptance never implies completion."""

    @property
    def state(self) -> str:
        """Returns the actual persisted lifecycle state."""
        return self["state"]

    async def poll(self) -> AsyncJob:
        """Fetches a fresh job state without modifying this object."""
        return await self._client.jobs.get(self.name)

    async def cancel(self) -> AsyncJob:
        """Cancels remotely; cancelling a local wait does not cancel the job."""
        return AsyncJob.from_response(
            self._client,
            self.name,
            await self._client.jobs.control(self.name, "cancel"),
        )

    async def wait(
        self, *, timeout: float = 120, poll_interval: float = 0.2
    ) -> AsyncJob:
        """Polls with a bounded total deadline and raises for terminal failure."""
        return await self._client.jobs.wait(
            self.name, timeout=timeout, poll_interval=poll_interval
        )

    async def results(self, *, offset: int = 0, limit: int = 100) -> Snapshot:
        """Gets one bounded page, retaining every skipped/rejected occurrence."""
        return Snapshot.from_dict(
            await self._client.jobs.results(self.name, offset=offset, limit=limit)
        )


@dataclass(frozen=True)
class AsyncDevelopment:
    """Remote development state, isolated from immutable revisions and production."""

    state: DevSession
    _client: Any = field(repr=False, compare=False)

    @property
    def id(self) -> str:
        """Returns the server-created remote session identifier."""
        return self.state.id

    @property
    def generation(self) -> int:
        """Returns the generation required for the next optimistic patch."""
        return self.state.generation

    async def refresh(self) -> AsyncDevelopment:
        """Fetches a new session snapshot."""
        return await self._client.dev_sessions.get(self.state.deployment, self.id)

    async def patch(self, package: Package) -> AsyncDevelopment:
        """Uploads a full replacement checked package using the captured generation."""
        return await self._client.dev_sessions.patch(self, package)

    async def predict(self, queries: Any, **options: Any) -> PredictionResult:
        """Scores only this development worker, never its production environment."""
        return await self._client.dev_sessions.predict(
            self.state.deployment, self.id, queries, **options
        )

    async def stop(self) -> AsyncDevelopment:
        """Stops this development worker without changing a production pointer."""
        return await self._client.dev_sessions.stop(self.state.deployment, self.id)
