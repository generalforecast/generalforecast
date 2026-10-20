"""Project-scoped versioned SDK. The server authorizes every operation."""

import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self, cast
from urllib.parse import urlencode

from generalforecast.contracts import (
    DatasetImportRef,
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
    Dataset,
    DatasetSpec,
    Deployment,
    Development,
    ImportReceipt,
    Job,
    JobSpec,
    RevisionDetails,
    Runtime,
    Snapshot,
    WriteReceipt,
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
from generalforecast.transport import Profile, Transport, connection

# Session-only onboarding and workspace-admin actions. The server checks the
# signed-in Google session and live workspace role on every call; project API
# keys receive its typed SessionRequired denial, never a local short-circuit.
WORKSPACE_ACTIONS: dict[str, tuple[str, str, str]] = {
    "onboarding.get": ("GET", "/onboarding", "onboarding"),
    "workspaces.list": ("GET", "/workspaces", "workspaces"),
    "workspaces.create": ("POST", "/workspaces", "new"),
    "workspaces.get": ("GET", "/workspaces/{workspace}", "{workspace}"),
    "workspaces.join": ("POST", "/workspaces/{workspace}/join", "{workspace}"),
    "workspaces.members": ("GET", "/workspaces/{workspace}/members", "{workspace}"),
    "invitations.preview": ("POST", "/invitations/preview", "invitation"),
    "invitations.accept": ("POST", "/invitations/accept", "invitation"),
    "invitations.list": ("GET", "/workspaces/{workspace}/invitations", "{workspace}"),
    "invitations.create": (
        "POST",
        "/workspaces/{workspace}/invitations",
        "{workspace}",
    ),
    "invitations.revoke": (
        "POST",
        "/workspaces/{workspace}/invitations/{id}/revoke",
        "{id}",
    ),
    "requests.list": ("GET", "/workspaces/{workspace}/join-requests", "{workspace}"),
    "requests.approve": (
        "POST",
        "/workspaces/{workspace}/join-requests/{id}/approve",
        "{id}",
    ),
    "requests.deny": (
        "POST",
        "/workspaces/{workspace}/join-requests/{id}/deny",
        "{id}",
    ),
    "requests.cancel": ("POST", "/join-requests/{id}/cancel", "{id}"),
    "domains.list": ("GET", "/workspaces/{workspace}/domains", "{workspace}"),
    "domains.create": ("POST", "/workspaces/{workspace}/domains", "{workspace}"),
    "domains.verify": ("POST", "/workspaces/{workspace}/domains/{id}/verify", "{id}"),
    "domains.configure": (
        "POST",
        "/workspaces/{workspace}/domains/{id}/policy",
        "{id}",
    ),
    "domains.revoke": ("POST", "/workspaces/{workspace}/domains/{id}/revoke", "{id}"),
}


def workspace_request(
    action: str, payload: JsonObject
) -> tuple[str, str, str, JsonObject | None, JsonObject | None]:
    """Binds one workspace action to (method, path, resource, body, query).

    Path identities are repeated in POST bodies so the server can check them;
    GET identities move into the path and only join requests accept `state`.
    """
    method, path, resource = WORKSPACE_ACTIONS[action]
    keys = [k for k in ("workspace", "id") if "{" + k + "}" in path]
    values = {k: name(payload.get(k)) for k in keys}
    path = path.format(**values)
    resource = resource.format(**values)
    if method == "POST":
        return method, path, resource, payload, None
    extra = set(payload) - set(keys)
    require(
        not extra or (action == "requests.list" and extra == {"state"}),
        "GET operation accepts only path identities",
    )
    params = None
    if "state" in extra:
        require(payload["state"] in ("pending", "all"), "invalid join request state")
        params = {"state": payload["state"]}
    return method, path, resource, None, params


class Client:
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
        self._transport = Transport(self.profile.url, token=key, **transport_options)
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

    def __enter__(self) -> Self:
        """Returns this synchronous client."""
        return self

    def __exit__(self, *_args) -> None:
        """Closes the client on scope exit."""
        self.close()

    def close(self) -> None:
        """Closes the transport; snapshots remain immutable and readable."""
        self._transport.close()

    def predict(
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
        payload = one_shot_payload(
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
            self._call("POST", "/predictions", "model", "predictions", payload)
        )

    def upload_snapshot(self, data: Any) -> JsonObject:
        """Uploads the full normalized Arrow snapshot and verifies the returned digest."""
        normalized = table(data)
        request_id = "r-" + uuid.uuid4().hex
        value = self._transport.send(
            "POST",
            f"/v1alpha1/projects/{self.project}/snapshots",
            arrow_bytes(normalized),
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
            "digest": table_digest(normalized),
            "rows": normalized.num_rows,
        }:
            raise ServiceError(
                "ProtocolError", "snapshot receipt does not match uploaded content", 502
            )
        return receipt

    def upload_python_source(
        self, deployment: str, archive: bytes, *, source: str
    ) -> Snapshot:
        """Uploads immutable Python source, not a client-qualified worker or image."""
        expected = python_source_upload(self.project, deployment, archive, source)
        request_id = "r-" + uuid.uuid4().hex
        value = self._transport.send(
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

    def _call(
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
        value = self._transport.send(
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

    def __init__(self, client: Client) -> None:
        """Uses the parent's authenticated, project-scoped transport."""
        self.client = client

    def list(self) -> list[JsonObject]:
        """Returns this project's actual resource metadata."""
        result = self.client._call(
            "GET", f"/{self.collection}", self.kind, self.collection
        )
        value = result.get(self.collection)
        if type(value) is not list or any(type(item) is not dict for item in value):
            raise ServiceError("ProtocolError", "invalid collection response", 502)
        return value

    def get(self, resource: str) -> Mapping[str, Any]:
        """Gets one project-qualified resource."""
        return self.client._call(
            "GET", f"/{self.collection}/{name(resource)}", self.kind, resource
        )

    def _post(self, resource: str, action: str, payload: JsonObject) -> JsonObject:
        suffix = f"/{name(resource)}" + (f"/{action}" if action else "")
        return self.client._call(
            "POST", f"/{self.collection}{suffix}", self.kind, resource, payload
        )

    def _create(self, resource: str, payload: JsonObject) -> JsonObject:
        return self.client._call(
            "POST", f"/{self.collection}", self.kind, resource, payload
        )


class Datasets(Collection):
    """Create-only observations and guarded component corrections."""

    kind: ResourceKind = "dataset"
    collection = "datasets"

    def create(self, config: JsonObject | DatasetSpec) -> "Dataset":
        """Creates a bounded immutable schema using the direct config payload."""
        value = config.to_dict() if isinstance(config, DatasetSpec) else config
        return Dataset.from_response(
            self.client, value["name"], self._create(value["name"], value)
        )

    def get(self, resource: str) -> "Dataset":
        """Returns a detached dataset snapshot with typed mutation methods."""
        return Dataset.from_response(self.client, resource, super().get(resource))

    def append(
        self, dataset: str, data: Any, *, id: str, idempotency_key: str
    ) -> "WriteReceipt":
        """Appends observations using the existing durable write-ID namespace."""
        return WriteReceipt.from_dict(
            self._post(
                dataset,
                "mutations",
                mutation(
                    data, id=id, operation="create_features", write_id=idempotency_key
                ),
            )
        )

    def label(
        self, dataset: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> "WriteReceipt":
        """Attaches outcome components independently of prediction requests."""
        return WriteReceipt.from_dict(
            self._post(
                dataset,
                "mutations",
                mutation(
                    data,
                    id=id,
                    column=column,
                    operation="attach_labels",
                    write_id=idempotency_key,
                ),
            )
        )

    def import_source(self, reference: DatasetImportRef) -> "ImportReceipt":
        """Binds a cloud or uploaded table to an existing logical dataset."""
        return ImportReceipt.from_dict(
            self._post(reference.dataset, "imports", reference.to_dict())
        )

    def import_file(self, dataset: str, data: Any) -> "ImportReceipt":
        """Uploads a local bounded table before binding it to a logical dataset."""
        receipt = self.client.upload_snapshot(data)
        return self.import_source(
            DatasetImportRef(
                dataset,
                "upload://" + receipt["digest"],
                "arrow",
                digest=receipt["digest"],
            )
        )

    def refresh(self, dataset: str) -> "Dataset":
        """Requests an explicit provider refresh; unsupported adapters fail visibly."""
        return Dataset.from_response(
            self.client, dataset, self._post(dataset, "refresh", {})
        )

    def mutate(self, dataset: str, mutation: Mutation) -> AcceptedWrite:
        """Accepts an atomic write, not serving readiness. Preserve write_id."""
        return cast(AcceptedWrite, self._post(dataset, "mutations", dict(mutation)))

    def snapshot(
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
            self._post(
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

    def pin(self, spec: JsonObject) -> SourcePin:
        """Pins server-relative bytes; this does not upload a client-side path."""
        return cast(SourcePin, self._create("new", spec))


class Models(Collection):
    """Curated model metadata and pure caller-context inference."""

    kind: ResourceKind = "model"
    collection = "models"

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Never ingests queries; returns genuine outputs or a typed gate."""
        return cast(
            InferenceResult,
            self._post(resource_name(request["model"]), "infer", dict(request)),
        )


class Jobs(Collection):
    """Durable jobs with linked retries and paginated per-occurrence reports."""

    kind: ResourceKind = "job"
    collection = "jobs"

    def submit(self, spec: JsonObject | JobSpec) -> "Job":
        """Preserves submission_id; acceptance is not job completion."""
        value = spec.to_dict() if isinstance(spec, JobSpec) else spec
        return Job.from_response(
            self.client, value["name"], self._create(value["name"], value)
        )

    def get(self, resource: str) -> "Job":
        """Returns a durable job snapshot; polling creates new snapshots."""
        return Job.from_response(self.client, resource, super().get(resource))

    def wait(
        self, job: str, *, timeout: float = 120, poll_interval: float = 0.2
    ) -> Job:
        """Polls a durable job with one deadline; failures are never successful results."""
        number(timeout, 0.001, 86400, "timeout")
        number(poll_interval, 0.001, 30, "poll_interval")
        deadline = time.monotonic() + timeout
        while True:
            state = Job.from_response(
                self.client,
                job,
                self.client._call(
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
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def control(
        self, job: str, action: Literal["pause", "resume", "cancel", "retry"]
    ) -> JsonObject:
        """Retry links a NEW job; it does not rewrite a completed report."""
        require(action in ("pause", "resume", "cancel", "retry"), "invalid job action")
        return self._post(job, action, {})

    def results(self, job: str, *, offset: int = 0, limit: int = 100) -> JsonObject:
        """Pages every occurrence, including skipped and rejected records."""
        require(type(offset) is int and 0 <= offset <= 2**53 - 1, "invalid offset")
        require(type(limit) is int and 1 <= limit <= 100, "invalid limit")
        return self.client._call(
            "GET",
            f"/jobs/{name(job)}/results",
            self.kind,
            job,
            params={"offset": offset, "limit": limit},
        )

    def iter_results(self, job: str) -> Iterator[JsonObject]:
        """Iterates all report pages, rejecting a non-progressing cursor."""
        offset = 0
        while True:
            page = self.results(job, offset=offset)
            rows, next_offset = page.get("results"), page.get("next_offset")
            if type(rows) is not list or any(type(row) is not dict for row in rows):
                raise ServiceError("ProtocolError", "invalid job results", 502)
            yield from rows
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
    def load(path: str | Path = ".") -> Package:
        """Loads a declarative package without network access."""
        return Package.load(path)

    @staticmethod
    def check(package: Package, **options: Any) -> Package:
        """Returns a new checked package using the canonical compiler."""
        return package.check(**options)

    def inspect(self, manifest: JsonObject) -> JsonObject:
        """Validates metadata without executing untrusted code."""
        return self.client._call(
            "POST", "/packages/inspect", self.kind, "new", manifest
        )

    def register(self, manifest: JsonObject) -> JsonObject:
        """Registers a digest-verified manifest, not a running deployment."""
        return self._create(manifest["name"], manifest)

    def test(self, package: str) -> JsonObject:
        """Preserves typed IsolationUnavailable when the boundary is absent."""
        return self._post(package, "test", {})


class Deployments(Collection):
    """Immutable package revisions and runtime-bound environment operations."""

    kind: ResourceKind = "deployment"
    collection = "deployments"

    def get(self, resource: str, *, env: str | None = None) -> "Deployment":
        """Returns immutable deployment metadata and optional environment state."""
        state = super().get(resource)
        environment = self.environment(resource, env) if env is not None else None
        return Deployment.from_response(self.client, resource, state, environment)

    def create(self, deployment: str) -> "Deployment":
        """Creates only the deployment identity, not serving readiness."""
        return Deployment.from_response(
            self.client,
            deployment,
            self._create(deployment, {"name": name(deployment)}),
        )

    def revisions(self, deployment: str) -> tuple[Revision, ...]:
        """Lists immutable revisions without inferring environment readiness."""
        value = self.client._call(
            "GET", f"/deployments/{name(deployment)}/revisions", self.kind, deployment
        )
        return tuple(Revision.from_dict(row) for row in value["revisions"])

    def revision(self, deployment: str, number: int) -> "RevisionDetails":
        """Reads the exact immutable compiled lock for a revision."""
        require(type(number) is int and number > 0, "invalid revision")
        value = self.client._call(
            "GET",
            f"/deployments/{name(deployment)}/revisions/{number}",
            self.kind,
            deployment,
        )
        return RevisionDetails(
            Revision.from_dict(value["revision"]), PackageLock.from_dict(value["lock"])
        )

    def environments(self, deployment: str) -> tuple[Environment, ...]:
        """Lists runtime-bound environment pointers and their actual states."""
        value = self.client._call(
            "GET",
            f"/deployments/{name(deployment)}/environments",
            self.kind,
            deployment,
        )
        return tuple(Environment.from_dict(row) for row in value["environments"])

    def environment(
        self, deployment: str, env: str | None = None, *, deadline: float | None = None
    ) -> Environment:
        """Gets one environment with revision and CAS generation."""
        selected = environment_name(env, self.client.profile.environment)
        return Environment.from_dict(
            self.client._call(
                "GET",
                f"/deployments/{name(deployment)}/environments/{selected}",
                self.kind,
                deployment,
                deadline=deadline,
            )
        )

    def create_environment(
        self, deployment: str, *, env: str, runtime: str
    ) -> Environment:
        """Binds a runtime separately from package content and authentication."""
        return Environment.from_dict(
            self._post(
                deployment,
                "environments",
                {"name": name(env), "runtime": name(runtime)},
            )
        )

    def promote(
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
            self.environment(deployment, selected).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            self._post(
                deployment,
                f"environments/{name(selected)}/promote",
                PromoteRequest(revision, generation).to_dict(),
            )
        )

    def rollback(
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
            self.environment(deployment, selected).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            self._post(
                deployment,
                f"environments/{name(selected)}/rollback",
                PromoteRequest(revision, generation).to_dict(),
            )
        )

    def predict(
        self, deployment: str, queries: Any, *, env: str | None = None, **options: Any
    ) -> PredictionResult:
        """Predicts through the runtime-bound environment with actual context metadata."""
        return inference_result(
            self._post(
                deployment,
                f"environments/{environment_name(env, self.client.profile.environment)}/predict",
                prediction_payload(queries, **options),
            )
        )

    def push(
        self,
        package: Package,
        *,
        env: str | None = None,
        runtime: str | None = None,
        wait: bool = False,
        timeout: float = 120,
    ) -> Revision:
        """Uploads a checked package, registers an immutable revision and CAS-promotes."""
        lock = checked_lock(package)
        deployment, selected = (
            lock.spec.name,
            environment_name(env, self.client.profile.environment),
        )
        try:
            self.get(deployment)
        except ServiceError as error:
            if error.status != 404:
                raise
            self.create(deployment)
        snapshot = None
        if isinstance(lock, PythonLock):
            self.client.upload_python_source(
                deployment,
                checked_python_source(package, lock),
                source=lock.python.source,
            )
        elif lock.spec.context.mode == "snapshot":
            snapshot = self.client.upload_snapshot(package.snapshot_table())["digest"]
            require(
                snapshot == lock.data_digest, "snapshot differs from checked package"
            )
        revision = Revision.from_dict(
            self._post(
                deployment, "revisions", {"lock": lock.to_dict(), "snapshot": snapshot}
            )
        )
        try:
            environment = self.environment(deployment, selected)
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
            environment = self.create_environment(
                deployment, env=selected, runtime=chosen
            )
        promoted = self.promote(
            deployment,
            revision=revision.number,
            env=selected,
            expected_generation=environment.generation,
        )
        if wait:
            self.wait(
                deployment,
                env=selected,
                revision=revision.number,
                generation=promoted.generation,
                timeout=timeout,
            )
        return revision

    def wait(
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
            state = self.environment(deployment, env, deadline=deadline)
            if (revision is not None and state.revision != revision) or (
                generation is not None and state.generation != generation
            ):
                raise ServiceError(
                    "Conflict", "environment pointer changed while waiting", 409
                )
            revision, generation = state.revision, state.generation
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
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def metrics(self, deployment: str, *, env: str | None = None) -> Snapshot:
        """Gets reported metrics without fabricating missing samples."""
        return Snapshot.from_dict(
            self.client._call(
                "GET",
                f"/deployments/{name(deployment)}/environments/{environment_name(env, self.client.profile.environment)}/metrics",
                self.kind,
                deployment,
            )
        )

    def logs(self, deployment: str, *, env: str | None = None) -> Snapshot:
        """Gets bounded safe runtime logs."""
        return Snapshot.from_dict(
            self.client._call(
                "GET",
                f"/deployments/{name(deployment)}/environments/{environment_name(env, self.client.profile.environment)}/logs",
                self.kind,
                deployment,
            )
        )

    def delete(self, deployment: str) -> Snapshot:
        """Requests explicit stop/delete, never treating unsupported cleanup as success."""
        return Snapshot.from_dict(self._post(deployment, "delete", {}))


class DevSessions:
    """Remote dev-only worker sessions with full compiled replacement patches."""

    def __init__(self, client: Client) -> None:
        """Uses the same authenticated scoped transport as deployment operations."""
        self.client = client

    def _call(
        self, deployment, session, action="", payload=None, method="POST"
    ) -> JsonObject:
        path = f"/deployments/{name(deployment)}/dev-sessions"
        if session is not None:
            path += "/" + name(session)
        if action:
            path += "/" + action
        return self.client._call(method, path, "deployment", deployment, payload)

    def start(
        self,
        deployment: str,
        *,
        base_revision: int,
        runtime: str,
        ttl_seconds: int = 3600,
    ) -> Development:
        """Starts from an immutable base without selecting any production environment."""
        from generalforecast.contracts import DevSession

        value = self._call(
            deployment,
            None,
            payload={
                "base_revision": base_revision,
                "runtime": name(runtime),
                "ttl_seconds": ttl_seconds,
            },
        )
        return Development(DevSession.from_dict(value), self.client)

    def get(self, deployment: str, session: str) -> Development:
        """Reads actual development lifecycle state."""
        from generalforecast.contracts import DevSession

        return Development(
            DevSession.from_dict(self._call(deployment, session, method="GET")),
            self.client,
        )

    def patch(self, development: Development, package: Package) -> Development:
        """CAS-patches a dev worker; stale handles fail without automatic rebase."""
        from generalforecast.contracts import DevSession

        require(
            development._client is self.client,
            "development session belongs to another client",
        )
        lock = checked_lock(package)
        require(
            lock.spec.name == development.state.deployment,
            "patch deployment differs from development session",
        )
        if isinstance(lock, PythonLock):
            self.client.upload_python_source(
                development.state.deployment,
                checked_python_source(package, lock),
                source=lock.python.source,
            )
        snapshot = (
            self.client.upload_snapshot(package.snapshot_table())["digest"]
            if lock.spec.context.mode == "snapshot"
            else None
        )
        request = PatchRequest(
            development.generation, development.state.base_revision, lock, snapshot
        )
        return Development(
            DevSession.from_dict(
                self._call(
                    development.state.deployment,
                    development.id,
                    "patch",
                    request.to_dict(),
                )
            ),
            self.client,
        )

    def predict(
        self, deployment: str, session: str, queries: Any, **options: Any
    ) -> PredictionResult:
        """Runs the isolated remote development package, not production."""
        return inference_result(
            self._call(
                deployment, session, "predict", prediction_payload(queries, **options)
            )
        )

    def stop(self, deployment: str, session: str) -> Development:
        """Stops only the named remote development session."""
        from generalforecast.contracts import DevSession

        return Development(
            DevSession.from_dict(self._call(deployment, session, "stop", {})),
            self.client,
        )


class Runtimes:
    """Authenticated runtime inventory; provisioning is never inferred from a plan."""

    def __init__(self, client: Client) -> None:
        """Shares project scope and transport, not provider credentials in config."""
        self.client = client

    def list(self) -> tuple[Runtime, ...]:
        """Lists configured runtime metadata."""
        value = self.client._call("GET", "/runtimes", "operator", "runtimes")
        return tuple(
            Runtime.from_response(self.client, row["name"], row)
            for row in value["runtimes"]
        )

    def get(self, runtime: str) -> Runtime:
        """Gets actual runtime state or a typed unsupported/provider error."""
        return Runtime.from_response(
            self.client,
            runtime,
            self.client._call("GET", f"/runtimes/{name(runtime)}", "operator", runtime),
        )

    def preflight(self, runtime: str) -> Snapshot:
        """Runs explicit provider preflight separately from authentication."""
        return Snapshot.from_dict(
            self.client._call(
                "POST", f"/runtimes/{name(runtime)}/preflight", "operator", runtime, {}
            )
        )

    def connect(self, runtime: str) -> Snapshot:
        """Explicitly connects a configured provider, never changing authentication."""
        return self._action(runtime, "connect")

    def cleanup(self, runtime: str) -> Snapshot:
        """Cleans up only the named configured runtime through provider policy."""
        return self._action(runtime, "cleanup")

    def _action(self, runtime: str, action: str) -> Snapshot:
        return Snapshot.from_dict(
            self.client._call(
                "POST", f"/runtimes/{name(runtime)}/{action}", "operator", runtime, {}
            )
        )


class Identity:
    """Allowlisted management routes; authentication never implies membership."""

    def __init__(self, client: Client) -> None:
        """Uses one scoped client; Google cookie login belongs to the console."""
        self.client = client

    def manage(self, action: str, payload: JsonObject | None = None) -> JsonObject:
        """Calls concrete identity actions, not a user-selected server method."""
        payload = {} if payload is None else json_object(payload)
        if action in WORKSPACE_ACTIONS:
            method, path, resource, body, params = workspace_request(action, payload)
            return self.client._call(
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
            return self.client._call(
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
                else (
                    "new"
                    if domain == "keys" and operation == "create"
                    else payload.get("id" if domain == "keys" else "name", "")
                )
            )
            path = f"/{domain}"
            if operation not in ("list", "create"):
                path += f"/{name(resource)}" + (
                    "" if operation == "get" else f"/{operation}"
                )
            return self.client._call(
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
                after, limit = payload.get("after", 0), payload.get("limit", 100)
                require(
                    type(after) is int and 0 <= after <= 2**53 - 1,
                    "invalid audit cursor",
                )
                require(type(limit) is int and 1 <= limit <= 100, "invalid audit limit")
                params = {"after": after, "limit": limit}
            elif operation in ("get", "list"):
                require(not payload, "GET operation requires empty payload")
            path = "/" + domain + (f"/{operation}" if operation == "remove" else "")
            return self.client._call(
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

    def create(
        self, *, name: str, permissions: list[str], expires_in: float | None = None
    ) -> JsonObject:
        """Creates only the permissions the server allows the caller to grant."""
        payload: JsonObject = {"name": name, "permissions": permissions}
        if expires_in is not None:
            payload["expires_in"] = expires_in
        return self.client.identity.manage("keys.create", payload)

    def revoke(self, key: str) -> JsonObject:
        """Revokes a key by public identifier, not by supplying its token."""
        return self.client.identity.manage("keys.revoke", {"id": key})

    def rotate(self, key: str) -> JsonObject:
        """Returns a replacement token once; does not persist it locally."""
        return self.client.identity.manage("keys.rotate", {"id": key})


class Secrets(Collection):
    """Encrypted secret writes and metadata-only reads/listings."""

    kind: ResourceKind = "secret"
    collection = "secrets"

    def create(self, *, name: str, value: str) -> JsonObject:
        """Sends secret material only in an authenticated TLS request body."""
        return self.client.identity.manage(
            "secrets.create", {"name": name, "value": value}
        )

    def rotate(self, *, name: str, value: str, expected_version: int) -> JsonObject:
        """Uses an explicit version guard instead of blindly overwriting."""
        return self.client.identity.manage(
            "secrets.rotate",
            {"name": name, "value": value, "expected_version": expected_version},
        )

    def delete(self, secret: str) -> JsonObject:
        """Deletes by public name; it never returns the previous value."""
        return self.client.identity.manage("secrets.delete", {"name": secret})


class Operators:
    """Project-safe doctor only. Full-store operations require host authority."""

    def __init__(self, client: Client) -> None:
        """Keeps operator authority separate from ordinary project credentials."""
        self.client = client

    def doctor(self) -> JsonObject:
        """Returns redacted capability metadata, never host backup contents."""
        return self.client._call("GET", "/operators/doctor", "operator", "doctor")

    def backup(self, _spec: JsonObject) -> JsonObject:
        """Fails closed until an independent host authority exists."""
        raise ServiceError(
            "HostAuthorityUnavailable",
            "full-store backup is not a project operation",
            501,
        )

    def restore_check(self, _manifest: JsonObject) -> JsonObject:
        """Fails closed; a project administrator is not a host administrator."""
        raise ServiceError(
            "HostAuthorityUnavailable",
            "full-store restore check requires independent host authority",
            501,
        )
