# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native asynchronous SDK with the same scoped resources and immutable values."""

from __future__ import annotations

import asyncio
import builtins
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast
from urllib.parse import urlencode

from generalforecast.client import WORKSPACE_ACTIONS, workspace_request
from generalforecast.contracts import (
    ContextImportRef,
    Environment,
    PredictionResult,
    PromoteRequest,
    Revision,
    parse_lock,
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
    Mutation,
    SourcePin,
)
from generalforecast.deployments.sdk import (
    BoundSnapshot,
    ContextSpec,
    ImportReceipt,
    RevisionDetails,
    Snapshot,
    WriteReceipt,
    append_payload,
    captured_environment,
    environment_name,
    inference_result,
    label_payload,
    pushed,
    table,
    unwrap,
)
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
        base_url: str | None = None,
        **transport_options: Any,
    ) -> None:
        """Resolves scoped GF_API_KEY/profile auth without persisting credentials.

        base_url targets one endpoint, such as a local `gf serve`, with no
        credential unless api_key is given; profiles and GF_* are not read.
        """
        self.profile, key = connection(
            url, project, token, api_key, profile, config_path, base_url
        )
        self.project = self.profile.project
        self._transport = AsyncTransport(
            self.profile.url, token=key, **transport_options
        )
        self.contexts = Contexts(self)
        self.sources = Sources(self)
        self.models = Models(self)
        self.predictors = Predictors(self)
        self.identity = Identity(self)
        self.keys = Keys(self)
        self.secrets = Secrets(self)

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
        model: str,
        target: str,
        context: Any,
        queries: Any,
        *,
        inputs: Sequence[str] | None = None,
        id: str | None = None,
        time: str | None = None,
        series: Sequence[str] | None = None,
        frequency: Any = None,
        horizon: int | None = None,
        quantiles: Sequence[float] | None = None,
        outputs: Sequence[str] | None = None,
        selection: JsonObject | None = None,
    ) -> PredictionResult:
        """Predicts with an ad hoc predictor; see `Client.predict`."""
        from generalforecast.predictor import PredictRequest

        request = await asyncio.to_thread(
            PredictRequest.build,
            model=model,
            target=target,
            context=context,
            queries=queries,
            inputs=inputs,
            id=id,
            time=time,
            series=series,
            frequency=frequency,
            horizon=horizon,
            quantiles=quantiles,
            outputs=outputs,
            selection=selection,
        )
        return inference_result(
            await self._call(
                "POST", "/predict", "predictor", "ad-hoc", request.payload()
            )
        )

    async def upload_snapshot(self, data: Any) -> JsonObject:
        """Uploads the full normalized Arrow snapshot and verifies the returned digest."""
        normalized = await asyncio.to_thread(table, data)
        encoded = await asyncio.to_thread(arrow_bytes, normalized)
        digest = await asyncio.to_thread(table_digest, normalized)
        request_id = "r-" + uuid.uuid4().hex
        value = await self._transport.send(
            "POST",
            f"/v1/projects/{self.project}/snapshots",
            encoded,
            {
                "Content-Type": "application/vnd.apache.arrow.stream",
                "X-GF-Project": self.project,
                "X-GF-Request-ID": request_id,
            },
        )
        request = Request(
            Scope(self.project, request_id), Resource("context", "snapshots"), {}
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
        prefix = "/v1" if global_route else f"/v1/projects/{self.project}"
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


class Contexts(Collection):
    """Create-only observations and guarded component corrections."""

    kind: ResourceKind = "context"
    collection = "contexts"

    async def create(self, config: JsonObject | ContextSpec) -> AsyncLiveContext:
        """Creates a live context from its schema."""
        value = config.to_dict() if isinstance(config, ContextSpec) else config
        return AsyncLiveContext.from_response(
            self.client, value["name"], await self._create(value["name"], value)
        )

    async def get(self, resource: str) -> AsyncLiveContext:
        """Returns a detached live-context snapshot with typed write methods."""
        return AsyncLiveContext.from_response(
            self.client, resource, await super().get(resource)
        )

    async def append(
        self, context: str, data: Any, *, id: str, idempotency_key: str
    ) -> WriteReceipt:
        """Appends rows under a durable write id."""
        return WriteReceipt.from_dict(
            await self._post(
                context,
                "append",
                await asyncio.to_thread(
                    append_payload, data, id=id, write_id=idempotency_key
                ),
            )
        )

    async def label(
        self, context: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches one target's values to rows already in the context."""
        return WriteReceipt.from_dict(
            await self._post(
                context,
                "label",
                await asyncio.to_thread(
                    label_payload, data, id=id, column=column, write_id=idempotency_key
                ),
            )
        )

    async def import_source(self, reference: ContextImportRef) -> ImportReceipt:
        """Binds a cloud or uploaded table to an existing live context."""
        return ImportReceipt.from_dict(
            await self._post(reference.context, "imports", reference.to_dict())
        )

    async def import_file(self, context: str, data: Any) -> ImportReceipt:
        """Uploads a local bounded table before binding it to a live context."""
        receipt = await self.client.upload_snapshot(data)
        return await self.import_source(
            ContextImportRef(
                context,
                "upload://" + receipt["digest"],
                "arrow",
                digest=receipt["digest"],
            )
        )

    async def refresh(self, context: str) -> AsyncLiveContext:
        """Requests an explicit provider refresh; unsupported adapters fail visibly."""
        return AsyncLiveContext.from_response(
            self.client, context, await self._post(context, "refresh", {})
        )

    async def mutate(self, context: str, mutation: Mutation) -> AcceptedWrite:
        """Accepts an atomic write, not serving readiness. Preserve write_id."""
        return cast(
            AcceptedWrite, await self._post(context, "mutations", dict(mutation))
        )

    async def snapshot(
        self,
        context: str,
        *,
        target: str,
        knowledge_at: float | None = None,
        event_before: float | None = None,
    ) -> ContextSnapshot:
        """Keeps knowledge/acceptance time independent from event time."""
        return cast(
            ContextSnapshot,
            await self._post(
                context,
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


class Predictors(Collection):
    """Hosted predictors: immutable revisions and runtime-bound environments."""

    kind: ResourceKind = "predictor"
    collection = "predictors"

    async def get(
        self, resource: str, *, env: str | None = None
    ) -> AsyncHostedPredictor:
        """Returns the predictor's facts and optional environment state."""
        state = await super().get(resource)
        environment = await self.environment(resource, env) if env is not None else None
        return AsyncHostedPredictor.from_response(
            self.client, resource, state, environment
        )

    async def create(self, predictor: str) -> AsyncHostedPredictor:
        """Creates only the predictor name, not serving readiness."""
        return AsyncHostedPredictor.from_response(
            self.client,
            predictor,
            await self._create(predictor, {"name": name(predictor)}),
        )

    async def revisions(self, predictor: str) -> tuple[Revision, ...]:
        """Lists immutable revisions without inferring environment readiness."""
        value = await self.client._call(
            "GET", f"/predictors/{name(predictor)}/revisions", self.kind, predictor
        )
        return tuple(Revision.from_dict(row) for row in value["revisions"])

    async def revision(self, predictor: str, number: int) -> RevisionDetails:
        """Reads the exact immutable compiled lock for a revision."""
        require(type(number) is int and number > 0, "invalid revision")
        value = await self.client._call(
            "GET",
            f"/predictors/{name(predictor)}/revisions/{number}",
            self.kind,
            predictor,
        )
        return RevisionDetails(
            Revision.from_dict(value["revision"]), parse_lock(value["lock"])
        )

    async def environments(self, predictor: str) -> tuple[Environment, ...]:
        """Lists runtime-bound environment pointers and their actual states."""
        value = await self.client._call(
            "GET",
            f"/predictors/{name(predictor)}/environments",
            self.kind,
            predictor,
        )
        return tuple(Environment.from_dict(row) for row in value["environments"])

    async def environment(
        self, predictor: str, env: str | None = None, *, deadline: float | None = None
    ) -> Environment:
        """Gets one environment with revision and CAS generation."""
        selected = environment_name(env, self.client.profile.environment)
        return Environment.from_dict(
            await self.client._call(
                "GET",
                f"/predictors/{name(predictor)}/environments/{selected}",
                self.kind,
                predictor,
                deadline=deadline,
            )
        )

    async def create_environment(
        self, predictor: str, *, env: str, runtime: str
    ) -> Environment:
        """Binds a runtime separately from predictor content and authentication."""
        return Environment.from_dict(
            await self._post(
                predictor,
                "environments",
                {"name": name(env), "runtime": name(runtime)},
            )
        )

    async def promote(
        self,
        predictor: str,
        *,
        revision: int,
        env: str | None = None,
        expected_generation: int | None = None,
    ) -> Environment:
        """CAS-promotes an explicit revision; races remain visible conflicts."""
        selected = environment_name(env, self.client.profile.environment)
        generation = (
            (await self.environment(predictor, selected)).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            await self._post(
                predictor,
                f"environments/{name(selected)}/promote",
                PromoteRequest(revision, generation).to_dict(),
            )
        )

    async def rollback(
        self,
        predictor: str,
        *,
        revision: int,
        env: str | None = None,
        expected_generation: int | None = None,
    ) -> Environment:
        """Rolls back to an explicit immutable revision under the same CAS guard."""
        selected = environment_name(env, self.client.profile.environment)
        generation = (
            (await self.environment(predictor, selected)).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            await self._post(
                predictor,
                f"environments/{name(selected)}/rollback",
                PromoteRequest(revision, generation).to_dict(),
            )
        )

    async def predict(
        self,
        predictor: str,
        queries: Any,
        *,
        context: Any = None,
        as_of: Any = None,
        env: str | None = None,
    ) -> PredictionResult:
        """Predicts with a declared predictor; see `Predictors.predict`."""
        from generalforecast.predictor import PredictRequest

        payload = await asyncio.to_thread(
            PredictRequest.declared_payload, queries, context=context, as_of=as_of
        )
        params = (
            {"env": environment_name(env, "production")} if env is not None else None
        )
        return inference_result(
            await self.client._call(
                "POST",
                f"/{self.collection}/{predictor}/predict",
                "predictor",
                predictor,
                payload,
                params=params,
            )
        )

    async def push(
        self,
        predictor: str | Path,
        *,
        env: str | None = None,
        runtime: str | None = None,
        wait: bool = False,
        timeout: float = 120,
    ) -> Revision:
        """Pushes a checked predictor: registers a revision and points an environment at it."""
        checked = await asyncio.to_thread(pushed, predictor)
        lock = checked.lock
        predictor, selected = (
            checked.name,
            environment_name(env, self.client.profile.environment),
        )
        try:
            await self.get(predictor)
        except ServiceError as error:
            if error.status != 404:
                raise
            await self.create(predictor)
        snapshot = None
        if checked.snapshot is not None:
            snapshot = (await self.client.upload_snapshot(checked.snapshot))["digest"]
            require(snapshot == lock.data_digest, "snapshot differs from gf.lock")
        revision = Revision.from_dict(
            await self._post(
                predictor, "revisions", {"lock": lock.to_dict(), "snapshot": snapshot}
            )
        )
        try:
            environment = await self.environment(predictor, selected)
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
                predictor, env=selected, runtime=chosen
            )
        promoted = await self.promote(
            predictor,
            revision=revision.number,
            env=selected,
            expected_generation=environment.generation,
        )
        if wait:
            await self.wait(
                predictor,
                env=selected,
                revision=revision.number,
                generation=promoted.generation,
                timeout=timeout,
            )
        return revision

    async def wait(
        self,
        predictor: str,
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
            state = await self.environment(predictor, env, deadline=deadline)
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
                    "EnvironmentFailed", "environment did not become ready", 409
                )
            if time.monotonic() >= deadline:
                raise ServiceError(
                    "WaitTimeout", "environment readiness timed out", 504
                )
            await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    async def metrics(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets reported metrics without fabricating missing samples."""
        return Snapshot.from_dict(
            await self.client._call(
                "GET",
                f"/predictors/{name(predictor)}/environments/{environment_name(env, self.client.profile.environment)}/metrics",
                self.kind,
                predictor,
            )
        )

    async def logs(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets bounded safe runtime logs."""
        return Snapshot.from_dict(
            await self.client._call(
                "GET",
                f"/predictors/{name(predictor)}/environments/{environment_name(env, self.client.profile.environment)}/logs",
                self.kind,
                predictor,
            )
        )

    async def delete(self, predictor: str) -> Snapshot:
        """Requests explicit stop/delete, never treating unsupported cleanup as success."""
        return Snapshot.from_dict(await self._post(predictor, "delete", {}))


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
            "auth.logout": ("workspace", "/auth/logout", "auth"),
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
            post = action.endswith("create") or action == "auth.logout"
            require(
                action.endswith("create") or not payload,
                "GET operation requires empty payload",
            )
            kind, path, resource = globals_[action]
            return await self.client._call(
                "POST" if post else "GET",
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


@dataclass(frozen=True, eq=False)
class AsyncHostedPredictor(BoundSnapshot):
    """Immutable predictor metadata with explicit environment operations."""

    environment: Environment | None = None

    @classmethod
    def from_response(
        cls,
        client,
        resource: str,
        value: Mapping[str, Any],
        environment: Environment | None = None,
    ):
        """Captures state without mutating older predictor handles."""
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

    async def refresh(self) -> AsyncHostedPredictor:
        """Fetches a new snapshot without modifying this handle."""
        return await self._client.predictors.get(
            self.name, env=self.environment.name if self.environment else None
        )

    async def predict(self, queries: Any, **options: Any) -> PredictionResult:
        """Uses the captured environment selection, not a mutable global default."""
        captured_environment(self.environment, options)
        return await self._client.predictors.predict(self.name, queries, **options)

    async def wait(self, **options: Any) -> Environment:
        """Waits for the captured pointer by default."""
        captured_environment(self.environment, options, pointer=True)
        return await self._client.predictors.wait(self.name, **options)

    async def promote(self, *, revision: int, **options: Any) -> Environment:
        """Promotes under this snapshot's generation when an environment was selected."""
        self._guard(options)
        return await self._client.predictors.promote(
            self.name, revision=revision, **options
        )

    async def rollback(self, *, revision: int, **options: Any) -> Environment:
        """Rolls back under this snapshot's generation."""
        self._guard(options)
        return await self._client.predictors.rollback(
            self.name, revision=revision, **options
        )

    def _guard(self, options):
        captured_environment(self.environment, options, guard=True)

    async def metrics(self, **options: Any) -> Snapshot:
        """Fetches actual metrics for the selected environment."""
        captured_environment(self.environment, options)
        return await self._client.predictors.metrics(self.name, **options)

    async def logs(self, **options: Any) -> Snapshot:
        """Fetches bounded runtime logs for the selected environment."""
        captured_environment(self.environment, options)
        return await self._client.predictors.logs(self.name, **options)

    async def delete(self) -> Snapshot:
        """Requests deletion; the server owns authorization and cleanup."""
        return await self._client.predictors.delete(self.name)


@dataclass(frozen=True, eq=False)
class AsyncLiveContext(BoundSnapshot):
    """Logical context snapshot and explicit append/label/import/refresh actions."""

    @property
    def sequence(self) -> int | None:
        """Returns the actual accepted transaction sequence."""
        return self.get("sequence")

    async def append(self, data: Any, *, id: str, idempotency_key: str) -> WriteReceipt:
        """Appends tabular observations without mutating this state snapshot."""
        return await self._client.contexts.append(
            self.name, data, id=id, idempotency_key=idempotency_key
        )

    async def label(
        self, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches labels using an explicit stable write identity."""
        return await self._client.contexts.label(
            self.name, data, id=id, column=column, idempotency_key=idempotency_key
        )

    async def import_source(self, reference) -> ImportReceipt:
        """Binds an import only when its logical context matches this handle."""
        require(reference.context == self.name, "import context differs from handle")
        return await self._client.contexts.import_source(reference)

    async def refresh(self) -> AsyncLiveContext:
        """Explicitly requests provider refresh rather than polling source silently."""
        return await self._client.contexts.refresh(self.name)

    async def status(self) -> AsyncLiveContext:
        """Fetches metadata without triggering provider ingestion."""
        return await self._client.contexts.get(self.name)
