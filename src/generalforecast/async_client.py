# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native asynchronous SDK: the same operations as `Client`, awaited.

Every operation is defined once, in `generalforecast._operations`. This
client awaits its requests on the event loop and runs its CPU-bound steps,
such as table conversion, in worker threads.
"""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from generalforecast import _operations as ops
from generalforecast._operations import Call, Operation, Pause, Work
from generalforecast.contracts import (
    ContextImportRef,
    Environment,
    PredictionResult,
    Revision,
)
from generalforecast.service.boundary import JsonObject, ResourceKind
from generalforecast.service.interfaces import (
    AcceptedWrite,
    ContextSnapshot,
    Mutation,
    SourcePin,
)
from generalforecast.service.sdk import (
    ContextHandle,
    ContextSpec,
    ImportReceipt,
    PredictorHandle,
    RevisionDetails,
    Snapshot,
    WriteReceipt,
)
from generalforecast.transport import AsyncTransport, Profile, connection

__all__ = ["AsyncClient", "AsyncHostedPredictor", "AsyncLiveContext"]


@dataclass(frozen=True, eq=False)
class AsyncHostedPredictor(PredictorHandle):
    """A hosted predictor's facts with explicit environment operations, awaited."""

    async def refresh(self) -> AsyncHostedPredictor:
        """Fetches a new snapshot without modifying this handle."""
        return await self._client.predictors.get(self.name, env=self._refresh_env())

    async def predict(self, queries: Any, **options: Any) -> PredictionResult:
        """Uses the captured environment selection, not a mutable global default."""
        return await self._client.predictors.predict(
            self.name, queries, **self._selected(options)
        )

    async def wait(self, **options: Any) -> Environment:
        """Waits for the captured pointer by default."""
        return await self._client.predictors.wait(
            self.name, **self._selected(options, pointer=True)
        )

    async def promote(self, *, revision: int, **options: Any) -> Environment:
        """Promotes under this snapshot's generation when an environment was selected."""
        return await self._client.predictors.promote(
            self.name, revision=revision, **self._selected(options, guard=True)
        )

    async def rollback(self, *, revision: int, **options: Any) -> Environment:
        """Rolls back under this snapshot's generation."""
        return await self._client.predictors.rollback(
            self.name, revision=revision, **self._selected(options, guard=True)
        )

    async def metrics(self, **options: Any) -> Snapshot:
        """Fetches actual metrics for the selected environment."""
        return await self._client.predictors.metrics(
            self.name, **self._selected(options)
        )

    async def logs(self, **options: Any) -> Snapshot:
        """Fetches bounded runtime logs for the selected environment."""
        return await self._client.predictors.logs(self.name, **self._selected(options))

    async def delete(self) -> Snapshot:
        """Requests deletion; the server owns authorization and cleanup."""
        return await self._client.predictors.delete(self.name)


@dataclass(frozen=True, eq=False)
class AsyncLiveContext(ContextHandle):
    """A live context's facts and explicit append/label/import/refresh actions, awaited."""

    async def append(self, data: Any, *, id: str, idempotency_key: str) -> WriteReceipt:
        """Appends rows without mutating this state snapshot."""
        return await self._client.contexts.append(
            self.name, data, id=id, idempotency_key=idempotency_key
        )

    async def label(
        self, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches target values using an explicit stable write identity."""
        return await self._client.contexts.label(
            self.name, data, id=id, column=column, idempotency_key=idempotency_key
        )

    async def import_source(self, reference) -> ImportReceipt:
        """Binds an import only when its live context matches this handle."""
        return await self._client.contexts.import_source(self._owned(reference))

    async def refresh(self) -> ImportReceipt:
        """Explicitly requests provider refresh rather than polling source silently."""
        return await self._client.contexts.refresh(self.name)

    async def status(self) -> AsyncLiveContext:
        """Fetches metadata without triggering provider ingestion."""
        return await self._client.contexts.get(self.name)


class AsyncClient:
    """One explicit endpoint/project and independent application/provider keys.

    Uses only versioned routes. Request IDs correlate; callers must preserve
    write_id/submission_id on retries. No automatic retries or cookie login.
    """

    _hosted_predictor = AsyncHostedPredictor
    _live_context = AsyncLiveContext

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
        """Closes the transport; later calls raise `ClientClosed`."""
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
        return await self._run(
            ops.predict(
                self,
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
        )

    async def upload_snapshot(self, data: Any) -> JsonObject:
        """Uploads the full normalized Arrow snapshot and verifies the returned digest."""
        return await self._run(ops.upload_snapshot(self, data))

    async def _call(
        self,
        method: str,
        path: str,
        kind: ResourceKind,
        resource: str,
        payload: JsonObject | None = None,
        *,
        global_route: bool = False,
        params: Mapping[str, Any] | None = None,
        deadline: float | None = None,
    ) -> JsonObject:
        """Sends one versioned request and returns its unwrapped result.

        Every operation's calls go through here, so this one method sends
        each versioned request a client makes.
        """
        prepared = ops.prepare(
            self.project,
            Call(method, path, kind, resource, payload, global_route, params, deadline),
        )
        value = await self._transport.send(
            prepared.method,
            prepared.path,
            prepared.body,
            prepared.headers,
            deadline=prepared.deadline,
        )
        return ops.answer(prepared, value)

    async def _run[T](self, operation: Operation[T]) -> T:
        """Awaits an operation's steps, throwing failures back into it."""
        result: Any = None
        failure: Exception | None = None
        while True:
            try:
                step = (
                    operation.send(result)
                    if failure is None
                    else operation.throw(failure)
                )
            except StopIteration as stop:
                return stop.value
            result, failure = None, None
            try:
                result = await self._step(step)
            except Exception as error:  # noqa: BLE001 - thrown into the operation
                failure = error

    async def _step(self, step: ops.Step) -> Any:
        if isinstance(step, Work):
            return await asyncio.to_thread(step.function, *step.args, **step.kwargs)
        if isinstance(step, Pause):
            await asyncio.sleep(step.seconds)
            return None
        if isinstance(step, Call):
            return await self._call(
                step.method,
                step.path,
                step.kind,
                step.resource,
                step.payload,
                global_route=step.global_route,
                params=step.params,
                deadline=step.deadline,
            )
        prepared = ops.prepare(self.project, step)
        value = await self._transport.send(
            prepared.method, prepared.path, prepared.body, prepared.headers
        )
        return ops.answer(prepared, value)


class Listing:
    """Lists one collection of the project's resources."""

    kind: ResourceKind
    collection: str

    def __init__(self, client: AsyncClient) -> None:
        """Uses the parent's authenticated, project-scoped transport."""
        self.client = client

    async def list(self) -> builtins.list[JsonObject]:
        """Returns this project's actual resource metadata."""
        return await self.client._run(ops.listed(self.kind, self.collection))


class Collection(Listing):
    """A listed collection whose resources are also read one at a time."""

    async def get(self, resource: str) -> Mapping[str, Any]:
        """Gets one project-qualified resource."""
        return await self.client._run(ops.got(self.kind, self.collection, resource))


class Contexts(Collection):
    """Create-only observations and guarded component corrections."""

    kind: ResourceKind = "context"
    collection = "contexts"

    async def create(self, config: JsonObject | ContextSpec) -> AsyncLiveContext:
        """Creates a live context from its schema."""
        return await self.client._run(ops.context_create(self.client, config))

    async def get(self, resource: str) -> AsyncLiveContext:
        """Returns a detached live-context snapshot with typed write methods."""
        return await self.client._run(ops.context_get(self.client, resource))

    async def append(
        self, context: str, data: Any, *, id: str, idempotency_key: str
    ) -> WriteReceipt:
        """Appends rows under a durable write id."""
        return await self.client._run(
            ops.context_append(context, data, id=id, idempotency_key=idempotency_key)
        )

    async def label(
        self, context: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches one target's values to rows already in the context."""
        return await self.client._run(
            ops.context_label(
                context, data, id=id, column=column, idempotency_key=idempotency_key
            )
        )

    async def import_source(self, reference: ContextImportRef) -> ImportReceipt:
        """Binds a cloud or uploaded table to an existing live context."""
        return await self.client._run(ops.context_import(reference))

    async def import_file(self, context: str, data: Any) -> ImportReceipt:
        """Uploads a local bounded table before binding it to a live context."""
        return await self.client._run(
            ops.context_import_file(self.client, context, data)
        )

    async def refresh(self, context: str) -> ImportReceipt:
        """Requests an explicit provider refresh; unsupported adapters fail visibly."""
        return await self.client._run(ops.context_refresh(context))

    async def mutate(self, context: str, mutation: Mutation) -> AcceptedWrite:
        """Accepts an atomic write, not serving readiness. Preserve write_id."""
        return await self.client._run(ops.context_mutate(context, mutation))

    async def snapshot(
        self,
        context: str,
        *,
        target: str,
        knowledge_at: float | None = None,
        event_before: float | None = None,
    ) -> ContextSnapshot:
        """Keeps knowledge/acceptance time independent from event time."""
        return await self.client._run(
            ops.context_snapshot(
                context,
                target=target,
                knowledge_at=knowledge_at,
                event_before=event_before,
            )
        )


class Sources(Collection):
    """Immutable pins of files under the server's approved import root."""

    kind: ResourceKind = "source"
    collection = "sources"

    async def pin(self, spec: JsonObject) -> SourcePin:
        """Pins server-relative bytes; this does not upload a client-side path."""
        return await self.client._run(ops.source_pin(spec))


class Models(Collection):
    """Curated model metadata and pure caller-context inference."""

    kind: ResourceKind = "model"
    collection = "models"


class Predictors(Collection):
    """Hosted predictors: immutable revisions and environments."""

    kind: ResourceKind = "predictor"
    collection = "predictors"

    async def get(
        self, resource: str, *, env: str | None = None
    ) -> AsyncHostedPredictor:
        """Returns the predictor's facts and optional environment state."""
        return await self.client._run(ops.predictor_get(self.client, resource, env))

    async def create(self, predictor: str) -> AsyncHostedPredictor:
        """Creates only the predictor name, not serving readiness."""
        return await self.client._run(ops.predictor_create(self.client, predictor))

    async def revisions(self, predictor: str) -> tuple[Revision, ...]:
        """Lists immutable revisions without inferring environment readiness."""
        return await self.client._run(ops.predictor_revisions(predictor))

    async def revision(self, predictor: str, number: int) -> RevisionDetails:
        """Reads the exact immutable compiled lock for a revision."""
        return await self.client._run(ops.predictor_revision(predictor, number))

    async def environments(self, predictor: str) -> tuple[Environment, ...]:
        """Lists environment pointers and their actual states."""
        return await self.client._run(ops.predictor_environments(predictor))

    async def environment(
        self, predictor: str, env: str | None = None, *, deadline: float | None = None
    ) -> Environment:
        """Gets one environment with revision and CAS generation."""
        return await self.client._run(
            ops.predictor_environment(self.client, predictor, env, deadline)
        )

    async def create_environment(self, predictor: str, *, env: str) -> Environment:
        """Creates an environment, a pointer to the revision it serves."""
        return await self.client._run(ops.predictor_create_environment(predictor, env))

    async def promote(
        self,
        predictor: str,
        *,
        revision: int,
        env: str | None = None,
        expected_generation: int | None = None,
    ) -> Environment:
        """CAS-promotes an explicit revision; races remain visible conflicts."""
        return await self.client._run(
            ops.predictor_move(
                self.client,
                predictor,
                "promote",
                revision=revision,
                env=env,
                expected_generation=expected_generation,
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
        return await self.client._run(
            ops.predictor_move(
                self.client,
                predictor,
                "rollback",
                revision=revision,
                env=env,
                expected_generation=expected_generation,
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
        """Predicts with a declared predictor; see `Client.predictors.predict`."""
        return await self.client._run(
            ops.predictor_predict(
                self.client, predictor, queries, context=context, as_of=as_of, env=env
            )
        )

    async def push(
        self,
        predictor: str | Path,
        *,
        env: str | None = None,
        wait: bool = False,
        timeout: float = 120,
    ) -> Revision:
        """Pushes a checked predictor: registers a revision and points an environment at it."""
        return await self.client._run(
            ops.predictor_push(
                self.client,
                predictor,
                env=env,
                wait=wait,
                timeout=timeout,
            )
        )

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
        """Polls an exact environment pointer; see `Client.predictors.wait`."""
        return await self.client._run(
            ops.predictor_wait(
                self.client,
                predictor,
                env=env,
                revision=revision,
                generation=generation,
                timeout=timeout,
                poll_interval=poll_interval,
            )
        )

    async def metrics(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets reported metrics without fabricating missing samples."""
        return await self.client._run(
            ops.predictor_report(self.client, predictor, env, "metrics")
        )

    async def logs(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets bounded safe runtime logs."""
        return await self.client._run(
            ops.predictor_report(self.client, predictor, env, "logs")
        )

    async def delete(self, predictor: str) -> Snapshot:
        """Requests explicit stop/delete, never treating unsupported cleanup as success."""
        return await self.client._run(ops.predictor_delete(predictor))


class Identity:
    """Allowlisted management routes; authentication never implies membership."""

    def __init__(self, client: AsyncClient) -> None:
        """Uses one scoped client; Google cookie login belongs to the console."""
        self.client = client

    async def manage(
        self, action: str, payload: JsonObject | None = None
    ) -> JsonObject:
        """Calls concrete identity actions, not a user-selected server method."""
        return await self.client._run(ops.manage(action, payload))


class Keys(Listing):
    """Revocable API keys; create/rotate return the token once.

    Keys are listed, never read one at a time: the service has no
    `GET /keys/{id}`.
    """

    kind: ResourceKind = "api-key"
    collection = "keys"

    async def create(
        self,
        *,
        name: str,
        permissions: builtins.list[str],
        expires_in: float | None = None,
    ) -> JsonObject:
        """Creates only the permissions the server allows the caller to grant."""
        return await self.client._run(
            ops.key_create(name=name, permissions=permissions, expires_in=expires_in)
        )

    async def revoke(self, key: str) -> JsonObject:
        """Revokes a key by public identifier, not by supplying its token."""
        return await self.client._run(ops.manage("keys.revoke", {"id": key}))

    async def rotate(self, key: str) -> JsonObject:
        """Returns a replacement token once; does not persist it locally."""
        return await self.client._run(ops.manage("keys.rotate", {"id": key}))


class Secrets(Collection):
    """Encrypted secret writes and metadata-only reads/listings."""

    kind: ResourceKind = "secret"
    collection = "secrets"

    async def create(self, *, name: str, value: str) -> JsonObject:
        """Sends secret material only in an authenticated TLS request body."""
        return await self.client._run(
            ops.manage("secrets.create", {"name": name, "value": value})
        )

    async def rotate(
        self, *, name: str, value: str, expected_version: int
    ) -> JsonObject:
        """Uses an explicit version guard instead of blindly overwriting."""
        return await self.client._run(
            ops.manage(
                "secrets.rotate",
                {"name": name, "value": value, "expected_version": expected_version},
            )
        )

    async def delete(self, secret: str) -> JsonObject:
        """Deletes by public name; it never returns the previous value."""
        return await self.client._run(ops.manage("secrets.delete", {"name": secret}))
