# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Project-scoped versioned SDK. The server authorizes every operation.

Every operation is defined once, in `generalforecast._operations`; this client
runs its steps in the calling thread, and `AsyncClient` awaits the same steps.
"""

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from generalforecast import _operations as ops
from generalforecast._operations import (
    WORKSPACE_ACTIONS,
    Call,
    Operation,
    Pause,
    Work,
    workspace_request,
)
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
    ContextSpec,
    HostedPredictor,
    ImportReceipt,
    LiveContext,
    RevisionDetails,
    Snapshot,
    WriteReceipt,
)
from generalforecast.transport import Profile, Transport, connection

__all__ = ["WORKSPACE_ACTIONS", "Client", "workspace_request"]


class Client:
    """One explicit endpoint/project and independent application/provider keys.

    Uses only versioned routes. Request IDs correlate; callers must preserve
    write_id/submission_id on retries. No automatic retries or cookie login.
    Errors are `ServiceError`s of the `gf.Error` category their code names,
    so `except gf.NotFound` catches a missing resource.
    """

    _hosted_predictor = HostedPredictor
    _live_context = LiveContext

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
        self._transport = Transport(self.profile.url, token=key, **transport_options)
        self.contexts = Contexts(self)
        self.sources = Sources(self)
        self.models = Models(self)
        self.predictors = Predictors(self)
        self.identity = Identity(self)
        self.keys = Keys(self)
        self.secrets = Secrets(self)

    def __enter__(self) -> Self:
        """Returns this synchronous client."""
        return self

    def __exit__(self, *_args) -> None:
        """Closes the client on scope exit."""
        self.close()

    def close(self) -> None:
        """Closes the transport; later calls raise `ClientClosed`."""
        self._transport.close()

    def predict(
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
        """Predicts with an ad hoc predictor: a manifest fragment plus rows.

        Args:
            model: A catalog id such as `general-forecast/tabicl-v2-classifier`.
            target: The column of `context` to predict.
            context: Rows the model learns from.
            queries: Rows to score.
            inputs: The columns the model may read; every other column if None.
            id: The id column, if any.
            time: The time column, for ordered rows or a forecast.
            series: The series columns, for a forecast.
            frequency: A forecast's step, such as `1d` or seconds.
            horizon: Steps a forecast covers.
            quantiles: Levels a regressor or forecaster reports.
            outputs: Which outputs to return; the model's default if None.
            selection: How each query's context is drawn: core, neighbors, kind.
        """
        return self._run(
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

    def upload_snapshot(self, data: Any) -> JsonObject:
        """Uploads the full normalized Arrow snapshot and verifies the returned digest."""
        return self._run(ops.upload_snapshot(self, data))

    def _call(
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
        """Sends one versioned request and returns its unwrapped result."""
        return self._step(
            Call(method, path, kind, resource, payload, global_route, params, deadline)
        )

    def _run[T](self, operation: Operation[T]) -> T:
        """Runs an operation's steps in this thread, throwing failures back into it."""
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
                result = self._step(step)
            except Exception as error:  # noqa: BLE001 - thrown into the operation
                failure = error

    def _step(self, step: ops.Step) -> Any:
        if isinstance(step, Work):
            return step.function(*step.args, **step.kwargs)
        if isinstance(step, Pause):
            time.sleep(step.seconds)
            return None
        prepared = ops.prepare(self.project, step)
        value = self._transport.send(
            prepared.method,
            prepared.path,
            prepared.body,
            prepared.headers,
            deadline=prepared.deadline,
        )
        return ops.answer(prepared, value)


class Listing:
    """Lists one collection of the project's resources."""

    kind: ResourceKind
    collection: str

    def __init__(self, client: Client) -> None:
        """Uses the parent's authenticated, project-scoped transport."""
        self.client = client

    def list(self) -> list[JsonObject]:
        """Returns this project's actual resource metadata."""
        return self.client._run(ops.listed(self.kind, self.collection))


class Collection(Listing):
    """A listed collection whose resources are also read one at a time."""

    def get(self, resource: str) -> Mapping[str, Any]:
        """Gets one project-qualified resource."""
        return self.client._run(ops.got(self.kind, self.collection, resource))


class Contexts(Collection):
    """Create-only observations and guarded component corrections."""

    kind: ResourceKind = "context"
    collection = "contexts"

    def create(self, config: JsonObject | ContextSpec) -> LiveContext:
        """Creates a live context from its schema."""
        return self.client._run(ops.context_create(self.client, config))

    def get(self, resource: str) -> LiveContext:
        """Returns a detached live-context snapshot with typed write methods."""
        return self.client._run(ops.context_get(self.client, resource))

    def append(
        self, context: str, data: Any, *, id: str, idempotency_key: str
    ) -> WriteReceipt:
        """Appends rows under a durable write id."""
        return self.client._run(
            ops.context_append(context, data, id=id, idempotency_key=idempotency_key)
        )

    def label(
        self, context: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches one target's values to rows already in the context."""
        return self.client._run(
            ops.context_label(
                context, data, id=id, column=column, idempotency_key=idempotency_key
            )
        )

    def import_source(self, reference: ContextImportRef) -> ImportReceipt:
        """Binds a cloud or uploaded table to an existing live context."""
        return self.client._run(ops.context_import(reference))

    def import_file(self, context: str, data: Any) -> ImportReceipt:
        """Uploads a local bounded table before binding it to a live context."""
        return self.client._run(ops.context_import_file(self.client, context, data))

    def refresh(self, context: str) -> ImportReceipt:
        """Requests an explicit provider refresh; unsupported adapters fail visibly."""
        return self.client._run(ops.context_refresh(context))

    def mutate(self, context: str, mutation: Mutation) -> AcceptedWrite:
        """Accepts an atomic write, not serving readiness. Preserve write_id."""
        return self.client._run(ops.context_mutate(context, mutation))

    def snapshot(
        self,
        context: str,
        *,
        target: str,
        knowledge_at: float | None = None,
        event_before: float | None = None,
    ) -> ContextSnapshot:
        """Keeps knowledge/acceptance time independent from event time."""
        return self.client._run(
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

    def pin(self, spec: JsonObject) -> SourcePin:
        """Pins server-relative bytes; this does not upload a client-side path."""
        return self.client._run(ops.source_pin(spec))


class Models(Collection):
    """Curated model metadata and pure caller-context inference."""

    kind: ResourceKind = "model"
    collection = "models"


class Predictors(Collection):
    """Hosted predictors: immutable revisions and runtime-bound environments."""

    kind: ResourceKind = "predictor"
    collection = "predictors"

    def get(self, resource: str, *, env: str | None = None) -> HostedPredictor:
        """Returns the predictor's facts and optional environment state."""
        return self.client._run(ops.predictor_get(self.client, resource, env))

    def create(self, predictor: str) -> HostedPredictor:
        """Creates only the predictor name, not serving readiness."""
        return self.client._run(ops.predictor_create(self.client, predictor))

    def revisions(self, predictor: str) -> tuple[Revision, ...]:
        """Lists immutable revisions without inferring environment readiness."""
        return self.client._run(ops.predictor_revisions(predictor))

    def revision(self, predictor: str, number: int) -> RevisionDetails:
        """Reads the exact immutable compiled lock for a revision."""
        return self.client._run(ops.predictor_revision(predictor, number))

    def environments(self, predictor: str) -> tuple[Environment, ...]:
        """Lists runtime-bound environment pointers and their actual states."""
        return self.client._run(ops.predictor_environments(predictor))

    def environment(
        self, predictor: str, env: str | None = None, *, deadline: float | None = None
    ) -> Environment:
        """Gets one environment with revision and CAS generation."""
        return self.client._run(
            ops.predictor_environment(self.client, predictor, env, deadline)
        )

    def create_environment(
        self, predictor: str, *, env: str, runtime: str
    ) -> Environment:
        """Binds a runtime separately from predictor content and authentication."""
        return self.client._run(
            ops.predictor_create_environment(predictor, env, runtime)
        )

    def promote(
        self,
        predictor: str,
        *,
        revision: int,
        env: str | None = None,
        expected_generation: int | None = None,
    ) -> Environment:
        """CAS-promotes an explicit revision; races remain visible conflicts."""
        return self.client._run(
            ops.predictor_move(
                self.client,
                predictor,
                "promote",
                revision=revision,
                env=env,
                expected_generation=expected_generation,
            )
        )

    def rollback(
        self,
        predictor: str,
        *,
        revision: int,
        env: str | None = None,
        expected_generation: int | None = None,
    ) -> Environment:
        """Rolls back to an explicit immutable revision under the same CAS guard."""
        return self.client._run(
            ops.predictor_move(
                self.client,
                predictor,
                "rollback",
                revision=revision,
                env=env,
                expected_generation=expected_generation,
            )
        )

    def predict(
        self,
        predictor: str,
        queries: Any,
        *,
        context: Any = None,
        as_of: Any = None,
        env: str | None = None,
    ) -> PredictionResult:
        """Predicts with a declared predictor in an environment.

        Args:
            predictor: The predictor's name.
            queries: Rows to score.
            context: Rows the model learns from; only for a predictor whose
                context kind is `request`.
            as_of: The time the context is read as of, for a live context.
            env: The environment; the profile's (`production` by default).
        """
        return self.client._run(
            ops.predictor_predict(
                self.client, predictor, queries, context=context, as_of=as_of, env=env
            )
        )

    def push(
        self,
        predictor: str | Path,
        *,
        env: str | None = None,
        runtime: str | None = None,
        wait: bool = False,
        timeout: float = 120,
    ) -> Revision:
        """Pushes a checked predictor: registers a revision and points an environment at it."""
        return self.client._run(
            ops.predictor_push(
                self.client,
                predictor,
                env=env,
                runtime=runtime,
                wait=wait,
                timeout=timeout,
            )
        )

    def wait(
        self,
        predictor: str,
        *,
        env: str | None = None,
        revision: int | None = None,
        generation: int | None = None,
        timeout: float = 120,
        poll_interval: float = 0.2,
    ) -> Environment:
        """Polls an exact environment pointer; superseded and failed states never succeed.

        Raises `WaitTimeout` when `timeout` seconds pass first; a `timeout`
        of 0 checks once.
        """
        return self.client._run(
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

    def metrics(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets reported metrics without fabricating missing samples."""
        return self.client._run(
            ops.predictor_report(self.client, predictor, env, "metrics")
        )

    def logs(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets bounded safe runtime logs."""
        return self.client._run(
            ops.predictor_report(self.client, predictor, env, "logs")
        )

    def delete(self, predictor: str) -> Snapshot:
        """Requests explicit stop/delete, never treating unsupported cleanup as success."""
        return self.client._run(ops.predictor_delete(predictor))


class Identity:
    """Allowlisted management routes; authentication never implies membership."""

    def __init__(self, client: Client) -> None:
        """Uses one scoped client; Google cookie login belongs to the console."""
        self.client = client

    def manage(self, action: str, payload: JsonObject | None = None) -> JsonObject:
        """Calls concrete identity actions, not a user-selected server method."""
        return self.client._run(ops.manage(action, payload))


class Keys(Listing):
    """Revocable API keys; create/rotate return the token once.

    Keys are listed, never read one at a time: the service has no
    `GET /keys/{id}`.
    """

    kind: ResourceKind = "api-key"
    collection = "keys"

    def create(
        self, *, name: str, permissions: list[str], expires_in: float | None = None
    ) -> JsonObject:
        """Creates only the permissions the server allows the caller to grant."""
        return self.client._run(
            ops.key_create(name=name, permissions=permissions, expires_in=expires_in)
        )

    def revoke(self, key: str) -> JsonObject:
        """Revokes a key by public identifier, not by supplying its token."""
        return self.client._run(ops.manage("keys.revoke", {"id": key}))

    def rotate(self, key: str) -> JsonObject:
        """Returns a replacement token once; does not persist it locally."""
        return self.client._run(ops.manage("keys.rotate", {"id": key}))


class Secrets(Collection):
    """Encrypted secret writes and metadata-only reads/listings."""

    kind: ResourceKind = "secret"
    collection = "secrets"

    def create(self, *, name: str, value: str) -> JsonObject:
        """Sends secret material only in an authenticated TLS request body."""
        return self.client._run(
            ops.manage("secrets.create", {"name": name, "value": value})
        )

    def rotate(self, *, name: str, value: str, expected_version: int) -> JsonObject:
        """Uses an explicit version guard instead of blindly overwriting."""
        return self.client._run(
            ops.manage(
                "secrets.rotate",
                {"name": name, "value": value, "expected_version": expected_version},
            )
        )

    def delete(self, secret: str) -> JsonObject:
        """Deletes by public name; it never returns the previous value."""
        return self.client._run(ops.manage("secrets.delete", {"name": secret}))
