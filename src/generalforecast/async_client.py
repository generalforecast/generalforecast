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
import inspect
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from generalforecast import _operations as ops
from generalforecast._operations import (
    Call,
    Emit,
    Operation,
    PartialWrite,
    Pause,
    Upload,
    Work,
)
from generalforecast.contracts import (
    ContextImportRef,
    Environment,
    PredictionResult,
    Revision,
)
from generalforecast.errors import Error
from generalforecast.retry import DEFAULT_RETRY, WRITE, RetryPolicy, Tries
from generalforecast.service.boundary import JsonObject, ResourceKind
from generalforecast.service.contracts import require
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
from generalforecast.transport import AsyncTransport, ConnectionSettings, Profile

__all__ = ["AsyncClient", "AsyncHostedPredictor", "AsyncLiveContext", "PartialWrite"]


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
    write_id/submission_id on retries. Transient failures of the requests
    that are safe to send again (predictions, reads, idempotent creates and
    pointer moves) are retried under `retry`, a `RetryPolicy`; any other
    request is sent once. No cookie login.
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
        settings: ConnectionSettings | None = None,
        retry: RetryPolicy | int | bool | None = DEFAULT_RETRY,
        **transport_options: Any,
    ) -> None:
        """Resolves the connection as the CLI does, without persisting credentials.

        A profile (named, GF_PROFILE, GF_TOKEN or a saved `default`), else
        GF_URL or the hosted service with GF_API_KEY or the `gf login` key;
        see `ConnectionSettings.resolve`. base_url targets one endpoint, such
        as a local `gf serve`, with no credential unless api_key is given.
        `settings` is a connection resolved already, such as the CLI's, in
        place of the other connection arguments.

        `retry` is the `RetryPolicy` transient failures are retried under,
        for the requests that are safe to send again (`generalforecast.retry`):
        by default for up to two minutes. None or 0 sends every request once; a number is the most tries.
        """
        self._retry = RetryPolicy.of(retry)
        self._retry_sleep = asyncio.sleep
        if settings is None:
            settings = ConnectionSettings.resolve(
                url, project, token, api_key, profile, config_path, base_url=base_url
            )
        else:
            require(
                (url, project, token, api_key, profile, config_path, base_url)
                == (None,) * 7,
                "pass settings or connection arguments, not both",
            )
        self.profile = settings.profile
        # None until the service says which project an unscoped key is for.
        self.project: str | None = (
            None if settings.project_from_key else settings.profile.project
        )
        key = settings.token
        self._transport = AsyncTransport(
            self.profile.url, token=key, **transport_options
        )
        self.contexts = Contexts(self)
        self.sources = Sources(self)
        self.snapshots = Snapshots(self)
        self.runs = Runs(self)
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

    async def upload_snapshot(self, data: Any, *, prefix: str = "table") -> JsonObject:
        """Sends a table as the snapshot named after `prefix` and its digest, verified.

        The name is `<prefix>-<first 12 hex digits of the digest>`, so the
        same table sent again returns the same snapshot.
        """
        return await self._run(ops.upload_snapshot(self, data, prefix=prefix))

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
        retry: str | None = None,
    ) -> JsonObject:
        """Sends one versioned request and returns its unwrapped result.

        Every operation's calls go through here, so this one method sends
        each versioned request a client makes. `retry` is the call's retry
        class (`Call.retry`).
        """
        project = self.project if self.project is not None else await self._scope()
        call = Call(
            method, path, kind, resource, payload, global_route, params, deadline, retry
        )
        prepared = ops.prepare(project, call)
        return await self._send(prepared, call.retry_class)

    async def _send(
        self, prepared: ops.Prepared, retry_class: str | None
    ) -> JsonObject:
        """Sends a prepared request, again after a transient failure when its class allows.

        Every try sends the same request, its request id included, so the
        service can tell a try again from a new request.
        """
        tries = (
            None
            if self._retry is None or retry_class is None
            else Tries(
                self._retry,
                retry_class,
                prepared.method,
                prepared.path,
                prepared.headers.get("X-GF-Request-ID"),
                prepared.deadline,
            )
        )
        while True:
            try:
                value = await self._transport.send(
                    prepared.method,
                    prepared.path,
                    prepared.body,
                    prepared.headers,
                    deadline=prepared.deadline,
                )
            except Error as error:
                wait = None if tries is None else tries.failed(error)
                if wait is None:
                    raise
                await self._retry_sleep(wait)
                continue
            return ops.answer(prepared, value)

    async def _run[T](
        self, operation: Operation[T], emit: Callable[[Any], Any] | None = None
    ) -> T:
        """Awaits an operation's steps, throwing failures back into it.

        What the operation emits goes to `emit`, if given, which may be a
        plain function or a coroutine function; a failure there is thrown
        back into the operation as a step's would be.
        """
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
                if isinstance(step, Emit):
                    if emit is not None:
                        called = emit(step.value)
                        if inspect.isawaitable(called):
                            await called
                else:
                    result = await self._step(step)
            except Exception as error:  # noqa: BLE001 - thrown into the operation
                failure = error

    async def _watch(self, operation: Operation[Any]) -> AsyncIterator[Any]:
        """Awaits an operation's steps, yielding what it emits."""
        result: Any = None
        failure: Exception | None = None
        while True:
            try:
                step = (
                    operation.send(result)
                    if failure is None
                    else operation.throw(failure)
                )
            except StopIteration:
                return
            result, failure = None, None
            if isinstance(step, Emit):
                yield step.value
                continue
            try:
                result = await self._step(step)
            except Exception as error:  # noqa: BLE001 - thrown into the operation
                failure = error

    async def _step(self, step: ops.Step) -> Any:
        if isinstance(step, Emit):
            return None
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
                retry=step.retry,
            )
        project = self.project if self.project is not None else await self._scope()
        prepared = ops.prepare(project, step)
        # A snapshot is stored by its content, so sending it again is safe; a
        # training upload's chunk is sent once.
        return await self._send(prepared, WRITE if isinstance(step, Upload) else None)

    async def _scope(self) -> str:
        """The project an unscoped key is for, asked of the service once."""
        prepared = ops.prepare(ops.UNSCOPED, ops.KEY_SCOPE)
        value = await self._send(prepared, ops.KEY_SCOPE.retry_class)
        self.project = ops.key_project(value)
        return self.project


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
        """Appends rows under a durable write id.

        More rows than one call takes go in chunks, each under a write id
        derived from `idempotency_key`; a chunk that fails after others were
        accepted raises `PartialWrite`, and retrying with the same key resumes.
        """
        return await self.client._run(
            ops.context_append(context, data, id=id, idempotency_key=idempotency_key)
        )

    async def label(
        self, context: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches one target's values to rows already in the context.

        Chunked as `append` is, with the same `PartialWrite` on a failure.
        """
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


class Snapshots(Collection):
    """Named, immutable tables: uploaded in chunks for training, or streamed as Arrow.

    A name is unique in the project and never reused, even once its
    snapshot is deleted. Runs read snapshots by name; the same name with
    the same table again returns the snapshot, and with another table is
    refused with `Conflict`.
    """

    kind: ResourceKind = "snapshot"
    collection = "snapshots"

    async def create(
        self,
        name: str,
        data: Any,
        *,
        target: str,
        task: str,
        column_types: Mapping[str, str] | None = None,
        time: str | None = None,
        group: str | None = None,
        origin: JsonObject | None = None,
        wait: bool = True,
    ) -> JsonObject:
        """Uploads a training table as `name`: a dataframe, an Arrow table, Parquet bytes, or a file.

        A file is a path to .parquet (sent as it is), or .csv, .tsv, .arrow,
        .feather or .jsonl (written as Parquet first); other bytes or files
        are refused before the upload opens.

        It goes in chunks, straight to object storage where the service
        says. With `wait`, returns once the service has verified it and the
        snapshot is `ready`; a rejected upload raises `ValidationError`.
        """
        metadata = ops.upload_metadata(
            target,
            task,
            column_types=column_types,
            time=time,
            group=group,
            origin=origin,
        )
        return await self.client._run(
            ops.training_upload(name, data, metadata, wait=wait)
        )

    async def stream(
        self,
        name: str,
        data: Any,
        *,
        target: str | None = None,
        task: str | None = None,
        time: str | None = None,
        group: str | None = None,
    ) -> JsonObject:
        """Sends a small table (100,000 rows, 32 MiB) as one Arrow stream, `ready` at once.

        Its training metadata is optional: a run reads it once it names a
        `target` and `task`.
        """
        metadata = {
            key: value
            for key, value in (
                ("target", target),
                ("task", task),
                ("time", time),
                ("group", group),
            )
            if value is not None
        }
        return await self.client._run(
            ops.upload_snapshot(self.client, data, snapshot=name, metadata=metadata)
        )

    async def list(self) -> builtins.list[JsonObject]:
        """The project's snapshots that are not deleted, oldest first."""
        return await self.client._run(ops.listed(self.kind, self.collection))

    async def delete(self, name: str) -> JsonObject:
        """Deletes a snapshot; its name stays taken.

        Refused with `Conflict` (reason `snapshot_held`) while a run that
        has not ended reads it. Deleting again changes nothing.
        """
        return await self.client._run(
            ops.posted("snapshot", "snapshots", name, "delete", {})
        )


class Runs(Collection):
    """Fine-tuning runs, one per version of a fine-tune, followed until they end.

    As `Client.runs`: a run is named by the model it publishes, its
    `model`, `<project>/<name>@ft<version>`, or `<project>/<name>` for the
    name's latest run, with the project left out for the client's own.
    """

    kind: ResourceKind = "run"
    collection = "runs"

    async def get(self, resource: str) -> Mapping[str, Any]:
        """A run by its model, `<project>/<name>@ft<version>`, or a name's latest run."""
        return await self.client._run(ops.run_got(resource, self.client.project))

    async def create(
        self,
        name: str,
        *,
        base: str,
        data: builtins.list[str],
        method: JsonObject,
        plan: JsonObject,
        evaluation: JsonObject | None = None,
        budget: JsonObject | None = None,
        seed: int | None = None,
        tags: builtins.list[str] | None = None,
    ) -> JsonObject:
        """Starts a run that publishes the fine-tune `name`'s next version.

        The first run of a name publishes `<project>/<name>@ft1`, and each
        create after it the next version, `@ft2` and on; the run's `model`
        is that version's id from the start, and names the run for `get`,
        `wait`, `watch`, `cancel` and `lineage`. `base` is the catalog model
        the run trains from, and every version of a name trains afresh from
        the same one: another base is refused with `Conflict`, a fine-tune
        with `ValidationError`, and so is a new run while one of the name
        is still queued or running.

        `data` names ready snapshots that together are one table (the same
        columns, target and task); the run pins each to its digest, and
        runs of the same data share its `data_digest`. `method` is how the
        network changes, `{"type": "lora", "rank": 16}` (with
        `"input_adapter": true` for an adapter beside it),
        `{"type": "input_adapter"}` or `{"type": "full", "components":
        [...]}`. `plan` gives its `steps` and the trainer's settings,
        `{"steps": 200, "learning_rate": 5e-5, "episode_sampling": {"rows":
        1024}}`; anything the trainer does not read is refused with
        `ValidationError` (`generalforecast._api.RunPlan`).

        `evaluation` says how the run holds out rows and keeps a step,
        `{"holdout": 0.2, "context_rows": 32768, "contexts": 3,
        "early_stopping_threshold": {"standard_errors": 2.0},
        "report_share": 0.5}`, each setting left out at that default
        (`context_rows` the base's trained row cap): a step is kept only on
        a gain over the base of more than that many standard errors, and
        the published version's `report` scores it against the base on
        rows that played no part in choosing it.

        A try sent again after a transient failure carries the same request
        id, so the service returns the run the first try made rather than
        start a second.
        """
        payload = ops.run_spec(
            name,
            base=base,
            data=data,
            method=method,
            plan=plan,
            evaluation=evaluation,
            budget=budget,
            seed=seed,
            tags=tags,
        )
        return await self.client._run(ops.run_create(name, payload))

    async def list(
        self,
        *,
        state: str | None = None,
        data_digest: str | None = None,
        tag: str | None = None,
    ) -> builtins.list[JsonObject]:
        """The project's runs by name and version, only those matching what is given.

        `state` lists the runs in it, `data_digest` those over the data
        with that digest, and `tag` those carrying it. A state that is not
        a run state is refused with `ValidationError`.
        """
        return await self.client._run(ops.run_list(state, data_digest, tag))

    async def cancel(self, run: str, *, save: bool = False) -> JsonObject:
        """Cancels a run; one that is training stops at its next step.

        With `save`, a run that is training publishes the weights it has
        trained so far as the version it was created for, then ends
        `cancelled`; its lineage names the version. A run not yet training
        has nothing to save and is cancelled as it would be without `save`.
        """
        return await self.client._run(
            ops.run_cancel(run, save=save, project=self.client.project)
        )

    async def lineage(self, run: str) -> JsonObject:
        """Every change to a run, oldest first, as `{"model", "events"}`."""
        return await self.client._run(ops.run_lineage(run, self.client.project))

    async def wait(
        self,
        run: str,
        *,
        timeout: float | None = None,
        poll: float = 5.0,
        on_event: Callable[[JsonObject], Any] | None = None,
    ) -> JsonObject:
        """Polls a run until it finishes, is cancelled or fails, and returns it.

        As `Client.runs.wait`: the run's `published` names the version it
        made, or is None. `on_event`, a function or a coroutine function,
        is called with each lineage event once.
        """
        project = self.client.project
        if on_event is None:
            return await self.client._run(
                ops.run_wait(run, timeout=timeout, poll=poll, project=project)
            )
        return await self.client._run(
            ops.run_watch(run, timeout=timeout, poll=poll, project=project), on_event
        )

    def watch(
        self, run: str, *, timeout: float | None = None, poll: float = 5.0
    ) -> AsyncIterator[JsonObject]:
        """Yields a run's lineage events once each, oldest first, until it ends.

        As `Client.runs.watch`, with `async for`.
        """
        return self.client._watch(
            ops.run_watch(run, timeout=timeout, poll=poll, project=self.client.project)
        )


class Models(Collection):
    """Curated model metadata and pure caller-context inference."""

    kind: ResourceKind = "model"
    collection = "models"

    async def get(self, resource: str) -> Mapping[str, Any]:
        """Gets a catalog model, or one of the project's fine-tunes, by its id.

        A catalog model's id is `general-forecast/<name>`. A fine-tune's is
        `<project>/<name>@ft<version>` for one version, or `<project>/<name>`
        for its latest version with every version under `versions`, newest
        first; the run that trained a version is `runs.get` of its id. The reply is the
        `Model` schema of the API (`generalforecast._api.Model`).
        """
        return await self.client._run(ops.model_got(resource, self.client.project))


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
