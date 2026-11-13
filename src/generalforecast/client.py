# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Project-scoped versioned SDK. The server authorizes every operation.

Every operation is defined once, in `generalforecast._operations`; this client
runs its steps in the calling thread, and `AsyncClient` awaits the same steps.

Public, and stable across releases: `Client`, `PartialWrite`, and the
identity action table a hosted service must serve, `WORKSPACE_ACTIONS`
(each action's method, path template and resource name) with
`workspace_request`, which binds one action and its payload to the request
`Client.identity.manage` sends.
"""

import builtins
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from generalforecast import _operations as ops
from generalforecast._operations import (
    WORKSPACE_ACTIONS,
    Call,
    Emit,
    Operation,
    PartialWrite,
    Pause,
    Upload,
    Work,
    workspace_request,
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
    ContextSpec,
    HostedPredictor,
    ImportReceipt,
    LiveContext,
    RevisionDetails,
    Snapshot,
    WriteReceipt,
)
from generalforecast.transport import ConnectionSettings, Profile, Transport

__all__ = ["WORKSPACE_ACTIONS", "Client", "PartialWrite", "workspace_request"]


class Client:
    """One explicit endpoint/project and independent application/provider keys.

    Uses only versioned routes. Request IDs correlate; callers must preserve
    write_id/submission_id on retries. Transient failures of the requests
    that are safe to send again (predictions, reads, idempotent creates and
    pointer moves) are retried under `retry`, a `RetryPolicy`; any other
    request is sent once. No cookie login.
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
        self._retry_sleep = time.sleep
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
        self._transport = Transport(self.profile.url, token=key, **transport_options)
        self.contexts = Contexts(self)
        self.sources = Sources(self)
        self.snapshots = Snapshots(self)
        self.runs = Runs(self)
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

    def upload_snapshot(self, data: Any, *, prefix: str = "table") -> JsonObject:
        """Sends a table as the snapshot named after `prefix` and its digest, verified.

        The name is `<prefix>-<first 12 hex digits of the digest>`, so the
        same table sent again returns the same snapshot.
        """
        return self._run(ops.upload_snapshot(self, data, prefix=prefix))

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
        retry: str | None = None,
    ) -> JsonObject:
        """Sends one versioned request and returns its unwrapped result.

        Every operation's calls go through here, so this one method sends
        each versioned request a client makes. `retry` is the call's retry
        class (`Call.retry`).
        """
        project = self.project if self.project is not None else self._scope()
        call = Call(
            method, path, kind, resource, payload, global_route, params, deadline, retry
        )
        prepared = ops.prepare(project, call)
        return self._send(prepared, call.retry_class)

    def _send(self, prepared: ops.Prepared, retry_class: str | None) -> JsonObject:
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
                value = self._transport.send(
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
                self._retry_sleep(wait)
                continue
            return ops.answer(prepared, value)

    def _run[T](
        self, operation: Operation[T], emit: Callable[[Any], Any] | None = None
    ) -> T:
        """Runs an operation's steps in this thread, throwing failures back into it.

        What the operation emits goes to `emit`, if given; a failure there
        is thrown back into the operation as a step's would be.
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
                        emit(step.value)
                else:
                    result = self._step(step)
            except Exception as error:  # noqa: BLE001 - thrown into the operation
                failure = error

    def _watch(self, operation: Operation[Any]) -> Iterator[Any]:
        """Runs an operation's steps in this thread, yielding what it emits."""
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
                result = self._step(step)
            except Exception as error:  # noqa: BLE001 - thrown into the operation
                failure = error

    def _step(self, step: ops.Step) -> Any:
        if isinstance(step, Emit):
            return None
        if isinstance(step, Work):
            return step.function(*step.args, **step.kwargs)
        if isinstance(step, Pause):
            time.sleep(step.seconds)
            return None
        if isinstance(step, Call):
            return self._call(
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
        project = self.project if self.project is not None else self._scope()
        prepared = ops.prepare(project, step)
        # A snapshot is stored by its content, so sending it again is safe; a
        # training upload's chunk is sent once.
        return self._send(prepared, WRITE if isinstance(step, Upload) else None)

    def _scope(self) -> str:
        """The project an unscoped key is for, asked of the service once."""
        prepared = ops.prepare(ops.UNSCOPED, ops.KEY_SCOPE)
        value = self._send(prepared, ops.KEY_SCOPE.retry_class)
        self.project = ops.key_project(value)
        return self.project


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
        """Appends rows under a durable write id.

        More rows than one call takes go in chunks, each under a write id
        derived from `idempotency_key`; a chunk that fails after others were
        accepted raises `PartialWrite`, and retrying with the same key resumes.
        """
        return self.client._run(
            ops.context_append(context, data, id=id, idempotency_key=idempotency_key)
        )

    def label(
        self, context: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches one target's values to rows already in the context.

        Chunked as `append` is, with the same `PartialWrite` on a failure.
        """
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


class Snapshots(Collection):
    """Named, immutable tables: uploaded in chunks for training, or streamed as Arrow.

    A name is unique in the project and never reused, even once its
    snapshot is deleted. Runs read snapshots by name; the same name with
    the same table again returns the snapshot, and with another table is
    refused with `Conflict`.
    """

    kind: ResourceKind = "snapshot"
    collection = "snapshots"

    def create(
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
        """Uploads a training table (a frame, an Arrow table, or a Parquet path or bytes) as `name`.

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
        return self.client._run(ops.training_upload(name, data, metadata, wait=wait))

    def stream(
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
        return self.client._run(
            ops.upload_snapshot(self.client, data, snapshot=name, metadata=metadata)
        )

    def list(self) -> builtins.list[JsonObject]:
        """The project's snapshots that are not deleted, oldest first."""
        return self.client._run(ops.listed(self.kind, self.collection))

    def delete(self, name: str) -> JsonObject:
        """Deletes a snapshot; its name stays taken.

        Refused with `Conflict` (reason `snapshot_held`) while a run that
        has not ended reads it. Deleting again changes nothing.
        """
        return self.client._run(ops.posted("snapshot", "snapshots", name, "delete", {}))


class Runs(Collection):
    """Fine-tuning runs, one per version of a fine-tune, followed until they end.

    A run is named by the model it publishes, its `model`,
    `<project>/<name>@ft<version>`, as `acme/credit-default@ft1`; every
    method takes that, or `<project>/<name>` for the name's latest run.
    The project may be left out, `credit-default@ft1` or `credit-default`,
    for a run of the client's own project; a model of another project is
    refused with `ValidationError`.
    """

    kind: ResourceKind = "run"
    collection = "runs"

    def get(self, resource: str) -> Mapping[str, Any]:
        """A run by its model, `<project>/<name>@ft<version>`, or a name's latest run."""
        return self.client._run(ops.run_got(resource, self.client.project))

    def create(
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
        return self.client._run(ops.run_create(name, payload))

    def list(
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
        return self.client._run(ops.run_list(state, data_digest, tag))

    def cancel(self, run: str, *, save: bool = False) -> JsonObject:
        """Cancels a run; one that is training stops at its next step.

        With `save`, a run that is training publishes the weights it has
        trained so far as the version it was created for, then ends
        `cancelled`; its lineage names the version. A run not yet training
        has nothing to save and is cancelled as it would be without `save`.
        """
        return self.client._run(
            ops.run_cancel(run, save=save, project=self.client.project)
        )

    def lineage(self, run: str) -> JsonObject:
        """Every change to a run, oldest first, as `{"model", "events"}`."""
        return self.client._run(ops.run_lineage(run, self.client.project))

    def wait(
        self,
        run: str,
        *,
        timeout: float | None = None,
        poll: float = 5.0,
        on_event: Callable[[JsonObject], Any] | None = None,
    ) -> JsonObject:
        """Polls a run until it finishes, is cancelled or fails, and returns it.

        The run's `published` names the version it made, as `{"model":
        "<project>/<name>@ft<version>", "digest", "kept_step",
        "validation_loss", "base_validation_loss"}`, or is None when it made
        none. With `on_event`, each lineage event is passed to it once, as
        `watch` yields them. A name alone waits for the run that was its
        latest when the wait began. Past `timeout` seconds it raises
        `WaitTimeout` (`gf.ModelNotReady`), and the run carries on.
        """
        project = self.client.project
        if on_event is None:
            return self.client._run(
                ops.run_wait(run, timeout=timeout, poll=poll, project=project)
            )
        return self.client._run(
            ops.run_watch(run, timeout=timeout, poll=poll, project=project), on_event
        )

    def watch(
        self, run: str, *, timeout: float | None = None, poll: float = 5.0
    ) -> Iterator[JsonObject]:
        """Yields a run's lineage events once each, oldest first, until it ends.

        Each event is `{"at", "kind", "body"}`, as `lineage` lists them:
        `state` for each change of state, `metrics` for each stretch of
        training (`steps`, `loss`, `validation_loss`, `seconds`), and
        `published` when it publishes. Events from before the watch began
        come first. The iteration ends once the run has ended; `get` or
        `wait` then reads it. Past `timeout` seconds it raises `WaitTimeout`.
        """
        return self.client._watch(
            ops.run_watch(run, timeout=timeout, poll=poll, project=self.client.project)
        )


class Models(Collection):
    """Curated model metadata and pure caller-context inference."""

    kind: ResourceKind = "model"
    collection = "models"

    def get(self, resource: str) -> Mapping[str, Any]:
        """Gets a catalog model, or one of the project's fine-tunes, by its id.

        A catalog model's id is `general-forecast/<name>`. A fine-tune's is
        `<project>/<name>@ft<version>` for one version, or `<project>/<name>`
        for its latest version with every version under `versions`, newest
        first; the run that trained a version is `runs.get` of its id. The reply is the
        `Model` schema of the API (`generalforecast._api.Model`).
        """
        return self.client._run(ops.model_got(resource, self.client.project))


class Predictors(Collection):
    """Hosted predictors: immutable revisions and environments."""

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
        """Lists environment pointers and their actual states."""
        return self.client._run(ops.predictor_environments(predictor))

    def environment(
        self, predictor: str, env: str | None = None, *, deadline: float | None = None
    ) -> Environment:
        """Gets one environment with revision and CAS generation."""
        return self.client._run(
            ops.predictor_environment(self.client, predictor, env, deadline)
        )

    def create_environment(self, predictor: str, *, env: str) -> Environment:
        """Creates an environment, a pointer to the revision it serves."""
        return self.client._run(ops.predictor_create_environment(predictor, env))

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
        wait: bool = False,
        timeout: float = 120,
    ) -> Revision:
        """Pushes a checked predictor: registers a revision and points an environment at it."""
        return self.client._run(
            ops.predictor_push(
                self.client,
                predictor,
                env=env,
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
