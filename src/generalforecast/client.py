# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Project-scoped versioned SDK. The server authorizes every operation."""

import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self, cast
from urllib.parse import urlencode

from generalforecast.contracts import (
    ContextImportRef,
    Environment,
    PredictionResult,
    PromoteRequest,
    Revision,
    parse_lock,
)
from generalforecast.service.boundary import (
    JsonObject,
    Request,
    Resource,
    ResourceKind,
    Scope,
    json_object,
)
from generalforecast.service.contracts import ServiceError, name, number, require
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
    append_payload,
    environment_name,
    inference_result,
    label_payload,
    pushed,
    table,
    unwrap,
)
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
    "workspaces.update": ("POST", "/workspaces/{workspace}", "{workspace}"),
    "workspaces.join": ("POST", "/workspaces/{workspace}/join", "{workspace}"),
    "workspaces.members": ("GET", "/workspaces/{workspace}/members", "{workspace}"),
    "members.update": (
        "POST",
        "/workspaces/{workspace}/members/role",
        "{workspace}",
    ),
    "members.remove": (
        "POST",
        "/workspaces/{workspace}/members/remove",
        "{workspace}",
    ),
    "invitations.preview": ("POST", "/invitations/preview", "invitation"),
    "device.preview": ("POST", "/cli-logins/preview", "cli-login"),
    "device.approve": ("POST", "/cli-logins/approve", "cli-login"),
    "device.deny": ("POST", "/cli-logins/deny", "cli-login"),
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
    "waitlist.join": ("POST", "/waitlist", "waitlist"),
    "waitlist.get": ("GET", "/waitlist/{id}", "{id}"),
    "waitlist.approve": ("POST", "/waitlist/{id}/approve", "{id}"),
    "domains.list": ("GET", "/workspaces/{workspace}/domains", "{workspace}"),
    "domains.configure": ("POST", "/workspaces/{workspace}/domains", "{workspace}"),
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
        """Closes the transport; snapshots remain immutable and readable."""
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
        from generalforecast.predictor import PredictRequest

        request = PredictRequest.build(
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
            self._call("POST", "/predict", "predictor", "ad-hoc", request.payload())
        )

    def upload_snapshot(self, data: Any) -> JsonObject:
        """Uploads the full normalized Arrow snapshot and verifies the returned digest."""
        normalized = table(data)
        request_id = "r-" + uuid.uuid4().hex
        value = self._transport.send(
            "POST",
            f"/v1/projects/{self.project}/snapshots",
            arrow_bytes(normalized),
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
            "digest": table_digest(normalized),
            "rows": normalized.num_rows,
        }:
            raise ServiceError(
                "ProtocolError", "snapshot receipt does not match uploaded content", 502
            )
        return receipt

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
        prefix = "/v1" if global_route else f"/v1/projects/{self.project}"
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


class Contexts(Collection):
    """Create-only observations and guarded component corrections."""

    kind: ResourceKind = "context"
    collection = "contexts"

    def create(self, config: JsonObject | ContextSpec) -> "LiveContext":
        """Creates a live context from its schema."""
        value = config.to_dict() if isinstance(config, ContextSpec) else config
        return LiveContext.from_response(
            self.client, value["name"], self._create(value["name"], value)
        )

    def get(self, resource: str) -> "LiveContext":
        """Returns a detached live-context snapshot with typed write methods."""
        return LiveContext.from_response(self.client, resource, super().get(resource))

    def append(
        self, context: str, data: Any, *, id: str, idempotency_key: str
    ) -> "WriteReceipt":
        """Appends rows under a durable write id."""
        return WriteReceipt.from_dict(
            self._post(
                context, "append", append_payload(data, id=id, write_id=idempotency_key)
            )
        )

    def label(
        self, context: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> "WriteReceipt":
        """Attaches one target's values to rows already in the context."""
        return WriteReceipt.from_dict(
            self._post(
                context,
                "label",
                label_payload(data, id=id, column=column, write_id=idempotency_key),
            )
        )

    def import_source(self, reference: ContextImportRef) -> "ImportReceipt":
        """Binds a cloud or uploaded table to an existing live context."""
        return ImportReceipt.from_dict(
            self._post(reference.context, "imports", reference.to_dict())
        )

    def import_file(self, context: str, data: Any) -> "ImportReceipt":
        """Uploads a local bounded table before binding it to a live context."""
        receipt = self.client.upload_snapshot(data)
        return self.import_source(
            ContextImportRef(
                context,
                "upload://" + receipt["digest"],
                "arrow",
                digest=receipt["digest"],
            )
        )

    def refresh(self, context: str) -> "LiveContext":
        """Requests an explicit provider refresh; unsupported adapters fail visibly."""
        return LiveContext.from_response(
            self.client, context, self._post(context, "refresh", {})
        )

    def mutate(self, context: str, mutation: Mutation) -> AcceptedWrite:
        """Accepts an atomic write, not serving readiness. Preserve write_id."""
        return cast(AcceptedWrite, self._post(context, "mutations", dict(mutation)))

    def snapshot(
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
            self._post(
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

    def pin(self, spec: JsonObject) -> SourcePin:
        """Pins server-relative bytes; this does not upload a client-side path."""
        return cast(SourcePin, self._create("new", spec))


class Models(Collection):
    """Curated model metadata and pure caller-context inference."""

    kind: ResourceKind = "model"
    collection = "models"


class Predictors(Collection):
    """Hosted predictors: immutable revisions and runtime-bound environments."""

    kind: ResourceKind = "predictor"
    collection = "predictors"

    def get(self, resource: str, *, env: str | None = None) -> "HostedPredictor":
        """Returns the predictor's facts and optional environment state."""
        state = super().get(resource)
        environment = self.environment(resource, env) if env is not None else None
        return HostedPredictor.from_response(self.client, resource, state, environment)

    def create(self, predictor: str) -> "HostedPredictor":
        """Creates only the predictor name, not serving readiness."""
        return HostedPredictor.from_response(
            self.client,
            predictor,
            self._create(predictor, {"name": name(predictor)}),
        )

    def revisions(self, predictor: str) -> tuple[Revision, ...]:
        """Lists immutable revisions without inferring environment readiness."""
        value = self.client._call(
            "GET", f"/predictors/{name(predictor)}/revisions", self.kind, predictor
        )
        return tuple(Revision.from_dict(row) for row in value["revisions"])

    def revision(self, predictor: str, number: int) -> "RevisionDetails":
        """Reads the exact immutable compiled lock for a revision."""
        require(type(number) is int and number > 0, "invalid revision")
        value = self.client._call(
            "GET",
            f"/predictors/{name(predictor)}/revisions/{number}",
            self.kind,
            predictor,
        )
        return RevisionDetails(
            Revision.from_dict(value["revision"]), parse_lock(value["lock"])
        )

    def environments(self, predictor: str) -> tuple[Environment, ...]:
        """Lists runtime-bound environment pointers and their actual states."""
        value = self.client._call(
            "GET",
            f"/predictors/{name(predictor)}/environments",
            self.kind,
            predictor,
        )
        return tuple(Environment.from_dict(row) for row in value["environments"])

    def environment(
        self, predictor: str, env: str | None = None, *, deadline: float | None = None
    ) -> Environment:
        """Gets one environment with revision and CAS generation."""
        selected = environment_name(env, self.client.profile.environment)
        return Environment.from_dict(
            self.client._call(
                "GET",
                f"/predictors/{name(predictor)}/environments/{selected}",
                self.kind,
                predictor,
                deadline=deadline,
            )
        )

    def create_environment(
        self, predictor: str, *, env: str, runtime: str
    ) -> Environment:
        """Binds a runtime separately from predictor content and authentication."""
        return Environment.from_dict(
            self._post(
                predictor,
                "environments",
                {"name": name(env), "runtime": name(runtime)},
            )
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
        selected = environment_name(env, self.client.profile.environment)
        generation = (
            self.environment(predictor, selected).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            self._post(
                predictor,
                f"environments/{name(selected)}/promote",
                PromoteRequest(revision, generation).to_dict(),
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
        selected = environment_name(env, self.client.profile.environment)
        generation = (
            self.environment(predictor, selected).generation
            if expected_generation is None
            else expected_generation
        )
        return Environment.from_dict(
            self._post(
                predictor,
                f"environments/{name(selected)}/rollback",
                PromoteRequest(revision, generation).to_dict(),
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
            env: The environment; the profile's, else `production`.
        """
        from generalforecast.predictor import PredictRequest

        payload = PredictRequest.declared_payload(queries, context=context, as_of=as_of)
        params = (
            {"env": environment_name(env, "production")} if env is not None else None
        )
        return inference_result(
            self.client._call(
                "POST",
                f"/{self.collection}/{predictor}/predict",
                "predictor",
                predictor,
                payload,
                params=params,
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
        checked = pushed(predictor)
        lock = checked.lock
        predictor, selected = (
            checked.name,
            environment_name(env, self.client.profile.environment),
        )
        try:
            self.get(predictor)
        except ServiceError as error:
            if error.status != 404:
                raise
            self.create(predictor)
        snapshot = None
        if checked.snapshot is not None:
            snapshot = self.client.upload_snapshot(checked.snapshot)["digest"]
            require(snapshot == lock.data_digest, "snapshot differs from gf.lock")
        revision = Revision.from_dict(
            self._post(
                predictor, "revisions", {"lock": lock.to_dict(), "snapshot": snapshot}
            )
        )
        try:
            environment = self.environment(predictor, selected)
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
                predictor, env=selected, runtime=chosen
            )
        promoted = self.promote(
            predictor,
            revision=revision.number,
            env=selected,
            expected_generation=environment.generation,
        )
        if wait:
            self.wait(
                predictor,
                env=selected,
                revision=revision.number,
                generation=promoted.generation,
                timeout=timeout,
            )
        return revision

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
        """Polls an exact environment pointer; superseded and failed states never succeed."""
        number(timeout, 0, 86400, "timeout")
        number(poll_interval, 0.001, 30, "poll_interval")
        deadline = time.monotonic() + timeout
        while True:
            state = self.environment(predictor, env, deadline=deadline)
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
                    "EnvironmentFailed", "environment did not become ready", 409
                )
            if time.monotonic() >= deadline:
                raise ServiceError(
                    "WaitTimeout", "environment readiness timed out", 504
                )
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def metrics(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets reported metrics without fabricating missing samples."""
        return Snapshot.from_dict(
            self.client._call(
                "GET",
                f"/predictors/{name(predictor)}/environments/{environment_name(env, self.client.profile.environment)}/metrics",
                self.kind,
                predictor,
            )
        )

    def logs(self, predictor: str, *, env: str | None = None) -> Snapshot:
        """Gets bounded safe runtime logs."""
        return Snapshot.from_dict(
            self.client._call(
                "GET",
                f"/predictors/{name(predictor)}/environments/{environment_name(env, self.client.profile.environment)}/logs",
                self.kind,
                predictor,
            )
        )

    def delete(self, predictor: str) -> Snapshot:
        """Requests explicit stop/delete, never treating unsupported cleanup as success."""
        return Snapshot.from_dict(self._post(predictor, "delete", {}))


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
            return self.client._call(
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
