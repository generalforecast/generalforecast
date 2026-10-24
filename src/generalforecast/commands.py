# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `gf model`, `gf context` and `gf admin` commands, and the connection plumbing.

Model commands read the catalog registry; context commands call the live
context (dataset) SDK methods; admin nouns bind to the SDK's identity and
collection operations or start a node or router on this machine.
"""

import argparse
import datetime
import json
import os
import ssl
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from generalforecast import term
from generalforecast.client import Client
from generalforecast.command_specs import (
    COLLECTIONS,
    IDENTITY_GROUPS,
    OPERATIONS,
)
from generalforecast.deployments.boundary import JsonObject
from generalforecast.deployments.contracts import ServiceError, name
from generalforecast.deployments.sdk import Snapshot
from generalforecast.tabular import file_format, read_table, table_digest
from generalforecast.transport import ConnectionSettings, Profile

# The largest table a command reads from disk.
MAX_DATA_BYTES = 128 * 1024 * 1024


class CLIError(ServiceError):
    """Local validation with a constant, safe terminal message."""


from generalforecast.credentials import CONNECTION, uses_profiles
from generalforecast.credentials import SEEN as _CREDENTIAL_VALUES


def require(condition: bool, message: str) -> None:
    """Separates local actionable validation from untrusted remote errors."""
    if not condition:
        raise CLIError("ValidationError", message, 422)


def read_object(path: str | None) -> dict:
    """Reads one bounded JSON/YAML object without reflecting input on failure."""
    if path is None:
        return {}
    from generalforecast.deployments.boundary import MAX_BYTES, json_object

    try:
        with Path(path).open("rb") if path != "-" else _Stdin() as stream:
            raw = stream.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "input exceeds 1 MiB")
        return json_object(yaml.safe_load(raw))
    except (ValueError, UnicodeError, yaml.YAMLError):
        raise ServiceError(
            "ValidationError", "invalid bounded JSON/YAML input", 422
        ) from None


class _Stdin:
    def __enter__(self):
        return sys.stdin.buffer

    def __exit__(self, *_args):
        return False


def read_data(path: str, root: Path):
    """Reads a bounded local table relative to the predictor directory."""
    target = Path(path) if Path(path).is_absolute() else root / path
    with target.open("rb") as stream:
        raw = stream.read(MAX_DATA_BYTES + 1)
    require(len(raw) <= MAX_DATA_BYTES, "tabular input too large")
    return read_table(raw, file_format(target.name))


def records(table) -> list[dict]:
    """Converts date/time values losslessly for bounded JSON prediction requests."""

    def scalar(value):
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.isoformat()
        raise ServiceError(
            "UnsupportedCapability", "binary values require Arrow transport", 422
        )

    return json.loads(json.dumps(table.to_pylist(), default=scalar, allow_nan=False))


# Server and credential source of the last resolved connection, for diagnostics.
def _profile_resolver(args: argparse.Namespace) -> bool:
    """Explicit endpoint flags, GF_PROFILE or a saved profile keep the old resolver."""
    return uses_profiles(args)


def connection_for(args: argparse.Namespace) -> ConnectionSettings:
    """GF_API_KEY or the `gf login` key for GF_URL (default: the hosted service).

    `--local` targets a `gf serve` on this machine without credentials. Saved
    profiles and the hidden --url/--profile/--key-env/--base-url flags still
    resolve through the SDK's profile resolver.
    """
    from generalforecast.credentials import ENV_KEY, CredentialStore, server_url
    from generalforecast.transport import credential

    CONNECTION.clear()
    if getattr(args, "local", False):
        from generalforecast.serve.listener import DEFAULT_PORT

        local = os.environ.get("GF_LOCAL_URL", f"http://127.0.0.1:{DEFAULT_PORT}")
        CONNECTION.update(server=local, source="--local", project_pending="")
        return ConnectionSettings.resolve(
            project=getattr(args, "project", None), base_url=local
        )
    if not _profile_resolver(args):
        server = server_url()
        found = CredentialStore().resolve(server)
        if found is None:
            from generalforecast.cli import Error

            raise Error("Unauthorized", "not logged in", hint="run `gf login`")
        CONNECTION.update(server=server, source=found.source)
        project = (
            getattr(args, "project", None)
            or os.environ.get("GF_PROJECT")
            or (found.login.project if found.login else None)
        )
        CONNECTION["project_pending"] = "" if project else "1"
        profile = Profile(
            server,
            project or "default",
            os.environ.get("GF_ENV", "production"),
            os.environ.get("GF_RUNTIME"),
            key_env=None,
        )
        env = ENV_KEY if found.source == ENV_KEY else None
        return ConnectionSettings(None, profile, env, credential(found.key, "API key"))
    settings = _resolve(args)
    CONNECTION.update(
        server=settings.profile.url,
        source=settings.credential_env or "profile",
        project_pending="",
    )
    return settings


def _key_project(server: str, key: str) -> str:
    """The project a bare GF_API_KEY is scoped to, asked of the server once."""
    client = Client(profile=Profile(server, "local", key_env=None), token=key)
    scoped = client.identity.manage("auth.status").get("key") or {}
    project = scoped.get("project")
    if not isinstance(project, str):
        raise CLIError("ValidationError", "set GF_PROJECT for this API key", 400)
    return project


def _resolve(args: argparse.Namespace) -> ConnectionSettings:
    return ConnectionSettings.resolve(
        getattr(args, "url", None),
        getattr(args, "project", None),
        profile=getattr(args, "profile", None),
        key_env=getattr(args, "key_env", None),
        new_profile=getattr(args, "group", None) == "login",
        base_url=getattr(args, "base_url", None),
    )


def client_for(
    args: argparse.Namespace, settings: ConnectionSettings | None = None
) -> Client:
    """Uses named environment references; never persists credential values."""
    settings = connection_for(args) if settings is None else settings
    token = settings.token
    # Hosted provider credentials stay with profile endpoints, never --base-url.
    provider = (
        {}
        if getattr(args, "base_url", None) is not None or getattr(args, "local", False)
        else {
            "auth_mode": os.environ.get("GF_AUTH_MODE", "bearer"),
            "platform_token": os.environ.get("GF_PLATFORM_TOKEN"),
            "modal_key": os.environ.get("GF_MODAL_KEY"),
            "modal_secret": os.environ.get("GF_MODAL_SECRET"),
        }
    )
    _CREDENTIAL_VALUES.update(
        value
        for value in (
            token,
            provider.get("platform_token"),
            provider.get("modal_key"),
            provider.get("modal_secret"),
        )
        if value
    )
    context = None
    ca, identity = os.environ.get("GF_TLS_CA"), os.environ.get("GF_TLS_IDENTITY")
    if ca or identity:
        try:
            context = ssl.create_default_context(cafile=ca)
            if identity:
                context.load_cert_chain(identity)
        except (OSError, ssl.SSLError):
            raise ServiceError(
                "ClientError", "cannot load verified TLS configuration", 422
            ) from None
    if CONNECTION.get("project_pending") and token:
        return _KeyScopedClient(
            key=token,
            profile=settings.profile,
            token=token,
            timeout=getattr(args, "timeout", 40),
            tls_context=context,
            **provider,
        )
    return Client(
        profile=settings.profile,
        token=token,
        timeout=getattr(args, "timeout", 40),
        tls_context=context,
        **provider,
    )


class _KeyScopedClient(Client):
    """A bare GF_API_KEY learns its project from the server on first use."""

    def __init__(self, *args: Any, key: str, **kwargs: Any) -> None:
        """Defers the project lookup so building a client never touches the network."""
        self._key, self._project, self._ready = key, None, False
        super().__init__(*args, **kwargs)
        self._ready = True

    @property
    def project(self) -> str:
        """The key's project, asked of the server once."""
        if self._project is None:
            self._project = _key_project(self.profile.url, self._key)
        return self._project

    @project.setter
    def project(self, value: str) -> None:
        if self._ready:
            self._project = value


def upload_snapshot(client: Client, table) -> dict:
    """Uploads the full canonical table through the SDK's authenticated transport."""
    result = client.upload_snapshot(table)
    require(
        type(result) is dict
        and result.get("digest") == table_digest(table)
        and type(result.get("rows")) is int
        and result.get("rows") == table.num_rows,
        "snapshot integrity mismatch",
    )
    return result


def execute(args: argparse.Namespace, out: term.Output) -> dict | None:
    """Runs `gf model`, `gf context` or `gf admin`."""
    if args.command == "model":
        return _model(args, out)
    if args.command == "context":
        return _context(args, out)
    return _admin(args, out)


def _admin(args: argparse.Namespace, out: term.Output) -> dict | None:
    """`gf admin <noun> <action>`: API collections, hosted predictors, nodes and routers."""
    noun = args.noun
    if noun in ("node", "router"):
        from generalforecast.predictor_commands import serve_process

        return serve_process(args, out)
    if noun in ("logs", "diff", "delete"):
        from generalforecast.predictor_commands import PredictorCommands

        return getattr(PredictorCommands(args, out), noun)()
    args.group = noun
    args.name = getattr(args, "name", None)
    args.file = getattr(args, "file", None)
    for field, default in (("after", 0), ("limit", 100)):
        if not hasattr(args, field):
            setattr(args, field, default)
    client = client_for(args)
    return _scoped(client, args)


def configured_login(args: argparse.Namespace) -> dict:
    """Verifies a key named by --key-env/--profile and saves the profile."""
    settings = connection_for(args)
    client = client_for(args, settings)
    client.identity.manage("auth.status")
    profile = settings.profile_name or "default"
    stored = Profile.load_all(missing_ok=True).get(profile, {})
    fields: dict[str, Any] = stored | {
        "url": client._transport.url,
        "project": client.project,
        "key_env": settings.credential_env or settings.profile.key_env,
    }
    Profile(**fields).save(profile)
    return {
        "authenticated": True,
        "profile": profile,
        "project": client.project,
        "method": "configured-key",
    }


def _scoped(client: Client, args: argparse.Namespace) -> dict:
    payload = read_object(args.file)
    if args.group in IDENTITY_GROUPS:
        return _identity(client, args, payload)
    return _collection(client, args, payload)


def _identity(client: Client, args: argparse.Namespace, payload: JsonObject) -> dict:
    """Binds CLI arguments to the SDK's allowlisted identity operations."""
    group, action, resource = args.group, args.action, getattr(args, "name", None)
    require(action in OPERATIONS[group], "unsupported operation")
    if action in ("list", "get", "status"):
        require(not payload, "GET operations do not accept a body")
    if group == "audit":
        require(
            0 <= args.after <= 2**53 - 1 and 1 <= args.limit <= 100,
            "invalid audit page",
        )
        payload = {"after": args.after, "limit": args.limit}
    elif group == "secret" and action == "get":
        payload = {"name": name(resource or "")}
    elif action == "create":
        ident = "new" if group == "key" else name(payload.get("name", ""))
        require(
            resource is None or resource == ident,
            "resource name must match config/spec name",
        )
    elif group == "key" and action in ("revoke", "rotate"):
        require(payload.get("id") == name(resource or ""), "key id must match request")
    elif group == "secret" and action in ("rotate", "delete"):
        require(
            payload.get("name") == name(resource or ""),
            "secret name must match request",
        )
    return client.identity.manage(IDENTITY_GROUPS[group] + "." + action, payload)


def _wire(value: Any) -> dict:
    """Returns the exact wire object behind an SDK snapshot DTO."""
    return value.to_dict() if isinstance(value, Snapshot) else value


def _collection(client: Client, args: argparse.Namespace, payload: JsonObject) -> dict:
    """Binds CLI arguments to public SDK collection methods; output stays wire JSON."""
    group, action, resource = args.group, args.action, getattr(args, "name", None)
    require(action in OPERATIONS[group], "unsupported operation")
    if action in ("list", "get", "doctor"):
        require(not payload, "GET operations do not accept a body")
    if group == "operator":
        return client._call("GET", "/operators/doctor", "operator", "doctor")
    collection = COLLECTIONS[group][1]
    sdk = getattr(client, collection)
    if action == "list":
        return {collection: sdk.list()}
    if action in ("create", "register"):
        ident = payload.get("name", "")
        if resource is not None:
            require(resource == ident, "resource name must match config/spec name")
        name(ident)
        return _wire(getattr(sdk, action)(payload))
    if action in ("pin", "inspect"):
        if resource is not None:
            require(resource == "new", "resource name must match config/spec name")
        return getattr(sdk, action)(payload)
    ident = name(resource or "")
    if action == "get":
        return _wire(sdk.get(ident))
    if action == "mutate":
        return sdk.mutate(ident, payload)
    if action == "snapshot":
        require(
            "target" in payload
            and set(payload) <= {"target", "knowledge_at", "event_before"},
            "invalid snapshot request",
        )
        return sdk.snapshot(ident, **payload)
    require(not payload, "operation does not accept a body")
    if action == "test":
        return sdk.test(ident)
    return getattr(sdk, action)(ident)


def _context(args: argparse.Namespace, out: term.Output) -> dict:
    """`gf context create|append|label|show`, over the live context SDK methods."""
    client = client_for(args)
    ident = name(args.name)
    if args.action == "show":
        return client.contexts.get(ident).to_dict()
    if args.action == "create":
        payload = read_object(args.file)
        require(payload.get("name") == ident, "the config's name differs from NAME")
        result = client.contexts.create(payload).to_dict()
        out.status("Created", f"context {ident}")
        return result
    table = read_data(args.file, Path.cwd())
    require(table.num_rows > 0, "the file has no rows")
    write_id = args.write_id or uuid.uuid4().hex
    require(args.id in table.column_names, f"column `{args.id}` is not in the file")
    if args.action == "label":
        require(
            args.target in table.column_names,
            f"column `{args.target}` is not in the file",
        )
        result = client.contexts.label(
            ident, table, id=args.id, column=args.target, idempotency_key=write_id
        )
        out.status("Labeled", f"{term.plural(table.num_rows, 'row')} in {ident}")
    else:
        result = client.contexts.append(
            ident, table, id=args.id, idempotency_key=write_id
        )
        out.status("Appended", f"{term.plural(table.num_rows, 'row')} to {ident}")
    return _wire(result) if not isinstance(result, dict) else result


def _model(args: argparse.Namespace, out: term.Output) -> dict:
    """`gf model list|show|pull`, over the model registry."""
    registry = model_registry()
    if args.action == "list":
        models = registry.catalog(task=args.task)
        return {"models": [_model_info(registry, info) for info in models]}
    ref = registry.canonical(args.model)
    if args.action == "show":
        info = next(m for m in registry.catalog() if ref.startswith(m.id))
        return _model_info(registry, info) | {"id": ref}
    if registry.is_cached(ref):
        out.status("Cached", ref)
        return {"id": ref, "cached": True}
    out.status("Pulling", ref)
    path = registry.pull(ref, progress=out.debug)
    out.status("Pulled", f"{ref} to {path}")
    return {"id": ref, "cached": True, "path": str(path)}


def _model_info(registry: Any, info: Any) -> dict:
    limits = registry.limits(info.id)
    return {
        "id": info.id,
        "task": info.task,
        "limits": {
            "context_rows": limits.context_rows,
            "queries": limits.queries,
            "inputs": limits.inputs,
        },
        "cached": registry.is_cached(info.id),
    }


def model_registry() -> Any:
    """The model registry: session F's module, or the fallback over the catalog."""
    return _CatalogRegistry()


@dataclass(frozen=True)
class ModelInfo:
    """One catalog model as `gf model list` shows it."""

    id: str
    task: str


@dataclass(frozen=True)
class Limits:
    """A model's per-call caps."""

    context_rows: int
    queries: int
    inputs: int


class _CatalogRegistry:
    """The registry interface over `generalforecast.models.catalog`."""

    def _entries(self) -> list[Any]:
        from generalforecast.models.catalog import entries

        return list(entries())

    def catalog(self, task: str | None = None) -> list[ModelInfo]:
        """Catalog models, newest first, optionally for one task."""
        found = [
            ModelInfo(e.id, e.capability.task)
            for e in reversed(self._entries())
            if task is None or e.capability.task == task
        ]
        return found

    def default(self, task: str) -> str:
        """The model `gf init` picks for a task when none is named."""
        from generalforecast.models.catalog import DEFAULTS

        return DEFAULTS[task]

    def canonical(self, ref: str) -> str:
        """`general-forecast/<name>[@variant]`, or an error with the right id."""
        from generalforecast.cli import Error
        from generalforecast.models.catalog import ORGANIZATION, known_as, not_canonical

        base, _, variant = ref.partition("@")
        suffix = f"@{variant}" if variant else ""
        for e in self._entries():
            if base == e.id:
                return ref
            if base == e.name or base in e.aliases:
                raise Error(
                    "InvalidInput", not_canonical(ref, f"{e.id}{suffix}"), param="model"
                )
        if (known := known_as(base)) is not None:
            raise Error(
                "InvalidInput", not_canonical(ref, f"{known}{suffix}"), param="model"
            )
        close = term.suggest(base.split("/")[-1], [e.name for e in self._entries()])
        hint = (
            f"use {ORGANIZATION}/{close}"
            if close
            else "run `gf model list` to see the catalog"
        )
        raise Error(
            "NotFound", f"model `{ref}` is not in the catalog", hint=hint, param="model"
        )

    def _entry(self, ref: str) -> Any:
        base = self.canonical(ref).partition("@")[0]
        return next(e for e in self._entries() if e.id == base)

    def task_of(self, ref: str) -> str:
        """classification, regression or forecast."""
        return self._entry(ref).capability.task

    def limits(self, ref: str) -> Limits:
        """The model's caps from the serving policy and its capability."""
        from generalforecast.models.catalog import V1_POLICY

        capability = self._entry(ref).capability
        return Limits(
            min(
                capability.context_rows or V1_POLICY.context_rows,
                V1_POLICY.context_rows,
            ),
            V1_POLICY.query_rows,
            min(capability.context_features or V1_POLICY.features, V1_POLICY.features),
        )

    def is_cached(self, ref: str) -> bool:
        """Whether the weights are in the local cache."""
        from generalforecast.models.artifacts import ArtifactError, checkpoint

        try:
            checkpoint(self._entry(ref), allow_downloads=False)
        except ArtifactError:
            return False
        return True

    def pull(self, ref: str, *, progress: Any = None) -> Path:
        """Downloads the weights and returns the cache directory."""
        from generalforecast.models.artifacts import checkpoint

        return checkpoint(self._entry(ref), allow_downloads=True)
