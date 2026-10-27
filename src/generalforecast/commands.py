# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `gf model`, `gf context` and `gf admin` commands, and the connection plumbing.

Model commands read the catalog registry; context commands call the live
context SDK methods; admin nouns bind to the SDK's identity and collection
operations, act on hosted predictors, or run a node or router on this
machine. Every command returns its result; `generalforecast.render` writes it.
"""

import argparse
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
    ADMIN_PREDICTOR_NOUNS,
    ADMIN_SERVE_ACTIONS,
    COLLECTIONS,
    IDENTITY_GROUPS,
    NAME_FIELDS,
    admin_action,
)
from generalforecast.credentials import CONNECTION, uses_profiles
from generalforecast.credentials import SEEN as _CREDENTIAL_VALUES
from generalforecast.errors import InvalidInput
from generalforecast.service.boundary import JsonObject
from generalforecast.service.contracts import name
from generalforecast.service.sdk import Snapshot
from generalforecast.tabular import file_format, read_table, table_digest
from generalforecast.transport import ConnectionSettings, Profile

# The largest table a command reads from disk.
MAX_DATA_BYTES = 128 * 1024 * 1024


def require(condition: bool, message: str, *, hint: str | None = None) -> None:
    """Raises a local, actionable validation error unless `condition` holds."""
    if not condition:
        raise InvalidInput(message, hint=hint)


def read_object(path: str) -> dict:
    """Reads one bounded JSON/YAML object without reflecting input on failure."""
    from generalforecast.service.boundary import MAX_BYTES, json_object

    try:
        if path == "-":
            raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        else:
            with Path(path).expanduser().open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "input exceeds 1 MiB")
        return json_object(yaml.safe_load(raw))
    except (ValueError, UnicodeError, yaml.YAMLError):
        raise InvalidInput(f"`{path}` is not one JSON or YAML object") from None


def read_data(path: str, root: Path):
    """Reads a bounded local table relative to the predictor directory."""
    given = Path(path).expanduser()
    target = given if given.is_absolute() else root / given
    with target.open("rb") as stream:
        raw = stream.read(MAX_DATA_BYTES + 1)
    require(len(raw) <= MAX_DATA_BYTES, "tabular input too large")
    return read_table(raw, file_format(target.name))


def connection_for(args: argparse.Namespace) -> ConnectionSettings:
    """GF_API_KEY or the `gf login` key for GF_URL (default: the hosted service).

    `GF_PROFILE`, `GF_TOKEN` or a saved default profile resolve through the
    SDK's profile resolver instead.
    """
    from generalforecast.account import not_logged_in
    from generalforecast.credentials import ENV_KEY, CredentialStore, server_url
    from generalforecast.transport import credential

    CONNECTION.clear()
    if not uses_profiles(args):
        server = server_url()
        found = CredentialStore().resolve(server)
        if found is None:
            raise not_logged_in()
        CONNECTION.update(server=server, source=found.source)
        project = os.environ.get("GF_PROJECT") or (
            found.login.project if found.login else None
        )
        CONNECTION["project_pending"] = "" if project else "1"
        profile = Profile(
            server,
            project or "default",
            os.environ.get("GF_ENV", "production"),
            key_env=None,
        )
        env = ENV_KEY if found.source == ENV_KEY else None
        return ConnectionSettings(None, profile, env, credential(found.key, "API key"))
    settings = ConnectionSettings.resolve()
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
        raise InvalidInput("set GF_PROJECT for this API key")
    return project


def client_for(
    args: argparse.Namespace, settings: ConnectionSettings | None = None
) -> Client:
    """Uses named environment references; never persists credential values."""
    settings = connection_for(args) if settings is None else settings
    token = settings.token
    provider = {
        "auth_mode": os.environ.get("GF_AUTH_MODE", "bearer"),
        "platform_token": os.environ.get("GF_PLATFORM_TOKEN"),
        "modal_key": os.environ.get("GF_MODAL_KEY"),
        "modal_secret": os.environ.get("GF_MODAL_SECRET"),
    }
    _CREDENTIAL_VALUES.update(
        value
        for value in (
            token,
            provider["platform_token"],
            provider["modal_key"],
            provider["modal_secret"],
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
            raise InvalidInput(
                "cannot load the TLS configuration",
                hint="check GF_TLS_CA and GF_TLS_IDENTITY",
            ) from None
    if CONNECTION.get("project_pending") and token:
        return _KeyScopedClient(
            key=token,
            profile=settings.profile,
            token=token,
            timeout=args.timeout,
            tls_context=context,
            **provider,
        )
    return Client(
        profile=settings.profile,
        token=token,
        timeout=args.timeout,
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
    if noun in ADMIN_SERVE_ACTIONS:
        from generalforecast.serve import listener

        return listener.admin(noun, args, out)
    if noun in ADMIN_PREDICTOR_NOUNS:
        from generalforecast.predictor_commands import PredictorCommands

        return getattr(PredictorCommands(args, out), noun)()
    return _api(args, out)


def _api(args: argparse.Namespace, out: term.Output) -> dict:
    """An API-backed admin action, its NAME and `--file` bound to the request."""
    noun, spec = args.noun, admin_action(args.noun, args.action)
    payload: JsonObject = read_object(args.file) if spec.body else {}
    resource = name(args.name) if spec.target is not None else None
    if resource is not None:
        field = NAME_FIELDS[noun]
        require(
            payload.get(field, resource) == resource,
            f"the file's `{field}` is not `{resource}`",
            hint=f"drop `{field}` from the file, or pass its value as NAME",
        )
        payload[field] = resource
    if noun == "audit":
        require(args.after >= 0, "--after must be 0 or more")
        require(1 <= args.limit <= 100, "--limit must be from 1 to 100")
        payload = {"after": args.after, "limit": args.limit}
    if spec.destructive:
        words = [spec.name.capitalize(), noun, resource]
        question = " ".join(word for word in words if word) + "?"
        term.Prompter(out, yes=args.yes).confirm(question)
    client = client_for(args)
    if noun in IDENTITY_GROUPS:
        return client.identity.manage(f"{IDENTITY_GROUPS[noun]}.{spec.name}", payload)
    if noun == "operator":
        return client._call("GET", "/operators/doctor", "operator", "doctor")
    collection = COLLECTIONS[noun][1]
    sdk = getattr(client, collection)
    if spec.name == "list":
        return {collection: sdk.list()}
    return _wire(getattr(sdk, spec.name)(payload))


def _wire(value: Any) -> dict:
    """Returns the exact wire object behind an SDK snapshot DTO."""
    return value.to_dict() if isinstance(value, Snapshot) else value


def _context(args: argparse.Namespace, out: term.Output) -> dict:
    """`gf context create|append|label|show`, over the live context SDK methods."""
    ident = name(args.name)
    if args.action == "show":
        return client_for(args).contexts.get(ident).to_dict()
    if args.action == "create":
        payload = read_object(args.file)
        require(payload.get("name") == ident, "the config's name differs from NAME")
        result = client_for(args).contexts.create(payload).to_dict()
        out.status("Created", f"context {ident}")
        return result
    table = read_data(args.file, Path.cwd())
    require(table.num_rows > 0, "the file has no rows")
    write_id = args.write_id or uuid.uuid4().hex
    require(args.id in table.column_names, f"column `{args.id}` is not in the file")
    client = client_for(args)
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
        return {
            "models": [_model_info(registry, info.id, info.task) for info in models]
        }
    ref = registry.canonical(args.model)
    if args.action == "show":
        return _model_info(registry, ref, registry.task_of(ref))
    if registry.is_cached(ref):
        out.status("Cached", ref)
        return {"id": ref, "cached": True}
    path = registry.pull(ref, progress=out.status)
    out.status("Pulled", f"{ref} to {path}")
    return {"id": ref, "cached": True, "path": str(path)}


def _model_info(registry: Any, ref: str, task: str) -> dict:
    limits = registry.limits(ref)
    return {
        "id": ref,
        "task": task,
        "limits": {
            "context_rows": limits.context_rows,
            "queries": limits.queries,
            "inputs": limits.inputs,
        },
        "cached": registry.is_cached(ref),
    }


def model_registry() -> Any:
    """The model registry over the catalog."""
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
        return [
            ModelInfo(e.id, e.capability.task)
            for e in reversed(self._entries())
            if task is None or e.capability.task == task
        ]

    def default(self, task: str) -> str:
        """The model `gf init` picks for a task when none is named."""
        from generalforecast.models.catalog import DEFAULTS

        return DEFAULTS[task]

    def canonical(self, ref: str) -> str:
        """`general-forecast/<name>[@variant]`, or an error with the right id."""
        from generalforecast.errors import NotFound
        from generalforecast.models.catalog import ORGANIZATION, known_as, not_canonical

        base, _, variant = ref.partition("@")
        suffix = f"@{variant}" if variant else ""
        for e in self._entries():
            if base == e.id:
                return ref
            if base == e.name or base in e.aliases:
                raise InvalidInput(not_canonical(ref, f"{e.id}{suffix}"), param="model")
        if (known := known_as(base)) is not None:
            raise InvalidInput(not_canonical(ref, f"{known}{suffix}"), param="model")
        close = term.suggest(base.split("/")[-1], [e.name for e in self._entries()])
        hint = (
            f"use {ORGANIZATION}/{close}"
            if close
            else "run `gf model list` to see the catalog"
        )
        raise NotFound(f"model `{ref}` is not in the catalog", hint=hint, param="model")

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

    def pull(self, ref: str, *, progress: Any) -> Path:
        """Downloads the weights, reporting the download through `progress`."""
        from generalforecast.models.artifacts import checkpoint

        return checkpoint(self._entry(ref), allow_downloads=True, progress=progress)
