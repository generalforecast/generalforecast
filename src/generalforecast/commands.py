# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scoped API operations, local input adapters and non-secret CLI profiles."""

import argparse
import datetime
import json
import os
import re
import ssl
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from generalforecast.client import Client
from generalforecast.command_specs import (
    COLLECTIONS,
    IDENTITY_GROUPS,
    LEGACY_DEPLOY,
    OPERATIONS,
)
from generalforecast.contracts import DatasetImportRef
from generalforecast.deployments.boundary import JsonObject, ResourceKind
from generalforecast.deployments.contracts import ServiceError, name
from generalforecast.deployments.sdk import Snapshot
from generalforecast.packages.files import MAX_DATA_BYTES
from generalforecast.tabular import file_format, read_table, table_digest
from generalforecast.transport import ConnectionSettings, Profile


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
    """Reads bounded local tabular data relative to the selected package directory."""
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
def _legacy(args: argparse.Namespace) -> bool:
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
    if not _legacy(args):
        server = server_url()
        found = CredentialStore().resolve(server)
        if found is None:
            raise CLIError("NotLoggedIn", "not logged in", 401)
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


def upload_python_source(client: Client, deployment: str, package) -> None:
    """Uploads a verified archive and requires an exact scoped source receipt."""
    from generalforecast.packages import PythonLock

    require(isinstance(package.lock, PythonLock), "expected a Python package lock")
    archive = package.source_archive()
    upload = getattr(client, "upload_python_source", None)
    if not callable(upload):
        raise ServiceError(
            "RuntimeUnavailable", "Python source upload is unavailable", 503
        )
    result = upload(deployment, archive, source=package.lock.python.source)
    if isinstance(result, Mapping):
        result = dict(result)
    expected = {
        "receipt_version": 1,
        "kind": "python-source",
        "project": client.project,
        "deployment": deployment,
        "source": package.lock.python.source,
        "size": len(archive),
    }
    if (
        type(result) is not dict
        or result != expected
        or type(result.get("receipt_version")) is not int
        or type(result.get("size")) is not int
    ):
        raise ServiceError(
            "ProtocolError", "Python source receipt identity mismatch", 502
        )


def execute(args: argparse.Namespace) -> dict:
    """Executes authenticated commands; receipt acceptance never implies readiness."""
    if args.group == "login":
        return configured_login(args)
    client = client_for(args)
    if args.group == "datasets":
        return _dataset(client, args)
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
    if args.group == "runtimes":
        ident = name(args.name)
        require(args.action != "cleanup" or args.yes, "cleanup requires --yes")
        payload = read_object(args.file)
        # Reject inline secrets before any remote provisioning operation.
        from generalforecast.packages.validation import text

        def references(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    require(
                        not re.search(r"(?i)(token|password|secret|credential)", key)
                        or key.endswith(("_ref", "_refs")),
                        "runtime credentials require trusted refs",
                    )
                    references(item)
            elif isinstance(value, list):
                for item in value:
                    references(item)
            elif isinstance(value, str):
                text(value)

        references(payload)
        if args.action == "status":
            return client.runtimes.get(ident).to_dict()
        require(not payload, "runtime operations do not accept a body")
        return getattr(client.runtimes, args.action)(ident).to_dict()
    payload = read_object(args.file)
    if args.group in IDENTITY_GROUPS:
        return _identity(client, args, payload)
    return _collection(client, args, payload)


def _identity(client: Client, args: argparse.Namespace, payload: JsonObject) -> dict:
    """Binds CLI arguments to the SDK's allowlisted identity operations."""
    group, action, resource = args.group, args.action, args.name
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
    group, action, resource = args.group, args.action, args.name
    if group == "operator" and action in ("backup", "restore-check"):
        raise ServiceError(
            "HostAuthorityUnavailable",
            "full-store operations require independent host authority",
            501,
        )
    require(action in OPERATIONS[group], "unsupported operation")
    if group == "deploy" and action in LEGACY_DEPLOY:
        method, path, kind, ident = route(group, action, resource, payload)
        return client._call(method, path, kind, ident, payload)
    if action in ("list", "get", "results", "doctor"):
        require(not payload, "GET operations do not accept a body")
    if group == "operator":
        return client.operators.doctor()
    collection = COLLECTIONS[group][1]
    sdk = getattr(client, collection)
    if action == "list":
        return {collection: sdk.list()}
    if action in ("create", "submit", "register"):
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
    if action == "results":
        require(
            0 <= args.offset <= 2**53 - 1 and 1 <= args.limit <= 100,
            "invalid result page",
        )
        return sdk.results(ident, offset=args.offset, limit=args.limit)
    if action == "infer":
        require(payload.get("model") == ident, "model must match request")
        return sdk.infer(payload)
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
    return sdk.control(ident, action)


def _dataset(client: Client, args: argparse.Namespace) -> dict:
    ident = name(args.name)
    if args.action == "status":
        return client.datasets.get(ident).to_dict()
    if args.action == "refresh":
        return client.datasets.refresh(ident).to_dict()
    if args.action == "create":
        payload = read_object(args.file)
        require(payload.get("name") == ident, "dataset name differs from config")
        return client.datasets.create(payload).to_dict()
    if args.action == "import":
        require(
            bool(args.source) != bool(args.input),
            "provide one cloud source or local table",
        )
        if args.input:
            uploaded = upload_snapshot(client, read_data(args.input, Path.cwd()))
            source, digest = (
                "upload://" + uploaded["digest"].replace(":", "-"),
                uploaded["digest"],
            )
        else:
            source, digest = args.source, None
        spec = DatasetImportRef(
            dataset=ident,
            source=source,
            format="arrow" if args.input else args.format or "table",
            credential_ref=args.credential_ref,
            digest=digest,
        )
        return client.datasets.import_source(spec).to_dict()
    require(
        bool(args.input) and bool(args.id) and bool(args.idempotency_key),
        "writes require a local table, --id and --idempotency-key",
    )
    rows = records(read_data(args.input, Path.cwd()))
    require(bool(rows), "empty dataset writes are not supported")
    if args.action == "label":
        require(bool(args.column), "label requires --column")
    observations = []
    for row in rows:
        require(args.id in row and row[args.id] is not None, "row lacks declared id")
        item = {"key": str(row[args.id])}
        if args.action == "label":
            require(
                args.column in row and row[args.column] is not None,
                "row lacks declared label",
            )
            item["labels"] = {args.column: row[args.column]}
        else:
            item["features"] = {
                key: value
                for key, value in row.items()
                if key not in {args.id, args.column}
            }
            if args.column and row.get(args.column) is not None:
                item["labels"] = {args.column: row[args.column]}
        observations.append(item)
    mutation: Any = {
        "write_id": args.idempotency_key,
        "operation": "attach_labels" if args.action == "label" else "create_features",
        "records": observations,
    }
    return dict(client.datasets.mutate(ident, mutation))


def route(
    group: str, action: str, resource: str | None, payload: JsonObject
) -> tuple[str, str, ResourceKind, str]:
    """Resolves legacy deploy operations that have no public SDK method."""
    require(group == "deploy" and action in LEGACY_DEPLOY, "unsupported operation")
    kind = COLLECTIONS[group][0]
    if action == "apply":
        ident = payload.get("name", "")
        if resource is not None:
            require(resource == ident, "resource name must match config/spec name")
        return "POST", "/deployments", kind, name(ident)
    ident = name(resource or "")
    suffix = {"effect-status": "effects/status", "effect-wait": "effects/wait"}.get(
        action, action
    )
    return "POST", f"/deployments/{ident}/{suffix}", kind, ident
