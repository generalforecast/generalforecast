"""Immutable package publication and isolated, bounded remote patch sessions."""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from generalforecast.commands import (
    client_for,
    read_data,
    read_object,
    records,
    require,
    upload_python_source,
    upload_snapshot,
)
from generalforecast.contracts import (
    DevSession,
    Environment,
    PatchRequest,
    PredictionResult,
    PromoteRequest,
    Revision,
)
from generalforecast.deployments.contracts import ServiceError, name, number
from generalforecast.packages import (
    Package,
    PackageError,
    PackageLock,
    PythonLock,
    PythonPackageSpec,
)
from generalforecast.packages.files import (
    MAX_DATA_BYTES,
    MAX_MANIFEST_BYTES,
    read,
    relative,
)
from generalforecast.python_packages.compiler import MAX_ARCHIVE, MAX_FILE


class PackageCommands:
    """Uses the SDK transport and foundation values, never a parallel control plane."""

    def __init__(self, args: argparse.Namespace):
        """Binds one explicit endpoint/project/profile for the complete operation."""
        self.args = args
        self.client = client_for(args)
        self.timeout = number(getattr(args, "timeout", 40), 0.01, 300, "timeout")
        if args.group == "push" and args.watch:
            self.validate_watch()

    def call(
        self,
        method: str,
        deployment: str,
        suffix: str = "",
        payload: dict | None = None,
    ) -> dict:
        """Calls only the canonical scoped deployment routes."""
        return self.client._call(
            method,
            f"/deployments/{name(deployment)}" + suffix,
            "deployment",
            deployment,
            payload,
        )

    def checked(self, *, write: bool = True) -> Package:
        """Compiles offline; upgrades and online qualification are always explicit."""
        args = self.args
        data = read_data(args.data, Path(args.path)) if args.data else None
        return Package.load(args.path).check(
            data=data,
            locked=getattr(args, "locked", False),
            frozen=getattr(args, "frozen", False),
            write=write,
        )

    def validate_watch(self) -> None:
        """Rejects invalid session limits before any remote publication or launch."""
        number(self.args.interval, 0.05, 60, "interval")
        number(self.args.watch_seconds, 0.05, 86400, "watch_seconds")
        number(self.args.ttl, 1, 86400, "ttl")

    def run(self) -> dict:
        """Dispatches package operations with exact readiness and CAS identities."""
        args = self.args
        if args.group == "push":
            require(not args.resume or args.watch, "--resume requires --watch")
            if args.resume:
                package = self.checked(write=False)
                session = DevSession.from_dict(
                    self.call(
                        "GET", package.spec.name, f"/dev-sessions/{name(args.resume)}"
                    )
                )
                require(session.id == args.resume, "resume session identity mismatch")
                return self.watch(package, session)
            package = self.checked()
            revision, environment = self.push(package)
            if args.watch:
                session = DevSession.from_dict(
                    self.call(
                        "POST",
                        package.spec.name,
                        "/dev-sessions",
                        {
                            "base_revision": revision.number,
                            "runtime": args.runtime,
                            "ttl_seconds": args.ttl,
                        },
                    )
                )
                return self.watch(package, session)
            return {
                "revision": revision.to_dict(),
                "environment": environment.to_dict(),
            }
        if args.group == "predict":
            return self.predict()
        deployment = name(args.name)
        env_path = f"/environments/{name(args.env)}"
        if args.group == "delete":
            return self.call("POST", deployment, "/delete", {})
        if args.group == "logs":
            return self.call("GET", deployment, env_path + "/logs")
        environment = Environment.from_dict(self.call("GET", deployment, env_path))
        require(
            (environment.deployment, environment.name) == (deployment, args.env),
            "environment response identity mismatch",
        )
        if args.group == "status":
            return {
                "environment": environment.to_dict(),
                "deployment": self.call("GET", deployment),
            }
        if args.group == "diff":
            require(environment.revision is not None, "environment has no revision")
            remote = self.call("GET", deployment, f"/revisions/{environment.revision}")
            local = self.checked(write=False)
            assert local.lock is not None
            require(local.spec.name == deployment, "package deployment differs")
            lock = PackageLock.from_dict(remote["lock"])
            return {
                "local": local.lock.package_digest,
                "remote": lock.package_digest,
                "changed": _changes(local.lock.to_dict(), lock.to_dict()),
            }
        expected = args.expected_generation
        request = PromoteRequest(
            args.revision, environment.generation if expected is None else expected
        )
        result = Environment.from_dict(
            self.call(
                "POST", deployment, env_path + "/" + args.group, request.to_dict()
            )
        )
        require(
            (
                result.deployment,
                result.name,
                result.runtime,
                result.revision,
                result.generation,
            )
            == (
                deployment,
                args.env,
                environment.runtime,
                request.revision,
                request.expected_generation + 1,
            ),
            "promotion response identity mismatch",
        )
        return self.wait_environment(result).to_dict()

    def snapshot(self, package: Package) -> str | None:
        """Uploads scoped source or full snapshot content, never selected context."""
        assert package.lock is not None
        if isinstance(package.lock, PythonLock):
            upload_python_source(self.client, package.spec.name, package)
            return None
        if package.spec.context.mode != "snapshot":
            return None
        result = upload_snapshot(self.client, package.snapshot_table())
        require(
            result["digest"] == package.lock.data_digest,
            "uploaded snapshot differs from lock",
        )
        return result["digest"]

    def push(self, package: Package) -> tuple[Revision, Environment]:
        """Creates immutable content, CAS promotes once and waits for actual readiness."""
        assert package.lock is not None
        deployment = package.spec.name
        runtime, env = name(self.args.runtime), name(self.args.env)
        try:
            self.call("GET", deployment)
        except ServiceError as error:
            if error.status != 404:
                raise
            self.client._call(
                "POST", "/deployments", "deployment", deployment, {"name": deployment}
            )
        snapshot = self.snapshot(package)
        revision = Revision.from_dict(
            self.call(
                "POST",
                deployment,
                "/revisions",
                {"lock": package.lock.to_dict(), "snapshot": snapshot},
            )
        )
        require(
            revision.deployment == deployment
            and revision.package_digest == package.lock.package_digest
            and revision.model == package.lock.identity,
            "revision response identity mismatch",
        )
        path = f"/environments/{env}"
        try:
            current = Environment.from_dict(self.call("GET", deployment, path))
        except ServiceError as error:
            if error.status != 404:
                raise
            current = Environment.from_dict(
                self.call(
                    "POST",
                    deployment,
                    "/environments",
                    {"name": env, "runtime": runtime},
                )
            )
        require(
            (current.deployment, current.name, current.runtime)
            == (deployment, env, runtime),
            "environment runtime differs; select a matching environment",
        )
        if current.revision == revision.number and current.status == "ready":
            return revision, current
        request = PromoteRequest(revision.number, current.generation)
        promoted = Environment.from_dict(
            self.call("POST", deployment, path + "/promote", request.to_dict())
        )
        require(
            (
                promoted.deployment,
                promoted.name,
                promoted.runtime,
                promoted.revision,
                promoted.generation,
            )
            == (deployment, env, runtime, revision.number, current.generation + 1),
            "promotion response identity mismatch",
        )
        return revision, self.wait_environment(promoted)

    def wait_environment(self, environment: Environment) -> Environment:
        """Polls the exact pointer and rejects failed states or concurrent promotion."""
        deadline = time.monotonic() + self.timeout
        current = environment
        while True:
            require(
                current.deployment == environment.deployment
                and current.name == environment.name
                and current.runtime == environment.runtime,
                "environment response identity mismatch",
            )
            if (
                current.revision != environment.revision
                or current.generation != environment.generation
            ):
                raise ServiceError("Conflict", "environment changed while waiting", 409)
            if current.status == "ready":
                return current
            if current.status in {"failed", "stopped"}:
                raise ServiceError(
                    "ReadinessFailed", "environment failed readiness", 503
                )
            if time.monotonic() >= deadline:
                raise ServiceError("Timeout", "readiness timed out", 504)
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            current = Environment.from_dict(
                self.call(
                    "GET", environment.deployment, f"/environments/{environment.name}"
                )
            )

    def predict(self) -> dict:
        """Reads JSON requests or local tabular queries and preserves result metadata."""
        args = self.args
        deployment = name(args.name or Package.load(args.path).spec.name)
        if args.local:
            require(
                self.client._transport.url.startswith(
                    ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
                ),
                "--local requires a loopback endpoint",
            )
        input_path = (
            Path(args.input)
            if Path(args.input).is_absolute()
            else Path(args.path) / args.input
        )
        payload = None
        if input_path.suffix in {".json", ".yaml", ".yml"}:
            try:
                payload = read_object(str(input_path))
            except ServiceError:
                if input_path.suffix != ".json":
                    raise
        if payload is None:
            payload = {"queries": records(read_data(args.input, Path(args.path)))}
        require(
            "queries" in payload
            and set(payload) <= {"queries", "context", "target", "features", "options"},
            "invalid prediction request",
        )
        if args.context:
            payload["context"] = records(read_data(args.context, Path(args.path)))
        if args.target:
            payload["target"] = args.target
        if args.features:
            payload["features"] = args.features
        path = (
            f"/dev-sessions/{name(args.session)}"
            if args.session
            else f"/environments/{name(args.env)}"
        )
        result = PredictionResult.from_dict(
            self.call("POST", deployment, path + "/predict", payload)
        )
        require(
            result.deployment == deployment, "prediction deployment identity mismatch"
        )
        require(
            result.environment == (None if args.session else args.env)
            and result.revision is not None,
            "prediction environment identity mismatch",
        )
        output = getattr(args, "output", None)
        if output and output != "json":
            target = (
                Path(output) if Path(output).is_absolute() else Path(args.path) / output
            )
            require(not target.exists(), "output exists; refusing overwrite")
            table = result.to_arrow()
            if target.suffix == ".parquet":
                import pyarrow.parquet as pq

                pq.write_table(table, target)
            elif target.suffix in {".arrow", ".ipc"}:
                from generalforecast.tabular import arrow_bytes

                target.write_bytes(arrow_bytes(table))
            elif target.suffix == ".csv":
                from pyarrow import csv

                csv.write_csv(table, target)
            else:
                raise ServiceError(
                    "UnsupportedCapability",
                    "output requires csv, parquet or arrow",
                    422,
                )
        return result.to_dict()

    def watch(self, package: Package, session: DevSession) -> dict:
        """Runs a dev-only bounded patch loop and cleans up on Ctrl-C or normal exit."""
        from generalforecast.cli import emit

        args = self.args
        self.validate_watch()
        require(session.deployment == package.spec.name, "session deployment differs")
        watcher = _Watcher(self, package, session)
        deadline = time.monotonic() + args.watch_seconds
        stop = True
        try:
            watcher.session = watcher.ready(session)
            emit(
                {
                    "event": "watching",
                    "session": watcher.session.to_dict(),
                    "resume": session.id,
                }
            )
            while time.monotonic() < deadline:
                watcher.tick()
                time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        except ServiceError as error:
            if error.code in {"TransportError", "Conflict"}:
                stop = False
                emit(
                    {
                        "event": "disconnected"
                        if error.code == "TransportError"
                        else "conflict",
                        "session": session.id,
                        "resume": session.id,
                    },
                    error=True,
                )
            raise
        finally:
            active_error = sys.exc_info()[0] is not None
            if stop:
                try:
                    stopped = DevSession.from_dict(
                        self.call(
                            "POST",
                            session.deployment,
                            f"/dev-sessions/{session.id}/stop",
                            {},
                        )
                    )
                    require(
                        stopped.status == "stopped"
                        and (stopped.id, stopped.deployment, stopped.base_revision)
                        == (session.id, session.deployment, session.base_revision),
                        "session cleanup was not confirmed",
                    )
                    emit({"event": "stopped", "session": stopped.to_dict()})
                    watcher.session = stopped
                except (ServiceError, PackageError, OSError):
                    emit(
                        {
                            "event": "cleanup_failed",
                            "session": session.id,
                            "action": "retry stop before expiry",
                        },
                        error=True,
                    )
                    if not active_error:
                        raise ServiceError(
                            "CleanupFailed", "session stop was not confirmed", 503
                        ) from None
        return {"session": watcher.session.to_dict(), "watch_complete": True}


class _Watcher:
    def __init__(
        self, commands: PackageCommands, package: Package, session: DevSession
    ):
        self.commands, self.package, self.session = commands, package, session
        self.fingerprint = None
        self.failures = 0
        self.pending: Package | None = None

    def ready(self, expected: DevSession) -> DevSession:
        deadline = time.monotonic() + self.commands.timeout
        current = expected
        while True:
            if (
                current.id,
                current.deployment,
                current.base_revision,
                current.generation,
                current.package_digest,
            ) != (
                expected.id,
                expected.deployment,
                expected.base_revision,
                expected.generation,
                expected.package_digest,
            ):
                raise ServiceError("Conflict", "session changed while waiting", 409)
            if current.status == "ready" and current.expires_at > time.time():
                return current
            if (
                current.status in {"failed", "stopped"}
                or current.expires_at <= time.time()
            ):
                raise ServiceError("ReadinessFailed", "session is not ready", 503)
            if time.monotonic() >= deadline:
                raise ServiceError("Timeout", "session readiness timed out", 504)
            time.sleep(0.1)
            current = self.get()

    def get(self) -> DevSession:
        return DevSession.from_dict(
            self.commands.call(
                "GET", self.session.deployment, f"/dev-sessions/{self.session.id}"
            )
        )

    def signature(self) -> str:
        root = Path(self.commands.args.path)
        paths = {"gf.yaml", "gf.lock"}
        if self.commands.args.data:
            paths.add(relative(self.commands.args.data))
        try:
            authored = Package.load(root)
        except PackageError:
            authored = self.package
        python_files = set()
        if isinstance(authored.spec, PythonPackageSpec):
            spec = authored.spec.python
            python_files.update(spec.code + spec.dependencies + spec.artifacts)
            paths.update(python_files)
        elif authored.spec.context.mode == "snapshot":
            assert authored.spec.context.source is not None
            paths.add(authored.spec.context.source)
        digest = hashlib.sha256()
        source_bytes = 0
        for path in sorted(paths):
            digest.update(path.encode())
            try:
                limit = (
                    MAX_FILE
                    if path in python_files
                    else MAX_MANIFEST_BYTES
                    if path in {"gf.yaml", "gf.lock"}
                    else MAX_DATA_BYTES
                )
                raw = read(root, path, limit=limit)
                if path in python_files:
                    source_bytes += len(raw)
                    require(
                        source_bytes <= MAX_ARCHIVE, "Python watch source too large"
                    )
                digest.update(raw)
            except PackageError as error:
                if error.code != "MissingFile":
                    raise
                digest.update(b"missing")
        return digest.hexdigest()

    def tick(self) -> None:
        from generalforecast.cli import emit

        try:
            fingerprint = self.signature()
            current = self.get()
            require(
                (current.id, current.deployment, current.base_revision)
                == (
                    self.session.id,
                    self.session.deployment,
                    self.session.base_revision,
                ),
                "session response identity mismatch",
            )
            if current.expires_at <= time.time() or current.status in {
                "failed",
                "stopped",
            }:
                raise ServiceError("ReadinessFailed", "session is not ready", 503)
            if (
                self.pending is not None
                and self.pending.lock is not None
                and current.generation == self.session.generation + 1
                and current.package_digest == self.pending.lock.package_digest
            ):
                self.session = current
            if (current.generation, current.package_digest) != (
                self.session.generation,
                self.session.package_digest,
            ):
                raise ServiceError(
                    "Conflict", "another writer changed this session", 409
                )
            self.session = self.ready(current)
            if (
                self.pending is not None
                and self.pending.lock is not None
                and self.pending.lock.package_digest == self.session.package_digest
            ):
                self.complete_patch()
                return
            if fingerprint == self.fingerprint:
                self.failures = 0
                return
            self.fingerprint = fingerprint
            candidate = self.commands.checked(write=False)
            assert candidate.lock is not None
            if candidate.lock.package_digest == self.session.package_digest:
                self.failures = 0
                return
            snapshot = self.commands.snapshot(candidate)
            request = PatchRequest(
                self.session.generation,
                self.session.base_revision,
                candidate.lock,
                snapshot,
            )
            emit(
                {
                    "event": "patching",
                    "session": self.session.id,
                    "generation": self.session.generation,
                    "package": candidate.lock.package_digest,
                }
            )
            self.pending = candidate
            try:
                patched = DevSession.from_dict(
                    self.commands.call(
                        "POST",
                        self.session.deployment,
                        f"/dev-sessions/{self.session.id}/patch",
                        request.to_dict(),
                    )
                )
            except ServiceError as error:
                if error.code != "TransportError":
                    self.pending = None
                    raise
                # A lost response is not a failed write; reconcile before any retry.
                patched = self.get()
                if (
                    patched.generation == self.session.generation
                    and patched.package_digest == self.session.package_digest
                ):
                    self.fingerprint = None
                    raise
                if (
                    patched.generation != self.session.generation + 1
                    or patched.package_digest != candidate.lock.package_digest
                ):
                    raise ServiceError(
                        "Conflict", "unknown patch outcome", 409
                    ) from None
            require(
                patched.generation == self.session.generation + 1
                and patched.package_digest == candidate.lock.package_digest,
                "patch response differs from requested identity",
            )
            require(
                (patched.id, patched.deployment, patched.base_revision)
                == (
                    self.session.id,
                    self.session.deployment,
                    self.session.base_revision,
                ),
                "patch response session identity mismatch",
            )
            self.session = patched
            self.session = self.ready(patched)
            self.complete_patch()
        except PackageError as error:
            emit(
                {
                    "event": "patch_rejected",
                    "code": error.code,
                    "session": self.session.id,
                    "serving_package": self.session.package_digest,
                },
                error=True,
            )
        except OSError:
            emit(
                {
                    "event": "patch_rejected",
                    "code": "InputUnavailable",
                    "session": self.session.id,
                    "serving_package": self.session.package_digest,
                },
                error=True,
            )
        except ServiceError as error:
            if error.code == "TransportError" and self.failures < 3:
                self.failures += 1
                self.fingerprint = None
                emit(
                    {
                        "event": "reconnecting",
                        "session": self.session.id,
                        "attempt": self.failures,
                    },
                    error=True,
                )
                return
            if error.status == 422:
                emit(
                    {
                        "event": "patch_rejected",
                        "code": error.code,
                        "session": self.session.id,
                        "serving_package": self.session.package_digest,
                    },
                    error=True,
                )
                return
            raise

    def complete_patch(self) -> None:
        """Records a confirmed patch, then rechecks inputs on the next tick."""
        from generalforecast.cli import emit

        assert self.pending is not None and self.pending.lock is not None
        self.pending.lock.write(self.pending.root)
        self.package = self.pending
        self.pending = None
        self.fingerprint = None
        self.failures = 0
        emit({"event": "patched", "session": self.session.to_dict()})


def _changes(local: dict, remote: dict, prefix: str = "") -> list[str]:
    result = []
    for key in sorted(local.keys() | remote.keys()):
        path = prefix + key
        left, right = local.get(key), remote.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result.extend(_changes(left, right, path + "."))
        elif left != right:
            result.append(path)
    return result
