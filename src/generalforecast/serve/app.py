"""Transport-independent Models API routes, envelopes and bearer-key check."""

import hmac
import ipaddress
import json
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from generalforecast.deployments.boundary import (
    JsonObject,
    Request,
    Resource,
    Scope,
    failure,
    parse_body,
    success,
)
from generalforecast.deployments.contracts import ServiceError, name, require
from generalforecast.models.catalog import resource_name
from generalforecast.packages.validation import PackageError
from generalforecast.serve.models import Served

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class Response:
    """One JSON response."""

    status: int
    body: bytes
    headers: list[tuple[str, str]]


@dataclass(frozen=True)
class Route:
    """A resolved Models API route."""

    project: str
    action: str
    resource: Resource


def respond(status: int, value: Any) -> Response:
    """Encodes JSON with the hosted API's no-store, no-sniff headers."""
    return Response(
        status,
        json.dumps(value, allow_nan=False, separators=(",", ":")).encode(),
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
        ],
    )


def not_found() -> ServiceError:
    """The hosted API's unknown-route error."""
    return ServiceError("NotFound", "route not found", 404)


def resolve(method: str, path: str) -> Route:
    """Maps only the hosted Models API paths; everything else is NotFound."""
    parts = path.split("/")
    if parts[:3] != ["", "v1alpha1", "projects"] or len(parts) < 5:
        raise not_found()
    if any(not part for part in parts[3:]):
        raise not_found()
    project, tail = name(parts[3]), parts[4:]
    if method == "GET" and tail == ["models"]:
        return Route(project, "models.list", Resource("model", "models"))
    if method == "GET" and len(tail) == 2 and tail[0] == "models":
        return Route(project, "models.get", Resource("model", name(tail[1])))
    if method == "POST" and len(tail) == 3 and tail[::2] == ["models", "infer"]:
        return Route(project, "models.infer", Resource("model", name(tail[1])))
    if method == "POST" and tail == ["predictions"]:
        return Route(project, "predictions.infer", Resource("model", "predictions"))
    raise not_found()


class Application:
    """Serves loaded models; the models are shared, immutable and loaded once."""

    def __init__(
        self,
        models: list[Served],
        *,
        api_key: str | None = None,
        loopback: bool = True,
        timeout: float = 60.0,
        concurrency: int = 4,
    ) -> None:
        """Binds the served models, optional bearer key and execution bounds."""
        require(api_key is None or bool(api_key), "API key must be nonempty")
        require(0 < timeout <= 3600, "request timeout must be within (0, 3600]")
        require(1 <= concurrency <= 256, "concurrency must be within 1..256")
        names = [model.name for model in models]
        require(len(set(names)) == len(names), "served model names must be unique")
        self.models = {model.name: model for model in models}
        self._key = None if api_key is None else api_key.encode()
        self.loopback = loopback
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(concurrency, thread_name_prefix="gf-serve")
        self.ready = threading.Event()
        self.draining = threading.Event()

    def warm(self) -> None:
        """Loads every model, then reports ready."""
        for model in self.models.values():
            model.warm()
        self.ready.set()

    def close(self) -> None:
        """Stops readiness and waits for accepted inference to finish."""
        self.draining.set()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def handle(
        self, method: str, target: str, headers: list[tuple[str, str]], body: bytes
    ) -> Response:
        """Answers one request with the hosted API's envelopes and error codes."""
        request = None
        try:
            h: dict[str, str] = {}
            for key, value in headers:
                key = key.lower()
                require(key not in h, "duplicate HTTP header")
                require(
                    len(value) <= 8192 and "\r" not in value and "\n" not in value,
                    "invalid HTTP header",
                )
                h[key] = value
            require(
                len(headers) <= 64 and len(target) <= 8192, "request headers too large"
            )
            if method not in {"GET", "POST"}:
                raise ServiceError(
                    "MethodNotAllowed", "only GET and POST are supported", 405
                )
            require(method != "GET" or not body, "GET bodies are forbidden")
            if target in ("/livez", "/readyz"):
                if target == "/livez":
                    return respond(200, {"live": True})
                ready = self.ready.is_set() and not self.draining.is_set()
                return respond(200 if ready else 503, {"ready": ready})
            self._origin(h)
            parsed = urlsplit(target)
            require(
                not parsed.scheme
                and not parsed.netloc
                and not parsed.fragment
                and "%" not in parsed.path
                and "\\" not in parsed.path,
                "invalid request target",
            )
            require(not parsed.query, "query parameters are not supported")
            route = resolve(method, parsed.path)
            request_id = h.get("x-gf-request-id", "r-" + uuid.uuid4().hex)
            require(
                len(request_id) <= 128
                and request_id.isascii()
                and all(33 <= ord(c) <= 126 for c in request_id),
                "invalid request id",
            )
            if "x-gf-project" in h:
                require(
                    h["x-gf-project"] == route.project, "project header/path mismatch"
                )
            if method == "GET":
                request = Request(Scope(route.project, request_id), route.resource, {})
            else:
                require(
                    h.get("content-type", "").split(";")[0].strip().lower()
                    == "application/json",
                    "application/json required",
                )
                request = Request.parse(parse_body(body))
                require(
                    request.scope.project == route.project,
                    "project path/envelope mismatch",
                )
                if "x-gf-request-id" in h:
                    require(
                        request.scope.request_id == request_id, "request id mismatch"
                    )
                require(
                    request.resource == route.resource,
                    "resource route/envelope mismatch",
                )
            self._authenticate(h)
            result = self._dispatch(route, request)
            return respond(200, success(request, result))
        except PackageError as error:
            status = (
                409
                if error.code == "Conflict"
                else 503
                if error.code == "ModelUnavailable"
                else 422
            )
            safe = ServiceError(error.code, "package operation rejected", status)
            return respond(status, failure(request, safe) if request else safe.body())
        except ServiceError as error:
            return respond(
                error.status, failure(request, error) if request else error.body()
            )
        except (ValueError, UnicodeError):
            error = ServiceError("ValidationError", "invalid request", 422)
            return respond(422, failure(request, error) if request else error.body())
        except Exception:  # noqa: BLE001 - never serialize payload or model exceptions
            error = ServiceError("InternalError", "request failed safely", 500)
            return respond(500, failure(request, error) if request else error.body())

    def _origin(self, h: dict[str, str]) -> None:
        """On loopback, refuses DNS-rebound hosts and cross-origin browser calls."""
        if not self.loopback:
            return
        host = h.get("host", "")
        parsed = urlsplit("//" + host)
        try:
            hostname = parsed.hostname or ""
            local = (
                hostname in LOOPBACK_HOSTS or ipaddress.ip_address(hostname).is_loopback
            )
        except ValueError:
            local = False
        require(local and not parsed.username and not parsed.path, "untrusted Host")
        if "origin" in h:
            require(h["origin"] == "http://" + host, "untrusted Origin")

    def _authenticate(self, h: dict[str, str]) -> None:
        if self._key is None:
            return
        header = h.get("authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme != "Bearer" or not presented or " " in presented:
            raise ServiceError("Unauthorized", "application bearer required", 401)
        if not hmac.compare_digest(presented.encode(), self._key):
            raise ServiceError("Unauthorized", "invalid API key", 401)

    def _dispatch(self, route: Route, request: Request) -> JsonObject:
        if not self.ready.is_set() or self.draining.is_set():
            raise ServiceError("NotReady", "models are not ready", 503)
        scope, payload = request.scope, request.payload
        if route.action == "models.list":
            return {"models": [model.describe(scope) for model in self.models.values()]}
        if route.action == "predictions.infer":
            model = payload.get("model")
            served = self._served(model if type(model) is str else "")
            return self._run(lambda: served.predict(scope, payload))
        served = self._served(route.resource.name)
        if route.action == "models.get":
            return served.describe(scope)
        require(
            resource_name(payload.get("model")) == resource_name(route.resource.name),
            "model route mismatch",
        )
        return self._run(lambda: served.infer(scope, payload))

    def _served(self, model: str) -> Served:
        served = self.models.get(model) or self.models.get(resource_name(model))
        if served is None:
            raise ServiceError("ModelNotFound", "model is not served here", 404)
        return served

    def _run(self, call: Callable[[], JsonObject]) -> JsonObject:
        """Runs inference on the bounded pool and stops waiting at the deadline."""
        try:
            future = self.pool.submit(call)
        except RuntimeError:
            raise ServiceError("NotReady", "server is shutting down", 503) from None
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout:
            raise ServiceError("RequestTimeout", "inference timed out", 504) from None


__all__ = ["Application", "Response", "resolve", "respond"]
