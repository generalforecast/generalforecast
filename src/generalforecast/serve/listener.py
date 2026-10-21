"""The `gf serve` HTTP listener, command line and graceful shutdown."""

import argparse
import ipaddress
import json
import os
import signal
import socket
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Self

from generalforecast.deployments.boundary import MAX_BYTES
from generalforecast.deployments.contracts import ServiceError, require
from generalforecast.packages.validation import PackageError

if TYPE_CHECKING:
    from generalforecast.serve.app import Application

API_KEY_ENV = "GF_API_KEY"
DEFAULT_PORT = 8000


def is_loopback(host: str) -> bool:
    """Whether a bind address only accepts connections from this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def public_warning(host: str, port: int) -> str:
    """The warning printed when anyone on the network can run inference."""
    return (
        f"gf serve: warning: listening on {host}:{port} without an API key; "
        "anyone who can reach this address can run inference. "
        f"Pass --api-key or set {API_KEY_ENV}."
    )


def _respond(status: int, value: Any) -> Any:
    from generalforecast.serve.app import respond

    return respond(status, value)


class HTTPServer(ThreadingHTTPServer):
    """Bounded concurrent connections; in-flight requests finish on close."""

    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        address: tuple[str, int],
        application: "Application",
        *,
        connections: int = 64,
        read_timeout: float = 30.0,
    ) -> None:
        """Binds one listener for the application."""
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        self.application = application
        self.admissions = threading.BoundedSemaphore(connections)
        self.read_timeout = read_timeout
        super().__init__(address, Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        """Refuses excess connections instead of spawning unbounded threads."""
        if not self.admissions.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.admissions.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        """Releases admission after connection cleanup."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.admissions.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Never logs request targets, payloads or exception values."""


class Handler(BaseHTTPRequestHandler):
    """Bounded body reads; no access logging."""

    server: HTTPServer

    def setup(self) -> None:
        """Bounds slow header and body reads."""
        super().setup()
        self.connection.settimeout(self.server.read_timeout)

    def log_message(self, format: str, *args: Any) -> None:
        """Disables access logs, which would carry request targets."""

    def _handle(self) -> None:
        self.close_connection = True
        try:
            lengths = self.headers.get_all("Content-Length", [])
            require(
                len(lengths) <= 1 and "Transfer-Encoding" not in self.headers,
                "invalid message framing",
            )
            length = lengths[0] if lengths else "0"
            require(length.isascii() and length.isdecimal(), "invalid content length")
            if int(length) > MAX_BYTES:
                response = _respond(
                    413,
                    ServiceError(
                        "CapacityExceeded",
                        f"request body exceeds {MAX_BYTES} bytes",
                        413,
                    ).body(),
                )
            else:
                body = self.rfile.read(int(length))
                require(len(body) == int(length), "truncated request body")
                response = self.server.application.handle(
                    self.command, self.path, list(self.headers.items()), body
                )
        except (ServiceError, OSError):
            response = _respond(
                400, ServiceError("ValidationError", "invalid HTTP message", 400).body()
            )
        try:
            self.send_response(response.status)
            for key, value in response.headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(response.body)
        except OSError:
            pass

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_PATCH = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle


def emit(value: dict, *, error: bool = False) -> None:
    """Writes one NDJSON event."""
    print(
        json.dumps(value, allow_nan=False, ensure_ascii=False),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


class Server:
    """Loads the target, listens, warms the models and drains on stop."""

    def __init__(
        self,
        application: "Application",
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        read_timeout: float = 30.0,
    ) -> None:
        """Binds the listener immediately so liveness answers during loading."""
        self.application = application
        self.httpd = HTTPServer((host, port), application, read_timeout=read_timeout)
        self.host, self.port = host, self.httpd.server_address[1]
        self.failure: BaseException | None = None
        self.stopped = threading.Event()
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="gf-serve-http", daemon=True
        )
        self._loader = threading.Thread(
            target=self._warm, name="gf-serve-load", daemon=True
        )

    @property
    def url(self) -> str:
        """The base URL clients connect to."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> Self:
        """Starts serving and loading in the background."""
        self._thread.start()
        self._loader.start()
        return self

    def _warm(self) -> None:
        try:
            self.application.warm()
        except BaseException as error:  # noqa: BLE001 - reported by the caller
            self.failure = error
            self.stopped.set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Waits for loading to finish, returning False on failure or timeout."""
        ready = self.application.ready.wait(timeout)
        return ready and self.failure is None

    def stop(self) -> None:
        """Stops accepting, finishes in-flight requests, then releases models."""
        self.application.draining.set()
        if self._thread.is_alive():
            self.httpd.shutdown()
            self._thread.join()
        self.httpd.server_close()
        self.application.close()
        self.stopped.set()

    def __enter__(self) -> Self:
        """Starts the server for a `with` block."""
        return self.start()

    def __exit__(self, *_args: object) -> None:
        """Stops the server at the end of a `with` block."""
        self.stop()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The `gf serve` flags, shared by `gf` and `python -m generalforecast.serve`."""
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Package directory, gf.lock file or catalog model reference",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"Require this bearer key (default: ${API_KEY_ENV}, else no key)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only cached model weights; never download",
    )
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-concurrency", type=int, default=4)


def run(
    args: argparse.Namespace,
    *,
    events: Callable[[dict], None] = emit,
    warn: Callable[[str], None] | None = None,
    install_signals: bool = True,
) -> dict:
    """Serves until SIGTERM or SIGINT, then drains and returns the final event."""
    require(1 <= len(args.host) <= 255, "invalid host")
    require(type(args.port) is int and 0 <= args.port <= 65535, "invalid port")
    key = args.api_key if args.api_key is not None else os.environ.get(API_KEY_ENV)
    key = key or None
    from generalforecast.serve.app import Application
    from generalforecast.serve.models import load

    model = load(args.target, allow_downloads=not args.offline)
    loopback = is_loopback(args.host)
    application = Application(
        [model],
        api_key=key,
        loopback=loopback,
        timeout=args.request_timeout,
        concurrency=args.max_concurrency,
    )
    server = Server(application, args.host, args.port)
    if not loopback and key is None:
        (warn or (lambda text: print(text, file=sys.stderr, flush=True)))(
            public_warning(args.host, server.port)
        )
    previous: dict[int, Any] = {}
    if install_signals:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, lambda *_: server.stopped.set())
    try:
        server.start()
        events(
            {
                "event": "serving",
                "url": server.url,
                "models": [model.name],
                "auth": "bearer" if key else "none",
            }
        )
        while not server.stopped.wait(0.2):
            if application.ready.is_set():
                events({"event": "ready", "url": server.url})
                break
        server.stopped.wait()
    finally:
        server.stop()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if server.failure is not None:
        error = server.failure
        if isinstance(error, (ServiceError, PackageError)):
            raise error
        raise ServiceError("ModelUnavailable", "model failed to load", 503)
    return {"event": "stopped", "url": server.url}


def main(argv: list[str] | None = None) -> None:
    """`python -m generalforecast.serve`."""
    parser = argparse.ArgumentParser(
        prog="gf serve", description="Serve a prediction package or catalog model."
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        emit(run(args))
    except (ServiceError, PackageError) as error:
        emit({"error": {"code": error.code, "message": str(error)}}, error=True)
        raise SystemExit(1) from None
    except OSError:
        emit(
            {"error": {"code": "ClientError", "message": "cannot listen on address"}},
            error=True,
        )
        raise SystemExit(1) from None
