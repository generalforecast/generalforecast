# Contributing

## Servers in a checkout

`gf serve`, `gf admin node` and `gf admin router` run gf-server and
gf-router, which exist only under gf: a wheel carries them in
`generalforecast/bin`, and nothing installs them on PATH. In a source
checkout gf runs the newer of `target/release` and `target/debug`, built
with `cargo build -p gf-server -p gf-router`, and skips a build older than
the Rust sources.

To run a particular build, set `GF_SERVER_BIN` or `GF_ROUTER_BIN` to its
path. That override is for development and tests alone.

## Checks

`uv run pytest -m "not slow" -n auto` runs the fast suite, and
`tests/test_repository.py` holds every tracked file to the repository's
conventions. A test that starts a server is marked `server` (or `router`);
one that is not cannot start one.
