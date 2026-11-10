# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The CLI's fine-tune resolver: the project's own route, asked once per model."""

import pytest

from generalforecast import cli
from generalforecast.derived_service import ServiceResolver
from generalforecast.models import derived
from generalforecast.models.catalog import ModelRefError
from generalforecast.service.contracts import ServiceError

FINE_TUNE = "acme/credit-default@ft3"
BASE = "tabular-classifier"
BASE_DIGEST = "sha256:" + "0" * 64
DIGEST = "sha256:" + "1" * 64
MADE_BLOCK = {
    "base": BASE,
    "base_digest": BASE_DIGEST,
    "digest": DIGEST,
    "bags": 4,
    "kind": "lora",
    "run": "credit-default",
    "version": 3,
    "created": 0.0,
}
FOUND = {"name": FINE_TUNE, "derived": MADE_BLOCK}
MADE = derived.Derived(FINE_TUNE, BASE, BASE_DIGEST, DIGEST, 4)


class Connection:
    """A project's connection that answers each call with `answer`."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def _run(self, operation):
        call = next(operation)
        self.calls.append(call)
        if isinstance(self.answer, Exception):
            raise self.answer
        try:
            operation.send(self.answer)
        except StopIteration as done:
            return done.value
        raise AssertionError("one call reads a fine-tune")


def test_a_fine_tune_resolves_through_its_projects_route():
    connection = Connection(FOUND)
    projects = []
    resolver = ServiceResolver(lambda project: projects.append(project) or connection)

    made = resolver.resolve(FINE_TUNE)

    assert made == MADE
    assert projects == ["acme"]
    (call,) = connection.calls
    assert (call.method, call.path) == ("GET", "/models/credit-default@ft3")


def test_each_fine_tune_is_asked_for_once():
    connection = Connection(FOUND)
    resolver = ServiceResolver(lambda project: connection)

    assert resolver.resolve(FINE_TUNE) is resolver.resolve(FINE_TUNE)
    assert len(connection.calls) == 1


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_fine_tune_the_connection_cannot_read_is_unknown(status):
    refused = ServiceError("NotFound", "no such fine-tune", status)
    resolver = ServiceResolver(lambda project: Connection(refused))

    with pytest.raises(ModelRefError) as raised:
        resolver.resolve(FINE_TUNE)
    assert raised.value.code == "UnknownModel"


def test_other_service_errors_pass_through():
    failed = ServiceError("ServiceUnavailable", "down", 503)
    resolver = ServiceResolver(lambda project: Connection(failed))

    with pytest.raises(ServiceError):
        resolver.resolve(FINE_TUNE)


def test_a_catalog_answer_is_not_a_fine_tune():
    resolver = ServiceResolver(lambda project: Connection({"name": BASE}))

    with pytest.raises(ModelRefError) as raised:
        resolver.resolve(FINE_TUNE)
    assert raised.value.code == "UnknownModel"


def test_weights_stay_on_the_plane():
    resolver = ServiceResolver(lambda project: Connection(FOUND))

    with pytest.raises(ModelRefError) as raised:
        resolver.weights(MADE, None)
    assert raised.value.code == "UnsupportedCapability"


@pytest.fixture
def no_resolver(monkeypatch):
    monkeypatch.delenv(derived.RESOLVER, raising=False)
    derived.use(None)
    yield
    derived.use(None)


def test_the_cli_installs_the_service_resolver_when_none_is_set(no_resolver):
    cli._resolve_fine_tunes()
    assert isinstance(derived.resolver(), ServiceResolver)


def test_the_cli_keeps_a_resolver_already_installed(no_resolver):
    installed = ServiceResolver(lambda project: Connection(FOUND))
    derived.use(installed)
    cli._resolve_fine_tunes()
    assert derived.resolver() is installed


@pytest.mark.parametrize("kind", ["input_adapter", "adapter"])
def test_an_input_adapter_resolves_whichever_name_the_plane_answers(kind):
    answer = {**FOUND, "derived": {**MADE_BLOCK, "kind": kind}}
    made = ServiceResolver(lambda project: Connection(answer)).resolve(FINE_TUNE)
    assert made.kind == "input_adapter"
