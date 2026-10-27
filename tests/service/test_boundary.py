# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `gf/v1` envelope and its bounded JSON: wire conformance, not journeys."""

import copy
import json
from dataclasses import asdict

import pytest

from generalforecast.service.boundary import (
    MAX_BYTES,
    MAX_DEPTH,
    MAX_INTEGER,
    MAX_NODES,
    Artifact,
    Capability,
    Request,
    Resource,
    Scope,
    failure,
    json_object,
    success,
)
from generalforecast.service.contracts import ServiceError
from generalforecast.service.interfaces import (
    AcceptedWrite,
    ComponentVersions,
    Mutation,
)

ENVELOPE = {
    "api_version": "gf/v1",
    "project": "local",
    "request_id": "request-1",
    "resource": {"kind": "context", "name": "observations"},
}
SUITE = {
    "request": {
        **ENVELOPE,
        "payload": {
            "write_id": "write-1",
            "operation": "create_features",
            "records": [{"key": "row-1", "features": {"amount": 2}}],
        },
    },
    "result": {
        **ENVELOPE,
        "result": {
            "project": "local",
            "context": "observations",
            "write_id": "write-1",
            "operation": "create_features",
            "sequence": 1,
            "accepted_at": 1700000000.0,
            "durably_accepted": True,
            "changes": [{"key": "row-1", "feature_version": 1, "label_versions": {}}],
            "retry_valid_until": None,
        },
    },
    "failure": {
        **ENVELOPE,
        "error": {
            "code": "Conflict",
            "message": "expected version is stale; nothing changed",
        },
    },
    "capability": {
        "name": "custom-execution",
        "state": "unconfigured",
        "code": "IsolationUnavailable",
        "reason": "a verified isolated OCI runtime is required",
    },
}


@pytest.fixture
def fixture():
    return copy.deepcopy(SUITE)


def test_envelope_conformance(fixture):
    request = Request.parse(fixture["request"])
    assert request.to_dict() == fixture["request"]
    assert success(request, fixture["result"]["result"]) == fixture["result"]
    error = ServiceError(**fixture["failure"]["error"], status=409)
    result = failure(request, error)
    assert result == fixture["failure"]
    assert "result" not in result and "features" not in json.dumps(result)
    assert set(fixture["request"]["payload"]) == set(Mutation.__annotations__)
    assert set(fixture["result"]["result"]) == set(AcceptedWrite.__annotations__)
    assert set(fixture["result"]["result"]["changes"][0]) == set(
        ComponentVersions.__annotations__
    )


def test_request_detaches_input_and_result_never_echoes_payload(fixture):
    original = fixture["request"]
    request = Request.parse(original)
    original["payload"]["records"][0]["features"]["amount"] = 999
    assert request.payload["records"][0]["features"]["amount"] == 2
    assert "payload" not in success(request, {"durably_accepted": True})


@pytest.mark.parametrize(
    "change",
    [
        {"api_version": "generalforecast/v99"},
        {"unexpected": True},
        {"project": "../elsewhere"},
        {"request_id": ""},
        {"resource": {"kind": "widget", "name": "old"}},
        {"resource": {"kind": [], "name": "bad"}},
        {"resource": {"kind": "context", "name": "ok", "project": "other"}},
        {"payload": []},
    ],
)
def test_invalid_request_rejected(fixture, change):
    with pytest.raises(ServiceError) as error:
        Request.parse(fixture["request"] | change)
    assert error.value.code == "ValidationError"
    assert error.value.status == 422


@pytest.mark.parametrize(
    "value",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": MAX_INTEGER + 1},
        {1: "coerced"},
        {"x": (1, 2)},
        {"x": b"bytes"},
        {"x": "\ud800"},
        {"x": "x" * MAX_BYTES},
        {"x": [0] * MAX_NODES},
    ],
)
def test_nonportable_or_oversize_json(value):
    with pytest.raises(ServiceError) as error:
        json_object(value)
    assert error.value.code == "ValidationError"


def test_json_depth_and_finite_limits():
    assert json_object({"a": [None, False, MAX_INTEGER, -1.5]}) == {
        "a": [None, False, MAX_INTEGER, -1.5]
    }
    nested = {}
    for _ in range(MAX_DEPTH + 1):
        nested = {"x": nested}
    with pytest.raises(ServiceError):
        json_object(nested)


def test_capability_is_not_fake_success(fixture):
    capability = Capability(**fixture["capability"])
    with pytest.raises(ServiceError) as error:
        capability.enforce()
    assert error.value.code == "IsolationUnavailable"
    assert error.value.status == 501
    assert asdict(capability) == fixture["capability"]
    Capability("one-shot", "supported").enforce()
    for args in [
        ("one-shot", "unsupported"),
        ("one-shot", "supported", "Unsupported", "no"),
    ]:
        with pytest.raises(ServiceError):
            Capability(*args)


def test_artifact_and_identity_validation():
    artifact = Artifact("test-only", "revision-1", "sha256:" + "a" * 64, "unknown")
    assert artifact.repository == "test-only"  # Syntax, not integrity evidence.
    with pytest.raises(ServiceError):
        Artifact("test-only", "main", "not-a-digest", "unknown")
    with pytest.raises(ServiceError):
        Scope("../private", "request")
    with pytest.raises(ServiceError):
        Resource("context", "../private")
