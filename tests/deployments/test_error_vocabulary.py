# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every error the control plane can put on the wire is an OpenAPI `ErrorCode`."""

import ast
import json
from pathlib import Path

import pytest

import generalforecast
from generalforecast._api import ErrorCode
from generalforecast.deployments.contracts import (
    WIRE_ERROR_CODES,
    ServiceError,
    public_code,
)
from generalforecast.transport import response_value

CONSTRUCTORS = {
    "ServiceError",
    "PredictorError",
    "failure",
    "conflict",
    "source_error",
}
BOUNDARY_STATUSES = (400, 401, 403, 404, 409, 422, 429, 500, 503)


def _raised():
    """Every (code, status) a constructor in the package raises with literals."""
    found = set()
    root = Path(generalforecast.__file__).parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = getattr(func, "id", getattr(func, "attr", ""))
            args = node.args
            if called in {"Capability", "ensure"} and len(args) >= 3:
                args = args[2:]
            elif called not in CONSTRUCTORS:
                continue
            if not (
                args
                and isinstance(args[0], ast.Constant)
                and type(args[0].value) is str
                and args[0].value.isidentifier()
            ):
                continue
            status = None
            if len(args) >= 3 and isinstance(args[2], ast.Constant):
                status = args[2].value
            for keyword in node.keywords:
                if keyword.arg == "status" and isinstance(keyword.value, ast.Constant):
                    status = keyword.value.value
            found.add((args[0].value, status if type(status) is int else None))
    return found


def test_the_wire_vocabulary_is_the_openapi_enum():
    assert WIRE_ERROR_CODES == {code.value for code in ErrorCode}


def test_every_raised_code_reaches_the_wire_as_an_openapi_code():
    raised = _raised()
    assert raised
    for code, status in raised:
        statuses = (status,) if status is not None else BOUNDARY_STATUSES
        for held in statuses:
            wire = ServiceError(code, "safe detail", held)
            assert wire.public in WIRE_ERROR_CODES, (code, held)
            assert wire.public_body()["error"]["code"] == wire.public


@pytest.mark.parametrize(
    "code,status,expected",
    [
        ("Forbidden", 403, "Unauthorized"),
        ("SessionRequired", 403, "Unauthorized"),
        ("UnsupportedLockVersion", 422, "ValidationError"),
        ("InvalidLock", 422, "ValidationError"),
        ("AlreadyExists", 409, "Conflict"),
        ("StaleComponent", 409, "Conflict"),
        ("RateLimited", 429, "CapacityExceeded"),
        ("ModelUnavailable", 503, "ModelNotReady"),
        ("CredentialUnavailable", 503, "ModelNotReady"),
        ("SnapshotNotFound", 404, "NotFound"),
        ("InternalError", 500, "Internal"),
        ("InferenceFailed", 503, "ModelNotReady"),
        ("Conflict", 409, "Conflict"),
    ],
)
def test_internal_codes_fold_into_the_enum(code, status, expected):
    assert public_code(code, status) == expected
    assert ServiceError(code, "safe detail", status).public_body() == {
        "error": {"code": expected, "message": "safe detail"}
    }


@pytest.mark.parametrize(
    "code,expected",
    [
        ("Conflict", "Conflict"),
        ("ModelNotReady", "ModelNotReady"),
        ("FutureCodeFromNewerServer", "RemoteError"),
        ("ghp_credentialLookingCode", "RemoteError"),
    ],
)
def test_remote_codes_are_classified_without_echoing_unknowns(code, expected):
    message = "safe service detail"
    raw = json.dumps({"error": {"code": code, "message": message}}).encode()
    with pytest.raises(ServiceError) as error:
        response_value(409, raw)
    assert error.value.code == expected and error.value.status == 409
    assert str(error.value) == message
    assert expected != "RemoteError" or code not in str(error.value.code)


def test_api_envelopes_fold_codes_and_browser_sessions_keep_them():
    from generalforecast.deployments.boundary import Request, Resource, Scope, failure

    error = ServiceError("StaleComponent", "expected version is stale", 409)
    api = Request(Scope("p", "r-1"), Resource("predictor", "fraud"), {})
    assert failure(api, error)["error"]["code"] == "Conflict"
    console = Request(Scope("p", "r-2"), Resource("workspace", "members"), {})
    assert failure(console, error)["error"]["code"] == "Conflict"
    assert failure(console, error, typed=True)["error"]["code"] == "StaleComponent"
