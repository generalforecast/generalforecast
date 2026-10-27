# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every error gf can put on the wire is an OpenAPI `ErrorCode`, folded the same in every envelope."""

import ast
from pathlib import Path

import pytest

import generalforecast
from generalforecast._api import ErrorCode
from generalforecast.service.boundary import Request, Resource, Scope, failure
from generalforecast.service.contracts import (
    SESSION_REASONS,
    WIRE_ERROR_CODES,
    ServiceError,
    public_code,
    reason_of,
)

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
        ("LinkConfirmationRequired", 409, "Unauthorized"),
        ("WaitlistRequired", 403, "Unauthorized"),
        ("UnsupportedLockVersion", 422, "ValidationError"),
        ("InvalidLock", 422, "ValidationError"),
        ("AlreadyExists", 409, "Conflict"),
        ("StaleComponent", 409, "Conflict"),
        ("RateLimited", 429, "CapacityExceeded"),
        ("ModelUnavailable", 503, "ModelNotReady"),
        ("SnapshotNotFound", 404, "NotFound"),
        ("InternalError", 500, "Internal"),
        ("InferenceFailed", 503, "ModelNotReady"),
        ("Conflict", 409, "Conflict"),
    ],
)
def test_internal_codes_fold_into_the_enum(code, status, expected):
    assert public_code(code, status) == expected
    reason = {} if code in WIRE_ERROR_CODES else {"reason": reason_of(code)}
    assert ServiceError(code, "safe detail", status).public_body() == {
        "error": {"code": expected, "message": "safe detail", **reason}
    }


def test_every_envelope_folds_codes_and_browser_refusals_carry_a_reason():
    error = ServiceError("StaleComponent", "expected version is stale", 409)
    api = Request(Scope("p", "r-1"), Resource("predictor", "fraud"), {})
    assert failure(api, error)["error"] == {
        "code": "Conflict",
        "message": "expected version is stale",
        "reason": "stale_component",
    }
    plain = ServiceError("Conflict", "expected version is stale", 409)
    assert "reason" not in failure(api, plain)["error"]
    console = Request(Scope("p", "r-2"), Resource("workspace", "members"), {})
    assert failure(console, error)["error"]["code"] == "Conflict"
    assert SESSION_REASONS == {
        "ReauthenticationRequired": "reauthentication_required",
        "LinkConfirmationRequired": "link_confirmation_required",
        "SessionRequired": "session_required",
        "WaitlistRequired": "waitlist_required",
    }
    for code, reason in SESSION_REASONS.items():
        status = 409 if code == "LinkConfirmationRequired" else 401
        refused = ServiceError(code, "sign in", status)
        assert refused.status == status
        assert failure(console, refused)["error"] == {
            "code": "Unauthorized",
            "message": "sign in",
            "reason": reason,
        }
        assert refused.public_body()["error"]["reason"] == reason
