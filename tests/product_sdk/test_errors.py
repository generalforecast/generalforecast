# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One error hierarchy: every layer's errors are `gf.Error`s of one category."""

import copy
import json
import pickle
from pathlib import Path

import pytest

import generalforecast as gf
from generalforecast import errors
from generalforecast.client import Client
from generalforecast.service.contracts import InputError, ServiceError
from generalforecast.transport import HostedOnlyError, RelayedError, response_value


@pytest.mark.parametrize(
    "code,status,category",
    [
        ("ValidationError", 422, gf.InvalidInput),
        ("NotFound", 404, gf.NotFound),
        ("SnapshotNotFound", 422, gf.NotFound),
        ("Unauthorized", 401, gf.Unauthorized),
        ("SessionRequired", 403, gf.Unauthorized),
        ("WriteConflict", 409, gf.Conflict),
        ("ContextCapacityExceeded", 413, gf.CapacityExceeded),
        ("RateLimited", 429, gf.CapacityExceeded),
        ("ModelUnavailable", 503, gf.ModelNotReady),
        ("WaitTimeout", 504, gf.ModelNotReady),
        ("InternalError", 500, gf.Internal),
        ("ProtocolError", 502, gf.Internal),
        ("TransportError", 503, gf.Unavailable),
        ("Timeout", 504, gf.Unavailable),
        ("ClientClosed", 400, gf.InvalidInput),
    ],
)
def test_a_service_error_is_the_error_of_the_category_its_code_names(
    code, status, category
):
    error = ServiceError(code, "message", status)
    assert isinstance(error, ServiceError) and isinstance(error, category)
    assert isinstance(error, gf.Error)
    assert (error.code, error.status, error.message) == (code, status, "message")
    with pytest.raises(category):
        raise error


def test_wire_form_and_reason_are_unchanged_for_services():
    folded = ServiceError("AuthorizationPending", "waiting for browser approval", 400)
    assert folded.public_body() == {
        "error": {
            "code": "Unauthorized",
            "message": "waiting for browser approval",
            "reason": "authorization_pending",
        }
    }
    assert str(folded) == "waiting for browser approval"
    bounded = InputError("ContextCapacityExceeded", "too many", 413, limit=5, actual=6)
    assert bounded.error() == {
        "code": "CapacityExceeded",
        "message": "too many",
        "reason": "context_capacity_exceeded",
        "limit": 5,
        "actual": 6,
    }
    assert ServiceError("NotFound", "gone", 404).error() == {
        "code": "NotFound",
        "message": "gone",
    }


def test_errors_pickle_and_copy_with_their_category():
    for error in (
        ServiceError("NotFound", "gone", 404),
        InputError("ValidationError", "bad", param="inputs.x"),
        HostedOnlyError("GET", "/v1/projects/p/keys"),
        gf.Conflict("moved"),
        errors.Error("SlowDown", "poll later", 400),
    ):
        for copied in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
            assert type(copied) is type(error)
            assert copied.__dict__ == error.__dict__


def test_from_wire_is_the_one_decoder_and_keeps_reason():
    error = errors.from_wire(
        {"code": "Unauthorized", "message": "wait", "reason": "authorization_pending"},
        status=400,
        request_id="r-1",
    )
    assert type(error) is gf.Unauthorized
    assert (error.reason, error.request_id, error.status) == (
        "authorization_pending",
        "r-1",
        400,
    )
    relayed = None
    try:
        response_value(
            400,
            json.dumps(
                {
                    "error": {
                        "code": "ValidationError",
                        "message": "slow",
                        "reason": "slow_down",
                    }
                }
            ).encode(),
        )
    except RelayedError as caught:
        relayed = caught
    assert relayed is not None and isinstance(relayed, gf.InvalidInput)
    assert relayed.reason == "slow_down"
    unknown = errors.from_wire({"code": "Weird", "message": "x"}, status=500)
    assert (unknown.code, type(unknown)) == ("RemoteError", gf.Internal)
    malformed = errors.from_wire({"code": 1}, status=500)
    assert malformed.code == "ProtocolError" and isinstance(malformed, gf.Internal)


def test_except_category_catches_a_client_call(wire):
    wire.expect("GET", "/predictors/gone", "NotFound", status=404)
    with (
        Client(wire.url, project="test", api_key="fixture-key") as client,
        pytest.raises(gf.NotFound) as error,
    ):
        client.predictors.get("gone")
    assert isinstance(error.value, ServiceError)


def test_the_boundary_maps_by_code_and_cause_and_leaves_bugs_alone():
    from generalforecast.engine import ModelOutputError
    from generalforecast.predictors.validation import PredictorError

    assert isinstance(
        errors.convert(PredictorError("CapacityExceeded", "too many rows")),
        gf.CapacityExceeded,
    )
    assert isinstance(
        errors.convert(ModelOutputError("the model returned invalid outputs")),
        gf.Internal,
    )
    with pytest.raises(TypeError), errors.boundary():
        raise TypeError("f() missing 2 required keyword-only arguments")
    with pytest.raises(KeyError), errors.boundary():
        {}["missing"]
    framed = errors.convert(KeyError('Flight returned not found error: table "x"'))
    assert isinstance(framed, gf.NotFound) and framed.message == 'table "x"'
    with pytest.raises(gf.NotFound), errors.boundary():
        Path("/nonexistent/rows.csv").read_bytes()
