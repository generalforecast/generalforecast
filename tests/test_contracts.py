# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shared wire values: exact round trips, immutability and CAS transitions."""

import dataclasses
import json

import pyarrow as pa
import pytest

from generalforecast.contracts import (
    ContextImportRef,
    ContextVersion,
    Environment,
    PredictionResult,
    PromoteRequest,
    Revision,
    RuntimePlan,
)
from generalforecast.predictors.validation import PredictorError, canonical
from tests.utils.locks import directory, lock_of

HASH = "sha256:" + "a" * 64


@pytest.mark.parametrize(
    "value",
    [
        Revision("fraud", 1, HASH, "general-forecast/tabicl-v2-classifier@" + HASH, 1),
        Environment("fraud", "production", 1, 0),
        PromoteRequest(1, 0),
        ContextVersion("live", "transaction-1", 12, 1, name="events"),
        ContextImportRef(
            "events",
            "s3://bucket/events.parquet",
            "parquet",
            "secret/warehouse",
            HASH,
            (("x", "value"),),
        ),
        RuntimePlan(
            "local",
            "production",
            HASH,
            HASH,
            None,
            "us-east-1",
            ("us-east-1",),
            2,
            "8Gi",
            1,
            2,
            ("secret/warehouse",),
        ),
    ],
)
def test_typed_values_roundtrip_exact_wire_and_are_immutable(value):
    serialized = json.loads(canonical(value.to_dict()))
    assert type(value).from_dict(serialized) == value
    with pytest.raises(PredictorError):
        type(value).from_dict({**serialized, "unexpected_secret": "redacted"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.extra = True


def test_environment_promotion_and_rollback_are_cas_and_do_not_mutate_production():
    production = Environment("fraud", "production", 1, 0, "ready")
    promoted = production.promote(PromoteRequest(2, 0))
    assert production.revision == 1 and production.generation == 0
    assert promoted.revision == 2 and promoted.generation == 1
    with pytest.raises(PredictorError) as error:
        promoted.promote(PromoteRequest(3, 0))
    assert error.value.code == "Conflict"
    rollback = promoted.promote(PromoteRequest(1, 1))
    assert rollback.revision == 1 and rollback.generation == 2


def test_runtime_plan_holds_placement_to_its_rules(tmp_path):
    rows = pa.table({"x": [1, 2], "y": [2.0, 4.0]})
    lock = lock_of(
        directory(
            tmp_path,
            rows,
            context="request",
            model="general-forecast/tabicl-v2-regressor",
        )
    )
    plan = RuntimePlan(
        "byoc",
        "production",
        lock.digest,
        lock.predictor["model_digest"],
        lock.data_digest,
        None,
        (),
        1,
        "2Gi",
        1,
        1,
        ("secret/warehouse",),
    )
    assert plan.predictor_digest == lock.digest and plan.data_digest is None
    with pytest.raises(PredictorError):
        dataclasses.replace(plan, region="eu-west-1", residency=("us-east-1",))
    with pytest.raises(PredictorError):
        dataclasses.replace(plan, min_replicas=3, max_replicas=1)


@pytest.mark.parametrize(
    "source",
    [
        "s3://user:password@bucket/table",
        "https://bucket/table?signature=redacted",
        "s3://bucket/table?signature=redacted",
        "s3://bucket/../private",
        "s3://bucket/%2e%2e/private",
        "s3://bucket/%40password",
        "file:///etc/passwd",
    ],
)
def test_context_import_rejects_inline_credentials_signed_urls_and_traversal(source):
    with pytest.raises(PredictorError):
        ContextImportRef("events", source, "parquet")


def test_prediction_nested_outputs_round_trip_with_provenance():
    table = pa.table(
        {
            "label": [False, True],
            "probabilities": [{"false": 0.7, "true": 0.3}, {"false": 0.1, "true": 0.9}],
        }
    )
    result = PredictionResult(
        table,
        "tabicl-v2-classifier",
        "classification",
        ContextVersion("live", 3, 42, name="events"),
        "fraud",
        1,
        "production",
    )
    assert result.to_arrow().equals(table)
    assert result.to_numpy().shape == (2, 2)
    wire = json.loads(canonical(result.to_dict()))
    parsed = PredictionResult.from_dict(wire)
    assert parsed.to_dict() == wire
    assert parsed.context == ContextVersion("live", 3, 42, name="events")
    empty = PredictionResult(pa.table({}), "tabicl", "classification")
    assert empty.to_numpy().shape == (0, 0)
