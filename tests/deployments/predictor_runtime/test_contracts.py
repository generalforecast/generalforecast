# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pyarrow as pa
import pytest
import yaml

from generalforecast import predictors
from generalforecast.contracts import (
    ContextImportRef,
    ContextVersion,
    Environment,
    PredictionResult,
    PredictorLock,
    PromoteRequest,
    Revision,
    RuntimePlan,
)
from generalforecast.predictors.validation import PredictorError, canonical

HASH = "sha256:" + "a" * 64


def lock(tmp_path, context="request: {max_rows: 100}"):
    (tmp_path / "context.csv").write_text("x,y\n1,2.0\n2,4.0\n")
    (tmp_path / "gf.yaml").write_text(
        "name: fraud\nmodel: general-forecast/tabicl-v2-regressor\ntarget: y\n"
        f"inputs: [x]\ncontext: {{{context}}}\n"
    )
    predictors.write_lock(
        predictors.resolve(predictors.load(tmp_path), tmp_path), tmp_path
    )
    compiled = PredictorLock(yaml.safe_load((tmp_path / "gf.lock").read_text()))
    assert isinstance(compiled, PredictorLock)
    return compiled


@pytest.mark.parametrize(
    "value",
    [
        Revision("fraud", 1, HASH, "general-forecast/tabicl-v2-classifier@" + HASH, 1),
        Environment("fraud", "production", 1, 0, "local"),
        PromoteRequest(1, 0),
        ContextVersion("live/events", "transaction-1", 12, 1),
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
    production = Environment("fraud", "production", 1, 0, "local", "ready")
    promoted = production.promote(PromoteRequest(2, 0))
    assert production.revision == 1 and production.generation == 0
    assert promoted.revision == 2 and promoted.generation == 1
    with pytest.raises(PredictorError) as error:
        promoted.promote(PromoteRequest(3, 0))
    assert error.value.code == "Conflict"
    rollback = promoted.promote(PromoteRequest(1, 1))
    assert rollback.revision == 1 and rollback.generation == 2


def test_runtime_plan_derivation_is_pure_and_preserves_placement(tmp_path):
    compiled = lock(tmp_path)
    plan = RuntimePlan.for_lock(
        compiled,
        runtime="byoc",
        environment="production",
        secret_refs=("secret/warehouse",),
    )
    assert plan.predictor_digest == compiled.digest
    assert plan.model_digest == compiled.predictor["model_digest"]
    assert plan.cpu == 1 and plan.memory == "2Gi"
    assert plan.region is None and plan.residency == ()
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
def test_dataset_import_rejects_inline_credentials_signed_urls_and_traversal(source):
    with pytest.raises(PredictorError):
        ContextImportRef("events", source, "parquet")


def test_prediction_nested_outputs_have_conversion_and_provenance():
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
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
        ContextVersion("live/events", 3, 42),
        "fraud",
        1,
        "production",
    )
    assert result.to_arrow().equals(table)
    assert result.to_pandas().shape == (2, 2)
    assert result.to_polars().shape == (2, 2)
    assert result.to_numpy().shape == (2, 2)
    wire = json.loads(canonical(result.to_dict()))
    parsed = PredictionResult.from_dict(wire)
    assert parsed.to_dict() == wire
    assert parsed.context is not None
    assert parsed.context.version == 3
    empty = PredictionResult(pa.table({}), "tabicl", "classification")
    assert empty.to_numpy().shape == (0, 0)


def test_untrusted_lock_capabilities_require_admission_even_with_valid_content_hash(
    tmp_path,
):
    compiled = lock(tmp_path)
    forged = compiled.to_dict()
    forged["predictor"]["model_digest"] = HASH
    admitted = PredictorLock.from_dict(forged)
    assert admitted.identity != compiled.identity
    assert admitted.digest != compiled.digest
