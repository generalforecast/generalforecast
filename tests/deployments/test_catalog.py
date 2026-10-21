import copy
import hashlib
import json
from typing import Any, cast

import pytest

from generalforecast.deployments.boundary import Scope, json_object
from generalforecast.deployments.catalog import CatalogService, create_catalog
from generalforecast.deployments.contracts import ServiceError
from generalforecast.deployments.interfaces import ModelPort
from generalforecast.models import catalog

SCOPE = Scope("test-project", "test-catalog")


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_provider_factory_explicitly_enables_downloads(invalid):
    assert not CatalogService().allow_downloads
    assert create_catalog().allow_downloads
    with pytest.raises(ServiceError):
        CatalogService(allow_downloads=invalid)


@pytest.fixture
def request_data():
    return {
        "model": "tabicl-v2-classifier",
        "task": "classification",
        "target": "outcome",
        "features": ["x", "z"],
        "context": [
            {"key": "c1", "features": {"x": 1, "z": 2}, "label": False},
            {"key": "c2", "features": {"x": 3, "z": 4}, "label": True},
        ],
        "queries": [{"key": "q", "features": {"x": 2, "z": 3}}],
        "options": {},
    }


@pytest.fixture(autouse=True)
def no_native_inference(monkeypatch):
    def refuse(*_args, **_kwargs):
        pytest.fail("unit validation must not execute a native model")

    monkeypatch.setattr(CatalogService, "_predict", refuse)


def error_code(request, code="ValidationError", status=422):
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, request)
    assert (caught.value.code, caught.value.status) == (code, status)


def test_catalog_is_pure_and_returns_copies(monkeypatch):
    monkeypatch.setattr(
        "generalforecast.models.registry.load_ref",
        lambda *_, **__: pytest.fail("no model load"),
    )
    service: ModelPort = CatalogService()
    models = service.list(SCOPE)
    assert [model["name"] for model in models] == [
        "tabicl-v2-classifier",
        "tabicl-v2-regressor",
        *(
            f"kumo-tabular-{size}-{task}"
            for size in ("small", "medium", "large")
            for task in ("classifier", "regressor")
        ),
        "mitra-2-classifier",
        "mitra-2-regressor",
        "chronos-2",
        "timestral-209m",
        "timestral-209m-unmirrored",
        "timestral-28m",
        "timestral-28m-unmirrored",
    ]
    assert {model["runtime"] for model in models} == {"cpu"}
    assert {
        model["runtime"] for model in CatalogService(runtime="gpu").list(SCOPE)
    } == {"gpu"}
    assert service.list(Scope("other-project", "list")) == models
    chronos = next(model for model in models if model["name"] == "chronos-2")
    assert (
        chronos["artifact"]["revision"]
        == catalog.ENTRIES["chronos-2"].artifact.revision
    )
    assert all(
        capability["state"] == "supported" for capability in chronos["capabilities"]
    )
    models[0]["artifact"]["files"].clear()
    actual = service.get(SCOPE, "general-forecast/tabicl-v2-classifier")
    assert len(actual["artifact"]["files"]) == 3
    manifest = json.dumps(
        actual["artifact"]["files"], sort_keys=True, separators=(",", ":")
    )
    assert (
        actual["artifact"]["digest"]
        == "sha256:" + hashlib.sha256(manifest.encode()).hexdigest()
    )
    with pytest.raises(ServiceError, match="curated catalog") as caught:
        service.get(SCOPE, "untrusted.module:execute")
    assert caught.value.code == "ModelNotFound"
    for model in service.list(SCOPE):
        json_object(model)


@pytest.mark.parametrize("context", [[], [{"key": "c", "features": {"x": 1, "z": 2}}]])
def test_empty_usable_context(request_data, context):
    request_data["context"] = context
    error_code(request_data, "EmptyContext", 503)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "custom.py:run"},
        {"task": "forecast"},
        {"features": []},
        {"features": "x"},
        {"features": ["x", "x"]},
        {"features": ["outcome"]},
        {"features": [str(i) for i in range(33)]},
        {"queries": []},
        {"queries": {}},
        {"queries": [{"key": "q", "features": {"x": 1, "z": 2}, "label": True}]},
        {"queries": [{"key": "q", "features": {"x": True, "z": 2}}]},
        {"queries": [{"key": "q", "features": {"x": float("nan"), "z": 2}}]},
        {"queries": [{"key": "q", "features": {"x": 1e40, "z": 2}}]},
        {"queries": [{"key": "q", "features": {"x": 1}}]},
        {"queries": [{"key": "q", "features": {"x": 1, "z": 2, "extra": 3}}]},
        {"queries": [{"key": "q", "features": {"x": 1, "z": 2}}] * 101},
        {
            "context": [{"key": "c", "features": {"x": 1, "z": 2}, "label": False}]
            * 1025
        },
        {"context": [{"key": "c", "features": {"x": 1, "z": 2}, "label": False}] * 2},
        {"options": {"unknown": True}},
        {"options": {"quantiles": [0.5]}},
        {"target": ""},
        {"unexpected": "field"},
    ],
)
def test_bad_requests_do_not_load_model(request_data, change):
    request_data.update(change)
    expected = (
        "ModelNotFound"
        if "model" in change
        else "UnsupportedTask"
        if "task" in change
        else "ValidationError"
    )
    error_code(request_data, expected, 404 if "model" in change else 422)


@pytest.mark.parametrize("labels", [[True, 1], [None, True], [1.5, 2.5], ["", "a"]])
def test_invalid_class_labels(request_data, labels):
    for row, label in zip(request_data["context"], labels, strict=True):
        row["label"] = label
    error_code(request_data)


def test_too_many_classes(request_data):
    request_data["context"] = [
        {"key": str(i), "features": {"x": i, "z": 0}, "label": i} for i in range(11)
    ]
    error_code(request_data, "UnsupportedClassCount")


@pytest.mark.parametrize(
    "labels,names,encoded",
    [
        ([False], ["false", "true"], [0]),
        (["yes", "no"], ["no", "yes"], [1, 0]),
        ([23, -2], ["-2", "23"], [1, 0]),
        (["only"], ["only"], [0]),
    ],
)
def test_tabular_output_and_class_mapping(
    request_data, monkeypatch, labels, names, encoded
):
    request_data["context"] = request_data["context"][: len(labels)]
    for row, label in zip(request_data["context"], labels, strict=True):
        row["label"] = label
    request_data["context"].append({"key": "unlabeled", "features": {"x": 0, "z": 1}})
    before = copy.deepcopy(request_data)
    calls = []

    def predict(self, ref, features, targets, queries, classes):
        calls.append((ref.name, features, targets, queries, classes))
        return [[1 / classes] * classes]

    monkeypatch.setattr(CatalogService, "_predict", predict)
    result = CatalogService().infer(SCOPE, request_data)
    assert request_data == before
    assert len(calls) == 1 and calls[0][2] == encoded
    assert calls[0][3] == [[2, 3]]
    assert list(result["outputs"][0]["probabilities"]) == names
    assert result["context_provenance"]["usable_labeled_rows"] == len(labels)
    assert result["context_provenance"]["persisted"] is False
    assert "context" not in vars(CatalogService())
    assert (
        result["artifact"]["revision"]
        == catalog.ENTRIES["tabicl-v2-classifier"].artifact.revision
    )
    json_object(dict(result))


def test_regression_quantile_selection(request_data, monkeypatch):
    request_data.update(
        model="tabicl-v2-regressor",
        task="regression",
        options={"quantiles": [0.001, 0.5, 0.999]},
    )
    for i, row in enumerate(request_data["context"]):
        row["label"] = 10.0 + i

    def predict(self, ref, features, labels, queries, classes):
        assert ref.name == "tabicl-v2-regressor"
        assert labels == [10.0, 11.0] and classes == 0
        return [list(range(999))]

    monkeypatch.setattr(CatalogService, "_predict", predict)
    result = CatalogService().infer(SCOPE, request_data)
    assert result["outputs"] == [
        {"key": "q", "quantiles": {"0.001": 0, "0.5": 499, "0.999": 998}}
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"quantiles": []},
        {"quantiles": [0.5, 0.1]},
        {"quantiles": [0.5, 0.5]},
        {"quantiles": [0]},
        {"quantiles": [1]},
        {"quantiles": [0.1234]},
        {"quantiles": [True]},
    ],
)
def test_invalid_regression_quantiles(request_data, options):
    request_data.update(model="tabicl-v2-regressor", task="regression", options=options)
    for row in request_data["context"]:
        row["label"] = 1.0
    error_code(request_data)


def test_many_class_codes_serve_more_classes_than_slots(request_data, monkeypatch):
    request_data["context"] = [
        {"key": str(i), "features": {"x": i, "z": 0}, "label": f"c{i:02d}"}
        for i in range(25)
    ]
    calls = []

    def predict(self, ref, features, targets, queries, classes):
        calls.append((str(ref), classes))
        return [[1 / classes] * classes]

    monkeypatch.setattr(CatalogService, "_predict", predict)
    for model in ("tabicl-classifier@4", "general-forecast/tabicl-v2-classifier@4*2"):
        request_data["model"] = model
        result = CatalogService().infer(SCOPE, request_data)
        assert len(result["outputs"][0]["probabilities"]) == 25
    assert calls == [("tabicl-v2-classifier@4", 25), ("tabicl-v2-classifier@4*2", 25)]
    described = CatalogService().get(SCOPE, "tabicl-classifier@4*2")
    assert described["variant"] == {"codes": 4, "estimators": 2}
    assert described["bounds"]["classes"] == 1000
    assert (
        described["artifact"]
        == CatalogService().get(SCOPE, "tabicl-classifier")["artifact"]
    )


@pytest.mark.parametrize(
    "model,code,status",
    [
        ("tabicl-classifier@9", "CapacityExceeded", 422),
        ("tabicl-classifier*9", "CapacityExceeded", 422),
        ("tabicl-classifier@4*8", "CapacityExceeded", 422),
        ("tabicl-regressor@2", "UnsupportedCapability", 422),
        ("chronos-2*2", "UnsupportedCapability", 422),
        ("tabpfn-3-classifier", "UnsupportedModel", 422),
        ("timesfm-3", "UnsupportedModel", 422),
        ("limix-2-classifier", "UnsupportedModel", 422),
        ("tabicl-classifier*2@4", "ModelNotFound", 404),
        ("tabicl-classifier@0", "ModelNotFound", 404),
        ("os:system", "ModelNotFound", 404),
    ],
)
def test_variants_respect_gates_and_serving_policy(model, code, status):
    with pytest.raises(ServiceError) as caught:
        CatalogService().get(SCOPE, model)
    assert (caught.value.code, caught.value.status) == (code, status)


def test_self_hosted_serves_pinned_local_only_tabular_models():
    hosted = CatalogService()
    own = CatalogService(surface="self-hosted")
    for model in ("tabpfn-3-classifier", "tabpfn-3-regressor"):
        with pytest.raises(ServiceError) as caught:
            hosted.get(SCOPE, model)
        assert caught.value.code == "UnsupportedModel"
        described = own.get(SCOPE, model)
        assert described["license"] == catalog.ENTRIES[model].artifact.license
    assert own.get(SCOPE, "tabicl-classifier") == hosted.get(SCOPE, "tabicl-classifier")
    assert own.get(SCOPE, "timesfm-3")["task"] == "forecast"
    assert [m["name"] for m in own.list(SCOPE)] == [
        m["name"] for m in hosted.list(SCOPE)
    ]
    entry = catalog.ENTRIES["tabpfn-3-classifier"]
    assert entry.allows("self-hosted") and not entry.allows("self-hosted", variant=True)
    assert not entry.allows("remote") and not entry.gates.remote
    assert not catalog.ENTRIES["limix-2-classifier"].allows("self-hosted")
    kumo = catalog.ENTRIES["kumo-tabular-small-classifier"]
    assert kumo.allows("self-hosted", variant=True) == kumo.allows(
        "remote", variant=True
    )
    with pytest.raises(ServiceError):
        CatalogService(surface=cast(Any, "local"))


@pytest.mark.parametrize(
    "model,code,status",
    [
        ("tabpfn-3-classifier@4", "UnsupportedCapability", 422),
        ("limix-2-classifier", "UnsupportedModel", 422),
        ("tabicl-classifier@9", "CapacityExceeded", 422),
        ("unknown-model", "ModelNotFound", 404),
        ("os:system", "ModelNotFound", 404),
    ],
)
def test_self_hosted_refusals(model, code, status):
    with pytest.raises(ServiceError) as caught:
        CatalogService(surface="self-hosted").get(SCOPE, model)
    assert (caught.value.code, caught.value.status) == (code, status)
