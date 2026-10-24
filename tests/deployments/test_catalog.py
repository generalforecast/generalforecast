# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

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


def test_catalog_is_pure_and_returns_copies(monkeypatch):
    monkeypatch.setattr(
        "generalforecast.models.registry.load_ref",
        lambda *_, **__: pytest.fail("no model load"),
    )
    service: ModelPort = CatalogService()
    models = service.list(SCOPE)
    assert [model["name"] for model in models] == [
        "general-forecast/tabicl-v2-classifier",
        "general-forecast/tabicl-v2-regressor",
        *(
            f"general-forecast/kumo-tabular-{size}-{task}"
            for size in ("small", "medium", "large")
            for task in ("classifier", "regressor")
        ),
        "general-forecast/mitra-2-classifier",
        "general-forecast/mitra-2-regressor",
        "general-forecast/chronos-2",
        "general-forecast/timestral-209m",
        "general-forecast/timestral-209m-unmirrored",
        "general-forecast/timestral-28m",
        "general-forecast/timestral-28m-unmirrored",
    ]
    assert {model["runtime"] for model in models} == {"cpu"}
    assert {
        model["runtime"] for model in CatalogService(runtime="gpu").list(SCOPE)
    } == {"gpu"}
    assert service.list(Scope("other-project", "list")) == models
    chronos = next(
        model for model in models if model["name"] == "general-forecast/chronos-2"
    )
    assert (
        chronos["artifact"]["revision"]
        == catalog.ENTRIES["general-forecast/chronos-2"].artifact.revision
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


@pytest.mark.parametrize(
    "model,code,status",
    [
        ("general-forecast/tabicl-v2-classifier@9", "CapacityExceeded", 422),
        ("tabicl-classifier*9", "CapacityExceeded", 422),
        ("general-forecast/tabicl-v2-classifier@4*8", "CapacityExceeded", 422),
        ("tabicl-regressor@2", "UnsupportedCapability", 422),
        ("chronos-2*2", "UnsupportedCapability", 422),
        ("tabpfn-3-classifier", "UnsupportedModel", 422),
        ("timesfm-3", "UnsupportedModel", 422),
        ("limix-2-classifier", "UnsupportedModel", 422),
        ("tabicl-classifier*2@4", "ModelNotFound", 404),
        ("general-forecast/tabicl-v2-classifier@0", "ModelNotFound", 404),
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
    for model in (
        "general-forecast/tabpfn-3-classifier",
        "general-forecast/tabpfn-3-regressor",
    ):
        with pytest.raises(ServiceError) as caught:
            hosted.get(SCOPE, model)
        assert caught.value.code == "UnsupportedModel"
        described = own.get(SCOPE, model)
        assert described["license"] == catalog.ENTRIES[model].artifact.license
    assert own.get(SCOPE, "general-forecast/tabicl-v2-classifier") == hosted.get(
        SCOPE, "general-forecast/tabicl-v2-classifier"
    )
    assert own.get(SCOPE, "general-forecast/timesfm-3")["task"] == "forecast"
    assert [m["name"] for m in own.list(SCOPE)] == [
        m["name"] for m in hosted.list(SCOPE)
    ]
    entry = catalog.ENTRIES["general-forecast/tabpfn-3-classifier"]
    assert entry.allows("self-hosted") and not entry.allows("self-hosted", variant=True)
    assert not entry.allows("remote") and not entry.gates.remote
    assert not catalog.ENTRIES["general-forecast/limix-2-classifier"].allows(
        "self-hosted"
    )
    kumo = catalog.ENTRIES["general-forecast/kumo-tabular-small-classifier"]
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
        ("general-forecast/tabicl-v2-classifier@9", "CapacityExceeded", 422),
        ("unknown-model", "ModelNotFound", 404),
        ("os:system", "ModelNotFound", 404),
    ],
)
def test_self_hosted_refusals(model, code, status):
    with pytest.raises(ServiceError) as caught:
        CatalogService(surface="self-hosted").get(SCOPE, model)
    assert (caught.value.code, caught.value.status) == (code, status)
