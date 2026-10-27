# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The curated Models API: pure metadata, gated per serving surface."""

import hashlib
import json
from typing import Any, cast

import pytest

from generalforecast.models import catalog
from generalforecast.service.boundary import Scope, json_object
from generalforecast.service.catalog import CatalogService
from generalforecast.service.contracts import ServiceError
from generalforecast.service.interfaces import ModelPort

SCOPE = Scope("test-project", "test-catalog")
REMOTE = [
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


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_downloads_are_enabled_only_explicitly(invalid):
    assert not CatalogService().allow_downloads
    assert CatalogService(allow_downloads=True).allow_downloads
    with pytest.raises(ServiceError):
        CatalogService(allow_downloads=invalid)


def test_catalog_is_pure_and_returns_copies(monkeypatch):
    monkeypatch.setattr(
        "generalforecast.models.registry.load_ref",
        lambda *_, **__: pytest.fail("no model load"),
    )
    service: ModelPort = CatalogService()
    models = service.list(SCOPE)
    assert [model["name"] for model in models] == REMOTE
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
    with pytest.raises(ServiceError, match="runs locally only") as caught:
        service.get(SCOPE, "untrusted.module:execute")
    assert caught.value.code == "UnsupportedModel"
    remote = [entry for entry in catalog.entries() if entry.gates.allows("remote")]
    assert remote and service.list(SCOPE) == [
        service.get(SCOPE, entry.id) for entry in remote
    ]
    for model in service.list(SCOPE):
        json_object(model)


@pytest.mark.parametrize(
    "model,code,status",
    [
        ("general-forecast/tabicl-v2-classifier@9", "CapacityExceeded", 422),
        ("general-forecast/tabicl-v2-classifier*9", "CapacityExceeded", 422),
        ("general-forecast/tabicl-v2-classifier@4*8", "CapacityExceeded", 422),
        ("general-forecast/tabicl-v2-regressor@2", "UnsupportedCapability", 422),
        ("general-forecast/chronos-2*2", "UnsupportedCapability", 422),
        ("general-forecast/tabpfn-3-classifier", "UnsupportedModel", 422),
        ("general-forecast/timesfm-3", "UnsupportedModel", 422),
        ("general-forecast/limix-2-classifier", "UnsupportedModel", 422),
        # Bare names, aliases and upstream ids are not catalog ids.
        ("tabicl-v2-classifier", "ModelNotFound", 404),
        ("tabicl-classifier*9", "ModelNotFound", 404),
        ("tabicl-regressor@2", "ModelNotFound", 404),
        ("amazon/chronos-2", "ModelNotFound", 404),
        ("general-forecast/tabicl-v2-classifier*2@4", "ModelNotFound", 404),
        ("general-forecast/tabicl-v2-classifier@0", "ModelNotFound", 404),
        ("os:system", "UnsupportedModel", 422),
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
        ("general-forecast/tabpfn-3-classifier@4", "UnsupportedCapability", 422),
        ("general-forecast/limix-2-classifier", "UnsupportedModel", 422),
        ("general-forecast/tabicl-v2-classifier@9", "CapacityExceeded", 422),
        ("tabpfn-3-classifier@4", "ModelNotFound", 404),
        ("unknown-model", "ModelNotFound", 404),
        ("os:system", "UnsupportedModel", 422),
    ],
)
def test_self_hosted_refusals(model, code, status):
    with pytest.raises(ServiceError) as caught:
        CatalogService(surface="self-hosted").get(SCOPE, model)
    assert (caught.value.code, caught.value.status) == (code, status)
