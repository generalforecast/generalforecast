# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The native validator's catalog manifest and reference parsing match the catalog."""

import json
from pathlib import Path

import pytest

from generalforecast import _core
from generalforecast.deployments.boundary import json_bytes
from generalforecast.models import catalog

MANIFEST = Path(__file__).resolve().parents[2] / "crates" / "gf-core" / "catalog.json"
DIGEST = catalog.ENTRIES["general-forecast/tabicl-v2-classifier"].artifact.digest


def test_checked_in_manifest_is_the_catalogs() -> None:
    assert json.loads(MANIFEST.read_text()) == catalog.manifest(), (
        "regenerate with `python -m generalforecast.models.catalog > "
        "crates/gf-core/catalog.json`"
    )


@pytest.mark.parametrize(
    "spec",
    [
        "general-forecast/tabicl-v2-classifier",
        "tabicl-v2-classifier",
        "tabicl-classifier",
        "general-forecast/tabicl-classifier",
        "general-forecast/tabicl-v2-classifier@4*2",
        f"general-forecast/tabicl-v2-classifier@{DIGEST}",
        f"general-forecast/tabicl-v2-classifier@{DIGEST}@3*2",
        f"general-forecast/tabicl-v2-classifier@{DIGEST}0",
        "general-forecast/tabicl-v2-classifier@sha256:" + "0" * 64,
        "general-forecast/tabicl-v2-classifier@007",
        "general-forecast/tabicl-v2-classifier@0",
        "general-forecast/tabicl-v2-classifier*00",
        "general-forecast/tabicl-v2-classifier*2@4",
        "general-forecast/tabicl-v2-classifier@",
        "general-forecast/tabicl-v2-regressor@2",
        "general-forecast/tabicl-v2-regressor*3",
        "general-forecast/chronos-2*2",
        "amazon/chronos-2",
        "mistralai/Timestral-28M",
        "os:system",
        "catalog/",
        "catalog/os:system",
        "unknown-model",
        "general-forecast/tabpfn-3.5-classifier",
        "it's\n\x00\u00a0é",
        "",
    ],
)
def test_native_references_parse_as_the_catalog_does(spec: str) -> None:
    try:
        expected = str(catalog.parse(spec))
    except catalog.ModelRefError as error:
        expected = {
            "UnknownModel": ("ModelNotFound", "model is not in the curated catalog"),
            "InvalidModel": ("ModelNotFound", "model is not in the curated catalog"),
        }.get(error.code, (error.code, str(error)))
    if spec == "":
        expected = ("ValidationError", "invalid model")
    try:
        actual = _core.resolve_model(spec, "local")
    except _core.RequestError as error:
        actual = error.args[:2]
    assert actual == expected


def _tabular(context: list, queries: list, options: dict | None = None) -> bytes:
    return json_bytes(
        {
            "model": "general-forecast/tabicl-v2-classifier",
            "task": "classification",
            "target": "y",
            "features": ["x"],
            "context": context,
            "queries": queries,
            "options": options or {},
        }
    )


def test_fingerprints_address_the_context_not_the_queries() -> None:
    context = [
        {"key": "a", "features": {"x": 1}, "label": "yes"},
        {"key": "b", "features": {"x": 2.5}, "label": "no"},
        {"key": "c", "features": {"x": 3}},
    ]
    first = _core.parse_inference(
        _tabular(context, [{"key": "q", "features": {"x": 0}}]), "remote"
    )
    renamed = [{**row, "key": f"{row['key']}!"} for row in context]
    second = _core.parse_inference(
        _tabular(renamed, [{"key": "r", "features": {"x": 9}}]), "remote"
    )
    other = _core.parse_inference(
        _tabular(context[:1], [{"key": "q", "features": {"x": 0}}]), "remote"
    )
    assert isinstance(first, _core.TabularRequest)
    assert first.fingerprint == second.fingerprint != other.fingerprint
    assert len(first.fingerprint) == 64
    assert first.classes == ["no", "yes"]
    assert first.targets.tolist() == [1, 0]
    assert first.context.tolist() == [[1.0], [2.5]]
    assert (first.supplied_rows, first.query_keys) == (3, ["q"])


def test_payload_errors_carry_the_models_api_envelope() -> None:
    with pytest.raises(_core.RequestError) as caught:
        _core.parse_inference(b'{"model": 1, "model": 2}', "remote")
    assert caught.value.args == ("ValidationError", "duplicate JSON field", 422)
    with pytest.raises(_core.RequestError) as caught:
        _core.parse_inference(b"[]", "remote")
    assert caught.value.args == ("ValidationError", "expected JSON object", 422)
    with pytest.raises(_core.RequestError) as caught:
        _core.parse_inference(
            _tabular([], [{"key": "q", "features": {"x": 0}}]), "remote"
        )
    assert caught.value.args == (
        "EmptyContext",
        "no usable labeled rows; no score",
        503,
    )
