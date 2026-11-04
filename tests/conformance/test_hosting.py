# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The hosting API is a recorded surface, importable alone, that hosted serving builds on.

`generalforecast.hosting` names everything a hosted control plane uses. A
change to the surface is a deliberate, versioned change: this list records
it, and the migration note in the release names it.
"""

import ast
import inspect
from pathlib import Path

import pytest

from generalforecast import hosting

SOURCE = Path(__file__).resolve().parents[2] / "src" / "generalforecast"

RECORDED = (
    "ALIASES",
    "AcceptedWrite",
    "CONTEXT_ROWS_KEY",
    "Capability",
    "CatalogService",
    "CompiledPlan",
    "ContextImportRef",
    "ContextVersion",
    "DERIVED_KINDS",
    "Derived",
    "DerivedInputs",
    "DerivedResolver",
    "DerivedWeights",
    "Environment",
    "Error",
    "ImportOutcome",
    "InputError",
    "JsonObject",
    "ModelPort",
    "ModelRefError",
    "Mutation",
    "NativeExecutor",
    "ObservationInput",
    "PHYSICAL_TYPES",
    "PredictionRequest",
    "PredictionResult",
    "PredictorError",
    "PredictorExecutor",
    "PredictorLock",
    "PreparedExecution",
    "PromoteRequest",
    "RateLimit",
    "Request",
    "Resource",
    "ResourceKind",
    "Revision",
    "Scope",
    "ServiceError",
    "SnapshotRow",
    "SourcePin",
    "SourceRecord",
    "V1_POLICY",
    "VERSION",
    "ad_hoc_lock",
    "arrow_bytes",
    "binary",
    "bound_rows",
    "bytes_digest",
    "catalog_model",
    "coerce",
    "compile_lock",
    "configuration",
    "context_version",
    "diagnostic_lines",
    "document_digest",
    "entries",
    "failure",
    "fields",
    "format_as_of",
    "json_object",
    "json_rows",
    "keep_measures",
    "kept_measures",
    "limits",
    "live_context",
    "lock_digest",
    "lock_name",
    "measured",
    "measures_identity",
    "model_id",
    "name",
    "number",
    "padded_widths",
    "parse_as_of",
    "parse_body",
    "parse_lock",
    "parse_model",
    "pin",
    "prefetch_catalog",
    "published_entry",
    "read_table",
    "report_diagnostic",
    "request_table",
    "require",
    "resource_name",
    "served_context",
    "success",
    "table_digest",
    "text",
    "to_arrow",
    "training",
    "unique_object",
    "use_derived",
)


def test_the_hosting_surface_is_the_recorded_one():
    assert tuple(sorted(hosting.__all__)) == RECORDED
    for name in hosting.__all__:
        assert getattr(hosting, name) is not None, name


def test_every_hosting_function_says_what_it_does():
    for name in hosting.__all__:
        value = getattr(hosting, name)
        if inspect.isfunction(value) or inspect.isclass(value):
            assert inspect.getdoc(value), name


def test_compile_lock_cannot_be_called_without_naming_the_manifest():
    parameter = inspect.signature(hosting.compile_lock).parameters["manifest"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="manifest"):
        hosting.compile_lock({}, None)  # ty: ignore[missing-argument]


def test_hosting_imports_nothing_hosted_and_no_private_predictor_module():
    tree = ast.parse((SOURCE / "hosting.py").read_text())
    imported = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not [
        module for module in imported if module.startswith("generalforecast.hosted")
    ]
    assert "generalforecast.server" not in imported


def test_every_export_is_documented_in_the_module():
    import re

    documented = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", hosting.__doc__ or ""))
    assert not [name for name in hosting.__all__ if name not in documented]


def test_the_image_prefetch_is_a_hosting_entry_point(monkeypatch):
    from generalforecast.service import catalog_artifacts

    calls = []
    monkeypatch.setattr(catalog_artifacts, "preload", lambda: calls.append(1))
    assert hosting._main(["prefetch"]) == 0
    assert calls == [1]
    assert hosting._main([]) == 2


def test_the_training_surface_loads_a_tabular_model_as_gf_serves_it(monkeypatch):
    from generalforecast.models import registry
    from generalforecast.models.tabicl.model import tabicl_functions
    from tests.models.tabicl.tiny import tiny_tabicl

    training = hosting.training
    assert training.VERSION == 2
    assert sorted(training.__all__) == [
        "COMPONENTS",
        "COMPONENT_MAPS",
        "FULL_FORMAT",
        "LinearParameters",
        "TabularFunctions",
        "VERSION",
        "load",
        "pack_full",
        "replaced",
        "trainable",
        "unpack_full",
    ]
    assert training.FULL_FORMAT == "full/1"
    asked = []

    def load(spec, *, surface, allow_downloads):
        asked.append((spec, surface))
        return tabicl_functions, tiny_tabicl(regression=False)

    monkeypatch.setattr(registry, "load", load)
    functions, _ = training.load("general-forecast/tabicl-v2-classifier")
    assert functions is tabicl_functions and isinstance(
        functions, training.TabularFunctions
    )
    assert asked == [("general-forecast/tabicl-v2-classifier", "predictor")]


def test_importing_hosting_loads_no_jax():
    import subprocess
    import sys

    code = (
        "import sys; import generalforecast.hosting; "
        "assert not [m for m in sys.modules if m.split('.')[0] == 'jax'], 'jax'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
