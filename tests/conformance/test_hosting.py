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

import generalforecast.hosting as hosting

SOURCE = Path(__file__).resolve().parents[2] / "src" / "generalforecast"

RECORDED = (
    "ALIASES",
    "AcceptedWrite",
    "Capability",
    "CatalogService",
    "CompiledPlan",
    "ContextImportRef",
    "ContextSnapshot",
    "ContextVersion",
    "Environment",
    "Error",
    "ExecutionPlan",
    "ImportOutcome",
    "InputError",
    "JsonObject",
    "LOCK_VERSION",
    "ModelPin",
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
    "PreparedPredictor",
    "PromoteRequest",
    "RateLimit",
    "Request",
    "Resource",
    "ResourceKind",
    "Revision",
    "RuntimePlan",
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
    "bind_request",
    "bytes_digest",
    "catalog_model",
    "coerce",
    "compile_lock",
    "configuration",
    "context_version",
    "entries",
    "failure",
    "fields",
    "json_object",
    "json_rows",
    "limits",
    "live_context",
    "lock_digest",
    "lock_name",
    "model_id",
    "name",
    "number",
    "parse_body",
    "parse_lock",
    "parse_model",
    "pin",
    "published_entry",
    "read_table",
    "request_table",
    "require",
    "resource_name",
    "served_context",
    "success",
    "table_digest",
    "task_of",
    "temporal",
    "text",
    "to_arrow",
    "unique_object",
    "upgrade_lock",
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
