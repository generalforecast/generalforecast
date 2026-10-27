# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The hosting API: what a hosted control plane builds on to serve gf predictors.

A hosted plane stores pushed `gf.lock` documents (version 2) and their
context rows, compiles and binds them with the same code `gf.Predictor` and
`gf serve` run, and publishes revisions to gf-server. Everything it needs
from gf is here, under these names; its modules import this one and no
private gf module.

The flow, for one revision:

1. `parse_lock` the pushed document; `pin(lock.model).digest` must equal
   the lock's `model_digest`, and an uploaded snapshot's digest the lock's
   `data_digest`. A stored version 1 lock is rewritten with `upgrade_lock`.
2. Gather its rows: the uploaded snapshot for a snapshot context,
   `live_context(lock, snapshot)` for a live one, none for a request one.
3. `served_context(lock, table)` gives the exact rows every path binds:
   the lock's split, projection to its columns in order, its roles' kinds
   and a classifier's integer labels; an uploaded snapshot is those rows
   already and comes back as it is. `compile_lock(lock, rows)` compiles them
   into a `CompiledPlan`, whose `spec.selection` every binder applies per
   query. `PredictorExecutor().prepare(plan, rows)` predicts in process;
   for gf-server, `published_entry(plan)` is the entry (its `strategy` is the
   selection) and `json_rows(bound_rows(plan, rows))` the rows, in the order
   gf serve binds them. `lock.digest` is the revision's identity and
   gf-server's fingerprint.
4. A request carries `PredictionRequest` fields: queries, a request
   context's rows, and `as_of` for a live context.

An ad hoc call resolves its manifest fragment with `ad_hoc_lock(manifest,
rows)` and runs the same way, its rows as a request context.

Compatibility: these names, their signatures and the `gf.lock` version 2
format are stable within a minor version of `generalforecast`; a change to
any of them is a minor-version change with a migration note, and the
recorded surface (`tests/conformance/test_hosting.py`) changes with it.
"""

from typing import Any

from generalforecast.contracts import (
    LOCK_VERSION,
    ContextImportRef,
    ContextVersion,
    Environment,
    PredictionRequest,
    PredictionResult,
    PredictorLock,
    PromoteRequest,
    Revision,
    RuntimePlan,
    context_version,
    lock_digest,
    lock_name,
    parse_lock,
)
from generalforecast.errors import Error
from generalforecast.models.catalog import (
    ALIASES,
    V1_POLICY,
    ModelRefError,
    entries,
    resource_name,
)
from generalforecast.models.catalog import parse as parse_model
from generalforecast.predictors.lock import upgrade as upgrade_lock
from generalforecast.predictors.models import CatalogResolver, ModelPin
from generalforecast.predictors.runtime import PredictorExecutor, PreparedPredictor
from generalforecast.predictors.runtime.compile import compile_lock
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.context import (
    json_rows,
    live_context,
    served_context,
)
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.runtime.requests import (
    bind_request,
    request_table,
    temporal,
)
from generalforecast.predictors.runtime.tabular import bound_order as bound_rows
from generalforecast.predictors.spec import PredictorSpec
from generalforecast.predictors.validation import PredictorError, bytes_digest
from generalforecast.serve.listener import binary, configuration
from generalforecast.serve.models import catalog_model, published_entry
from generalforecast.service.boundary import (
    Capability,
    JsonObject,
    Request,
    Resource,
    ResourceKind,
    Scope,
    failure,
    json_object,
    parse_body,
    success,
    unique_object,
)
from generalforecast.service.catalog import CatalogService
from generalforecast.service.contracts import (
    VERSION,
    InputError,
    RateLimit,
    ServiceError,
    fields,
    model_id,
    name,
    number,
    require,
    text,
)
from generalforecast.service.interfaces import (
    AcceptedWrite,
    ContextSnapshot,
    ImportOutcome,
    ModelPort,
    Mutation,
    NativeExecutor,
    ObservationInput,
    PreparedExecution,
    SnapshotRow,
    SourcePin,
    SourceRecord,
)
from generalforecast.tabular import (
    PHYSICAL_TYPES,
    arrow_bytes,
    coerce,
    read_table,
    table_digest,
    to_arrow,
)


def pin(model: str) -> ModelPin:
    """The pinned model a hosted predictor may run, under the hosted predictor gate.

    Raises:
        PredictorError: `ModelResolutionRequired` for a model the catalog does
            not qualify for hosted predictors, or one it does not know.
    """
    return CatalogResolver("predictor").resolve(model)


def task_of(model: str) -> str:
    """`classification`, `regression` or `forecast`, for a catalog model id."""
    from generalforecast.predictors import _registry

    return _registry.task_of(model)


def limits(model: str) -> Any:
    """A catalog model's per-call bounds: `context_rows`, `queries` and `inputs`."""
    from generalforecast.predictors import _registry

    return _registry.limits(model)


def ad_hoc_lock(manifest: dict[str, Any], rows: Any) -> PredictorLock:
    """The lock of an ad hoc call: its manifest fragment resolved against its rows.

    `manifest` takes the keys `gf.yaml` does (`model`, `target`, `inputs`,
    `id`, `time`, `series`, `frequency`, `horizon`, `quantiles`,
    `selection`); its context is the call's `rows`, as a request context.
    The call then compiles and runs as a revision does: `compile_lock(lock,
    rows, outputs=...)`, `PredictorExecutor().prepare(plan, None)` and
    `predict(queries, context=rows)`. Its selection is honoured.

    Raises:
        InvalidInput: For a fragment `gf check` would refuse, pointing at the
            field.
    """
    from generalforecast.predictors import lock, resolve

    body = {"name": "ad-hoc", **manifest, "context": {"request": {}}}
    spec = PredictorSpec.from_dict(body)
    return PredictorLock.from_dict(lock.document(resolve(spec, rows=to_arrow(rows))))


__all__ = [
    "ALIASES",
    "LOCK_VERSION",
    "PHYSICAL_TYPES",
    "V1_POLICY",
    "VERSION",
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
    "ModelPin",
    "ModelPort",
    "ModelRefError",
    "Mutation",
    "NativeExecutor",
    "ObservationInput",
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
    "ad_hoc_lock",
    "arrow_bytes",
    "binary",
    "bind_request",
    "bound_rows",
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
]
