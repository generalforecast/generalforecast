# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The hosting API: what a hosted control plane builds on to serve gf predictors.

A hosted plane stores pushed `gf.lock` documents and their
context rows, compiles and binds them with the same code `gf.Predictor` and
`gf serve` run, and publishes revisions to gf-server. Everything it needs
from gf is here, under these names; its modules import this one and no
private gf module.

The flow, for one revision:

1. `parse_lock` the pushed document; `pin(lock.model).digest` must equal
   the lock's `model_digest`, and an uploaded snapshot's digest the lock's
   `data_digest`. A document in another format is refused, with the hint
   to run `gf check`.
2. Gather its rows: the uploaded snapshot for a snapshot context,
   `live_context(lock, snapshot)` for a live one, none for a request one.
3. `served_context(lock, table)` gives the exact rows every path binds:
   the lock's split, projection to its columns in order, its roles' kinds
   and a classifier's integer labels; an uploaded snapshot is those rows
   already and comes back as it is. `compile_lock(lock, rows, manifest=...)`,
   given the normalized gf.yaml the push carried (`RevisionRegister.manifest`),
   holds the lock to it field by field and compiles them into a `CompiledPlan`, whose `spec.selection` every binder applies per
   query. `PredictorExecutor().prepare(plan, rows)` predicts in process;
   for gf-server, `published_entry(plan)` is the entry (its `strategy` is the
   selection) and `json_rows(bound_rows(plan, rows))` the rows, in the order
   gf serve binds them. `lock.digest` is the revision's identity and
   gf-server's fingerprint.
4. A request carries `PredictionRequest` fields: queries, a request
   context's rows, and `as_of` for a live context, read by `parse_as_of`.
   A reply's `context` is a `ContextVersion`'s `to_dict()`, its `as_of`
   written by `format_as_of`; gf-server reads and writes both the same way.

An ad hoc call resolves its manifest fragment with `ad_hoc_lock(manifest,
rows)` and runs the same way, its rows as a request context.

`as_of` through gf-server: when a prediction for a live-context predictor
carries `as_of`, gf-server asks the control plane for the context as of
that time through the internal revision lock route, passing `as_of` in
RFC 3339 (`format_as_of`); the plane answers with that version's rows, or a
422 (or 409) when the time precedes the history it retains. A prediction
for a snapshot or request context that carries `as_of` is refused with a
422 `ValidationError`, `param` `as_of`.

An offline image runs `python -m generalforecast.hosting prefetch` (or
`prefetch_catalog()`) once, to download, convert and verify every model a
hosted plane may serve.

Exports:

    Locks and revisions
        PredictorLock, parse_lock, lock_digest, lock_name: a pushed lock,
            read and identified.
        document_digest: a lock document's own digest, of its canonical
            JSON without the `digest` key; a valid lock's `digest` is it.
        Revision, Environment, PromoteRequest: a predictor's revisions and
            the environments that serve them.
        ContextImportRef, ContextVersion, context_version: a live context's
            import source, and the context a prediction read (its wire form
            and the decoder of current and stored forms).
        PredictionRequest, PredictionResult: a prediction call and reply.
        parse_as_of, format_as_of: `as_of` as requests carry it and replies
            write it (contract C1): `generalforecast.as_of.parse` and
            `format`, the one reading gf serve, the SDK and gf-server share.
    Compiling and predicting
        compile_lock, CompiledPlan: a lock and its rows, held to the pushed
            manifest and compiled.
        served_context, live_context, bound_rows, json_rows: the rows a
            revision binds, and how gf-server takes them.
        PredictorExecutor: in-process prediction of a compiled plan.
        CONTEXT_ROWS_KEY: the prediction table's schema metadata key holding
            how many context rows the call read.
        ad_hoc_lock: an ad hoc call's lock.
        published_entry, catalog_model, configuration, binary: gf-server's
            models file entries, its configuration and its executables.
        request_table, to_arrow, coerce, read_table, arrow_bytes,
            table_digest, bytes_digest, PHYSICAL_TYPES: rows in and out of
            Arrow, and their digests.
    Models
        pin, limits, entries, parse_model, resource_name, ALIASES,
            ModelRefError, V1_POLICY, CatalogService, prefetch_catalog: the
            catalog, its pins and bounds, and the serving policy.
    Service plumbing
        Request, Resource, ResourceKind, Scope, Capability, RateLimit,
            JsonObject, VERSION, parse_body, success, failure, json_object,
            unique_object: the request envelope and reply.
        Error, ServiceError, InputError, PredictorError: errors with codes.
        diagnostic_lines, report_diagnostic: a diagnostic in gf's cargo
            shape, as lines or written to standard error.
        fields, require, name, text, number, model_id: field checks with
            the same rules gf applies.
        ModelPort, NativeExecutor, PreparedExecution, AcceptedWrite,
            ImportOutcome, Mutation, ObservationInput, SnapshotRow,
            SourcePin, SourceRecord: the ports and records a plane
            implements or passes through.

Compatibility: these names, their signatures and the `gf.lock`
format are stable within a minor version of `generalforecast`; a change to
any of them is a minor-version change with a migration note, and the
recorded surface (`tests/conformance/test_hosting.py`) changes with it.
"""

import sys
from typing import Any

from generalforecast.as_of import format as format_as_of
from generalforecast.as_of import parse as parse_as_of
from generalforecast.contracts import (
    ContextImportRef,
    ContextVersion,
    Environment,
    PredictionRequest,
    PredictionResult,
    PredictorLock,
    PromoteRequest,
    Revision,
    context_version,
    lock_digest,
    lock_name,
    parse_lock,
)
from generalforecast.diagnostics import lines as diagnostic_lines
from generalforecast.diagnostics import report as report_diagnostic
from generalforecast.errors import Error
from generalforecast.models.catalog import (
    ALIASES,
    V1_POLICY,
    ModelRefError,
    entries,
    resource_name,
)
from generalforecast.models.catalog import parse as parse_model
from generalforecast.predictors.lock import lock_digest as document_digest
from generalforecast.predictors.models import CatalogResolver, ModelPin
from generalforecast.predictors.runtime import CONTEXT_ROWS_KEY, PredictorExecutor
from generalforecast.predictors.runtime import compile as _compile
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.context import (
    json_rows,
    live_context,
    served_context,
)
from generalforecast.predictors.runtime.requests import request_table
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


def limits(model: str) -> Any:
    """A catalog model's per-call bounds: `context_rows`, `queries` and `inputs`."""
    from generalforecast.predictors import _registry

    return _registry.limits(model)


def compile_lock(
    lock: Any,
    rows: Any,
    *,
    manifest: dict[str, Any] | None,
    outputs: tuple[str, ...] | list[str] | None = None,
) -> CompiledPlan:
    """A revision's lock and context rows, held to its manifest and compiled.

    `manifest` is the normalized `gf.yaml` the push carried
    (`RevisionRegister.manifest`); the lock must be what `gf check` writes
    for it and these rows. It is required so a plane cannot skip the check
    by omission: pass None only for a lock built here by `ad_hoc_lock`.

    Args:
        lock: A `PredictorLock`, or a lock document.
        rows: `served_context`'s rows for a snapshot or live context; None
            for a request context.
        manifest: The pushed manifest, or None for an ad hoc lock.
        outputs: An ad hoc call's own outputs; the lock's otherwise.

    Raises:
        PredictorError: `LockMismatch` naming the first field that differs
            from the manifest, `SnapshotMismatch` for rows other than the
            lock's, and the refusals of compiling.
    """
    return _compile.compile_lock(lock, rows, outputs=outputs, manifest=manifest)


def ad_hoc_lock(manifest: dict[str, Any], rows: Any) -> PredictorLock:
    """The lock of an ad hoc call: its manifest fragment resolved against its rows.

    `manifest` takes the keys `gf.yaml` does (`model`, `target`, `inputs`,
    `id`, `time`, `series`, `frequency`, `horizon`, `quantiles`,
    `selection`); its context is the call's `rows`, as a request context.
    The call then compiles and runs as a revision does: `compile_lock(lock,
    rows, manifest=None, outputs=...)`, `PredictorExecutor().prepare(plan, None)` and
    `predict(queries, context=rows)`. Its selection is honoured.

    Raises:
        InvalidInput: For a fragment `gf check` would refuse, pointing at the
            field.
    """
    from generalforecast.predictors import lock, resolve

    body = {"name": "ad-hoc", **manifest, "context": {"request": {}}}
    spec = PredictorSpec.from_dict(body)
    return PredictorLock.from_dict(lock.document(resolve(spec, rows=to_arrow(rows))))


def prefetch_catalog() -> None:
    """Downloads, converts and verifies every model a hosted plane may serve.

    An offline image runs this once at build time, as `python -m
    generalforecast.hosting prefetch`; afterwards the plane serves with
    downloads off.

    Raises:
        ServiceError: For a download or a file that does not match its pin.
    """
    from generalforecast.service.catalog_artifacts import preload

    preload()


__all__ = [
    "ALIASES",
    "CONTEXT_ROWS_KEY",
    "PHYSICAL_TYPES",
    "V1_POLICY",
    "VERSION",
    "AcceptedWrite",
    "Capability",
    "CatalogService",
    "CompiledPlan",
    "ContextImportRef",
    "ContextVersion",
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
    "limits",
    "live_context",
    "lock_digest",
    "lock_name",
    "model_id",
    "name",
    "number",
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
    "unique_object",
]


def _main(argv: list[str]) -> int:
    """`python -m generalforecast.hosting prefetch`: the image build's model download."""
    if argv != ["prefetch"]:
        print("usage: python -m generalforecast.hosting prefetch", file=sys.stderr)
        return 2
    prefetch_catalog()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
