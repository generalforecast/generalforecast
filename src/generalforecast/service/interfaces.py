# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ports a predictor service implements: data, sources, models, operators, identity, execution.

`gf serve` and hosted serving implement them; an adapter authorizes a
request, invokes a port and wraps its result with `service.boundary`.
`NativeExecutor` is the one execution port: `PredictorExecutor` implements
it in process, binding a `CompiledPlan`. These are part of the hosting API,
`generalforecast.hosting`.
"""

from collections.abc import Iterator
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from generalforecast.service.boundary import JsonObject, Scope

if TYPE_CHECKING:
    import pyarrow as pa

    from generalforecast.predictors.runtime.compiled import CompiledPlan

WriteOperation = Literal[
    "create_features",
    "attach_labels",
    "correct_features",
    "correct_labels",
    "resupply_labels",
    "import_record",
]
LabelValue = bool | float | str


class ObservationInput(TypedDict):
    """Independently guarded components of one logical observation."""

    key: str
    features: NotRequired[dict[str, float]]
    labels: NotRequired[dict[str, LabelValue]]
    event_at: NotRequired[float]
    expected_feature_version: NotRequired[int]
    expected_label_versions: NotRequired[dict[str, int]]


class Mutation(TypedDict):
    """One bounded atomic ordinary request in the context write-ID namespace."""

    write_id: str
    operation: WriteOperation
    records: list[ObservationInput]


class ComponentVersions(TypedDict):
    """Independent current component versions, never a single row version."""

    key: str
    feature_version: int | None
    label_versions: dict[str, int]


class AcceptedWrite(TypedDict):
    """Original durable receipt; no serving readiness assertion."""

    project: str
    context: str
    write_id: str
    operation: WriteOperation
    sequence: int
    accepted_at: float
    durably_accepted: bool
    changes: list[ComponentVersions]
    retry_valid_until: float | None


class SnapshotRow(TypedDict):
    """Only the requested target and versions known at the requested instant."""

    key: str
    features: dict[str, float]
    label: LabelValue
    feature_version: int
    label_version: int
    feature_accepted_at: float
    label_accepted_at: float
    event_at: float | None


class ContextSnapshot(TypedDict):
    """A transactionally consistent bounded history view with explicit clocks."""

    project: str
    context: str
    target: str
    features: list[str]
    through_sequence: int
    knowledge_at: float
    event_before: float | None
    rows: list[SnapshotRow]
    source_wait_count: int
    pending_processing_count: int
    oldest_unprocessed_at: float | None


class EffectStatus(TypedDict):
    """Integration-owned progress for one exact revision and data receipt."""

    effect: JsonObject
    state: Literal[
        "queued", "processing", "source_wait", "visible", "failed", "superseded"
    ]
    accepted_at: float
    processed_at: float | None
    visible_at: float | None
    oldest_processing_at: float | None
    source_wait_count: int
    error: JsonObject | None


class ImportOutcome(TypedDict):
    """One source occurrence; infrastructure failures raise, not row-conflict."""

    ordinal: int
    status: Literal["accepted", "skipped", "conflict"]
    receipt: AcceptedWrite | None
    skipped_components: list[str]
    issue: JsonObject | None


class SourcePin(TypedDict):
    """Pinned bytes under the configured source root, not arbitrary URL access."""

    source_id: str
    digest: str
    size_bytes: int
    format: Literal["jsonl", "csv"]
    mapping: JsonObject


class SourceRecord(TypedDict):
    """Stable ordered occurrence; next_offset is the durable resume checkpoint."""

    ordinal: int
    offset: int
    next_offset: int
    record: ObservationInput | None
    issue: JsonObject | None


class DataPort(Protocol):
    """A: component admission/history, called by C and integration."""

    def create(self, scope: Scope, config: JsonObject) -> JsonObject:
        """Creates a bounded versioned context, never overwriting its schema."""
        ...

    def list(self, scope: Scope) -> list[JsonObject]:
        """Lists only contexts in this project."""
        ...

    def get(self, scope: Scope, context: str) -> JsonObject:
        """Returns schema, capacities and separate pending/processing counts."""
        ...

    def mutate(self, scope: Scope, context: str, mutation: Mutation) -> AcceptedWrite:
        """Atomically accepts all components or replays the original receipt."""
        ...

    def snapshot(
        self,
        scope: Scope,
        context: str,
        *,
        target: str,
        knowledge_at: float | None = None,
        event_before: float | None = None,
    ) -> ContextSnapshot:
        """Selects acceptance <= knowledge_at and event time < event_before."""
        ...

    def import_record(
        self,
        scope: Scope,
        context: str,
        *,
        write_id: str,
        ordinal: int,
        record: ObservationInput,
        skip_identical: bool,
    ) -> ImportOutcome:
        """Admits one record atomically; skips never renew versions or deadlines."""
        ...

    def expire(self, scope: Scope, context: str) -> int:
        """Durably expires due pending labels using the injected acceptance clock."""
        ...


class SourcePort(Protocol):
    """A: pinned source reader for C, without owning revision coordination."""

    def pin(self, scope: Scope, spec: JsonObject) -> SourcePin:
        """Copies bounded regular JSONL/CSV bytes into a private immutable spool."""
        ...

    def list(self, scope: Scope) -> list[SourcePin]:
        """Lists project-scoped pinned source metadata, not private file paths."""
        ...

    def read(
        self,
        scope: Scope,
        pin: SourcePin,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Iterator[SourceRecord]:
        """Verifies pinned identity and yields stable source order/checkpoints."""
        ...


class ModelPort(Protocol):
    """B: the curated catalog gf-server serves; predictions go to gf-server."""

    def list(self, scope: Scope) -> list[JsonObject]:
        """Lists pins, license, bounds and supported/unconfigured capabilities."""
        ...

    def get(self, scope: Scope, model: str) -> JsonObject:
        """Gets one curated model, without downloading or executing it."""
        ...


class OperatorPort(Protocol):
    """D: local-state lifecycle proof, not hosted placement or recovery."""

    def doctor(self, scope: Scope) -> JsonObject:
        """Returns redacted runtime/state/capability checks."""
        ...


class IdentityPort(Protocol):
    """E: authentication is separate from routing and project selection."""

    def authenticate(self, token: str) -> JsonObject:
        """Returns a redacted principal; never echoes tokens or secret material."""
        ...

    def authorize(self, principal: JsonObject, scope: Scope, permission: str) -> None:
        """Checks project and action on EVERY request, including read/list."""
        ...

    def manage(self, scope: Scope, action: str, payload: JsonObject) -> JsonObject:
        """Executes integration-allowlisted identity/key/secret metadata actions."""
        ...


@runtime_checkable
class PreparedExecution(Protocol):
    """One published worker lifecycle, shared by native and isolated executions.

    `retire` only stops new leases and performs no provider I/O, so callers may
    hold the journal lock. `close` retires, then attempts verified teardown
    outside that lock: True once resources are verified released, False while
    admitted predictions drain, raising when teardown is unverified.
    """

    @property
    def context_rows(self) -> int | None:
        """Bound context rows actually used, or None for request/no context."""
        ...

    @property
    def usable(self) -> bool:
        """Whether a published view may still admit new predictions."""
        ...

    def predict(
        self, queries: "pa.Table", *, context: "pa.Table | None" = None
    ) -> "pa.Table":
        """Score queries under a lease that retirement drains before teardown."""
        ...

    def retire(self) -> None:
        """Stop admitting new leases without provider I/O."""
        ...

    def close(self) -> bool:
        """Retire and attempt verified teardown; see the class contract."""
        ...


class ResourceFree:
    """Explicit lifecycle for prepared executions owning no external resources."""

    __slots__ = ()

    @property
    def usable(self) -> bool:
        """Owns nothing that can expire or be torn down."""
        return True

    def retire(self) -> None:
        """Holds no leases to stop."""

    def close(self) -> bool:
        """Has nothing to release, so teardown is trivially verified."""
        return True


class NativeExecutor(Protocol):
    """Trusted curated-model loading; never accepts uploaded Python code."""

    def prepare(
        self, lock: "CompiledPlan", context: "pa.Table | None"
    ) -> PreparedExecution:
        """Admit a compiled plan, load verified weights and bind any snapshot context."""
        ...
