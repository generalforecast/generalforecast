# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Types for generalforecast._core, the Rust control plane built from crates/gf-py."""

from typing import Literal, Self

import numpy as np
import numpy.typing as npt

type Keys = npt.NDArray[np.uint64]
"""128-bit keys as an (n, 2) array of each key's low and high 64 bits."""
type Features = npt.NDArray[np.float32]
type Targets = npt.NDArray[np.int32] | npt.NDArray[np.float32]
type Times = npt.NDArray[np.int64]
type RowIds = npt.NDArray[np.uint32]
type CoreKind = Literal["random", "stratified", "balanced", "diverse"]

# Rows a table's state is fit on unless it says otherwise.
DEFAULT_FIT_ROWS: int

def bucket_size(rows: int, minimum: int = 8) -> int:
    """The padded size for `rows` rows: `minimum` doubled until it fits."""

type Surface = Literal["local", "remote", "self-hosted"]
type Label = str | bool | int

class RequestError(ValueError):
    """A refused inference request: `args` are its code, message and HTTP status."""

def resolve_model(spec: object, surface: Surface) -> str:
    """The canonical reference a model spec names, if the surface may serve it.

    `surface` is `remote` (the hosted Models API), `self-hosted` (a server its
    operator runs) or `local` (a caller that admitted the model already).
    Raises `RequestError` as the Models API refuses the reference.
    """

def quantile_levels(payload: bytes) -> list[float]:
    """Requested quantile levels, from their JSON list: strictly increasing, in (0, 1).

    Raises `RequestError` for levels a request may not select.
    """

class TabularRequest:
    """A classification or regression request, validated before any weights load."""

    model: str
    """The canonical reference of the model it resolved to."""
    task: Literal["classification", "regression"]
    target: str
    features: list[str]
    context: Features
    """The labeled context rows' features, `(rows, features)`."""
    targets: Targets
    """Per labeled row, its class code into `classes` (int32) or its value (float32)."""
    classes: list[Label]
    """The classes in output order; empty for a regression."""
    queries: Features
    """The queries' features, `(queries, features)`."""
    query_keys: list[str]
    levels: list[float]
    """The quantile levels a regression returns; empty for a classification."""
    supplied_rows: int
    """Context rows supplied, labeled or not."""
    fingerprint: str
    """SHA-256 of the model and its context, as hex: same context, same table."""

class AlignedGroup:
    """Related series aligned on one time axis, unpadded: the last step is the origin.

    Rows are the group's targets in query order, then its covariates sorted
    by name; execution left-pads time to the model's patch.
    """

    values: Features
    """`(rows, steps)`, zero where unobserved."""
    observed: npt.NDArray[np.bool_]
    roles: npt.NDArray[np.int32]
    """Per row, `TARGET`, `PAST` or `KNOWN`."""
    future: Features
    """Known covariates' values over the horizon, `(rows, horizon)`."""
    future_observed: npt.NDArray[np.bool_]

class ForecastQuery:
    """One forecast query: a series forecast with its group from an origin."""

    key: str
    series: str
    origin: int | float
    """An integer period or floating UTC seconds."""
    members: list[str]
    """The group's target series, in row order."""
    member: int
    """The row of `series` among `members`."""
    times: list[int | float]
    """The forecast's times, one step after another past the origin."""
    group: AlignedGroup
    fingerprint: str
    """SHA-256 of the context fingerprint and this group, as hex: its members,
    origin, horizon and known-future covariates.
    """

class ForecastRequest:
    """A forecast request, validated and aligned before any weights load."""

    model: str
    """The canonical reference of the model it resolved to."""
    target: str
    step: int | float
    """An integer number of periods or floating seconds."""
    horizon: int
    levels: list[float]
    known_covariates: list[str]
    queries: list[ForecastQuery]
    time_axis: Literal["integer-periods", "utc-seconds"]
    """`integer-periods` or `utc-seconds`."""
    supplied_rows: int
    """Context rows supplied, observed or not."""
    usable_observations: int
    """Context rows with an observed value."""
    fingerprint: str
    """SHA-256 of the model and its context, as hex: same context, same table."""

class PredictorContext:
    """A tabular predictor's context, bound once.

    Its labeled rows encoded as the engine reads them, and the vocabularies
    its queries are encoded against.
    """

    @property
    def features(self) -> Features:
        """The labeled rows' features, `(rows, width)`."""
    @property
    def targets(self) -> Targets:
        """Per labeled row, its class code into `classes` (int32) or its value (float32)."""
    @property
    def classes(self) -> list[Label]:
        """The classes in output order; empty for a regression."""
    @property
    def rows(self) -> int:
        """Labeled context rows."""
    @property
    def vocabularies(self) -> list[list[str]]:
        """Each input column's words in code order; empty but for categorical columns."""
    def encode(self, queries: bytes) -> Features:
        """Query rows from an Arrow IPC stream, encoded as `(rows, width)`.

        Words are coded with the context's vocabularies; a word the context
        never held is NaN. Raises `RequestError` for a missing column or values of another kind.
        """

def bind_predictor(
    schema: bytes,
    task: Literal["classification", "regression"],
    target: str,
    context: bytes,
) -> PredictorContext:
    """Binds a tabular predictor's context from its lock's schema and its rows.

    `schema` is the lock's `TableSchema.to_dict()` as JSON, `task` is
    `classification` or `regression`, and `context` is the rows as an Arrow
    IPC stream. Only labeled rows are kept. Raises `RequestError`:
    `EmptyContext` for no labeled rows, `InvalidPredictor` for fewer than two
    classes or columns that hold values of another kind.
    """

def parse_inference(
    payload: bytes,
    surface: Surface,
    *,
    limits: Literal["request", "predictor"] = "request",
    levels: list[float] | None = None,
) -> TabularRequest | ForecastRequest:
    """Parses and validates an inference request's JSON payload, without the GIL.

    `surface` decides the gates its model passes, as for `resolve_model`.
    `limits` is `request` for one Models API request's bounds or `predictor`
    for a predictor's bound context, which its lock bounds instead. `levels`
    are the quantile levels a forecaster outputs, when its weights are
    loaded; the catalog's declared levels otherwise. Raises `RequestError`
    with the code, message and status the Models API answers.
    """

class Strategy:
    """How each query's context is chosen.

    Each constructor raises `ValueError` for a count that is negative, zero,
    or more rows than a table can hold.
    """

    @staticmethod
    def exact() -> Strategy:
        """Every live row, for every query."""
    @staticmethod
    def adaptive() -> Strategy:
        """What fits the table's size, resolved from its live rows at each plan.

        Exact up to the model's context cap, and past it a stratified core of
        a quarter of the cap plus the nearest rows that fill the rest, the
        neighbors leaning to the newest rows when the table has times.
        """
    def resolve(self, rows: int, cap: int | None, timed: bool = False) -> Strategy:
        """The strategy a table of `rows` live rows plans with.

        `cap` is the model's context rows, or None for no cap; `timed` whether
        the table's rows have times. An adaptive strategy is resolved by the
        server's rule, any other is itself.
        """
    @staticmethod
    def subsample(rows: int, seed: int = 0, kind: CoreKind = "random") -> Strategy:
        """The same deterministic sample of rows for every query: a core alone.

        `kind` is `random`, `stratified` (by class, or by target bin for a
        regression), `balanced` (an equal share per class or bin) or `diverse`
        (farthest-point sampling).
        """
    @staticmethod
    def neighbors(rows: int) -> Strategy:
        """Each query's nearest rows, without a core."""
    @staticmethod
    def retrieved(
        neighbors: int,
        core: int = 0,
        seed: int = 0,
        kind: CoreKind = "random",
        among: int | None = None,
        last: int | None = None,
        span: int | None = None,
    ) -> Strategy:
        """A shared core plus each query's nearest rows outside it, encoded together.

        The core is drawn as `subsample` draws one; a `core` of 0 is no core.
        With `among`, the neighbors are the latest of that many nearest rows.
        With `last` or `span`, the whole context comes from the window's
        latest rows, or its rows at most `span` before its newest row's time.
        """
    def spec(self) -> str:
        """The strategy as the JSON a server's `create_table` takes."""
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class GroupStrategy:
    """Which other series join a series in its group."""

    @staticmethod
    def alone() -> GroupStrategy:
        """Each series alone, with its own covariate columns."""
    @staticmethod
    def random(size: int, seed: int) -> GroupStrategy:
        """`size - 1` others by seeded hash with the series; a table holding fewer refuses the request."""
    @staticmethod
    def given(size: int) -> GroupStrategy:
        """The members each forecast names with `forecast_given`, up to `size` series in all; the table draws none."""
    def size(self) -> int:
        """The most series in a group, the forecast one included: 1 alone."""
    def spec(self) -> str:
        """The strategy as the JSON a server's `create_series_table` takes."""
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class CacheKey:
    """What a cache entry is keyed on.

    Its model, table, epoch and kind, and the fingerprint of its rows.
    """

    model: int
    table: int
    epoch: int
    kind: Literal["rows", "context"]
    """`rows` for a page's row representations, `context` for a unit's context."""
    rows: int
    def __new__(
        cls,
        model: int,
        table: int,
        epoch: int,
        kind: Literal["rows", "context"],
        rows: int,
    ) -> Self: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class Fit:
    """Rows to fit a table's state on, which also embeds them into `pages`."""

    table: int
    epoch: int
    rows: RowIds
    features: Features
    targets: Targets
    pages: list[int]
    """The step's pages these rows fill, in order."""

class Page:
    """The representations of one page's rows.

    `features` and `targets` are set when the worker embeds them.
    """

    table: int
    key: CacheKey
    rows: RowIds
    action: Literal["encode", "reuse", "transient"]
    """`encode` (embed and keep), `reuse` or `transient`."""
    slots: list[int]
    """The host slots kept for the representations of a page to encode."""
    features: Features | None
    targets: Targets | None
    copies: list[tuple[CacheKey, list[tuple[int, int]]]]
    """Cached pages to copy rows from, each with (position in the source,
    position in `rows`) pairs; `features` holds the other rows."""

class Unit:
    """One unit of a step: rows to encode or reuse.

    `features` and `targets` are set when the worker fits the unit's own state.
    """

    table: int
    key: CacheKey
    rows: RowIds
    action: Literal["encode", "reuse", "transient"]
    """`encode`, `reuse` or `transient`."""
    slots: list[int]
    features: Features | None
    targets: Targets | None

class QueryBatch:
    """One request's queries, and the unit each decodes against."""

    request: int
    table: int
    epoch: int
    features: Features
    units: list[int]

class Grid:
    """A group of series as the model reads it.

    A dense grid of values, which of them are observed, and each row's role.
    """

    values: Features
    observed: npt.NDArray[np.bool_]
    roles: npt.NDArray[np.int32]

class SeriesUnit:
    """A unit of series: a group's context to encode, reuse or forecast whole."""

    table: int
    key: CacheKey
    members: list[int]
    """The group's series, in the order the grid's rows take.

    Each has its target row, then one row per feature column.
    """
    cutoff: int
    grid: Grid
    action: Literal["encode", "reuse", "transient", "extend"]
    """`encode`, `reuse`, `extend` or `transient`; a unit forecast whole is
    transient.
    """
    slots: list[int]
    source: CacheKey | None
    """For an extension, the entry holding the shorter encoding it grows."""
    grown: int
    """For an extension, how many steps at the grid's end are new."""

class Forecast:
    """One series' forecast.

    The step's series unit it reads, the member it forecasts, its horizon and
    the group's known future values.
    """

    request: int
    table: int
    epoch: int
    unit: int
    target: int
    horizon: int
    future: Grid

class Movement:
    """A copy or drop the worker makes before running a step."""

    key: CacheKey
    source: Literal["device", "host", "disk"]
    target: Literal["device", "host", "disk"] | None
    """The tier the entry moves to, or None if it is dropped."""
    slots: list[int]
    """The slots it moves into there."""

class Step:
    """What a worker runs, in order: movements, fits, pages, units, queries, forecasts."""

    movements: list[Movement]
    fits: list[Fit]
    pages: list[Page]
    units: list[Unit]
    queries: list[QueryBatch]
    series_units: list[SeriesUnit]
    forecasts: list[Forecast]

class Node:
    """Many tables behind one request queue and one cache.

    Calls raise `KeyError` for a table or key that does not exist, `ValueError`
    for input the node refuses, such as a negative size, a label outside the
    table's classes or rows of the wrong width, and `RuntimeError` when the
    cache or planner fails. A call that fails changes nothing.

    Not for concurrent use: a call made while another thread's runs raises
    `RuntimeError`. `LocalClient` makes threads take turns.
    """

    def __init__(
        self,
        device_bytes: int,
        host_bytes: int,
        slot_bytes: int = 1 << 20,
        disk_bytes: int = 0,
        step_bytes: int | None = None,
        eviction: Literal["gdsf", "lru"] = "gdsf",
    ) -> None: ...
    def create_table(
        self,
        model: int,
        features: int,
        classes: int,
        strategy: Strategy,
        bytes_per_row: int,
        bytes_per_representation: int = 0,
        page_size: int = 128,
        bucket_minimum: int = 8,
        time_bucket: int | None = None,
        refit_after: float | None = None,
        refit_shift: float | None = None,
        fit_rows: int | None = None,
        context_rows: int | None = None,
        past_cap: Literal["warn", "clamp"] = "warn",
        region_rows: int | None = None,
        rebuild: Literal["always", "never", "churn", "every"] = "always",
        rebuild_at: float = 0.0,
        rebuild_steps: int = 8,
        rebuild_seconds: float | None = 1.0,
        refit_cutoffs: Literal["each", "changed", "never"] = "each",
        scale_rows: int | None = None,
        retrieval: int = 0,
        summary: Literal["table", "context"] = "table",
        state_bytes: int = 0,
        categorical: list[int] = [],
        retrieval_categoricals: Literal["codes", "one_hot", "target"] = "codes",
        unlabeled_context: bool = False,
        fit_bound: int | None = None,
    ) -> int:
        """Adds an empty table and returns its id. `classes` is 0 for regression.

        `summary` is `table` or `context`; a table that summarizes each context
        keeps `state_bytes` of state beside each unit's rows. A fit on more
        than `fit_bound` rows waits for `set_fit_limits`; None never waits.
        """
    def insert(
        self,
        table: int,
        keys: Keys,
        features: Features,
        targets: Targets,
        times: Times | None = None,
    ) -> RowIds:
        """Adds rows, at `times` if given; returns their row ids."""
    def update(
        self,
        table: int,
        keys: Keys,
        features: Features,
        targets: Targets,
        times: Times | None = None,
    ) -> RowIds:
        """Changes rows, and their times if given; returns their row ids."""
    def label(self, table: int, keys: Keys, targets: Targets) -> RowIds:
        """Sets rows' targets, keeping their features and times; returns their row ids."""
    def upsert(
        self,
        table: int,
        keys: Keys,
        features: Features,
        targets: Targets,
        times: Times | None = None,
    ) -> RowIds:
        """Adds rows or replaces them by key, at `times` if given; returns their row ids."""
    def present(self, table: int, keys: Keys) -> Keys:
        """The keys among `keys` that a table holds, in order."""
    def delete(self, table: int, keys: Keys) -> RowIds:
        """Deletes rows; returns their row ids."""
    def truncate(self, table: int) -> RowIds:
        """Deletes every row of a table; returns their row ids."""
    def len(self, table: int) -> int:
        """Number of live rows in a table."""
    def submit(self, table: int, queries: Features, before: Times | None = None) -> int:
        """Queues query rows for a table; returns the request id.

        Each query reads only rows before its `before` time, if given.
        """
    def create_series_table(
        self,
        model: int,
        features: int,
        step: int,
        patch: int,
        context: int,
        output_patch: int,
        group: GroupStrategy,
        bytes_per_row: int,
        pass_limit: int | None = None,
        patch_bytes: int = 0,
        known: list[int] = [],
        max_rows: int | None = None,
        neighbors: Literal["targets", "covariates"] = "targets",
        reusable: bool = True,
        bucket_minimum: int = 1,
        extendable: bool = False,
    ) -> int:
        """Adds an empty table of series and returns its id.

        Its rows are observations keyed by series and time, with `features`
        covariate columns, of which `known` are known over the horizon; the
        grid's `step` is in the table's time unit. `patch`, `output_patch`,
        `context`, `pass_limit`, `max_rows`, `reusable` and `extendable` come
        from the model's properties. A group's cache takes `bytes_per_row`
        per padded row and `patch_bytes` beside them, per patch.
        """
    def submit_forecast(
        self,
        table: int,
        series: npt.NDArray[np.uint64],
        horizon: int,
        before: Times | None = None,
        groups: list[npt.NDArray[np.uint64]] | None = None,
    ) -> int:
        """Queues forecasts of a series table's series over `horizon` steps.

        Each is from its `before` time, or the series' latest observed time plus a step
        without one; on a table of given groups, each series' group is in `groups`.
        Returns the request id.
        """
    def submit_keys(self, table: int, queries: Features, contexts: list[Keys]) -> int:
        """Queues query rows that each read exactly their given rows; returns the request id."""
    def cancel(self, request: int) -> bool:
        """Drops a queued request before it is planned; returns whether it was queued."""
    def schedule(self) -> Step | None:
        """Plans pending requests into one step, or None if none can run yet."""
    def take_drops(self) -> list[Movement]:
        """Drops of cached entries holding rows since changed or deleted, to apply now."""
    def complete(self, step: Step) -> list[Movement]:
        """Releases a finished step's units; returns drops of entries from ended epochs."""
    def fail(self, step: Step) -> list[Movement]:
        """Undoes a step the worker failed to run.

        Returns drops of the entries it was to encode, and of entries from ended
        epochs.
        """
    def contexts(
        self, table: int, queries: Features, before: Times | None = None
    ) -> list[Keys]:
        """Each query's context keys, as `submit` would plan them, queueing nothing."""
    def region_churn(self, table: int, queries: Features) -> npt.NDArray[np.float64]:
        """Each query's region churn, as `Rebuild::Churn` compares it; NaN if unbuilt or uncached."""
    def take_unmeasured(self) -> list[int]:
        """Tables that came to wait for their fit's rows to be measured, each once."""
    def set_fit_limits(self, table: int, state_rows: int, fused_rows: int) -> None:
        """Sets the rows a table's state alone and a fit with its rows take, as measured."""
    def effective_fit_rows(self, table: int) -> int:
        """The most rows a table's state is fit on, within the worker's call budget."""
    def region_stats(self, table: int) -> dict[str, float] | None:
        """What a table's regions served and their rebuilds cost; None without regions."""
    def take_past_cap(self) -> list[int]:
        """Tables whose contexts first passed their model's trained length, each once."""
    def take_refused(self) -> list[tuple[int, str, bool]]:
        """Requests planning refused since the last call, with why and whether it names what does not exist."""
    def forget_cache(self) -> None:
        """Forgets every cached entry and fit, as when the worker starts again empty.

        Call it with no step in flight.
        """
    def remove_table(self, table: int) -> list[Movement]:
        """Removes a table; returns drops of its cache entries.

        Its queued requests are refused; see `take_refused`.
        """
    def refit(self, table: int) -> list[Movement]:
        """Refits a table's state and retrieval scaling.

        Returns drops of old entries nothing reads.
        """
    def set_strategy(self, table: int, strategy: Strategy) -> None:
        """Changes how a table chooses contexts from here on; see `Node.strategy`."""
    def strategy(self, table: int) -> Strategy:
        """The strategy a table's next plan chooses contexts with.

        An adaptive strategy resolved against the table's live rows, or the
        one it was given.
        """

class BoundTable:
    """A context bound as a durable table, which lives until `Engine.unbind`."""

    @property
    def name(self) -> str:
        """The table's name."""
    @property
    def levels(self) -> list[float] | None:
        """The quantile levels a regression's outputs are; None for a classification."""

class Engine:
    """gf-server's shards in this process, sized as `gf serve` sizes them.

    One worker per GPU, or on CPU one per run of the configuration's
    `cpu_threads` cores, pinned to them.

    Calls release the GIL while they wait, so concurrent callers' queries
    share the shards' steps.
    """

    def __init__(self, config: str) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def shards(self) -> int:
        """How many shards, and so worker processes, the engine runs."""
    def serve(self) -> tuple[str, str]:
        """Serves gf-server's Flight `Service` on these shards to this process alone.

        Returns `(location, token)`: a Unix socket in an owner-only directory
        under the data directory, or a loopback port where there are none, and
        the random token it accepts. The first call starts it; `close` stops it.
        """
    def add_model(self, name: str, spec: str) -> None:
        """Has every worker load a model under `name` from `spec`.

        `spec` is a built-in model's repository id; a worker serving that
        name already keeps it.
        """
    def tabular(
        self, model: str, request: TabularRequest
    ) -> tuple[npt.NDArray[np.float32], list[float] | None]:
        """Scores a validated request's queries against its context.

        The context is the ephemeral table its fingerprint names. Returns
        `(queries, outputs)` and the quantile levels of a regression's outputs.
        """
    def forecast(
        self, model: str, request: ForecastRequest
    ) -> tuple[npt.NDArray[np.float32], list[float] | None]:
        """Forecasts every query of a validated request.

        Each aligned group reads the ephemeral table its fingerprint names.
        Returns `(queries, horizon, levels)` and the model's quantile levels.
        """
    def bind(
        self,
        model: str,
        features: npt.NDArray[np.float32],
        targets: npt.NDArray[np.int32] | npt.NDArray[np.float32],
        classes: int,
    ) -> BoundTable:
        """Binds labeled rows once as a durable table.

        `features` is `(rows, columns)`; `targets` holds class codes (int32)
        when `classes` is positive, else values (float32).
        """
    def predict(
        self, bound: BoundTable, features: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], list[float] | None]:
        """Scores query features `(queries, columns)` against a bound table."""
    def unbind(self, bound: BoundTable) -> None:
        """Drops a bound table and its log."""
    def close(self) -> None:
        """Stops the shards and their workers once calls in progress return.

        Later calls raise.
        """
