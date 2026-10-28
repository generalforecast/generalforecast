# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`generalforecast.Predictor`: one object for local, ad hoc, served and hosted predictors.

A predictor is a catalog model, a target, inputs and a context. A local
predictor's `gf.yaml` resolves as `gf check` resolves it, and an ad hoc
predictor's settings resolve as the hosted ad hoc route resolves them, into
a lock document. `run_local` compiles that document and binds its
context rows with `PredictorExecutor`, the executor hosted serving runs, so
types, past and future covariates, `max_rows` and selection are honoured
exactly as `gf serve` and hosted serving honour them. A served or hosted
predictor is called over HTTP.
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast import tabular
from generalforecast.errors import (
    Error,
    InvalidInput,
    boundary,
)
from generalforecast.evaluation import (
    Evaluation,
    _when,
    classification_metrics,
    forecast_metrics,
    regression_metrics,
)
from generalforecast.native import models as registry
from generalforecast.predict_request import (
    PredictRequest,
    _empty,
    _frame_of_result,
    _with_forecast_keys,
    run_local,
)
from generalforecast.predictors.validation import NAME
from generalforecast.references import resolve_reference

pc: Any = _compute
ENVIRONMENTS = ("staging", "production")
_HOLDOUT_FRACTION = 0.2


def _query_batches(
    table: pa.Table, series: Sequence[str], cap: int
) -> list[np.ndarray]:
    """The row indices of each call, at most `cap` rows apiece, in row order.

    With series columns, each series' rows stay in one call, as a
    forecaster needs a series' whole horizon at once; series are packed in
    order of first appearance. A series of more than `cap` rows is a call
    of its own, which the server bounds.
    """
    if not series:
        return [
            np.arange(start, min(start + cap, table.num_rows))
            for start in range(0, table.num_rows, cap)
        ]
    keys = list(zip(*(table[name].to_pylist() for name in series), strict=True))
    members: dict[Any, list[int]] = {}
    for index, key in enumerate(keys):
        members.setdefault(key, []).append(index)
    batches: list[list[int]] = [[]]
    for rows in members.values():
        if batches[-1] and len(batches[-1]) + len(rows) > cap:
            batches.append([])
        batches[-1].extend(rows)
    return [np.asarray(sorted(rows), dtype=np.int64) for rows in batches if rows]


@dataclass(frozen=True)
class Context:
    """What a predictor learns from: its kind and how many rows it holds."""

    kind: str
    rows: int | None = None
    name: str | None = None
    at: Any = None
    _owner: Any = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:  # noqa: D105
        rows = "" if self.rows is None else f" {self.rows:,} rows"
        return f"Context({self.kind}{rows})"

    def _live(self) -> Any:
        if self.kind != "live" or self._owner is None:
            raise InvalidInput(
                f"a {self.kind} context cannot change",
                hint="only a live context takes append, label and as_of",
            )
        return self._owner

    def append(self, rows: Any, *, idempotency_key: str | None = None) -> Any:
        """Appends rows to a live context, each carrying the predictor's id column.

        Args:
            rows: The rows to append.
            idempotency_key: The write's durable id, so a retried append is
                applied once; a fresh one if None.
        """
        owner = self._live()
        identity = owner._id_name()
        if identity is None:
            raise InvalidInput(
                "a live context's rows need an id column",
                hint="declare `id:` in gf.yaml so appended rows can be labeled later",
            )
        with boundary():
            return owner._client().contexts.append(
                self.name,
                tabular.frame(rows),
                id=identity,
                idempotency_key=idempotency_key or "w-" + uuid.uuid4().hex,
            )

    def label(
        self,
        ids: Iterable[Any],
        values: Iterable[Any],
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        """Sets the target of rows in a live context, by id."""
        owner = self._live()
        ids, values = list(ids), list(values)
        if len(ids) != len(values):
            raise InvalidInput("ids and values must have the same length")
        identity = owner._id_name() or "id"
        target = owner.target
        if target is None and owner._kind in ("served", "hosted"):
            target = owner._hosted_facts().get("target")
        if target is None:
            raise InvalidInput("the predictor names no target to label")
        rows = [
            {identity: key, target: value}
            for key, value in zip(ids, values, strict=True)
        ]
        with boundary():
            return owner._client().contexts.label(
                self.name,
                rows,
                id=identity,
                column=target,
                idempotency_key=idempotency_key or "w-" + uuid.uuid4().hex,
            )

    def as_of(self, ts: Any) -> Predictor:
        """The same predictor reading the live context as of `ts`."""
        owner = self._live()
        return owner._with_as_of(ts)


@dataclass(frozen=True)
class Resolution:
    """The resolved schema of a predictor, as `gf check` prints it."""

    # A served or hosted predictor's view, from what its server describes;
    # local and ad hoc predictors explain with `predictors.Resolution`.
    name: str
    model: str
    task: str
    target: str
    target_type: str | None = None
    classes: tuple[Any, ...] = ()
    id: str | None = None
    time: str | None = None
    series: tuple[str, ...] = ()
    inputs: tuple[tuple[str, str], ...] = ()
    context: Context | None = None
    selection: Mapping[str, Any] | None = None
    inferred: Mapping[str, str] = field(default_factory=dict)


class Predictor:
    """A predictor by reference: a local directory, a hosted name, a catalog model or a URL.

    Args:
        ref: What to predict with. A path to a directory holding `gf.yaml`
            (anything with a path separator, starting with `.`, or naming an
            existing directory with `gf.yaml`) is local and runs in process.
            A catalog id such as `general-forecast/tabicl-v2-classifier` is
            an ad hoc predictor: give `target` and the context. A URL is a
            `gf serve` elsewhere. Anything else is a hosted predictor's name.
        env: The hosted environment, `production` by default.
        target: An ad hoc predictor's target column.
        inputs: An ad hoc predictor's input columns; inferred if None.
        exclude: Columns an ad hoc predictor leaves out of the inferred inputs.
        id: The id column.
        time: The time column.
        series: A forecaster's series columns.
        frequency: A forecaster's step.
        horizon: A forecaster's steps.
        quantiles: A regressor's or forecaster's levels.
        context: An ad hoc predictor's context rows, so `predict` can omit them.
        selection: How each query's context is drawn: core, neighbors, kind.
        future: A forecaster's inputs whose values each query brings ahead;
            its other inputs are past covariates.
        client: The hosted client to use; one from the profile if None.
        download: Whether a model's files may download if not cached.
        project: A served predictor's project; `local` if None.
        api_key: A served predictor's key. When None, a served predictor is
            called without one, unless GF_URL names its origin and
            GF_API_KEY holds that server's key.
    """

    def __init__(
        self,
        ref: str | os.PathLike[str],
        *,
        env: str | None = None,
        target: str | None = None,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        id: str | None = None,
        time: str | None = None,
        series: Sequence[str] | str | None = None,
        frequency: Any = None,
        horizon: int | None = None,
        quantiles: Sequence[float] | None = None,
        context: Any = None,
        selection: Mapping[str, Any] | None = None,
        future: Sequence[str] = (),
        client: Any = None,
        download: bool = True,
        project: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Resolves the reference; nothing runs until `predict` or `eval`."""
        self._kind, self._where = resolve_reference(ref, env=env)
        self._client_given = client
        self._download = download
        self._as_of: Any = None
        self._root: Path | None = None
        self._spec: Any = None
        self._resolution: Any = None
        self._lock: dict[str, Any] | None = None
        self._snapshot: pa.Table | None = None
        self._context_frame: pa.Table | None = None
        self._query_frame: pa.Table | None = None
        self._context_kind = "request"
        self._live_name: str | None = None
        self._project = project
        self._api_key = api_key
        self._transport: Any = None
        self.future = tuple(future)
        self.target = target
        self.inputs: tuple[str, ...] | None = (
            tuple(inputs) if inputs is not None else None
        )
        self.exclude = tuple(exclude)
        self.id = id
        self.time = time
        self.series: tuple[str, ...] = (
            (series,) if isinstance(series, str) else tuple(series or ())
        )
        self.frequency = frequency
        self.horizon = horizon
        self.quantiles = tuple(quantiles) if quantiles is not None else None
        self.selection = dict(selection) if selection else None
        self._env: str | None = None
        if self._kind == "local":
            if env is not None:
                raise InvalidInput(
                    "a local predictor has no environment",
                    hint="pass the hosted name without a path to use an environment",
                )
            self._root = Path(self._where)
            self.name = self._root.name
            self._load_local()
        elif self._kind == "catalog":
            if env is not None:
                raise InvalidInput("an ad hoc predictor has no environment")
            if not target:
                raise InvalidInput(
                    f"an ad hoc predictor on {self._where} needs a target",
                    param="target",
                    hint='pass target="<column>"',
                )
            self.model = self._where
            self.name = "ad-hoc"
            self.task = registry.task_of(self.model)
            if context is not None:
                self._context_frame = tabular.frame(context)
        elif self._kind == "served":
            if env is not None:
                raise InvalidInput("a served predictor's environment is its URL")
            self._base, self.name = self._where
            self._env = self._base
            self.model = "?"
            self.task = "?"
        else:
            if env is not None and env not in ENVIRONMENTS:
                raise InvalidInput(
                    f"environment `{env}` is not one of staging, production",
                    param="env",
                )
            self.name = self._where
            self._env = env or "production"
            self.model = "?"
            self.task = "?"
            self._facts: dict[str, Any] | None = None

    @classmethod
    def local(cls, path: str | os.PathLike[str]) -> Predictor:
        """The local predictor in directory `path`, never a hosted, catalog or served one.

        A bare name such as `churn` is the directory `churn` here, so
        commands given a directory never read it as a hosted name.

        Raises:
            NotFound: If `path` is no directory or holds no `gf.yaml`.
        """
        return cls(Path(os.fspath(path)))

    def _load_local(self) -> None:
        """Resolves `gf.yaml` as `gf check` does, without writing `gf.lock`."""
        from generalforecast.predictors import lock, resolve, verified
        from generalforecast.predictors.spec import load

        assert self._root is not None
        with boundary():
            spec = load(self._root)
            # A gf.lock beside the manifest is what gf serve and gf push run;
            # predicting from anything else would answer differently, so it
            # must be exactly the lock gf check writes now. Without one, the
            # manifest resolves as gf check would resolve it.
            if (self._root / lock.LOCK).exists():
                resolution = verified(spec, self._root)
            else:
                resolution = resolve(spec, self._root)
        self._spec, self._resolution = spec, resolution
        self._lock = lock.document(resolution)
        self.name = spec.name
        self.model = spec.model
        self.task = resolution.task
        self.target = spec.target
        self.inputs = resolution.input_names
        self.exclude = spec.exclude
        self.id, self.time, self.series = spec.id, spec.time, spec.series
        self.frequency, self.horizon = resolution.frequency, spec.horizon
        self.quantiles = resolution.quantiles or None
        self.future = spec.future
        self.selection = resolution.selection.to_dict()
        self._context_kind = spec.context.kind
        self._live_name = spec.context.live
        self._context_frame = resolution.context_table
        self._query_frame = resolution.query_table

    # facts

    @property
    def env(self) -> str | None:
        """None for local and ad hoc, `staging`/`production`, or the served URL."""
        return self._env

    @property
    def context(self) -> Context:
        """The predictor's context: its kind and rows, and a live context's handles."""
        if self._kind in ("local", "catalog"):
            if self._kind == "local" and self._context_kind == "live":
                return Context("live", None, self._live_name, self._as_of, self)
            if self._kind == "local" and self._context_kind == "request":
                return Context("request", None)
            rows = None if self._context_frame is None else len(self._context_frame)
            return Context("snapshot" if self._kind == "local" else "request", rows)
        facts = self._hosted_facts()
        info = facts.get("context") or {}
        return Context(
            str(info.get("kind", "live")),
            info.get("rows"),
            info.get("name", self.name),
            self._as_of,
            self,
        )

    def _id_name(self) -> str | None:
        """The id column, asked of a served or hosted predictor once."""
        if self.id is None and self._kind in ("served", "hosted"):
            self.id = self._hosted_facts().get("id") or None
        return self.id

    def _hosted_facts(self) -> dict[str, Any]:
        if self._kind == "served":
            return self._served_get(f"/predictors/{self.name}")
        if self._facts is None:
            with boundary():
                found = self._client().predictors.get(self.name, env=self._env)
            self._facts = dict(found) if isinstance(found, Mapping) else {}
            self.model = str(self._facts.get("model", self.model))
            self.task = str(self._facts.get("task", self.task))
        return self._facts

    def _client(self) -> Any:
        if self._client_given is None:
            from generalforecast.client import Client

            with boundary():
                self._client_given = Client()
        return self._client_given

    def __repr__(self) -> str:  # noqa: D105
        inputs = "?" if self.inputs is None else str(len(self.inputs))
        if (
            self._kind == "catalog"
            and self.inputs is None
            and self._context_frame is not None
        ):
            roles = {self.target, self.id, self.time, *self.series, *self.exclude}
            inputs = str(
                sum(1 for name in self._context_frame.column_names if name not in roles)
            )
        try:
            context = (
                repr(self.context)
                if self._kind in ("local", "catalog")
                else "Context(?)"
            )
        except Error:
            context = "Context(?)"
        return (
            f"Predictor({self.name}, model={self.model}, inputs={inputs}, "
            f"context={context[len('Context(') : -1]}, env={self._env!r})"
        )

    # prediction

    def request(
        self, queries: Any, context: Any = None, *, as_of: Any = None
    ) -> PredictRequest:
        """The validated request `predict` runs for these rows."""
        if self._kind == "local":
            return self._local_request(queries, context, as_of=as_of)
        if context is None:
            context = self._context_frame
        if context is None:
            raise InvalidInput(
                "context is required",
                param="context",
                hint="pass context= to predict, or give the predictor a context",
            )
        assert self.target is not None
        return PredictRequest.build(
            model=self.model,
            target=self.target,
            context=context,
            queries=queries,
            inputs=self.inputs,
            exclude=self.exclude,
            id=self.id,
            time=self.time,
            series=self.series,
            frequency=self.frequency,
            horizon=self.horizon,
            quantiles=self.quantiles,
            selection=self.selection,
            as_of=as_of if as_of is not None else self._as_of,
            future=self.future,
        )

    def _local_request(
        self, queries: Any, context: Any = None, *, as_of: Any = None
    ) -> PredictRequest:
        """A local predictor's request: its lock, and the context rows the lock names."""
        assert self._lock is not None and self.target is not None
        if self._context_kind == "request":
            if context is None:
                raise InvalidInput(
                    "context is required",
                    param="context",
                    hint=f"predictor {self.name} takes its context with each call",
                )
            rows = tabular.frame(context)
        else:
            if context is not None:
                raise InvalidInput(
                    f"predictor {self.name} has a {self._context_kind} context",
                    param="context",
                    hint="its context is the one gf.yaml declares; pass queries only",
                )
            rows = self._context_frame
        if rows is None:
            raise InvalidInput(
                "a live context is read where it lives",
                hint="push the predictor and predict with the hosted name",
            )
        resolution = self._resolution
        return PredictRequest(
            model=self.model,
            task=self.task,
            target=self.target,
            context=rows,
            queries=tabular.frame(queries),
            inputs=resolution.input_names,
            id=self.id,
            time=self.time,
            series=self.series,
            frequency=self.frequency,
            horizon=self.horizon,
            quantiles=self.quantiles,
            selection=self.selection,
            as_of=as_of if as_of is not None else self._as_of,
            classes=tuple(resolution.classes or ()),
            future=self.future,
            lock=self._lock,
        )

    def predict(self, queries: Any, context: Any = None, *, as_of: Any = None) -> Any:
        """Scores rows.

        Any number of query rows score: past one call's bound
        (`contracts.MAX_QUERIES`, or the model's own) they go as several
        calls, a forecaster's series each whole within one, and come back
        in the order given.

        Args:
            queries: Rows to score: a frame, list of dicts, Arrow table, or a
                path or URL to a .csv, .parquet or .json file.
            context: Rows the model learns from, for an ad hoc predictor or a
                predictor whose context kind is `request`.
            as_of: For a live context, the time to read it as of.

        Returns:
            A pandas DataFrame: `id` (if any), `label` and `p_<class>`
            columns for a classifier, or `value` and `q_<level>` columns for
            a regressor or forecaster.
        """
        table = tabular.frame(queries)
        cap = self._query_cap()
        if table.num_rows <= cap and self._kind in ("local", "catalog"):
            return self._predict_once(table, context, as_of=as_of)
        import pandas as pd

        series = self._series_columns()
        parts, order = [], []
        pending = list(reversed(_query_batches(table, series, cap)))
        while pending:
            rows = pending.pop()
            try:
                parts.append(self._predict_once(table.take(rows), context, as_of=as_of))
            except Error as error:
                # A reply past the SDK's response bound comes back as a
                # protocol error; half as many rows answer in half the bytes.
                halves = _query_batches(
                    table.take(rows), series, max(1, len(rows) // 2)
                )
                if error.code != "ProtocolError" or len(halves) < 2:
                    raise
                pending.extend(reversed([rows[half] for half in halves]))
                continue
            order.append(rows)
        frame = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
        placed = np.empty(table.num_rows, dtype=np.int64)
        placed[np.concatenate(order)] = np.arange(table.num_rows)
        return frame.iloc[placed].reset_index(drop=True)

    def _query_cap(self) -> int:
        """Most query rows one call carries: the wire bound, or a model's lower one."""
        from generalforecast.contracts import MAX_QUERIES

        if self._kind == "catalog" and self.task != "forecast":
            # An ad hoc call is held to the model's own per-call bound.
            return max(1, min(MAX_QUERIES, registry.limits(self.model).queries))
        return MAX_QUERIES

    def _series_columns(self) -> tuple[str, ...]:
        """A forecaster's series columns, which a batch keeps whole; none otherwise."""
        if self._kind in ("served", "hosted") and not self.series:
            facts = self._hosted_facts()
            if facts.get("task") == "forecast":
                self.series = tuple(facts.get("series") or ())
        return tuple(self.series)

    def _predict_once(
        self, queries: Any, context: Any = None, *, as_of: Any = None
    ) -> Any:
        """One prediction call: at most `MAX_QUERIES` query rows."""
        if self._kind in ("local", "catalog") and not (
            self._kind == "local" and self._context_kind == "live"
        ):
            if (as_of if as_of is not None else self._as_of) is not None:
                # As gf serve and hosted serving refuse it: only a live
                # context changes over time.
                kind = self._context_kind if self._kind == "local" else "request"
                raise InvalidInput(
                    f"as_of reads a live context; this predictor's is a {kind} context",
                    param="as_of",
                )
            request = self.request(queries, context, as_of=as_of)
            with boundary():
                return run_local(request, download=self._download)
        if self._kind == "served":
            payload = PredictRequest.declared_payload(
                queries,
                context=context,
                as_of=as_of if as_of is not None else self._as_of,
            )
            value = self._served_post(f"/predictors/{self.name}/predict", payload)
            return self._keyed(_frame_of_result(value, self._id_name()), queries)
        with boundary():
            result = self._client().predictors.predict(
                self.name,
                queries,
                context=context,
                as_of=as_of if as_of is not None else self._as_of,
                env=self._env,
            )
        return self._keyed(_frame_of_result(result, self._id_name()), queries)

    def _keyed(self, frame: Any, queries: Any) -> Any:
        """A served or hosted forecast's frame with the series and time of its queries, as local frames have them.

        The wire reply carries each prediction in query order, so the keys
        come from the queries themselves.
        """
        facts = self._hosted_facts()
        if facts.get("task") != "forecast" or not facts.get("time"):
            return frame
        series = tuple(facts.get("series") or ())
        return _with_forecast_keys(
            frame, tabular.frame(queries), series, facts["time"], self._id_name()
        )

    def _with_as_of(self, ts: Any) -> Predictor:
        clone = Predictor.__new__(Predictor)
        clone.__dict__.update(self.__dict__)
        clone._as_of = ts
        return clone

    # served HTTP

    def _served_transport(self) -> Any:
        """The SDK transport to a `gf serve`, with its key, made once.

        The connection resolves as every SDK connection does
        (`ConnectionSettings.resolve`): HTTPS except on loopback, and a key
        only when `api_key` gives one, or when GF_URL names this very
        origin and GF_API_KEY is its key. An ambient hosted key never goes
        to another server.
        """
        if self._transport is None:
            from generalforecast.transport import (
                ConnectionSettings,
                Transport,
                endpoint,
                origin_of,
            )

            with boundary():
                base = endpoint(self._base)
                ambient = os.environ.get("GF_URL")
                owned = ambient is not None and origin_of(ambient) == origin_of(base)
                settings = ConnectionSettings.resolve(
                    base_url=base,
                    api_key=self._api_key,
                    project=self._project,
                    key_env="GF_API_KEY" if owned and self._api_key is None else None,
                )
                self._transport = Transport(base, token=settings.token, timeout=120)
        return self._transport

    def _served_post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._served("POST", path, payload)

    def _served_get(self, path: str) -> dict[str, Any]:
        return self._served("GET", path, None)

    def _served(
        self, method: str, path: str, payload: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """One call to a `gf serve`, enveloped and unwrapped as the SDK client does."""
        from generalforecast.service.boundary import Request, Resource, Scope
        from generalforecast.service.sdk import unwrap

        project = self._project or "local"
        request = Request(
            Scope(project, "r-" + uuid.uuid4().hex),
            Resource("predictor", self.name),
            dict(payload or {}),
        )
        with boundary():
            value = self._served_transport().send(
                method,
                f"/v1/projects/{project}{path}",
                request.to_dict() if method == "POST" else None,
                {
                    "X-GF-Request-ID": request.scope.request_id,
                    "X-GF-Project": project,
                },
            )
            return dict(unwrap(value, request))

    # evaluation

    def eval(self, queries: Any = None, *, models: Sequence[str] = ()) -> Evaluation:
        """Scores held-out rows whose target is known.

        Args:
            queries: Rows with the target, to score instead of the
                predictor's own query rows (a snapshot's empty-target rows,
                rows after `until`, or the holdout draw).
            models: Other catalog models to score on the same split.

        Returns:
            The metrics per model and the scored rows.
        """
        if self._kind not in ("local", "catalog"):
            raise InvalidInput(
                "eval runs in process; use a local or ad hoc predictor",
                hint="gf.Predictor('./dir').eval()",
            )
        context = self._context_frame
        if context is None:
            raise InvalidInput(
                "eval needs a context", hint="give the predictor a context"
            )
        if queries is not None:
            held = tabular.frame(queries)
        elif self._query_frame is not None and len(self._query_frame):
            held = self._query_frame
        elif self._kind == "local":
            raise InvalidInput(
                f"predictor {self.name} has no query rows to score",
                param="queries",
                because="every row of the context has a target, and gf.yaml "
                "sets no `until` or `holdout`",
                hint="pass rows with known targets: `gf eval --queries FILE` or "
                "`.eval(queries)`; or set `holdout` or `until` under `context`",
            )
        else:
            from generalforecast.predictors.tables import split_by

            assert self.target is not None
            context, held = split_by(
                context,
                self.target,
                holdout={"fraction": _HOLDOUT_FRACTION, "seed": 0},
            )
        assert self.target is not None
        if self.target not in held.column_names:
            raise InvalidInput(
                f"query rows have no `{self.target}` column to score against",
                param="target",
            )
        truth_column = held.column(self.target)
        if _empty(truth_column).to_numpy(zero_copy_only=False).any():
            raise InvalidInput(
                f"query rows have empty `{self.target}` values; eval needs the truth",
                param="target",
            )
        truth = truth_column.to_pylist()
        rows: dict[str, Any] = {}
        if self.id is not None and self.id in held.column_names:
            rows["id"] = held.column(self.id).to_pylist()
        rows[self.target] = truth
        metrics: dict[str, dict[str, float]] = {}
        import pandas as pd

        for ref in (self.model, *models):
            model = registry.canonical(ref)
            task = registry.task_of(model)
            from generalforecast.contracts import MAX_QUERIES

            cap = (
                MAX_QUERIES
                if task == "forecast"
                else max(1, min(MAX_QUERIES, registry.limits(model).queries))
            )
            frames, order = [], []
            lock = self._lock if model == self.model else self._lock_for(model)
            series = self.series if task == "forecast" else ()
            for indices in _query_batches(held, series, cap):
                given = None if self._kind == "local" else context
                batch = dataclasses.replace(
                    self.request(held.take(indices), given),
                    model=model,
                    task=task,
                    lock=lock,
                )
                if not frames:
                    request = batch
                with boundary():
                    frames.append(run_local(batch, download=self._download))
                order.append(indices)
            frame = (
                frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
            )
            placed = np.empty(len(frame), dtype=np.int64)
            placed[np.concatenate(order)] = np.arange(len(frame))
            frame = frame.iloc[placed].reset_index(drop=True)
            metrics[model] = self._score(request, frame, truth, held)
            key = (
                registry.short(model)
                if (len(models) or model != self.model)
                else "prediction"
            )
            rows[key] = frame[
                "label" if self.task == "classification" else "value"
            ].tolist()
        return Evaluation(metrics, pd.DataFrame(rows))

    def _lock_for(self, model: str) -> dict[str, Any] | None:
        """A local predictor's lock with another model, resolved as `gf check` would."""
        if self._kind != "local" or self._spec is None:
            return None
        from generalforecast.predictors import lock, resolve

        with boundary():
            spec = dataclasses.replace(self._spec, model=model)
            return lock.document(resolve(spec, self._root))

    def _score(
        self, request: PredictRequest, frame: Any, truth: list[Any], held: pa.Table
    ) -> dict[str, float]:
        if request.task == "classification":
            probabilities = {
                label: frame[f"p_{label}"].tolist()
                for label in request.classes
                if f"p_{label}" in frame.columns
            }
            return classification_metrics(truth, frame["label"].tolist(), probabilities)
        quantiles = {
            float(column[2:]): frame[column].tolist()
            for column in frame.columns
            if column.startswith("q_")
        }
        if request.task == "regression":
            return regression_metrics(truth, frame["value"].tolist(), quantiles or None)

        # Each series is keyed by every series column, and its history is
        # in time order, so the naive error scaling MASE is each series' own.
        def keys(table: pa.Table) -> list[tuple]:
            columns = [table.column(name).to_pylist() for name in request.series]
            return list(zip(*columns, strict=True)) if columns else [()] * len(table)

        context = request.context
        times = (
            context.column(request.time).to_pylist()
            if request.time
            else list(range(len(context)))
        )
        ordered: dict[tuple, list[tuple[Any, float]]] = {}
        for key, moment, value in zip(
            keys(context),
            times,
            context.column(request.target).to_pylist(),
            strict=True,
        ):
            if value is not None:
                ordered.setdefault(key, []).append((_when(moment), float(value)))
        history = {
            key: [value for _, value in sorted(values, key=lambda item: item[0])]
            for key, values in ordered.items()
        }
        held_keys = keys(held)
        if not quantiles:
            quantiles = {0.5: frame["value"].tolist()}
        return forecast_metrics(truth, quantiles, history=history, series=held_keys)

    # lifecycle

    def explain(self) -> Any:
        """The resolved schema: model, task, target, inputs, context and selection.

        A local predictor's is the `predictors.Resolution` `gf check` prints;
        an ad hoc one's is the same, resolved against its context rows; a
        served or hosted one's is what its server describes.
        """
        if self._kind == "local":
            return self._resolution
        if self._kind == "catalog":
            from generalforecast.predictors import resolve
            from generalforecast.predictors.spec import PredictorSpec

            context = self._context_frame
            if context is None:
                raise InvalidInput(
                    "an ad hoc predictor explains against its context",
                    hint="give the predictor a context",
                )
            request = self.request(context.slice(0, 1), context)
            with boundary():
                return resolve(
                    PredictorSpec.from_dict(request.manifest()), rows=context
                )
        facts = self._hosted_facts()
        return Resolution(
            name=self.name,
            model=str(facts.get("model", self.model)),
            task=str(facts.get("task", self.task)),
            target=str(facts.get("target", self.target or "?")),
            id=facts.get("id"),
            time=facts.get("time"),
            series=tuple(facts.get("series") or ()),
            inputs=tuple(
                (item["name"], item.get("type", "?"))
                if isinstance(item, Mapping)
                else (str(item), "?")
                for item in facts.get("inputs") or ()
            ),
            context=self.context,
            selection=facts.get("selection"),
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Writes `gf.yaml`, its snapshot and `gf.lock` for a local or ad hoc predictor at `path`.

        A local predictor's manifest is kept whole, its split, types, past
        and future covariates, selection and resources included, with the
        snapshot copied beside it; an ad hoc predictor's settings become a
        manifest with its context rows as the snapshot. The directory checks
        as the original does: `gf.Predictor(path)` predicts the same.
        """
        if self._kind not in ("local", "catalog"):
            raise InvalidInput(
                "only a local or ad hoc predictor can be saved",
                hint="a hosted predictor's manifest is its revision",
            )
        from pyarrow import parquet

        from generalforecast.predictors import resolve, write_lock, write_manifest
        from generalforecast.predictors.spec import Context as SpecContext
        from generalforecast.predictors.spec import PredictorSpec

        root = Path(path)
        name = root.resolve().name if self._kind == "catalog" else self.name
        if not NAME.fullmatch(name):
            raise InvalidInput(
                f"`{name}` is not a predictor name",
                hint="use lowercase letters, digits and dashes, starting with a letter",
            )
        snapshot = "context.parquet"
        with boundary():
            if self._kind == "local":
                assert self._spec is not None and self._root is not None
                spec = dataclasses.replace(self._spec, name=name, marks={})
                rows = None
                if spec.context.kind == "snapshot":
                    from generalforecast.predictors.tables import read_frame

                    assert spec.context.snapshot is not None
                    rows = read_frame(self._root, spec.context.snapshot)
                    spec = dataclasses.replace(
                        spec,
                        context=dataclasses.replace(spec.context, snapshot=snapshot),
                    )
            else:
                rows = self._context_frame
                if rows is None:
                    raise InvalidInput(
                        "an ad hoc predictor saves with its context rows",
                        hint="give the predictor a context",
                    )
                request = self.request(rows.slice(0, 1), rows)
                manifest = request.manifest()
                manifest["name"] = name
                manifest.pop("context")
                spec = dataclasses.replace(
                    PredictorSpec.from_dict(manifest),
                    context=SpecContext("snapshot", snapshot=snapshot),
                )
            root.mkdir(parents=True, exist_ok=True)
            if rows is not None:
                parquet.write_table(rows, root / snapshot)
            write_manifest(spec, root, force=True)
            write_lock(resolve(spec, root), root)
        return root

    def push(self, env: str = "staging") -> Predictor:
        """Deploys a local predictor to a hosted environment and returns it.

        Args:
            env: The environment, `staging` or `production`.
        """
        if env not in ENVIRONMENTS:
            raise InvalidInput(
                f"environment `{env}` is not one of staging, production", param="env"
            )
        if self._kind != "local":
            raise InvalidInput(
                "only a local predictor can be pushed",
                hint="save it first with .save(path), then push that directory",
            )
        assert self._root is not None
        client = self._client()
        with boundary():
            client.predictors.push(self._root, env=env)
        return Predictor(self.name, env=env, client=client)


__all__ = [
    "Context",
    "Evaluation",
    "PredictRequest",
    "Predictor",
    "Resolution",
    "classification_metrics",
    "forecast_metrics",
    "regression_metrics",
    "resolve_reference",
    "run_local",
]
