"""sklearn adapters over the existing optimized native tabular and series engine."""

import datetime
import uuid
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from generalforecast.columns import infer
from generalforecast.local_models import native_model
from generalforecast.native.estimators import Classifier, Regressor, _drop, _local
from generalforecast.native.package import (
    admitted_spec,
    copy_table,
    export_package,
    remote_result,
)
from generalforecast.native.package import (
    forecast as native_forecast,
)
from generalforecast.packages import Package, PackageSpec
from generalforecast.packages.models import describe
from generalforecast.packages.schema import infer_schema
from generalforecast.packages.validation import ensure
from generalforecast.tabular import physical_type, to_arrow

__all__ = ["Classifier", "Forecaster", "Regressor"]


class Forecaster(RegressorMixin, BaseEstimator):
    """Prepares series context, not foundation weights, for explicit future rows.

    fit accepts a frame containing time, series keys, declared covariates and
    target (or a separate y vector). predict accepts exactly horizon future
    rows per series. Series in one fitted context form one explicit group, with
    shared named covariates and no implicit context truncation.
    """

    def __init__(
        self,
        model: Any = "chronos-2",
        *,
        target: str = "target",
        time: str = "time",
        series: str | tuple[str, ...] = (),
        horizon: int = 1,
        frequency: str | float | None = None,
        past_inputs: tuple[str, ...] = (),
        future_inputs: tuple[str, ...] = (),
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        backend: Any = None,
        allow_downloads: bool = False,
    ) -> None:
        """Stores cloneable settings; local checkpoints are cache-only by default."""
        self.model = model
        self.target = target
        self.time = time
        self.series = series
        self.horizon = horizon
        self.frequency = frequency
        self.past_inputs = past_inputs
        self.future_inputs = future_inputs
        self.quantiles = quantiles
        self.backend = backend
        self.allow_downloads = allow_downloads

    def __sklearn_is_fitted__(self) -> bool:
        """Reports only successfully prepared context as fitted."""
        return hasattr(self, "context_")

    def set_params(self, **params: Any) -> "Forecaster":
        """Invalidates context when any constructor parameter changes."""
        super().set_params(**params)
        if params:
            self.close()
        return self

    def _require_fitted(self) -> None:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("call fit before predicting or exporting a package")

    def _intent(self) -> dict:
        series = [self.series] if isinstance(self.series, str) else list(self.series)
        return {
            "time": self.time,
            "series": series,
            "horizon": self.horizon,
            "past_inputs": list(self.past_inputs),
            "future_inputs": list(self.future_inputs),
            "quantiles": list(self.quantiles),
            "outputs": ["value", "quantiles"],
            **({"frequency": self.frequency} if self.frequency is not None else {}),
        }

    def fit(self, X: Any, y: Any = None) -> "Forecaster":
        """Copies bounded context and validates regular cadence for every series."""
        self.close()
        table = to_arrow(X)
        features = list(table.column_names)
        if y is not None:
            ensure(
                self.target not in table.column_names, "target supplied in both X and y"
            )
            values = np.asarray(y)
            ensure(
                values.ndim == 1 and len(values) == len(table),
                "expected one target per context row",
            )
            table = to_arrow(table.append_column(self.target, pa.array(values)))
        else:
            features = [column for column in features if column != self.target]
        pin = describe(self.model if isinstance(self.model, str) else "chronos-2")
        ensure("forecast" in pin.tasks, "model does not support forecasting")
        spec = PackageSpec.from_dict(
            {
                "name": "estimator",
                "model": pin.resolved,
                "task": "forecast",
                "target": self.target,
                "context": "context.arrow",
                **self._intent(),
            }
        )
        if not isinstance(self.model, str):
            ensure(
                self.backend is None,
                "remote backend requires an admitted catalog model",
            )
            functions, parameters = self.model
            properties = functions.properties(parameters)
            pin = replace(
                pin,
                max_context_rows=properties.context_limit,
                max_horizon=properties.pass_limit or properties.context_limit,
                quantiles=properties.quantiles,
            )
        schema = infer_schema(spec, pin, "forecast", table)
        identities = tuple(
            dict.fromkeys(
                tuple(row[name] for name in spec.series) for row in table.to_pylist()
            )
        )
        ensure(
            1 <= len(identities) <= 8,
            "forecast supports at most eight related series",
            "CapacityExceeded",
        )
        origins = {}
        covariates = {}
        for row in table.to_pylist():
            identity = tuple(row[name] for name in spec.series)
            instant = row[self.time]
            origins[identity] = max(instant, origins.get(identity, instant))
            for column in spec.forecast_inputs:
                key = (column, instant)
                ensure(
                    key not in covariates or covariates[key] == row[column],
                    "grouped targets must share named historical covariates",
                    "UnsupportedCovariates",
                )
                covariates[key] = row[column]
        ensure(
            len(set(origins.values())) == 1,
            "related series must share the same forecast origin",
        )
        ensure(
            set(spec.quantiles) <= set(pin.quantiles) and 0.5 in spec.quantiles,
            "forecast quantiles must include median and be published model levels",
            "UnsupportedCapability",
        )
        ensure(
            table.num_rows <= pin.max_context_rows,
            "context capacity exceeded",
            "CapacityExceeded",
        )
        clock = physical_type(table[self.time].type)
        ensure(
            clock in ("int64", "date", "timestamp"),
            "local forecast requires integer periods or datetime time",
            "UnsupportedCapability",
        )
        for column in spec.forecast_inputs:
            ensure(
                physical_type(table[column].type) in ("int64", "float64"),
                "forecast covariates must be numeric",
                "UnsupportedCapability",
            )
        self._spec = spec
        self._identities = identities
        assert schema.frequency is not None
        self._frequency = schema.frequency
        self._clock = clock
        self.n_features_in_ = len(features)
        self.feature_names_in_ = np.asarray(features, dtype=object)
        self.context_ = copy_table(table)
        try:
            if self.backend is None:
                if isinstance(self.model, str):
                    native_model(
                        self.model, "forecast", allow_downloads=self.allow_downloads
                    )
                else:
                    self._table()
        except BaseException:
            self.close()
            raise
        return self

    def _identity(self, row: dict) -> str:
        identity = tuple(row[name] for name in self._spec.series)
        ensure(
            identity in self._identities, "future series is absent from fitted context"
        )
        return f"gfseries:{self._identities.index(identity)}"

    def _engine_rows(self, table: pa.Table) -> pa.Table:
        rows = table.to_pylist()
        result = {
            "series": [self._identity(row) for row in rows],
            "time": table[self.time],
            "target": table[self.target],
        }
        for index, column in enumerate(self._spec.forecast_inputs):
            result[f"c{index}"] = table[column]
        return pa.table(result)

    def _table(self) -> Any:
        if hasattr(self, "table_"):
            return self.table_
        table, dropper = self._make_table(self.context_)
        self.table_, self._dropper = table, dropper
        return table

    def _make_table(self, context: pa.Table) -> tuple[Any, Any]:
        from generalforecast._core import GroupStrategy

        client = _local()
        name = "estimator-" + uuid.uuid4().hex
        rows = self._engine_rows(context)
        covariates = {
            f"c{i}": infer(self.context_.select([column]))[column]
            for i, column in enumerate(self._spec.forecast_inputs)
        }
        known = [
            f"c{i}"
            for i, column in enumerate(self._spec.forecast_inputs)
            if column in self.future_inputs
        ]
        step = (
            int(self._frequency)
            if self._clock == "int64"
            else datetime.timedelta(seconds=self._frequency)
        )
        table = client.create_series_table(
            name,
            model=native_model(
                self.model, "forecast", allow_downloads=self.allow_downloads
            ),
            series="series",
            time="time",
            target="target",
            step=step,
            columns=covariates or None,
            known=known,
            group=GroupStrategy.given(len(self._identities)),
            horizon=self.horizon,
            times="integer" if self._clock == "int64" else "datetime",
        )
        dropper = weakref.finalize(self, _drop, client, name)
        dropper.atexit = False
        try:
            table.insert(rows)
        except BaseException:
            dropper()
            raise
        return table, dropper

    def _future(self, X: Any) -> tuple[pa.Table, dict[str, list[int]]]:
        self._require_fitted()
        table = to_arrow(X)
        columns = [*self._spec.series, self.time, *self.future_inputs]
        ensure(
            set(table.column_names) == set(columns),
            "future rows require only series, time and known-future inputs",
        )
        table = table.select(columns)
        for column in columns:
            ensure(
                table[column].type == self.context_[column].type
                and not table[column].null_count,
                "future column types and non-missing values must match context",
            )
        history: dict[str, list] = {}
        for row in self.context_.to_pylist():
            history.setdefault(self._identity(row), []).append(row[self.time])
        groups: dict[str, list[int]] = {}
        rows = table.to_pylist()
        known = {}
        for index, row in enumerate(rows):
            key = self._identity(row)
            ensure(key in history, "future series is absent from fitted context")
            groups.setdefault(key, []).append(index)
            for column in self.future_inputs:
                item = (column, row[self.time])
                ensure(
                    item not in known or known[item] == row[column],
                    "grouped targets must share named future covariates",
                    "UnsupportedCovariates",
                )
                known[item] = row[column]
        step = (
            self._frequency
            if self._clock == "int64"
            else datetime.timedelta(seconds=self._frequency)
        )
        for key, indexes in groups.items():
            ensure(
                len(indexes) == self.horizon,
                "future rows must cover exactly the configured horizon",
            )
            indexes.sort(key=lambda index: rows[index][self.time])
            origin = max(history[key])
            ensure(
                [rows[index][self.time] for index in indexes]
                == [origin + step * (i + 1) for i in range(self.horizon)],
                "future times must immediately follow fitted context on its regular grid",
            )
        return table, groups

    def predict_quantiles(self, X: Any) -> np.ndarray:
        """Returns one row per supplied future row, in its original order."""
        queries, groups = self._future(X)
        if self.backend is not None:
            result = remote_result(
                self.backend,
                spec=self._spec,
                context=self.context_,
                queries=queries,
                features=[],
                options=self._intent(),
            )
            ensure(
                "quantiles" in result.table.column_names,
                "backend did not return forecast quantiles",
                "ProtocolError",
            )
            values = result.table["quantiles"].to_pylist()
            keys = [str(float(level)) for level in self.quantiles]
            ensure(
                all(isinstance(row, dict) and set(row) == set(keys) for row in values),
                "backend returned different forecast quantiles",
                "ProtocolError",
            )
            values = np.asarray(
                [[row[key] for key in keys] for row in values], dtype=float
            )
            ensure(
                np.isfinite(values).all(),
                "backend returned non-finite forecasts",
                "ProtocolError",
            )
            return values
        if isinstance(self.model, str):
            return native_forecast(
                self._spec,
                self.context_,
                queries,
                self._frequency,
                allow_downloads=self.allow_downloads,
            )
        table = self._table()
        temporary = None
        try:
            if self.future_inputs:
                future = pa.table(
                    {
                        **{column: queries[column] for column in queries.column_names},
                        self.target: pa.array(
                            [None] * len(queries), type=self.context_[self.target].type
                        ),
                        **{
                            column: pa.array(
                                [None] * len(queries), type=self.context_[column].type
                            )
                            for column in self.past_inputs
                            if column not in self.future_inputs
                        },
                    }
                )
                future = future.select(self.context_.column_names).cast(
                    self.context_.schema
                )
                table, temporary = self._make_table(
                    pa.concat_tables([self.context_, future])
                )
            forecasts = table.forecast_given(
                list(groups),
                [[f"gfseries:{i}" for i in range(len(self._identities))]] * len(groups),
                horizon=self.horizon,
                quantiles=self.quantiles,
                before=[
                    queries[self.time][indexes[0]].as_py()
                    for indexes in groups.values()
                ],
            )
        finally:
            if temporary is not None:
                temporary()
        values = np.empty((len(queries), len(self.quantiles)))
        for forecast, indexes in zip(forecasts, groups.values(), strict=True):
            values[indexes] = forecast
        ensure(
            np.isfinite(values).all(),
            "native forecast returned non-finite values",
            "ModelOutputError",
        )
        return values

    def predict(self, X: Any) -> np.ndarray:
        """Returns the forecast median on the original target scale."""
        return self.predict_quantiles(X)[:, list(self.quantiles).index(0.5)]

    def to_package(self, path: str | Path, *, name: str) -> Package:
        """Exports the exact context, cadence, horizon and explicit covariate roles."""
        self._require_fitted()
        spec = admitted_spec(
            self.model, "forecast", self.context_, target=self.target, **self._intent()
        )
        return export_package(path, name=name, spec=spec, context=self.context_)

    def close(self) -> None:
        """Releases native context; the estimator must be fitted again."""
        dropper = getattr(self, "_dropper", None)
        if dropper is not None:
            dropper()
        for name in (
            "table_",
            "_dropper",
            "context_",
            "n_features_in_",
            "feature_names_in_",
            "_spec",
            "_clock",
            "_frequency",
            "_identities",
        ):
            self.__dict__.pop(name, None)

    def __getstate__(self) -> dict[str, Any]:
        """Keeps copied context but recreates process-local native tables on demand."""
        state = dict(super().__getstate__())
        state.pop("table_", None)
        state.pop("_dropper", None)
        return state
