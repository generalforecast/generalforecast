# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sklearn adapters over the existing optimized native tabular and series engine."""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from generalforecast.errors import InvalidInput
from generalforecast.native import models as registry
from generalforecast.native.estimators import Classifier, Regressor
from generalforecast.native.predictor import (
    admitted_spec,
    copy_table,
    export_predictor,
    remote_result,
)
from generalforecast.native.predictor import (
    forecast as native_forecast,
)
from generalforecast.native.sklearn_shim import (
    BaseEstimator,
    NotFittedError,
    RegressorMixin,
    public,
)
from generalforecast.predictors import frequency as _frequency
from generalforecast.predictors.models import describe, for_task
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.runtime.schema import infer_schema, observed_frequency
from generalforecast.predictors.validation import ensure
from generalforecast.tabular import physical_type, to_arrow

if TYPE_CHECKING:
    from generalforecast.predictor import Predictor

__all__ = ["Classifier", "Forecaster", "Regressor"]


def _forecaster_model(model: Any, download: bool) -> Any:
    """A forecaster's model as the constructor and `set_params` take it.

    A string becomes its canonical catalog id; a custom model is refused.
    """
    if not isinstance(model, str):
        return model
    model = registry.canonical(model)
    if registry.is_custom(model):
        raise InvalidInput(
            f"Forecaster runs catalog forecasters, not custom model `{model}`",
            hint="run a custom forecaster through gf.engine.SeriesTable",
            param="model",
        )
    from generalforecast.native.estimators import note_download

    note_download(model, download)
    return model


class Forecaster(RegressorMixin, BaseEstimator):
    """Prepares series context, not foundation weights, for explicit future rows.

    fit accepts a frame containing time, series keys, declared covariates and
    target (or a separate y vector). predict accepts exactly horizon future
    rows per series. Series in one fitted context form one explicit group, with
    shared named covariates and no implicit context truncation.
    """

    def __init__(
        self,
        model: Any = "general-forecast/chronos-2",
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
        download: bool = True,
    ) -> None:
        """Stores cloneable settings; local checkpoints are cache-only by default."""
        self.model = _forecaster_model(model, download)
        self.target = target
        self.time = time
        self.series = series
        self.horizon = horizon
        self.frequency = frequency
        self.past_inputs = past_inputs
        self.future_inputs = future_inputs
        self.quantiles = quantiles
        self.backend = backend
        self.download = download

    def __sklearn_is_fitted__(self) -> bool:
        """Reports only successfully prepared context as fitted."""
        return hasattr(self, "context_")

    def set_params(self, **params: Any) -> "Forecaster":
        """Checks a new model as the constructor does; invalidates context on any change."""
        if "model" in params:
            params["model"] = _forecaster_model(
                params["model"], params.get("download", self.download)
            )
        super().set_params(**params)
        if params:
            self.close()
        return self

    def _require_fitted(self) -> None:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("call fit before predicting or saving a predictor")

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
            **(
                {"frequency": self._spelled_frequency()}
                if self.frequency is not None
                else {}
            ),
        }

    @public
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
        pin = describe(self.model)
        ensure("forecast" in pin.tasks, "model does not support forecasting")
        spec = ExecutionPlan.build(
            "estimator",
            pin.resolved,
            task="forecast",
            target=self.target,
            **self._intent(),
        )
        if spec.frequency is None:
            spec = replace(spec, frequency=observed_frequency(spec, table))
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
        assert schema.frequency is not None and spec.frequency is not None
        self._frequency = schema.frequency
        self._step = spec.frequency
        self._clock = clock
        self.n_features_in_ = len(features)
        self.feature_names_in_ = np.asarray(features, dtype=object)
        self.context_ = copy_table(table)
        try:
            if self.backend is None:
                for_task(self.model, "forecast")
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

    def _future(self, X: Any) -> pa.Table:
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
        for key, indexes in groups.items():
            ensure(
                len(indexes) == self.horizon,
                "future rows must cover exactly the configured horizon",
            )
            indexes.sort(key=lambda index: rows[index][self.time])
            origin = max(history[key])
            ensure(
                [rows[index][self.time] for index in indexes]
                == self._steps_after(origin),
                "future times must immediately follow fitted context on its regular grid",
            )
        return table

    def _spelled_frequency(self) -> str | int | float:
        try:
            return _frequency.normalize(self.frequency)
        except (TypeError, ValueError):
            raise InvalidInput(
                f"frequency `{self.frequency}` is not a frequency",
                param="frequency",
                hint="use a span such as `1d` or `15min`, a pandas offset alias "
                "such as `W-MON` or `MS`, or seconds as a number",
            ) from None

    def _steps_after(self, origin: Any) -> list[Any]:
        """The `horizon` times after `origin` on the fitted grid."""
        if self._clock == "int64":
            return [
                origin + int(self._frequency) * (i + 1) for i in range(self.horizon)
            ]
        return _frequency.grid(origin, self.horizon, self._step)

    @public
    def predict_quantiles(self, X: Any) -> np.ndarray:
        """Returns one row per supplied future row, in its original order."""
        queries = self._future(X)
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
        values = native_forecast(
            self._spec,
            self.context_,
            queries,
            self._step,
            allow_downloads=self.download,
        )
        ensure(
            np.isfinite(values).all(),
            "native forecast returned non-finite values",
            "ModelOutputError",
        )
        return values

    @public
    def predict(self, X: Any) -> np.ndarray:
        """Returns the forecast median on the original target scale."""
        return self.predict_quantiles(X)[:, list(self.quantiles).index(0.5)]

    @public
    def forecast(self, horizon: int | None = None, *, future: Any = None) -> np.ndarray:
        """Forecasts every fitted series past its last observation.

        Args:
            horizon: Steps to forecast; the fitted horizon if None. The
                model's horizon is fixed at fit, so another value refits
                the intent on the same context.
            future: The future rows, one per series and step, with the
                series, time and every `future_inputs` column; `future_grid()`
                lists the times. Required when the forecaster has
                `future_inputs`, and the only way to score with them.

        Returns:
            Quantiles `(series * horizon, levels)`, one row per series and
            step, series in the order they were first seen (the order of
            `future` when given).
        """
        self._require_fitted()
        if horizon is not None and horizon != self.horizon:
            context = self.context_
            self.set_params(horizon=horizon)
            self.fit(context)
        if future is None:
            if self.future_inputs:
                raise InvalidInput(
                    "this forecaster has known-future inputs; pass forecast(future=...)",
                    param="future",
                    hint="future rows need "
                    + ", ".join(self.future_inputs)
                    + f" at each of the {self.horizon} times future_grid() lists",
                )
            future = self.future_grid()
        return self.predict_quantiles(future)

    @public
    def future_grid(self) -> Any:
        """The time grid `forecast` scores: `horizon` steps after each series.

        Returns:
            A pandas DataFrame with the series and time columns, one row per
            series and step, series in the order they were first seen.
        """
        self._require_fitted()
        latest: dict[str, dict] = {}
        for row in self.context_.to_pylist():
            key = self._identity(row)
            if key not in latest or row[self.time] > latest[key][self.time]:
                latest[key] = row
        columns = [*self._spec.series, self.time]
        rows = [
            {**{name: last[name] for name in self._spec.series}, self.time: when}
            for last in latest.values()
            for when in self._steps_after(last[self.time])
        ]
        grid = pa.Table.from_pylist(rows, schema=self.context_.select(columns).schema)
        return grid.to_pandas()

    @public
    def save(self, path: str | Path, *, name: str | None = None) -> "Predictor":
        """Writes the fitted forecaster as a predictor directory at `path`.

        Returns:
            The saved predictor, `gf.Predictor(path)`, with `gf.yaml`,
            `gf.lock` and the context snapshot.
        """
        from generalforecast.predictor import Predictor

        self._require_fitted()
        if name is None:
            name = Path(path).resolve().name
        spec = admitted_spec(
            self.model, "forecast", self.context_, target=self.target, **self._intent()
        )
        return Predictor(
            export_predictor(path, name=name, spec=spec, context=self.context_)
        )

    def close(self) -> None:
        """Releases native context; the estimator must be fitted again."""
        for name in (
            "context_",
            "n_features_in_",
            "feature_names_in_",
            "_spec",
            "_clock",
            "_frequency",
            "_step",
            "_identities",
        ):
            self.__dict__.pop(name, None)
