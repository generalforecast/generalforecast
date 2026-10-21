# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real pinned CPU weights: @K, *N and grouped forecasts agree across surfaces.

The Models API (`CatalogService`, which `LocalModels` and HTTP routes share), a
package revision and trusted local loading all run the same verified weights
through the same contract calls, so one request answers the same everywhere.
Requires the pinned TabICL and Chronos-2 artifacts, downloaded if missing.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pytest

from generalforecast.deployments.boundary import Scope
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.interfaces import InferenceRequest
from generalforecast.deployments.package_runtime import PackageExecutor
from generalforecast.local_models import LocalModels
from generalforecast.models import inference, registry
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.outputs import class_key
from tests.product_runtime.test_executor import checked

pytestmark = pytest.mark.slow
SCOPE = Scope("local", "real-variants")
CLASSES = [f"c{i:02d}" for i in range(25)]


def _context(classes: bool) -> tuple[list, list]:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(100, 3)).round(3)
    labels = (
        [CLASSES[i % 25] for i in range(100)]
        if classes
        else (rows @ [1.0, -2.0, 0.5]).round(3).tolist()
    )
    return rows.tolist(), labels


def _request(
    model: str, task: Literal["classification", "regression"]
) -> InferenceRequest:
    rows, labels = _context(task == "classification")
    return {
        "model": model,
        "task": task,
        "target": "y",
        "features": ["a", "b", "c"],
        "context": [
            {
                "key": f"c{i}",
                "features": dict(zip("abc", row, strict=True)),
                "label": label,
            }
            for i, (row, label) in enumerate(zip(rows, labels, strict=True))
        ],
        "queries": [
            {"key": f"q{i}", "features": dict(zip("abc", row, strict=True))}
            for i, row in enumerate(rows[:5])
        ],
        "options": {} if task == "classification" else {"quantiles": [0.1, 0.5, 0.9]},
    }


def _local(model: str, task: str) -> np.ndarray:
    rows, labels = _context(task == "classification")
    functions, parameters = registry.load(model)
    assert isinstance(functions, TabularFunctions)
    loaded = (functions, parameters)
    classes = sorted(set(labels)) if task == "classification" else []
    targets = [classes.index(label) for label in labels] if classes else labels
    fitted = inference.fit(
        loaded,
        jnp.asarray(rows, jnp.float32),
        jnp.asarray(targets, jnp.int32 if classes else jnp.float32),
        len(classes),
    )
    return inference.predict(
        loaded, fitted, jnp.asarray(rows[:5], jnp.float32), len(classes)
    )


@pytest.mark.parametrize("model", ["tabicl-classifier@4", "tabicl-classifier@4*2"])
def test_many_classes_agree_across_models_api_sdk_and_local(model):
    request = _request(model, "classification")
    remote = CatalogService(allow_downloads=True).infer(SCOPE, request)
    sdk = LocalModels(allow_downloads=True).infer(request)
    local = _local(model, "classification")
    assert remote["outputs"] == sdk["outputs"]
    for output, row in zip(remote["outputs"], local, strict=True):
        assert list(output["probabilities"]) == [class_key(c) for c in CLASSES]
        np.testing.assert_allclose(
            list(output["probabilities"].values()), row, atol=1e-5
        )
    _, labels = _context(True)
    predicted = [
        max(o["probabilities"], key=o["probabilities"].get) for o in remote["outputs"]
    ]
    # Codes tell all 25 classes apart: most context rows are recalled.
    assert sum(p == label for p, label in zip(predicted, labels, strict=False)) >= 3


def test_regressor_ensemble_agrees_across_models_api_sdk_and_local():
    request = _request("tabicl-regressor*2", "regression")
    remote = CatalogService(allow_downloads=True).infer(SCOPE, request)
    sdk = LocalModels(allow_downloads=True).infer(request)
    local = _local("tabicl-regressor*2", "regression")
    assert remote["outputs"] == sdk["outputs"]
    for output, row in zip(remote["outputs"], local, strict=True):
        np.testing.assert_allclose(
            [output["quantiles"][k] for k in ("0.1", "0.5", "0.9")],
            row[[99, 499, 899]],
            rtol=1e-5,
        )
    base = CatalogService(allow_downloads=True).infer(
        SCOPE, {**request, "model": "tabicl-regressor"}
    )
    assert base["outputs"] != remote["outputs"]


def test_package_runtime_runs_the_same_variant_as_the_models_api(tmp_path):
    rows, labels = _context(True)
    context = pa.table(
        {
            "a": [r[0] for r in rows],
            "b": [r[1] for r in rows],
            "c": [r[2] for r in rows],
            "y": labels,
        }
    )
    package = checked(
        tmp_path, context, model="tabicl-classifier@4*2", outputs=["probabilities"]
    )
    assert package.lock.model.codes == 4 and package.lock.model.estimators == 2
    assert package.lock.model.max_classes == 1000
    prepared = PackageExecutor(allow_downloads=True).prepare(package.lock, context)
    predicted = prepared.predict(context.slice(0, 5).drop(["y"])).to_pylist()
    remote = CatalogService(allow_downloads=True).infer(
        SCOPE, _request("tabicl-classifier@4*2", "classification")
    )
    for row, output in zip(predicted, remote["outputs"], strict=True):
        np.testing.assert_allclose(
            list(row["probabilities"].values()),
            list(output["probabilities"].values()),
            atol=1e-5,
        )


def test_grouped_forecast_agrees_across_models_api_and_package(tmp_path):
    context = pa.Table.from_pylist(
        [
            {
                "series": s,
                "time": i,
                "y": float(i * (1 if s == "a" else 2) + np.sin(i)),
            }
            for s in ("a", "b")
            for i in range(40)
        ]
    )
    package = checked(
        tmp_path,
        context,
        model="chronos-2",
        time="time",
        series=["series"],
        horizon=3,
        outputs=["value", "quantiles"],
        quantiles=[0.1, 0.5, 0.9],
    )
    prepared = PackageExecutor(allow_downloads=True).prepare(package.lock, context)
    queries = pa.table({"series": ["a"] * 3 + ["b"] * 3, "time": [40, 41, 42] * 2})
    packaged = prepared.predict(queries).to_pylist()
    request: InferenceRequest = {
        "model": "chronos-2",
        "task": "forecast",
        "target": "y",
        "features": [],
        "context": [
            {
                "key": f"{r['series']}{r['time']}",
                "series": r["series"],
                "time": r["time"],
                "value": r["y"],
                "covariates": {},
            }
            for r in context.to_pylist()
        ],
        "queries": [
            {
                "key": s,
                "series": s,
                "origin": 39,
                "group": ["a", "b"],
                "future_covariates": {},
            }
            for s in ("a", "b")
        ],
        "options": {
            "series": {
                "step": 1,
                "horizon": 3,
                "quantiles": [0.1, 0.5, 0.9],
                "known_covariates": [],
            }
        },
    }
    remote = CatalogService(allow_downloads=True).infer(SCOPE, request)
    for index, output in enumerate(remote["outputs"]):
        for level in ("0.1", "0.5", "0.9"):
            np.testing.assert_allclose(
                output["quantiles"][level],
                [r["quantiles"][level] for r in packaged[index * 3 : index * 3 + 3]],
                rtol=1e-5,
            )
