# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real pinned CPU checkpoints, including the 478 MB Chronos-2 artifact."""

import copy

import numpy as np
import pytest

from generalforecast.deployments.boundary import Scope, json_object
from generalforecast.deployments.catalog import create_catalog
from generalforecast.deployments.interfaces import InferenceRequest

pytestmark = pytest.mark.slow
SCOPE = Scope("native-catalog-project", "test-catalog")


@pytest.mark.parametrize(
    "task,model",
    [
        (task, f"{name}-{kind}{variant if kind == 'classifier' else ''}")
        for name in (
            "tabicl-v2",
            "kumo-tabular-small",
            "kumo-tabular-medium",
            "kumo-tabular-large",
            "mitra-2",
        )
        for task, kind in (
            ("classification", "classifier"),
            ("regression", "regressor"),
        )
        for variant in ("", "@2*2")
        if not (variant and kind == "regressor")
    ],
)
def test_real_pinned_hosted_tabular_request_context(task, model, tmp_path, monkeypatch):
    request: InferenceRequest = {
        "model": model,
        "task": task,
        "target": "outcome",
        "features": ["x"],
        "context": [
            {
                "key": str(i),
                "features": {"x": i},
                "label": bool(i >= 2) if task == "classification" else float(i + 1),
            }
            for i in range(4)
        ],
        "queries": [{"key": "q", "features": {"x": 1.5}}],
        "options": {},
    }
    before = copy.deepcopy(request)
    monkeypatch.chdir(tmp_path)
    result = create_catalog().infer(SCOPE, request)
    assert request == before and list(tmp_path.iterdir()) == []
    assert result["artifact"]["verified"] is True
    assert result["context_provenance"]["persisted"] is False
    output = result["outputs"][0]
    if task == "classification":
        probabilities = output["probabilities"]
        assert set(probabilities) == {"false", "true"}
        assert sum(probabilities.values()) == pytest.approx(1, abs=1e-5)
        assert all(0 <= value <= 1 for value in probabilities.values())
    else:
        values = list(output["quantiles"].values())
        assert len(values) == 3 and np.isfinite(values).all()
        assert values == sorted(values)
    json_object(dict(result))


@pytest.mark.parametrize(
    "model",
    [
        *(
            f"kumo-tabular-{size}-{task}"
            for size in ("small", "medium", "large")
            for task in ("classifier", "regressor")
        ),
        "mitra-2-classifier",
        "mitra-2-regressor",
    ],
)
def test_real_pinned_local_kumo_and_mitra_weights(model):
    import jax.numpy as jnp

    from generalforecast.models import inference, registry

    classifier = model.endswith("-classifier")
    classes = 2 if classifier else 0
    loaded = registry.load_tabular(model)
    context = jnp.array(
        [[-1, 0], [-0.5, 0.25], [0.1, 0.4], [0.4, 0.5], [0.9, 1.1], [1.2, 1]],
        dtype=jnp.float32,
    )
    targets = jnp.array(
        [0, 0, 0, 1, 1, 1] if classifier else [-1, -0.5, 0.1, 0.4, 0.9, 1.2],
        dtype=jnp.int32 if classifier else jnp.float32,
    )
    fitted = inference.fit(loaded, context, targets, classes)
    outputs = inference.predict(
        loaded, fitted, jnp.array([[0.2, 0.3], [0.8, 0.9]], dtype=jnp.float32), classes
    )
    assert outputs.shape == (2, 2 if classifier else 999)
    assert np.isfinite(outputs).all()
    if classifier:
        np.testing.assert_allclose(outputs.sum(axis=1), 1, atol=1e-5)
    else:
        assert (np.diff(outputs, axis=1) >= 0).all()


def test_real_pinned_chronos_grouped_known_future_forecast(tmp_path, monkeypatch):
    request: InferenceRequest = {
        "model": "chronos-2",
        "task": "forecast",
        "target": "sales",
        "features": [],
        "context": [
            {
                "key": f"{name}-{i}",
                "series": name,
                "time": i,
                "value": (10 + i + (i % 3) if name == "a" else 50 + 2 * i + (i % 4) * 7)
                if i != 7
                else None,
                "covariates": {"promo": i % 2, "past": i / 10},
            }
            for name in ["a", "b"]
            for i in range(16)
        ],
        "queries": [
            {
                "key": "q-" + name,
                "series": name,
                "origin": 15,
                "group": ["a", "b"],
                "future_covariates": {"promo": [0, 1, 0]},
            }
            for name in ["a", "b"]
        ],
        "options": {
            "series": {
                "step": 1,
                "horizon": 3,
                "quantiles": [0.1, 0.5, 0.9],
                "known_covariates": ["promo"],
            }
        },
    }
    before = copy.deepcopy(request)
    monkeypatch.chdir(tmp_path)
    service = create_catalog()
    result = service.infer(SCOPE, request)
    assert request == before
    assert result["artifact"]["verified"] is True
    assert (
        result["artifact"]["digest"]
        == "sha256:11b8442ad8a70e4cae0f95fba1a4dac767ed6e60bbdc9870a4fccf5e4ff38c88"
    )
    assert result["context_provenance"]["usable_observations"] == 30
    assert result["context_provenance"]["persisted"] is False
    assert len(result["outputs"]) == 2
    for output in result["outputs"]:
        assert output["times"] == [16, 17, 18]
        assert set(output["quantiles"]) == {"0.1", "0.5", "0.9"}
        assert all(
            len(values) == 3 and np.isfinite(values).all()
            for values in output["quantiles"].values()
        )
    original = result["outputs"][0]["quantiles"]["0.5"]
    request["queries"] = request["queries"][:1]
    request["queries"][0]["future_covariates"]["promo"] = [100, -100, 100]
    changed_future = service.infer(SCOPE, request)["outputs"][0]["quantiles"]["0.5"]
    assert not np.array_equal(original, changed_future)
    request["queries"][0]["future_covariates"]["promo"] = [0, 1, 0]
    request["queries"][0]["group"] = ["a"]
    ungrouped = service.infer(SCOPE, request)["outputs"][0]["quantiles"]["0.5"]
    assert not np.array_equal(original, ungrouped)
    assert list(tmp_path.iterdir()) == [] and vars(service) == {"allow_downloads": True}
    json_object(dict(result))
