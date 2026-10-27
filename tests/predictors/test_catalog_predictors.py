# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every catalog model family resolves, locks and serves as a predictor, pinned.

A model's pin is `catalog.Artifact.digest`: its file manifest's digest, the
files as they load or as they convert from upstream; or, for a repository
published without file hashes, the digest of its immutable upstream commit.
`gf check` records it, serving compiles against the same pin, and hosted
serving admits by it (`hosting.pin`). Nothing here downloads weights.
"""

import datetime
from pathlib import Path

import pyarrow as pa
import pytest

from generalforecast import hosting
from generalforecast.models import catalog
from generalforecast.predictors.models import CatalogResolver
from generalforecast.predictors.runtime.compile import compile_lock
from generalforecast.predictors.runtime.context import served_context
from generalforecast.predictors.validation import PredictorError
from generalforecast.serialization import content_digest
from generalforecast.serve import models as serve_models
from tests.utils.locks import directory, lock_of

FAMILIES = [
    "tabicl-v2-classifier",
    "tabicl-v2-regressor",
    "kumo-tabular-small-classifier",
    "tabpfn-3-classifier",
    "tabpfn-3.5-regressor",
    "limix-2-classifier",
    "limix-2-regressor",
    "mitra-classifier",
    "mitra-2-regressor",
    "chronos-2",
    "timestral-28m",
    "timestral-209m-unmirrored",
    "timesfm-3",
]


def _rows(task: str) -> tuple[pa.Table, dict]:
    if task == "forecast":
        days = [
            datetime.date(2025, 1, 1) + datetime.timedelta(days=d) for d in range(20)
        ]
        rows = pa.table({"date": days, "units": [float(d % 7) for d in range(20)]})
        return rows, {"target": "units", "time": "date", "horizon": 2}
    labels = (
        [i % 2 for i in range(20)]
        if task == "classification"
        else [float(i) for i in range(20)]
    )
    rows = pa.table({"x": [float(i) for i in range(20)], "y": labels})
    return rows, {"target": "y"}


def test_every_catalog_entry_is_covered():
    families = {name.rsplit("-", 1)[0].split("-")[0] for name in FAMILIES}
    assert families >= {entry.name.split("-")[0] for entry in catalog.entries()}


@pytest.mark.parametrize("name", FAMILIES)
def test_a_family_resolves_locks_and_serves_by_its_pin(tmp_path, monkeypatch, name):
    model = f"general-forecast/{name}"
    entry = catalog.parse(model).entry
    rows, manifest = _rows(entry.capability.task)
    root = directory(tmp_path / "predictor", rows, model=model, **manifest)
    lock = lock_of(root)
    pinned = lock.predictor["model_digest"]
    assert pinned == entry.artifact.digest
    if not entry.artifact.files:
        assert pinned == content_digest(
            {
                "repository": entry.artifact.repository,
                "revision": entry.artifact.revision,
            }
        )
    plan = compile_lock(
        lock, served_context(lock, rows), resolver=CatalogResolver("local")
    )
    assert plan.model.model == model and plan.model.digest == pinned
    monkeypatch.setattr(serve_models, "checkpoint", lambda *_, **__: None)
    (tmp_path / "served").mkdir()
    entry_served = serve_models.served(str(root), tmp_path / "served")
    assert entry_served["predictor"]["lock_digest"] == lock.digest
    assert entry_served["resolved"].endswith(f"{name}@{pinned}")
    ad_hoc = hosting.ad_hoc_lock({"model": model, **manifest}, rows)
    assert ad_hoc.predictor["model_digest"] == pinned
    if entry.gates.predictor:
        assert hosting.pin(model).digest == pinned
    else:
        with pytest.raises(PredictorError) as caught:
            hosting.pin(model)
        assert caught.value.code == "ModelResolutionRequired"


def test_serving_refuses_a_revision_pin_that_moved(tmp_path):
    model = "general-forecast/limix-2-classifier"
    rows, manifest = _rows("classification")
    root = directory(tmp_path / "limix", rows, model=model, **manifest)
    document = lock_of(root).to_dict()
    from generalforecast.predictors.lock import lock_digest

    document["predictor"]["model_digest"] = content_digest(
        {"repository": "general-forecast/limix-2-classifier", "revision": "0" * 40}
    )
    document["digest"] = lock_digest(document)
    with pytest.raises(PredictorError) as caught:
        compile_lock(
            document,
            served_context(document, rows),
            resolver=CatalogResolver("local"),
        )
    assert caught.value.code == "ModelMismatch"
    assert Path(root / "gf.lock").is_file()
