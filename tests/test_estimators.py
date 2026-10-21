"""sklearn-style estimators on any client."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

import generalforecast
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.tabicl.tiny import classifier, tiny_tabicl

LABELS = np.array(["cat", "dog", "fish"])


def _data(rows: int = 32) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(rows, 5)).astype(np.float32)
    return features, LABELS[np.arange(rows) % 3]


def test_classifiers_learn_their_classes_from_the_labels() -> None:
    features, labels = _data()
    model = "tests.models.tabicl.tiny:classifier"

    estimator = generalforecast.Classifier(model).fit(features, labels)
    probabilities = estimator.predict_proba(features[:4])

    table = generalforecast.local().create_table(
        "check", model=model, features=5, classes=3
    )
    table.insert(np.arange(32), features, np.arange(32) % 3)
    np.testing.assert_array_equal(estimator.classes_, LABELS)
    np.testing.assert_allclose(probabilities, table.predict(features[:4]), atol=1e-6)
    np.testing.assert_array_equal(
        estimator.predict(features[:4]), LABELS[probabilities.argmax(1)]
    )
    assert 0.0 <= estimator.score(features, labels) <= 1.0


def test_regressors_predict_medians_and_quantiles() -> None:
    features, _ = _data()
    targets = features.sum(axis=1)
    model = (tabicl_functions, tiny_tabicl(regression=True))

    estimator = generalforecast.Regressor(model).fit(features, targets)
    quantiles = estimator.predict_quantiles(features[:4])

    assert quantiles.shape == (4, 999)
    np.testing.assert_array_equal(estimator.predict(features[:4]), quantiles[:, 499])
    assert np.isfinite(estimator.score(features, targets))


def test_estimators_work_with_sklearn_tools() -> None:
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    features, labels = _data(48)
    estimator = generalforecast.Classifier(
        classifier(), strategy=generalforecast.Strategy.exact()
    )

    copy = clone(estimator)
    scores = cross_val_score(estimator, features, labels, cv=2)

    assert copy.get_params()["strategy"] is estimator.get_params()["strategy"]
    assert scores.shape == (2,) and np.isfinite(scores).all()


def test_predicting_before_fitting_is_an_error() -> None:
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError, match="fit"):
        generalforecast.Classifier(classifier()).predict(
            np.zeros((1, 5), dtype=np.float32)
        )


def test_estimators_pass_sklearns_checks() -> None:
    from sklearn.utils.estimator_checks import check_estimator

    features, labels = _data()
    estimator = generalforecast.Classifier(classifier()).fit(features, labels)

    assert estimator.n_features_in_ == 5
    assert estimator.__sklearn_tags__().input_tags.allow_nan
    with pytest.raises(ValueError, match="5"):
        estimator.predict(features[:, :4])
    # The tiny random model need not beat chance.
    check_estimator(
        generalforecast.Classifier(classifier()),
        expected_failed_checks={
            "check_classifiers_train": "the tiny model's weights are random",
        },
    )
    check_estimator(
        generalforecast.Regressor((tabicl_functions, tiny_tabicl(regression=True))),
        expected_failed_checks={
            "check_regressors_train": "the tiny model's weights are random",
        },
    )


def test_frames_keep_their_string_and_datetime_columns() -> None:
    pa = pytest.importorskip("pyarrow")
    features, labels = _data()
    stores = np.array(["north", "south", "east", "west"])[np.arange(32) % 4]
    days = (np.datetime64("2026-01-01") + np.arange(32)).astype("datetime64[ms]")
    frame = pa.table(
        {"store": stores, "day": days, **{f"x{i}": features[:, i] for i in range(3)}}
    )
    model = "tests.models.tabicl.tiny:classifier"

    estimator = generalforecast.Classifier(model).fit(frame, labels)
    probabilities = estimator.predict_proba(frame.slice(0, 4))

    columns = {"store": "categorical", "day": "datetime", "x0": "numeric"}
    columns |= {"x1": "numeric", "x2": "numeric"}
    table = generalforecast.local().create_table(
        "typed", model=model, columns=columns, classes=3
    )
    table.insert(np.arange(32), frame, np.unique(labels, return_inverse=True)[1])
    np.testing.assert_allclose(
        probabilities, table.predict(frame.slice(0, 4)), atol=1e-6
    )
    assert list(estimator.feature_names_in_) == list(columns)
    assert estimator.n_features_in_ == 5
    copy = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_allclose(
        copy.predict_proba(frame.slice(0, 4)), probabilities, atol=1e-6
    )
    with pytest.raises(ValueError, match="frames"):
        estimator.predict(features)


def test_fitted_estimators_pickle_and_make_their_table_again() -> None:
    features, labels = _data()
    estimator = generalforecast.Classifier(
        classifier(), client=generalforecast.local()
    ).fit(features, labels)

    copy = pickle.loads(pickle.dumps(estimator))

    np.testing.assert_allclose(
        copy.predict_proba(features[:4]),
        estimator.predict_proba(features[:4]),
        atol=1e-6,
    )
    copy.close()
    with pytest.raises(NotFittedError):
        copy.predict(features[:4])


@pytest.fixture
def server(tmp_path: Path) -> Iterator[object]:
    """A server for the tests marked `server`."""
    pytest.importorskip("pyarrow")
    from tests.utils.server import Server, build

    running = Server(build(), tmp_path).start()
    yield running
    running.stop()


@pytest.mark.server
def test_estimators_fit_and_predict_on_a_server(server) -> None:
    features, labels = _data()
    from tests.utils.legacy_client import connect

    client = connect(f"grpc://127.0.0.1:{server.port}", token="secret")

    remote = generalforecast.Classifier("tiny", client=client).fit(features, labels)
    local = generalforecast.Classifier(classifier()).fit(features, labels)

    np.testing.assert_allclose(
        remote.predict_proba(features[:4]), local.predict_proba(features[:4]), atol=1e-5
    )
    # Each estimator holds one table, until it closes.
    remote.fit(features, labels)
    assert len(client.tables()) == 1
    remote.close()
    assert client.tables() == []
