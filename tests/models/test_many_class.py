# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Error-correcting output codes, for more classes than a classifier has slots."""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import catalog
from generalforecast.models.many_class import ManyClass, codebook, many_class_functions
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.tabicl.tiny import CLASS_SLOTS, tiny_tabicl
from tests.utils.engine_client import local
from tests.utils.predict import predict


def test_codes_tell_every_pair_of_classes_apart() -> None:
    book = codebook(8, 40, CLASS_SLOTS)

    assert book.shape == (8, 40) and book.max() == CLASS_SLOTS - 1
    # Groups are as even as they can be: 40 classes in 10 groups of 4.
    assert all((np.bincount(row) == 4).all() for row in book)
    assert all(
        (book[:, first] != book[:, second]).any()
        for first, second in itertools.combinations(range(40), 2)
    )
    # Random permutations alone leave two of 25 classes alike in both of two
    # codes over ten slots; trades between classes part them.
    tight = codebook(2, 25, CLASS_SLOTS)
    assert len({tuple(column) for column in tight.T}) == 25
    assert all(sorted(np.bincount(row)) == [2] * 5 + [3] * 5 for row in tight)


def test_tables_with_more_classes_than_slots_predict_every_class() -> None:
    generator = np.random.default_rng(0)
    classes = 25
    features = jnp.asarray(generator.normal(size=(100, 4)), dtype=jnp.float32)
    targets = jnp.asarray(np.arange(100) % classes, dtype=jnp.int32)
    queries = jnp.asarray(generator.normal(size=(6, 4)), dtype=jnp.float32)
    model = ManyClass(tiny_tabicl(regression=False), 6)

    probabilities = predict(
        many_class_functions(tabicl_functions), model, features, targets, queries
    )

    assert probabilities.shape == (6, classes)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, rtol=1e-5)
    # One code over ten slots cannot tell 25 classes apart.
    with pytest.raises(ValueError, match="more codes"):
        predict(
            many_class_functions(tabicl_functions),
            ManyClass(tiny_tabicl(regression=False), 1),
            features,
            targets,
            queries,
        )
    # A table the codes cannot serve is refused when it is created.
    with pytest.raises(ValueError, match="more codes"):
        local().create_table(
            "t", model="general-forecast/tabicl-v2-classifier@1", features=4, classes=25
        )
    for spec in ("classifier@0", "classifier@8@4"):
        with pytest.raises(ValueError, match="spec"):
            catalog.parse(f"general-forecast/tabicl-v2-{spec}")
    with pytest.raises(ValueError, match="classifier"):
        catalog.parse("general-forecast/tabicl-v2-regressor@2")


def test_an_unknown_class_stays_unknown_in_every_code() -> None:
    from generalforecast.models.limix.model import limix_functions
    from tests.models.limix.tiny import tiny_limix

    functions = many_class_functions(limix_functions)
    model = ManyClass(tiny_limix(regression=False), 6)
    features = jnp.asarray(np.random.default_rng(0).normal(size=(30, 5)), jnp.float32)
    targets = jnp.asarray(np.arange(30) % 25, jnp.int32).at[4].set(-1)

    state, rows = functions.fit(model, features, targets, 25)

    # Each code's estimator embeds the row as it does a query, by its mask.
    query = functions.embed(model, state, features[4:5])
    np.testing.assert_allclose(rows[4], query[0], atol=1e-6)
