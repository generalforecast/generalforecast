# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checks a model against the contract the engine relies on, on random inputs.

A model written against `generalforecast.models.contract` is run by the engine in ways a
quick test of its predictions would not show: rows padded to fixed sizes with
garbage past their counts, tables batched with `jax.vmap`, queries decoded
apart and in chunks, and contexts joined with a mask saying which keys each
query reads. `check` runs a model through each of these on small random tables
and raises if any changes what it predicts. `check_series` does the same for
a forecaster written against `generalforecast.models.series`, on small random groups.

    from generalforecast.models.conformance import check

    check(kernel_functions, Kernel(temperature=jnp.asarray(2.0), width=32))
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.models.series import (
    KNOWN,
    TARGET,
    SeriesFunctions,
    SeriesGroup,
    SeriesProperties,
)
from generalforecast.nn.attention import AttentionCache
from generalforecast.runtime.paging import cache_layout
from generalforecast.runtime.shapes import pad_rows


class Table(NamedTuple):
    """Context rows, their targets, query rows and the task's class count."""

    features: jax.Array
    targets: jax.Array
    queries: jax.Array
    num_classes: int


def table(
    regression: bool,
    rows: int = 32,
    queries: int = 8,
    features: int = 5,
    seed: int = 0,
) -> Table:
    """A random table with a missing value in the context and in the queries.

    Args:
        regression: Whether the targets are values rather than classes.
        rows: Context rows.
        queries: Query rows.
        features: Columns.
        seed: The random seed.

    Returns:
        The table, with three classes for a classifier.
    """
    generator = np.random.default_rng(seed)
    context = generator.normal(size=(rows, features)).astype(np.float32)
    query_rows = generator.normal(size=(queries, features)).astype(np.float32)
    context[3, 1] = np.nan
    query_rows[2, 0] = np.nan
    if regression:
        targets = jnp.asarray(np.nansum(context, axis=1), dtype=jnp.float32)
        num_classes = 0
    else:
        targets = jnp.asarray(generator.permutation(np.arange(rows) % 3), jnp.int32)
        num_classes = 3
    return Table(jnp.asarray(context), targets, jnp.asarray(query_rows), num_classes)


def predict(
    functions: TabularFunctions,
    model: Any,
    data: Table,
    num_rows: int | jax.Array | None = None,
) -> jax.Array:
    """The five contract calls in the order the engine makes them.

    Args:
        functions: The model's contract calls.
        model: Its parameters.
        data: The table.
        num_rows: Real context rows, the rest padding, if any.

    Returns:
        The query rows' predictions.
    """
    state, context_rows = functions.fit(
        model, data.features, data.targets, data.num_classes, num_rows=num_rows
    )
    cache = functions.encode(
        model, state, context_rows, data.targets, num_rows=num_rows
    )
    query_rows = functions.embed(model, state, data.queries)
    return functions.postprocess(
        model, state, functions.decode(model, state, cache, query_rows)
    )


def padded(data: Table, context_size: int, query_size: int, seed: int = 0) -> Table:
    """A table padded with large garbage rows and a NaN, which must not reach the real rows.

    Args:
        data: The table.
        context_size: Context rows after padding.
        query_size: Query rows after padding.
        seed: The random seed of the garbage.

    Returns:
        The padded table.
    """

    def garbage(values: jax.Array, size: int, offset: int) -> jax.Array:
        array = np.array(pad_rows(values, size, fill=0.0))
        tail = array[values.shape[0] :]
        tail[:] = 1e3 * np.random.default_rng(seed + offset).normal(size=tail.shape)
        tail[0, 0] = np.nan
        return jnp.asarray(array)

    fill = 1e4 if jnp.issubdtype(data.targets.dtype, jnp.floating) else 2
    return Table(
        garbage(data.features, context_size, 0),
        pad_rows(data.targets, context_size, fill=fill),
        garbage(data.queries, query_size, 1),
        data.num_classes,
    )


def close(actual: Any, expected: Any, tolerance: float = 1e-5) -> None:
    """Raises unless two trees of arrays match within `tolerance`."""
    jax.tree.map(
        lambda a, e: np.testing.assert_allclose(
            np.asarray(a, dtype=np.float32),
            np.asarray(e, dtype=np.float32),
            rtol=tolerance,
            atol=tolerance,
        ),
        actual,
        expected,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check(
    functions: TabularFunctions,
    parameters: Any,
    *,
    features: int = 5,
    tolerance: float = 1e-5,
) -> None:
    """Runs a model through every way the engine calls it, on random tables.

    Checks that its properties describe its outputs and need only parameter
    shapes; that `fit` returns the context rows `embed` would; that rows embed
    and queries decode apart as together; that it jits; that padding past
    `num_rows`, garbage and NaN included, reaches no real row; that columns
    constant or empty in the context are left out; that tables batch with
    `jax.vmap`;
    that `decode` reads only the keys a cache's mask allows; and that its
    caches page by rows.

    Args:
        functions: The model's contract calls.
        parameters: Its parameters.
        features: Columns of the random tables.
        tolerance: How far results may move between equivalent calls.

    Raises:
        AssertionError: Naming the first check the model fails.
    """
    properties = functions.properties(parameters)
    _require(
        isinstance(properties, TabularProperties),
        "properties must return TabularProperties",
    )
    regression = properties.regression
    for name, run in _CHECKS:
        try:
            run(functions, parameters, properties, regression, features, tolerance)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None


def _properties(functions, model, properties, regression, features, tolerance):
    """Properties describe the outputs, and follow from parameter shapes alone."""
    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), model)
    _require(
        functions.properties(abstract) == properties,
        "properties must follow from parameter shapes alone",
    )
    hash(properties)
    data = table(regression, features=features)
    state, context_rows = functions.fit(
        model, data.features, data.targets, data.num_classes
    )
    query_rows = functions.embed(model, state, data.queries)
    predictions = predict(functions, model, data)
    _require(
        context_rows.ndim == 2 and len(context_rows) == data.features.shape[0],
        f"fit returned context rows of shape {context_rows.shape}, "
        f"not one row each for {data.features.shape[0]} rows",
    )
    _require(
        query_rows.shape == (data.queries.shape[0], context_rows.shape[1]),
        f"embed returned query rows of shape {query_rows.shape}, unlike the "
        f"context's {context_rows.shape}",
    )
    columns = properties.num_outputs if regression else data.num_classes
    _require(
        predictions.shape == (data.queries.shape[0], columns),
        f"postprocess returned predictions of shape {predictions.shape}, "
        f"not ({data.queries.shape[0]}, {columns})",
    )
    if not regression:
        close(predictions.sum(axis=-1), jnp.ones(data.queries.shape[0]), tolerance)


def _fit_embeds(functions, model, properties, regression, features, tolerance):
    """`fit` returns the context rows `embed` would, and rows embed apart as together."""
    data = table(regression, features=features)
    state, context_rows = functions.fit(
        model, data.features, data.targets, data.num_classes
    )
    close(context_rows, functions.embed(model, state, data.features, data.targets))
    apart = jnp.concatenate(
        [
            functions.embed(model, state, data.queries[:3]),
            functions.embed(model, state, data.queries[3:]),
        ]
    )
    close(apart, functions.embed(model, state, data.queries), tolerance)


def _decodes_apart(functions, model, properties, regression, features, tolerance):
    """Queries decode apart as together."""
    data = table(regression, features=features)
    state, context_rows = functions.fit(
        model, data.features, data.targets, data.num_classes
    )
    cache = functions.encode(model, state, context_rows, data.targets)
    query_rows = functions.embed(model, state, data.queries)
    together = functions.decode(model, state, cache, query_rows)
    apart = jnp.concatenate(
        [
            functions.decode(model, state, cache, query_rows[:3]),
            functions.decode(model, state, cache, query_rows[3:]),
        ]
    )
    close(apart, together, tolerance)


def _jits(functions, model, properties, regression, features, tolerance):
    """The contract calls compile with `jax.jit`."""
    data = table(regression, features=features)
    jitted = TabularFunctions(
        fit=jax.jit(functions.fit, static_argnames="num_classes"),
        embed=jax.jit(functions.embed),
        encode=jax.jit(functions.encode),
        decode=jax.jit(functions.decode),
        postprocess=jax.jit(functions.postprocess),
        properties=functions.properties,
    )
    close(predict(jitted, model, data), predict(functions, model, data), tolerance)


def _padding(functions, model, properties, regression, features, tolerance):
    """Padding past `num_rows`, garbage and NaN included, reaches no real row."""
    data = table(regression, rows=20, queries=6, features=features)
    pad = padded(data, context_size=32, query_size=8)
    state, context_rows = functions.fit(
        model, data.features, data.targets, data.num_classes
    )
    padded_state, padded_rows = functions.fit(
        model, pad.features, pad.targets, data.num_classes, num_rows=20
    )
    close(padded_state, state, tolerance)
    close(padded_rows[:20], context_rows, tolerance)
    close(
        predict(functions, model, pad, num_rows=20)[:6],
        predict(functions, model, data),
        tolerance,
    )


def _constant_columns(functions, model, properties, regression, features, tolerance):
    """Columns constant or empty in the context are left out of predictions.

    Upstream preprocessing drops them, so what they hold, in the context or
    in the queries, changes nothing. The reference tests check that the rest
    predicts as upstream does without them.
    """
    data = table(regression, features=features)
    context, queries = np.array(data.features), np.array(data.queries)
    context[:, 0] = np.nan
    context[:, 1] = 5.0
    queries[:, 1] = 6.0
    predictions = predict(
        functions,
        model,
        Table(
            jnp.asarray(context), data.targets, jnp.asarray(queries), data.num_classes
        ),
    )
    _require(bool(np.isfinite(predictions).all()), "predictions are not finite")
    context[:, 1] = -3.0
    queries[:, 0] = np.linspace(-50, 50, len(queries))
    queries[:, 1] = np.nan
    changed = Table(
        jnp.asarray(context), data.targets, jnp.asarray(queries), data.num_classes
    )
    close(predictions, predict(functions, model, changed), tolerance)


def _vmap(functions, model, properties, regression, features, tolerance):
    """Tables of different sizes, padded to one, batch with `jax.vmap`."""
    tables = [
        table(regression, rows=rows, queries=6, features=features, seed=seed)
        for rows, seed in ((20, 0), (27, 1))
    ]
    pads = [padded(data, 32, 8, seed) for seed, data in enumerate(tables)]
    stacked = [
        jnp.stack(values) for values in zip(*(data[:3] for data in pads), strict=True)
    ]

    def one(context: jax.Array, targets: jax.Array, queries: jax.Array, rows):
        data = Table(context, targets, queries, tables[0].num_classes)
        return predict(functions, model, data, num_rows=rows)

    context, targets, queries = stacked
    rows = jnp.asarray([20, 27], dtype=jnp.int32)
    batched = jax.vmap(one)(context, targets, queries, rows)
    for index, data in enumerate(tables):
        close(batched[index, :6], predict(functions, model, data), tolerance)


def _mask(functions, model, properties, regression, features, tolerance):
    """`decode` reads only the keys a cache's mask allows."""
    data = table(regression, rows=24, queries=4, features=features)
    state, rows = functions.fit(model, data.features, data.targets, data.num_classes)
    first = functions.encode(model, state, rows[:12], data.targets[:12])
    second = functions.encode(model, state, rows[12:], data.targets[12:])
    queries = functions.embed(model, state, data.queries)
    # Queries 0 and 1 read the first part and 2 and 3 the second, as when the
    # engine decodes several units in one call.
    mask = jnp.array(
        [[True] * 12 + [False] * 12] * 2 + [[False] * 12 + [True] * 12] * 2
    )

    def join(a: AttentionCache, b: AttentionCache) -> AttentionCache:
        return AttentionCache(
            key=jnp.concatenate([a.key, b.key]),
            value=jnp.concatenate([a.value, b.value]),
            length=None,
            mask=mask,
        )

    joined = jax.tree.map(
        join, first, second, is_leaf=lambda node: isinstance(node, AttentionCache)
    )
    outputs = functions.decode(model, state, joined, queries)
    close(outputs[:2], functions.decode(model, state, first, queries[:2]), tolerance)
    close(outputs[2:], functions.decode(model, state, second, queries[2:]), tolerance)


def _pages(functions, model, properties, regression, features, tolerance):
    """Caches are trees of `AttentionCache`s with rows first, so they page by rows."""
    cache_layout(functions, model, features, slot_bytes=1 << 20)


_CHECKS = [
    ("properties", _properties),
    ("fit and embed", _fit_embeds),
    ("decoding apart", _decodes_apart),
    ("jit", _jits),
    ("padding", _padding),
    ("constant columns", _constant_columns),
    ("vmap", _vmap),
    ("masks", _mask),
    ("paging", _pages),
]


def group(
    rows: int = 3,
    steps: int = 32,
    horizon: int = 8,
    known: bool = True,
    seed: int = 0,
) -> SeriesGroup:
    """A random group with a missing step, and a known covariate if `known`.

    Args:
        rows: Series in the group, the last a known covariate if `known`.
        steps: Context steps.
        horizon: Steps of known future values.
        known: Whether the last row is a known covariate.
        seed: The random seed.

    Returns:
        The group.
    """
    generator = np.random.default_rng(seed)
    values = np.cumsum(generator.normal(size=(rows, steps)), axis=1) + 10
    observed = np.ones((rows, steps), dtype=bool)
    observed[0, steps // 2] = False
    roles = np.full(rows, TARGET, np.int32)
    future = future_observed = None
    if known:
        roles[-1] = KNOWN
        future = np.zeros((rows, horizon), np.float32)
        future[-1] = values[-1, -1] + np.cumsum(generator.normal(size=horizon))
        future_observed = np.zeros((rows, horizon), dtype=bool)
        future_observed[-1] = True
    return SeriesGroup(
        jnp.asarray(values, jnp.float32),
        jnp.asarray(observed),
        jnp.asarray(roles),
        None if future is None else jnp.asarray(future),
        None if future_observed is None else jnp.asarray(future_observed),
    )


def padded_group(
    data: SeriesGroup, rows: int, steps: int, seed: int = 0
) -> SeriesGroup:
    """A group padded with garbage rows and unobserved earlier steps.

    Args:
        data: The group.
        rows: Rows after padding, with large garbage and a NaN in the new ones.
        steps: Steps after padding, the new ones before the real ones.
        seed: The random seed of the garbage.

    Returns:
        The padded group, with `num_rows` set.
    """
    generator = np.random.default_rng(seed)
    real_rows, real_steps = data.values.shape

    def garbage(values: Array, fill: Any) -> Array:
        array = np.array(pad_rows(values, rows, fill=fill))
        if np.issubdtype(array.dtype, np.floating):
            array[real_rows:] = 1e3 * generator.normal(size=array[real_rows:].shape)
            array[real_rows, 0] = np.nan
        return jnp.asarray(array)

    before = np.full((rows, steps - real_steps), np.nan, np.float32)
    before[:real_rows] = 1e3 * generator.normal(size=(real_rows, steps - real_steps))
    return SeriesGroup(
        jnp.concatenate([jnp.asarray(before), garbage(data.values, 0.0)], axis=1),
        jnp.concatenate(
            [
                jnp.zeros((rows, steps - real_steps), bool),
                garbage(data.observed, False),
            ],
            axis=1,
        ),
        pad_rows(data.roles, rows, fill=TARGET),
        None if data.future is None else garbage(data.future, 0.0),
        None if data.future_observed is None else garbage(data.future_observed, False),
        real_rows,
    )


def check_series(
    functions: SeriesFunctions, parameters: Any, *, tolerance: float = 1e-5
) -> None:
    """Runs a forecaster through every way the engine calls it, on random groups.

    Checks that its properties describe its outputs and need only parameter
    shapes; that `forecast` returns a finite quantile per level for every row
    and step, for horizons of partial patches and past one pass; that padding
    rows past `num_rows` and unobserved steps before the real ones, garbage
    and NaN included, reach no real row; that groups batch with `jax.vmap`;
    that it jits; and, for a reusable model, that decoding its context cache
    forecasts as `forecast` does, read back padded past its real patches as
    the runtime reads it too.

    Args:
        functions: The model's contract calls.
        parameters: Its parameters.
        tolerance: How far results may move between equivalent calls.

    Raises:
        AssertionError: Naming the first check the model fails.
    """
    properties = functions.properties(parameters)
    _require(
        isinstance(properties, SeriesProperties),
        "properties must return SeriesProperties",
    )
    for name, run in _SERIES_CHECKS:
        try:
            run(functions, parameters, properties, tolerance)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None


def _series_properties(functions, model, properties, tolerance):
    """Properties describe the outputs, and follow from parameter shapes alone."""
    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), model)
    _require(
        functions.properties(abstract) == properties,
        "properties must follow from parameter shapes alone",
    )
    hash(properties)
    _require(
        properties.quantiles == tuple(sorted(set(properties.quantiles))),
        "quantile levels must increase",
    )
    _require(
        properties.reusable
        == (functions.encode is not None)
        == (functions.decode is not None)
        and (properties.reusable or functions.extend is None),
        "reusable must say whether encode and decode exist, and extend needs them",
    )
    for horizon in (properties.output_patch + 1, _long_horizon(properties)):
        data = group(steps=4 * properties.patch, horizon=horizon)
        predictions = functions.forecast(model, data, horizon)
        _require(
            predictions.shape == (3, horizon, len(properties.quantiles)),
            f"forecast returned predictions of shape {predictions.shape}, not "
            f"(3, {horizon}, {len(properties.quantiles)})",
        )
        _require(
            predictions.dtype == jnp.float32 and bool(jnp.isfinite(predictions).all()),
            "predictions must be finite float32",
        )


def _long_horizon(properties: SeriesProperties) -> int:
    """A horizon past one pass, when passes are bounded, else three output patches."""
    if properties.pass_limit is None:
        return 3 * properties.output_patch
    return properties.pass_limit + properties.output_patch


def paged_series(cache: Any, grows: list[bool], extra: int) -> Any:
    """A forecaster's cache as the runtime reads it back from its slots.

    Args:
        cache: The cache, its leaves in `jax.tree.leaves` order.
        grows: Which leaves have the patches on their first axis.
        extra: Zero patches to pad those leaves with.

    Returns:
        The padded cache, each attention cache's length its real patches.
    """
    leaves, treedef = jax.tree.flatten(cache)
    real = next(
        leaf.shape[0] for leaf, grown in zip(leaves, grows, strict=True) if grown
    )
    padded = [
        jnp.concatenate([leaf, jnp.zeros((extra, *leaf.shape[1:]), leaf.dtype)])
        if grown
        else leaf
        for leaf, grown in zip(leaves, grows, strict=True)
    ]
    return jax.tree.map(
        lambda layer: (
            layer._replace(length=jnp.int32(real))
            if isinstance(layer, AttentionCache)
            else layer
        ),
        jax.tree.unflatten(treedef, padded),
        is_leaf=lambda node: isinstance(node, AttentionCache),
    )


def _series_padding(functions, model, properties, tolerance):
    """Padding rows and unobserved earlier steps reach no real row."""
    horizon = _long_horizon(properties)
    data = group(steps=4 * properties.patch, horizon=horizon)
    pad = padded_group(data, rows=5, steps=6 * properties.patch)
    close(
        functions.forecast(model, pad, horizon)[:3],
        functions.forecast(model, data, horizon),
        tolerance,
    )


def _series_vmap(functions, model, properties, tolerance):
    """Groups of different sizes, padded to one, batch with `jax.vmap`."""
    horizon = _long_horizon(properties)
    groups = [
        group(rows=rows, steps=steps * properties.patch, horizon=horizon, seed=seed)
        for rows, steps, seed in ((2, 4, 0), (3, 3, 1))
    ]
    pads = [
        padded_group(data, rows=4, steps=4 * properties.patch, seed=seed)
        for seed, data in enumerate(groups)
    ]
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *pads)
    batched = jax.vmap(lambda data: functions.forecast(model, data, horizon))(stacked)
    for index, data in enumerate(groups):
        rows = data.values.shape[0]
        close(
            batched[index, :rows], functions.forecast(model, data, horizon), tolerance
        )


def _series_jits(functions, model, properties, tolerance):
    """`forecast` compiles with `jax.jit`."""
    data = group(steps=4 * properties.patch, horizon=properties.output_patch)
    horizon = properties.output_patch
    jitted = jax.jit(functions.forecast, static_argnums=2)
    close(
        jitted(model, data, horizon),
        functions.forecast(model, data, horizon),
        tolerance,
    )


def _series_reuse(functions, model, properties, tolerance):
    """A reusable model's cache decodes as `forecast` forecasts, past one pass too.

    A known row, which `encode` cannot refuse under a trace, must leave the
    cache forecasting NaN for the targets. With `extend`, a cache grown by a
    patch decodes as the longer group forecasts, when the new patch crosses
    zero after a context whose first patch is unobserved, so the bounds a
    cache keeps must grow with it.
    """
    if not properties.reusable:
        return
    horizon = _long_horizon(properties)
    data = group(steps=4 * properties.patch, horizon=horizon, known=False)
    cache = functions.encode(model, data)
    longer = functions.encode(
        model, group(steps=5 * properties.patch, horizon=horizon, known=False)
    )
    for leaf, grown in zip(
        jax.tree.leaves(cache), jax.tree.leaves(longer), strict=True
    ):
        _require(
            grown.shape == leaf.shape
            or (
                grown.shape[1:] == leaf.shape[1:]
                and grown.shape[0] == leaf.shape[0] + 1
            ),
            f"a cache leaf of shape {leaf.shape} became {grown.shape} with one more "
            "patch; leaves grow on their first axis or not at all",
        )
    close(
        functions.decode(model, cache, horizon),
        functions.forecast(model, data, horizon),
        tolerance,
    )
    # As the runtime reads a cache back from its slots: padded past its real
    # patches with zeros, its attention caches saying how many are real.
    grows = [
        leaf.ndim > 0 and grown.shape[0] == leaf.shape[0] + 1
        for leaf, grown in zip(
            jax.tree.leaves(cache), jax.tree.leaves(longer), strict=True
        )
    ]
    close(
        functions.decode(model, paged_series(cache, grows, 3), horizon),
        functions.decode(model, cache, horizon),
        tolerance,
    )
    # Under a trace, roles cannot be checked, so a known row must poison the
    # cache instead of slipping through as a target.
    known = group(steps=4 * properties.patch, horizon=horizon)
    stacked = jax.tree.map(lambda leaf: leaf[None], known)
    poisoned = jax.vmap(lambda data: functions.encode(model, data))(stacked)
    poisoned = jax.tree.map(lambda leaf: leaf[0], poisoned)
    targets = np.asarray(known.roles) == TARGET
    _require(
        bool(np.isnan(functions.decode(model, poisoned, horizon))[targets].all()),
        "a known row encoded under a trace must leave the target rows' forecasts NaN",
    )
    if functions.extend is None:
        return
    patch = properties.patch
    values = np.array(data.values)
    values[:, -patch:] = -np.abs(values[:, -patch:])
    observed = np.array(data.observed)
    observed[:, :patch] = False
    data = data._replace(values=jnp.asarray(values), observed=jnp.asarray(observed))
    shorter = SeriesGroup(
        data.values[:, :-patch], data.observed[:, :-patch], data.roles
    )
    added = SeriesGroup(data.values[:, -patch:], data.observed[:, -patch:], data.roles)
    grown = functions.extend(model, functions.encode(model, shorter), added)
    close(grown, functions.encode(model, data), tolerance)
    close(
        functions.decode(model, grown, horizon),
        functions.forecast(model, data, horizon),
        tolerance,
    )


_SERIES_CHECKS = [
    ("properties", _series_properties),
    ("padding", _series_padding),
    ("vmap", _series_vmap),
    ("jit", _series_jits),
    ("reuse", _series_reuse),
]
