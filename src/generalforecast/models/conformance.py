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
from generalforecast.models.outputs import quantile_index
from generalforecast.models.series import (
    KNOWN,
    TARGET,
    SeriesFunctions,
    SeriesGroup,
    SeriesProperties,
)
from generalforecast.nn.attention import AttentionCache
from generalforecast.runtime.device import load_cache, to_cache_dtype
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
    classes: int = 3,
) -> Table:
    """A random table with a missing value in the context and in the queries.

    Args:
        regression: Whether the targets are values rather than classes.
        rows: Context rows.
        queries: Query rows.
        features: Columns.
        seed: The random seed.
        classes: Classes of a classifier, which the context rows cycle through.

    Returns:
        The table.
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
        labels = generator.permutation(np.arange(rows) % classes)
        targets = jnp.asarray(labels, jnp.int32)
        num_classes = classes
    return Table(jnp.asarray(context), targets, jnp.asarray(query_rows), num_classes)


def predict(
    functions: TabularFunctions,
    model: Any,
    data: Table,
    num_rows: int | jax.Array | None = None,
    categorical: jax.Array | None = None,
) -> jax.Array:
    """The five contract calls in the order, and with the arguments, the engine makes them.

    Args:
        functions: The model's contract calls.
        model: Its parameters.
        data: The table.
        num_rows: Real context rows, the rest padding, if any.
        categorical: Which columns hold category codes, if any.

    Returns:
        The query rows' predictions.
    """
    state, context_rows = functions.fit(
        model,
        data.features,
        data.targets,
        data.num_classes,
        num_rows=num_rows,
        categorical=categorical,
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


def coded(data: Table, column: int) -> tuple[Table, jax.Array]:
    """The table with `column` holding category codes, and the mask that says so.

    The context holds codes 0 to 3 and a missing one; the queries a code the
    context lacks, as the engine's tables give unseen categories.
    """
    context, queries = np.array(data.features), np.array(data.queries)
    context[:, column] = np.arange(len(context)) % 4
    context[1, column] = np.nan
    queries[:, column] = np.arange(len(queries)) % 5
    mask = jnp.zeros(context.shape[1], bool).at[column].set(True)
    changed = data._replace(features=jnp.asarray(context), queries=jnp.asarray(queries))
    return changed, mask


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


def _forecasts_close(actual: Any, expected: Any, tolerance: float) -> None:
    """Raises unless two forecasts match within `tolerance` at each series' scale.

    A forecast near zero in a series that reaches tens carries float32's last
    places at the series' scale, which a tolerance on each value alone misses,
    so each row's allowance grows with its largest finite forecast, from 1.
    """
    expected = np.asarray(expected, dtype=np.float32)
    size = np.where(np.isfinite(expected), np.abs(expected), 0)
    scale = size.reshape(len(expected), -1).max(axis=1, initial=1.0)
    scale = scale.reshape((-1,) + (1,) * (expected.ndim - 1))
    # |a - e| <= tolerance * (scale + |e|), both sides over the row's scale.
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32) / scale,
        expected / scale,
        rtol=tolerance,
        atol=tolerance,
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
    stored_tolerance: float = 1e-3,
) -> None:
    """Runs a model through every way the engine calls it, on random tables.

    Checks that its calls take the arguments the engine passes, a categorical
    mask among them; that its properties describe its outputs and need only
    parameter shapes; that `fit` returns the context rows `embed` would; that
    rows embed and queries decode apart as together; that it jits; that
    padding past `num_rows`, garbage and NaN included, reaches no real row;
    that columns constant or empty in the context are left out; that tables
    batch with `jax.vmap`; that `decode` reads only the keys a cache's mask
    allows within its length; that it serves class counts past three, up
    to its slots; and that its caches page by rows. What its properties and
    functions claim is checked too: that it reads stored caches, scales and
    all; that it takes context rows without a target; that `decode_reading`
    decodes as `decode`; and that `crowded_categories` counts columns.

    Args:
        functions: The model's contract calls.
        parameters: Its parameters.
        features: Columns of the random tables.
        tolerance: How far results may move between equivalent calls.
        stored_tolerance: How far a model that reads stored caches may move
            its predictions, probabilities or medians per target deviation,
            between a float16, bfloat16 or int8 cache and the same cache
            loaded back to float32.

    Raises:
        AssertionError: Naming the first check the model fails, whether its
            results differ or one of its calls raises.
    """
    try:
        properties = functions.properties(parameters)
    except Exception as error:
        raise AssertionError(f"properties: {_reason(error)}") from error
    _require(
        isinstance(properties, TabularProperties),
        "properties: properties must return TabularProperties",
    )
    regression = properties.regression
    limits = Limits(features, tolerance, stored_tolerance)
    for name, run in _CHECKS:
        try:
            run(functions, parameters, properties, regression, limits)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None
        except Exception as error:
            raise AssertionError(f"{name}: {_reason(error)}") from error


class Limits(NamedTuple):
    """The table width and tolerances a check runs with."""

    features: int
    tolerance: float
    stored_tolerance: float


def _reason(error: Exception) -> str:
    """A call's error, as a failed check reports it."""
    return f"a call raised {type(error).__name__}: {error}"


def _fit(
    functions: TabularFunctions,
    model: Any,
    data: Table,
    num_rows: int | jax.Array | None = None,
    categorical: jax.Array | None = None,
) -> tuple[Any, jax.Array]:
    """`fit`, with every argument the engine passes it."""
    return functions.fit(
        model,
        data.features,
        data.targets,
        data.num_classes,
        num_rows=num_rows,
        categorical=categorical,
    )


def _calls(functions, model, properties, regression, limits):
    """The calls take the arguments the engine passes, a categorical mask among them.

    The engine passes `fit` its row count and categorical mask, None or not,
    and `encode` its row count, by keyword; padding stays invisible with
    category codes in the table.
    """
    data, mask = coded(table(regression, features=limits.features), 0)
    for categorical in (None, mask):
        predictions = predict(functions, model, data, categorical=categorical)
        _require(bool(np.isfinite(predictions).all()), "predictions are not finite")
    small = coded(table(regression, rows=20, queries=6, features=limits.features), 0)[0]
    close(
        predict(functions, model, padded(small, 32, 8), 20, mask)[:6],
        predict(functions, model, small, categorical=mask),
        limits.tolerance,
    )


def _properties(functions, model, properties, regression, limits):
    """Properties describe the outputs, and follow from parameter shapes alone."""
    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), model)
    _require(
        functions.properties(abstract) == properties,
        "properties must follow from parameter shapes alone",
    )
    hash(properties)
    data = table(regression, features=limits.features)
    state, context_rows = _fit(functions, model, data)
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
    _outputs(properties, data, predictions, limits.tolerance)


def _outputs(
    properties: TabularProperties, data: Table, predictions: jax.Array, tolerance: float
) -> None:
    """Predictions have a column per class or quantile, and classes' sum to 1."""
    columns = properties.num_outputs if properties.regression else data.num_classes
    _require(
        predictions.shape == (data.queries.shape[0], columns),
        f"postprocess returned predictions of shape {predictions.shape}, "
        f"not ({data.queries.shape[0]}, {columns})",
    )
    _require(bool(np.isfinite(predictions).all()), "predictions are not finite")
    if not properties.regression:
        close(predictions.sum(axis=-1), jnp.ones(data.queries.shape[0]), tolerance)


def _fit_embeds(functions, model, properties, regression, limits):
    """`fit` returns the context rows `embed` would, and rows embed apart as together."""
    data = table(regression, features=limits.features)
    state, context_rows = _fit(functions, model, data)
    close(context_rows, functions.embed(model, state, data.features, data.targets))
    apart = jnp.concatenate(
        [
            functions.embed(model, state, data.queries[:3]),
            functions.embed(model, state, data.queries[3:]),
        ]
    )
    close(apart, functions.embed(model, state, data.queries), limits.tolerance)


def _decodes_apart(functions, model, properties, regression, limits):
    """Queries decode apart as together."""
    data = table(regression, features=limits.features)
    state, context_rows = _fit(functions, model, data)
    cache = functions.encode(model, state, context_rows, data.targets, num_rows=None)
    query_rows = functions.embed(model, state, data.queries)
    together = functions.decode(model, state, cache, query_rows)
    apart = jnp.concatenate(
        [
            functions.decode(model, state, cache, query_rows[:3]),
            functions.decode(model, state, cache, query_rows[3:]),
        ]
    )
    close(apart, together, limits.tolerance)


def _jits(functions, model, properties, regression, limits):
    """The contract calls compile with `jax.jit`."""
    data = table(regression, features=limits.features)
    jitted = TabularFunctions(
        fit=jax.jit(functions.fit, static_argnames="num_classes"),
        embed=jax.jit(functions.embed),
        encode=jax.jit(functions.encode),
        decode=jax.jit(functions.decode),
        postprocess=jax.jit(functions.postprocess),
        properties=functions.properties,
    )
    close(
        predict(jitted, model, data), predict(functions, model, data), limits.tolerance
    )


def _padding(functions, model, properties, regression, limits):
    """Padding past `num_rows`, garbage and NaN included, reaches no real row."""
    tolerance = limits.tolerance
    data = table(regression, rows=20, queries=6, features=limits.features)
    pad = padded(data, context_size=32, query_size=8)
    state, context_rows = _fit(functions, model, data)
    padded_state, padded_rows = _fit(functions, model, pad, num_rows=20)
    close(padded_state, state, tolerance)
    close(padded_rows[:20], context_rows, tolerance)
    close(
        predict(functions, model, pad, num_rows=20)[:6],
        predict(functions, model, data),
        tolerance,
    )


def _constant_columns(functions, model, properties, regression, limits):
    """Columns constant or empty in the context are left out of predictions.

    Upstream preprocessing drops them, so what they hold, in the context or
    in the queries, changes nothing. The reference tests check that the rest
    predicts as upstream does without them.
    """
    data = table(regression, features=limits.features)
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
    close(predictions, predict(functions, model, changed), limits.tolerance)


def _vmap(functions, model, properties, regression, limits):
    """Tables of different sizes, padded to one, batch with `jax.vmap`."""
    tables = [
        table(regression, rows=rows, queries=6, features=limits.features, seed=seed)
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
        close(batched[index, :6], predict(functions, model, data), limits.tolerance)


def _mask(functions, model, properties, regression, limits):
    """`decode` reads only the keys a cache's mask allows, within its length.

    The engine joins units' contexts under a mask of the keys each query
    reads, and gives a unit that reads its whole context a mask over the
    call's padded rows, which only its length bounds.
    """
    tolerance = limits.tolerance
    data = table(regression, rows=24, queries=4, features=limits.features)
    state, rows = _fit(functions, model, data)
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

    joined = _caches(join, first, second)
    outputs = functions.decode(model, state, joined, queries)
    close(outputs[:2], functions.decode(model, state, first, queries[:2]), tolerance)
    close(outputs[2:], functions.decode(model, state, second, queries[2:]), tolerance)
    # Padding like the real rows but labeled otherwise, which would move
    # predictions if a query read it.
    copies = data.targets[:8]
    relabeled = copies + 10 if regression else (copies + 1) % data.num_classes
    pad = data._replace(
        features=jnp.concatenate([data.features, data.features[:8][::-1]]),
        targets=jnp.concatenate([data.targets, relabeled[::-1]]),
    )
    state, rows = _fit(functions, model, pad, num_rows=24)
    cache = functions.encode(model, state, rows, pad.targets, num_rows=24)
    queries = functions.embed(model, state, pad.queries)
    every = jnp.ones((queries.shape[0], rows.shape[0]), bool)
    close(
        functions.decode(
            model, state, _caches(lambda c: c._replace(mask=every), cache), queries
        ),
        functions.decode(model, state, cache, queries),
        tolerance,
    )


def _caches(change: Any, *caches: Any) -> Any:
    """Each attention cache of a context cache, changed."""
    return jax.tree.map(
        change, *caches, is_leaf=lambda node: isinstance(node, AttentionCache)
    )


def _class_counts(functions, model, properties, regression, limits):
    """A classifier serves class counts past three, its slots among them up to 32."""
    if regression:
        return
    slots = properties.class_slots
    for classes in sorted({2, min(slots, 7), min(slots, 32)}):
        data = table(False, features=limits.features, classes=classes)
        _outputs(properties, data, predict(functions, model, data), limits.tolerance)


def _stored(functions, model, properties, regression, limits):
    """A model that reads stored caches decodes one as it decodes it loaded back.

    The runtime stores caches in float16, bfloat16 or int8 with a scale per
    row and head, and hands a model that reads them so the stored cache; one
    loaded back to float32, its scales applied, leaves only the rounding of
    the model's own arithmetic between the two, whatever the storage's.
    """
    if not properties.stored_cache:
        return
    data = table(regression, rows=40, queries=12, features=limits.features)
    state, rows = _fit(functions, model, data)
    cache = functions.encode(model, state, rows, data.targets)
    queries = functions.embed(model, state, data.queries)

    def predictions(cache: Any) -> jax.Array:
        outputs = functions.decode(model, state, cache, queries)
        return functions.postprocess(model, state, outputs)

    for dtype in ("float16", "bfloat16", "int8"):
        stored = to_cache_dtype(cache, jnp.dtype(dtype))
        loaded = predictions(load_cache(stored))
        moved = jnp.abs(predictions(stored) - loaded)
        if regression:
            middle = quantile_index(0.5)
            drift = float(jnp.max(moved[:, middle]) / jnp.std(data.targets))
        else:
            drift = float(jnp.max(moved))
        _require(
            drift <= limits.stored_tolerance,
            f"a {dtype} cache decodes {drift:.3g} from the same cache loaded "
            f"back, past {limits.stored_tolerance}",
        )


def _unlabeled(functions, model, properties, regression, limits):
    """A model that takes context rows without a target does, padding still invisible."""
    if not properties.unlabeled_context:
        return
    data = table(regression, rows=20, queries=6, features=limits.features)
    unknown = jnp.nan if regression else -1
    data = data._replace(targets=data.targets.at[jnp.asarray([2, 9])].set(unknown))
    predictions = predict(functions, model, data)
    _outputs(properties, data, predictions, limits.tolerance)
    close(
        predict(functions, model, padded(data, 32, 8), num_rows=20)[:6],
        predictions,
        limits.tolerance,
    )


def _reading(functions, model, properties, regression, limits):
    """`decode_reading` decodes as `decode`, whatever reads the cache's parts."""
    if functions.decode_reading is None:
        return
    data = table(regression, features=limits.features)
    state, rows = _fit(functions, model, data)
    cache = functions.encode(model, state, rows, data.targets)
    queries = functions.embed(model, state, data.queries)

    def read(after: Any, part: Any) -> Any:
        return part(cache)

    close(
        functions.decode_reading(model, state, read, queries),
        functions.decode(model, state, cache, queries),
        limits.tolerance,
    )


def _crowded(functions, model, properties, regression, limits):
    """`crowded_categories` counts columns, of one state or of states stacked."""
    count = functions.crowded_categories
    if count is None:
        return
    data, mask = coded(table(regression, features=limits.features), 0)
    for categorical in (None, mask):
        state, _ = _fit(functions, model, data, categorical=categorical)
        stacked = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), state)
        for fitted in (state, stacked):
            crowded = count(fitted)
            _require(
                type(crowded) is int and 0 <= crowded <= limits.features,
                f"crowded_categories returned {crowded!r}, not a column count",
            )


def _pages(functions, model, properties, regression, limits):
    """Caches are trees of `AttentionCache`s with rows first, so they page by rows."""
    cache_layout(functions, model, limits.features, slot_bytes=1 << 20)


_CHECKS = [
    ("calls", _calls),
    ("properties", _properties),
    ("fit and embed", _fit_embeds),
    ("decoding apart", _decodes_apart),
    ("jit", _jits),
    ("padding", _padding),
    ("constant columns", _constant_columns),
    ("vmap", _vmap),
    ("masks", _mask),
    ("class counts", _class_counts),
    ("stored caches", _stored),
    ("unlabeled context", _unlabeled),
    ("decode reading", _reading),
    ("crowded categories", _crowded),
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
        AssertionError: Naming the first check the model fails, whether its
            results differ or one of its calls raises.
    """
    try:
        properties = functions.properties(parameters)
    except Exception as error:
        raise AssertionError(f"properties: {_reason(error)}") from error
    _require(
        isinstance(properties, SeriesProperties),
        "properties: properties must return SeriesProperties",
    )
    for name, run in _SERIES_CHECKS:
        try:
            run(functions, parameters, properties, tolerance)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None
        except Exception as error:
            raise AssertionError(f"{name}: {_reason(error)}") from error


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
        and (properties.reusable or functions.extend is None)
        and (functions.decode is not None or functions.decode_within is None),
        "reusable must say whether encode and decode exist, and extend and "
        "decode_within need them",
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
    patches = [
        leaf.shape[0] for leaf, grown in zip(leaves, grows, strict=True) if grown
    ]
    _require(bool(patches), "no cache leaf grows with the context's patches")
    real = patches[0]
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
    _forecasts_close(
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
        _forecasts_close(
            batched[index, :rows], functions.forecast(model, data, horizon), tolerance
        )


def _series_jits(functions, model, properties, tolerance):
    """`forecast` compiles with `jax.jit`."""
    data = group(steps=4 * properties.patch, horizon=properties.output_patch)
    horizon = properties.output_patch
    jitted = jax.jit(functions.forecast, static_argnums=2)
    _forecasts_close(
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
    _forecasts_close(
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
    _forecasts_close(
        functions.decode(model, paged_series(cache, grows, 3), horizon),
        functions.decode(model, cache, horizon),
        tolerance,
    )
    if (
        functions.decode_within is not None
        and 4 * properties.patch + horizon <= properties.context_limit
    ):
        # As the runtime decodes caches stored at the limit's bucket whose
        # real steps and horizon stay within the limit, under `jax.vmap`.
        extra = properties.context_limit // properties.patch - 4
        paged = jax.tree.map(lambda leaf: leaf[None], paged_series(cache, grows, extra))
        within, general = (
            jax.jit(jax.vmap(lambda c, decode=decode: decode(model, c, horizon)))(paged)
            for decode in (functions.decode_within, functions.decode)
        )
        _require(
            np.array_equal(within, general),
            "decode_within must forecast as decode does, bit for bit",
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
    _forecasts_close(
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
