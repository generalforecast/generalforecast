# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Protocol

from jax import Array


class Averaging(NamedTuple):
    """How a model's ensembles combine estimators, as its own package's do.

    Attributes:
        over: `outputs` averages the decoder outputs and postprocesses the mean;
            `predictions` postprocesses each estimator and averages the
            predictions; `buckets` mixes each estimator's distribution over
            regression buckets, the softmax of its outputs over `temperature`,
            and postprocesses the mixture.
        trim: Fraction of estimators left out at each end of each sorted
            prediction before averaging, with `predictions`.
        temperature: What `postprocess` divides bucket logits by, with `buckets`.
    """

    over: Literal["outputs", "predictions", "buckets"] = "outputs"
    trim: float = 0.0
    temperature: float = 1.0


@dataclass(frozen=True)
class TabularProperties:
    """What the engine and model wrappers need to know about a model before running it.

    Every field is read from parameter shapes and types, so it holds for abstract
    parameters too. The values are hashable, which lets them be static under jit.

    Attributes:
        regression: Whether the model predicts quantiles instead of classes.
        class_slots: Most classes the model can serve, or 0 for regression.
        num_outputs: Columns of each prediction `postprocess` returns: the class
            slots, whose first columns a table's classes take, or the quantiles.
        averaging: How an ensemble of the model combines its estimators.
        context_rows: The most context rows the model was trained on, if
            known. Larger contexts still run, but past what it learned from;
            a table warns or cuts them to this length.
        context_features: The most features it was trained on, if known.
        context_classes: The most classes it was trained on, if known.
        stored_cache: Whether `decode` reads caches as the runtime stores them:
            cast to a reduced precision, or int8 with a float32 scale per row
            and head in each
            `AttentionCache`'s `key_scale` and `value_scale`, as
            `generalforecast.nn.attention` reads them. Otherwise the runtime hands them
            caches back in float32, converting each before it is read.
        column_summary: Whether the state holds a summary pooled over the
            context's rows, such as inducing points per column. A table may
            then summarize each context: the engine fits the unit's own state,
            preprocessing and column summary, on its rows alone with `fit`,
            and embeds its queries against it.
        unlabeled_context: Whether `fit`, `embed` and `encode` take context
            rows whose target is not known yet, NaN or for a class -1, through
            a learned mask; they leave such rows out of target statistics.
            LimiX and Mitra were trained with the mask on query rows; their
            papers keep every context row labeled, though LimiX's regressor
            masks a non-finite target in any row. A masked context row asks
            of them what they were not trained on. Otherwise such rows stay
            out of fits and contexts until they are labeled.
    """

    regression: bool
    class_slots: int
    num_outputs: int
    averaging: Averaging = field(default_factory=Averaging)
    stored_cache: bool = False
    unlabeled_context: bool = False
    context_rows: int | None = None
    context_features: int | None = None
    context_classes: int | None = None
    column_summary: bool = False


class Fit(Protocol):
    """Computes the table state and context row representations from the whole table.

    A fit on more rows than a call embeds keeps only the state and embeds
    pages with `embed` as they are read, so the state must not depend on the
    rows `fit` returns: computed from them, XLA cannot drop them.
    """

    def __call__(
        self,
        model: Any,
        features: Array,
        targets: Array,
        num_classes: int,
        /,
        *,
        num_rows: Array | int | None = None,
        categorical: Array | None = None,
    ) -> tuple[Any, Array]:
        """Fits a table: real rows first, `num_rows` of them, the rest padding."""
        ...


class Embed(Protocol):
    """Computes row representations from rows and the state."""

    def __call__(
        self,
        model: Any,
        state: Any,
        features: Array,
        targets: Array | None = None,
        /,
    ) -> Array:
        """Embeds rows, with targets for context rows and none for queries."""
        ...


class Encode(Protocol):
    """Computes the context cache queries read."""

    def __call__(
        self,
        model: Any,
        state: Any,
        rows: Array,
        targets: Array,
        /,
        *,
        num_rows: Array | int | None = None,
    ) -> Any:
        """Encodes context rows."""
        ...


class Decode(Protocol):
    """Computes raw outputs from query rows and a context cache.

    Every attention over the cache must honor each `AttentionCache`'s `length`
    and, when set, its `mask`: the engine decodes queries that read different
    parts of one context together, and the mask says which keys each reads.
    """

    def __call__(self, model: Any, state: Any, cache: Any, rows: Array, /) -> Array:
        """Decodes query rows, reading only the keys the cache's mask allows."""
        ...


class ReadCache(Protocol):
    """Reads a part of the context cache for `DecodeReading`."""

    def __call__(self, after: Any, part: Callable[[Any], Any], /) -> Any:
        """`part` of the cache, read once `after` is computed."""
        ...


class DecodeReading(Protocol):
    """Computes raw outputs as `Decode` does, reading the cache a part at a time.

    A decode that reads each layer's keys as that layer needs them holds only
    one layer's beside the slots it reads from.
    """

    def __call__(
        self, model: Any, state: Any, read: ReadCache, rows: Array, /
    ) -> Array:
        """Decodes query rows, reading the cache's parts through `read`."""
        ...


class Postprocess(Protocol):
    """Computes predictions from raw outputs and the state."""

    def __call__(self, model: Any, state: Any, outputs: Array, /) -> Array:
        """Turns outputs into probabilities or quantiles."""
        ...


class Properties(Protocol):
    """Describes a model from its parameters, once at load."""

    def __call__(self, model: Any, /) -> TabularProperties:
        """The model's properties."""
        ...


class CrowdedCategories(Protocol):
    """Counts categorical columns with more categories than the state keeps exactly."""

    def __call__(self, state: Any, /) -> int:
        """How many columns' ranks are interpolated, for a fitted state."""
        ...


class TabularFunctions(NamedTuple):
    """The calls the engine makes on a model, each with its own cache boundary.

    Every call works on one table with no batch axis; the engine batches with
    `jax.vmap`. Targets are class ids `0 .. C-1` or regression targets on their
    original scale; for a model whose properties set `unlabeled_context`, a
    context row's target may also be unknown: -1 for a class, NaN for a value.

    A context cache is a tree of `AttentionCache`s: their keys, values and any
    scales have the context rows on their first axis and the heads on their
    second, and their lengths are scalars. The engine pages caches by rows and
    ensembles fold estimators into heads, so `cache_layout` refuses others.
    When the engine decodes queries that read different parts of one context
    together, it sets each cache's `mask` to the keys each query reads, and
    `decode` must honor it; `generalforecast.nn.attention.attend` does.

    Attributes:
        fit: Computes the table state and context row reps from the whole context
            table; with `categorical`, a mask of the columns that hold category
            codes, for models that treat them apart.
        embed: Computes a row representation from a row and the state.
        encode: Computes the context cache (used by queries) from the set of context
            rows and the state.
        decode: Computes raw outputs using the queries and the cache.
        postprocess: Computes predictions using the outputs and the state.
        properties: Describes the model from its parameters, once at load.
        crowded_categories: For a model whose state keeps a bounded number of
            each categorical column's categories, counts the columns past it,
            so the engine can say their predictions are approximate; None for
            other models.
        decode_reading: `decode`, reading the cache a part at a time, for a
            model that reads its layers' keys in turn; None for other models.
    """

    fit: Fit
    embed: Embed
    encode: Encode
    decode: Decode
    postprocess: Postprocess
    properties: Properties
    crowded_categories: CrowdedCategories | None = None
    decode_reading: DecodeReading | None = None
