// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `generalforecast._core`: thin Python bindings over `gf-core`.
//!
//! Bindings convert arguments, arrays and errors and nothing else, so behavior
//! lives in `gf-core`, where it is tested without Python. Calls that may take
//! a while, such as planning a step or inserting many rows, release the GIL,
//! so other Python threads run meanwhile.

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::time::Duration;

use gf_core::cache::{Eviction, Kind, ModelId, TierConfig};
use gf_core::catalog::{self, Surface};
use gf_core::categories::CategoryEncoding;
use gf_core::forecasting::{self, SeriesSpec};
use gf_core::node::{
    self, CutoffRefit, DEFAULT_FIT_ROWS, EntrySizes, ErrorKind, NodeError, PastCap, RefitPolicy,
    RequestId, StepRows, SummaryScope, TableId, TableSpec, Targets, Task,
};
use gf_core::planner;
use gf_core::predictor;
use gf_core::regions::{Rebuild, RebuildWait, RegionSpec};
use gf_core::request::{self, Inference, Label, Limits, Num, Options};
use gf_core::table::RowId;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod engine;

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// A node error as the Python exception its kind calls for: a missing table or
/// key is a `KeyError`, a bad argument a `ValueError`, and an invariant the
/// node broke a `RuntimeError`.
fn raised(error: NodeError) -> PyErr {
    let message = error.to_string();
    match error.kind() {
        ErrorKind::NotFound => PyKeyError::new_err(message),
        ErrorKind::Invalid => PyValueError::new_err(message),
        ErrorKind::Internal => PyRuntimeError::new_err(message),
    }
}

fn positive(value: i64, name: &str) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(count(value, name)?)
        .ok_or_else(|| invalid(format!("{name} must be positive")))
}

/// A count or size from Python: a negative one is a `ValueError`, as a zero
/// one is where it must be positive, rather than an `OverflowError`.
fn count<T: TryFrom<i64>>(value: i64, name: &str) -> PyResult<T> {
    T::try_from(value).map_err(|_| {
        invalid(if value < 0 {
            format!("{name} must not be negative")
        } else {
            format!("{name} is too large")
        })
    })
}

/// The padded size for `rows` rows: `minimum` doubled until it fits.
#[pyfunction]
#[pyo3(signature = (rows, minimum = 8))]
fn bucket_size(rows: i64, minimum: i64) -> PyResult<usize> {
    Ok(gf_core::shapes::bucket_size(
        count(rows, "rows")?,
        positive(minimum, "minimum")?,
    ))
}

/// How each query's context is chosen.
///
/// Each constructor raises `ValueError` for a count that is negative, zero,
/// or more rows than a table can hold.
#[pyclass(frozen)]
struct Strategy(planner::Strategy);

impl Strategy {
    /// A strategy whose counts are all positive.
    fn checked(strategy: planner::Strategy) -> PyResult<Self> {
        strategy.check().map_err(invalid)?;
        Ok(Self(strategy))
    }
}

#[pymethods]
impl Strategy {
    /// Every live row, for every query.
    #[staticmethod]
    fn exact() -> Self {
        Self(planner::Strategy::Exact)
    }

    /// What fits the table's size, resolved from its live rows at each plan.
    ///
    /// Exact up to the model's context cap, and past it a stratified core of
    /// a quarter of the cap plus the nearest rows that fill the rest, the
    /// neighbors leaning to the newest rows when the table has times.
    #[staticmethod]
    fn adaptive() -> Self {
        Self(planner::Strategy::Adaptive)
    }

    /// The strategy a table of `rows` live rows plans with.
    ///
    /// `cap` is the model's context rows, or None for no cap; `timed` whether
    /// the table's rows have times. An adaptive strategy is resolved by the
    /// server's rule, any other is itself.
    #[pyo3(signature = (rows, cap, timed = false))]
    fn resolve(&self, rows: i64, cap: Option<i64>, timed: bool) -> PyResult<Self> {
        let cap = cap
            .map(|cap| count::<u64>(cap, "cap"))
            .transpose()?
            .and_then(std::num::NonZeroU64::new);
        Ok(Self(self.0.resolve(count(rows, "rows")?, cap, timed)))
    }

    /// The same deterministic sample of rows for every query: a core alone.
    ///
    /// `kind` is `random`, `stratified` (by class, or by target bin for a
    /// regression), `balanced` (an equal share per class or bin) or `diverse`
    /// (farthest-point sampling).
    #[staticmethod]
    #[pyo3(signature = (rows, seed = 0, kind = "random"))]
    fn subsample(rows: i64, seed: i64, kind: &str) -> PyResult<Self> {
        let kind = kind.parse::<planner::CoreKind>().map_err(invalid)?;
        Self::checked(planner::Strategy::subsample(
            count(rows, "rows")?,
            count(seed, "seed")?,
            kind,
        ))
    }

    /// Each query's nearest rows, without a core.
    #[staticmethod]
    fn neighbors(rows: i64) -> PyResult<Self> {
        Self::retrieved(rows, 0, 0, "random", None, None, None)
    }

    /// A shared core plus each query's nearest rows outside it, encoded together.
    ///
    /// The core is drawn as `subsample` draws one; a `core` of 0 is no core.
    /// With `among`, the neighbors are the latest of that many nearest rows.
    /// With `last` or `span`, the whole context comes from the window's
    /// latest rows, or its rows at most `span` before its newest row's time.
    #[staticmethod]
    #[pyo3(signature = (
        neighbors, core = 0, seed = 0, kind = "random", among = None, last = None, span = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn retrieved(
        neighbors: i64,
        core: i64,
        seed: i64,
        kind: &str,
        among: Option<i64>,
        last: Option<i64>,
        span: Option<i64>,
    ) -> PyResult<Self> {
        let core = match count::<u64>(core, "core")? {
            0 => None,
            rows => Some(planner::Core {
                rows,
                seed: count(seed, "seed")?,
                kind: kind.parse::<planner::CoreKind>().map_err(invalid)?,
            }),
        };
        let recency = match (last, span) {
            (Some(_), Some(_)) => return Err(invalid("give last or span, not both")),
            (Some(rows), None) => Some(planner::Recency::Last(count(rows, "last")?)),
            (None, Some(span)) => Some(planner::Recency::Span(count(span, "span")?)),
            (None, None) => None,
        };
        Self::checked(planner::Strategy::Retrieved {
            core,
            neighbors: count(neighbors, "neighbors")?,
            among: among.map(|among| count(among, "among")).transpose()?,
            recency,
        })
    }

    /// The call that makes the strategy, such as `Strategy.neighbors(512)`.
    fn __repr__(&self) -> String {
        let planner::Strategy::Retrieved {
            core,
            neighbors,
            among,
            recency,
        } = self.0
        else {
            return match self.0 {
                planner::Strategy::Adaptive => "Strategy.adaptive()",
                _ => "Strategy.exact()",
            }
            .to_string();
        };
        let mut more = String::new();
        if let Some(among) = among {
            more += &format!(", among={among}");
        }
        match recency {
            Some(planner::Recency::Last(rows)) => more += &format!(", last={rows}"),
            Some(planner::Recency::Span(span)) => more += &format!(", span={span}"),
            None => {}
        }
        let Some(core) = core else {
            return if more.is_empty() {
                format!("Strategy.neighbors({neighbors})")
            } else {
                format!("Strategy.retrieved({neighbors}{more})")
            };
        };
        let kind = match core.kind {
            planner::CoreKind::Random => String::new(),
            kind => format!(", kind={:?}", kind.name()),
        };
        if neighbors == 0 && more.is_empty() {
            format!(
                "Strategy.subsample({}, seed={}{kind})",
                core.rows, core.seed
            )
        } else {
            format!(
                "Strategy.retrieved({neighbors}, core={}, seed={}{kind}{more})",
                core.rows, core.seed
            )
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.spec().hash(&mut hasher);
        hasher.finish()
    }

    /// The strategy as the JSON a server's `create_table` takes.
    fn spec(&self) -> String {
        let planner::Strategy::Retrieved {
            core,
            neighbors,
            among,
            recency,
        } = self.0
        else {
            return match self.0 {
                planner::Strategy::Adaptive => r#"{"kind": "adaptive"}"#,
                _ => r#"{"kind": "exact"}"#,
            }
            .to_string();
        };
        let among = among.map_or_else(|| "null".to_string(), |among| among.to_string());
        let recency = match recency {
            Some(planner::Recency::Last(rows)) => format!(r#"{{"last": {rows}}}"#),
            Some(planner::Recency::Span(span)) => format!(r#"{{"span": {span}}}"#),
            None => "null".to_string(),
        };
        let core = core.map_or_else(
            || "null".to_string(),
            |core| {
                format!(
                    r#"{{"rows": {}, "seed": {}, "kind": "{}"}}"#,
                    core.rows,
                    core.seed,
                    core.kind.name()
                )
            },
        );
        format!(
            r#"{{"kind": "retrieved", "core": {core}, "neighbors": {neighbors}, "among": {among}, "recency": {recency}}}"#
        )
    }

    /// Strategies are immutable, so a copy is the same object.
    fn __copy__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Strategies are immutable, so a deep copy is the same object.
    fn __deepcopy__(slf: Py<Self>, _memo: &Bound<'_, PyAny>) -> Py<Self> {
        slf
    }
}

/// Which other series join a series in its group.
#[pyclass(frozen)]
struct GroupStrategy(forecasting::GroupStrategy);

#[pymethods]
impl GroupStrategy {
    /// Each series alone, with its own covariate columns.
    #[staticmethod]
    fn alone() -> Self {
        Self(forecasting::GroupStrategy::Alone)
    }

    /// `size - 1` others by seeded hash with the series; a table holding fewer refuses the request.
    #[staticmethod]
    fn random(size: i64, seed: i64) -> PyResult<Self> {
        Ok(Self(forecasting::GroupStrategy::Random {
            size: positive(size, "size")?,
            seed: seed as u64,
        }))
    }

    /// The members each forecast names with `forecast_given`, up to `size` series in all; the table draws none.
    #[staticmethod]
    fn given(size: i64) -> PyResult<Self> {
        Ok(Self(forecasting::GroupStrategy::Given {
            size: positive(size, "size")?,
        }))
    }

    /// The most series in a group, the forecast one included: 1 alone.
    fn size(&self) -> usize {
        self.0.size()
    }

    /// The strategy as the JSON a server's `create_series_table` takes.
    fn spec(&self) -> String {
        match self.0 {
            forecasting::GroupStrategy::Alone => r#"{"kind":"alone"}"#.to_string(),
            forecasting::GroupStrategy::Random { size, seed } => {
                format!(r#"{{"kind":"random","size":{size},"seed":{seed}}}"#)
            }
            forecasting::GroupStrategy::Given { size } => {
                format!(r#"{{"kind":"given","size":{size}}}"#)
            }
        }
    }

    fn __repr__(&self) -> String {
        match self.0 {
            forecasting::GroupStrategy::Alone => "GroupStrategy.alone()".to_string(),
            forecasting::GroupStrategy::Random { size, seed } => {
                format!("GroupStrategy.random({size}, {seed})")
            }
            forecasting::GroupStrategy::Given { size } => format!("GroupStrategy.given({size})"),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

/// What a cache entry is keyed on.
///
/// Its model, table, epoch and kind, and the fingerprint of its rows.
#[pyclass(frozen, eq, hash, get_all, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    model: u32,
    table: u32,
    epoch: u64,
    /// `rows` for a page's row representations, `context` for a unit's context.
    kind: &'static str,
    rows: u128,
}

#[pymethods]
impl CacheKey {
    /// A key, as a worker reads one from a server's frame.
    #[new]
    fn py_new(model: u32, table: u32, epoch: u64, kind: &str, rows: u128) -> PyResult<Self> {
        let kind = [Kind::Rows, Kind::Context]
            .into_iter()
            .map(Kind::name)
            .find(|name| *name == kind)
            .ok_or_else(|| {
                invalid(format!(
                    "a cache key's kind is rows or context, not {kind:?}"
                ))
            })?;
        Ok(Self {
            model,
            table,
            epoch,
            kind,
            rows,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheKey(model={}, table={}, epoch={}, kind={:?}, rows={})",
            self.model, self.table, self.epoch, self.kind, self.rows
        )
    }
}

impl From<&gf_core::cache::CacheKey> for CacheKey {
    fn from(key: &gf_core::cache::CacheKey) -> Self {
        Self {
            model: key.model.0,
            table: key.table.0,
            epoch: key.epoch.0,
            kind: key.kind.name(),
            rows: key.rows.bits(),
        }
    }
}

fn slot_numbers(slots: &[gf_core::cache::SlotId]) -> Vec<u32> {
    slots.iter().map(|slot| slot.0).collect()
}

fn row_array(py: Python<'_>, rows: &[RowId]) -> Py<PyArray1<u32>> {
    rows.iter()
        .map(|row| row.0)
        .collect::<Vec<_>>()
        .into_pyarray(py)
        .unbind()
}

/// Each row's time, copied from a contiguous array.
fn row_times(times: Option<&PyReadonlyArray1<'_, i64>>) -> PyResult<Option<Vec<Option<i64>>>> {
    times
        .map(|times| Ok(times.as_slice()?.iter().copied().map(Some).collect()))
        .transpose()
}

/// A C-contiguous matrix of `table`'s rows, refused unless its rows are the
/// table's width: its values alone would be read as rows of that width,
/// however the array was shaped. An empty matrix holds no rows to measure.
fn table_rows<'a>(
    node: &node::Node,
    table: u32,
    matrix: &'a PyReadonlyArray2<'_, f32>,
    what: &str,
) -> PyResult<&'a [f32]> {
    let width = node.spec(TableId(table)).map_err(raised)?.features;
    if let [rows, columns] = *matrix.shape()
        && rows > 0
        && columns != width
    {
        return Err(invalid(format!(
            "{what} are rows of {columns} values, and table {table} takes rows of {width}"
        )));
    }
    Ok(matrix.as_slice()?)
}

/// Keys from Python: a C-contiguous (n, 2) array of each key's low and high
/// 64 bits, as `generalforecast.keys` lays out 128-bit keys.
fn wide_keys(keys: &PyReadonlyArray2<'_, u64>) -> PyResult<Vec<u128>> {
    if keys.as_array().ncols() != 2 {
        return Err(invalid(
            "keys are an (n, 2) array of each key's low and high 64 bits",
        ));
    }
    Ok(keys
        .as_slice()?
        .chunks_exact(2)
        .map(|pair| u128::from(pair[0]) | (u128::from(pair[1]) << 64))
        .collect())
}

/// Keys as `wide_keys` takes them.
fn key_matrix(py: Python<'_>, keys: &[u128]) -> PyResult<Py<PyArray2<u64>>> {
    let pairs: Vec<u64> = keys
        .iter()
        .flat_map(|&key| [key as u64, (key >> 64) as u64])
        .collect();
    Ok(pairs.into_pyarray(py).reshape([keys.len(), 2])?.unbind())
}

/// Row-major values as a matrix of `rows` rows, owning them without a copy.
fn matrix(py: Python<'_>, values: Vec<f32>, rows: usize) -> PyResult<Py<PyArray2<f32>>> {
    let width = values.len().checked_div(rows).unwrap_or(0);
    Ok(values.into_pyarray(py).reshape([rows, width])?.unbind())
}

/// Targets as an array, owning them without a copy.
fn target_array(py: Python<'_>, targets: Targets) -> Py<PyAny> {
    match targets {
        Targets::Classes(labels) => labels.into_pyarray(py).into_any().unbind(),
        Targets::Values(values) => values.into_pyarray(py).into_any().unbind(),
    }
}

fn targets(array: &Bound<'_, PyAny>) -> PyResult<Targets> {
    if let Ok(labels) = array.extract::<PyReadonlyArray1<i32>>() {
        return Ok(Targets::Classes(labels.as_slice()?.to_vec()));
    }
    let values = array.extract::<PyReadonlyArray1<f32>>().map_err(|_| {
        invalid(
            "targets must be a contiguous int32 array of class labels or a float32 \
             array of values",
        )
    })?;
    Ok(Targets::Values(values.as_slice()?.to_vec()))
}

/// Rows to fit a table's state on, which also embeds them into `pages`.
#[pyclass(frozen, get_all)]
struct Fit {
    table: u32,
    epoch: u64,
    rows: Py<PyArray1<u32>>,
    features: Py<PyArray2<f32>>,
    targets: Py<PyAny>,
    /// The step's pages these rows fill, in order.
    pages: Vec<usize>,
}

/// The representations of one page's rows.
///
/// `features` and `targets` are set when the worker embeds them.
#[pyclass(frozen, get_all)]
struct Page {
    table: u32,
    key: CacheKey,
    rows: Py<PyArray1<u32>>,
    /// `encode` (embed and keep), `reuse` or `transient`.
    action: &'static str,
    /// The host slots kept for the representations of a page to encode.
    slots: Vec<u32>,
    features: Option<Py<PyArray2<f32>>>,
    targets: Option<Py<PyAny>>,
    /// Cached pages to copy rows from, each with (position in the source,
    /// position in `rows`) pairs; `features` holds the other rows.
    copies: Vec<(CacheKey, Vec<(usize, usize)>)>,
}

/// One unit of a step: rows to encode or reuse.
///
/// `features` and `targets` are set when the worker fits the unit's own state.
#[pyclass(frozen, get_all)]
struct Unit {
    table: u32,
    key: CacheKey,
    rows: Py<PyArray1<u32>>,
    /// `encode`, `reuse` or `transient`.
    action: &'static str,
    slots: Vec<u32>,
    features: Option<Py<PyArray2<f32>>>,
    targets: Option<Py<PyAny>>,
}

/// One request's queries, and the unit each decodes against.
#[pyclass(frozen, get_all)]
struct QueryBatch {
    request: u64,
    table: u32,
    epoch: u64,
    features: Py<PyArray2<f32>>,
    units: Vec<usize>,
}

/// A group of series as the model reads it.
///
/// A dense grid of values, which of them are observed, and each row's role.
#[pyclass(frozen, get_all)]
struct Grid {
    values: Py<PyArray2<f32>>,
    observed: Py<PyArray2<bool>>,
    roles: Py<PyArray1<i32>>,
}

fn grid(py: Python<'_>, grid: forecasting::Grid) -> PyResult<Grid> {
    let observed = PyArray1::from_vec(py, grid.observed)
        .reshape([grid.rows, grid.steps])?
        .unbind();
    Ok(Grid {
        values: matrix(py, grid.values, grid.rows)?,
        observed,
        roles: PyArray1::from_vec(py, grid.roles).unbind(),
    })
}

/// A unit of series: a group's context to encode, reuse or forecast whole.
#[pyclass(frozen, get_all)]
struct SeriesUnit {
    table: u32,
    key: CacheKey,
    /// The group's series, in the order the grid's rows take.
    ///
    /// Each has its target row, then one row per feature column.
    members: Vec<u64>,
    cutoff: i64,
    grid: Py<Grid>,
    /// `encode`, `reuse`, `extend` or `transient`; a unit forecast whole is
    /// transient.
    action: &'static str,
    slots: Vec<u32>,
    /// For an extension, the entry holding the shorter encoding it grows.
    source: Option<CacheKey>,
    /// For an extension, how many steps at the grid's end are new.
    grown: usize,
}

/// One series' forecast.
///
/// The step's series unit it reads, the member it forecasts, its horizon and
/// the group's known future values.
#[pyclass(frozen, get_all)]
struct Forecast {
    request: u64,
    table: u32,
    epoch: u64,
    unit: usize,
    target: usize,
    horizon: usize,
    future: Py<Grid>,
}

/// A copy or drop the worker makes before running a step.
#[pyclass(frozen, get_all)]
struct Movement {
    key: CacheKey,
    source: &'static str,
    /// The tier the entry moves to, or None if it is dropped.
    target: Option<&'static str>,
    /// The slots it moves into there.
    slots: Vec<u32>,
}

fn movements(movements: &[gf_core::cache::Movement]) -> Vec<Movement> {
    movements
        .iter()
        .map(|movement| Movement {
            key: CacheKey::from(&movement.key),
            source: movement.from.name(),
            target: movement.to.as_ref().map(|placement| placement.tier.name()),
            slots: movement
                .to
                .as_ref()
                .map(|placement| slot_numbers(&placement.slots))
                .unwrap_or_default(),
        })
        .collect()
}

/// What a worker runs, in order: movements, fits, pages, units, queries, forecasts.
#[pyclass(frozen)]
struct Step {
    #[pyo3(get)]
    movements: Vec<Py<Movement>>,
    #[pyo3(get)]
    fits: Vec<Py<Fit>>,
    #[pyo3(get)]
    pages: Vec<Py<Page>>,
    #[pyo3(get)]
    units: Vec<Py<Unit>>,
    #[pyo3(get)]
    queries: Vec<Py<QueryBatch>>,
    #[pyo3(get)]
    series_units: Vec<Py<SeriesUnit>>,
    #[pyo3(get)]
    forecasts: Vec<Py<Forecast>>,
    /// The step as the node planned it, for `complete` and `fail`.
    inner: node::Step,
}

impl Step {
    /// The step's Python objects, built once, so reading an attribute copies
    /// nothing; the rows' features and targets move into them without a copy.
    fn new(py: Python<'_>, mut step: node::Step) -> PyResult<Self> {
        let StepRows {
            fits: fit_rows,
            pages: page_rows,
            units: unit_rows,
            queries: query_rows,
            series_units: unit_grids,
            forecasts: future_grids,
        } = step.take_rows();
        let fits = step
            .fits()
            .iter()
            .zip(fit_rows)
            .map(|(fit, (features, targets))| {
                Py::new(
                    py,
                    Fit {
                        table: fit.table.0,
                        epoch: fit.epoch.0,
                        rows: row_array(py, &fit.rows),
                        features: matrix(py, features, fit.rows.len())?,
                        targets: target_array(py, targets),
                        pages: fit.pages.iter().map(|page| page.0).collect(),
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        let pages = step
            .pages()
            .iter()
            .zip(page_rows)
            .map(|(page, rows)| {
                let copied: usize = page
                    .copies
                    .iter()
                    .map(|(_, positions)| positions.len())
                    .sum();
                let (features, targets) = match rows {
                    Some((features, targets)) => (
                        Some(matrix(py, features, page.rows.len() - copied)?),
                        Some(target_array(py, targets)),
                    ),
                    None => (None, None),
                };
                Py::new(
                    py,
                    Page {
                        table: page.table.0,
                        key: CacheKey::from(&page.key),
                        rows: row_array(py, &page.rows),
                        action: page.placed.action().name(),
                        slots: slot_numbers(page.placed.slots()),
                        features,
                        targets,
                        copies: page
                            .copies
                            .iter()
                            .map(|(source, positions)| (CacheKey::from(source), positions.clone()))
                            .collect(),
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        let units = step
            .units()
            .iter()
            .zip(unit_rows)
            .map(|(unit, rows)| {
                let (features, targets) = match rows {
                    Some((features, targets)) => (
                        Some(matrix(py, features, unit.rows.len())?),
                        Some(target_array(py, targets)),
                    ),
                    None => (None, None),
                };
                Py::new(
                    py,
                    Unit {
                        table: unit.table.0,
                        key: CacheKey::from(&unit.key),
                        rows: row_array(py, &unit.rows),
                        action: unit.placed.action().name(),
                        slots: slot_numbers(unit.placed.slots()),
                        features,
                        targets,
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        let queries = step
            .queries()
            .iter()
            .zip(query_rows)
            .map(|(batch, features)| {
                Py::new(
                    py,
                    QueryBatch {
                        request: batch.request.0,
                        table: batch.table.0,
                        epoch: batch.epoch.0,
                        features: matrix(py, features, batch.units.len())?,
                        units: batch.units.iter().map(|unit| unit.0).collect(),
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        let series_units = step
            .series_units()
            .iter()
            .zip(unit_grids)
            .map(|(unit, laid)| {
                Py::new(
                    py,
                    SeriesUnit {
                        table: unit.table.0,
                        key: CacheKey::from(&unit.key),
                        members: unit.members.iter().map(|member| member.0).collect(),
                        cutoff: unit.cutoff,
                        grid: Py::new(py, grid(py, laid)?)?,
                        action: unit.placed.action().name(),
                        slots: slot_numbers(unit.placed.slots()),
                        source: unit.source.as_ref().map(CacheKey::from),
                        grown: unit.grown,
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        let forecasts = step
            .forecasts()
            .iter()
            .zip(future_grids)
            .map(|(forecast, laid)| {
                Py::new(
                    py,
                    Forecast {
                        request: forecast.request.0,
                        table: forecast.table.0,
                        epoch: forecast.epoch.0,
                        unit: forecast.unit.0,
                        target: forecast.target,
                        horizon: forecast.horizon,
                        future: Py::new(py, grid(py, laid)?)?,
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        let movements = movements(step.movements())
            .into_iter()
            .map(|movement| Py::new(py, movement))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            movements,
            fits,
            pages,
            units,
            queries,
            series_units,
            forecasts,
            inner: step,
        })
    }
}

/// Many tables behind one request queue and one cache.
///
/// Calls raise `KeyError` for a table or key that does not exist, `ValueError`
/// for input the node refuses, such as a negative size, a label outside the
/// table's classes or rows of the wrong width, and `RuntimeError` when the
/// cache or planner fails. A call that fails changes nothing.
///
/// Not for concurrent use: a call made while another thread's runs raises
/// `RuntimeError`. `LocalClient` makes threads take turns.
#[pyclass]
struct Node(node::Node);

#[pymethods]
impl Node {
    #[new]
    #[pyo3(signature = (
        device_bytes, host_bytes, slot_bytes = 1 << 20, disk_bytes = 0, step_bytes = None,
        eviction = "gdsf",
    ))]
    fn new(
        device_bytes: i64,
        host_bytes: i64,
        slot_bytes: i64,
        disk_bytes: i64,
        step_bytes: Option<i64>,
        eviction: &str,
    ) -> PyResult<Self> {
        let eviction = Eviction::ALL
            .into_iter()
            .find(|policy| policy.name() == eviction)
            .ok_or_else(|| invalid(format!("eviction is gdsf or lru, not {eviction:?}")))?;
        let slot = NonZeroU64::new(count(slot_bytes, "slot_bytes")?)
            .ok_or_else(|| invalid("slot_bytes must be positive"))?;
        let tier = |bytes: i64| -> PyResult<TierConfig> {
            let bytes: u64 = count(bytes, "a tier's bytes")?;
            let slots = u32::try_from(bytes / slot.get()).map_err(|_| invalid("too many slots"))?;
            Ok(TierConfig {
                slot_bytes: slot,
                slots,
            })
        };
        let (device, host, disk) = (tier(device_bytes)?, tier(host_bytes)?, tier(disk_bytes)?);
        let mut node = node::Node::new(device, host)
            .and_then(|node| node.with_disk(disk))
            .map_err(raised)?
            .with_eviction(eviction);
        if let Some(bytes) = step_bytes {
            let bytes = NonZeroU64::new(count(bytes, "step_bytes")?)
                .ok_or_else(|| invalid("step_bytes must be positive"))?;
            node.set_step_bytes(bytes.get());
        }
        Ok(Self(node))
    }

    /// Adds an empty table and returns its id. `classes` is 0 for regression.
    ///
    /// `summary` is `table` or `context`; a table that summarizes each context
    /// keeps `state_bytes` of state beside each unit's rows. A fit on more
    /// than `fit_bound` rows waits for `set_fit_limits`; None never waits.
    #[pyo3(signature = (
        model, features, classes, strategy, bytes_per_row,
        bytes_per_representation = 0, page_size = 128, bucket_minimum = 8,
        time_bucket = None, refit_after = None, refit_shift = None, fit_rows = None,
        context_rows = None, past_cap = "warn", region_rows = None, rebuild = "always",
        rebuild_at = 0.0, rebuild_steps = 8, rebuild_seconds = Some(1.0),
        refit_cutoffs = "each", scale_rows = None, retrieval = 0,
        summary = "table", state_bytes = 0,
        categorical = Vec::new(), retrieval_categoricals = "codes",
        unlabeled_context = false, fit_bound = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create_table(
        &mut self,
        model: i64,
        features: i64,
        classes: i64,
        strategy: &Strategy,
        bytes_per_row: i64,
        bytes_per_representation: i64,
        page_size: i64,
        bucket_minimum: i64,
        time_bucket: Option<i64>,
        refit_after: Option<f64>,
        refit_shift: Option<f64>,
        fit_rows: Option<i64>,
        context_rows: Option<i64>,
        past_cap: &str,
        region_rows: Option<i64>,
        rebuild: &str,
        rebuild_at: f64,
        rebuild_steps: i64,
        rebuild_seconds: Option<f64>,
        refit_cutoffs: &str,
        scale_rows: Option<i64>,
        retrieval: i64,
        summary: &str,
        state_bytes: i64,
        categorical: Vec<usize>,
        retrieval_categoricals: &str,
        unlabeled_context: bool,
        fit_bound: Option<i64>,
    ) -> PyResult<u32> {
        let cutoffs = refit_cutoffs.parse::<CutoffRefit>().map_err(invalid)?;
        let refit = RefitPolicy::new(refit_after, refit_shift)
            .and_then(|policy| policy.with_cutoffs(cutoffs))
            .map_err(raised)?;
        let summary = SummaryScope::of_name(summary)
            .ok_or_else(|| invalid(format!("summary is table or context, not {summary:?}")))?;
        let spec = TableSpec {
            model: ModelId(count(model, "model")?),
            features: count(features, "features")?,
            retrieval: count(retrieval, "retrieval")?,
            categorical,
            categories: retrieval_categoricals
                .parse::<CategoryEncoding>()
                .map_err(invalid)?,
            unlabeled_context,
            task: Task::of_classes(count(classes, "classes")?),
            page_size: positive(page_size, "page_size")?,
            strategy: strategy.0,
            sizes: EntrySizes {
                bytes_per_row: count(bytes_per_row, "bytes_per_row")?,
                bytes_per_representation: count(
                    bytes_per_representation,
                    "bytes_per_representation",
                )?,
                bucket_minimum: positive(bucket_minimum, "bucket_minimum")?,
                state_bytes: match summary {
                    SummaryScope::Table => 0,
                    SummaryScope::Context => count(state_bytes, "state_bytes")?,
                },
                state_rows: usize::MAX,
                fused_rows: usize::MAX,
                fit_bound: fit_bound
                    .map(|rows| count(rows, "fit_bound"))
                    .transpose()?
                    .unwrap_or(usize::MAX),
            },
            time_bucket: time_bucket
                .map(|width| {
                    NonZeroU64::new(count(width, "time_bucket")?)
                        .ok_or_else(|| invalid("time_bucket must be positive"))
                })
                .transpose()?,
            refit,
            scale_rows: scale_rows
                .map(|rows| {
                    NonZeroU64::new(count(rows, "scale_rows")?)
                        .ok_or_else(|| invalid("scale_rows must be positive"))
                })
                .transpose()?,
            fit_rows: positive(fit_rows.unwrap_or(DEFAULT_FIT_ROWS as i64), "fit_rows")?,
            context_rows: context_rows
                .map(|rows| {
                    NonZeroU64::new(count(rows, "context_rows")?)
                        .ok_or_else(|| invalid("context_rows must be positive"))
                })
                .transpose()?,
            past_cap: past_cap.parse::<PastCap>().map_err(invalid)?,
            regions: region_rows
                .map(|rows| -> PyResult<RegionSpec> {
                    let rebuild = match rebuild {
                        "always" => Rebuild::Always,
                        "never" => Rebuild::Never,
                        "churn" if rebuild_at > 0.0 && rebuild_at.is_finite() => {
                            Rebuild::Churn(rebuild_at)
                        }
                        "every" if rebuild_at >= 0.0 && rebuild_at.is_finite() => {
                            Rebuild::Every(Duration::from_secs_f64(rebuild_at))
                        }
                        _ => {
                            return Err(invalid(format!(
                                "rebuild is always, never, a positive churn share or a \
                                 duration, not {rebuild} at {rebuild_at}"
                            )));
                        }
                    };
                    let time = rebuild_seconds
                        .map(|seconds| {
                            Duration::try_from_secs_f64(seconds).map_err(|_| {
                                invalid(format!(
                                    "a rebuild's wait is a finite number of seconds, \
                                     not {seconds}"
                                ))
                            })
                        })
                        .transpose()?;
                    let steps = u32::try_from(rebuild_steps).map_err(|_| {
                        invalid(format!(
                            "a rebuild's wait in steps is from 0 to {}, not {rebuild_steps}",
                            u32::MAX
                        ))
                    })?;
                    Ok(RegionSpec {
                        rows: positive(rows, "region_rows")?.get() as u64,
                        rebuild,
                        wait: RebuildWait { steps, time },
                    })
                })
                .transpose()?,
            summary,
            series: None,
        };
        Ok(self.0.create_table(spec).map_err(raised)?.0)
    }

    /// Adds an empty table of series and returns its id.
    ///
    /// Its rows are observations keyed by series and time, with `features`
    /// covariate columns, of which `known` are known over the horizon; the
    /// grid's `step` is in the table's time unit. `patch`, `output_patch`,
    /// `context`, `pass_limit`, `max_rows`, `reusable` and `extendable` come
    /// from the model's properties. A group's cache takes `bytes_per_row`
    /// per padded row and `patch_bytes` beside them, per patch.
    #[pyo3(signature = (
        model, features, step, patch, context, output_patch, group, bytes_per_row,
        pass_limit = None, patch_bytes = 0, known = Vec::new(), max_rows = None,
        neighbors = "targets", reusable = true, bucket_minimum = 1, extendable = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create_series_table(
        &mut self,
        model: i64,
        features: i64,
        step: i64,
        patch: i64,
        context: i64,
        output_patch: i64,
        group: &GroupStrategy,
        bytes_per_row: i64,
        pass_limit: Option<i64>,
        patch_bytes: i64,
        known: Vec<i64>,
        max_rows: Option<i64>,
        neighbors: &str,
        reusable: bool,
        bucket_minimum: i64,
        extendable: bool,
    ) -> PyResult<u32> {
        let neighbors = match neighbors {
            "targets" => forecasting::Neighbors::Targets,
            "covariates" => forecasting::Neighbors::Covariates,
            other => {
                return Err(invalid(format!(
                    "neighbors is targets or covariates, not {other:?}"
                )));
            }
        };
        let series = SeriesSpec {
            step: NonZeroU64::new(count(step, "step")?)
                .ok_or_else(|| invalid("step must be positive"))?,
            patch: positive(patch, "patch")?,
            context: positive(context, "context")?,
            max_horizon: forecasting::horizon_limit(
                positive(output_patch, "output_patch")?,
                pass_limit
                    .map(|limit| positive(limit, "pass_limit"))
                    .transpose()?,
            ),
            max_rows: max_rows
                .map(|rows| positive(rows, "max_rows"))
                .transpose()?,
            known: known
                .into_iter()
                .map(|column| count(column, "known"))
                .collect::<PyResult<_>>()?,
            group: group.0,
            neighbors,
            reusable,
            extendable,
            patch_bytes: count(patch_bytes, "patch_bytes")?,
        };
        let spec = TableSpec {
            model: ModelId(count(model, "model")?),
            features: count(features, "features")?,
            categorical: Vec::new(),
            categories: CategoryEncoding::Codes,
            unlabeled_context: false,
            task: Task::Values,
            page_size: positive(128, "page_size")?,
            strategy: planner::Strategy::Exact,
            sizes: EntrySizes {
                bytes_per_row: count(bytes_per_row, "bytes_per_row")?,
                bytes_per_representation: 0,
                state_bytes: 0,
                bucket_minimum: positive(bucket_minimum, "bucket_minimum")?,
                state_rows: usize::MAX,
                fused_rows: usize::MAX,
                fit_bound: usize::MAX,
            },
            time_bucket: None,
            refit: RefitPolicy::default(),
            fit_rows: positive(DEFAULT_FIT_ROWS as i64, "fit_rows")?,
            context_rows: None,
            past_cap: PastCap::Warn,
            regions: None,
            scale_rows: None,
            retrieval: 0,
            summary: SummaryScope::Table,
            series: Some(series),
        };
        Ok(self.0.create_table(spec).map_err(raised)?.0)
    }

    /// Queues forecasts of a series table's series over `horizon` steps.
    ///
    /// Each is from its `before` time, or the series' latest observed time plus a step
    /// without one; on a table of given groups, each series' group is in `groups`.
    /// Returns the request id.
    #[pyo3(signature = (table, series, horizon, before = None, groups = None))]
    fn submit_forecast(
        &mut self,
        table: u32,
        series: PyReadonlyArray1<'_, u64>,
        horizon: i64,
        before: Option<PyReadonlyArray1<'_, i64>>,
        groups: Option<Vec<PyReadonlyArray1<'_, u64>>>,
    ) -> PyResult<u64> {
        let before = before
            .as_ref()
            .map(PyReadonlyArray1::as_slice)
            .transpose()?;
        let groups: Option<Vec<Vec<u64>>> = groups
            .map(|groups| {
                groups
                    .iter()
                    .map(|group| group.as_slice().map(<[u64]>::to_vec).map_err(PyErr::from))
                    .collect::<PyResult<_>>()
            })
            .transpose()?;
        Ok(self
            .0
            .submit_forecast(
                TableId(table),
                series.as_slice()?,
                before,
                positive(horizon, "horizon")?,
                groups.as_deref(),
            )
            .map_err(raised)?
            .0)
    }

    /// Adds rows, at `times` if given; returns their row ids.
    #[pyo3(signature = (table, keys, features, targets, times = None))]
    fn insert<'py>(
        &mut self,
        py: Python<'py>,
        table: u32,
        keys: PyReadonlyArray2<'py, u64>,
        features: PyReadonlyArray2<'py, f32>,
        targets: &Bound<'py, PyAny>,
        times: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<Py<PyArray1<u32>>> {
        // Copied while the GIL is held: another thread may write to the arrays
        // once it is released.
        let times = row_times(times.as_ref())?;
        let (keys, features, targets) = (
            wide_keys(&keys)?,
            table_rows(&self.0, table, &features, "features")?.to_vec(),
            self::targets(targets)?,
        );
        let node = &mut self.0;
        let rows = py
            .detach(|| node.insert_at(TableId(table), &keys, &features, targets, times.as_deref()))
            .map_err(raised)?;
        Ok(row_array(py, &rows))
    }

    /// Changes rows, and their times if given; returns their row ids.
    #[pyo3(signature = (table, keys, features, targets, times = None))]
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        table: u32,
        keys: PyReadonlyArray2<'py, u64>,
        features: PyReadonlyArray2<'py, f32>,
        targets: &Bound<'py, PyAny>,
        times: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<Py<PyArray1<u32>>> {
        // Copied while the GIL is held: another thread may write to the arrays
        // once it is released.
        let times = row_times(times.as_ref())?;
        let (keys, features, targets) = (
            wide_keys(&keys)?,
            table_rows(&self.0, table, &features, "features")?.to_vec(),
            self::targets(targets)?,
        );
        let node = &mut self.0;
        let rows = py
            .detach(|| node.update_at(TableId(table), &keys, &features, targets, times.as_deref()))
            .map_err(raised)?;
        Ok(row_array(py, &rows))
    }

    /// Sets rows' targets, keeping their features and times; returns their row ids.
    fn label<'py>(
        &mut self,
        py: Python<'py>,
        table: u32,
        keys: PyReadonlyArray2<'py, u64>,
        targets: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyArray1<u32>>> {
        let (keys, targets) = (wide_keys(&keys)?, self::targets(targets)?);
        let node = &mut self.0;
        let rows = py
            .detach(|| node.label(TableId(table), &keys, targets))
            .map_err(raised)?;
        Ok(row_array(py, &rows))
    }

    /// Adds rows or replaces them by key, at `times` if given; returns their row ids.
    #[pyo3(signature = (table, keys, features, targets, times = None))]
    fn upsert<'py>(
        &mut self,
        py: Python<'py>,
        table: u32,
        keys: PyReadonlyArray2<'py, u64>,
        features: PyReadonlyArray2<'py, f32>,
        targets: &Bound<'py, PyAny>,
        times: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<Py<PyArray1<u32>>> {
        // Copied while the GIL is held: another thread may write to the arrays
        // once it is released.
        let times = row_times(times.as_ref())?;
        let (keys, features, targets) = (
            wide_keys(&keys)?,
            table_rows(&self.0, table, &features, "features")?.to_vec(),
            self::targets(targets)?,
        );
        let node = &mut self.0;
        let rows = py
            .detach(|| node.upsert_at(TableId(table), &keys, &features, targets, times.as_deref()))
            .map_err(raised)?;
        Ok(row_array(py, &rows))
    }

    /// The keys among `keys` that a table holds, in order.
    fn present<'py>(
        &self,
        py: Python<'py>,
        table: u32,
        keys: PyReadonlyArray2<'py, u64>,
    ) -> PyResult<Py<PyArray2<u64>>> {
        let present = self
            .0
            .present(TableId(table), &wide_keys(&keys)?)
            .map_err(raised)?;
        key_matrix(py, &present)
    }

    /// Deletes rows; returns their row ids.
    fn delete<'py>(
        &mut self,
        py: Python<'py>,
        table: u32,
        keys: PyReadonlyArray2<'py, u64>,
    ) -> PyResult<Py<PyArray1<u32>>> {
        let keys = wide_keys(&keys)?;
        let node = &mut self.0;
        let rows = py
            .detach(|| node.delete(TableId(table), &keys))
            .map_err(raised)?;
        Ok(row_array(py, &rows))
    }

    /// Deletes every row of a table; returns their row ids.
    fn truncate(&mut self, py: Python<'_>, table: u32) -> PyResult<Py<PyArray1<u32>>> {
        let node = &mut self.0;
        let rows = py
            .detach(|| node.truncate(TableId(table)))
            .map_err(raised)?;
        Ok(row_array(py, &rows))
    }

    /// Number of live rows in a table.
    fn len(&self, table: u32) -> PyResult<u64> {
        self.0.len(TableId(table)).map_err(raised)
    }

    /// Queues query rows for a table; returns the request id.
    ///
    /// Each query reads only rows before its `before` time, if given.
    #[pyo3(signature = (table, queries, before = None))]
    fn submit(
        &mut self,
        table: u32,
        queries: PyReadonlyArray2<'_, f32>,
        before: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<u64> {
        let before = before
            .as_ref()
            .map(PyReadonlyArray1::as_slice)
            .transpose()?;
        Ok(self
            .0
            .submit_before(
                TableId(table),
                table_rows(&self.0, table, &queries, "queries")?,
                before,
            )
            .map_err(raised)?
            .0)
    }

    /// Queues query rows that each read exactly their given rows; returns the request id.
    fn submit_keys(
        &mut self,
        table: u32,
        queries: PyReadonlyArray2<'_, f32>,
        contexts: Vec<PyReadonlyArray2<'_, u64>>,
    ) -> PyResult<u64> {
        let contexts = contexts
            .iter()
            .map(wide_keys)
            .collect::<PyResult<Vec<_>>>()?;
        let queries = table_rows(&self.0, table, &queries, "queries")?;
        Ok(self
            .0
            .submit_keys(TableId(table), queries, &contexts)
            .map_err(raised)?
            .0)
    }

    /// Each query's context keys, as `submit` would plan them, queueing nothing.
    #[pyo3(signature = (table, queries, before = None))]
    fn contexts(
        &mut self,
        py: Python<'_>,
        table: u32,
        queries: PyReadonlyArray2<'_, f32>,
        before: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<Vec<Py<PyArray2<u64>>>> {
        // Copied out, so planning, which may draw regions, runs without the GIL.
        let before = before
            .as_ref()
            .map(|before| before.as_slice().map(<[i64]>::to_vec))
            .transpose()?;
        let queries = table_rows(&self.0, table, &queries, "queries")?.to_vec();
        let node = &mut self.0;
        py.detach(|| node.contexts(TableId(table), &queries, before.as_deref()))
            .map_err(raised)?
            .iter()
            .map(|keys| key_matrix(py, keys))
            .collect()
    }

    /// Drops a queued request before it is planned; returns whether it was queued.
    fn cancel(&mut self, request: u64) -> bool {
        self.0.cancel(RequestId(request))
    }

    /// Plans pending requests into one step, or None if none can run yet.
    fn schedule(&mut self, py: Python<'_>) -> PyResult<Option<Step>> {
        let node = &mut self.0;
        let step = py.detach(|| node.schedule()).map_err(raised)?;
        step.map(|step| Step::new(py, step)).transpose()
    }

    /// Releases a finished step's units; returns drops of entries from ended epochs.
    fn complete(&mut self, step: &Step) -> Vec<Movement> {
        movements(&self.0.complete(&step.inner))
    }

    /// Undoes a step the worker failed to run.
    ///
    /// Returns drops of the entries it was to encode, and of entries from ended
    /// epochs.
    fn fail(&mut self, step: &Step) -> Vec<Movement> {
        movements(&self.0.fail(&step.inner))
    }

    /// Each query's region churn, as `Rebuild::Churn` compares it; NaN if unbuilt or uncached.
    fn region_churn(
        &self,
        py: Python<'_>,
        table: u32,
        queries: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let queries = table_rows(&self.0, table, &queries, "queries")?;
        let churn: Vec<f64> = self
            .0
            .region_churn(TableId(table), queries)
            .map_err(raised)?
            .into_iter()
            .map(|churn| churn.unwrap_or(f64::NAN))
            .collect();
        Ok(churn.into_pyarray(py).unbind())
    }

    /// Tables that came to wait for their fit's rows to be measured, each once.
    fn take_unmeasured(&mut self) -> Vec<u32> {
        self.0
            .take_unmeasured()
            .into_iter()
            .map(|table| table.0)
            .collect()
    }

    /// Sets the rows a table's state alone and a fit with its rows take, as measured.
    fn set_fit_limits(&mut self, table: u32, state_rows: i64, fused_rows: i64) -> PyResult<()> {
        self.0
            .set_fit_limits(
                TableId(table),
                positive(state_rows, "state_rows")?.get(),
                count(fused_rows, "fused_rows")?,
            )
            .map_err(raised)
    }

    /// The most rows a table's state is fit on, within the worker's call budget.
    fn effective_fit_rows(&self, table: u32) -> PyResult<usize> {
        Ok(self
            .0
            .spec(TableId(table))
            .map_err(raised)?
            .effective_fit_rows())
    }

    /// What a table's regions served and their rebuilds cost; None without regions.
    fn region_stats(&self, py: Python<'_>, table: u32) -> PyResult<Option<Py<PyDict>>> {
        let Some(stats) = self.0.region_stats(TableId(table)).map_err(raised)? else {
            return Ok(None);
        };
        let described = PyDict::new(py);
        described.set_item("regions", stats.regions)?;
        described.set_item("mean_rows", stats.mean_rows)?;
        described.set_item("levels", stats.levels)?;
        described.set_item("hits", stats.hits)?;
        described.set_item("stale", stats.stale)?;
        described.set_item("misses", stats.misses)?;
        described.set_item("fresh", stats.fresh)?;
        described.set_item("rebuilds", stats.rebuilds)?;
        described.set_item("rebuilt_rows", stats.rebuilt_rows)?;
        described.set_item("overdue_seconds", stats.overdue.as_secs_f64())?;
        Ok(Some(described.unbind()))
    }

    /// Tables whose contexts first passed their model's trained length, each once.
    fn take_past_cap(&mut self) -> Vec<u32> {
        self.0
            .take_past_cap()
            .into_iter()
            .map(|table| table.0)
            .collect()
    }

    /// Requests planning refused since the last call, with why and whether it names what does not exist.
    fn take_refused(&mut self) -> Vec<(u64, String, bool)> {
        self.0
            .take_refused()
            .into_iter()
            .map(|(request, error)| {
                let missing = matches!(error.kind(), ErrorKind::NotFound);
                (request.0, error.to_string(), missing)
            })
            .collect()
    }

    /// Drops of cached entries holding rows since changed or deleted, to apply now.
    fn take_drops(&mut self) -> Vec<Movement> {
        movements(&self.0.take_drops())
    }

    /// Forgets every cached entry and fit, as when the worker starts again empty.
    ///
    /// Call it with no step in flight.
    fn forget_cache(&mut self) {
        self.0.forget_cache();
    }

    /// Removes a table; returns drops of its cache entries.
    ///
    /// Its queued requests are refused; see `take_refused`.
    fn remove_table(&mut self, table: u32) -> PyResult<Vec<Movement>> {
        Ok(movements(
            &self.0.remove_table(TableId(table)).map_err(raised)?,
        ))
    }

    /// Refits a table's state and retrieval scaling.
    ///
    /// Returns drops of old entries nothing reads.
    fn refit(&mut self, py: Python<'_>, table: u32) -> PyResult<Vec<Movement>> {
        let node = &mut self.0;
        let dropped = py.detach(|| node.refit(TableId(table))).map_err(raised)?;
        Ok(movements(&dropped))
    }

    /// Changes how a table chooses contexts from here on; see `Node.strategy`.
    fn set_strategy(&mut self, table: u32, strategy: &Strategy) -> PyResult<()> {
        self.0
            .set_strategy(TableId(table), strategy.0)
            .map_err(raised)
    }

    /// The strategy a table's next plan chooses contexts with.
    ///
    /// An adaptive strategy resolved against the table's live rows, or the
    /// one it was given.
    fn strategy(&self, table: u32) -> PyResult<Strategy> {
        Ok(Strategy(self.0.strategy(TableId(table)).map_err(raised)?))
    }
}

pyo3::create_exception!(
    _core,
    RequestError,
    PyValueError,
    "A refused inference request: `args` are its code, message and HTTP status."
);

fn refused(error: request::RequestError) -> PyErr {
    RequestError::new_err((error.code, error.message, error.status))
}

fn surface(name: &str) -> PyResult<Surface> {
    name.parse().map_err(invalid)
}

fn number(py: Python<'_>, value: Num) -> PyResult<Py<PyAny>> {
    match value {
        Num::Int(value) => value.into_py_any(py),
        Num::Float(value) => value.into_py_any(py),
    }
}

fn label(py: Python<'_>, value: &Label) -> PyResult<Py<PyAny>> {
    match value {
        Label::Bool(value) => value.into_py_any(py),
        Label::Int(value) => value.into_py_any(py),
        Label::Text(value) => value.into_py_any(py),
    }
}

/// The canonical reference a model spec names, if the surface may serve it.
///
/// `surface` is `remote` (the hosted Models API), `self-hosted` (a server its
/// operator runs) or `local` (a caller that admitted the model already).
/// Raises `RequestError` as the Models API refuses the reference.
#[pyfunction]
fn resolve_model(spec: &Bound<'_, PyAny>, surface: &str) -> PyResult<String> {
    let spec = spec.extract::<String>().ok();
    catalog::resolve(spec.as_deref(), self::surface(surface)?)
        .map(|model| model.to_string())
        .map_err(refused)
}

/// Requested quantile levels, from their JSON list: strictly increasing, in (0, 1).
///
/// Raises `RequestError` for levels a request may not select.
#[pyfunction]
fn quantile_levels(payload: &[u8]) -> PyResult<Vec<f64>> {
    request::quantile_levels(payload).map_err(refused)
}

/// A classification or regression request, validated before any weights load.
#[pyclass(frozen)]
struct TabularRequest {
    /// The canonical reference of the model it resolved to.
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    task: &'static str,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    features: Vec<String>,
    /// The labeled context rows' features, `(rows, features)`.
    #[pyo3(get)]
    context: Py<PyArray2<f32>>,
    /// Per labeled row, its class code into `classes` (int32) or its value (float32).
    #[pyo3(get)]
    targets: Py<PyAny>,
    /// The classes in output order; empty for a regression.
    #[pyo3(get)]
    classes: Vec<Py<PyAny>>,
    /// The queries' features, `(queries, features)`.
    #[pyo3(get)]
    queries: Py<PyArray2<f32>>,
    #[pyo3(get)]
    query_keys: Vec<String>,
    /// The quantile levels a regression returns; empty for a classification.
    #[pyo3(get)]
    levels: Vec<f64>,
    /// Context rows supplied, labeled or not.
    #[pyo3(get)]
    supplied_rows: usize,
    /// SHA-256 of the model and its context, as hex: same context, same table.
    #[pyo3(get)]
    fingerprint: String,
    /// The request as parsed, which the engine runs.
    parsed: Arc<request::Tabular>,
}

/// Related series aligned on one time axis, unpadded: the last step is the origin.
///
/// Rows are the group's targets in query order, then its covariates sorted
/// by name; execution left-pads time to the model's patch.
#[pyclass(frozen, get_all)]
struct AlignedGroup {
    /// `(rows, steps)`, zero where unobserved.
    values: Py<PyArray2<f32>>,
    observed: Py<PyArray2<bool>>,
    /// Per row, `TARGET`, `PAST` or `KNOWN`.
    roles: Py<PyArray1<i32>>,
    /// Known covariates' values over the horizon, `(rows, horizon)`.
    future: Py<PyArray2<f32>>,
    future_observed: Py<PyArray2<bool>>,
}

/// One forecast query: a series forecast with its group from an origin.
#[pyclass(frozen, get_all)]
struct ForecastQuery {
    key: String,
    series: String,
    /// An integer period or floating UTC seconds.
    origin: Py<PyAny>,
    /// The group's target series, in row order.
    members: Vec<String>,
    /// The row of `series` among `members`.
    member: usize,
    /// The forecast's times, one step after another past the origin.
    times: Vec<Py<PyAny>>,
    group: Py<AlignedGroup>,
    /// SHA-256 of the context fingerprint and this group, as hex: its members,
    /// origin, horizon and known-future covariates.
    fingerprint: String,
}

/// A forecast request, validated and aligned before any weights load.
#[pyclass(frozen)]
struct ForecastRequest {
    /// The canonical reference of the model it resolved to.
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    target: String,
    /// An integer number of periods or floating seconds.
    #[pyo3(get)]
    step: Py<PyAny>,
    #[pyo3(get)]
    horizon: usize,
    #[pyo3(get)]
    levels: Vec<f64>,
    #[pyo3(get)]
    known_covariates: Vec<String>,
    #[pyo3(get)]
    queries: Vec<Py<ForecastQuery>>,
    /// `integer-periods` or `utc-seconds`.
    #[pyo3(get)]
    time_axis: &'static str,
    /// Context rows supplied, observed or not.
    #[pyo3(get)]
    supplied_rows: usize,
    /// Context rows with an observed value.
    #[pyo3(get)]
    usable_observations: usize,
    /// SHA-256 of the model and its context, as hex: same context, same table.
    #[pyo3(get)]
    fingerprint: String,
    /// The request as parsed, which the engine runs.
    parsed: Arc<request::Forecast>,
}

fn boolean_matrix(py: Python<'_>, values: Vec<bool>, rows: usize) -> PyResult<Py<PyArray2<bool>>> {
    let width = values.len().checked_div(rows).unwrap_or(0);
    Ok(PyArray1::from_vec(py, values)
        .reshape([rows, width])?
        .unbind())
}

fn tabular_request(py: Python<'_>, parsed: request::Tabular) -> PyResult<TabularRequest> {
    let kept = Arc::new(parsed.clone());
    let rows = parsed.usable_rows();
    let queries = parsed.query_keys.len();
    Ok(TabularRequest {
        parsed: kept,
        model: parsed.model.to_string(),
        task: parsed.task.as_str(),
        target: parsed.target,
        features: parsed.features,
        context: matrix(py, parsed.context, rows)?,
        targets: target_array(py, parsed.targets),
        classes: parsed
            .classes
            .iter()
            .map(|class| label(py, class))
            .collect::<PyResult<_>>()?,
        queries: matrix(py, parsed.queries, queries)?,
        query_keys: parsed.query_keys,
        levels: parsed.levels,
        supplied_rows: parsed.supplied_rows,
        fingerprint: request::hex(&parsed.fingerprint),
    })
}

fn forecast_request(py: Python<'_>, parsed: request::Forecast) -> PyResult<ForecastRequest> {
    let kept = Arc::new(parsed.clone());
    let queries = parsed
        .queries
        .into_iter()
        .map(|query| {
            let group = query.group;
            let group = AlignedGroup {
                values: matrix(py, group.values, group.rows)?,
                observed: boolean_matrix(py, group.observed, group.rows)?,
                roles: PyArray1::from_vec(py, group.roles).unbind(),
                future: matrix(py, group.future, group.rows)?,
                future_observed: boolean_matrix(py, group.future_observed, group.rows)?,
            };
            Py::new(
                py,
                ForecastQuery {
                    key: query.key,
                    series: query.series,
                    origin: number(py, query.origin)?,
                    members: query.members,
                    member: query.member,
                    times: query
                        .times
                        .into_iter()
                        .map(|time| number(py, time))
                        .collect::<PyResult<_>>()?,
                    group: Py::new(py, group)?,
                    fingerprint: request::hex(&query.fingerprint),
                },
            )
        })
        .collect::<PyResult<_>>()?;
    Ok(ForecastRequest {
        parsed: kept,
        model: parsed.model.to_string(),
        target: parsed.target,
        step: number(py, parsed.step)?,
        horizon: parsed.horizon,
        levels: parsed.levels,
        known_covariates: parsed.known,
        queries,
        time_axis: parsed.clock.as_str(),
        supplied_rows: parsed.supplied_rows,
        usable_observations: parsed.usable_observations,
        fingerprint: request::hex(&parsed.fingerprint),
    })
}

/// Parses and validates an inference request's JSON payload, without the GIL.
///
/// `surface` decides the gates its model passes, as for `resolve_model`.
/// `limits` is `request` for one Models API request's bounds or `predictor`
/// for a predictor's bound context, which its lock bounds instead. `levels`
/// are the quantile levels a forecaster outputs, when its weights are
/// loaded; the catalog's declared levels otherwise. Raises `RequestError`
/// with the code, message and status the Models API answers.
#[pyfunction]
#[pyo3(signature = (payload, surface, *, limits = "request", levels = None))]
fn parse_inference(
    py: Python<'_>,
    payload: &[u8],
    surface: &str,
    limits: &str,
    levels: Option<Vec<f64>>,
) -> PyResult<Py<PyAny>> {
    let limits = match limits {
        "request" => Limits::Request,
        "predictor" => Limits::Predictor,
        _ => return Err(invalid(format!("unknown request limits {limits:?}"))),
    };
    let options = Options {
        surface: self::surface(surface)?,
        limits,
        levels: levels.as_deref(),
    };
    match py
        .detach(|| request::parse(payload, &options))
        .map_err(refused)?
    {
        Inference::Tabular(parsed) => Py::new(py, tabular_request(py, parsed)?)?.into_py_any(py),
        Inference::Forecast(parsed) => Py::new(py, forecast_request(py, parsed)?)?.into_py_any(py),
    }
}

/// Rows from an Arrow IPC stream, as one batch.
fn record_batch(payload: &[u8]) -> PyResult<arrow_array::RecordBatch> {
    let reader = arrow_ipc::reader::StreamReader::try_new(payload, None).map_err(invalid)?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().map_err(invalid)?;
    arrow_select::concat::concat_batches(&schema, &batches).map_err(invalid)
}

/// A tabular predictor's context, bound once.
///
/// Its labeled rows encoded as the engine reads them, and the vocabularies
/// its queries are encoded against.
#[pyclass(frozen)]
struct PredictorContext(predictor::Context);

#[pymethods]
impl PredictorContext {
    /// The labeled rows' features, `(rows, width)`.
    #[getter]
    fn features(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f32>>> {
        matrix(py, self.0.features.clone(), self.0.rows)
    }

    /// Per labeled row, its class code into `classes` (int32) or its value (float32).
    #[getter]
    fn targets(&self, py: Python<'_>) -> Py<PyAny> {
        target_array(py, self.0.targets.clone())
    }

    /// The classes in output order; empty for a regression.
    #[getter]
    fn classes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.0
            .classes
            .iter()
            .map(|class| label(py, class))
            .collect()
    }

    /// Labeled context rows.
    #[getter]
    fn rows(&self) -> usize {
        self.0.rows
    }

    /// Each input column's words in code order; empty but for categorical columns.
    #[getter]
    fn vocabularies(&self) -> Vec<Vec<String>> {
        self.0
            .encoder
            .vocabularies()
            .map(<[String]>::to_vec)
            .collect()
    }

    /// Query rows from an Arrow IPC stream, encoded as `(rows, width)`.
    ///
    /// Words are coded with the context's vocabularies; a word the context
    /// never held is NaN. Raises `RequestError` for a missing column or values of another kind.
    fn encode(&self, py: Python<'_>, queries: &[u8]) -> PyResult<Py<PyArray2<f32>>> {
        let rows = record_batch(queries)?;
        let count = rows.num_rows();
        let encoded = py
            .detach(|| self.0.encoder.encode(&rows))
            .map_err(refused)?;
        matrix(py, encoded, count)
    }
}

/// Binds a tabular predictor's context from its lock's schema and its rows.
///
/// `schema` is the lock's `TableSchema.to_dict()` as JSON, `task` is
/// `classification` or `regression`, and `context` is the rows as an Arrow
/// IPC stream. Only labeled rows are kept. Raises `RequestError`:
/// `EmptyContext` for no labeled rows, `InvalidPredictor` for fewer than two
/// classes or columns that hold values of another kind.
#[pyfunction]
fn bind_predictor(
    py: Python<'_>,
    schema: &[u8],
    task: &str,
    target: &str,
    context: &[u8],
) -> PyResult<PredictorContext> {
    let schema: predictor::Schema = serde_json::from_slice(schema).map_err(invalid)?;
    let task = match task {
        "classification" => catalog::Task::Classification,
        "regression" => catalog::Task::Regression,
        _ => return Err(invalid(format!("a tabular predictor has no task {task:?}"))),
    };
    let rows = record_batch(context)?;
    py.detach(|| predictor::bind(&schema, task, target, &rows))
        .map(PredictorContext)
        .map_err(refused)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bucket_size, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_model, module)?)?;
    module.add_function(wrap_pyfunction!(parse_inference, module)?)?;
    module.add_function(wrap_pyfunction!(quantile_levels, module)?)?;
    module.add_function(wrap_pyfunction!(bind_predictor, module)?)?;
    module.add_class::<PredictorContext>()?;
    module.add_class::<engine::Engine>()?;
    module.add_class::<engine::BoundTable>()?;
    module.add("RequestError", module.py().get_type::<RequestError>())?;
    module.add_class::<TabularRequest>()?;
    module.add_class::<ForecastRequest>()?;
    module.add_class::<ForecastQuery>()?;
    module.add_class::<AlignedGroup>()?;
    module.add("DEFAULT_FIT_ROWS", DEFAULT_FIT_ROWS)?;
    module.add_class::<Strategy>()?;
    module.add_class::<GroupStrategy>()?;
    module.add_class::<Grid>()?;
    module.add_class::<SeriesUnit>()?;
    module.add_class::<Forecast>()?;
    module.add_class::<Node>()?;
    module.add_class::<Step>()?;
    module.add_class::<Fit>()?;
    module.add_class::<Page>()?;
    module.add_class::<Unit>()?;
    module.add_class::<QueryBatch>()?;
    module.add_class::<Movement>()?;
    module.add_class::<CacheKey>()?;
    Ok(())
}
