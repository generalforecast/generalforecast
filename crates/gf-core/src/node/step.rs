// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! What a step carries to the worker, and where planning spent its time.

use super::*;

/// Fit a table's state on these rows: every live row, page by page, which the
/// fit also embeds, or for a table past the rows a fit embeds with its state
/// (see `EntrySizes::fused_rows`), its rows or a sample of them past its
/// effective `fit_rows`, filling no pages.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub table: TableId,
    pub epoch: Epoch,
    /// Every live row, page by page, or the rows a state alone is fit on.
    pub rows: Vec<RowId>,
    /// Row-major, `rows.len()` by the features the model reads, NaN for missing.
    pub features: Vec<f32>,
    pub targets: Targets,
    /// The step's pages these rows fill, in order: their rows, joined, are
    /// `rows`. None for a state alone: its pages are embedded once read.
    pub pages: Vec<StepPage>,
}

/// The representations of one page's rows, which encoding units gather from.
///
/// Pages live on the host: they are read only while a step encodes from them.
/// A changed row changes its page's key, so only its page is computed again,
/// and the page's earlier entries, still cached, serve as sources: the rows
/// they hold that were not written since are copied, and only the rest embed.
#[derive(Clone, Debug, PartialEq)]
pub struct Page {
    pub table: TableId,
    pub key: CacheKey,
    /// The page's live rows, in row order.
    pub rows: Vec<RowId>,
    /// What to do with the page, and the host slots for its representations.
    pub placed: Placed,
    /// Row-major features and targets of the rows to embed, those of `rows`
    /// no source copies, in row order; None when the page is cached or a fit
    /// computes it.
    pub embed: Option<(Vec<f32>, Targets)>,
    /// Cached entries to copy rows from.
    pub copies: Copies,
}

/// Entries a page copies rows from, each with the positions to copy, as
/// (position in the source's rows, position in the page's rows).
pub type Copies = Vec<(CacheKey, Vec<(usize, usize)>)>;

/// A unit of a step: rows whose representations to encode or reuse.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub table: TableId,
    pub key: CacheKey,
    /// The context's rows: by key, the order the model reads them in, for
    /// a unit to encode.
    pub rows: Vec<RowId>,
    /// What to do with the unit, and the device slots for its context.
    pub placed: Placed,
    /// Row-major features and targets of a unit to encode whose table
    /// summarizes each context: its state is fit on them. None otherwise.
    pub fit: Option<(Vec<f32>, Targets)>,
}

/// One request's queries, and the unit each decodes against.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryBatch {
    pub request: RequestId,
    pub table: TableId,
    pub epoch: Epoch,
    /// Row-major, one row per query.
    pub features: Vec<f32>,
    /// The step's unit each query reads.
    pub units: Vec<StepUnit>,
}

/// What a worker runs, in order: movements, fits, pages to embed, units,
/// queries, then series units and forecasts.
///
/// Units that are not reused gather their rows' representations from `pages`,
/// or, for a table that summarizes each context, are fit on their own rows.
/// Only the node makes steps, and it trusts the steps it is handed back, so
/// they are read-only.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    pub(crate) movements: Vec<Movement>,
    pub(crate) fits: Vec<Fit>,
    pub(crate) pages: Vec<Page>,
    pub(crate) units: Vec<Unit>,
    pub(crate) queries: Vec<QueryBatch>,
    pub(crate) series_units: Vec<SeriesUnit>,
    pub(crate) forecasts: Vec<Forecast>,
}

impl Step {
    /// Copies and drops between tiers, to make before anything else.
    #[must_use]
    pub fn movements(&self) -> &[Movement] {
        &self.movements
    }

    /// Tables to fit.
    #[must_use]
    pub fn fits(&self) -> &[Fit] {
        &self.fits
    }

    /// Pages whose rows units read.
    #[must_use]
    pub fn pages(&self) -> &[Page] {
        &self.pages
    }

    /// Units to encode or reuse.
    #[must_use]
    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    /// One of the step's units.
    ///
    /// # Panics
    ///
    /// If the step has no such unit: only its own fits, units and queries name them.
    #[must_use]
    pub fn unit(&self, unit: StepUnit) -> &Unit {
        &self.units[unit.0]
    }

    /// One of the step's pages.
    ///
    /// # Panics
    ///
    /// If the step has no such page: only its own fits name them.
    #[must_use]
    pub fn page(&self, page: StepPage) -> &Page {
        &self.pages[page.0]
    }

    /// Queries to decode, by request.
    #[must_use]
    pub fn queries(&self) -> &[QueryBatch] {
        &self.queries
    }

    /// Groups of series to encode, reuse or forecast whole.
    #[must_use]
    pub fn series_units(&self) -> &[SeriesUnit] {
        &self.series_units
    }

    /// Forecasts to make, each from a series unit.
    #[must_use]
    pub fn forecasts(&self) -> &[Forecast] {
        &self.forecasts
    }

    /// Moves the rows' features and targets out of the step, in order, so an
    /// executor can own them without a copy. What the node reads back when the
    /// step completes or fails stays.
    pub fn take_rows(&mut self) -> StepRows {
        StepRows {
            fits: self
                .fits
                .iter_mut()
                .map(|fit| {
                    let empty = Targets::empty_like_targets(&fit.targets);
                    (
                        std::mem::take(&mut fit.features),
                        std::mem::replace(&mut fit.targets, empty),
                    )
                })
                .collect(),
            pages: self
                .pages
                .iter_mut()
                .map(|page| page.embed.take())
                .collect(),
            units: self.units.iter_mut().map(|unit| unit.fit.take()).collect(),
            queries: self
                .queries
                .iter_mut()
                .map(|batch| std::mem::take(&mut batch.features))
                .collect(),
            series_units: self
                .series_units
                .iter_mut()
                .map(|unit| std::mem::take(&mut unit.grid))
                .collect(),
            forecasts: self
                .forecasts
                .iter_mut()
                .map(|forecast| std::mem::take(&mut forecast.future))
                .collect(),
        }
    }
}

/// A step's row data, as `Step::take_rows` moves it out.
#[derive(Clone, Debug, PartialEq)]
pub struct StepRows {
    /// Each fit's features and targets.
    pub fits: Vec<(Vec<f32>, Targets)>,
    /// Each page's features and targets to embed, if it has any.
    pub pages: Vec<Option<(Vec<f32>, Targets)>>,
    /// Each unit's features and targets to fit, if it has any.
    pub units: Vec<Option<(Vec<f32>, Targets)>>,
    /// Each query batch's features.
    pub queries: Vec<Vec<f32>>,
    /// Each series unit's grid.
    pub series_units: Vec<forecasting::Grid>,
    /// Each forecast's known future.
    pub forecasts: Vec<forecasting::Grid>,
}

/// Where the last `schedule` spent its time, by part: how often each ran and
/// its seconds. The parts cover the plan's time but the table loop between
/// them. `checks` is what a step
/// refuses, waits for or refits, `snapshot` the tables' snapshots, `regions`
/// the queries' assignment to regions, `retrieve` the retrieval of contexts
/// to build, `fit` a table's fit, counted when the step plans one, `units`
/// the placement of contexts in the cache, `pages` the placement of their
/// pages and the copies they take, `series` the whole plan of a series
/// table, and `assemble` the step's queries and bookkeeping. Untimed parts,
/// the default, read no clock and stay empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanParts {
    pub(super) timed: bool,
    pub(super) parts: BTreeMap<&'static str, (u64, Duration)>,
}

impl PlanParts {
    /// No parts yet, timed if `timed`.
    #[must_use]
    pub fn new(timed: bool) -> Self {
        Self {
            timed,
            parts: BTreeMap::new(),
        }
    }

    /// When a part begins, if the parts are timed.
    pub(super) fn start(&self) -> Option<Instant> {
        self.timed.then(Instant::now)
    }

    /// Counts a run of `part` that began at `since`.
    pub(super) fn add(&mut self, part: &'static str, since: Option<Instant>) {
        let Some(since) = since else { return };
        let (calls, seconds) = self.parts.entry(part).or_default();
        *calls += 1;
        *seconds += since.elapsed();
    }

    /// Adds another plan's parts to these.
    pub fn extend(&mut self, other: &Self) {
        for (&part, &(calls, seconds)) in &other.parts {
            let entry = self.parts.entry(part).or_default();
            entry.0 += calls;
            entry.1 += seconds;
        }
    }

    /// Each part, how often it ran and its time, by name.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, u64, Duration)> + '_ {
        self.parts
            .iter()
            .map(|(&part, &(calls, seconds))| (part, calls, seconds))
    }
}
