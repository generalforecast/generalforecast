// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A node: many tables, one request queue, one cache.
//!
//! The node owns each table's rows, so it is the single writer for them. A
//! step carries everything a worker needs, in order: tables to fit, rows to
//! embed since the fit, units to encode or reuse, and queries to decode. Small
//! tables share the node's cache and its steps, so many of them cost what one
//! large table does.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::{Duration, Instant};

use roaring::RoaringBitmap;

use crate::cache::{
    Cache, CacheError, CacheKey, Epoch, Eviction, Kind, ModelId, Movement, Tier, TierConfig,
};
use crate::categories::{CategoryEncoding, Lookup};
use crate::context::{Context, Fingerprint};
use crate::forecasting::{self, Forecast, ForecastError, SeriesRequest, SeriesSpec, SeriesUnit};
use crate::pages::{Bucket, IndexError, IndexLayout, PageId, PageIndex, PageLayout, Vectors};
use crate::planner::{
    Core, CoreKind, Cores, Plan, PlanError, Ranking, Recency, Source, Strata, Strategy,
    StrategyError, UnitId, Windows, draw_core, plan_given, plan_with, recent, smallest,
};
use crate::regions::{Built, Rebuild, RegionSpec, RegionStats, Regions, RegionsError};
use crate::rows::{RowMemory, RowSnapshot, RowStore, RowsError, Sealer};
use crate::series::{SeriesIndex, SeriesKey};
use crate::shapes::bucket_size;
use crate::table::{Change, Keys, ROW_IDS, RowId, RowKey, Snapshot, Table, TableError, Version};
use crate::time_index::TimeIndex;

pub use crate::step::{Action, Placed, RequestId, StepPage, StepUnit};
pub use crate::table::TableId;

/// How much an encoded context and a row representation take, from the
/// model's memory profile, and how many rows a fit takes within the worker's
/// call budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntrySizes {
    pub bytes_per_row: u64,
    /// One row's representation.
    pub bytes_per_representation: u64,
    /// Contexts are padded to `bucket_size(rows, bucket_minimum)` rows.
    pub bucket_minimum: NonZeroUsize,
    /// A unit's own state, preprocessing and column summary, kept in whole
    /// slots after its rows by a table that summarizes each context; 0 for
    /// one that summarizes the table.
    pub state_bytes: u64,
    /// The most rows the worker fits a state on within its call budget, at
    /// this width: a table holding more, or more than its `fit_rows`, fits on
    /// a seeded sample of the fewer.
    pub state_rows: usize,
    /// The most rows a fit embeds in the same call as its state, within the
    /// worker's call budget. A fit on more computes the state alone, and each
    /// page is embedded once a plan reads it.
    pub fused_rows: usize,
    /// The most rows a fit takes before the worker measures `state_rows` and
    /// `fused_rows`, which until then stand unlimited: a table that would
    /// fit on more waits, and `Node::take_unmeasured` names it, until
    /// `Node::set_fit_limits` sets them. `usize::MAX` once they are set.
    pub fit_bound: usize,
}

impl EntrySizes {
    /// Bytes a context of `rows` rows takes: whole slots, each holding as many
    /// padded rows as fit, as the worker packs them, then its state's slots.
    fn bytes(&self, rows: usize, slot_bytes: u64) -> u64 {
        let padded = bucket_size(rows, self.bucket_minimum) as u64;
        let per_row = self.bytes_per_row.max(1);
        let state = self.state_bytes.div_ceil(slot_bytes) * slot_bytes;
        if per_row > slot_bytes {
            // A row wider than a slot takes whole slots of its own.
            return padded * per_row.div_ceil(slot_bytes) * slot_bytes + state;
        }
        padded.div_ceil(slot_bytes / per_row) * slot_bytes + state
    }

    /// Bytes a series unit's cache takes: whole slots of patches, as the
    /// worker packs them, for a group of `members` series of `1 + features`
    /// rows each and `steps` steps.
    ///
    /// Rows pad to their bucket and patches to theirs, within the model's
    /// context; a patch takes `bytes_per_row` per padded row and the spec's
    /// `patch_bytes` beside them.
    fn series_bytes(
        &self,
        series: &SeriesSpec,
        features: usize,
        members: usize,
        steps: usize,
        slot_bytes: u64,
    ) -> u64 {
        let patches = self.series_patches(series, steps);
        self.series_bytes_of(series, features, members, patches, slot_bytes)
    }

    /// The patches a grid of `steps` pads to, as the worker buckets them,
    /// within the model's context.
    fn series_patches(&self, series: &SeriesSpec, steps: usize) -> usize {
        let patch = series.patch.get();
        let limit = (series.context.get() / patch).max(1);
        bucket_size(steps / patch, self.bucket_minimum).min(limit)
    }

    /// Bytes a series unit's cache of exactly `patches` patches takes, as
    /// `series_bytes` sizes a padded grid's, for an extension that holds the
    /// source's patches and the new ones.
    fn series_bytes_of(
        &self,
        series: &SeriesSpec,
        features: usize,
        members: usize,
        patches: usize,
        slot_bytes: u64,
    ) -> u64 {
        let rows = bucket_size(members * (1 + features), self.bucket_minimum) as u64;
        let patches = patches as u64;
        let per_patch = (self.bytes_per_row * rows + series.patch_bytes).max(1);
        if per_patch > slot_bytes {
            return patches * per_patch.div_ceil(slot_bytes) * slot_bytes;
        }
        patches.div_ceil(slot_bytes / per_patch) * slot_bytes
    }
}

/// What embedding one feature of one row costs, in the units of a context's
/// cost (its rows squared). Measured on TabICL v2, where embedding 4,096 rows of
/// 10 features takes about a ninth of encoding them.
const EMBED_COST_PER_CELL: f64 = 40.0;

/// What a table predicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// Class labels `0 .. count`.
    Classes(u32),
    Values,
}

impl Task {
    /// The task a class count names, as clients and logs give it: values for 0.
    #[must_use]
    pub fn of_classes(classes: u32) -> Self {
        if classes == 0 {
            Self::Values
        } else {
            Self::Classes(classes)
        }
    }

    /// The class count, or 0 for values.
    #[must_use]
    pub fn classes(self) -> u32 {
        match self {
            Self::Classes(classes) => classes,
            Self::Values => 0,
        }
    }
}

/// Targets for a batch of rows, matching the table's task.
#[derive(Clone, Debug, PartialEq)]
pub enum Targets {
    Classes(Vec<i32>),
    Values(Vec<f32>),
}

/// The class target of a row whose label is not known yet. A value target
/// not known yet is NaN.
pub const UNLABELED: i32 = -1;

impl Targets {
    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Classes(labels) => labels.len(),
            Self::Values(values) => values.len(),
        }
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the target at `index` has a value: a class always, a value
    /// unless it is NaN, which a series table stores for a step without one.
    #[must_use]
    pub fn observed(&self, index: usize) -> bool {
        match self {
            Self::Classes(_) => true,
            Self::Values(values) => values[index].is_finite(),
        }
    }

    /// Each target's bits, as rows store them.
    fn bits(&self) -> Vec<u32> {
        match self {
            Self::Classes(labels) => labels.iter().map(|&label| label as u32).collect(),
            Self::Values(values) => values.iter().map(|value| value.to_bits()).collect(),
        }
    }

    /// The targets `keep` marks.
    fn picked(&self, keep: &[bool]) -> Self {
        let kept = |index: usize| keep[index];
        match self {
            Self::Classes(labels) => Self::Classes(
                (0..labels.len())
                    .filter(|&i| kept(i))
                    .map(|i| labels[i])
                    .collect(),
            ),
            Self::Values(values) => Self::Values(
                (0..values.len())
                    .filter(|&i| kept(i))
                    .map(|i| values[i])
                    .collect(),
            ),
        }
    }

    /// Whether each target is known: a class from 0, or a finite value.
    fn labeled(&self) -> Vec<bool> {
        match self {
            Self::Classes(labels) => labels.iter().map(|&label| label >= 0).collect(),
            Self::Values(values) => values.iter().map(|value| value.is_finite()).collect(),
        }
    }

    /// Targets for the task from the bits rows store.
    fn from_bits(task: Task, bits: &[u32]) -> Self {
        match task {
            Task::Classes(_) => Self::Classes(bits.iter().map(|&bits| bits as i32).collect()),
            Task::Values => Self::Values(bits.iter().map(|&bits| f32::from_bits(bits)).collect()),
        }
    }

    /// No targets, of the same kind as `targets`.
    fn empty_like_targets(targets: &Self) -> Self {
        match targets {
            Self::Classes(_) => Self::Classes(Vec::new()),
            Self::Values(_) => Self::Values(Vec::new()),
        }
    }

    /// The targets at `indices` of a batch, in that order.
    fn pick(&self, indices: &[usize]) -> Self {
        match self {
            Self::Classes(labels) => Self::Classes(indices.iter().map(|&i| labels[i]).collect()),
            Self::Values(values) => Self::Values(indices.iter().map(|&i| values[i]).collect()),
        }
    }
}

/// What a table's column summary, and the rest of its state, is fit on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SummaryScope {
    /// The table's rows, once per fit, so every row embeds against one state
    /// and pages cache row representations.
    #[default]
    Table,
    /// Each context's own rows, as upstream packages fit each context: a unit
    /// carries its rows' features, and the unit's own state, preprocessing
    /// and column summary, is fit on them, kept beside its rows, and read by
    /// its queries. The worker fits no table and embeds no pages, and drift
    /// and cutoffs refit nothing, since a unit depends on its rows alone.
    Context,
}

impl SummaryScope {
    /// The scope's name, as clients and workers spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Context => "context",
        }
    }

    /// The scope a name spells, if any.
    #[must_use]
    pub fn of_name(name: &str) -> Option<Self> {
        [Self::Table, Self::Context]
            .into_iter()
            .find(|scope| scope.name() == name)
    }
}

/// When a table's state is refit on its own, besides explicit refits; none by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefitPolicy {
    changed: Option<f64>,
    shift: Option<f64>,
    cutoffs: CutoffRefit,
}

/// When a query's time cutoff inside the table's history calls for a state
/// fit on other rows than the state it would read. A state fit on a row at
/// or after the cutoff always refits, whatever the setting, so no query reads
/// a later row. A cutoff after every row, as a live stream's are, reads the
/// state fit on every row, which only the policy's changed share or shift
/// fits again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CutoffRefit {
    /// At each new cutoff, on the rows before it, and back on every row for
    /// a query without one, so alternating them refits each time.
    #[default]
    Each,
    /// Once rows between the state's horizon and the cutoff reach the
    /// policy's changed share of the rows it was fit on.
    Changed,
    /// Never, unless the state was fit on a row from at or after the cutoff;
    /// queries without a cutoff then keep the state fit for it until the
    /// policy's changed share or shift refits.
    Never,
}

impl CutoffRefit {
    /// The setting's name, as clients and logs spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Each => "each",
            Self::Changed => "changed",
            Self::Never => "never",
        }
    }
}

impl std::str::FromStr for CutoffRefit {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        [Self::Each, Self::Changed, Self::Never]
            .into_iter()
            .find(|cutoffs| cutoffs.name() == name)
            .ok_or_else(|| format!("refit_cutoffs is each, changed or never, not {name:?}"))
    }
}

impl RefitPolicy {
    /// A policy that refits once rows inserted, updated or deleted since the
    /// fit reach `changed`, a fraction of the rows the state was fit on, or
    /// once a column's mean moves by more than `shift` of its standard
    /// deviations at the fit.
    ///
    /// # Errors
    ///
    /// `RefitPolicy` if a limit given is not positive and finite.
    pub fn new(changed: Option<f64>, shift: Option<f64>) -> Result<Self, NodeError> {
        let positive = [changed, shift]
            .into_iter()
            .flatten()
            .all(|limit| limit.is_finite() && limit > 0.0);
        if positive {
            Ok(Self {
                changed,
                shift,
                cutoffs: CutoffRefit::Each,
            })
        } else {
            Err(NodeError::RefitPolicy)
        }
    }

    /// The policy with `cutoffs` deciding when a cutoff refits.
    ///
    /// # Errors
    ///
    /// `RefitPolicy` for `CutoffRefit::Changed` without a changed share.
    pub fn with_cutoffs(self, cutoffs: CutoffRefit) -> Result<Self, NodeError> {
        if cutoffs == CutoffRefit::Changed && self.changed.is_none() {
            return Err(NodeError::RefitPolicy);
        }
        Ok(Self { cutoffs, ..self })
    }

    /// When a query's cutoff calls for a new state.
    #[must_use]
    pub fn cutoffs(&self) -> CutoffRefit {
        self.cutoffs
    }

    /// The fraction of changed rows that refits, if any.
    #[must_use]
    pub fn changed(&self) -> Option<f64> {
        self.changed
    }

    /// The shift of a column's mean, in deviations, that refits, if any.
    #[must_use]
    pub fn shift(&self) -> Option<f64> {
        self.shift
    }
}

/// What a table does with a context larger than its model was trained on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PastCap {
    /// Read it whole, and say so once; see `Node::take_past_cap`.
    #[default]
    Warn,
    /// Cut it to the trained length: every row becomes a random sample of
    /// that many under a fixed seed, and a retrieved context keeps at most
    /// that many of its core, under its own seed, and fewer neighbors. A
    /// table wanting samples of other seeds subsamples to the length itself.
    /// Partitions would each cut their own share, so a table of several
    /// takes no clamp.
    Clamp,
}

impl PastCap {
    /// The setting's name, as clients and logs spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Warn => "warn",
            Self::Clamp => "clamp",
        }
    }
}

impl std::str::FromStr for PastCap {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        [Self::Warn, Self::Clamp]
            .into_iter()
            .find(|past| past.name() == name)
            .ok_or_else(|| format!("past_cap is warn or clamp, not {name:?}"))
    }
}

/// How a table is served.
#[derive(Clone, Debug, PartialEq)]
pub struct TableSpec {
    pub model: ModelId,
    /// Values in each row: the model's features, then `retrieval` values.
    pub features: usize,
    /// Values at the end of each row that place it for retrieval, which the
    /// model never reads: its index space is then exactly these, unscaled,
    /// and otherwise the standardized features.
    pub retrieval: usize,
    /// The model features that hold category codes, in order.
    pub categorical: Vec<usize>,
    /// How those features place rows for retrieval, for a table without
    /// retrieval values.
    pub categories: CategoryEncoding,
    /// Whether the model reads context rows whose target is not known yet,
    /// through a mask; otherwise such rows wait out of every fit and context
    /// until they are labeled.
    pub unlabeled_context: bool,
    pub task: Task,
    pub page_size: NonZeroUsize,
    pub strategy: Strategy,
    pub sizes: EntrySizes,
    /// A window of time pages never straddle, if rows are grouped by time first.
    /// Pages wholly before a cutoff then keep their rows, and their caches, as
    /// later rows arrive.
    pub time_bucket: Option<NonZeroU64>,
    pub refit: RefitPolicy,
    /// Rows the retrieval scaling may be fit on before it stays as it is, if
    /// set; otherwise it is fit again each time the rows double, up to
    /// `fit_rows`. Each new scaling cuts every page anew and redraws regions.
    pub scale_rows: Option<NonZeroU64>,
    /// The most rows the table's state is fit on, as asked. A table holding
    /// more fits on a seeded sample of this many, or of fewer where the
    /// worker's call budget takes fewer (see `effective_fit_rows`), and embeds
    /// each page only once a plan reads it, so a fit's cost stays bounded
    /// however large the table grows.
    pub fit_rows: NonZeroUsize,
    /// The most context rows the model was trained on, if it says.
    pub context_rows: Option<NonZeroU64>,
    /// What a context past `context_rows` does.
    pub past_cap: PastCap,
    /// Regions whose queries share a context, rebuilt on churn; without
    /// them, each query reads a context planned for it alone.
    pub regions: Option<RegionSpec>,
    /// What the column summary is fit on: the table, or each context.
    pub summary: SummaryScope,
    /// How the table's series are forecast, for a table of series: its rows
    /// are observations keyed by series and time, it is never fit, and it
    /// takes forecast requests in place of query rows.
    pub series: Option<SeriesSpec>,
}

/// Rows a table's state is fit on unless it says otherwise: about as many as
/// a fit of every row could take before, so tables up to it fit as before.
pub const DEFAULT_FIT_ROWS: usize = 1 << 20;

/// Bins a regression table's targets fall in for stratified and balanced cores.
const TARGET_BINS: usize = 8;

/// Seeds the sample a table larger than its `fit_rows` is fit on.
const FIT_SEED: u64 = 0x6669_745f_726f_7773;

impl TableSpec {
    /// The values of each row the model reads.
    #[must_use]
    pub fn model_features(&self) -> usize {
        self.features - self.retrieval
    }

    /// The most rows the table's state is fit on: its `fit_rows`, or fewer
    /// where the worker's call budget takes fewer at this width.
    #[must_use]
    pub fn effective_fit_rows(&self) -> usize {
        self.fit_rows.get().min(self.sizes.state_rows.max(1))
    }

    /// Dimensions of the retrieval space: the retrieval values, or the
    /// features with each categorical one as wide as its encoding.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        if self.retrieval > 0 {
            return self.retrieval;
        }
        let categorical = self.categorical.len();
        self.features - categorical + categorical * self.categories.width(self.task)
    }

    /// Row-major `rows` with only the values the model reads.
    fn model_values(&self, rows: &[f32]) -> Vec<f32> {
        if self.retrieval == 0 {
            return rows.to_vec();
        }
        rows.chunks(self.features)
            .flat_map(|row| &row[..self.model_features()])
            .copied()
            .collect()
    }

    /// The bucket a row at `time` belongs to; rows without a time share the first.
    fn bucket(&self, time: Option<i64>) -> Bucket {
        match (self.time_bucket, time) {
            (Some(width), Some(time)) => {
                let window = time.div_euclid(i64::try_from(width.get()).unwrap_or(i64::MAX));
                // Flipping the sign bit keeps windows in order as unsigned numbers.
                Bucket((window as u64) ^ (1 << 63))
            }
            _ => Bucket(0),
        }
    }
}

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
    timed: bool,
    parts: BTreeMap<&'static str, (u64, Duration)>,
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
    fn start(&self) -> Option<Instant> {
        self.timed.then(Instant::now)
    }

    /// Counts a run of `part` that began at `since`.
    fn add(&mut self, part: &'static str, since: Option<Instant>) {
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

/// Why a change or request was refused. The node is left unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeError {
    UnknownTable(TableId),
    /// A table was given no feature columns.
    NoFeatures,
    /// A retrieval vector was requested before its standardizer was fitted.
    Unscaled,
    Strategy(StrategyError),
    /// Queries were sent to a table with no rows to learn from.
    EmptyTable(TableId),
    /// A query's time cutoff precedes every row of the table.
    EmptyContext(TableId),
    /// The table's strategy keeps to a span of time, and no row has a time.
    Untimed(TableId),
    UnknownKey(RowKey),
    DuplicateKey(RowKey),
    /// A batch has a different number of `what` (such as targets) than keys.
    Count {
        what: Counted,
        expected: usize,
        got: usize,
    },
    /// A refit policy limit that is not positive and finite.
    RefitPolicy,
    /// Rows with a different number of columns than the table.
    Width {
        expected: usize,
        got: usize,
    },
    /// A batch of queries with no rows.
    NoQueries,
    /// A class label outside the table's classes.
    Label {
        label: i32,
        classes: u32,
    },
    /// Targets of the wrong kind for the table's task.
    Task,
    /// A compacted log's rows or layout that do not rebuild a table.
    Restore(&'static str),
    /// A given context that is empty or names rows the table does not hold.
    Given,
    /// Rows that could not be read or written.
    Rows(RowsError),
    Table(TableError),
    Index(IndexError),
    Plan(PlanError),
    Cache(CacheError),
    Regions(RegionsError),
    Forecast(ForecastError),
    /// A table's categorical features or their encoding do not fit it.
    Categories(&'static str),
}

impl fmt::Display for NodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTable(table) => write!(formatter, "table {} does not exist", table.0),
            Self::NoFeatures => write!(formatter, "a table needs at least one feature column"),
            Self::Unscaled => write!(formatter, "the table has no fitted retrieval space"),
            Self::Strategy(error) => error.fmt(formatter),
            Self::EmptyTable(_) => write!(
                formatter,
                "the table has no rows, or none labeled, to predict from"
            ),
            Self::EmptyContext(_) => write!(
                formatter,
                "a query's time cutoff precedes every row of the table, so it has no context"
            ),
            Self::Untimed(_) => write!(
                formatter,
                "the table's strategy keeps to a span of time, but none of its rows has a \
                 time, so a query has no context"
            ),
            Self::UnknownKey(key) => write!(formatter, "row key {key} does not exist"),
            Self::DuplicateKey(key) => write!(formatter, "row key {key} appears twice"),
            Self::Count {
                what,
                expected,
                got,
            } => write!(formatter, "expected {expected} {what}, got {got}"),
            Self::Width { expected, got } => write!(
                formatter,
                "rows have {got} columns, but the table has {expected} feature columns"
            ),
            Self::NoQueries => write!(formatter, "there are no query rows"),
            Self::RefitPolicy => write!(
                formatter,
                "a refit policy's limits must be positive and finite, and \
                 refit_cutoffs = \"changed\" needs refit_after"
            ),
            Self::Label { label, classes } => {
                write!(
                    formatter,
                    "label {label} is outside 0 .. {}",
                    classes.saturating_sub(1)
                )
            }
            Self::Task => write!(formatter, "targets do not match the table's task"),
            Self::Restore(why) => write!(formatter, "the table cannot be restored: {why}"),
            Self::Given => write!(formatter, "each given context holds rows of the table"),
            Self::Rows(error) => error.fmt(formatter),
            Self::Table(error) => error.fmt(formatter),
            Self::Index(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
            Self::Cache(error) => error.fmt(formatter),
            Self::Regions(error) => error.fmt(formatter),
            Self::Forecast(error) => error.fmt(formatter),
            Self::Categories(why) => write!(formatter, "{why}"),
        }
    }
}

impl std::error::Error for NodeError {}

/// What kind of failure an error is, which servers and bindings report alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A table or key that does not exist.
    NotFound,
    /// Input the node refuses, such as rows of the wrong width.
    Invalid,
    /// A fault of the node's own, such as its cache refusing an operation.
    Internal,
}

impl NodeError {
    /// What kind of failure the error is.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::UnknownTable(_)
            | Self::UnknownKey(_)
            | Self::Forecast(ForecastError::UnknownMember(_))
            | Self::Table(TableError::UnknownKey(_)) => ErrorKind::NotFound,
            Self::Index(error) | Self::Plan(PlanError::Index(error)) => match error {
                IndexError::Dimensions { .. } | IndexError::NotFinite => ErrorKind::Invalid,
                IndexError::Duplicate(_)
                | IndexError::Unknown(_)
                | IndexError::UnknownPage(_)
                | IndexError::Rows(_) => ErrorKind::Internal,
            },
            Self::Cache(_)
            | Self::Plan(_)
            | Self::Restore(_)
            | Self::Rows(_)
            | Self::Regions(RegionsError::Rows(_)) => ErrorKind::Internal,
            _ => ErrorKind::Invalid,
        }
    }
}

/// What a batch counts one of per key, or per query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counted {
    Targets,
    Times,
    Cutoffs,
    Groups,
    Rows,
    Contexts,
}

impl fmt::Display for Counted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Targets => "targets",
            Self::Times => "times",
            Self::Cutoffs => "cutoffs",
            Self::Groups => "groups",
            Self::Rows => "rows of features",
            Self::Contexts => "contexts",
        })
    }
}

/// Every live row of a table, in key order.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRows {
    pub keys: Vec<u128>,
    /// Row-major, one row per key.
    pub features: Vec<f32>,
    pub targets: Targets,
    /// Each row's time, or None for a row without one.
    pub times: Vec<Option<i64>>,
}

/// A row a table offers for a query's context, and how it ranks there, so
/// contexts can be assembled from the rows of several partitions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The query it is offered for, or None for a row every query shares,
    /// such as one of a sample's.
    pub query: Option<u32>,
    pub row: RowId,
    /// Its squared distance from the query in the table's standardized
    /// space; 0 for a shared row.
    pub score: f64,
}

/// What a table holds besides its rows and pages, as a compacted log records
/// it, so a copy rebuilt from the log serves the table as the original did:
/// rows added later land on the same pages, and refits come at the same writes.
#[derive(Clone, Debug, PartialEq)]
pub struct TableLayout {
    /// Live rows the compaction's records hold, in all.
    pub rows: u64,
    /// Row ids taken, live or since deleted.
    pub ids: u32,
    pub index: IndexLayout,
    /// The retrieval space's scaling, if the table has had rows.
    pub scaling: Option<Scaling>,
    /// Running sums and counts of each column's finite values over live rows.
    pub column_sums: Vec<f64>,
    pub column_counts: Vec<u64>,
    /// The columns' moments when the state was last fit, if it has been.
    pub fitted: Option<FittedColumns>,
    /// Rows inserted, updated or deleted since the state was last fit.
    pub changed: u64,
}

/// The center and scale of each column in the retrieval space, and the live
/// rows they were fit on.
#[derive(Clone, Debug, PartialEq)]
pub struct Scaling {
    pub center: Vec<f32>,
    pub scale: Vec<f32>,
    /// Each categorical feature's codes and vectors, if they are encoded.
    pub lookups: Vec<(Vec<f32>, Vec<f32>)>,
    pub rows: u64,
}

/// The columns' means and deviations when a table's state was fit, and the
/// live rows it then had.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedColumns {
    pub rows: u64,
    pub means: Vec<f64>,
    pub deviations: Vec<f64>,
}

/// Rows of a compacted table, in page order: each row's id and page, and the
/// layout of each page that begins among them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowLayout {
    pub ids: Vec<RowId>,
    pub pages: Vec<PageId>,
    pub new_pages: Vec<PageLayout>,
}

/// What a compaction reads, taken at once and cheaply: the table's layout,
/// each live row's id, page and time in page order, and a snapshot of its
/// rows, which reads them as they were while the table goes on changing.
#[derive(Clone, Debug)]
pub struct CompactionSource {
    pub table: TableLayout,
    pub layout: RowLayout,
    pub times: Vec<Option<i64>>,
    pub rows: RowSnapshot,
    pub task: Task,
}

impl CompactionSource {
    /// The rows of positions `start..end` in page order, with their layout:
    /// the pages that begin among them, as a log splits a compaction into records.
    ///
    /// # Errors
    ///
    /// If a row cannot be read.
    pub fn part(&self, start: usize, end: usize) -> Result<(TableRows, RowLayout), RowsError> {
        let ids = &self.layout.ids[start..end];
        let stored = self.rows.rows(ids)?;
        let bits: Vec<u32> = stored.iter().map(|row| row.target).collect();
        let rows = TableRows {
            keys: stored.iter().map(|row| row.key).collect(),
            features: stored
                .iter()
                .flat_map(|row| row.features.iter().copied())
                .collect(),
            targets: Targets::from_bits(self.task, &bits),
            times: self.times[start..end].to_vec(),
        };
        let pages = &self.layout.pages[start..end];
        let begun: HashSet<PageId> = (start..end)
            .filter(|&position| {
                position == 0 || self.layout.pages[position - 1] != self.layout.pages[position]
            })
            .map(|position| self.layout.pages[position])
            .collect();
        let layout = RowLayout {
            ids: ids.to_vec(),
            pages: pages.to_vec(),
            new_pages: self
                .layout
                .new_pages
                .iter()
                .filter(|page| begun.contains(&page.id))
                .cloned()
                .collect(),
        };
        Ok((rows, layout))
    }
}

/// A table as a compaction writes it: every live row in page order, with
/// each row's id and page, and what the table holds besides.
#[derive(Clone, Debug, PartialEq)]
pub struct Compaction {
    pub table: TableLayout,
    pub rows: TableRows,
    pub layout: RowLayout,
}

/// A kept series unit a later one of the same group may extend: its cutoff,
/// grid length, the patches its entry holds and the entry.
#[derive(Clone, Copy, Debug)]
struct Recent {
    cutoff: i64,
    steps: usize,
    /// Patches the worker stored: the padded grid's for an encoding, the
    /// source's plus the new ones for an extension.
    patches: usize,
    key: CacheKey,
}

/// Groups whose latest kept units a table remembers before it prunes them:
/// past this many, and again each time they double, those whose entries are
/// gone are forgotten.
const RECENT_PRUNE: usize = 4096;

/// A table being rebuilt from a compacted log, which takes its requests only
/// once every row is back.
#[derive(Debug)]
struct Restoring {
    layout: TableLayout,
    pages: HashMap<PageId, PageLayout>,
    members: BTreeMap<PageId, Vec<RowId>>,
    restored: u64,
}

impl From<RowsError> for NodeError {
    fn from(error: RowsError) -> Self {
        Self::Rows(error)
    }
}

impl From<TableError> for NodeError {
    fn from(error: TableError) -> Self {
        Self::Table(error)
    }
}

impl From<IndexError> for NodeError {
    fn from(error: IndexError) -> Self {
        Self::Index(error)
    }
}

impl From<PlanError> for NodeError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<StrategyError> for NodeError {
    fn from(error: StrategyError) -> Self {
        Self::Strategy(error)
    }
}

impl From<ForecastError> for NodeError {
    fn from(error: ForecastError) -> Self {
        Self::Forecast(error)
    }
}

impl From<CacheError> for NodeError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

/// Per-column standardization for retrieval vectors, fit on a table's first
/// batch and again each time its rows double and at each refit.
#[derive(Clone, Debug)]
struct Standardizer {
    /// Leading values of a row it leaves out: the model's, when the table
    /// has retrieval values.
    skip: usize,
    center: Vec<f32>,
    scale: Vec<f32>,
    /// Each encoded categorical feature's lookup, in feature order; none
    /// when categoricals keep their codes.
    lookups: Vec<(usize, Lookup)>,
}

/// Each column's mean and standard deviation over its finite values, in one
/// pass over the rows, as Welford's update keeps them; 0 for a column without any.
fn moments(features: &[f32], width: usize) -> (Vec<f64>, Vec<f64>) {
    let mut counts = vec![0u64; width];
    let mut means = vec![0.0; width];
    let mut squares = vec![0.0; width];
    for row in features.chunks_exact(width.max(1)) {
        for (column, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let value = f64::from(value);
            counts[column] += 1;
            let delta = value - means[column];
            means[column] += delta / counts[column] as f64;
            squares[column] += delta * (value - means[column]);
        }
    }
    let deviations = squares
        .iter()
        .zip(&counts)
        .map(|(&squares, &count)| {
            if count == 0 {
                0.0
            } else {
                (squares / count as f64).sqrt()
            }
        })
        .collect();
    (means, deviations)
}

impl Standardizer {
    /// Standardization of `spec`'s features over row-major `features`, with
    /// its categorical features encoded from them and their rows' `targets`,
    /// or for a table with retrieval values, those values as they are.
    fn fit(features: &[f32], targets: &Targets, spec: &TableSpec) -> Self {
        if spec.retrieval > 0 {
            let ones = vec![1.0; spec.retrieval];
            return Self::of(spec, vec![0.0; spec.retrieval], ones, Vec::new());
        }
        let (means, deviations) = moments(features, spec.features);
        let lookups = match spec.categories {
            CategoryEncoding::Codes => Vec::new(),
            encoding => spec
                .categorical
                .iter()
                .map(|&feature| {
                    let values: Vec<f32> = features
                        .chunks(spec.features)
                        .map(|row| row[feature])
                        .collect();
                    (feature, Lookup::fit(encoding, &values, targets, spec.task))
                })
                .collect(),
        };
        Self {
            skip: 0,
            center: means.iter().map(|&mean| mean as f32).collect(),
            scale: deviations
                .iter()
                .map(|&deviation| {
                    if deviation > 0.0 {
                        deviation as f32
                    } else {
                        1.0
                    }
                })
                .collect(),
            lookups,
        }
    }

    /// A standardization of `spec`'s rows by `center`, `scale` and `lookups`.
    fn of(
        spec: &TableSpec,
        center: Vec<f32>,
        scale: Vec<f32>,
        lookups: Vec<(usize, Lookup)>,
    ) -> Self {
        Self {
            skip: spec.model_features() * usize::from(spec.retrieval > 0),
            center,
            scale,
            lookups,
        }
    }

    /// A row's retrieval vector: standardized, with missing values at 0, and
    /// each encoded categorical feature's block in its place.
    fn vector(&self, row: &[f32]) -> Vec<f32> {
        let mut lookups = self.lookups.iter().peekable();
        let mut vector = Vec::with_capacity(row.len() - self.skip);
        let values = row[self.skip..]
            .iter()
            .zip(self.center.iter().zip(&self.scale));
        for (feature, (&value, (center, scale))) in values.enumerate() {
            if let Some((_, lookup)) = lookups.next_if(|(at, _)| *at == feature) {
                lookup.extend(value, &mut vector);
                continue;
            }
            let standardized = (value - center) / scale;
            vector.push(if standardized.is_finite() {
                standardized
            } else {
                0.0
            });
        }
        vector
    }
}

/// A table's rows in its retrieval space, which its index reads its vectors from.
struct Scaled<'a> {
    rows: &'a RowStore,
    standardizer: Option<&'a Standardizer>,
}

impl Vectors for Scaled<'_> {
    fn vector(&self, row: RowId) -> Result<Vec<f32>, RowsError> {
        let features = self.rows.features(row)?;
        Ok(self
            .standardizer
            .expect("rows are scaled before they are placed")
            .vector(&features))
    }
}

/// Rows' strata: a row's class, or the bin among `edges` its value falls in.
struct Labels<'a> {
    rows: &'a RowStore,
    task: Task,
    edges: &'a [f32],
}

impl Strata for Labels<'_> {
    fn stratum(&self, row: RowId) -> u32 {
        let bits = self.rows.target(row).unwrap_or(0);
        match self.task {
            Task::Classes(_) => bits,
            Task::Values => {
                let value = f32::from_bits(bits);
                self.edges.partition_point(|&edge| edge <= value) as u32
            }
        }
    }
}

/// Per-column sums over live rows' finite values, for each column's mean.
///
/// Recounted from the rows at each fit of every row, so the rounding of
/// adding and removing values never builds up past one such fit.
#[derive(Debug)]
struct Columns {
    sums: Vec<f64>,
    counts: Vec<u64>,
}

impl Columns {
    fn new(width: usize) -> Self {
        Self {
            sums: vec![0.0; width],
            counts: vec![0; width],
        }
    }

    /// The sums of row-major `features`.
    fn of(features: &[f32], width: usize) -> Self {
        let mut columns = Self::new(width);
        for row in features.chunks(width) {
            columns.add(row, 1.0);
        }
        columns
    }

    /// Adds a row's model values, or with `sign` -1 removes them.
    fn add(&mut self, row: &[f32], sign: f64) {
        for (column, &value) in row.iter().take(self.sums.len()).enumerate() {
            if value.is_finite() {
                self.sums[column] += sign * f64::from(value);
                self.counts[column] = if sign > 0.0 {
                    self.counts[column] + 1
                } else {
                    self.counts[column].saturating_sub(1)
                };
            }
        }
    }

    fn mean(&self, column: usize) -> f64 {
        self.sums[column] / self.counts[column].max(1) as f64
    }
}

/// What drift is measured against: the state's rows and columns at its fit.
#[derive(Debug)]
struct Fitted {
    /// The served rows the fit drew on: those it read, or for a fit on a
    /// sample, those the sample was drawn from; labeled ones, and unlabeled
    /// ones for a model that reads them.
    rows: u64,
    means: Vec<f64>,
    deviations: Vec<f64>,
}

#[derive(Debug)]
struct TableState {
    spec: TableSpec,
    /// The feature count, which is positive.
    dimensions: NonZeroUsize,
    table: Table,
    index: PageIndex,
    epoch: Epoch,
    /// Every row id's features, target and key, live or since deleted.
    rows: RowStore,
    standardizer: Option<Standardizer>,
    /// Live rows when the standardizer was fit; it is fit again once they double.
    scaled_rows: u64,
    /// The epoch whose fit has been scheduled.
    fitted: Option<Epoch>,
    /// Each row's time; rows inserted without one are before every cutoff.
    times: TimeIndex,
    /// Each series' rows by time, for a table of series.
    series: Option<SeriesIndex>,
    /// Forecast requests, for a table of series.
    forecasts: Vec<(RequestId, SeriesRequest)>,
    /// The latest kept unit of each group of series, by the group's print:
    /// what a unit of the group at a later cutoff may extend. Only units a
    /// step has written are here, so an extension never reads an entry a
    /// step in flight is still encoding, nor one a failed step forgot.
    recent: HashMap<Fingerprint, Recent>,
    /// Kept units of groups a step in flight writes, which join `recent` as
    /// their step completes.
    writing: HashMap<CacheKey, (Fingerprint, Recent)>,
    /// How many groups `recent` held after it was last pruned.
    recent_pruned: usize,
    /// The fingerprint of the rows plans read at each time, rows without one first
    /// (None sorts first), so a window before a cutoff is fingerprinted by a
    /// sum over times rather than over rows.
    prints: BTreeMap<Option<i64>, Fingerprint>,
    /// The rows plans read ranked by the strategy's core seed, if it has a core.
    ranking: Option<Ranking>,
    /// Live rows whose target is not known yet.
    unlabeled: RoaringBitmap,
    /// The quantiles a regression table's targets are binned by for
    /// stratified and balanced cores, set once per state, at its first plan.
    edges: Option<(Epoch, Vec<f32>)>,
    /// Diverse cores drawn for the state of an epoch.
    cores: (Epoch, Cores),
    /// Running per-column sums over live rows, for drift.
    columns: Columns,
    /// The columns' means and deviations when the state was last fit.
    fitted_columns: Option<Fitted>,
    /// Writes to rows plans read, before or after the write, since the state
    /// was last fit, so they count in `Fitted::rows`' units.
    changed: u64,
    /// Requests: their queries, and each query's cutoff if it has one.
    pending: Vec<(RequestId, Vec<f32>, Option<Vec<i64>>)>,
    /// Requests whose cutoffs need a state fit on other rows than `pending`'s,
    /// which wait until those are served.
    later: Vec<(RequestId, Vec<f32>, Option<Vec<i64>>)>,
    /// The rows the state was fit on: those before this time, or every row.
    horizon: Option<i64>,
    /// The latest time of any row the state was fit on, if any had one. Rows
    /// may since have gone or moved, so a cutoff at or before it refits.
    fitted_latest: Option<i64>,
    /// The context of each query of its one pending request, if they were given.
    given: Option<Vec<Context>>,
    /// Requests with given contexts waiting for the table's pending requests
    /// to be planned; they go before requests that wait for another state.
    givens: std::collections::VecDeque<(RequestId, Vec<f32>, Vec<Context>)>,
    /// Whether the last step served given requests, whose turn is then over.
    gave: bool,
    /// Counts changes to the table or its requests, so a plan made for them
    /// is known to still hold.
    generation: u64,
    /// The plan of requests the step budget left waiting, and the generation
    /// it was made at, so a later step need not plan them again.
    deferred: Option<(u64, Plan)>,
    /// A rebuild from a compacted log under way, if one is.
    restoring: Option<Restoring>,
    /// Whether a plan read a context past the model's trained length.
    past_cap: bool,
    /// The regions of the state queries last read, if the table has them.
    regions: Option<Regions>,
    region_stats: RegionStats,
    /// Regions drawn so far, which numbers each draw.
    region_draws: u64,
    /// Queries of the latest plan of the pending requests, counted into
    /// `region_stats` once it joins a step.
    planned_counts: RegionStats,
    /// The generation regions' contexts were last aged at.
    aged_at: Option<u64>,
    /// Request steps taken since a region context fell due, without a
    /// rebuild step.
    rebuild_waited: u32,
    /// Whether the table holds a context one request brought; see
    /// `Node::set_single_use`.
    single_use: bool,
}

/// Dead row ids past which, and past the live rows, a table is renumbered
/// before its next insert, so ids and the arrays they index stay near the
/// live rows however many rows come and go.
const SPARSE_ROWS: u64 = 1 << 12;

impl TableState {
    /// Whether the table holds more dead row ids than live rows, and enough
    /// to be worth renumbering.
    /// Empties the table to restore it from a compacted log's layout: every
    /// row id it took, none live, and its scaling and refit state as recorded.
    fn begin_restore(&mut self, layout: TableLayout) -> Result<(), NodeError> {
        self.table = Table::with_ids(layout.ids);
        self.index = PageIndex::new(self.dimensions, self.spec.page_size);
        self.rows.clear()?;
        self.rows.reserve(layout.ids)?;
        self.times = TimeIndex::new();
        self.reindex_series();
        self.prints = BTreeMap::new();
        self.ranking = self
            .ranking
            .as_ref()
            .map(|ranking| Ranking::empty(ranking.seed()));
        if let Some(scaling) = &layout.scaling {
            let encoded =
                self.spec.retrieval == 0 && self.spec.categories != CategoryEncoding::Codes;
            let lookups = if encoded {
                self.spec.categorical.len()
            } else {
                0
            };
            let width = if self.spec.retrieval > 0 {
                self.spec.retrieval
            } else {
                self.spec.features
            };
            let block = self.spec.categories.width(self.spec.task);
            let shaped = scaling.lookups.iter().all(|(codes, vectors)| {
                vectors.len() == (codes.len() + 1) * block
                    && codes.iter().all(|code| code.is_finite() && *code >= 0.0)
            });
            if scaling.center.len() != width
                || scaling.scale.len() != width
                || scaling.lookups.len() != lookups
                || !shaped
            {
                return Err(NodeError::Restore(
                    "a layout's scaling differs from the table's retrieval space",
                ));
            }
        }
        self.standardizer = layout.scaling.as_ref().map(|scaling| {
            let lookups = self
                .spec
                .categorical
                .iter()
                .zip(&scaling.lookups)
                .map(|(&feature, (codes, vectors))| {
                    let lookup = Lookup::new(self.spec.categories, codes.clone(), vectors.clone());
                    (feature, lookup)
                })
                .collect();
            Standardizer::of(
                &self.spec,
                scaling.center.clone(),
                scaling.scale.clone(),
                lookups,
            )
        });
        self.scaled_rows = layout.scaling.as_ref().map_or(0, |scaling| scaling.rows);
        self.columns = Columns {
            sums: layout.column_sums.clone(),
            counts: layout.column_counts.clone(),
        };
        self.fitted_columns = layout.fitted.as_ref().map(|fitted| Fitted {
            rows: fitted.rows,
            means: fitted.means.clone(),
            deviations: fitted.deviations.clone(),
        });
        self.changed = layout.changed;
        self.fitted = None;
        self.deferred = None;
        self.restoring = Some(Restoring {
            layout,
            pages: HashMap::new(),
            members: BTreeMap::new(),
            restored: 0,
        });
        Ok(())
    }

    /// Builds the restored table's pages, once its last row is back.
    fn finish_restore(&mut self) -> Result<(), NodeError> {
        let Restoring {
            layout,
            mut pages,
            members,
            ..
        } = self.restoring.take().expect("a restore is under way");
        if self.spec.series.is_some() {
            // Its rows came back on one page without a layout; the series
            // index places them.
            self.reprint();
            return Ok(());
        }
        let mut paged = Vec::with_capacity(members.len());
        for (page, rows) in members {
            let page = pages
                .remove(&page)
                .ok_or(NodeError::Restore("a row's page has no layout"))?;
            paged.push((page, rows));
        }
        if !pages.is_empty() {
            return Err(NodeError::Restore("a page's layout has no rows"));
        }
        if self.standardizer.is_none() && !paged.is_empty() {
            return Err(NodeError::Restore("rows come back without a scaling"));
        }
        self.index = PageIndex::restore(self.dimensions, self.spec.page_size, layout.index, paged)?;
        self.reprint();
        Ok(())
    }

    fn sparse(&self) -> bool {
        let live = self.table.len();
        self.table.ids_used() - live > live.max(SPARSE_ROWS)
    }

    /// How many more rows can be inserted, counting the dead ids a sparse
    /// table frees when it is renumbered before its next insert.
    fn ids_left(&self) -> u64 {
        if self.sparse() {
            ROW_IDS - self.table.len()
        } else {
            self.table.ids_left()
        }
    }

    /// One row's features.
    fn row(&self, row: RowId) -> Result<Vec<f32>, NodeError> {
        Ok(self.rows.features(row)?)
    }

    /// The live rows plans read: all of them, or for a model that reads no
    /// unlabeled context rows, the labeled ones.
    fn served(&self) -> Snapshot {
        self.table.snapshot().without(self.hidden())
    }

    /// Whether each target is known. A series' rows are observations, not
    /// labels, so they are known even when missing.
    fn labeled(&self, targets: &Targets) -> Vec<bool> {
        if self.spec.series.is_some() {
            vec![true; targets.len()]
        } else {
            targets.labeled()
        }
    }

    /// Drops pending requests, and from a pending given batch their contexts
    /// with them, so the others keep theirs.
    fn drop_pending(&mut self, requests: &[RequestId]) {
        let width = self.spec.features;
        if let Some(given) = &mut self.given {
            let mut offset = 0;
            let mut kept = Vec::with_capacity(given.len());
            for (request, queries, _) in &self.pending {
                let count = queries.len() / width;
                if !requests.contains(request) {
                    kept.extend(given[offset..offset + count].iter().cloned());
                }
                offset += count;
            }
            *given = kept;
        }
        self.pending
            .retain(|(request, _, _)| !requests.contains(request));
    }

    /// The pending requests a plan gives a context of unlabeled rows alone,
    /// which says nothing of any target.
    fn blind(&self, planned: &Plan) -> Vec<RequestId> {
        if !self.spec.unlabeled_context || self.unlabeled.is_empty() {
            return Vec::new();
        }
        let width = self.spec.features;
        let mut offset = 0;
        let mut found = Vec::new();
        for (request, queries, _) in &self.pending {
            let count = queries.len() / width;
            let reads = &planned.queries[offset..offset + count];
            if reads.iter().any(|unit| {
                planned.units[unit.0]
                    .rows
                    .bitmap()
                    .is_subset(&self.unlabeled)
            }) {
                found.push(*request);
            }
            offset += count;
        }
        found
    }

    /// Whether the table's next fit, a first one or a refit its policy or
    /// cutoffs ask for, would read more rows than it may before the worker
    /// measures how many its fits take.
    fn unmeasured_fit(&self) -> bool {
        let wanted = self.horizon_of(
            self.pending
                .first()
                .and_then(|(_, _, cutoffs)| cutoffs.as_deref()),
        );
        self.spec.sizes.fit_bound != usize::MAX
            && self.spec.summary == SummaryScope::Table
            && self.spec.series.is_none()
            && (self.fitted != Some(self.epoch) || self.refit_for(wanted))
            && self.table.len() > self.spec.sizes.fit_bound as u64
    }

    /// Rows plans leave out: unlabeled ones, unless the model reads them.
    fn hidden(&self) -> &RoaringBitmap {
        static NONE: std::sync::LazyLock<RoaringBitmap> =
            std::sync::LazyLock::new(RoaringBitmap::new);
        if self.spec.unlabeled_context {
            &NONE
        } else {
            &self.unlabeled
        }
    }

    /// Whether plans read a row of this label.
    fn serves(&self, labeled: bool) -> bool {
        labeled || self.spec.unlabeled_context
    }

    /// The labeled rows of `rows`' row-major model `features`, which drift reads.
    fn labeled_values(&self, rows: &[RowId], features: &[f32]) -> Vec<f32> {
        features
            .chunks(self.spec.model_features().max(1))
            .zip(rows)
            .filter(|(_, row)| !self.unlabeled.contains(row.0))
            .flat_map(|(values, _)| values)
            .copied()
            .collect()
    }

    /// The live rows whose target is known, which scaling and drift read.
    fn labeled_rows(&self) -> Context {
        Context::from_bitmap(self.table.snapshot().rows().bitmap() - &self.unlabeled)
    }

    /// Sets what a plan for the state of `epoch` draws cores with: a
    /// regression table's target bins, from its live rows, and no diverse
    /// cores of an earlier state.
    fn prepare_cores(&mut self, epoch: Epoch) {
        let Strategy::Retrieved {
            core: Some(core), ..
        } = self.spec.strategy
        else {
            return;
        };
        if self.cores.0 != epoch {
            self.cores = (epoch, Cores::default());
        }
        let binned = matches!(core.kind, CoreKind::Stratified | CoreKind::Balanced)
            && self.spec.task == Task::Values;
        if binned && self.edges.as_ref().is_none_or(|(at, _)| *at != epoch) {
            let mut values: Vec<f32> = self
                .served()
                .rows()
                .iter()
                .filter_map(|row| self.rows.target(row).map(f32::from_bits))
                .filter(|value| value.is_finite())
                .collect();
            values.sort_by(f32::total_cmp);
            let edges = (1..TARGET_BINS)
                .filter_map(|bin| values.get(bin * values.len() / TARGET_BINS).copied())
                .collect();
            self.edges = Some((epoch, edges));
        }
    }

    /// Whether serving requests that need a state fit on the rows before
    /// `wanted` refits: once the table is fit, if its policy asks for a refit
    /// or the state is fit on other rows.
    ///
    /// A table that summarizes each context never does: each unit is fit on
    /// its own rows and keyed by them, so units outlive drift and serve every
    /// cutoff whose window holds the same rows.
    fn refit_for(&self, wanted: Option<i64>) -> bool {
        self.spec.summary == SummaryScope::Table
            && self.fitted == Some(self.epoch)
            && (self.refit_due() || self.stale_for(wanted))
    }

    /// Plans the pending requests for the state of `epoch`.
    ///
    /// With regions, a query without a cutoff reads its region's built
    /// context while `cached` says its entry is held, stale or not, and one
    /// is built for it otherwise, as when the entry is doomed by a write; a query with a cutoff reads a context built
    /// around its region's center within its window. A region's context
    /// keeps to the strategy's recency window as it stood when built, and
    /// its rows the window has since left behind count toward a rebuild.
    fn plan_pending(
        &mut self,
        epoch: Epoch,
        snapshot: &Snapshot,
        cached: &dyn Fn(Fingerprint) -> bool,
        now: Instant,
        parts: &mut PlanParts,
    ) -> Result<Plan, NodeError> {
        let since = parts.start();
        self.prepare_cores(epoch);
        let width = self.spec.features;
        let vectors: Vec<Vec<f32>> = self
            .pending
            .iter()
            .flat_map(|(_, queries, _)| queries.chunks(width))
            .map(|row| self.vector(row))
            .collect::<Result<_, _>>()?;
        let (cutoffs, prints, window_of) = self.windows();
        let Some(spec) = self.spec.regions else {
            let planned = self.plan_around(snapshot, &vectors, &cutoffs, &prints, &window_of);
            parts.add("retrieve", since);
            return planned;
        };
        self.prepare_regions(epoch, snapshot, spec)?;
        self.age_regions(now);
        let mut regions = self.regions.take().expect("prepared above");
        // Periods that have ended mark their regions due.
        regions.due(spec.rebuild, now);
        // Counted once the plan joins a step, so a plan made again counts once.
        let mut counts = RegionStats::default();
        let mut plan = Plan::default();
        let mut ids: HashMap<Fingerprint, UnitId> = HashMap::new();
        let mut planned = vec![UnitId(0); vectors.len()];
        let mut fresh: Vec<(usize, usize)> = Vec::new();
        for (query, vector) in vectors.iter().enumerate() {
            let region = regions.of(vector);
            match &mut regions.built[region] {
                Some(built) if window_of[query].is_none() && cached(built.key) => {
                    if built.due.is_some() {
                        counts.stale += 1;
                    } else {
                        counts.hits += 1;
                    }
                    planned[query] = add_unit(&mut plan, &mut ids, &built.rows, built.key);
                }
                _ => fresh.push((query, region)),
            }
        }
        // One retrieval per region to build, not per query: its queries
        // without a window share the context. A query with a window keeps
        // a retrieval of its own, since its rows end at its cutoff.
        let mut first_of: HashMap<usize, usize> = HashMap::new();
        let mut distinct: Vec<(usize, Option<usize>)> = Vec::new();
        let mut slot_of = Vec::with_capacity(fresh.len());
        for &(query, region) in &fresh {
            let slot = match window_of[query] {
                None => *first_of.entry(region).or_insert_with(|| {
                    distinct.push((region, None));
                    distinct.len() - 1
                }),
                window => {
                    distinct.push((region, window));
                    distinct.len() - 1
                }
            };
            slot_of.push(slot);
        }
        let centers: Vec<Vec<f32>> = distinct
            .iter()
            .map(|&(region, _)| regions.centers[region].clone())
            .collect();
        let windows: Vec<Option<usize>> = distinct.iter().map(|&(_, window)| window).collect();
        parts.add("regions", since);
        let since = parts.start();
        let around = if distinct.is_empty() {
            Ok(Plan::default())
        } else {
            self.plan_around(snapshot, &centers, &cutoffs, &prints, &windows)
        };
        parts.add("retrieve", since);
        let since = parts.start();
        let around = match around {
            Ok(around) => around,
            Err(error) => {
                self.regions = Some(regions);
                return Err(error);
            }
        };
        let mut built_now: HashSet<usize> = HashSet::new();
        for (&(query, region), &slot) in fresh.iter().zip(&slot_of) {
            let unit = &around.units[around.queries[slot].0];
            if window_of[query].is_none() {
                if built_now.insert(region) {
                    let times = self.times.times_of(&unit.rows);
                    regions.build(region, Built::new(unit.rows.clone(), unit.key, times, now));
                }
                counts.misses += 1;
            } else {
                counts.fresh += 1;
            }
            planned[query] = add_unit(&mut plan, &mut ids, &unit.rows, unit.key);
        }
        self.regions = Some(regions);
        self.planned_counts = counts;
        plan.queries = planned;
        parts.add("regions", since);
        Ok(plan)
    }

    /// Draws the regions of the state of `epoch` if they are not drawn yet,
    /// or the retrieval scaling changed since. A strategy without neighbors
    /// gives every center the same context, so it has one region.
    fn prepare_regions(
        &mut self,
        epoch: Epoch,
        snapshot: &Snapshot,
        spec: RegionSpec,
    ) -> Result<(), NodeError> {
        let drawn = self
            .regions
            .as_ref()
            .is_some_and(|regions| regions.epoch == epoch && regions.scaling == self.scaled_rows);
        if !drawn {
            let per_region = match self.spec.strategy {
                Strategy::Retrieved { neighbors, .. } if neighbors > 0 => spec.rows,
                _ => u64::MAX,
            };
            self.region_draws += 1;
            let live = snapshot.rows();
            let regions = Regions::draw(
                (epoch, self.scaled_rows, self.region_draws),
                &live,
                per_region,
                self.dimensions.get(),
                &self.scaled(),
                &self.table,
            )
            .map_err(NodeError::Regions)?;
            let stats = &mut self.region_stats;
            stats.regions = regions.centers.len() as u64;
            stats.mean_rows = live.len() as f64 / stats.regions as f64;
            stats.levels = regions.levels();
            self.regions = Some(regions);
        }
        Ok(())
    }

    /// Plans queries at `vectors`, each within its window among the rows
    /// before `cutoffs`, or every live row.
    fn plan_around(
        &mut self,
        snapshot: &Snapshot,
        vectors: &[Vec<f32>],
        cutoffs: &[i64],
        prints: &[Fingerprint],
        window_of: &[Option<usize>],
    ) -> Result<Plan, NodeError> {
        let windows = Windows::Before {
            times: &self.times,
            cutoffs,
            prints,
        };
        let mut cores = std::mem::take(&mut self.cores.1);
        let source = Source {
            snapshot,
            index: &self.index,
            vectors: &self.scaled(),
            strata: &self.labels(),
            keys: &self.table,
            ranking: self.ranking.as_ref(),
            times: &self.times,
        };
        let planned = plan_with(
            &self.spec.strategy,
            &source,
            &mut cores,
            vectors,
            windows,
            window_of,
        );
        self.cores.1 = cores;
        Ok(planned?)
    }

    /// Refuses query rows of the wrong width, or cutoffs other than one a query.
    fn check_queries(&self, queries: &[f32], cutoffs: Option<&[i64]>) -> Result<(), NodeError> {
        let width = self.spec.features;
        if queries.is_empty() {
            return Err(NodeError::NoQueries);
        }
        if !queries.len().is_multiple_of(width) {
            return Err(NodeError::Width {
                expected: width,
                got: queries.len(),
            });
        }
        if let Some(cutoffs) = cutoffs
            && cutoffs.len() * width != queries.len()
        {
            return Err(NodeError::Count {
                what: Counted::Cutoffs,
                expected: queries.len() / width,
                got: cutoffs.len(),
            });
        }
        Ok(())
    }

    /// Counts the rows of built region contexts that a recency window has
    /// left behind since they were built, once after each change to the table.
    fn age_regions(&mut self, now: Instant) {
        let (
            Some(spec),
            Strategy::Retrieved {
                recency: Some(recency),
                ..
            },
        ) = (self.spec.regions, self.spec.strategy)
        else {
            return;
        };
        if self.regions.is_none() || self.aged_at == Some(self.generation) {
            return;
        }
        self.aged_at = Some(self.generation);
        let live = self.served().rows();
        let oldest = match recency {
            Recency::Last(rows) => Some(self.times.latest_from(&live, rows)),
            Recency::Span(span) => self
                .times
                .newest(&live)
                .map(|newest| newest.saturating_sub(i64::try_from(span).unwrap_or(i64::MAX))),
        };
        if let (Some(oldest), Some(regions)) = (oldest, self.regions.as_mut()) {
            regions.age(oldest, spec.rebuild, now);
        }
    }

    /// Counts a changed row, with these features, toward its region's rebuild.
    fn note_change(&mut self, features: &[f32]) -> Result<(), NodeError> {
        if let Some(spec) = self.spec.regions
            && self.regions.is_some()
        {
            let vector = self.vector(features)?;
            let regions = self.regions.as_mut().expect("checked above");
            regions.change(&vector, spec.rebuild, Instant::now());
        }
        Ok(())
    }

    /// The rows' strata, by class or by target bin.
    fn labels(&self) -> Labels<'_> {
        Labels {
            rows: &self.rows,
            task: self.spec.task,
            edges: self.edges.as_ref().map_or(&[], |(_, edges)| edges),
        }
    }

    /// The table's rows in its retrieval space, as its index reads them.
    fn scaled(&self) -> Scaled<'_> {
        Scaled {
            rows: &self.rows,
            standardizer: self.standardizer.as_ref(),
        }
    }

    /// Whether the table's refit policy asks for a new fit.
    fn refit_due(&self) -> bool {
        let Some(fitted) = &self.fitted_columns else {
            return false;
        };
        let policy = self.spec.refit;
        let changed = policy
            .changed
            .is_some_and(|fraction| self.changed as f64 >= fraction * fitted.rows.max(1) as f64);
        let shifted = policy.shift.is_some_and(|limit| {
            (0..self.spec.model_features()).any(|column| {
                let scale = if fitted.deviations[column] > 0.0 {
                    fitted.deviations[column]
                } else {
                    1.0
                };
                (self.columns.mean(column) - fitted.means[column]).abs() > limit * scale
            })
        });
        changed || shifted
    }

    /// The rows a request's state must be fit on so no query sees a row from
    /// after its cutoff: those before its earliest cutoff, named by the first
    /// row time at or after it, so cutoffs with the same rows before them
    /// share a state; or every row, for a request without cutoffs or one
    /// whose cutoffs come after every row.
    fn horizon_of(&self, cutoffs: Option<&[i64]>) -> Option<i64> {
        let cutoff = cutoffs?.iter().min()?;
        self.times.first_outside(*cutoff, self.hidden())
    }

    /// Whether requests needing a state fit on the rows before `wanted` need
    /// a new fit. A state fit on any row after their cutoffs would leak it,
    /// so it always refits. One fit on fewer, earlier rows leaks nothing, so
    /// the table's `CutoffRefit` decides: `Each` refits, so each cutoff reads
    /// the state its rows give; `Changed` keeps it until rows after it reach
    /// the policy's share of the rows it was fit on; `Never` keeps it.
    fn stale_for(&self, wanted: Option<i64>) -> bool {
        // A state fit on a row from at or after any pending cutoff would leak it.
        let earliest = self
            .pending
            .iter()
            .filter_map(|(_, _, cutoffs)| cutoffs.as_deref()?.iter().min().copied())
            .min();
        if let (Some(latest), Some(cutoff)) = (self.fitted_latest, earliest)
            && latest >= cutoff
        {
            return true;
        }
        if self.horizon == wanted {
            return false;
        }
        match (self.horizon, wanted, self.spec.refit.cutoffs) {
            // Fit on rows from at or after the cutoff.
            (Some(fitted), Some(wanted), _) if fitted > wanted => true,
            (_, _, CutoffRefit::Never) => false,
            (_, _, CutoffRefit::Each) => true,
            (fitted, wanted, CutoffRefit::Changed) => {
                // A state fit on every row holds those up to its latest time.
                let fitted = fitted.unwrap_or_else(|| {
                    self.fitted_latest
                        .map_or(i64::MIN, |latest| latest.saturating_add(1))
                });
                let share = self.spec.refit.changed.unwrap_or(0.0);
                let between = self.times.between(fitted, wanted.unwrap_or(i64::MAX));
                let since = (between.bitmap() - self.hidden()).len();
                let rows = self.fitted_columns.as_ref().map_or(0, |fitted| fitted.rows);
                since as f64 >= share * rows.max(1) as f64
            }
        }
    }

    /// Removes requests with a query that would read no rows: one whose cutoff
    /// precedes every row, or any once the table has been emptied.
    fn take_empty(&mut self, table: TableId) -> Vec<(RequestId, NodeError)> {
        let mut refused = Vec::new();
        let width = self.spec.features;
        // Requests promoted in place of refused ones are checked in turn.
        while !self.pending.is_empty() {
            // A given context whose rows have all gone, or are all unlabeled
            // for a model that reads no such rows, reads nothing.
            let live = self.given.as_ref().map(|_| self.served().rows());
            let given = self.given.take();
            let (mut kept, mut kept_given, mut offset) = (Vec::new(), Vec::new(), 0);
            for (request, queries, cutoffs) in std::mem::take(&mut self.pending) {
                let count = queries.len() / width;
                let contexts = given.as_ref().map(|given| &given[offset..offset + count]);
                offset += count;
                let gone = contexts.zip(live.as_ref()).is_some_and(|(contexts, live)| {
                    contexts.iter().any(|context| (context & live).is_empty())
                });
                let error = if gone {
                    Some(NodeError::Given)
                } else {
                    self.empty(table, cutoffs.as_deref())
                };
                match error {
                    Some(error) => refused.push((request, error)),
                    None => {
                        kept_given.extend(contexts.into_iter().flatten().cloned());
                        kept.push((request, queries, cutoffs));
                    }
                }
            }
            self.pending = kept;
            self.given = given.map(|_| kept_given);
            if !self.pending.is_empty() {
                break;
            }
            self.promote();
        }
        self.generation += refused.len() as u64;
        refused
    }

    /// Once no request is pending, makes pending the queued given requests,
    /// as one batch, or the others that wait. The two take turns, so neither
    /// waits more than one step behind the other.
    fn promote(&mut self) {
        if !self.pending.is_empty() {
            return;
        }
        self.given = None;
        if self.givens.is_empty() || (self.gave && !self.later.is_empty()) {
            self.pending = std::mem::take(&mut self.later);
            return;
        }
        let mut given = Vec::new();
        for (request, queries, contexts) in self.givens.drain(..) {
            self.pending.push((request, queries, None));
            given.extend(contexts);
        }
        self.given = Some(given);
    }

    /// Why a request with these cutoffs would read no rows, if it would.
    fn empty(&self, table: TableId, cutoffs: Option<&[i64]>) -> Option<NodeError> {
        // A context needs a labeled row, even for a model that reads unlabeled ones.
        let hidden = &self.unlabeled;
        if self.table.len() == hidden.len() {
            return Some(NodeError::EmptyTable(table));
        }
        // The times of the labeled rows.
        let (untimed, timed) = (
            self.times.has_untimed_outside(hidden),
            self.times.first_outside(i64::MIN, hidden),
        );
        // Rows without a time are before every cutoff.
        let earliest = (!untimed).then_some(timed).flatten();
        let before = |first: i64| {
            cutoffs.is_some_and(|cutoffs| cutoffs.iter().any(|&cutoff| cutoff <= first))
        };
        if earliest.is_some_and(before) {
            return Some(NodeError::EmptyContext(table));
        }
        // A span of time holds timed rows alone, so it is empty with none of
        // them before the cutoff.
        let span = matches!(
            self.spec.strategy,
            Strategy::Retrieved {
                recency: Some(Recency::Span(_)),
                ..
            }
        );
        match timed {
            _ if !span => None,
            None => Some(NodeError::Untimed(table)),
            Some(timed) if before(timed) => Some(NodeError::EmptyContext(table)),
            Some(_) => None,
        }
    }

    /// Adds a row's fingerprint at `time`, or takes it away.
    fn print(&mut self, time: Option<i64>, print: Fingerprint, add: bool) {
        let at = self.prints.entry(time).or_default();
        *at = if add { *at + print } else { *at - print };
        if *at == Fingerprint::default() {
            self.prints.remove(&time);
        }
    }

    /// A live row's fingerprint, from its version.
    fn print_of(&self, row: RowId) -> Fingerprint {
        Fingerprint::of(row, self.table.version(row).unwrap_or(Version(0)))
    }

    /// Places every live row in its series again, as after a renumbering, from
    /// the series and time its key packs.
    fn reindex_series(&mut self) {
        let Some(spec) = &self.spec.series else {
            return;
        };
        let mut index = SeriesIndex::new(spec.step).expect("the step was accepted at creation");
        let keyed: Vec<(RowKey, RowId)> = self.table.keys().collect();
        let rows: Vec<RowId> = keyed.iter().map(|&(_, row)| row).collect();
        let (_, targets) = self.gather(&rows).expect("live rows are stored");
        for (offset, &(key, row)) in keyed.iter().enumerate() {
            let (_, time) = SeriesKey::unpack(key);
            index
                .insert(row, key, time, targets.observed(offset))
                .expect("live rows were placed when inserted");
        }
        self.series = Some(index);
    }

    /// Checks that a batch of rows could be placed in their series, for a
    /// table of series: the whole batch against each series' grid, rows
    /// earlier in the batch included. With `replacing`, a time the table
    /// holds may be written over, as an update or upsert does.
    fn check_series(
        &self,
        keys: &[u128],
        times: Option<&[Option<i64>]>,
        replacing: bool,
    ) -> Result<(), NodeError> {
        let Some(index) = &self.series else {
            return Ok(());
        };
        let times = times.ok_or(ForecastError::Untimed)?;
        let rows = keys
            .iter()
            .zip(times)
            .map(|(&key, &time)| time.map(|time| (RowKey(key), time)))
            .collect::<Option<Vec<_>>>()
            .ok_or(ForecastError::Untimed)?;
        index
            .check_batch(rows, |key| replacing && self.table.row(key).is_some())
            .map_err(|error| ForecastError::from(error).into())
    }

    /// The request a table has waited longest for, if any.
    fn first_request(&self) -> Option<RequestId> {
        let pending = self.pending.first().map(|(request, _, _)| *request);
        let forecast = self.forecasts.first().map(|(request, _)| *request);
        pending.into_iter().chain(forecast).min()
    }

    /// Fingerprints and ranks every live row again, as after a renumbering.
    fn reprint(&mut self) {
        let rows: Vec<RowId> = self.table.snapshot().rows().iter().collect();
        self.reindex_series();
        self.prints.clear();
        self.unlabeled.clear();
        let task = self.spec.task;
        for &row in &rows {
            let bits = self.rows.target(row).unwrap_or_default();
            let labeled = self.labeled(&Targets::from_bits(task, &[bits]))[0];
            if !labeled {
                self.unlabeled.insert(row.0);
            }
            if self.serves(labeled) {
                let (time, print) = (self.times.time(row), self.print_of(row));
                self.print(time, print, true);
            }
        }
        let served = self.served().rows();
        if let Some(ranking) = &mut self.ranking {
            *ranking = Ranking::new(ranking.seed(), &served, &self.table);
        }
        // Cores and regions' contexts name rows by id, which a renumbering changes.
        self.cores.1.clear();
        self.regions = None;
    }

    /// The distinct cutoffs pending queries read, ascending, the fingerprint
    /// of the rows before each, and each query's window among them, or None
    /// for every row. Planning grows each window from the one before.
    fn windows(&self) -> (Vec<i64>, Vec<Fingerprint>, Vec<Option<usize>>) {
        let mut distinct: Vec<i64> = self
            .pending
            .iter()
            .filter_map(|(_, _, cutoffs)| cutoffs.as_deref())
            .flatten()
            .copied()
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        // The fingerprint of the rows before each cutoff, in one pass over times.
        let mut prints = Vec::with_capacity(distinct.len());
        let mut sum = Fingerprint::default();
        let mut times = self.prints.iter().peekable();
        for &cutoff in &distinct {
            while let Some((_, print)) = times.next_if(|(time, _)| **time < Some(cutoff)) {
                sum = sum + *print;
            }
            prints.push(sum);
        }
        let width = self.spec.features;
        let mut window_of = Vec::new();
        for (_, queries, cutoffs) in &self.pending {
            let count = queries.len() / width;
            match cutoffs {
                None => window_of.extend(std::iter::repeat_n(None, count)),
                Some(cutoffs) => window_of.extend(cutoffs.iter().map(|cutoff| {
                    Some(
                        distinct
                            .binary_search(cutoff)
                            .expect("every cutoff is listed"),
                    )
                })),
            }
        }
        (distinct, prints, window_of)
    }

    /// Rows' features, row-major, and their targets, in order.
    fn gather(&self, rows: &[RowId]) -> Result<(Vec<f32>, Targets), NodeError> {
        let (features, bits) = self.rows.gather(rows)?;
        Ok((features, Targets::from_bits(self.spec.task, &bits)))
    }

    /// As `gather`, with only the features the model reads.
    fn inputs(&self, rows: &[RowId]) -> Result<(Vec<f32>, Targets), NodeError> {
        let (features, targets) = self.gather(rows)?;
        Ok((self.spec.model_values(&features), targets))
    }

    /// Fits retrieval scaling on the live rows and places every row again with
    /// it, in row order: a table's first batch alone may say little.
    fn rescale(&mut self) -> Result<(), NodeError> {
        let rows: Vec<RowId> = self.table.snapshot().rows().iter().collect();
        if rows.is_empty() {
            return Ok(());
        }
        // Unlabeled rows place nothing until they are labeled, unless none is.
        let labeled: Vec<RowId> = self.labeled_rows().iter().collect();
        let fit = if labeled.is_empty() { &rows } else { &labeled };
        // A scaling past `scale_rows` stays; the rows are only placed again.
        let frozen = self
            .spec
            .scale_rows
            .is_some_and(|limit| self.scaled_rows >= limit.get());
        let (standardizer, scaled_rows) = match &self.standardizer {
            Some(kept) if frozen => (kept.clone(), self.scaled_rows),
            _ => (
                {
                    let (features, targets) = self.gather(fit)?;
                    Standardizer::fit(&features, &targets, &self.spec)
                },
                labeled.len() as u64,
            ),
        };
        let mut index = PageIndex::new(self.dimensions, self.spec.page_size);
        let scaled = Scaled {
            rows: &self.rows,
            standardizer: Some(&standardizer),
        };
        let placed = rows
            .iter()
            .map(|&row| {
                Ok((
                    row,
                    self.spec.bucket(self.times.time(row)),
                    scaled.vector(row)?,
                ))
            })
            .collect::<Result<Vec<_>, RowsError>>()?;
        index.insert_many(&placed, &scaled)?;
        self.index = index;
        self.standardizer = Some(standardizer);
        self.scaled_rows = scaled_rows;
        Ok(())
    }

    /// Gives the live rows new ids, densely in key order, and places them
    /// again, so the table's ids and pages depend on its rows alone, not on
    /// the order of the writes that made them.
    fn renumber(&mut self) -> Result<(), NodeError> {
        let mut keyed: Vec<(RowKey, RowId)> = self.table.keys().collect();
        keyed.sort_unstable();
        let rows: Vec<RowId> = keyed.iter().map(|&(_, row)| row).collect();
        let mut table = Table::new();
        let mut times = TimeIndex::new();
        for &(key, row) in &keyed {
            let renumbered = table.apply(Change::Insert(key))?;
            times.set(renumbered, self.times.time(row));
        }
        let (features, targets) = self.gather(&rows)?;
        let keys: Vec<u128> = keyed.iter().map(|&(key, _)| key.0).collect();
        self.rows.clear()?;
        self.rows.push(&features, &targets.bits(), &keys)?;
        self.table = table;
        self.times = times;
        self.reprint();
        if self.spec.series.is_some() {
            // A table of series places no rows in pages.
            return Ok(());
        }
        self.rescale()
    }

    /// Checks a batch's shape, keys and targets before anything changes.
    fn check(&self, keys: &[u128], features: &[f32], targets: &Targets) -> Result<(), NodeError> {
        check_rows(keys.len(), features, self.spec.features)?;
        if targets.len() != keys.len() {
            return Err(NodeError::Count {
                what: Counted::Targets,
                expected: keys.len(),
                got: targets.len(),
            });
        }
        match (self.spec.task, targets) {
            (Task::Classes(classes), Targets::Classes(labels)) => {
                if let Some(&label) = labels
                    .iter()
                    .find(|&&label| label < UNLABELED || label >= 0 && label as u32 >= classes)
                {
                    return Err(NodeError::Label { label, classes });
                }
            }
            (Task::Values, Targets::Values(_)) => {}
            _ => return Err(NodeError::Task),
        }
        let mut seen = BTreeSet::new();
        if let Some(&key) = keys.iter().find(|&&key| !seen.insert(key)) {
            return Err(NodeError::DuplicateKey(RowKey(key)));
        }
        Ok(())
    }

    fn vector(&self, row: &[f32]) -> Result<Vec<f32>, NodeError> {
        self.standardizer
            .as_ref()
            .map(|standardizer| standardizer.vector(row))
            .ok_or(NodeError::Unscaled)
    }
}

/// The order a step takes waiting tables in, while its budget lasts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scheduling {
    /// Each step starts after the table the last one started with, so a busy
    /// table cannot starve the rest.
    #[default]
    Turns,
    /// The table whose oldest request has waited longest goes first.
    Arrival,
}

impl Scheduling {
    /// Every order, for parsing names.
    pub const ALL: [Self; 2] = [Self::Turns, Self::Arrival];

    /// The order's name, as configurations spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Turns => "turns",
            Self::Arrival => "arrival",
        }
    }
}

/// Many tables behind one request queue and one cache.
///
/// Each step has a budget: the contexts it newly encodes, kept or not, take at
/// most `step_bytes`, the device tier's size by default, or no limit without a
/// device tier, since then nothing is kept. Requests past it wait for a later
/// step, in the order `scheduling` picks.
#[derive(Debug)]
pub struct Node {
    cache: Cache,
    tables: BTreeMap<TableId, TableState>,
    next_table: u32,
    next_request: u64,
    step_bytes: Option<u64>,
    scheduling: Scheduling,
    /// Memory the tables' rows share.
    rows: RowMemory,
    /// The table the next step starts from.
    turn: TableId,
    /// Stale entries a running step still reads, dropped once every step
    /// reading them completes.
    doomed: HashSet<CacheKey>,
    /// Drops from purges, for the executor to apply.
    drops: Vec<Movement>,
    /// Requests planning refused, for the executor to answer.
    refused: Vec<(RequestId, NodeError)>,
    /// Tables that started an epoch since the executor last asked.
    advanced: Vec<TableId>,
    /// Tables whose contexts first passed their model's trained length.
    past_cap: Vec<TableId>,
    /// Tables waiting for the worker to measure how many rows their fit
    /// takes, and those the executor was not yet told of.
    unmeasured: BTreeSet<TableId>,
    untold: Vec<TableId>,
    /// Region contexts a step in flight builds again, by their entries, and
    /// the regions they are for.
    rebuilding: HashMap<CacheKey, Vec<Rebuilding>>,
    rebuilt_regions: HashSet<(TableId, usize)>,
    /// Entries a step in flight moves between tiers: later steps read them
    /// only once it completes, since until then their slots may not hold them.
    moving: HashSet<CacheKey>,
    /// Entries a step in flight encodes, and tables it fits. Requests that
    /// would read them wait for it: if it fails, the worker never holds them.
    encoding: HashSet<CacheKey>,
    fitting: HashSet<TableId>,
    /// Where the last `schedule` spent its time.
    parts: PlanParts,
}

impl Node {
    /// An empty node over a device and a host pool.
    ///
    /// # Errors
    ///
    /// If the pools' slots differ in size.
    pub fn new(device: TierConfig, host: TierConfig) -> Result<Self, NodeError> {
        Ok(Self {
            cache: Cache::new(device, host)?,
            tables: BTreeMap::new(),
            next_table: 0,
            next_request: 0,
            step_bytes: None,
            scheduling: Scheduling::Turns,
            rows: RowMemory::unbounded(),
            turn: TableId(0),
            doomed: HashSet::new(),
            drops: Vec::new(),
            refused: Vec::new(),
            advanced: Vec::new(),
            past_cap: Vec::new(),
            unmeasured: BTreeSet::new(),
            untold: Vec::new(),
            rebuilding: HashMap::new(),
            rebuilt_regions: HashSet::new(),
            moving: HashSet::new(),
            encoding: HashSet::new(),
            fitting: HashSet::new(),
            parts: PlanParts::default(),
        })
    }

    /// Where the last `schedule` spent its time, by part.
    #[must_use]
    pub fn plan_parts(&self) -> &PlanParts {
        &self.parts
    }

    /// Times each part of planning a step from the next `schedule` on, if `timed`.
    pub fn set_timers(&mut self, timed: bool) {
        self.parts = PlanParts::new(timed);
    }

    /// Drops of cached entries that held rows since changed or deleted, for the
    /// executor to apply now: no cache keeps a row's old values once it changes.
    pub fn take_drops(&mut self) -> Vec<Movement> {
        std::mem::take(&mut self.drops)
    }

    /// Requests planning refused since the last call, with why: their queries
    /// would read no rows. Every other request goes into a step.
    pub fn take_refused(&mut self) -> Vec<(RequestId, NodeError)> {
        std::mem::take(&mut self.refused)
    }

    /// Tables that started an epoch, by a refit asked for or due, since the
    /// last call, once per epoch.
    pub fn take_advanced(&mut self) -> Vec<TableId> {
        std::mem::take(&mut self.advanced)
    }

    /// Tables that, since the last call, first read a context of more rows
    /// than their model was trained on, under `PastCap::Warn`; each table once.
    pub fn take_past_cap(&mut self) -> Vec<TableId> {
        std::mem::take(&mut self.past_cap)
    }

    /// Tables that, since the last call, came to wait for the worker to
    /// measure how many rows their fit takes; each table once.
    ///
    /// A table waits once a fit would read more than its `EntrySizes::fit_bound`
    /// rows, and its requests stay pending until `set_fit_limits`.
    pub fn take_unmeasured(&mut self) -> Vec<TableId> {
        std::mem::take(&mut self.untold)
    }

    /// Sets the most rows a table's state alone is fit on and the most a fit
    /// embeds with its state, as the worker measured them, so it fits again.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn set_fit_limits(
        &mut self,
        table: TableId,
        state_rows: usize,
        fused_rows: usize,
    ) -> Result<(), NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.spec.sizes.state_rows = state_rows.max(1);
        state.spec.sizes.fused_rows = fused_rows;
        state.spec.sizes.fit_bound = usize::MAX;
        self.unmeasured.remove(&table);
        self.untold.retain(|&untold| untold != table);
        Ok(())
    }

    /// What a table's regions served and what rebuilding them cost, if it has regions.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn region_stats(&self, table: TableId) -> Result<Option<RegionStats>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        Ok(state.spec.regions.map(|_| state.region_stats))
    }

    /// Each query's region churn: rows changed in its region since its
    /// context was built, and the context's rows aged out of its window, as
    /// a share of the context's rows, which `Rebuild::Churn` compares. Aged
    /// rows count as of the table's latest plan or schedule. None for a query
    /// whose region has no built context or whose context is no longer
    /// cached, as the next read builds it again, and for every query until a
    /// plan draws the table's regions.
    ///
    /// # Errors
    ///
    /// If the table does not exist, or the query rows are of the wrong width.
    pub fn region_churn(
        &self,
        table: TableId,
        queries: &[f32],
    ) -> Result<Vec<Option<f64>>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.check_queries(queries, None)?;
        let rows = queries.chunks(state.spec.features);
        let Some(regions) = state
            .regions
            .as_ref()
            .filter(|regions| regions.epoch == state.epoch && regions.scaling == state.scaled_rows)
        else {
            return Ok(vec![None; rows.len()]);
        };
        // A context a write doomed, or one still being encoded, is not built.
        let cached = |built: &&Built| {
            let key = CacheKey {
                model: state.spec.model,
                table,
                epoch: state.epoch,
                kind: Kind::Context,
                rows: built.key,
            };
            self.cache.contains(&key)
                && !self.doomed.contains(&key)
                && !self.encoding.contains(&key)
        };
        rows.map(|row| {
            state
                .vector(row)
                .map(|vector| regions.built_of(&vector).filter(cached).map(Built::churn))
        })
        .collect::<Result<_, _>>()
    }

    /// How a table is served: its strategy as cut to the model's trained
    /// length under `PastCap::Clamp`.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn spec(&self, table: TableId) -> Result<&TableSpec, NodeError> {
        Ok(&self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .spec)
    }

    /// Drops every cached context of a table that holds one of `rows`; entries
    /// a running step reads are dropped when it completes. A page's entry
    /// stays, with the rows marked stale, as a source for the page's next
    /// version, which copies the rest of its rows instead of embedding them.
    fn purge(&mut self, table: TableId, rows: &[RowId]) {
        let changed: Context = rows.iter().copied().collect();
        let epoch = self.tables.get(&table).map(|state| state.epoch);
        for key in self.cache.holding(table, &changed) {
            // A page's entry is a source while it holds a row no write touched.
            if key.kind == Kind::Rows && Some(key.epoch) == epoch && !self.doomed.contains(&key) {
                let fresh = match (self.cache.rows(&key), self.cache.stale(&key)) {
                    (Some(held), Some(stale)) => held
                        .iter()
                        .any(|row| !stale.contains(row) && !changed.contains(row)),
                    _ => false,
                };
                if fresh {
                    let _ = self.cache.mark_stale(&key, &changed);
                    continue;
                }
            }
            let dropped = self.drop_or_doom(&key);
            self.drops.extend(dropped);
        }
    }

    /// Drops a stale entry now, or once the steps reading it complete.
    fn drop_or_doom(&mut self, key: &CacheKey) -> Option<Movement> {
        if self.cache.pinned(key) {
            self.doomed.insert(*key);
            return None;
        }
        self.doomed.remove(key);
        self.cache.remove(key)
    }

    /// The node with a disk tier that contexts spill to from the host.
    ///
    /// # Errors
    ///
    /// If the disk's slots differ in size from the other pools'.
    pub fn with_disk(mut self, disk: TierConfig) -> Result<Self, NodeError> {
        self.cache = self.cache.with_disk(disk)?;
        Ok(self)
    }

    /// The node with another eviction policy; set before anything is cached.
    #[must_use]
    pub fn with_eviction(self, eviction: Eviction) -> Self {
        Self {
            cache: self.cache.with_eviction(eviction),
            ..self
        }
    }

    /// Bytes of a cache tier's slots that entries hold.
    #[must_use]
    pub fn cache_bytes(&self, tier: Tier) -> u64 {
        self.cache.used_bytes(tier)
    }

    /// Limits the context bytes one step may newly encode.
    pub fn set_step_bytes(&mut self, bytes: u64) {
        self.step_bytes = Some(bytes);
    }

    /// Whether every table with requests waiting is a single-use context
    /// not yet fitted, whose requests a step would fit, encode and decode
    /// in one call.
    #[must_use]
    pub fn only_new_contexts_wait(&self) -> bool {
        let mut waiting = false;
        for state in self.tables.values() {
            if state.pending.is_empty()
                && state.later.is_empty()
                && state.givens.is_empty()
                && state.deferred.is_none()
            {
                continue;
            }
            if !state.single_use || state.fitted == Some(state.epoch) || state.restoring.is_some() {
                return false;
            }
            waiting = true;
        }
        waiting && self.drops.is_empty() && self.rebuilding.is_empty()
    }

    /// Marks a table as holding a context one request brought, which may
    /// never be read again. Its first step, when the queries waiting read one
    /// context of every row, fits its state, encodes the context and decodes
    /// them in one call, writing no pages and keeping no context; a later
    /// step that reads it again caches it as any table's.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn set_single_use(&mut self, table: TableId) -> Result<(), NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.single_use = true;
        Ok(())
    }

    /// The node with tables' rows in `memory`, which bounds what tables with
    /// files keep in memory; set before any table is made.
    #[must_use]
    pub fn with_row_memory(self, memory: RowMemory) -> Self {
        Self {
            rows: memory,
            ..self
        }
    }

    /// Bytes of rows the node's tables keep in memory.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        self.rows.bytes()
    }

    /// Bytes a table's page index holds, about, and its rows in memory.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn index_bytes(&self, table: TableId) -> Result<u64, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        Ok(state.index.bytes() + state.rows.memory_bytes())
    }

    /// Sets the order a step takes waiting tables in.
    pub fn set_scheduling(&mut self, scheduling: Scheduling) {
        self.scheduling = scheduling;
    }

    /// Adds an empty table.
    ///
    /// # Errors
    ///
    /// If the table has no feature columns, or its strategy a zero count.
    pub fn create_table(&mut self, spec: TableSpec) -> Result<TableId, NodeError> {
        let rows = RowStore::new(&self.rows, spec.features);
        self.add_table(spec, rows)
    }

    /// Adds an empty table whose rows spill to a file at `path` when the
    /// node's row memory runs short, sealed by `sealer`.
    ///
    /// # Errors
    ///
    /// As `create_table`, or if the file cannot be made.
    pub fn create_table_on_disk(
        &mut self,
        spec: TableSpec,
        path: &std::path::Path,
        sealer: std::sync::Arc<dyn Sealer>,
    ) -> Result<TableId, NodeError> {
        let rows = RowStore::on_disk(&self.rows, spec.features, path, sealer)?;
        self.add_table(spec, rows)
    }

    fn add_table(&mut self, spec: TableSpec, rows: RowStore) -> Result<TableId, NodeError> {
        if spec.retrieval >= spec.features && (spec.retrieval > 0 || spec.series.is_none()) {
            return Err(NodeError::NoFeatures);
        }
        let ordered = spec.categorical.windows(2).all(|pair| pair[0] < pair[1]);
        if !ordered
            || spec
                .categorical
                .iter()
                .any(|&feature| feature >= spec.model_features())
        {
            return Err(NodeError::Categories(
                "categorical features are distinct model features, in order",
            ));
        }
        if spec.retrieval > 0 && spec.categories != CategoryEncoding::Codes {
            return Err(NodeError::Categories(
                "retrieval columns place rows themselves, so categoricals keep their codes",
            ));
        }
        // A table of series may hold targets alone: it places no rows in its
        // page index, which stays empty.
        let space = spec.dimensions().max(usize::from(spec.series.is_some()));
        let dimensions = NonZeroUsize::new(space).ok_or(NodeError::NoFeatures)?;
        spec.strategy.check()?;
        if let Some(series) = &spec.series {
            if let Some(&column) = series.known.iter().find(|&&column| column >= spec.features) {
                return Err(ForecastError::KnownColumn {
                    column,
                    features: spec.features,
                }
                .into());
            }
            // A group that would pass the model's cap on rows is refused
            // here, never drawn smaller than the table asked.
            series.check_fit(spec.features)?;
        }
        let spec = match (spec.past_cap, spec.context_rows) {
            (PastCap::Clamp, Some(rows)) => TableSpec {
                strategy: spec.strategy.capped(rows.get()),
                ..spec
            },
            _ => spec,
        };
        let series = spec
            .series
            .as_ref()
            .map(|series| SeriesIndex::new(series.step).map_err(ForecastError::from))
            .transpose()?;
        let (model_features, page_size) = (spec.model_features(), spec.page_size);
        let ranking = spec.strategy.seed().map(Ranking::empty);
        let id = TableId(self.next_table);
        self.next_table += 1;
        self.tables.insert(
            id,
            TableState {
                spec,
                dimensions,
                table: Table::new(),
                index: PageIndex::new(dimensions, page_size),
                epoch: Epoch(0),
                rows,
                standardizer: None,
                scaled_rows: 0,
                prints: BTreeMap::new(),
                ranking,
                unlabeled: RoaringBitmap::new(),
                edges: None,
                cores: (Epoch(0), Cores::default()),
                fitted: None,
                times: TimeIndex::new(),
                series,
                forecasts: Vec::new(),
                recent: HashMap::new(),
                writing: HashMap::new(),
                recent_pruned: 0,
                columns: Columns::new(model_features),
                fitted_columns: None,
                changed: 0,
                pending: Vec::new(),
                later: Vec::new(),
                horizon: None,
                fitted_latest: None,
                given: None,
                givens: std::collections::VecDeque::new(),
                gave: false,
                generation: 0,
                deferred: None,
                restoring: None,
                past_cap: false,
                regions: None,
                region_stats: RegionStats::default(),
                region_draws: 0,
                planned_counts: RegionStats::default(),
                aged_at: None,
                rebuild_waited: 0,
                single_use: false,
            },
        );
        Ok(id)
    }

    /// Removes a table and returns drops for its cache entries.
    ///
    /// Requests still queued for the table are refused; see `take_refused`.
    /// Entries a step in flight reads are dropped once it completes.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn remove_table(&mut self, table: TableId) -> Result<Vec<Movement>, NodeError> {
        let state = self
            .tables
            .remove(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        self.unmeasured.remove(&table);
        self.untold.retain(|&untold| untold != table);
        let queued = state
            .pending
            .iter()
            .chain(&state.later)
            .map(|(request, _, _)| *request);
        let given = state.givens.iter().map(|(request, _, _)| *request);
        self.refused.extend(
            queued
                .chain(given)
                .chain(state.forecasts.iter().map(|(request, _)| *request))
                .map(|request| (request, NodeError::UnknownTable(table))),
        );
        let after = Epoch(state.epoch.0 + 1);
        Ok(self
            .cache
            .drop_epochs_before(state.spec.model, table, after))
    }

    /// A table to change, or whose requests change.
    fn state(&mut self, table: TableId) -> Result<&mut TableState, NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.generation += 1;
        Ok(state)
    }

    /// Number of live rows in a table.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn len(&self, table: TableId) -> Result<u64, NodeError> {
        Ok(self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .table
            .len())
    }

    /// Every live row of a table, in key order: keys, row-major features, targets.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn rows(&self, table: TableId) -> Result<(Vec<u128>, Vec<f32>, Targets), NodeError> {
        let rows = self.rows_with_times(table)?;
        Ok((rows.keys, rows.features, rows.targets))
    }

    /// The rows a table offers each query under its strategy, for contexts
    /// assembled across partitions: every live row for `Exact`; for
    /// `Retrieved`, its core of `share` rows (the strategy's own count
    /// without one), drawn as a plan draws it, and each query's nearest rows
    /// outside the core. Shared rows come first, each once. A diverse core is
    /// spread over this partition's rows alone, so the cores partitions offer
    /// together are not one farthest-point sample of the whole table. With a
    /// cutoff, only rows from before it are offered, as a query with that
    /// cutoff reads.
    ///
    /// # Errors
    ///
    /// As `submit`, for the queries, or if the table's rows cannot be read.
    pub fn candidates(
        &mut self,
        table: TableId,
        queries: &[f32],
        share: Option<u64>,
        cutoff: Option<i64>,
    ) -> Result<Vec<Candidate>, NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let width = state.spec.features;
        if queries.is_empty() {
            return Err(NodeError::NoQueries);
        }
        if !queries.len().is_multiple_of(width) {
            return Err(NodeError::Width {
                expected: width,
                got: queries.len(),
            });
        }
        // The bins and diverse cores of the state a prediction would read.
        let wanted = state.horizon_of(cutoff.as_ref().map(std::slice::from_ref));
        let refit = state.refit_for(wanted);
        state.prepare_cores(Epoch(state.epoch.0 + u64::from(refit)));
        let live = state.served().rows();
        let live = match cutoff {
            Some(cutoff) => &state.times.before(cutoff) & &live,
            None => live,
        };
        let vectors: Vec<Vec<f32>> = queries
            .chunks(width)
            .map(|row| state.vector(row))
            .collect::<Result<_, _>>()?;
        // Its fields alone, so the table's cores can be drawn meanwhile.
        let scaled = Scaled {
            rows: &state.rows,
            standardizer: state.standardizer.as_ref(),
        };
        let mut offered = Vec::new();
        let shared = |offered: &mut Vec<Candidate>, rows: &Context| {
            offered.extend(rows.iter().map(|row| Candidate {
                query: None,
                row,
                score: 0.0,
            }));
        };
        let Strategy::Retrieved {
            core,
            neighbors,
            among,
            recency,
        } = state.spec.strategy
        else {
            shared(&mut offered, &live);
            return Ok(offered);
        };
        let live = match recency {
            Some(recency) => recent(&state.times, &live, recency, cutoff),
            None => live,
        };
        let core = match core {
            Some(core) => {
                let core = Core {
                    rows: share.unwrap_or(core.rows),
                    ..core
                };
                let mut cores = std::mem::take(&mut state.cores.1);
                let source = Source {
                    snapshot: &state.served(),
                    index: &state.index,
                    vectors: &scaled,
                    strata: &state.labels(),
                    keys: &state.table,
                    ranking: state.ranking.as_ref(),
                    times: &state.times,
                };
                let drawn = draw_core(core, &source, &mut cores, &live);
                state.cores.1 = cores;
                drawn?
            }
            None => Context::new(),
        };
        shared(&mut offered, &core);
        let outside = &live - &core;
        for (query, vector) in vectors.iter().enumerate() {
            let nearest = state.index.nearest_rows(
                vector,
                among.unwrap_or(neighbors),
                &outside,
                &scaled,
                &state.table,
            )?;
            let nearest = match among {
                Some(_) => state.times.latest_of(&nearest, neighbors as u64),
                None => nearest,
            };
            for row in nearest.iter() {
                let score = scaled
                    .vector(row)?
                    .iter()
                    .zip(vector)
                    .map(|(&value, &target)| (f64::from(value) - f64::from(target)).powi(2))
                    .sum();
                offered.push(Candidate {
                    query: Some(query as u32),
                    row,
                    score,
                });
            }
        }
        Ok(offered)
    }

    /// Rows of a table by id, with their keys and times, in the order given.
    ///
    /// # Errors
    ///
    /// If the table does not exist or a row cannot be read.
    pub fn rows_of(&self, table: TableId, rows: &[RowId]) -> Result<TableRows, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let keys = state.rows.rows(rows)?.iter().map(|row| row.key).collect();
        let times = rows.iter().map(|&row| state.times.time(row)).collect();
        let (features, targets) = state.gather(rows)?;
        Ok(TableRows {
            keys,
            features,
            targets,
            times,
        })
    }

    /// Every live row of a table, in key order, with each row's time.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn rows_with_times(&self, table: TableId) -> Result<TableRows, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let mut keyed: Vec<(u128, RowId)> =
            state.table.keys().map(|(key, row)| (key.0, row)).collect();
        keyed.sort_unstable();
        let rows: Vec<RowId> = keyed.iter().map(|&(_, row)| row).collect();
        let keys = keyed.into_iter().map(|(key, _)| key).collect();
        let times = rows.iter().map(|&row| state.times.time(row)).collect();
        let (features, targets) = state.gather(&rows)?;
        Ok(TableRows {
            keys,
            features,
            targets,
            times,
        })
    }

    /// Every live row of a table in page order, with each row's id and page,
    /// and what the table holds besides: what a compacted log records, so that
    /// `restore` builds the same table again without searching or renumbering.
    ///
    /// # Errors
    ///
    /// If there is no such table.
    pub fn compaction(&self, table: TableId) -> Result<Compaction, NodeError> {
        let source = self.compaction_source(table)?;
        let (rows, layout) = source.part(0, source.layout.ids.len())?;
        Ok(Compaction {
            table: source.table,
            rows,
            layout,
        })
    }

    /// What a compaction of a table reads, as `compaction` would write it,
    /// with its rows left to read part by part, as the table was now.
    ///
    /// # Errors
    ///
    /// If there is no such table.
    pub fn compaction_source(&self, table: TableId) -> Result<CompactionSource, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let (index, pages) = state.index.layout();
        let mut layout = RowLayout::default();
        if state.spec.series.is_some() {
            // A table of series places no rows in pages: its rows go in key
            // order, as one page without a layout.
            let mut keyed: Vec<(RowKey, RowId)> = state.table.keys().collect();
            keyed.sort_unstable();
            layout.ids.extend(keyed.iter().map(|&(_, row)| row));
            layout
                .pages
                .extend(std::iter::repeat_n(PageId(0), keyed.len()));
        }
        for (page, rows) in pages {
            layout.ids.extend_from_slice(rows);
            layout
                .pages
                .extend(std::iter::repeat_n(page.id, rows.len()));
            layout.new_pages.push(page);
        }
        let times = layout
            .ids
            .iter()
            .map(|&row| state.times.time(row))
            .collect();
        let fitted = state.fitted_columns.as_ref().map(|fitted| FittedColumns {
            rows: fitted.rows,
            means: fitted.means.clone(),
            deviations: fitted.deviations.clone(),
        });
        let table = TableLayout {
            rows: layout.ids.len() as u64,
            ids: u32::try_from(state.table.ids_used()).expect("row ids are u32"),
            index,
            scaling: state.standardizer.as_ref().map(|standardizer| Scaling {
                center: standardizer.center.clone(),
                scale: standardizer.scale.clone(),
                lookups: standardizer
                    .lookups
                    .iter()
                    .map(|(_, lookup)| (lookup.codes.clone(), lookup.vectors.clone()))
                    .collect(),
                rows: state.scaled_rows,
            }),
            column_sums: state.columns.sums.clone(),
            column_counts: state.columns.counts.clone(),
            fitted,
            changed: state.changed,
        };
        Ok(CompactionSource {
            table,
            layout,
            times,
            rows: state.rows.snapshot(),
            task: state.spec.task,
        })
    }

    /// Rebuilds a table from a compacted log's records, as `compaction` wrote
    /// them: the first with the table's layout, then any that continue it. The
    /// table takes no requests until its last row is back; then it holds the
    /// same rows under the same ids, on the same pages, as the table the log
    /// was compacted from, and fits its state again on its next request.
    ///
    /// Drops of the cached entries a table being restored over held are left
    /// for `take_drops`.
    ///
    /// # Errors
    ///
    /// If the rows or their layout are malformed, or continue no restore; the
    /// table is then left as the records found it.
    pub fn restore(
        &mut self,
        table: TableId,
        rows: &TableRows,
        layout: &RowLayout,
        table_layout: Option<&TableLayout>,
    ) -> Result<(), NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let times: Vec<Option<i64>> = rows.times.clone();
        state.check(&rows.keys, &rows.features, &rows.targets)?;
        check_times(&rows.keys, Some(&times))?;
        state.check_series(&rows.keys, Some(&times), true)?;
        if layout.ids.len() != rows.keys.len() || layout.pages.len() != rows.keys.len() {
            return Err(NodeError::Restore(
                "a record's rows and layout differ in length",
            ));
        }
        if let Some(layout) = table_layout {
            let width = state.spec.model_features();
            if layout.column_sums.len() != width || layout.column_counts.len() != width {
                return Err(NodeError::Restore(
                    "a layout's columns differ from the table's",
                ));
            }
            for key in self.cache.keys_of(table) {
                let dropped = self.drop_or_doom(&key);
                self.drops.extend(dropped);
            }
            let state = self.state(table)?;
            state.begin_restore(layout.clone())?;
        }
        let state = self.state(table)?;
        let Some(restoring) = &mut state.restoring else {
            return Err(NodeError::Restore("rows with a layout continue no restore"));
        };
        for page in &layout.new_pages {
            restoring.pages.insert(page.id, page.clone());
        }
        if layout.ids.iter().any(|row| row.0 >= state.rows.len()) {
            return Err(NodeError::Restore("a row id past those the layout took"));
        }
        state.rows.set(
            &layout.ids,
            &rows.features,
            &rows.targets.bits(),
            Some(&rows.keys),
        )?;
        for (index, (&row, &page)) in layout.ids.iter().zip(&layout.pages).enumerate() {
            state.table.restore(RowKey(rows.keys[index]), row)?;
            state.times.set(row, times[index]);
            if let Some(ranking) = &mut state.ranking {
                ranking.insert(row, rows.keys[index]);
            }
            restoring.members.entry(page).or_default().push(row);
        }
        restoring.restored += layout.ids.len() as u64;
        if restoring.restored == restoring.layout.rows {
            state.finish_restore()?;
        }
        Ok(())
    }

    /// Makes a table hold exactly these rows, as a snapshot says: deletes the
    /// others, updates rows whose features, target or time differ, and inserts
    /// new ones, then renumbers the rows as `renumber` does, so every copy built
    /// from the snapshot holds the same ids and pages.
    ///
    /// Drops of the entries the renumbering forgets are left for `take_drops`.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn replace_rows(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        check_times(keys, times)?;
        if let Some(state) = self.tables.get(&table) {
            state.check_series(keys, times, true)?;
        }
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.check(keys, features, &targets)?;
        let width = state.spec.features;
        let wanted: HashSet<u128> = keys.iter().copied().collect();
        let (mut updates, mut inserts) = (Vec::new(), Vec::new());
        for (index, &key) in keys.iter().enumerate() {
            let values = &features[index * width..(index + 1) * width];
            match state.table.row(RowKey(key)) {
                None => inserts.push(index),
                Some(row) => {
                    let stored = state.rows.rows(&[row])?.remove(0);
                    let unchanged = stored
                        .features
                        .iter()
                        .zip(values)
                        .all(|(old, new)| old.to_bits() == new.to_bits())
                        && stored.target == targets.bits()[index]
                        && times.is_none_or(|times| state.times.time(row) == times[index]);
                    if !unchanged {
                        updates.push(index);
                    }
                }
            }
        }
        if inserts.len() as u64 > state.ids_left() {
            return Err(NodeError::Table(TableError::Full));
        }
        // In key order: the order of deletes decides which pages merge.
        let mut deletes: Vec<u128> = state
            .table
            .keys()
            .map(|(key, _)| key.0)
            .filter(|key| !wanted.contains(key))
            .collect();
        deletes.sort_unstable();
        let updates = Picked::new(keys, features, width, &targets, times, &updates);
        let inserts = Picked::new(keys, features, width, &targets, times, &inserts);
        self.apply_delete(table, &deletes)?;
        updates.update(self, table)?;
        inserts.insert(self, table)?;
        self.renumber(table)
    }

    /// Renumbers a table's rows densely in key order and places them again,
    /// as copies of it built from a snapshot of its log do, so that from here
    /// on its row ids, pages and samples match theirs. Its cached entries hold
    /// the old ids, so they are dropped, or once read by the steps in flight;
    /// the drops are left for `take_drops`. Its fitted state is kept.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn renumber(&mut self, table: TableId) -> Result<(), NodeError> {
        self.state(table)?.renumber()?;
        for key in self.cache.keys_of(table) {
            let dropped = self.drop_or_doom(&key);
            self.drops.extend(dropped);
        }
        Ok(())
    }

    /// Adds rows, replacing the rows of keys the table already holds, at `times`
    /// if given; returns the rows' ids in the order of `keys`.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn upsert_at(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        self.check_upsert(table, keys, features, &targets, times)?;
        let state = &self.tables[&table];
        let (known, new): (Vec<usize>, Vec<usize>) =
            (0..keys.len()).partition(|&index| state.table.row(RowKey(keys[index])).is_some());
        let width = state.spec.features;
        let updates = Picked::new(keys, features, width, &targets, times, &known);
        let inserts = Picked::new(keys, features, width, &targets, times, &new);
        updates.update(self, table)?;
        inserts.insert(self, table)?;
        let state = &self.tables[&table];
        Ok(keys
            .iter()
            .map(|&key| state.table.row(RowKey(key)).expect("just written"))
            .collect())
    }

    /// The keys among `keys` that a table holds, in order.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn present(&self, table: TableId, keys: &[u128]) -> Result<Vec<u128>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        Ok(keys
            .iter()
            .copied()
            .filter(|&key| state.table.row(RowKey(key)).is_some())
            .collect())
    }

    /// Checks an insert without applying it, so it can be logged before it is applied.
    ///
    /// # Errors
    ///
    /// The error `insert` would return.
    pub fn check_insert(
        &self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        check_times(keys, times)?;
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.check(keys, features, targets)?;
        state.check_series(keys, times, false)?;
        if let Some(&key) = keys
            .iter()
            .find(|&&key| state.table.row(RowKey(key)).is_some())
        {
            return Err(NodeError::Table(TableError::DuplicateKey(RowKey(key))));
        }
        if keys.len() as u64 > state.ids_left() {
            return Err(NodeError::Table(TableError::Full));
        }
        Ok(())
    }

    /// Checks an update without applying it.
    ///
    /// # Errors
    ///
    /// The error `update` would return.
    pub fn check_update(
        &self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        check_times(keys, times)?;
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.check(keys, features, targets)?;
        state.check_series(keys, times, true)?;
        match keys
            .iter()
            .find(|&&key| state.table.row(RowKey(key)).is_none())
        {
            Some(&key) => Err(NodeError::UnknownKey(RowKey(key))),
            None => Ok(()),
        }
    }

    /// Checks an upsert without applying it.
    ///
    /// # Errors
    ///
    /// The error `upsert_at` would return.
    pub fn check_upsert(
        &self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        check_times(keys, times)?;
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.check(keys, features, targets)?;
        state.check_series(keys, times, true)?;
        let new = keys
            .iter()
            .filter(|&&key| state.table.row(RowKey(key)).is_none())
            .count();
        if new as u64 > state.ids_left() {
            return Err(NodeError::Table(TableError::Full));
        }
        Ok(())
    }

    /// Checks a delete without applying it.
    ///
    /// # Errors
    ///
    /// The error `delete` would return.
    pub fn check_delete(&self, table: TableId, keys: &[u128]) -> Result<(), NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let mut seen = BTreeSet::new();
        for &key in keys {
            if !seen.insert(key) {
                return Err(NodeError::DuplicateKey(RowKey(key)));
            }
            if state.table.row(RowKey(key)).is_none() {
                return Err(NodeError::UnknownKey(RowKey(key)));
            }
        }
        Ok(())
    }

    /// Adds labeled rows without times. The first batch sets the table's retrieval scaling.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing is inserted then.
    pub fn insert(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
    ) -> Result<Vec<RowId>, NodeError> {
        self.insert_at(table, keys, features, targets, None)
    }

    /// Adds labeled rows, each at its entry of `times` if given. A row without
    /// a time is before every cutoff.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing is inserted then.
    pub fn insert_at(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        self.check_insert(table, keys, features, &targets, times)?;
        self.apply_insert(table, keys, features, &targets, times)
    }

    /// Adds rows that `check_insert` and `check_times` passed.
    fn apply_insert(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        // Only once the write is checked, so every copy applying the same
        // writes renumbers at the same one.
        if self.tables.get(&table).is_some_and(TableState::sparse) {
            self.renumber(table)?;
        }
        let state = self.state(table)?;
        let width = state.spec.features;
        let labeled = state.labeled(targets);
        // A table of series places no rows in pages, so it fits no scaling.
        let paged = state.spec.series.is_none();
        if paged && state.standardizer.is_none() {
            // Labeled rows set the scaling, unless the batch has none.
            let fit: Vec<f32> = features
                .chunks(width)
                .zip(&labeled)
                .filter(|&(_, &labeled)| labeled)
                .flat_map(|(row, _)| row)
                .copied()
                .collect();
            // Labeled rows it was fit on, so the first labeled ones fit it again.
            state.scaled_rows = (fit.len() / width) as u64;
            let standardizer = if fit.is_empty() {
                Standardizer::fit(features, targets, &state.spec)
            } else {
                Standardizer::fit(&fit, &targets.picked(&labeled), &state.spec)
            };
            state.standardizer = Some(standardizer);
        }
        let mut rows = Vec::with_capacity(keys.len());
        let mut placed = Vec::with_capacity(keys.len());
        state.rows.push(features, &targets.bits(), keys)?;
        for (offset, &key) in keys.iter().enumerate() {
            let values = &features[offset * width..(offset + 1) * width];
            let row = state.table.apply(Change::Insert(RowKey(key)))?;
            let time = times.and_then(|times| times[offset]);
            if paged {
                placed.push((row, state.spec.bucket(time), state.vector(values)?));
            }
            state.times.set(row, time);
            if let (Some(index), Some(time)) = (&mut state.series, time) {
                index
                    .insert(row, RowKey(key), time, targets.observed(offset))
                    .expect("series rows were checked before the write");
            }
            if state.serves(labeled[offset]) {
                let print = state.print_of(row);
                state.print(time, print, true);
            }
            if !labeled[offset] {
                state.unlabeled.insert(row.0);
            }
            if state.serves(labeled[offset])
                && let Some(ranking) = &mut state.ranking
            {
                ranking.insert(row, key);
            }
            if labeled[offset] {
                state.columns.add(values, 1.0);
            }
            rows.push(row);
        }
        if let (Some(regions), Some(spec)) = (&mut state.regions, state.spec.regions) {
            let now = Instant::now();
            for ((_, _, vector), &labeled) in placed.iter().zip(&labeled) {
                if labeled || state.spec.unlabeled_context {
                    regions.change(vector, spec.rebuild, now);
                }
            }
        }
        let scaled = Scaled {
            rows: &state.rows,
            standardizer: state.standardizer.as_ref(),
        };
        state.index.insert_many(&placed, &scaled)?;
        state.changed += labeled
            .iter()
            .filter(|&&labeled| state.serves(labeled))
            .count() as u64;
        self.rescale_if_due(table)?;
        Ok(rows)
    }

    /// Fits a table's retrieval scaling again once the labeled rows that set
    /// it are half the table's labeled rows or fewer, which depends only on
    /// the writes, so every copy does the same. Once it was fit on `fit_rows`
    /// rows it stays, so a large table is not cut into pages anew each time it
    /// doubles.
    fn rescale_if_due(&mut self, table: TableId) -> Result<(), NodeError> {
        let state = self.state(table)?;
        let labeled = state.table.len() - state.unlabeled.len();
        if state.spec.series.is_none()
            && state.spec.retrieval == 0
            && labeled >= 2 * state.scaled_rows.max(1)
            && state.scaled_rows < state.spec.fit_rows.get() as u64
            && state
                .spec
                .scale_rows
                .is_none_or(|rows| state.scaled_rows < rows.get())
        {
            state.rescale()?;
            // Pages are cut anew, so the old pages' representations would
            // only hold slots until evicted.
            for key in self.cache.keys_of(table) {
                if key.kind == Kind::Rows {
                    let dropped = self.drop_or_doom(&key);
                    self.drops.extend(dropped);
                }
            }
        }
        Ok(())
    }

    /// Changes existing rows: new features and targets, and a new version.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn update(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
    ) -> Result<Vec<RowId>, NodeError> {
        self.update_at(table, keys, features, targets, None)
    }

    /// Changes existing rows, and their times to the entries of `times` if given.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn update_at(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        self.check_update(table, keys, features, &targets, times)?;
        self.apply_update(table, keys, features, &targets, times)
    }

    /// Changes rows that `check_update` and `check_times` passed.
    fn apply_update(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        let state = self.state(table)?;
        let width = state.spec.features;
        let labeled = state.labeled(targets);
        let mut rows = Vec::with_capacity(keys.len());
        let mut changed = 0;
        for (offset, &key) in keys.iter().enumerate() {
            let values = &features[offset * width..(offset + 1) * width];
            let before = state.table.row(RowKey(key)).map(|row| state.print_of(row));
            let row = state.table.apply(Change::Update(RowKey(key)))?;
            let was = !state.unlabeled.contains(row.0);
            let old = state.row(row)?;
            let old_time = state.times.time(row);
            let time = times.map_or(old_time, |times| times[offset]);
            // Times' fingerprints sum the rows plans read.
            if let Some(before) = before.filter(|_| state.serves(was)) {
                state.print(old_time, before, false);
            }
            if state.serves(labeled[offset]) {
                let after = state.print_of(row);
                state.print(time, after, true);
            }
            let moved = old
                .iter()
                .zip(values)
                .any(|(old, new)| old.to_bits() != new.to_bits())
                || state.spec.bucket(time) != state.spec.bucket(old_time);
            // A row whose features and window are unchanged keeps its page; a
            // table of series has none.
            let paged = state.spec.series.is_none();
            if moved && paged {
                let scaled = Scaled {
                    rows: &state.rows,
                    standardizer: state.standardizer.as_ref(),
                };
                state.index.remove(row, &scaled)?;
            }
            let target = targets.bits()[offset];
            state.rows.set(&[row], values, &[target], None)?;
            if let Some(index) = &mut state.series {
                index.observe(RowKey(key), targets.observed(offset));
            }
            if moved && paged {
                let bucket = state.spec.bucket(time);
                let vector = state.vector(values)?;
                let scaled = Scaled {
                    rows: &state.rows,
                    standardizer: state.standardizer.as_ref(),
                };
                state.index.insert(row, bucket, &vector, &scaled)?;
            }
            let is = labeled[offset];
            if is {
                state.unlabeled.remove(row.0);
            } else {
                state.unlabeled.insert(row.0);
            }
            if let Some(ranking) = &mut state.ranking {
                match (
                    state.spec.unlabeled_context || was,
                    state.spec.unlabeled_context || is,
                ) {
                    (false, true) => ranking.insert(row, key),
                    (true, false) => ranking.remove(row, key),
                    _ => {}
                }
            }
            if was {
                state.columns.add(&old, -1.0);
            }
            if is {
                state.columns.add(values, 1.0);
            }
            state.times.set(row, time);
            // A row a plan reads before or after counts toward a refit and its
            // region's rebuild: a label is when an unlabeled row enters
            // contexts, or one update to it for a model that reads it unlabeled.
            if state.serves(was) || state.serves(is) {
                state.note_change(values)?;
                changed += 1;
            }
            rows.push(row);
        }
        state.changed += changed;
        self.purge(table, &rows);
        // Labels that arrive may make enough labeled rows to scale again.
        self.rescale_if_due(table)?;
        Ok(rows)
    }

    /// Sets rows' targets, keeping their features and times, such as labels
    /// that arrive after their rows.
    ///
    /// # Errors
    ///
    /// As `update`. Nothing changes then.
    pub fn label(
        &mut self,
        table: TableId,
        keys: &[u128],
        targets: Targets,
    ) -> Result<Vec<RowId>, NodeError> {
        let features = self.features(table, keys)?;
        self.update_at(table, keys, &features, targets, None)
    }

    /// Rows' features by key, row-major.
    ///
    /// # Errors
    ///
    /// If the table or a key does not exist.
    pub fn features(&self, table: TableId, keys: &[u128]) -> Result<Vec<f32>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let mut features = Vec::with_capacity(keys.len() * state.spec.features);
        for &key in keys {
            let row = state
                .table
                .row(RowKey(key))
                .ok_or(NodeError::UnknownKey(RowKey(key)))?;
            features.extend(state.row(row)?);
        }
        Ok(features)
    }

    /// Deletes rows.
    ///
    /// # Errors
    ///
    /// If a key does not exist or repeats. Nothing changes then.
    pub fn delete(&mut self, table: TableId, keys: &[u128]) -> Result<Vec<RowId>, NodeError> {
        self.check_delete(table, keys)?;
        self.apply_delete(table, keys)
    }

    /// Deletes rows that `check_delete` passed.
    fn apply_delete(&mut self, table: TableId, keys: &[u128]) -> Result<Vec<RowId>, NodeError> {
        let state = self.state(table)?;
        let mut rows = Vec::with_capacity(keys.len());
        let mut changed = 0;
        for &key in keys {
            let before = state.table.row(RowKey(key)).map(|row| state.print_of(row));
            let row = state.table.apply(Change::Delete(RowKey(key)))?;
            if state.spec.series.is_none() {
                let scaled = Scaled {
                    rows: &state.rows,
                    standardizer: state.standardizer.as_ref(),
                };
                state.index.remove(row, &scaled)?;
            }
            let time = state.times.time(row);
            state.times.remove(row);
            if let Some(index) = &mut state.series {
                index.remove(RowKey(key));
            }
            let labeled = !state.unlabeled.remove(row.0);
            if let Some(print) = before.filter(|_| state.serves(labeled)) {
                state.print(time, print, false);
            }
            if let Some(ranking) = &mut state.ranking {
                ranking.remove(row, key);
            }
            let values = state.row(row)?;
            if labeled {
                state.columns.add(&values, -1.0);
            }
            if state.serves(labeled) {
                state.note_change(&values)?;
                changed += 1;
            }
            rows.push(row);
        }
        state.changed += changed;
        self.purge(table, &rows);
        Ok(rows)
    }

    /// Deletes every row, in key order, as `delete` deletes them.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn truncate(&mut self, table: TableId) -> Result<Vec<RowId>, NodeError> {
        let mut keys: Vec<u128> = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .table
            .keys()
            .map(|(key, _)| key.0)
            .collect();
        keys.sort_unstable();
        self.delete(table, &keys)
    }

    /// Queues query rows for a table's next step.
    ///
    /// # Errors
    ///
    /// If the table does not exist or has no rows, or the queries have the wrong width.
    pub fn submit(&mut self, table: TableId, queries: &[f32]) -> Result<RequestId, NodeError> {
        self.submit_before(table, queries, None)
    }

    /// Queues query rows that each read only rows from before their cutoff.
    ///
    /// # Errors
    ///
    /// As `submit`, or if there is not one cutoff per query.
    pub fn submit_before(
        &mut self,
        table: TableId,
        queries: &[f32],
        cutoffs: Option<&[i64]>,
    ) -> Result<RequestId, NodeError> {
        let request = RequestId(self.next_request);
        let state = self.state(table)?;
        if state.spec.series.is_some() {
            return Err(ForecastError::OnlyForecasts.into());
        }
        state.check_queries(queries, cutoffs)?;
        if state.table.is_empty() {
            return Err(NodeError::EmptyTable(table));
        }
        let queued = (request, queries.to_vec(), cutoffs.map(<[i64]>::to_vec));
        // A given request is planned alone, so others wait until it is.
        if state.given.is_some() || !state.givens.is_empty() {
            state.later.push(queued);
        } else {
            state.pending.push(queued);
        }
        self.next_request += 1;
        Ok(request)
    }

    /// The keys of the rows each query's context holds, in the order the
    /// model reads them, ascending, planned as a request of these queries
    /// alone would be, with its refit, but neither queued nor
    /// encoded, and changing nothing a later plan reads. A query whose cutoff
    /// precedes every row has none.
    ///
    /// # Errors
    ///
    /// As `submit_before`, or if a row cannot be read.
    pub fn contexts(
        &mut self,
        table: TableId,
        queries: &[f32],
        cutoffs: Option<&[i64]>,
    ) -> Result<Vec<Vec<u128>>, NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        if state.spec.series.is_some() {
            return Err(ForecastError::OnlyForecasts.into());
        }
        state.check_queries(queries, cutoffs)?;
        if let Some(error) = state.empty(table, cutoffs) {
            return Err(error);
        }
        let alone = (
            RequestId(u64::MAX),
            queries.to_vec(),
            cutoffs.map(<[i64]>::to_vec),
        );
        let pending = std::mem::replace(&mut state.pending, vec![alone]);
        let refit = state.refit_for(state.horizon_of(cutoffs));
        let epoch = Epoch(state.epoch.0 + u64::from(refit));
        // Planned against copies of what plans keep, so a refit this plan
        // foresees drops no core, bins or region the current state reads,
        // and counted as no region's hit or miss.
        let kept = (
            state.cores.clone(),
            state.edges.clone(),
            state.regions.clone(),
            state.region_stats,
            state.planned_counts,
        );
        let model = state.spec.model;
        let cached = |rows| {
            let key = CacheKey {
                model,
                table,
                epoch,
                kind: Kind::Context,
                rows,
            };
            self.cache.contains(&key) && !self.doomed.contains(&key)
        };
        let planned = state.plan_pending(
            epoch,
            &state.served(),
            &cached,
            Instant::now(),
            &mut PlanParts::default(),
        );
        (
            state.cores,
            state.edges,
            state.regions,
            state.region_stats,
            state.planned_counts,
        ) = kept;
        state.pending = pending;
        let planned = planned?;
        let alone = vec![(
            RequestId(u64::MAX),
            queries.to_vec(),
            cutoffs.map(<[i64]>::to_vec),
        )];
        let pending = std::mem::replace(&mut state.pending, alone);
        let blind = !state.blind(&planned).is_empty();
        state.pending = pending;
        if blind {
            return Err(NodeError::EmptyContext(table));
        }
        planned
            .queries
            .iter()
            .map(|unit| {
                let rows: Vec<RowId> = planned.units[unit.0].rows.iter().collect();
                let mut keys: Vec<u128> =
                    state.rows.rows(&rows)?.iter().map(|row| row.key).collect();
                keys.sort_unstable();
                Ok(keys)
            })
            .collect()
    }

    /// Queues queries whose contexts were chosen elsewhere: each query reads
    /// exactly its given rows, encoded together, such as rows gathered across
    /// a table's partitions. The request is planned alone: behind the table's
    /// pending requests, if any, and before requests that come after it.
    ///
    /// # Errors
    ///
    /// As `submit`, `Count` if there is not one context per query, or `Given`
    /// if a context names a row the table does not hold, or a query is given
    /// no rows.
    pub fn submit_given(
        &mut self,
        table: TableId,
        queries: &[f32],
        given: Vec<Context>,
    ) -> Result<RequestId, NodeError> {
        let state = self.state(table)?;
        let width = state.spec.features;
        if queries.len() != given.len() * width {
            return Err(NodeError::Count {
                what: Counted::Contexts,
                expected: queries.len() / width,
                got: given.len(),
            });
        }
        // Rows the table holds; plans leave out those they may not read.
        let live = state.table.snapshot().rows();
        let held = given
            .iter()
            .all(|rows| !rows.is_empty() && (rows - &live).is_empty());
        if !held {
            return Err(NodeError::Given);
        }
        state.check_queries(queries, None)?;
        if state.table.is_empty() {
            return Err(NodeError::EmptyTable(table));
        }
        let request = RequestId(self.next_request);
        self.next_request += 1;
        let state = self.state(table)?;
        let pending = (request, queries.to_vec(), None);
        match &mut state.given {
            // Given requests not yet planned join one batch.
            Some(batch) => {
                state.pending.push(pending);
                batch.extend(given);
            }
            None if state.pending.is_empty() => {
                state.pending.push(pending);
                state.given = Some(given);
            }
            // Behind requests of other contexts, until their turn.
            None => state.givens.push_back((request, queries.to_vec(), given)),
        }
        Ok(request)
    }

    /// Queues forecasts of a series table's series, each from its cutoff, or
    /// from the table's latest time plus a step without one, over `horizon`
    /// steps.
    ///
    /// # Errors
    ///
    /// If the table does not exist or holds no series, no series is named, or
    /// there is not one cutoff per series.
    pub fn submit_forecast(
        &mut self,
        table: TableId,
        series: &[u64],
        cutoffs: Option<&[i64]>,
        horizon: NonZeroUsize,
        groups: Option<&[Vec<u64>]>,
    ) -> Result<RequestId, NodeError> {
        let request = RequestId(self.next_request);
        let state = self.state(table)?;
        let max_horizon = state
            .spec
            .series
            .as_ref()
            .ok_or(ForecastError::NotSeries)?
            .max_horizon;
        if horizon > max_horizon {
            return Err(ForecastError::HorizonTooLong {
                horizon: horizon.get(),
                limit: max_horizon.get(),
            }
            .into());
        }
        if series.is_empty() {
            return Err(NodeError::NoQueries);
        }
        if let Some(cutoffs) = cutoffs
            && cutoffs.len() != series.len()
        {
            return Err(NodeError::Count {
                what: Counted::Cutoffs,
                expected: series.len(),
                got: cutoffs.len(),
            });
        }
        if let Some(groups) = groups
            && groups.len() != series.len()
        {
            return Err(NodeError::Count {
                what: Counted::Groups,
                expected: series.len(),
                got: groups.len(),
            });
        }
        let cutoffs = match cutoffs {
            Some(cutoffs) => cutoffs.iter().map(|&cutoff| Some(cutoff)).collect(),
            None => vec![None; series.len()],
        };
        let groups = groups.map(|groups| {
            groups
                .iter()
                .map(|group| group.iter().map(|&member| SeriesKey(member)).collect())
                .collect()
        });
        state.forecasts.push((
            request,
            SeriesRequest {
                series: series.iter().map(|&series| SeriesKey(series)).collect(),
                cutoffs,
                horizon,
                groups,
            },
        ));
        self.next_request += 1;
        Ok(request)
    }

    /// Queues queries that each read exactly the rows of the keys given
    /// them, encoded together, such as rows an external index chose. A
    /// context past the model's trained length warns, or under `Clamp`, keeps
    /// its first keys up to that length, in the order given: an index's
    /// ranking, best first.
    ///
    /// # Errors
    ///
    /// As `submit_given`, or `UnknownKey` if a key names no row.
    pub fn submit_keys(
        &mut self,
        table: TableId,
        queries: &[f32],
        contexts: &[Vec<u128>],
    ) -> Result<RequestId, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let cap = match (state.spec.past_cap, state.spec.context_rows) {
            (PastCap::Clamp, Some(rows)) => usize::try_from(rows.get()).unwrap_or(usize::MAX),
            _ => usize::MAX,
        };
        let given = contexts
            .iter()
            .map(|keys| {
                let mut rows = BTreeSet::new();
                for &key in keys {
                    let row = state
                        .table
                        .row(RowKey(key))
                        .ok_or(NodeError::UnknownKey(RowKey(key)))?;
                    if rows.len() < cap {
                        rows.insert(row);
                    }
                }
                Ok(rows.into_iter().collect::<Context>())
            })
            .collect::<Result<Vec<Context>, NodeError>>()?;
        self.submit_given(table, queries, given)
    }

    /// Drops a queued request before it is planned; returns whether it was queued.
    pub fn cancel(&mut self, request: RequestId) -> bool {
        self.tables.values_mut().any(|state| {
            let queued = state.pending.len()
                + state.later.len()
                + state.givens.len()
                + state.forecasts.len();
            state.drop_pending(&[request]);
            state.later.retain(|(id, _, _)| *id != request);
            state.givens.retain(|(id, _, _)| *id != request);
            state.forecasts.retain(|(id, _)| *id != request);
            let left = state.pending.len()
                + state.later.len()
                + state.givens.len()
                + state.forecasts.len();
            let cancelled = left < queued;
            state.promote();
            state.generation += u64::from(cancelled);
            cancelled
        })
    }

    /// Refits a table, as asked: renumbers its rows, fits its retrieval
    /// scaling on them, places every row again, and starts a new epoch.
    ///
    /// Only these refits renumber and rescale, not those a refit policy
    /// starts, since they are what table logs record: copies that replay a log
    /// place rows alike.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn refit(&mut self, table: TableId) -> Result<Vec<Movement>, NodeError> {
        self.state(table)?.renumber()?;
        self.advance_epoch(table)
    }

    /// Starts a new epoch for a table after its state is to be refit.
    ///
    /// Returns drops of its old entries that nothing reads; the rest are dropped
    /// when the steps reading them complete.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn advance_epoch(&mut self, table: TableId) -> Result<Vec<Movement>, NodeError> {
        let state = self.state(table)?;
        state.epoch.0 += 1;
        // The old epoch's kept units are dropped, so none is left to extend.
        state.recent.clear();
        state.writing.clear();
        let (model, epoch) = (state.spec.model, state.epoch);
        self.advanced.push(table);
        Ok(self.cache.drop_epochs_before(model, table, epoch))
    }

    /// Plans pending requests into one step, or None if none can run yet.
    ///
    /// Requests whose queries would read no rows are refused instead; see
    /// `take_refused`. Tables are taken in turn from the one after the last
    /// step's first. A table's requests join the step if the contexts they
    /// newly encode fit the remaining budget; the step's first table always
    /// joins, so a table larger than the budget still runs, alone. A table
    /// whose requests would read what a step in flight encodes or fits waits
    /// for that step, since if it fails the worker never holds them.
    ///
    /// # Errors
    ///
    /// If the cache refuses an operation for a reason other than room. The
    /// requests stay queued, and nothing of the step is left pinned or admitted.
    pub fn schedule(&mut self) -> Result<Option<Step>, NodeError> {
        self.parts = PlanParts::new(self.parts.timed);
        let since = self.parts.start();
        for (&id, state) in &mut self.tables {
            if !state.pending.is_empty() {
                self.refused.extend(state.take_empty(id));
            }
        }
        let mut waiting: Vec<TableId> = self
            .tables
            .range(self.turn..)
            .chain(self.tables.range(..self.turn))
            // A table being restored takes requests once its last row is back.
            .filter(|(_, state)| {
                (!state.pending.is_empty() || !state.forecasts.is_empty())
                    && state.restoring.is_none()
            })
            .map(|(&id, _)| id)
            .collect();
        // A table whose fit would read more rows than it may before the
        // worker measures its limits waits for them.
        waiting.retain(|id| {
            let state = &self.tables[id];
            let held = state.unmeasured_fit();
            if held && self.unmeasured.insert(*id) {
                self.untold.push(*id);
            }
            !held
        });
        if self.scheduling == Scheduling::Arrival {
            waiting.sort_by_key(|id| self.tables[id].first_request());
        }
        // Region contexts due a rebuild take a step of their own once no
        // request waits, or once one has waited out its table's
        // `RebuildWait`, so requests keep reading the old contexts meanwhile.
        let now = Instant::now();
        let due = self.oldest_due(now);
        for (id, state) in &mut self.tables {
            if !due.iter().any(|(table, _)| table == id) {
                state.rebuild_waited = 0;
            }
        }
        let waited = due.iter().any(|(id, due)| {
            let state = &self.tables[id];
            state
                .spec
                .regions
                .is_some_and(|spec| spec.wait.over(state.rebuild_waited, *due, now))
        });
        self.parts.add("checks", since);
        if !due.is_empty()
            && (waiting.is_empty() || waited)
            && let Some(step) = self.rebuild_step(now)?
        {
            for state in self.tables.values_mut() {
                state.rebuild_waited = 0;
            }
            return Ok(Some(step));
        }
        let mut step = Step::default();
        let mut scheduled = Vec::new();
        if let Err(error) = self.plan_step(&waiting, &mut step, &mut scheduled, now) {
            // Undo the half-planned step as a failed one, and keep the drops it
            // already made for the executor, which never sees the step.
            let dropped = self.fail(&step);
            self.drops.extend(dropped);
            self.drops.extend(
                step.movements
                    .into_iter()
                    .filter(|movement| movement.to.is_none()),
            );
            return Err(error);
        }
        let Some(&first) = scheduled.first() else {
            return Ok(None);
        };
        let since = self.parts.start();
        for &id in &scheduled {
            if let Some(state) = self.tables.get_mut(&id) {
                state.gave = state.given.is_some();
                state.pending.clear();
                state.forecasts.clear();
                state.promote();
                state.generation += 1;
            }
        }
        self.moving.extend(
            step.movements
                .iter()
                .filter(|movement| movement.to.is_some())
                .map(|movement| movement.key),
        );
        self.encoding.extend(encoded(&step));
        self.fitting.extend(fitted_tables(&step));
        self.turn = TableId(first.0 + 1);
        for (id, _) in &due {
            if let Some(state) = self.tables.get_mut(id) {
                state.rebuild_waited += 1;
            }
        }
        self.parts.add("assemble", since);
        Ok(Some(step))
    }

    /// Each table with a region context due a rebuild, among those of current
    /// states no step rebuilds yet, and when its longest due fell due.
    fn oldest_due(&mut self, now: Instant) -> Vec<(TableId, Instant)> {
        let rebuilt = &self.rebuilt_regions;
        self.tables
            .iter_mut()
            .filter_map(|(&id, state)| {
                let spec = state.spec.regions?;
                state.age_regions(now);
                let regions = state.regions.as_mut()?;
                if regions.epoch != state.epoch || state.fitted != Some(state.epoch) {
                    return None;
                }
                regions
                    .due(spec.rebuild, now)
                    .into_iter()
                    .filter(|&(region, _)| !rebuilt.contains(&(id, region)))
                    .map(|(_, due)| due)
                    .min()
                    .map(|due| (id, due))
            })
            .collect()
    }

    /// A step that only builds again region contexts due a rebuild, around
    /// their centers, as many as the step budget allows; None if none can be.
    ///
    /// # Errors
    ///
    /// As `schedule`.
    fn rebuild_step(&mut self, now: Instant) -> Result<Option<Step>, NodeError> {
        let budget = self.step_bytes.unwrap_or(match self.cache.device_bytes() {
            0 => u64::MAX,
            bytes => bytes,
        });
        let slot_bytes = self.cache.slot_bytes();
        let mut used = 0;
        let mut step = Step::default();
        let tables: Vec<TableId> = self
            .tables
            .range(self.turn..)
            .chain(self.tables.range(..self.turn))
            .map(|(&id, _)| id)
            .collect();
        for id in tables {
            if (!step.units.is_empty() && used >= budget) || self.fitting.contains(&id) {
                continue;
            }
            let state = self.tables.get_mut(&id).expect("listed above");
            let Some(spec) = state.spec.regions else {
                continue;
            };
            let epoch = state.epoch;
            let current = state.fitted == Some(epoch)
                && state.restoring.is_none()
                && !state.refit_for(None)
                && state
                    .regions
                    .as_ref()
                    .is_some_and(|regions| regions.epoch == epoch);
            if !current {
                continue;
            }
            let since = self.parts.start();
            let regions = state.regions.as_mut().expect("checked above");
            let generation = regions.generation;
            let due: Vec<(usize, Instant, u64)> = regions
                .due(spec.rebuild, now)
                .into_iter()
                .filter(|&(region, _)| !self.rebuilt_regions.contains(&(id, region)))
                .filter_map(|(region, due)| {
                    Some((region, due, regions.built[region].as_ref()?.changes))
                })
                .collect();
            let centers: Vec<Vec<f32>> = due
                .iter()
                .map(|&(region, _, _)| regions.centers[region].clone())
                .collect();
            self.parts.add("regions", since);
            if due.is_empty() {
                continue;
            }
            let since = self.parts.start();
            let snapshot = state.served();
            self.parts.add("snapshot", since);
            let since = self.parts.start();
            state.prepare_cores(epoch);
            let mut planned =
                state.plan_around(&snapshot, &centers, &[], &[], &vec![None; centers.len()])?;
            self.parts.add("retrieve", since);
            let since = self.parts.start();
            let key_of = |rows| CacheKey {
                model: state.spec.model,
                table: id,
                epoch,
                kind: Kind::Context,
                rows,
            };
            let mut kept = Vec::new();
            let mut keys = Vec::new();
            // Regions join while the budget lasts; the step's first always does.
            for (&(region, due, seen), unit) in due.iter().zip(&planned.queries) {
                let unit = &planned.units[unit.0];
                let key = key_of(unit.key);
                if self.encoding.contains(&key) {
                    continue;
                }
                let encodes = !self.rebuilding.contains_key(&key)
                    && !keys.contains(&key)
                    && !self.cache.contains(&key);
                let cost = if encodes {
                    state.spec.sizes.bytes(unit.rows.len() as usize, slot_bytes)
                } else {
                    0
                };
                if (!step.units.is_empty() || !keys.is_empty()) && used + cost > budget {
                    break;
                }
                used += cost;
                self.rebuilding.entry(key).or_default().push(Rebuilding {
                    table: id,
                    region,
                    epoch,
                    generation,
                    rows: unit.rows.clone(),
                    seen,
                    due,
                });
                self.rebuilt_regions.insert((id, region));
                if encodes {
                    keys.push(key);
                    kept.push(unit.clone());
                }
            }
            planned.units = kept;
            self.parts.add("checks", since);
            let busy = Busy {
                moving: &self.moving,
                doomed: &self.doomed,
                encoding: &self.encoding,
            };
            let placed = place_units(
                &mut self.cache,
                busy,
                id,
                state,
                &snapshot,
                planned.units,
                keys,
                &BTreeMap::new(),
                false,
                &mut step,
                &mut self.parts,
            );
            if let Err(error) = placed {
                let dropped = self.fail(&step);
                self.drops.extend(dropped);
                return Err(error);
            }
        }
        let since = self.parts.start();
        // Contexts the cache held already replace their regions' at once.
        let held: Vec<CacheKey> = self
            .rebuilding
            .keys()
            .filter(|key| {
                self.cache.contains(key) && !step.units.iter().any(|unit| unit.key == **key)
            })
            .filter(|key| !self.encoding.contains(key))
            .copied()
            .collect();
        for key in held {
            let dropped = self.finish_rebuild(key, now);
            self.drops.extend(dropped);
        }
        if step.units.is_empty() {
            self.parts.add("assemble", since);
            return Ok(None);
        }
        self.moving.extend(
            step.movements
                .iter()
                .filter(|movement| movement.to.is_some())
                .map(|movement| movement.key),
        );
        self.encoding.extend(encoded(&step));
        self.parts.add("assemble", since);
        Ok(Some(step))
    }

    /// Makes the contexts rebuilt as `key` their regions' built ones, and
    /// drops the entries they replace; returns the drops.
    fn finish_rebuild(&mut self, key: CacheKey, now: Instant) -> Vec<Movement> {
        let mut replaced = Vec::new();
        for rebuilt in self.rebuilding.remove(&key).unwrap_or_default() {
            self.rebuilt_regions
                .remove(&(rebuilt.table, rebuilt.region));
            let Some(state) = self.tables.get_mut(&rebuilt.table) else {
                continue;
            };
            // Regions drawn again since the rebuild was planned are other regions.
            let Some(regions) = state.regions.as_mut().filter(|regions| {
                regions.epoch == rebuilt.epoch && regions.generation == rebuilt.generation
            }) else {
                continue;
            };
            let rows = rebuilt.rows.len();
            let old = regions.built[rebuilt.region].clone();
            let times = state.times.times_of(&rebuilt.rows);
            regions.rebuilt(
                rebuilt.region,
                (rebuilt.rows, key.rows, times),
                rebuilt.seen,
                state
                    .spec
                    .regions
                    .map_or(Rebuild::Never, |spec| spec.rebuild),
                now,
            );
            if let Some(old) = old.filter(|old| old.key != key.rows) {
                replaced.push(CacheKey {
                    rows: old.key,
                    ..key
                });
            }
            let stats = &mut state.region_stats;
            stats.rebuilds += 1;
            stats.rebuilt_rows += rows;
            stats.overdue += now.saturating_duration_since(rebuilt.due);
        }
        // An entry another region still reads stays.
        replaced.retain(|old| {
            !self.tables.get(&old.table).is_some_and(|state| {
                state.regions.as_ref().is_some_and(|regions| {
                    regions
                        .built
                        .iter()
                        .flatten()
                        .any(|built| built.key == old.rows)
                })
            })
        });
        replaced
            .into_iter()
            .flat_map(|old| self.drop_or_doom(&old))
            .collect()
    }

    /// Adds each waiting table's requests to the step while the budget allows.
    fn plan_step(
        &mut self,
        waiting: &[TableId],
        step: &mut Step,
        scheduled: &mut Vec<TableId>,
        now: Instant,
    ) -> Result<(), NodeError> {
        let budget = self.step_bytes.unwrap_or(match self.cache.device_bytes() {
            0 => u64::MAX,
            bytes => bytes,
        });
        let slot_bytes = self.cache.slot_bytes();
        let mut used = 0;
        for &id in waiting {
            // Planning costs time, so tables wait without it once the budget is
            // spent, or while a step in flight fits them.
            if (!scheduled.is_empty() && used >= budget) || self.fitting.contains(&id) {
                continue;
            }
            let Some(state) = self.tables.get_mut(&id) else {
                continue;
            };
            if state.spec.series.is_some() {
                let since = self.parts.start();
                let planned = self.plan_series(id, budget, slot_bytes, &mut used, step, scheduled);
                self.parts.add("series", since);
                planned?;
                continue;
            }
            // One state serves a step: requests whose cutoffs need a state fit
            // on other rows wait until these are served. Units fit on their
            // own rows share no state, so serve every cutoff together.
            let since = self.parts.start();
            let wanted = state.horizon_of(state.pending[0].2.as_deref());
            let mixed = state.spec.summary == SummaryScope::Table
                && state
                    .pending
                    .iter()
                    .any(|(_, _, cutoffs)| state.horizon_of(cutoffs.as_deref()) != wanted);
            if mixed {
                let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut state.pending)
                    .into_iter()
                    .partition(|(_, _, cutoffs)| state.horizon_of(cutoffs.as_deref()) == wanted);
                state.pending = now;
                state.later.extend(later);
                state.generation += 1;
            }
            // A refit the policy asks for, or one for other rows, starts a new
            // epoch, but only once the table joins the step, so a table left
            // waiting keeps its cache.
            let refit = state.refit_for(wanted);
            let epoch = Epoch(state.epoch.0 + u64::from(refit));
            self.parts.add("checks", since);
            let since = self.parts.start();
            let snapshot = state.served();
            self.parts.add("snapshot", since);
            let planned = loop {
                // A table the budget left waiting keeps its plan while nothing changes.
                let planned = match state.deferred.take() {
                    Some((generation, planned)) if generation == state.generation => planned,
                    _ if state.given.is_some() => {
                        let since = self.parts.start();
                        let planned =
                            plan_given(&snapshot, state.given.as_ref().expect("matched above"));
                        self.parts.add("retrieve", since);
                        planned?
                    }
                    _ => {
                        let model = state.spec.model;
                        let cached = |rows| {
                            let key = CacheKey {
                                model,
                                table: id,
                                epoch,
                                kind: Kind::Context,
                                rows,
                            };
                            self.cache.contains(&key) && !self.doomed.contains(&key)
                        };
                        state.plan_pending(epoch, &snapshot, &cached, now, &mut self.parts)?
                    }
                };
                // Requests a context of unlabeled rows alone would answer are
                // refused, and the rest planned again.
                let blind = state.blind(&planned);
                if blind.is_empty() {
                    break Some(planned);
                }
                state.drop_pending(&blind);
                state.generation += 1;
                self.refused.extend(
                    blind
                        .into_iter()
                        .map(|request| (request, NodeError::EmptyContext(id))),
                );
                if state.pending.is_empty() {
                    state.promote();
                    if state.pending.is_empty() {
                        break None;
                    }
                }
            };
            let Some(planned) = planned else {
                continue;
            };
            let since = self.parts.start();
            if let (PastCap::Warn, Some(cap)) = (state.spec.past_cap, state.spec.context_rows)
                && !state.past_cap
                && planned.units.iter().any(|unit| unit.rows.len() > cap.get())
            {
                state.past_cap = true;
                self.past_cap.push(id);
            }
            let keys: Vec<CacheKey> = planned
                .units
                .iter()
                .map(|unit| CacheKey {
                    model: state.spec.model,
                    table: id,
                    epoch,
                    kind: Kind::Context,
                    rows: unit.key,
                })
                .collect();
            // Contexts a step in flight encodes, or, for those not cached
            // yet, pages it embeds, are read only once it completes.
            let waits = keys.iter().any(|key| self.encoding.contains(key))
                || (keys.iter().any(|key| !self.cache.contains(key))
                    && self
                        .encoding
                        .iter()
                        .any(|key| key.table == id && key.kind == Kind::Rows));
            self.parts.add("checks", since);
            if waits {
                state.deferred = Some((state.generation, planned));
                continue;
            }
            let cost: u64 = planned
                .units
                .iter()
                .zip(&keys)
                .filter(|(_, key)| !self.cache.contains(key))
                .map(|(unit, _)| state.spec.sizes.bytes(unit.rows.len() as usize, slot_bytes))
                .sum();
            if !scheduled.is_empty() && used + cost > budget {
                state.deferred = Some((state.generation, planned));
                continue;
            }
            used += cost;
            let state = self.tables.get_mut(&id).expect("planned above");
            let counts = std::mem::take(&mut state.planned_counts);
            state.region_stats.count(&counts);
            if refit {
                step.movements.extend(self.advance_epoch(id)?);
            }
            let busy = Busy {
                moving: &self.moving,
                doomed: &self.doomed,
                encoding: &self.encoding,
            };
            let state = self.tables.get_mut(&id).expect("planned above");
            schedule_table(
                &mut self.cache,
                busy,
                id,
                state,
                &snapshot,
                planned,
                keys,
                step,
                &mut self.parts,
            )?;
            scheduled.push(id);
        }
        Ok(())
    }

    /// Adds a series table's forecast requests to the step while the budget
    /// allows: its groups as units to encode, reuse or forecast whole, and a
    /// forecast per series.
    fn plan_series(
        &mut self,
        id: TableId,
        budget: u64,
        slot_bytes: u64,
        used: &mut u64,
        step: &mut Step,
        scheduled: &mut Vec<TableId>,
    ) -> Result<(), NodeError> {
        let state = self.tables.get_mut(&id).expect("waiting tables exist");
        let spec = state
            .spec
            .series
            .clone()
            .expect("a series table has a spec");
        let index = state.series.as_ref().expect("a series table has an index");
        let snapshot = state.table.snapshot();
        let requests: Vec<SeriesRequest> = state
            .forecasts
            .iter()
            .map(|(_, request)| request.clone())
            .collect();
        let features = state.spec.features;
        let planned = match forecasting::plan(&spec, index, &snapshot, features, &requests) {
            Ok(planned) => planned,
            Err(_) => {
                // A request planning refuses, such as one with a cutoff per
                // series missing, is refused alone; the rest plan again.
                let mut kept = Vec::with_capacity(state.forecasts.len());
                let mut refused = Vec::new();
                for (request, asked) in state.forecasts.drain(..) {
                    let alone = std::slice::from_ref(&asked);
                    match forecasting::plan(&spec, index, &snapshot, features, alone) {
                        Ok(_) => kept.push((request, asked)),
                        Err(error) => refused.push((request, NodeError::Forecast(error))),
                    }
                }
                state.forecasts = kept;
                state.generation += refused.len() as u64;
                self.refused.extend(refused);
                if state.forecasts.is_empty() {
                    return Ok(());
                }
                let requests: Vec<SeriesRequest> = state
                    .forecasts
                    .iter()
                    .map(|(_, asked)| asked.clone())
                    .collect();
                forecasting::plan(&spec, index, &snapshot, features, &requests)?
            }
        };
        let keys: Vec<CacheKey> = planned
            .units
            .iter()
            .map(|unit| CacheKey {
                model: state.spec.model,
                table: id,
                epoch: state.epoch,
                kind: Kind::Context,
                rows: unit.key,
            })
            .collect();
        // Contexts a step in flight encodes are read only once it completes.
        if keys.iter().any(|key| self.encoding.contains(key)) {
            return Ok(());
        }
        let features = state.spec.features;
        let unit_bytes = |unit: &forecasting::PlannedUnit| {
            state.spec.sizes.series_bytes(
                &spec,
                features,
                unit.members.len(),
                unit.steps,
                slot_bytes,
            )
        };
        // A unit that may extend its group's entry holds the source's patches
        // and the new ones, which can pad past a fresh encoding's; it is
        // charged the larger, since placement may fall back to either.
        let extended_bytes = |unit: &forecasting::PlannedUnit| {
            let prev = state.recent.get(&unit.group).filter(|prev| {
                spec.extendable
                    && self.cache.contains(&prev.key)
                    && !self.encoding.contains(&prev.key)
            })?;
            let (_, patches) = extension_size(&spec, unit, prev)?;
            let sizes = &state.spec.sizes;
            Some(extension_bytes(
                &spec, sizes, features, unit, patches, slot_bytes,
            ))
        };
        let cost: u64 = planned
            .units
            .iter()
            .zip(&keys)
            .filter(|(unit, key)| unit.whole || !self.cache.contains(key))
            .map(|(unit, _)| {
                let fresh = unit_bytes(unit);
                match extended_bytes(unit) {
                    Some(bytes) if !unit.whole => bytes.max(fresh),
                    _ => fresh,
                }
            })
            .sum();
        if !scheduled.is_empty() && *used + cost > budget {
            return Ok(());
        }
        *used += cost;
        let busy = Busy {
            moving: &self.moving,
            doomed: &self.doomed,
            encoding: &self.encoding,
        };
        let state = self.tables.get_mut(&id).expect("planned above");
        let first_unit = step.series_units.len();
        for (unit, key) in planned.units.iter().zip(keys) {
            let grid = forecasting::grid(&spec, unit, features, |rows| state.rows.rows(rows))?;
            let (mut placed, mut source, mut grown) = (Placed::Transient, None, 0);
            if !unit.whole {
                let bytes = state.spec.sizes.series_bytes(
                    &spec,
                    features,
                    unit.members.len(),
                    unit.steps,
                    slot_bytes,
                );
                // A kept unit whose entry is gone is nothing to extend.
                let recent = match state.recent.get(&unit.group).copied() {
                    Some(prev) if !self.cache.contains(&prev.key) => {
                        state.recent.remove(&unit.group);
                        None
                    }
                    Some(prev) if self.encoding.contains(&prev.key) => None,
                    recent => recent,
                };
                let extended = match recent {
                    Some(prev) if spec.extendable && !self.cache.contains(&key) => extend(
                        &mut self.cache,
                        busy,
                        &state.table.snapshot(),
                        &spec,
                        &state.spec.sizes,
                        features,
                        unit,
                        key,
                        prev,
                        slot_bytes,
                        &mut step.movements,
                    )?,
                    _ => None,
                };
                let mut patches = None;
                placed = match extended {
                    Some(extension) => {
                        source = Some(extension.source);
                        grown = extension.grown;
                        patches = Some(extension.patches);
                        extension.placed
                    }
                    None => place(
                        &mut self.cache,
                        busy,
                        key,
                        bytes,
                        unit.rows(),
                        &mut step.movements,
                    )?,
                };
                if placed.action() == Action::Encode {
                    patches = Some(state.spec.sizes.series_patches(&spec, unit.steps));
                }
                if let Some(patches) = patches {
                    let recent = Recent {
                        cutoff: unit.cutoff,
                        steps: unit.steps,
                        patches,
                        key,
                    };
                    state.writing.insert(key, (unit.group, recent));
                }
            }
            step.series_units.push(SeriesUnit {
                table: id,
                key,
                members: unit.members.clone(),
                cutoff: unit.cutoff,
                grid,
                placed,
                source,
                grown,
            });
        }
        for ((request, _), forecasts) in state.forecasts.iter().zip(planned.forecasts) {
            for forecast in forecasts {
                let unit = &planned.units[forecast.unit];
                let future = forecasting::future(&spec, unit, &forecast, features, |rows| {
                    state.rows.rows(rows)
                })?;
                step.forecasts.push(Forecast {
                    request: *request,
                    table: id,
                    epoch: state.epoch,
                    unit: StepUnit(first_unit + forecast.unit),
                    target: forecast.target,
                    horizon: forecast.horizon,
                    future,
                });
            }
        }
        scheduled.push(id);
        Ok(())
    }

    /// Forgets every cached entry and fit, as when the worker starts again
    /// empty; call it with no step in flight. Each table starts a new epoch,
    /// so a step that completes late cannot touch the entries that follow, and
    /// is fit again on its next step.
    pub fn forget_cache(&mut self) {
        for (&id, state) in &mut self.tables {
            let _ = self
                .cache
                .drop_epochs_before(state.spec.model, id, Epoch(u64::MAX));
            state.epoch.0 += 1;
            state.fitted = None;
            state.recent.clear();
            state.writing.clear();
        }
        self.doomed.clear();
        self.drops.clear();
        self.moving.clear();
        self.encoding.clear();
        self.fitting.clear();
    }

    /// Releases a finished step's entries; returns drops of entries from ended epochs.
    pub fn complete(&mut self, step: &Step) -> Vec<Movement> {
        self.release(step, &Failed::None)
    }

    /// Completes a step that the worker ran for every table except `failed`.
    /// The failed tables' parts are undone as `fail` undoes a whole step,
    /// except that their tier moves stand, since the worker made them before
    /// any table's part ran. Returns drops as `fail` does.
    pub fn complete_except(&mut self, step: &Step, failed: &HashSet<TableId>) -> Vec<Movement> {
        if failed.is_empty() {
            return self.complete(step);
        }
        self.unfit(step, |table| failed.contains(&table));
        self.release(step, &Failed::Tables(failed))
    }

    /// Undoes a step the worker failed to run. The entries it was to encode,
    /// which the worker may hold only in part, are forgotten, and so are the
    /// entries it moved between tiers, which the worker may not have moved, so
    /// a later step encodes them again; the rest are released as `complete`
    /// releases them, and fits it held are planned again. Returns drops for the
    /// forgotten entries and those from ended epochs.
    pub fn fail(&mut self, step: &Step) -> Vec<Movement> {
        self.unfit(step, |_| true);
        self.release(step, &Failed::All)
    }

    /// Plans again the fits a step held for the tables `failed` names.
    fn unfit(&mut self, step: &Step, failed: impl Fn(TableId) -> bool) {
        let fits = step.fits.iter().map(|fit| (fit.table, fit.epoch));
        let fused = step
            .units
            .iter()
            .filter(|unit| unit.placed == Placed::Fuse)
            .map(|unit| (unit.table, unit.key.epoch));
        for (table, epoch) in fits.chain(fused).filter(|(table, _)| failed(*table)) {
            if let Some(state) = self.tables.get_mut(&table)
                && state.fitted == Some(epoch)
            {
                state.fitted = None;
            }
        }
    }

    /// Releases a step's pins. An entry that is stale, because the step failed
    /// to encode it, a change purged it, or its epoch ended, is dropped once no
    /// step reads it any more.
    fn release(&mut self, step: &Step, failed: &Failed) -> Vec<Movement> {
        for movement in &step.movements {
            self.moving.remove(&movement.key);
        }
        for key in encoded(step) {
            self.encoding.remove(&key);
        }
        for table in fitted_tables(step) {
            self.fitting.remove(&table);
        }
        let kept = step
            .units
            .iter()
            .map(|unit| (unit.table, unit.key, unit.placed.action()))
            .chain(
                step.pages
                    .iter()
                    .map(|page| (page.table, page.key, page.placed.action())),
            )
            .chain(
                step.series_units
                    .iter()
                    .map(|unit| (unit.table, unit.key, unit.placed.action())),
            );
        let mut dropped = Vec::new();
        // A page's sources were pinned for the step too. One a kept page
        // copied from has served its purpose; one a transient page or a
        // failed part copied from stays for the page's next version.
        for page in &step.pages {
            for (source, _) in &page.copies {
                let _ = self.cache.unpin(source);
                let current = self.tables.get(&page.table).map(|state| state.epoch);
                let served = page.placed.action() == Action::Encode && !failed.includes(page.table);
                if served
                    || self.doomed.contains(source)
                    || current.is_none_or(|epoch| source.epoch < epoch)
                {
                    dropped.extend(self.drop_or_doom(source));
                }
            }
        }
        // An extension's source was pinned for the step too, and is dropped
        // like any entry the step read if it was doomed or its epoch ended; a
        // kept unit the step wrote is now one a later unit of its group may
        // extend.
        for unit in &step.series_units {
            if let Some(source) = unit.source {
                let _ = self.cache.unpin(&source);
                let current = self.tables.get(&unit.table).map(|state| state.epoch);
                if self.doomed.contains(&source) || current.is_none_or(|epoch| source.epoch < epoch)
                {
                    dropped.extend(self.drop_or_doom(&source));
                }
            }
            let Some(state) = self.tables.get_mut(&unit.table) else {
                continue;
            };
            let Some((group, recent)) = state.writing.remove(&unit.key) else {
                continue;
            };
            let written = !failed.includes(unit.table)
                && !self.doomed.contains(&unit.key)
                && unit.key.epoch >= state.epoch;
            // A group extends forward, so an earlier cutoff written after a
            // later one, in this step or an earlier, leaves the later as the
            // group's latest while its entry lasts.
            let latest = match state.recent.get(&group) {
                Some(prev) => {
                    prev.cutoff <= recent.cutoff
                        || !self.cache.contains(&prev.key)
                        || self.doomed.contains(&prev.key)
                        || prev.key.epoch < state.epoch
                }
                None => true,
            };
            if written && latest {
                state.recent.insert(group, recent);
                let held = state.recent.len();
                if held > RECENT_PRUNE && held >= 2 * state.recent_pruned {
                    let cache = &self.cache;
                    state.recent.retain(|_, recent| cache.contains(&recent.key));
                    state.recent_pruned = state.recent.len();
                }
            }
        }
        for (table, key, action) in kept {
            if matches!(action, Action::Transient | Action::Fuse) {
                continue;
            }
            let _ = self.cache.unpin(&key);
            let current = self.tables.get(&table).map(|state| state.epoch);
            // A removed table's entries are stale too, and so is an entry a
            // failed step was to write, whether encoded or extended.
            let written = matches!(action, Action::Encode | Action::Extend);
            let stale = (failed.includes(table) && written)
                || self.doomed.contains(&key)
                || current.is_none_or(|epoch| key.epoch < epoch);
            if stale {
                dropped.extend(self.drop_or_doom(&key));
            }
        }
        let now = Instant::now();
        for unit in &step.units {
            if self.rebuilding.contains_key(&unit.key) {
                if !failed.includes(unit.table) && self.cache.contains(&unit.key) {
                    dropped.extend(self.finish_rebuild(unit.key, now));
                } else {
                    for rebuilt in self.rebuilding.remove(&unit.key).unwrap_or_default() {
                        self.rebuilt_regions
                            .remove(&(rebuilt.table, rebuilt.region));
                    }
                }
            }
        }
        if matches!(failed, Failed::All) {
            // The worker may not have moved these, so they are encoded again.
            for movement in step
                .movements
                .iter()
                .filter(|movement| movement.to.is_some())
            {
                dropped.extend(self.drop_or_doom(&movement.key));
            }
        }
        dropped
    }
}

/// The unit of `rows` in `plan`, added once.
fn add_unit(
    plan: &mut Plan,
    ids: &mut HashMap<Fingerprint, UnitId>,
    rows: &Context,
    key: Fingerprint,
) -> UnitId {
    *ids.entry(key).or_insert_with(|| {
        plan.units.push(crate::planner::Unit {
            rows: rows.clone(),
            key,
        });
        UnitId(plan.units.len() - 1)
    })
}

/// `rows` in the order of their keys.
fn by_key(table: &Table, rows: &[RowId]) -> Vec<RowId> {
    let mut keyed: Vec<(u128, RowId)> = rows
        .iter()
        .map(|&row| (Keys::key(table, row), row))
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, row)| row).collect()
}

/// A region's context a rebuild step encodes, to replace its built one.
#[derive(Debug)]
struct Rebuilding {
    table: TableId,
    region: usize,
    /// The state and draw of the regions it was planned in.
    epoch: Epoch,
    generation: u64,
    rows: Context,
    /// Changes the region had counted when it was planned.
    seen: u64,
    /// Since when the region was due.
    due: Instant,
}

/// The tables a step fits a state for: by a fit, or by a fused unit.
fn fitted_tables(step: &Step) -> impl Iterator<Item = TableId> + '_ {
    let fused = step
        .units
        .iter()
        .filter(|unit| unit.placed == Placed::Fuse)
        .map(|unit| unit.table);
    step.fits.iter().map(|fit| fit.table).chain(fused)
}

/// The entries a step encodes and keeps: its contexts and its pages.
fn encoded(step: &Step) -> impl Iterator<Item = CacheKey> + '_ {
    let units = step.units.iter().map(|unit| (unit.key, &unit.placed));
    let pages = step.pages.iter().map(|page| (page.key, &page.placed));
    let series = step
        .series_units
        .iter()
        .map(|unit| (unit.key, &unit.placed));
    units
        .chain(pages)
        .chain(series)
        .filter(|(_, placed)| matches!(placed, Placed::Encode(_) | Placed::Extend { .. }))
        .map(|(key, _)| key)
}

/// Which tables' parts of a step the worker failed to run.
enum Failed<'a> {
    None,
    Tables(&'a HashSet<TableId>),
    All,
}

impl Failed<'_> {
    fn includes(&self, table: TableId) -> bool {
        match self {
            Failed::None => false,
            Failed::Tables(tables) => tables.contains(&table),
            Failed::All => true,
        }
    }
}

/// Entries a new step must not read: those a step in flight moves between
/// tiers, whose slots may not hold them yet, and stale ones a step in flight
/// still reads. A unit or page on such an entry is computed again, transiently.
#[derive(Clone, Copy)]
struct Busy<'a> {
    moving: &'a HashSet<CacheKey>,
    doomed: &'a HashSet<CacheKey>,
    /// Entries a step in flight computes, which the worker holds only once
    /// it completes; a later step reads them, but copies from none.
    encoding: &'a HashSet<CacheKey>,
}

impl Busy<'_> {
    fn contains(&self, key: &CacheKey) -> bool {
        self.moving.contains(key) || self.doomed.contains(key)
    }

    /// Whether a step may copy rows from the entry.
    fn source(&self, key: &CacheKey) -> bool {
        !self.contains(key) && !self.encoding.contains(key)
    }
}

/// Adds one table's pending requests to a step: its fit if due, its units and
/// the pages they need, and its queries. The requests stay pending until the
/// whole step is planned.
#[allow(clippy::too_many_arguments)]
fn schedule_table(
    cache: &mut Cache,
    busy: Busy,
    id: TableId,
    state: &mut TableState,
    snapshot: &Snapshot,
    planned: Plan,
    keys: Vec<CacheKey>,
    step: &mut Step,
    parts: &mut PlanParts,
) -> Result<(), NodeError> {
    let since = parts.start();
    let mut placed: BTreeMap<PageId, StepPage> = BTreeMap::new();
    let wanted = state.horizon_of(state.pending[0].2.as_deref());
    // A table that summarizes each context fits each unit on its own rows
    // instead: the worker gets no fit and no pages.
    let own = state.spec.summary == SummaryScope::Context;
    let fuse = !own && wanted.is_none() && fuses(state, snapshot, &planned);
    let fits = step.fits.len();
    if state.fitted != Some(state.epoch) {
        state.horizon = wanted;
    }
    if own {
        state.fitted = Some(state.epoch);
    } else if fuse {
        // The unit fits the state on its rows, every row by key, in the call
        // that encodes and decodes it, so the step has no fit and no pages.
        let rows: Vec<RowId> = snapshot.rows().iter().collect();
        let rows = by_key(&state.table, &rows);
        let (features, _) = state.inputs(&rows)?;
        fitted_on(state, &rows, &features);
        state.fitted_latest = rows.iter().filter_map(|&row| state.times.time(row)).max();
    } else if let Some(horizon) = wanted.filter(|_| state.fitted != Some(state.epoch)) {
        // Fit only on rows before the cutoffs, so no query's state saw a later row.
        let before = &state.times.before(horizon) & &snapshot.rows();
        fit_sample(id, state, &before, step)?;
    } else if state.fitted != Some(state.epoch)
        && snapshot.rows().len()
            > state
                .spec
                .effective_fit_rows()
                .min(state.spec.sizes.fused_rows) as u64
    {
        // Past the rows a fit embeds with its state, it computes the state
        // alone, on a sample past the rows its call takes, and each page is
        // embedded once a plan reads it, so a fit never holds every row's
        // representation.
        fit_sample(id, state, &snapshot.rows(), step)?;
    } else if state.fitted != Some(state.epoch) {
        let pages: Vec<PageId> = state
            .index
            .pages()
            .filter(|&page| {
                state
                    .index
                    .rows(page)
                    .is_some_and(|rows| rows.iter().any(|row| snapshot.version(row).is_some()))
            })
            .collect();
        let first = step.pages.len();
        place_pages(cache, busy, id, state, snapshot, &pages, true, step)?;
        let indices: Vec<StepPage> = (first..step.pages.len()).map(StepPage).collect();
        for (&page, &index) in pages.iter().zip(&indices) {
            placed.insert(page, index);
        }
        let rows: Vec<RowId> = indices
            .iter()
            .flat_map(|&index| step.page(index).rows.clone())
            .collect();
        let (features, targets) = state.inputs(&rows)?;
        fitted_on(state, &rows, &features);
        step.fits.push(Fit {
            table: id,
            epoch: state.epoch,
            features,
            targets,
            rows,
            pages: indices,
        });
    }
    if let Some(fit) = step.fits.get(fits) {
        state.fitted_latest = fit
            .rows
            .iter()
            .filter_map(|&row| state.times.time(row))
            .max();
        parts.add("fit", since);
    }

    let width = state.spec.features;
    let first_unit = step.units.len();
    place_units(
        cache,
        busy,
        id,
        state,
        snapshot,
        planned.units,
        keys,
        &placed,
        fuse,
        step,
        parts,
    )?;

    let since = parts.start();
    let mut queries = planned.queries.into_iter();
    for (request, features, _) in &state.pending {
        let count = features.len() / width;
        let units = queries
            .by_ref()
            .take(count)
            .map(|unit| StepUnit(first_unit + unit.0))
            .collect();
        step.queries.push(QueryBatch {
            request: *request,
            table: id,
            epoch: state.epoch,
            features: state.spec.model_values(features),
            units,
        });
    }
    parts.add("assemble", since);
    Ok(())
}

/// Adds units of a table to a step, placing each, and the pages holding the
/// rows of those to encode that `placed` does not hold already.
#[allow(clippy::too_many_arguments)]
fn place_units(
    cache: &mut Cache,
    busy: Busy,
    id: TableId,
    state: &TableState,
    snapshot: &Snapshot,
    units: Vec<crate::planner::Unit>,
    keys: Vec<CacheKey>,
    placed: &BTreeMap<PageId, StepPage>,
    fuse: bool,
    step: &mut Step,
    parts: &mut PlanParts,
) -> Result<(), NodeError> {
    let since = parts.start();
    // The rows of units to encode, whose pages the step needs.
    let mut encoding = Context::new();
    for (unit, key) in units.into_iter().zip(keys) {
        let mut rows: Vec<RowId> = unit.rows.iter().collect();
        let bytes = state.spec.sizes.bytes(rows.len(), cache.slot_bytes());
        let held = unit.rows.clone();
        let placement = if fuse {
            Placed::Fuse
        } else {
            place(cache, busy, key, bytes, unit.rows, &mut step.movements)?
        };
        // A unit fit on its own rows carries the values the model reads of
        // them, and reads no pages.
        let mut fit = None;
        if placement.action() != Action::Reuse {
            // The model reads a context's rows by key, so their order does
            // not depend on the order they were written in.
            rows = by_key(&state.table, &rows);
            if fuse || state.spec.summary == SummaryScope::Context {
                fit = Some(state.inputs(&rows)?);
            } else {
                encoding |= &held;
            }
        }
        step.units.push(Unit {
            table: id,
            key,
            rows,
            placed: placement,
            fit,
        });
    }
    parts.add("units", since);
    let since = parts.start();
    let missing: Vec<PageId> = state
        .index
        .pages_holding(&encoding)
        .into_iter()
        .filter(|page| !placed.contains_key(page))
        .collect();
    let pages = place_pages(cache, busy, id, state, snapshot, &missing, false, step);
    parts.add("pages", since);
    pages
}

/// Whether a table's step fuses its one unit: the table holds a single-use
/// context not fit this epoch, its plan reads one context of every row, and
/// the context and the queries waiting stay within the rows a fit embeds
/// with its state.
fn fuses(state: &TableState, snapshot: &Snapshot, planned: &Plan) -> bool {
    let most = state
        .spec
        .effective_fit_rows()
        .min(state.spec.sizes.fused_rows) as u64;
    let width = state.spec.features.max(1);
    let queries: usize = state
        .pending
        .iter()
        .map(|(_, features, _)| features.len() / width)
        .sum();
    state.single_use
        && state.fitted != Some(state.epoch)
        && queries as u64 <= most
        && matches!(planned.units.as_slice(), [unit]
            if unit.rows.len() <= most && unit.rows == snapshot.rows())
}

/// Records a fit of a table's state on `rows`, every live row, whose model
/// values are `features`: their column summary and moments.
fn fitted_on(state: &mut TableState, rows: &[RowId], features: &[f32]) {
    let width = state.spec.model_features();
    let labeled = state.labeled_values(rows, features);
    let (means, deviations) = moments(&labeled, width);
    state.columns = Columns::of(&labeled, width);
    let hidden = state.hidden();
    state.fitted_columns = Some(Fitted {
        rows: rows.iter().filter(|row| !hidden.contains(row.0)).count() as u64,
        means,
        deviations,
    });
    state.changed = 0;
    state.fitted = Some(state.epoch);
}

/// Fits a table's state alone on a seeded sample of at most the effective
/// `fit_rows` of `from`, every row of it if it has no more: its live rows
/// when they outgrow the rows a fit embeds with its state, or those before a
/// request's cutoffs.
///
/// The sample's representations fill no pages: each page is embedded when a
/// plan first reads it. Column sums keep running over every live row, for
/// drift, while the fitted moments come from the sample.
fn fit_sample(
    id: TableId,
    state: &mut TableState,
    from: &Context,
    step: &mut Step,
) -> Result<(), NodeError> {
    let sample: Vec<RowId> = smallest(
        FIT_SEED,
        from,
        state.spec.effective_fit_rows(),
        &state.table,
    )
    .iter()
    .collect();
    // The fit reads the sample by key, like a context, so its moments and
    // the model's fit do not depend on the order rows were written in.
    let rows = by_key(&state.table, &sample);
    let (features, targets) = state.inputs(&rows)?;
    let labeled = state.labeled_values(&rows, &features);
    let (means, deviations) = moments(&labeled, state.spec.model_features());
    state.fitted_columns = Some(Fitted {
        rows: (from.bitmap() - state.hidden()).len(),
        means,
        deviations,
    });
    state.changed = 0;
    step.fits.push(Fit {
        table: id,
        epoch: state.epoch,
        features,
        targets,
        rows,
        pages: Vec::new(),
    });
    state.fitted = Some(state.epoch);
    Ok(())
}

/// Places pages of a table in the host tier and adds them to the step, pinning kept ones.
///
/// Cached pages are pinned before any new page is admitted, so admitting one
/// cannot evict another this step reads. A fit computes its pages, so they
/// carry no rows to embed.
#[allow(clippy::too_many_arguments)]
fn place_pages(
    cache: &mut Cache,
    busy: Busy,
    table: TableId,
    state: &TableState,
    snapshot: &Snapshot,
    pages: &[PageId],
    fit: bool,
    step: &mut Step,
) -> Result<(), NodeError> {
    let mut entries = Vec::with_capacity(pages.len());
    for &page in pages {
        let context = state.index.rows(page).unwrap_or_default();
        let rows: Vec<RowId> = context
            .iter()
            .filter(|&row| snapshot.version(row).is_some())
            .collect();
        let key = CacheKey {
            model: state.spec.model,
            table,
            epoch: state.epoch,
            kind: Kind::Rows,
            rows: snapshot.fingerprint(&context),
        };
        let cached = cache.lookup(&key);
        if cached.is_some() && !busy.contains(&key) {
            cache.pin(&key)?;
        }
        // Sources are pinned with the cached pages, before any admission.
        let copies = if cached.is_none() && !fit {
            page_sources(cache, busy, table, state.epoch, &rows)?
        } else {
            Vec::new()
        };
        entries.push((key, rows, cached, copies));
    }
    for (key, rows, cached, copies) in entries {
        let placed = if let Some(placement) = cached {
            if busy.contains(&key) {
                Placed::Transient
            } else {
                Placed::Reuse(placement.slots)
            }
        } else {
            let bytes = state.spec.sizes.bytes_per_representation * rows.len() as u64;
            let cost = EMBED_COST_PER_CELL * (rows.len() * state.spec.model_features()) as f64;
            let held = rows.iter().copied().collect();
            match cache.admit_host(key, bytes, cost, held) {
                Ok(admission) => {
                    step.movements.extend(admission.movements);
                    cache.pin(&key)?;
                    Placed::Encode(admission.slots)
                }
                Err(CacheError::NoRoom | CacheError::TooLarge { .. }) => Placed::Transient,
                Err(error) => return Err(error.into()),
            }
        };
        let embed = (placed.action() != Action::Reuse && !fit)
            .then(|| {
                let copied: HashSet<usize> = copies
                    .iter()
                    .flat_map(|(_, positions)| positions.iter().map(|&(_, to)| to))
                    .collect();
                let embedded: Vec<RowId> = rows
                    .iter()
                    .enumerate()
                    .filter(|(at, _)| !copied.contains(at))
                    .map(|(_, &row)| row)
                    .collect();
                state.inputs(&embedded)
            })
            .transpose()?;
        step.pages.push(Page {
            table,
            key,
            rows,
            placed,
            embed,
            copies,
        });
    }
    Ok(())
}

/// The cached entries of this epoch holding rows of a page to compute that
/// were not written since, pinned, with the positions to copy from each.
///
/// Each row is copied from the first source holding it, in key order.
fn page_sources(
    cache: &mut Cache,
    busy: Busy,
    table: TableId,
    epoch: Epoch,
    rows: &[RowId],
) -> Result<Copies, NodeError> {
    let wanted: Context = rows.iter().copied().collect();
    let mut copies = Vec::new();
    let mut covered = HashSet::new();
    for source in cache.holding(table, &wanted) {
        if source.kind != Kind::Rows || source.epoch != epoch || !busy.source(&source) {
            continue;
        }
        let (Some(held), Some(stale)) = (cache.rows(&source), cache.stale(&source)) else {
            continue;
        };
        let from: HashMap<RowId, usize> =
            held.iter().enumerate().map(|(at, row)| (row, at)).collect();
        let positions: Vec<(usize, usize)> = rows
            .iter()
            .enumerate()
            .filter(|&(to, row)| !covered.contains(&to) && !stale.contains(*row))
            .filter_map(|(to, row)| from.get(row).map(|&at| (at, to)))
            .collect();
        if positions.is_empty() {
            continue;
        }
        covered.extend(positions.iter().map(|&(_, to)| to));
        cache.pin(&source)?;
        copies.push((source, positions));
    }
    Ok(copies)
}

/// Some of a batch's rows, picked by index.
struct Picked {
    keys: Vec<u128>,
    features: Vec<f32>,
    targets: Targets,
    times: Option<Vec<Option<i64>>>,
}

impl Picked {
    fn new(
        keys: &[u128],
        features: &[f32],
        width: usize,
        targets: &Targets,
        times: Option<&[Option<i64>]>,
        indices: &[usize],
    ) -> Self {
        Self {
            keys: indices.iter().map(|&index| keys[index]).collect(),
            features: indices
                .iter()
                .flat_map(|&index| &features[index * width..(index + 1) * width])
                .copied()
                .collect(),
            targets: targets.pick(indices),
            times: times.map(|times| indices.iter().map(|&index| times[index]).collect()),
        }
    }

    /// Inserts the rows, which the whole batch's checks passed.
    fn insert(&self, node: &mut Node, table: TableId) -> Result<Vec<RowId>, NodeError> {
        let times = self.times.as_deref();
        node.apply_insert(table, &self.keys, &self.features, &self.targets, times)
    }

    /// Updates the rows, which the whole batch's checks passed.
    fn update(&self, node: &mut Node, table: TableId) -> Result<Vec<RowId>, NodeError> {
        let times = self.times.as_deref();
        node.apply_update(table, &self.keys, &self.features, &self.targets, times)
    }
}

/// Checks that `times`, if given, has one entry per key.
fn check_times(keys: &[u128], times: Option<&[Option<i64>]>) -> Result<(), NodeError> {
    match times {
        Some(times) if times.len() != keys.len() => Err(NodeError::Count {
            what: Counted::Times,
            expected: keys.len(),
            got: times.len(),
        }),
        _ => Ok(()),
    }
}

/// Checks that row-major `features` holds one row of `width` per key.
fn check_rows(keys: usize, features: &[f32], width: usize) -> Result<(), NodeError> {
    if features.len() == keys * width {
        return Ok(());
    }
    if keys > 0 && features.len().is_multiple_of(keys) {
        return Err(NodeError::Width {
            expected: width,
            got: features.len() / keys,
        });
    }
    Err(NodeError::Count {
        what: Counted::Rows,
        expected: keys,
        got: features.len() / width.max(1),
    })
}

/// A series unit placed as an extension: where, from which entry, by how many
/// steps, and the patches its entry will hold.
struct Extension {
    placed: Placed,
    source: CacheKey,
    grown: usize,
    patches: usize,
}

/// The steps a unit's grid grows `prev`'s by and the patches its extension
/// holds, the source's and the new ones: None unless the cutoff advanced by
/// whole patches, the grid grew by as many steps and the grown context still
/// fits the model's.
fn extension_size(
    spec: &SeriesSpec,
    unit: &forecasting::PlannedUnit,
    prev: &Recent,
) -> Option<(usize, usize)> {
    let step = spec.step.get() as i64;
    let advanced = unit.cutoff.checked_sub(prev.cutoff).unwrap_or(0);
    if advanced <= 0 || advanced % step != 0 {
        return None;
    }
    let grown = usize::try_from(advanced / step).unwrap_or(usize::MAX);
    let patch = spec.patch.get();
    if grown % patch != 0 || unit.steps != prev.steps + grown {
        return None;
    }
    let patches = prev.patches + grown / patch;
    (patches * patch <= spec.context.get()).then_some((grown, patches))
}

/// Bytes an extension of `patches` patches takes: it is stored padded to the
/// bucket of its patches, as an encoding of as many is, so the programs
/// reading it serve every count in the bucket.
fn extension_bytes(
    spec: &SeriesSpec,
    sizes: &EntrySizes,
    features: usize,
    unit: &forecasting::PlannedUnit,
    patches: usize,
    slot_bytes: u64,
) -> u64 {
    let bucket = sizes.series_patches(spec, patches * spec.patch.get());
    sizes.series_bytes_of(spec, features, unit.members.len(), bucket, slot_bytes)
}

/// Places a series unit as an extension of its group's latest kept entry,
/// when the cutoff advanced by whole patches, the grid grew by as many steps
/// and the rows before the earlier cutoff are the entry's: the new entry is
/// admitted, sized for the patches the source holds plus the new ones, and
/// both are pinned for the step, the source before room is made so that
/// making it cannot move the source off the device. None when any of that
/// fails, the grown context would pass the model's, or the device has no
/// room, so the unit is placed as any other.
#[allow(clippy::too_many_arguments)]
fn extend(
    cache: &mut Cache,
    busy: Busy,
    snapshot: &Snapshot,
    spec: &SeriesSpec,
    sizes: &EntrySizes,
    features: usize,
    unit: &forecasting::PlannedUnit,
    key: CacheKey,
    prev: Recent,
    slot_bytes: u64,
    movements: &mut Vec<Movement>,
) -> Result<Option<Extension>, NodeError> {
    let Some((grown, patches)) = extension_size(spec, unit, &prev) else {
        return Ok(None);
    };
    let bytes = extension_bytes(spec, sizes, features, unit, patches, slot_bytes);
    if busy.contains(&prev.key) || busy.contains(&key) {
        return Ok(None);
    }
    let Some(placement) = cache.lookup(&prev.key) else {
        return Ok(None);
    };
    if placement.tier != Tier::Device {
        return Ok(None);
    }
    let prefix = forecasting::prefix_rows(unit, prev.steps);
    let print = forecasting::unit_key(
        snapshot.fingerprint(&prefix),
        prev.cutoff,
        prev.steps,
        &unit.members,
        unit.target,
    );
    if print != prev.key.rows {
        return Ok(None);
    }
    let cost = (unit.rows().len() as f64).powi(2);
    cache.pin(&prev.key)?;
    match cache.admit(key, bytes, cost, unit.rows()) {
        Ok(admission) => {
            movements.extend(admission.movements);
            cache.pin(&key)?;
            Ok(Some(Extension {
                placed: Placed::Extend {
                    source: placement.slots,
                    slots: admission.slots,
                },
                source: prev.key,
                grown,
                patches,
            }))
        }
        Err(error) => {
            let _ = cache.unpin(&prev.key);
            match error {
                CacheError::NoRoom | CacheError::TooLarge { .. } => Ok(None),
                error => Err(error.into()),
            }
        }
    }
}

/// Finds or makes room for a unit's entry on the device, pinning kept entries.
fn place(
    cache: &mut Cache,
    busy: Busy,
    key: CacheKey,
    bytes: u64,
    rows: Context,
    movements: &mut Vec<Movement>,
) -> Result<Placed, NodeError> {
    if busy.contains(&key) {
        return Ok(Placed::Transient);
    }
    let placed = if let Some(placement) = cache.lookup(&key) {
        if placement.tier == Tier::Device {
            Ok(Placed::Reuse(placement.slots))
        } else {
            cache.promote(&key).map(|admission| {
                movements.extend(admission.movements);
                Placed::Reuse(admission.slots)
            })
        }
    } else {
        let cost = (rows.len() as f64).powi(2);
        cache.admit(key, bytes, cost, rows).map(|admission| {
            movements.extend(admission.movements);
            Placed::Encode(admission.slots)
        })
    };
    match placed {
        Ok(placed) => {
            cache.pin(&key)?;
            Ok(placed)
        }
        Err(CacheError::NoRoom | CacheError::TooLarge { .. }) => Ok(Placed::Transient),
        Err(error) => Err(error.into()),
    }
}
