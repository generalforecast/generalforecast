// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! How a table is served: its model, task, sizes, strategy and refit policy.

use super::*;

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
    pub(super) fn bytes(&self, rows: usize, slot_bytes: u64) -> u64 {
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
    pub(super) fn series_bytes(
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
    pub(super) fn series_patches(&self, series: &SeriesSpec, steps: usize) -> usize {
        let patch = series.patch.get();
        let limit = (series.context.get() / patch).max(1);
        bucket_size(steps / patch, self.bucket_minimum).min(limit)
    }

    /// Bytes a series unit's cache of exactly `patches` patches takes, as
    /// `series_bytes` sizes a padded grid's, for an extension that holds the
    /// source's patches and the new ones.
    pub(super) fn series_bytes_of(
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
    pub(super) fn bits(&self) -> Vec<u32> {
        match self {
            Self::Classes(labels) => labels.iter().map(|&label| label as u32).collect(),
            Self::Values(values) => values.iter().map(|value| value.to_bits()).collect(),
        }
    }

    /// Whether each target is known: a class from 0, or a finite value.
    pub(super) fn labeled(&self) -> Vec<bool> {
        match self {
            Self::Classes(labels) => labels.iter().map(|&label| label >= 0).collect(),
            Self::Values(values) => values.iter().map(|value| value.is_finite()).collect(),
        }
    }

    /// Targets for the task from the bits rows store.
    pub(super) fn from_bits(task: Task, bits: &[u32]) -> Self {
        match task {
            Task::Classes(_) => Self::Classes(bits.iter().map(|&bits| bits as i32).collect()),
            Task::Values => Self::Values(bits.iter().map(|&bits| f32::from_bits(bits)).collect()),
        }
    }

    /// No targets, of the same kind as `targets`.
    pub(super) fn empty_like_targets(targets: &Self) -> Self {
        match targets {
            Self::Classes(_) => Self::Classes(Vec::new()),
            Self::Values(_) => Self::Values(Vec::new()),
        }
    }

    /// The targets at `indices` of a batch, in that order.
    pub(super) fn pick(&self, indices: &[usize]) -> Self {
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
    /// Each context's own rows, as upstream predictors fit each context: a unit
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
    pub(super) changed: Option<f64>,
    pub(super) shift: Option<f64>,
    pub(super) cutoffs: CutoffRefit,
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

impl TableSpec {
    /// The most rows a context of this table holds: the model's trained
    /// length under `PastCap::Clamp`, or None for any number.
    pub(super) fn clamp(&self) -> Option<u64> {
        match (self.past_cap, self.context_rows) {
            (PastCap::Clamp, Some(rows)) => Some(rows.get()),
            _ => None,
        }
    }

    /// `strategy` cut to the contexts `clamp` allows.
    pub(super) fn clamped(&self, strategy: Strategy) -> Strategy {
        self.clamp().map_or(strategy, |rows| strategy.capped(rows))
    }

    /// The keys of this table's entries in its state of `epoch`.
    pub(super) fn entry_keys(&self, table: TableId, epoch: Epoch) -> EntryKeys {
        EntryKeys {
            model: self.model,
            table,
            epoch,
        }
    }

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
    pub(super) fn model_values(&self, rows: &[f32]) -> Vec<f32> {
        if self.retrieval == 0 {
            return rows.to_vec();
        }
        rows.chunks(self.features)
            .flat_map(|row| &row[..self.model_features()])
            .copied()
            .collect()
    }

    /// The bucket a row at `time` belongs to; rows without a time share the first.
    pub(super) fn bucket(&self, time: Option<i64>) -> Bucket {
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

/// The keys of one table's cache entries in one state, the one place they
/// are made, so every check of what the cache holds names entries alike.
#[derive(Clone, Copy, Debug)]
pub(super) struct EntryKeys {
    model: ModelId,
    table: TableId,
    epoch: Epoch,
}

impl EntryKeys {
    /// The key of an entry of `kind` computed from the rows `rows` fingerprints.
    pub(super) fn of(self, kind: Kind, rows: Fingerprint) -> CacheKey {
        CacheKey {
            model: self.model,
            table: self.table,
            epoch: self.epoch,
            kind,
            rows,
        }
    }

    /// The key of a context of those rows.
    pub(super) fn context(self, rows: Fingerprint) -> CacheKey {
        self.of(Kind::Context, rows)
    }
}
