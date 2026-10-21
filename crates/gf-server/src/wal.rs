//! The write-ahead log: every pushed change, encrypted and synced before it applies.
//!
//! Each table has its own log, `<table>.log`: a plain header naming the format
//! and the key, then the table's definition, then its records. A record is a
//! frame, sealed with XChaCha20-Poly1305 under a random nonce, with its
//! sequence number, the table's name and creation id, and the previous
//! record's tag as associated data, so changed, swapped, spliced or foreign
//! records fail to open, including records from another table's log or from
//! an earlier table of the same name. A crash can only leave a partial final
//! record, which replay drops, and a failed append is cut back off; a log that
//! cannot cut it off takes no more writes until the server restarts.
//!
//! A log sealed under a previous key still opens, and the server rewrites it
//! under the current key at startup, entry for entry, keeping every sequence
//! number, so it stays the same log as its copies elsewhere.
//!
//! Beside each log, `<table>.head` holds a length the log was synced through,
//! sealed with the table's name. A log shorter than its head, or without one,
//! lost whole records from its end and does not open. A record is
//! acknowledged once it is synced, so a crash can only tear an append that
//! was never acknowledged. Writing the head costs three more syncs, so while
//! a table is busy it trails the synced length by up to `HEAD_LAG` records or
//! `HEAD_INTERVAL`, and it catches up once the table's writes go idle and
//! when the log closes. A record before the head's length that does not open
//! is refused as damage; one at or past it is taken for a torn append, and
//! opening cuts it off with everything after it. So a crash loses no
//! acknowledged record, and only damage to one synced within the head's lag
//! before a crash could be taken for a torn append.
//! Restoring an older copy of both files together still opens; replicas are
//! what catch that.
//!
//! Opening a log replays it one record at a time, so a long log never sits in
//! memory at once.
//!
//! Records that add rows carry the words they added to the vocabularies of
//! the table's categorical columns, so replay codes categories as the primary did.
//!
//! Replicas copy a table by reading its log from a sequence number: entry 0 is
//! the definition, and each later entry is one record. Records that change
//! rows may carry the client's write id, so a write sent twice applies once.
//!
//! Compacting rewrites a log as its definition and a snapshot of the live rows,
//! which keeps the snapshot's sequence number: later records continue the same
//! numbering, so replicas can follow a compaction like any other record.

use std::collections::HashSet;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use std::num::{NonZeroU64, NonZeroUsize};

use gf_core::categories::CategoryEncoding;
use gf_core::forecasting::GroupStrategy;
use gf_core::node::{
    CutoffRefit, DEFAULT_FIT_ROWS, FittedColumns, PastCap, RefitPolicy, RowLayout, Scaling,
    SummaryScope, TableLayout, Targets, Task,
};
use gf_core::pages::{Bucket, CellLayout, IndexLayout, PageId, PageLayout};
use gf_core::planner::Strategy;
use gf_core::regions::{Rebuild, RebuildWait, RegionSpec};
use gf_core::rows::Sealer;
use gf_core::table::RowId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use zeroize::Zeroize;

use crate::name::TableName;
use crate::protocol::{
    Frame, FrameError, StrategySpec, SummarySpec, f32_bytes, f32_values, f64_bytes, f64_values,
    i64_bytes, i64_values, key_bytes, key_values, target_bytes, targets_from, u32_bytes,
    u32_values, u64_bytes, u64_values,
};
use gf_core::columns::{Column, Words};

const NONCE: usize = 24;
/// Numbers each staged compaction file, unique in the process, so a dropped
/// table's late compaction never shares a file with its successor's.
static COMPACTIONS: AtomicU64 = AtomicU64::new(0);
/// A record's length, then its complement: a torn write leaves the pair short,
/// while a damaged length breaks the pair instead of hiding later records.
const PREFIX: usize = 8;
/// The authentication tag that ends each sealed record.
const TAG: usize = 16;
/// The file's plain header: a magic, the format, and the id of the key.
const MAGIC: &[u8; 6] = b"GF_LOG";
const FORMAT: u8 = 4;
const HEADER: usize = MAGIC.len() + 2 + KEY_ID;
/// Bytes of a key's id.
const KEY_ID: usize = 8;

/// How a table is served, as recorded at the head of its log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableDefinition {
    pub name: TableName,
    pub model: String,
    /// The feature columns, in order.
    pub columns: Vec<Column>,
    /// Number of classes, or 0 for regression.
    pub classes: u32,
    #[serde(default)]
    pub strategy: StrategySpec,
    #[serde(default = "page_size")]
    pub page_size: usize,
    /// A window of time pages never straddle, if any.
    #[serde(default)]
    pub time_bucket: Option<u64>,
    /// Refit once this fraction of the fitted rows has changed, if set.
    #[serde(default)]
    pub refit_after: Option<f64>,
    /// Refit once a column's mean moves this many deviations, if set.
    #[serde(default)]
    pub refit_shift: Option<f64>,
    /// The most rows the table's state is fit on; `DEFAULT_FIT_ROWS` if unset.
    #[serde(default)]
    pub fit_rows: Option<u64>,
    /// What a context larger than the model was trained on does.
    #[serde(default)]
    pub past_cap: PastCapSpec,
    /// When a query's cutoff calls for a new state: `each`, `changed` or `never`.
    #[serde(default = "each")]
    pub refit_cutoffs: String,
    /// Rows the retrieval scaling stays fixed after, once fit on at least
    /// that many, if set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_rows: Option<u64>,
    /// How categorical columns place rows for retrieval: `codes`, `one_hot`
    /// or `target`.
    #[serde(default = "codes")]
    pub retrieval_categoricals: String,
    /// Regions whose queries share a context, if the table has them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regions: Option<RegionsSpec>,
    /// What the column summary is fit on: the table, or each context.
    #[serde(default)]
    pub summary: SummarySpec,
    /// How the table's series are forecast, for a table of series.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub series: Option<Box<SeriesDefinition>>,
    /// An id the server gives each creation, so copies of an earlier table of
    /// the same name never mix with this one's; None until it is created.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "CreationId::optional"
    )]
    pub created: Option<CreationId>,
    /// What row times are, as clients convert them; the node reads only numbers.
    #[serde(default)]
    pub times: TimeKind,
    /// The column whose value a router sends rows and queries to one
    /// partition by, for a table it splits; a node serves such a partition
    /// as any table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_by: Option<RouteBy>,
    /// Whether the table's rows outlive the server: `durable` tables keep a
    /// log and replicate, `ephemeral` ones hold a request's context in memory
    /// alone, under the name `ctx-<fingerprint>`, until the shard evicts them.
    #[serde(default, skip_serializing_if = "Durability::is_durable")]
    pub durability: Durability,
}

/// Whether a table's rows are logged, or held in memory for requests alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Durability {
    /// Every change is logged before it applies, and replicas copy the log.
    #[default]
    Durable,
    /// Rows live in memory only: never logged, never replicated, and gone
    /// once the shard evicts the table or the server stops.
    Ephemeral,
}

impl Durability {
    /// Whether this is the default, which definitions leave unwritten.
    #[must_use]
    pub fn is_durable(&self) -> bool {
        *self == Self::Durable
    }
}

/// The id a server gives a table when it creates it: 16 hex digits, the
/// milliseconds since 1970 by its clock and then random, so a later creation
/// of a name sorts after an earlier one.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CreationId(String);

impl CreationId {
    /// The id of a creation at `millis` since 1970, with two random bytes.
    #[must_use]
    pub fn new(millis: u128, random: [u8; 2]) -> Self {
        Self(format!(
            "{:012x}{:02x}{:02x}",
            millis & 0xffff_ffff_ffff,
            random[0],
            random[1]
        ))
    }

    /// An id as another node or a router sent it; an empty one names no creation.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        (!text.is_empty()).then(|| Self(text.to_string()))
    }

    /// The id's text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Reads an optional id, taking an empty string as none, as older
    /// definitions and heads hold.
    pub(crate) fn optional<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Self>, D::Error> {
        let text = Option::<String>::deserialize(deserializer)?;
        Ok(text.as_deref().and_then(Self::parse))
    }
}

impl fmt::Display for CreationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// What a table's row times and cutoffs are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeKind {
    /// Integers, in a unit of the client's.
    #[default]
    Integer,
    /// Dates and times, as microseconds since 1970-01-01.
    Datetime,
}

impl TableDefinition {
    /// The series settings, checked, for a table of series.
    ///
    /// # Errors
    ///
    /// If the step, a group's size or the horizon is 0, a known column is
    /// not a covariate column or is named twice, the series, time and target
    /// columns share a name with each other or a covariate, or the table has
    /// retrieval columns, regions, a strategy or a time bucket, which a table
    /// of series never reads: it plans its own units.
    pub fn series_settings(&self) -> Result<Option<SeriesSettings>, String> {
        let Some(series) = &self.series else {
            return Ok(None);
        };
        if self.regions.is_some() || self.time_bucket.is_some() {
            return Err("a table of series takes no regions or time bucket".into());
        }
        if self.strategy != StrategySpec::default() {
            return Err("a table of series takes no strategy: its groups are its contexts".into());
        }
        if series.known.iter().collect::<HashSet<_>>().len() != series.known.len() {
            return Err("a known column is named twice".into());
        }
        let step = NonZeroU64::new(series.step).ok_or("a series table's step must be positive")?;
        let fixed = [&series.series, &series.time, &series.target];
        if fixed.iter().collect::<HashSet<_>>().len() != fixed.len() {
            return Err("the series, time and target columns need names of their own".into());
        }
        if self
            .columns
            .iter()
            .any(|column| fixed.contains(&&column.name))
        {
            return Err(
                "covariate columns need names other than the series, time and target columns"
                    .into(),
            );
        }
        if gf_core::columns::retrieval(&self.columns) > 0 {
            return Err("a table of series takes no retrieval columns".into());
        }
        let starts = gf_core::columns::offsets(&self.columns);
        let mut known = Vec::new();
        for name in &series.known {
            let index = self
                .columns
                .iter()
                .position(|column| column.name == *name)
                .ok_or_else(|| format!("known column {name:?} is not a covariate column"))?;
            let width = self.columns[index].kind.width();
            known.extend(starts[index]..starts[index] + width);
        }
        let sized = |size: u64| {
            usize::try_from(size)
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or("a group's size must be positive")
        };
        let group = match series.group {
            GroupSpec::Alone => GroupStrategy::Alone,
            GroupSpec::Random { size, seed } => GroupStrategy::Random {
                size: sized(size)?,
                seed,
            },
            GroupSpec::Given { size } => GroupStrategy::Given { size: sized(size)? },
        };
        if series.horizon == Some(0) {
            return Err("a horizon is at least one step".into());
        }
        Ok(Some(SeriesSettings { step, known, group }))
    }

    /// Number of values its columns become in a row: one per column, and
    /// more for a datetime column.
    #[must_use]
    pub fn features(&self) -> usize {
        gf_core::columns::width(&self.columns)
    }

    /// Number of those values the model reads: all but the retrieval values.
    #[must_use]
    pub fn model_features(&self) -> usize {
        self.features() - gf_core::columns::retrieval(&self.columns)
    }

    /// The definition's settings, checked once, as the node serves the table.
    ///
    /// # Errors
    ///
    /// If a setting is out of range: a page size, time bucket or fit row
    /// count of 0, or a refit limit that is not positive and finite.
    pub fn config(&self) -> Result<TableConfig, String> {
        let cutoffs: CutoffRefit = self.refit_cutoffs.parse()?;
        Ok(TableConfig {
            task: Task::of_classes(self.classes),
            strategy: self.strategy.strategy(),
            page_size: NonZeroUsize::new(self.page_size).ok_or("page_size must be positive")?,
            time_bucket: self
                .time_bucket
                .map(|width| NonZeroU64::new(width).ok_or("time_bucket must be positive"))
                .transpose()?,
            refit: RefitPolicy::new(self.refit_after, self.refit_shift)
                .and_then(|policy| policy.with_cutoffs(cutoffs))
                .map_err(|error| error.to_string())?,
            scale_rows: self
                .scale_rows
                .map(|rows| NonZeroU64::new(rows).ok_or("scale_rows must be positive"))
                .transpose()?,
            categories: self.retrieval_categoricals.parse()?,
            fit_rows: self
                .fit_rows
                .map_or(Some(DEFAULT_FIT_ROWS), |rows| usize::try_from(rows).ok())
                .and_then(NonZeroUsize::new)
                .ok_or("fit_rows must be positive")?,
            past_cap: match self.past_cap {
                PastCapSpec::Warn => PastCap::Warn,
                PastCapSpec::Clamp => PastCap::Clamp,
            },
            regions: self
                .regions
                .map(|regions| -> Result<RegionSpec, String> {
                    let rebuild = match regions.rebuild {
                        RebuildSpec::Always => Rebuild::Always,
                        RebuildSpec::Never => Rebuild::Never,
                        RebuildSpec::Churn(share) if share > 0.0 && share.is_finite() => {
                            Rebuild::Churn(share)
                        }
                        RebuildSpec::Every(seconds) if seconds >= 0.0 && seconds.is_finite() => {
                            Rebuild::Every(std::time::Duration::from_secs_f64(seconds))
                        }
                        _ => return Err("a rebuild's churn share or period is out of range".into()),
                    };
                    if regions.rows == 0 {
                        return Err("a region's rows must be positive".into());
                    }
                    let time = regions
                        .wait
                        .seconds
                        .map(std::time::Duration::try_from_secs_f64)
                        .transpose()
                        .map_err(|_| "a rebuild's wait in seconds is out of range")?;
                    Ok(RegionSpec {
                        rows: regions.rows,
                        rebuild,
                        wait: RebuildWait {
                            steps: regions.wait.steps,
                            time,
                        },
                    })
                })
                .transpose()?,
            summary: self.summary.scope(),
        })
    }
}

/// What a table does with a context larger than its model was trained on;
/// see `PastCap`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PastCapSpec {
    #[default]
    Warn,
    Clamp,
}

/// How a table's queries share contexts by region; see `RegionSpec`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionsSpec {
    /// Live rows per region.
    pub rows: u64,
    #[serde(default)]
    pub rebuild: RebuildSpec,
    #[serde(default, skip_serializing_if = "RebuildWaitSpec::is_default")]
    pub wait: RebuildWaitSpec,
}

/// How a router sends a split table's rows and queries to partitions by one
/// column's value, so each partition answers its own queries. The value
/// places a row, so it cannot change for a key once the row is written. A
/// row whose value changes is deleted and inserted again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteBy {
    /// The numeric column routed by.
    pub column: String,
    #[serde(default)]
    pub kind: RouteKind,
    /// The increasing values that cut the column's range into partitions,
    /// one fewer than the partitions, for `ordered` routes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bounds: Vec<f64>,
}

/// How a route maps a value to the key space its partitions slice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    /// By a hash of the value, so partitions hold equal shares of the values.
    #[default]
    Hashed,
    /// In order of the value, so a partition holds one range of it.
    Ordered,
}

impl RouteBy {
    /// Checks the route against the table's columns.
    ///
    /// # Errors
    ///
    /// If the column is not a numeric column of the table, or a hashed route
    /// has bounds, or an ordered route's bounds are not finite and increasing.
    pub fn check(&self, columns: &[Column]) -> Result<(), String> {
        let column = columns
            .iter()
            .find(|column| column.name == self.column)
            .ok_or_else(|| format!("route_by column {:?} is not a column", self.column))?;
        if column.kind != gf_core::columns::ColumnKind::Numeric {
            return Err(format!("route_by column {:?} must be numeric", self.column));
        }
        match self.kind {
            RouteKind::Hashed if !self.bounds.is_empty() => {
                Err("a hashed route takes no bounds".into())
            }
            RouteKind::Hashed => Ok(()),
            RouteKind::Ordered => {
                let finite = self.bounds.iter().all(|bound| bound.is_finite());
                let increasing = self.bounds.windows(2).all(|pair| pair[0] < pair[1]);
                (finite && increasing)
                    .then_some(())
                    .ok_or_else(|| "an ordered route's bounds must be finite and increasing".into())
            }
        }
    }

    /// Where a value sits in the route's key space. Negative zero sits where
    /// zero does.
    #[must_use]
    pub fn key(&self, value: f64) -> gf_assign::Key {
        let value = value + 0.0;
        match self.kind {
            RouteKind::Hashed => gf_assign::key::hashed(&value.to_bits().to_le_bytes()),
            RouteKind::Ordered => gf_assign::key::ordered(value),
        }
    }

    /// The partitions' slices of the key space: equal shares of it for a
    /// hashed route, or the ranges between the bounds for an ordered one,
    /// each held by the partition of its index. An ordered route gives one
    /// slice more than it has bounds, whatever `count` says.
    #[must_use]
    pub fn assignment(&self, count: usize) -> gf_assign::Assignment {
        match self.kind {
            RouteKind::Hashed => gf_assign::Assignment::even(count, count, true),
            RouteKind::Ordered => gf_assign::Assignment {
                generation: 0,
                slices: std::iter::once(0)
                    .chain(self.bounds.iter().map(|&bound| self.key(bound)))
                    .enumerate()
                    .map(|(index, start)| gf_assign::Slice {
                        start,
                        holders: vec![index],
                    })
                    .collect(),
            },
        }
    }
}

/// How long a context due a rebuild is read stale while requests queue; see
/// `RebuildWait`. Without `seconds`, only request steps count.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RebuildWaitSpec {
    pub steps: u32,
    pub seconds: Option<f64>,
}

impl Default for RebuildWaitSpec {
    fn default() -> Self {
        let wait = RebuildWait::default();
        Self {
            steps: wait.steps,
            seconds: wait.time.map(|time| time.as_secs_f64()),
        }
    }
}

impl RebuildWaitSpec {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// How a table of series is forecast; see `SeriesSpec`.
///
/// The series, time and target columns name what clients' frames hold. The
/// server reads observations as keyed rows: the client packs the series'
/// digest and the time into each row key, as `generalforecast.keys.series_keys` does.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesDefinition {
    /// The frame column naming each observation's series.
    pub series: String,
    /// The frame column holding each observation's time.
    pub time: String,
    /// The frame column holding the value forecast.
    pub target: String,
    /// The grid's step, in the table's time unit.
    pub step: u64,
    /// The covariate columns known over the horizon, by name.
    #[serde(default)]
    pub known: Vec<String>,
    #[serde(default)]
    pub group: GroupSpec,
    /// Steps a forecast covers when its call names none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon: Option<u64>,
}

/// Which other series join a series in its group; see `GroupStrategy`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GroupSpec {
    #[default]
    Alone,
    /// `size - 1` others by seeded hash with the series; a table holding fewer refuses the request.
    Random { size: u64, seed: u64 },
    /// The members each forecast names, up to `size` series; the table draws none.
    Given { size: u64 },
}

/// A series definition checked: what the node's `SeriesSpec` takes from it,
/// beside what the model's profile gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeriesSettings {
    pub step: NonZeroU64,
    /// The model features of the known columns.
    pub known: Vec<usize>,
    pub group: GroupStrategy,
}

/// When a region's context is built again; see `Rebuild`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RebuildSpec {
    #[default]
    Always,
    /// Once changed rows reach this share of the context's.
    Churn(f64),
    /// Once this many seconds passed since the build.
    Every(f64),
    Never,
}

/// A table's definition, checked: the settings its node serves it with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableConfig {
    pub task: Task,
    pub strategy: Strategy,
    pub page_size: NonZeroUsize,
    pub time_bucket: Option<NonZeroU64>,
    pub refit: RefitPolicy,
    pub scale_rows: Option<NonZeroU64>,
    pub categories: CategoryEncoding,
    pub fit_rows: NonZeroUsize,
    pub past_cap: PastCap,
    pub regions: Option<RegionSpec>,
    pub summary: SummaryScope,
}

fn each() -> String {
    "each".to_string()
}

fn codes() -> String {
    "codes".to_string()
}

fn page_size() -> usize {
    128
}

/// Rows a record carries: keys, row-major features, targets, and times if any.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub keys: Vec<u128>,
    pub features: Vec<f32>,
    pub targets: Targets,
    /// Each row's time if it has one; None when the rows were written
    /// without times. Only a snapshot mixes rows with and without times.
    pub times: Option<Vec<Option<i64>>>,
    /// The client's id for the write, if it sent one, so a retry applies once.
    pub id: Option<String>,
    /// Words the record adds to columns' vocabularies, in code order; a
    /// snapshot's are every word.
    pub words: Vec<Words>,
    /// A snapshot's: the writes the table last applied, by client id, so a
    /// retry still applies once after the log is compacted.
    pub writes: Vec<PastWrite>,
    /// A compaction's: each row's id and page, rows in page order, and the
    /// pages that begin among them, so a copy rebuilt from the log keeps them.
    pub layout: Option<RowLayout>,
    /// A compaction's first record's: what the table holds besides its rows.
    pub table: Option<Box<TableLayout>>,
}

/// A write a table applied, as a snapshot remembers it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PastWrite {
    pub id: String,
    pub rows: u64,
    pub log_length: u64,
}

/// A change to a table.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Insert(Rows),
    Update(Rows),
    /// Rows replacing those of keys the table holds, and added otherwise.
    Upsert(Rows),
    Delete {
        keys: Vec<u128>,
        /// The client's id for the write, if it sent one.
        id: Option<String>,
    },
    Refit,
    /// Every row deleted, as when the source table is truncated.
    Truncate,
    /// Every live row at this point: the table is exactly these rows.
    Snapshot(Rows),
    /// From here on the table's writes go to the node named `primary`, under
    /// fencing epoch `term`.
    Term {
        term: u64,
        primary: String,
    },
}

impl Record {
    /// The client's id for the write, if the record carries one.
    #[must_use]
    pub fn write_id(&self) -> Option<&str> {
        match self {
            Self::Insert(rows) | Self::Update(rows) | Self::Upsert(rows) => rows.id.as_deref(),
            Self::Delete { id, .. } => id.as_deref(),
            Self::Refit | Self::Truncate | Self::Snapshot(_) | Self::Term { .. } => None,
        }
    }

    /// The keys of the rows the record changes; none for a refit or term.
    #[must_use]
    pub fn keys(&self) -> &[u128] {
        match self {
            Self::Insert(rows) | Self::Update(rows) | Self::Upsert(rows) | Self::Snapshot(rows) => {
                &rows.keys
            }
            Self::Delete { keys, .. } => keys,
            Self::Refit | Self::Truncate | Self::Term { .. } => &[],
        }
    }

    /// The words the record adds to vocabularies, if it carries rows.
    #[must_use]
    pub fn words(&self) -> &[Words] {
        match self {
            Self::Insert(rows) | Self::Update(rows) | Self::Upsert(rows) | Self::Snapshot(rows) => {
                &rows.words
            }
            Self::Delete { .. } | Self::Refit | Self::Truncate | Self::Term { .. } => &[],
        }
    }
}

/// Who takes a table's writes: the epoch a promotion began, the node it named,
/// and the sequence number of the record that began it. Term 0, the table's
/// first, has no named primary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    pub term: u64,
    pub primary: String,
    pub start: u64,
}

/// Every term a table has had, oldest first. Never empty: a table begins at term 0.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<Term>", into = "Vec<Term>")]
pub struct Terms(Vec<Term>);

impl Default for Terms {
    /// A new table's: term 0 alone.
    fn default() -> Self {
        Self(vec![Term::default()])
    }
}

impl TryFrom<Vec<Term>> for Terms {
    type Error = String;

    fn try_from(terms: Vec<Term>) -> Result<Self, Self::Error> {
        if terms.is_empty() {
            return Err("a table has term 0 at least".to_string());
        }
        Ok(Self(terms))
    }
}

impl From<Terms> for Vec<Term> {
    fn from(terms: Terms) -> Self {
        terms.0
    }
}

impl std::ops::Deref for Terms {
    type Target = [Term];

    fn deref(&self) -> &[Term] {
        &self.0
    }
}

impl Terms {
    /// The latest term: who takes the table's writes now.
    #[must_use]
    pub fn latest(&self) -> &Term {
        self.0.last().expect("terms are never empty")
    }

    /// Begins a later term.
    pub fn begin(&mut self, term: Term) {
        self.0.push(term);
    }

    /// Forgets the terms begun at `length` or later, keeping term 0.
    fn cut(&mut self, length: u64) {
        self.0.retain(|term| term.start < length || term.term == 0);
        if self.0.is_empty() {
            self.0.push(Term::default());
        }
    }
}

/// One entry of a log, as replicas receive it.
#[derive(Clone, Debug, PartialEq)]
pub enum LogEntry {
    /// The table's definition, and its terms when the log was last compacted.
    Definition(TableDefinition, Terms),
    Record(Record),
}

/// Decodes an entry read from a log.
///
/// # Errors
///
/// If the bytes are not a definition or a record.
pub fn decode(bytes: &[u8]) -> Result<LogEntry, LogError> {
    let frame = Frame::from_bytes(bytes)?;
    if frame.header.get("record").and_then(Value::as_str) == Some("table") {
        return Ok(LogEntry::Definition(
            definition_from(&frame)?,
            terms_from(&frame)?,
        ));
    }
    Ok(LogEntry::Record(record_from(&frame)?))
}

/// Strategies tables could once name, whose stitched contexts this server no
/// longer builds.
const RETIRED_STRATEGIES: [&str; 2] = ["pages", "global_local"];

fn definition_from(frame: &Frame) -> Result<TableDefinition, LogError> {
    let definition = &frame.header["definition"];
    if let Some(kind) = definition.pointer("/strategy/kind").and_then(Value::as_str)
        && RETIRED_STRATEGIES.contains(&kind)
    {
        return Err(LogError::RetiredStrategy(kind.to_string()));
    }
    serde_json::from_value(definition.clone())
        .map_err(|error| LogError::Malformed(error.to_string()))
}

fn terms_from(frame: &Frame) -> Result<Terms, LogError> {
    match frame.header.get("terms") {
        Some(terms) => serde_json::from_value(terms.clone())
            .map_err(|error| LogError::Malformed(error.to_string())),
        None => Ok(Terms::default()),
    }
}

/// The key that seals log records: 32 bytes, wiped when dropped.
#[derive(Clone)]
pub struct EncryptionKey(Key);

impl EncryptionKey {
    /// A key from 64 hex digits.
    ///
    /// # Errors
    ///
    /// If the string is not 32 bytes of hex.
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        if hex.len() != 64 || !hex.is_ascii() {
            return Err("an encryption key is 64 hex digits".to_string());
        }
        let mut bytes: Vec<u8> = (0..32)
            .map(|index| u8::from_str_radix(&hex[2 * index..2 * index + 2], 16))
            .collect::<Result<_, _>>()
            .map_err(|_| "an encryption key is 64 hex digits".to_string())?;
        let key = Self(*Key::from_slice(&bytes));
        bytes.zeroize();
        Ok(key)
    }

    /// An id of the key that reveals nothing about it: the tag of an empty
    /// message sealed under it, as hex.
    #[must_use]
    pub fn id(&self) -> String {
        let tag = XChaCha20Poly1305::new(&self.0)
            .encrypt(
                XNonce::from_slice(&[0; NONCE]),
                Payload {
                    msg: &[],
                    aad: b"gf log key",
                },
            )
            .expect("sealing an empty message cannot fail");
        tag[..KEY_ID]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn id_bytes(&self) -> [u8; KEY_ID] {
        let id = self.id();
        let mut bytes = [0; KEY_ID];
        for (byte, pair) in bytes.iter_mut().zip(id.as_bytes().chunks(2)) {
            *byte = u8::from_str_radix(std::str::from_utf8(pair).expect("hex"), 16).expect("hex");
        }
        bytes
    }
}

impl Drop for EncryptionKey {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

/// The key records are sealed under, and earlier keys logs may still be sealed under.
#[derive(Clone)]
pub struct Keys {
    pub current: EncryptionKey,
    pub previous: Vec<EncryptionKey>,
}

impl Keys {
    /// Only one key, current and past.
    #[must_use]
    pub fn only(current: EncryptionKey) -> Self {
        Self {
            current,
            previous: Vec::new(),
        }
    }

    fn find(&self, id: &[u8; KEY_ID]) -> Option<(&EncryptionKey, bool)> {
        if self.current.id_bytes() == *id {
            return Some((&self.current, true));
        }
        self.previous
            .iter()
            .find(|key| key.id_bytes() == *id)
            .map(|key| (key, false))
    }
}

/// Why a log could not be read or written.
#[derive(Debug)]
pub enum LogError {
    Io(std::io::Error),
    /// The file is not a log of a format this server reads.
    Format(String),
    /// The log is sealed under a key the server was not given.
    Key {
        id: String,
    },
    /// The log is sealed under a previous key; compact it before appending.
    Rotate,
    /// A complete record failed to open: the wrong key, or changed bytes.
    Decrypt {
        offset: u64,
    },
    /// A record's length prefix was changed.
    Length {
        offset: u64,
    },
    /// An earlier write failed in a way that left the file unknown.
    Poisoned(String),
    /// The log is shorter than its head says it was synced to, or its head is
    /// missing or does not open.
    Head(String),
    /// A read from a sequence number past the log's end.
    Beyond {
        from: u64,
        length: u64,
    },
    Malformed(String),
    /// The table names a strategy this server no longer serves; its log is
    /// whole and stays where it is.
    RetiredStrategy(String),
}

impl LogError {
    /// Whether the file itself is damaged, rather than unreadable for now, sealed
    /// under a key the server lacks, or short of records its head expects.
    #[must_use]
    pub fn damaged(&self) -> bool {
        matches!(
            self,
            Self::Format(_) | Self::Decrypt { .. } | Self::Length { .. } | Self::Malformed(_)
        )
    }
}

impl fmt::Display for LogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Format(reason) => write!(formatter, "not a log this server reads: {reason}"),
            Self::Key { id } => write!(
                formatter,
                "the log is sealed under key {id}, which the server was not given; \
                 name it in previous_key_envs to rotate it"
            ),
            Self::Rotate => write!(
                formatter,
                "the log is sealed under a previous key; compact it before appending"
            ),
            Self::Decrypt { offset } => {
                write!(formatter, "the log record at byte {offset} does not open")
            }
            Self::Length { offset } => {
                write!(
                    formatter,
                    "the log record at byte {offset} has a damaged length"
                )
            }
            Self::Poisoned(reason) => write!(
                formatter,
                "the log takes no more writes after a failed one ({reason}); \
                 restart the server to recover it"
            ),
            Self::Head(reason) => write!(formatter, "the log lost records from its end: {reason}"),
            Self::Beyond { from, length } => write!(
                formatter,
                "the log holds {length} entries, so it cannot be read from entry {from}"
            ),
            Self::Malformed(message) => formatter.write_str(message),
            Self::RetiredStrategy(kind) => write!(
                formatter,
                "the table uses the retired {kind} strategy; recreate it with \
                 an exact or retrieved one"
            ),
        }
    }
}

impl std::error::Error for LogError {}

impl From<std::io::Error> for LogError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<FrameError> for LogError {
    fn from(error: FrameError) -> Self {
        Self::Malformed(error.0)
    }
}

/// An open log, positioned to append.
pub struct Log {
    path: PathBuf,
    /// The table's name and creation id, which every record is sealed with.
    table: TableName,
    created: Option<CreationId>,
    file: File,
    /// Seals new records and the head, under the current key, whose id
    /// heads the head file.
    cipher: XChaCha20Poly1305,
    key_id: [u8; KEY_ID],
    /// Opens the records on disk: the current key, or the previous key they
    /// were sealed under until the log is compacted.
    reader: XChaCha20Poly1305,
    rotate: bool,
    /// Where each entry starts, then where the next one will.
    offsets: Vec<u64>,
    /// How many of `offsets` are synced to disk, and the tag then.
    durable: usize,
    durable_tag: [u8; TAG],
    /// The tag of the last entry, which the next is sealed with.
    last_tag: [u8; TAG],
    /// The sequence number of the first record; later than 1 after compaction.
    base: u64,
    terms: Terms,
    /// The length the head file holds, and when it was written.
    head: u64,
    head_at: Instant,
    /// Why the file may hold more or less than `offsets` says, after a failed write.
    poisoned: Option<String>,
}

/// Records a sync may run past the head, and how long the head may stay
/// behind a sync, before a sync writes the head too. Writing it costs three
/// more syncs, so a busy table writes it every so often rather than with
/// every commit; a record past the head that does not open is cut off as a
/// torn append, rather than refused, until the head catches up, as it does
/// when the table's writes go idle.
const HEAD_LAG: u64 = 64;
/// How long the head may stay behind a sync before a sync writes it too.
pub const HEAD_INTERVAL: Duration = Duration::from_secs(1);

fn definition_frame(
    definition: &TableDefinition,
    base: u64,
    terms: &Terms,
) -> Result<Frame, LogError> {
    let header =
        serde_json::to_value(definition).map_err(|error| LogError::Malformed(error.to_string()))?;
    Ok(Frame::header(
        json!({"record": "table", "definition": header, "base": base, "terms": terms}),
    ))
}

fn file_header(key: &EncryptionKey) -> [u8; HEADER] {
    let mut header = [0; HEADER];
    header[..MAGIC.len()].copy_from_slice(MAGIC);
    header[MAGIC.len()] = FORMAT;
    header[MAGIC.len() + 2..].copy_from_slice(&key.id_bytes());
    header
}

impl Log {
    /// Creates a table's log, starting with its definition and terms: term 0
    /// for a new table, or its source's for a replica.
    ///
    /// # Errors
    ///
    /// If the file exists or cannot be written.
    pub fn create(
        path: &Path,
        key: &EncryptionKey,
        definition: &TableDefinition,
        terms: Terms,
    ) -> Result<Self, LogError> {
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .read(true)
            .open(path)?;
        remove_staged_compactions(path);
        let cipher = XChaCha20Poly1305::new(&key.0);
        let mut log = Self {
            path: path.to_path_buf(),
            table: definition.name.clone(),
            created: definition.created.clone(),
            file,
            reader: cipher.clone(),
            cipher,
            key_id: key.id_bytes(),
            rotate: false,
            offsets: vec![HEADER as u64],
            durable: 1,
            durable_tag: [0; TAG],
            last_tag: [0; TAG],
            base: 1,
            terms,
            head: 1,
            head_at: Instant::now(),
            poisoned: None,
        };
        // The head goes first, so a crash never leaves the log without one.
        let written = log
            .write_head(1)
            .map_err(LogError::from)
            .and_then(|()| log.seal(&definition_frame(definition, 1, &log.terms)?, 0))
            .and_then(|bytes| {
                log.file.write_all(&file_header(key))?;
                log.file.write_all(&bytes)?;
                log.file.sync_data()?;
                sync_directory(path)?;
                Ok(bytes)
            });
        match written {
            Ok(bytes) => {
                log.offsets.push(HEADER as u64 + bytes.len() as u64);
                log.last_tag = tag_of(&bytes);
                log.durable = 2;
                log.durable_tag = log.last_tag;
                Ok(log)
            }
            Err(error) => {
                let _ = remove(path);
                Err(error)
            }
        }
    }

    /// Opens a log to replay: its definition is read now, and its records come
    /// from the returned `Replay`, which `Replay::finish` turns into the log.
    ///
    /// # Errors
    ///
    /// If the file is not a log, is sealed under a key not in `keys`, has no
    /// head that opens, or the file cannot be read.
    pub fn open(path: &Path, keys: &Keys) -> Result<Replay, LogError> {
        Replay::start(path, keys, true)
    }

    /// Opens a log to read only, as a check does: a torn final record is
    /// reported by `Replay::finish` but left on disk.
    ///
    /// # Errors
    ///
    /// As `open`.
    pub fn open_read_only(path: &Path, keys: &Keys) -> Result<Replay, LogError> {
        Replay::start(path, keys, false)
    }

    /// Writes the head, sealed and synced: staged beside it, then renamed over
    /// it, so a crash leaves the old head or the new one whole.
    fn write_head(&self, length: u64) -> std::io::Result<()> {
        let mut nonce = [0u8; NONCE];
        getrandom::fill(&mut nonce).map_err(|error| std::io::Error::other(error.to_string()))?;
        let body = serde_json::to_vec(&Head {
            length,
            created: self.created.clone(),
        })
        .map_err(std::io::Error::other)?;
        let sealed = self
            .cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: &body,
                    aad: &head_associated(&self.table),
                },
            )
            .map_err(|_| std::io::Error::other("sealing the head failed"))?;
        let mut bytes = self.key_id.to_vec();
        bytes.extend_from_slice(&nonce);
        bytes.extend_from_slice(&sealed);
        let path = head_path(&self.path);
        let staged = self.path.with_extension("head-new");
        let mut file = File::create(&staged)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&staged, &path)?;
        sync_directory(&path)
    }

    /// Writes the head if records were synced since it was last written.
    ///
    /// # Errors
    ///
    /// If the head cannot be written; it then still holds an earlier length.
    pub fn save_head(&mut self) -> std::io::Result<()> {
        let synced = self.durable_length();
        if self.head < synced {
            self.write_head(synced)?;
            self.head_written(synced);
        }
        Ok(())
    }

    /// Writes the head through every synced record, so a log closed cleanly
    /// checks all its acknowledged records.
    ///
    /// # Errors
    ///
    /// If the head cannot be written.
    pub fn close(mut self) -> std::io::Result<()> {
        self.save_head()
    }

    /// Notes a head written at `length`.
    fn head_written(&mut self, length: u64) {
        self.head = length;
        self.head_at = Instant::now();
    }

    /// The length the log had at its last sync.
    fn durable_length(&self) -> u64 {
        self.base + self.durable as u64 - 2
    }

    /// The sequence number the next record gets: entries so far, counting the
    /// definition and any records compaction folded into a snapshot.
    #[must_use]
    pub fn length(&self) -> u64 {
        self.base + self.offsets.len() as u64 - 2
    }

    /// The sequence number of the first record on disk; later than 1 after compaction.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Bytes the log takes on disk.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.offsets.last().copied().unwrap_or_default()
    }

    /// Whether the log is sealed under a previous key and awaits compaction.
    #[must_use]
    pub fn needs_rotation(&self) -> bool {
        self.rotate
    }

    /// The table's creation id, as its definition records it.
    #[must_use]
    pub fn created(&self) -> Option<&CreationId> {
        self.created.as_ref()
    }

    /// The creation id entry `sequence` is sealed with: none for the
    /// definition, which names the creation itself.
    fn sealed_creation(&self, sequence: u64) -> &str {
        match &self.created {
            Some(created) if sequence != 0 => created.as_str(),
            _ => "",
        }
    }

    /// Who takes the table's writes, as its records last said.
    #[must_use]
    pub fn term(&self) -> &Term {
        self.terms.latest()
    }

    /// Every term the table has had, oldest first.
    #[must_use]
    pub fn terms(&self) -> &Terms {
        &self.terms
    }

    /// Takes later terms from another copy's definition, as when that copy's
    /// log was compacted past the records that began them; the snapshot that
    /// follows makes them durable.
    pub fn adopt(&mut self, terms: Terms) {
        if terms.latest().term > self.term().term {
            self.terms = terms;
        }
    }

    /// The sequence number of the entry at `index` in the file.
    fn sequence(&self, index: usize) -> u64 {
        if index == 0 {
            0
        } else {
            self.base + index as u64 - 1
        }
    }

    /// Entries with sequence numbers from `from` on, at most `limit` of them,
    /// decrypted, for a replica to apply. Records compaction folded away are
    /// not there: a replica that far behind starts from the definition, for
    /// its term, and the snapshot.
    ///
    /// # Errors
    ///
    /// `Beyond` if `from` is past the log's synced end, or if the file cannot
    /// be read or an entry does not open.
    pub fn read(&self, from: u64, limit: usize) -> Result<Vec<(u64, Vec<u8>)>, LogError> {
        let length = self.durable_length();
        if from > length {
            return Err(LogError::Beyond { from, length });
        }
        let first = if from < self.base {
            0
        } else {
            usize::try_from(from - self.base + 1).unwrap_or(usize::MAX)
        };
        let mut entries = Vec::new();
        let mut previous = [0; TAG];
        if first > 0 {
            let before = self.offsets[first - 1]..self.offsets[first];
            let mut sealed = vec![0u8; usize::try_from(before.end - before.start).expect("fits")];
            self.file.read_exact_at(&mut sealed, before.start)?;
            previous = tag_of(&sealed[PREFIX..]);
        }
        let synced = &self.offsets[..self.durable];
        for (index, window) in synced.windows(2).enumerate().skip(first) {
            if entries.len() >= limit {
                break;
            }
            let sequence = self.sequence(index);
            let mut sealed = vec![0u8; usize::try_from(window[1] - window[0]).expect("fits")];
            self.file.read_exact_at(&mut sealed, window[0])?;
            let (nonce, ciphertext) = sealed[PREFIX..].split_at(NONCE);
            let created = self.sealed_creation(sequence);
            let plain = self
                .reader
                .decrypt(
                    XNonce::from_slice(nonce),
                    Payload {
                        msg: ciphertext,
                        aad: &associated(&self.table, created, sequence, &previous),
                    },
                )
                .map_err(|_| LogError::Decrypt { offset: window[0] })?;
            previous = tag_of(&sealed[PREFIX..]);
            entries.push((sequence, plain));
        }
        Ok(entries)
    }

    /// Refuses writes once a failed one left the file unknown, or the log
    /// awaits rewriting under the current key.
    /// Why the log takes no more writes until the server restarts, if it does not.
    #[must_use]
    pub fn poisoned(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    fn writable(&self) -> Result<(), LogError> {
        if self.rotate {
            return Err(LogError::Rotate);
        }
        match &self.poisoned {
            Some(reason) => Err(LogError::Poisoned(reason.clone())),
            None => Ok(()),
        }
    }

    /// Appends a record and syncs it to disk before returning.
    ///
    /// A failed write or sync is cut back off the file, so the log holds
    /// exactly its entries before it; if that fails too, the log takes no more
    /// writes.
    ///
    /// # Errors
    ///
    /// If the write or sync fails, or an earlier one left the log unusable.
    pub fn append(&mut self, record: &Record) -> Result<(), LogError> {
        self.stage(record)?;
        self.sync()
    }

    /// Writes a record without syncing it, for `sync` to make durable with
    /// others; until then it is not an entry a reader or replica may see.
    ///
    /// # Errors
    ///
    /// If the write fails, which cuts the staged records back off, or an
    /// earlier one left the log unusable.
    pub fn stage(&mut self, record: &Record) -> Result<(), LogError> {
        self.writable()?;
        let sequence = self.length();
        let bytes = self.seal(&record_frame(record), sequence)?;
        let start = self.offsets.last().copied().unwrap_or_default();
        if let Err(error) = self.file.write_all(&bytes) {
            self.cut_back();
            return Err(error.into());
        }
        self.offsets.push(start + bytes.len() as u64);
        self.last_tag = tag_of(&bytes);
        if let Record::Term { term, primary } = record {
            self.terms.begin(Term {
                term: *term,
                primary: primary.clone(),
                start: sequence,
            });
        }
        Ok(())
    }

    /// Syncs the staged records to disk; a failure cuts them back off, so
    /// they were never written. The head follows every `HEAD_LAG` records or
    /// `HEAD_INTERVAL`, and the shard saves it when its writes go idle.
    ///
    /// # Errors
    ///
    /// If the sync fails, or an earlier failure left the log unusable.
    pub fn sync(&mut self) -> Result<(), LogError> {
        self.writable()?;
        if self.offsets.len() == self.durable {
            return Ok(());
        }
        if let Err(error) = self.file.sync_data() {
            self.cut_back();
            return Err(error.into());
        }
        self.durable = self.offsets.len();
        self.durable_tag = self.last_tag;
        let length = self.length();
        if length - self.head >= HEAD_LAG || self.head_at.elapsed() >= HEAD_INTERVAL {
            // The records are durable; a head that failed to write still
            // holds an earlier length, which only weakens the check until
            // the next save.
            match self.write_head(length) {
                Ok(()) => self.head_written(length),
                Err(error) => {
                    tracing::error!(log = %self.path.display(), "writing the head: {error}");
                }
            }
        }
        Ok(())
    }

    /// Whether records are staged but not yet synced.
    #[must_use]
    pub fn has_staged(&self) -> bool {
        self.offsets.len() > self.durable
    }

    /// Cuts staged records back off, for a caller that will not sync them.
    pub fn discard_staged(&mut self) {
        if self.has_staged() {
            self.cut_back();
        }
    }

    /// Cuts the file back to its last synced entry, and the log's state with it.
    fn cut_back(&mut self) {
        let start = self.offsets[self.durable - 1];
        self.offsets.truncate(self.durable);
        self.last_tag = self.durable_tag;
        let length = self.length();
        self.terms.cut(length);
        let cut = self
            .file
            .set_len(start)
            .and_then(|()| self.file.sync_data());
        if let Err(cut) = cut {
            self.poisoned = Some(format!("cutting a failed write off: {cut}"));
        }
    }

    /// Rewrites the log as its definition and `records`, a snapshot and any
    /// records that follow it, from sequence number `at` on; later records
    /// continue from there. The new log is sealed under the current key.
    ///
    /// The new log is written beside the old one, synced, and renamed over it,
    /// so a crash leaves one or the other whole.
    ///
    /// # Errors
    ///
    /// If the files cannot be written or renamed.
    pub fn compact(
        &mut self,
        definition: &TableDefinition,
        at: u64,
        records: &[Record],
    ) -> Result<(), LogError> {
        let mut compactor = self.begin_compaction(definition, at)?;
        for record in records {
            compactor.append(record)?;
        }
        self.install_compaction(compactor.finish()?)
    }

    /// Rewrites the log under the current key, entry for entry, keeping every
    /// sequence number, so it stays the same log as its copies elsewhere,
    /// where a compaction would add an entry this copy alone holds.
    ///
    /// # Errors
    ///
    /// If records are staged, an entry does not open, or the files cannot be
    /// written or renamed.
    pub fn rotate(&mut self) -> Result<(), LogError> {
        if !self.rotate {
            return Ok(());
        }
        if let Some(reason) = &self.poisoned {
            return Err(LogError::Poisoned(reason.clone()));
        }
        if self.has_staged() {
            return Err(LogError::Malformed(
                "a log with records staged cannot be rewritten".to_string(),
            ));
        }
        let staged = self.path.with_extension("rotating");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&staged)?;
        file.write_all(&self.key_header())?;
        let mut offsets = vec![HEADER as u64];
        let (mut read, mut written) = ([0; TAG], [0; TAG]);
        for (index, window) in self.offsets.windows(2).enumerate() {
            let sequence = self.sequence(index);
            let mut sealed = vec![0u8; usize::try_from(window[1] - window[0]).expect("fits")];
            self.file.read_exact_at(&mut sealed, window[0])?;
            let (nonce, ciphertext) = sealed[PREFIX..].split_at(NONCE);
            let created = self.sealed_creation(sequence);
            let plain = self
                .reader
                .decrypt(
                    XNonce::from_slice(nonce),
                    Payload {
                        msg: ciphertext,
                        aad: &associated(&self.table, created, sequence, &read),
                    },
                )
                .map_err(|_| LogError::Decrypt { offset: window[0] })?;
            read = tag_of(&sealed);
            let bytes = self.seal_plain_after(&plain, sequence, &written)?;
            file.write_all(&bytes)?;
            written = tag_of(&bytes);
            offsets.push(offsets.last().copied().unwrap_or_default() + bytes.len() as u64);
        }
        file.sync_all()?;
        let base = self.base;
        self.install(&staged, offsets, written, base)
    }

    /// Renames a rewritten file over the log and takes up its entries; the
    /// head is written again, under the current key.
    fn install(
        &mut self,
        staged: &Path,
        offsets: Vec<u64>,
        last_tag: [u8; TAG],
        base: u64,
    ) -> Result<(), LogError> {
        std::fs::rename(staged, &self.path)?;
        // The old file is gone now, so a failure past here leaves the log unknown.
        let reopened = sync_directory(&self.path)
            .and_then(|()| OpenOptions::new().read(true).append(true).open(&self.path));
        match reopened {
            Ok(file) => {
                self.file = file;
                self.offsets = offsets;
                self.durable = self.offsets.len();
                self.last_tag = last_tag;
                self.durable_tag = last_tag;
                self.base = base;
                self.reader = self.cipher.clone();
                self.rotate = false;
                // A head that failed to write still holds an earlier length,
                // which only weakens the check until the next sync saves it.
                let length = self.durable_length();
                match self.write_head(length) {
                    Ok(()) => self.head_written(length),
                    Err(error) => {
                        tracing::error!(log = %self.path.display(), "writing the head: {error}");
                    }
                }
                Ok(())
            }
            Err(error) => {
                self.poisoned = Some(format!("reopening a rewritten log: {error}"));
                Err(error.into())
            }
        }
    }

    /// The file header naming the current key, which seals the log from the
    /// next compaction on.
    fn key_header(&self) -> [u8; HEADER] {
        let mut header = [0; HEADER];
        header[..MAGIC.len()].copy_from_slice(MAGIC);
        header[MAGIC.len()] = FORMAT;
        let tag = self
            .cipher
            .encrypt(
                XNonce::from_slice(&[0; NONCE]),
                Payload {
                    msg: &[],
                    aad: b"gf log key",
                },
            )
            .expect("sealing an empty message cannot fail");
        header[MAGIC.len() + 2..].copy_from_slice(&tag[..KEY_ID]);
        header
    }

    /// A frame sealed under its sequence number after the last entry, with its length prefix.
    fn seal(&self, frame: &Frame, sequence: u64) -> Result<Vec<u8>, LogError> {
        self.seal_after(frame, sequence, &self.last_tag)
    }

    fn seal_after(
        &self,
        frame: &Frame,
        sequence: u64,
        previous: &[u8; TAG],
    ) -> Result<Vec<u8>, LogError> {
        self.seal_plain_after(&frame.to_bytes(), sequence, previous)
    }

    /// An entry's plaintext sealed under its sequence number after `previous`,
    /// with its length prefix.
    fn seal_plain_after(
        &self,
        plain: &[u8],
        sequence: u64,
        previous: &[u8; TAG],
    ) -> Result<Vec<u8>, LogError> {
        seal_entry(
            &self.cipher,
            &self.table,
            self.sealed_creation(sequence),
            plain,
            sequence,
            previous,
        )
    }

    /// Starts rewriting the log as a compaction at `at`, its length now or, for
    /// a snapshot a replica skips ahead to, later: the
    /// definition sealed into a staged file, for `Compactor` to add the
    /// compaction's records to off the shard, and `install_compaction` to put
    /// in the log's place.
    ///
    /// # Errors
    ///
    /// If the log takes no writes, has records staged, or the staged file
    /// cannot be made.
    pub fn begin_compaction(
        &self,
        definition: &TableDefinition,
        at: u64,
    ) -> Result<Compactor, LogError> {
        if let Some(reason) = &self.poisoned {
            return Err(LogError::Poisoned(reason.clone()));
        }
        if self.has_staged() {
            return Err(LogError::Malformed(
                "a log with records staged cannot be compacted".to_string(),
            ));
        }
        let number = COMPACTIONS.fetch_add(1, Ordering::Relaxed);
        let staged = self.path.with_extension(format!("compacting.{number}"));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staged)?;
        file.write_all(&self.key_header())?;
        let mut compactor = Compactor {
            staged,
            number,
            file,
            cipher: self.cipher.clone(),
            table: self.table.clone(),
            created: self.created.clone(),
            offsets: vec![HEADER as u64],
            previous: [0; TAG],
            next: at,
            at,
            begun: self.length(),
        };
        let frame = definition_frame(definition, at, &self.terms)?;
        compactor.seal(&frame, 0)?;
        Ok(compactor)
    }

    /// Puts a finished compaction in the log's place.
    ///
    /// # Errors
    ///
    /// If the log took records since the compaction began, which the
    /// compaction does not hold, or the file cannot be put in place.
    pub fn install_compaction(&mut self, compacted: Compacted) -> Result<(), LogError> {
        if self.length() != compacted.begun || self.has_staged() {
            let _ = std::fs::remove_file(&compacted.staged);
            return Err(LogError::Malformed(
                "the log took records while it was compacted".to_string(),
            ));
        }
        self.install(
            &compacted.staged,
            compacted.offsets,
            compacted.previous,
            compacted.at,
        )
    }
}

/// A log being opened, replayed one record at a time: iterate it for the
/// records, then `finish` it for the log.
pub struct Replay {
    entries: Entries,
    definition: TableDefinition,
    writable: bool,
}

/// Reads a log's entries in order, checking each as it goes.
struct Entries {
    log: Log,
    reader: BufReader<File>,
    /// Bytes of the file when it was opened.
    size: u64,
    /// The length the head says was synced: entries before it must open.
    synced: u64,
    /// Where a torn final append begins, if one was found.
    torn: Option<u64>,
    done: bool,
}

impl Replay {
    fn start(path: &Path, keys: &Keys, writable: bool) -> Result<Self, LogError> {
        let table = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| TableName::new(stem).ok())
            .ok_or_else(|| LogError::Malformed("a log is named <table>.log".to_string()))?;
        let file = if writable {
            OpenOptions::new().read(true).append(true).open(path)?
        } else {
            File::open(path)?
        };
        let size = file.metadata()?.len();
        let mut reader = BufReader::new(file.try_clone()?);
        if size < HEADER as u64 {
            return Err(LogError::Format("missing header".to_string()));
        }
        let mut header = [0; HEADER];
        reader.read_exact(&mut header)?;
        if &header[..MAGIC.len()] != MAGIC {
            return Err(LogError::Format("missing header".to_string()));
        }
        if header[MAGIC.len()] != FORMAT {
            return Err(LogError::Format(format!(
                "format {} rather than {FORMAT}",
                header[MAGIC.len()]
            )));
        }
        let id: [u8; KEY_ID] = header[MAGIC.len() + 2..].try_into().expect("sized");
        let (key, current) = keys.find(&id).ok_or_else(|| LogError::Key {
            id: id.iter().map(|byte| format!("{byte:02x}")).collect(),
        })?;
        let head = read_head(path, &table, keys)?;
        let mut entries = Entries {
            log: Log {
                path: path.to_path_buf(),
                table,
                created: None,
                file,
                cipher: XChaCha20Poly1305::new(&keys.current.0),
                key_id: keys.current.id_bytes(),
                reader: XChaCha20Poly1305::new(&key.0),
                rotate: !current,
                offsets: vec![HEADER as u64],
                durable: 1,
                durable_tag: [0; TAG],
                last_tag: [0; TAG],
                base: 1,
                terms: Terms::default(),
                head: head.length,
                head_at: Instant::now(),
                poisoned: None,
            },
            reader,
            size,
            synced: head.length.max(1),
            torn: None,
            done: false,
        };
        let frame = match entries.next_entry()? {
            Some(plain) => Frame::from_bytes(&plain)?,
            None => return Err(LogError::Malformed("the log has no definition".to_string())),
        };
        let definition = definition_from(&frame)?;
        if definition.name != entries.log.table {
            return Err(LogError::Malformed(format!(
                "{} holds table {:?}",
                path.display(),
                definition.name
            )));
        }
        if head.created != definition.created {
            return Err(LogError::Head(
                "its head belongs to another creation of the table".to_string(),
            ));
        }
        entries.log.base = frame.header["base"].as_u64().unwrap_or(1);
        entries.log.created.clone_from(&definition.created);
        entries.log.terms = terms_from(&frame)?;
        if writable {
            remove_staged_compactions(path);
        }
        Ok(Self {
            entries,
            definition,
            writable,
        })
    }

    /// The table's definition, as the log's first entry records it.
    #[must_use]
    pub fn definition(&self) -> &TableDefinition {
        &self.definition
    }

    /// The log's length so far: the sequence number of the next record.
    #[must_use]
    pub fn length(&self) -> u64 {
        self.entries.log.length()
    }

    /// Whether the log is sealed under a previous key and awaits compaction.
    #[must_use]
    pub fn needs_rotation(&self) -> bool {
        self.entries.log.rotate
    }

    /// Reads the rest of the log, cuts off a torn final append, and returns
    /// the log, positioned to append.
    ///
    /// # Errors
    ///
    /// If a record before the head's length does not open, the log ends
    /// before it, or a torn append cannot be cut off.
    pub fn finish(mut self) -> Result<Log, LogError> {
        for record in self.by_ref() {
            record?;
        }
        let Entries {
            mut log,
            synced,
            torn,
            ..
        } = self.entries;
        if log.length() < synced {
            return Err(LogError::Head(format!(
                "it ends before entry {}, but it was synced through entry {}",
                log.length(),
                synced - 1
            )));
        }
        if let Some(offset) = torn {
            tracing::warn!(
                log = %log.path.display(),
                offset,
                "cutting off a record past the head that does not open, and everything after it: an append a crash tore before it was synced, or damage to a record synced since the head was last written"
            );
            if self.writable {
                log.file.set_len(offset)?;
                log.file.sync_data()?;
            }
        }
        log.durable = log.offsets.len();
        log.durable_tag = log.last_tag;
        Ok(log)
    }
}

impl Iterator for Replay {
    type Item = Result<Record, LogError>;

    fn next(&mut self) -> Option<Self::Item> {
        let entries = &mut self.entries;
        let plain = match entries.next_entry() {
            Ok(Some(plain)) => plain,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
        };
        let record = Frame::from_bytes(&plain)
            .map_err(LogError::from)
            .and_then(|frame| record_from(&frame));
        match &record {
            Ok(Record::Term { term, primary }) => {
                let start = entries.log.length() - 1;
                entries.log.terms.begin(Term {
                    term: *term,
                    primary: primary.clone(),
                    start,
                });
            }
            Ok(_) => {}
            Err(_) => entries.done = true,
        }
        Some(record)
    }
}

impl Entries {
    /// The next entry's plaintext, or None at the end. Damage at or past the
    /// head's length is a torn append, which ends the log.
    fn next_entry(&mut self) -> Result<Option<Vec<u8>>, LogError> {
        if self.done {
            return Ok(None);
        }
        let offset = *self.log.offsets.last().expect("the header at least");
        let sequence = self.log.sequence(self.log.offsets.len() - 1);
        match self.read_sealed(offset, sequence) {
            Ok(Some(opened)) => {
                self.log.offsets.push(opened.end);
                self.log.last_tag = opened.tag;
                Ok(Some(opened.plain))
            }
            Ok(None) => {
                self.done = true;
                Ok(None)
            }
            Err(error) => {
                self.done = true;
                if sequence >= self.synced && !matches!(error, LogError::Io(_)) {
                    self.torn = Some(offset);
                    return Ok(None);
                }
                Err(error)
            }
        }
    }

    /// The entry at `offset`, opened; None at the end of the file.
    fn read_sealed(&mut self, offset: u64, sequence: u64) -> Result<Option<Opened>, LogError> {
        let remaining = self.size - offset;
        if remaining == 0 {
            return Ok(None);
        }
        let synced = self.synced;
        let short = || {
            LogError::Head(format!(
                "it ends inside entry {sequence}, but it was synced through entry {}",
                synced - 1
            ))
        };
        if remaining < PREFIX as u64 {
            return Err(short());
        }
        let mut prefix = [0; PREFIX];
        self.reader.read_exact(&mut prefix)?;
        let length = u32::from_le_bytes(prefix[..4].try_into().expect("four bytes"));
        let complement = u32::from_le_bytes(prefix[4..].try_into().expect("four bytes"));
        if complement != !length || (length as usize) < NONCE + TAG {
            return Err(LogError::Length { offset });
        }
        let length = u64::from(length);
        if remaining - (PREFIX as u64) < length {
            return Err(short());
        }
        let mut sealed = vec![0; usize::try_from(length).expect("a u32 fits")];
        self.reader.read_exact(&mut sealed)?;
        let (nonce, ciphertext) = sealed.split_at(NONCE);
        let created = self.log.sealed_creation(sequence);
        let plain = self
            .log
            .reader
            .decrypt(
                XNonce::from_slice(nonce),
                Payload {
                    msg: ciphertext,
                    aad: &associated(&self.log.table, created, sequence, &self.log.last_tag),
                },
            )
            .map_err(|_| LogError::Decrypt { offset })?;
        Ok(Some(Opened {
            plain,
            tag: tag_of(&sealed),
            end: offset + PREFIX as u64 + length,
        }))
    }
}

/// An entry read and opened: its plaintext, its tag, and where it ends.
struct Opened {
    plain: Vec<u8>,
    tag: [u8; TAG],
    end: u64,
}

/// Reads a log's head, which must open under one of `keys`.
fn read_head(path: &Path, table: &str, keys: &Keys) -> Result<Head, LogError> {
    let bytes = match std::fs::read(head_path(path)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(LogError::Head("its head file is missing".to_string()));
        }
        Err(error) => return Err(error.into()),
    };
    let unreadable = || LogError::Head("its head file does not open".to_string());
    if bytes.len() < KEY_ID + NONCE {
        return Err(unreadable());
    }
    let (id, rest) = bytes.split_at(KEY_ID);
    let (nonce, ciphertext) = rest.split_at(NONCE);
    let (key, _) = keys
        .find(&id.try_into().expect("sized"))
        .ok_or_else(unreadable)?;
    let plain = XChaCha20Poly1305::new(&key.0)
        .decrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad: &head_associated(table),
            },
        )
        .map_err(|_| unreadable())?;
    serde_json::from_slice(&plain).map_err(|_| unreadable())
}

/// What a log's head file holds.
#[derive(Serialize, Deserialize)]
struct Head {
    length: u64,
    #[serde(default, deserialize_with = "CreationId::optional")]
    created: Option<CreationId>,
}

fn head_path(log: &Path) -> PathBuf {
    log.with_extension("head")
}

fn remove_staged_compactions(log: &Path) {
    let Some(stem) = log.file_stem().and_then(|stem| stem.to_str()) else {
        return;
    };
    let plain = format!("{stem}.compacting");
    let numbered = format!("{stem}.compacting.");
    let directory = log.parent().unwrap_or_else(|| Path::new("."));
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(directory = %directory.display(), "listing staged compactions: {error}");
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(directory = %directory.display(), "reading staged compaction entry: {error}");
                continue;
            }
        };
        let matches = entry.file_name().to_str().is_some_and(|name| {
            name == plain
                || name.strip_prefix(&numbered).is_some_and(|number| {
                    !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit())
                })
        });
        if !matches {
            continue;
        }
        if let Err(error) = std::fs::remove_file(entry.path())
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(staged = %entry.path().display(), "removing staged compaction: {error}");
        }
    }
}

/// What a head is sealed with besides its bytes.
fn head_associated(table: &str) -> Vec<u8> {
    let mut data = b"gf log head".to_vec();
    data.push(FORMAT);
    data.extend_from_slice(table.as_bytes());
    data
}

/// Removes a log and its head, and syncs their directory.
///
/// # Errors
///
/// If the log cannot be removed or the directory synced.
pub fn remove(path: &Path) -> std::io::Result<()> {
    std::fs::remove_file(path)?;
    match std::fs::remove_file(head_path(path)) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    sync_directory(path)
}

/// Renames a log and its head out of the way with `suffix` after their
/// extensions, so restarts leave them alone; returns where the log went.
///
/// # Errors
///
/// If the log cannot be renamed or the directory synced.
pub fn set_aside(path: &Path, suffix: &str) -> std::io::Result<PathBuf> {
    // A log set aside before under the same suffix stays, for whoever looks.
    let suffix = (0..)
        .map(|copy| match copy {
            0 => suffix.to_string(),
            copy => format!("{suffix}.{copy}"),
        })
        .find(|suffix| !path.with_extension(format!("log.{suffix}")).exists())
        .expect("some suffix is free");
    let aside = path.with_extension(format!("log.{suffix}"));
    std::fs::rename(path, &aside)?;
    let head = head_path(path);
    match std::fs::rename(&head, head.with_extension(format!("head.{suffix}"))) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    sync_directory(path)?;
    Ok(aside)
}

/// Sets a damaged log aside as `<table>.log.unreadable`, so the other tables
/// still start and a replica can be copied back, keeping it for inspection.
pub fn set_aside_damaged(path: &Path, error: &LogError) {
    match set_aside(path, "unreadable") {
        Ok(aside) => tracing::error!(
            log = %path.display(),
            aside = %aside.display(),
            "setting aside a log that does not open: {error}"
        ),
        Err(rename) => tracing::error!(
            log = %path.display(),
            "skipping a log that does not open ({error}) and cannot be moved: {rename}"
        ),
    }
}

/// The authentication tag ending a sealed entry, with or without its length prefix.
fn tag_of(sealed: &[u8]) -> [u8; TAG] {
    sealed[sealed.len() - TAG..].try_into().expect("sized")
}

/// What a record is sealed with besides its bytes: the format, its sequence
/// number, the previous entry's tag, and the table's creation id and name.
fn associated(table: &str, created: &str, sequence: u64, previous: &[u8; TAG]) -> Vec<u8> {
    let mut data = vec![FORMAT];
    data.extend_from_slice(&sequence.to_le_bytes());
    data.extend_from_slice(previous);
    data.extend_from_slice(created.as_bytes());
    data.push(0);
    data.extend_from_slice(table.as_bytes());
    data
}

/// Syncs the directory holding `path`, so a file created, renamed or removed there stays so.
///
/// # Errors
///
/// If the directory cannot be opened or synced.
pub fn sync_directory(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(directory) => File::open(directory)?.sync_all(),
        None => Ok(()),
    }
}

fn record_frame(record: &Record) -> Frame {
    let mut frame = Frame::header(Value::Null);
    frame.header = match record {
        Record::Insert(rows)
        | Record::Update(rows)
        | Record::Upsert(rows)
        | Record::Snapshot(rows) => {
            let kind = match record {
                Record::Insert(_) => "insert",
                Record::Update(_) => "update",
                Record::Upsert(_) => "upsert",
                _ => "snapshot",
            };
            let (target_buffer, target_type) = target_bytes(&rows.targets);
            let mut header = json!({
                "record": kind,
                "keys": frame.push(key_bytes(&rows.keys)),
                "features": frame.push(f32_bytes(&rows.features)),
                "targets": frame.push(target_buffer),
                "target_type": target_type,
            });
            if let Some(times) = &rows.times {
                let values: Vec<i64> = times.iter().map(|time| time.unwrap_or(0)).collect();
                header["times"] = json!(frame.push(i64_bytes(&values)));
                // Which rows have a time, when some do not.
                if times.iter().any(Option::is_none) {
                    let timed = times.iter().map(|time| u8::from(time.is_some())).collect();
                    header["timed"] = json!(frame.push(timed));
                }
            }
            if let Some(id) = &rows.id {
                header["id"] = json!(id);
            }
            if !rows.words.is_empty() {
                header["words"] = json!(rows.words);
            }
            if !rows.writes.is_empty() {
                header["writes"] = json!(rows.writes);
            }
            if let Some(layout) = &rows.layout {
                header["layout"] = row_layout_header(&mut frame, layout);
            }
            if let Some(table) = &rows.table {
                header["table_layout"] = table_layout_header(&mut frame, table);
            }
            header
        }
        Record::Delete { keys, id } => {
            let mut header = json!({"record": "delete", "keys": frame.push(key_bytes(keys))});
            if let Some(id) = id {
                header["id"] = json!(id);
            }
            header
        }
        Record::Refit => json!({"record": "refit"}),
        Record::Truncate => json!({"record": "truncate"}),
        Record::Term { term, primary } => {
            json!({"record": "term", "term": term, "primary": primary})
        }
    };
    frame
}

/// A log being rewritten as a compaction, off the shard that owns the log:
/// records sealed in order after the definition into a staged file.
pub struct Compactor {
    staged: PathBuf,
    number: u64,
    file: File,
    cipher: XChaCha20Poly1305,
    table: TableName,
    created: Option<CreationId>,
    offsets: Vec<u64>,
    previous: [u8; TAG],
    next: u64,
    at: u64,
    /// The log's length when the compaction began, which it must still have.
    begun: u64,
}

/// A compaction written and synced, for `Log::install_compaction`.
#[derive(Debug)]
pub struct Compacted {
    staged: PathBuf,
    offsets: Vec<u64>,
    previous: [u8; TAG],
    at: u64,
    begun: u64,
}

impl Compacted {
    /// Removes the staged file, for a compaction that will not be installed.
    pub fn discard(self) {
        let _ = std::fs::remove_file(&self.staged);
    }
}

impl Compactor {
    /// The process-wide number in this compaction's staged file name.
    #[must_use]
    pub fn number(&self) -> u64 {
        self.number
    }

    /// Seals the compaction's next record into the staged file.
    ///
    /// # Errors
    ///
    /// If the record is too large or cannot be written.
    pub fn append(&mut self, record: &Record) -> Result<(), LogError> {
        let sequence = self.next;
        self.seal(&record_frame(record), sequence)?;
        self.next += 1;
        Ok(())
    }

    /// Syncs the staged file, ready to be put in the log's place.
    ///
    /// # Errors
    ///
    /// If the file cannot be synced.
    pub fn finish(mut self) -> Result<Compacted, LogError> {
        self.file.sync_all()?;
        Ok(Compacted {
            staged: std::mem::take(&mut self.staged),
            offsets: std::mem::take(&mut self.offsets),
            previous: self.previous,
            at: self.at,
            begun: self.begun,
        })
    }

    fn seal(&mut self, frame: &Frame, sequence: u64) -> Result<(), LogError> {
        let created = match &self.created {
            Some(created) if sequence != 0 => created.as_str(),
            _ => "",
        };
        let bytes = seal_entry(
            &self.cipher,
            &self.table,
            created,
            &frame.to_bytes(),
            sequence,
            &self.previous,
        )?;
        self.file.write_all(&bytes)?;
        self.previous = tag_of(&bytes);
        let end = self.offsets.last().copied().unwrap_or_default() + bytes.len() as u64;
        self.offsets.push(end);
        Ok(())
    }
}

impl Drop for Compactor {
    fn drop(&mut self) {
        if !self.staged.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.staged);
        }
    }
}

impl fmt::Debug for Compactor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Compactor({})", self.staged.display())
    }
}

/// An entry sealed for a log: its length twice, then the nonce and ciphertext.
fn seal_entry(
    cipher: &XChaCha20Poly1305,
    table: &str,
    created: &str,
    plain: &[u8],
    sequence: u64,
    previous: &[u8; TAG],
) -> Result<Vec<u8>, LogError> {
    let mut nonce = [0u8; NONCE];
    getrandom::fill(&mut nonce).map_err(|error| LogError::Malformed(error.to_string()))?;
    let ciphertext = cipher
        .encrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: plain,
                aad: &associated(table, created, sequence, previous),
            },
        )
        .map_err(|_| LogError::Malformed("encryption failed".to_string()))?;
    let length = u32::try_from(NONCE + ciphertext.len())
        .map_err(|_| LogError::Malformed("record too large".to_string()))?;
    let mut bytes = Vec::with_capacity(PREFIX + length as usize);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&(!length).to_le_bytes());
    bytes.extend_from_slice(&nonce);
    bytes.extend_from_slice(&ciphertext);
    Ok(bytes)
}

/// Seals a table's row blocks under the log key, as its log is sealed: each
/// bound to the table, its creation, and the block's place and version, so a
/// block from another table, an earlier table of the same name, another place
/// or an older write does not open.
pub struct RowSealer {
    cipher: XChaCha20Poly1305,
    binding: Vec<u8>,
}

impl RowSealer {
    /// A sealer for the rows of `table` as created at `creation`.
    #[must_use]
    pub fn new(key: &EncryptionKey, table: &TableName, creation: &str) -> Self {
        let mut binding = b"gf rows\0".to_vec();
        binding.extend_from_slice(table.as_str().as_bytes());
        binding.push(0);
        binding.extend_from_slice(creation.as_bytes());
        binding.push(0);
        Self {
            cipher: XChaCha20Poly1305::new(&key.0),
            binding,
        }
    }

    fn associated(&self, block: u32, version: u32) -> Vec<u8> {
        let mut associated = self.binding.clone();
        associated.extend_from_slice(&block.to_le_bytes());
        associated.extend_from_slice(&version.to_le_bytes());
        associated
    }
}

impl fmt::Debug for RowSealer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RowSealer")
    }
}

impl Sealer for RowSealer {
    fn seal(&self, block: u32, version: u32, plain: &[u8]) -> Vec<u8> {
        let mut nonce = [0u8; NONCE];
        getrandom::fill(&mut nonce).expect("the system has randomness");
        let sealed = self
            .cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: plain,
                    aad: &self.associated(block, version),
                },
            )
            .expect("sealing a block in memory does not fail");
        [nonce.as_slice(), &sealed].concat()
    }

    fn open(&self, block: u32, version: u32, sealed: &[u8]) -> Option<Vec<u8>> {
        let (nonce, sealed) = sealed.split_at_checked(NONCE)?;
        self.cipher
            .decrypt(
                XNonce::from_slice(nonce),
                Payload {
                    msg: sealed,
                    aad: &self.associated(block, version),
                },
            )
            .ok()
    }
}

/// A compaction record's row layout, as frame buffers.
fn row_layout_header(frame: &mut Frame, layout: &RowLayout) -> Value {
    let ids: Vec<u32> = layout.ids.iter().map(|row| row.0).collect();
    let pages: Vec<u32> = layout.pages.iter().map(|page| page.0).collect();
    let new: Vec<u32> = layout.new_pages.iter().map(|page| page.id.0).collect();
    let buckets: Vec<u64> = layout.new_pages.iter().map(|page| page.bucket.0).collect();
    let sums: Vec<f64> = layout
        .new_pages
        .iter()
        .flat_map(|page| page.sum.iter().copied())
        .collect();
    let radii: Vec<f64> = layout.new_pages.iter().map(|page| page.radius).collect();
    json!({
        "ids": frame.push(u32_bytes(&ids)),
        "pages": frame.push(u32_bytes(&pages)),
        "new_pages": frame.push(u32_bytes(&new)),
        "buckets": frame.push(u64_bytes(&buckets)),
        "sums": frame.push(f64_bytes(&sums)),
        "radii": frame.push(f64_bytes(&radii)),
    })
}

/// A compaction record's row layout, from frame buffers.
fn row_layout_from(frame: &Frame, header: &Value) -> Result<RowLayout, LogError> {
    let new = u32_values(frame.buffer(header, "new_pages")?)?;
    let buckets = u64_values(frame.buffer(header, "buckets")?)?;
    let radii = f64_values(frame.buffer(header, "radii")?)?;
    let sums = f64_values(frame.buffer(header, "sums")?)?;
    if buckets.len() != new.len() || radii.len() != new.len() {
        return Err(malformed(
            "a layout's pages and their buckets or radii differ in number",
        ));
    }
    let width = per(sums.len(), new.len(), "a layout's page sums")?;
    let new_pages = new
        .iter()
        .zip(&buckets)
        .zip(&radii)
        .enumerate()
        .map(|(index, ((&id, &bucket), &radius))| PageLayout {
            id: PageId(id),
            bucket: Bucket(bucket),
            sum: sums[index * width..(index + 1) * width].to_vec(),
            radius,
        })
        .collect();
    Ok(RowLayout {
        ids: u32_values(frame.buffer(header, "ids")?)?
            .into_iter()
            .map(RowId)
            .collect(),
        pages: u32_values(frame.buffer(header, "pages")?)?
            .into_iter()
            .map(PageId)
            .collect(),
        new_pages,
    })
}

/// A compaction's table layout, as frame buffers.
fn table_layout_header(frame: &mut Frame, table: &TableLayout) -> Value {
    let cells = &table.index.cells;
    let centers: Vec<f64> = cells
        .iter()
        .flat_map(|cell| cell.center.iter().copied())
        .collect();
    let radii: Vec<f64> = cells.iter().map(|cell| cell.radius).collect();
    let reaches: Vec<f64> = cells.iter().map(|cell| cell.reach).collect();
    let groups: Vec<u32> = cells.iter().map(|cell| cell.pages.len() as u32).collect();
    let buckets: Vec<u64> = cells
        .iter()
        .flat_map(|cell| cell.pages.iter().map(|(bucket, _)| bucket.0))
        .collect();
    let sizes: Vec<u32> = cells
        .iter()
        .flat_map(|cell| cell.pages.iter().map(|(_, pages)| pages.len() as u32))
        .collect();
    let pages: Vec<u32> = cells
        .iter()
        .flat_map(|cell| {
            cell.pages
                .iter()
                .flat_map(|(_, pages)| pages.iter().map(|page| page.0))
        })
        .collect();
    let scaling = table.scaling.as_ref().map_or(Value::Null, |scaling| {
        let lookups: Vec<Value> = scaling
            .lookups
            .iter()
            .map(|(codes, vectors)| {
                json!({
                    "codes": frame.push(f32_bytes(codes)),
                    "vectors": frame.push(f32_bytes(vectors)),
                })
            })
            .collect();
        json!({
            "center": frame.push(f32_bytes(&scaling.center)),
            "scale": frame.push(f32_bytes(&scaling.scale)),
            "lookups": lookups,
            "rows": scaling.rows,
        })
    });
    let fitted = table.fitted.as_ref().map_or(Value::Null, |fitted| {
        json!({
            "rows": fitted.rows,
            "means": frame.push(f64_bytes(&fitted.means)),
            "deviations": frame.push(f64_bytes(&fitted.deviations)),
        })
    });
    json!({
        "rows": table.rows,
        "ids": table.ids,
        "next_page": table.index.next_page,
        "changed": table.changed,
        "reach": frame.push(f64_bytes(&[table.index.reach, table.index.built_reach])),
        "cells": {
            "centers": frame.push(f64_bytes(&centers)),
            "radii": frame.push(f64_bytes(&radii)),
            "reaches": frame.push(f64_bytes(&reaches)),
            "groups": frame.push(u32_bytes(&groups)),
            "buckets": frame.push(u64_bytes(&buckets)),
            "sizes": frame.push(u32_bytes(&sizes)),
            "pages": frame.push(u32_bytes(&pages)),
        },
        "scaling": scaling,
        "column_sums": frame.push(f64_bytes(&table.column_sums)),
        "column_counts": frame.push(u64_bytes(&table.column_counts)),
        "fitted": fitted,
    })
}

/// A compaction's table layout, from frame buffers.
fn table_layout_from(frame: &Frame, header: &Value) -> Result<TableLayout, LogError> {
    let number = |field: &str| {
        header
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed(&format!("a table layout needs {field}")))
    };
    let reach = f64_values(frame.buffer(header, "reach")?)?;
    let [reach, built_reach] = reach[..] else {
        return Err(malformed("a table layout's reach is two numbers"));
    };
    let cells = header.get("cells").unwrap_or(&Value::Null);
    let radii = f64_values(frame.buffer(cells, "radii")?)?;
    let reaches = f64_values(frame.buffer(cells, "reaches")?)?;
    let groups = u32_values(frame.buffer(cells, "groups")?)?;
    let buckets = u64_values(frame.buffer(cells, "buckets")?)?;
    let sizes = u32_values(frame.buffer(cells, "sizes")?)?;
    let pages = u32_values(frame.buffer(cells, "pages")?)?;
    let centers = f64_values(frame.buffer(cells, "centers")?)?;
    let count = radii.len();
    if reaches.len() != count
        || groups.len() != count
        || sizes.len() != buckets.len()
        || groups.iter().map(|&groups| groups as usize).sum::<usize>() != buckets.len()
        || sizes.iter().map(|&size| size as usize).sum::<usize>() != pages.len()
    {
        return Err(malformed("a table layout's cells do not add up"));
    }
    let width = per(centers.len(), count, "a table layout's cell centers")?;
    let (mut group, mut page) = (0, 0);
    let mut laid = Vec::with_capacity(count);
    for index in 0..count {
        let mut members = Vec::new();
        for _ in 0..groups[index] {
            let size = sizes[group] as usize;
            let ids = pages[page..page + size]
                .iter()
                .map(|&id| PageId(id))
                .collect();
            members.push((Bucket(buckets[group]), ids));
            group += 1;
            page += size;
        }
        laid.push(CellLayout {
            center: centers[index * width..(index + 1) * width].to_vec(),
            radius: radii[index],
            reach: reaches[index],
            pages: members,
        });
    }
    let scaling = match header.get("scaling") {
        Some(scaling) if !scaling.is_null() => Some(Scaling {
            center: f32_values(frame.buffer(scaling, "center")?)?,
            scale: f32_values(frame.buffer(scaling, "scale")?)?,
            lookups: scaling
                .get("lookups")
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .map(|lookup| {
                    Ok((
                        f32_values(frame.buffer(lookup, "codes")?)?,
                        f32_values(frame.buffer(lookup, "vectors")?)?,
                    ))
                })
                .collect::<Result<_, LogError>>()?,
            rows: scaling
                .get("rows")
                .and_then(Value::as_u64)
                .ok_or_else(|| malformed("a scaling needs its rows"))?,
        }),
        _ => None,
    };
    let fitted = match header.get("fitted") {
        Some(fitted) if !fitted.is_null() => Some(FittedColumns {
            rows: fitted
                .get("rows")
                .and_then(Value::as_u64)
                .ok_or_else(|| malformed("fitted columns need their rows"))?,
            means: f64_values(frame.buffer(fitted, "means")?)?,
            deviations: f64_values(frame.buffer(fitted, "deviations")?)?,
        }),
        _ => None,
    };
    Ok(TableLayout {
        rows: number("rows")?,
        ids: u32::try_from(number("ids")?).map_err(|_| malformed("too many row ids"))?,
        index: IndexLayout {
            cells: laid,
            next_page: u32::try_from(number("next_page")?)
                .map_err(|_| malformed("too many pages"))?,
            reach,
            built_reach,
        },
        scaling,
        column_sums: f64_values(frame.buffer(header, "column_sums")?)?,
        column_counts: u64_values(frame.buffer(header, "column_counts")?)?,
        fitted,
        changed: number("changed")?,
    })
}

/// How many values each of `count` items has, where `total` values hold them all.
fn per(total: usize, count: usize, what: &str) -> Result<usize, LogError> {
    match count {
        0 if total == 0 => Ok(0),
        0 => Err(malformed(&format!("{what} hold values for nothing"))),
        count if total.is_multiple_of(count) => Ok(total / count),
        _ => Err(malformed(&format!("{what} do not divide evenly"))),
    }
}

fn malformed(why: &str) -> LogError {
    LogError::Malformed(why.to_string())
}

/// A record's row times, each None where its `timed` mask says the row has none.
fn times_from(frame: &Frame, header: &Value) -> Result<Vec<Option<i64>>, LogError> {
    let values = i64_values(frame.buffer(header, "times")?)?;
    let Some(_) = header.get("timed") else {
        return Ok(values.into_iter().map(Some).collect());
    };
    let timed = frame.buffer(header, "timed")?;
    if timed.len() != values.len() || timed.iter().any(|&flag| flag > 1) {
        return Err(LogError::Malformed(
            "a record's time mask does not match its times".to_string(),
        ));
    }
    Ok(values
        .into_iter()
        .zip(timed)
        .map(|(time, &flag)| (flag == 1).then_some(time))
        .collect())
}

fn record_from(frame: &Frame) -> Result<Record, LogError> {
    let header = &frame.header;
    let kind = header
        .get("record")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let id = header.get("id").and_then(Value::as_str).map(str::to_string);
    let keys = || -> Result<Vec<u128>, LogError> { Ok(key_values(frame.buffer(header, "keys")?)?) };
    let rows = || -> Result<Rows, LogError> {
        let target_type = header
            .get("target_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let times = match header.get("times") {
            Some(_) => Some(times_from(frame, header)?),
            None => None,
        };
        Ok(Rows {
            keys: keys()?,
            features: f32_values(frame.buffer(header, "features")?)?,
            targets: targets_from(frame.buffer(header, "targets")?, target_type)?,
            times,
            id: id.clone(),
            words: match header.get("words") {
                Some(words) => serde_json::from_value(words.clone())
                    .map_err(|error| LogError::Malformed(error.to_string()))?,
                None => Vec::new(),
            },
            writes: match header.get("writes") {
                Some(writes) => serde_json::from_value(writes.clone())
                    .map_err(|error| LogError::Malformed(error.to_string()))?,
                None => Vec::new(),
            },
            layout: header
                .get("layout")
                .map(|layout| row_layout_from(frame, layout))
                .transpose()?,
            table: header
                .get("table_layout")
                .map(|table| table_layout_from(frame, table).map(Box::new))
                .transpose()?,
        })
    };
    Ok(match kind {
        "insert" => Record::Insert(rows()?),
        "update" => Record::Update(rows()?),
        "upsert" => Record::Upsert(rows()?),
        "snapshot" => Record::Snapshot(rows()?),
        "delete" => Record::Delete { keys: keys()?, id },
        "refit" => Record::Refit,
        "truncate" => Record::Truncate,
        "term" => Record::Term {
            term: header
                .get("term")
                .and_then(Value::as_u64)
                .ok_or_else(|| LogError::Malformed("a term record needs a term".to_string()))?,
            primary: header
                .get("primary")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        _ => return Err(LogError::Malformed(format!("unknown record {kind:?}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup;
    use gf_core::columns::ColumnKind;

    /// Writes a log whose table names `strategy`, as an earlier server could.
    fn log_with_strategy(path: &Path, key: &EncryptionKey, strategy: Value) {
        let definition = TableDefinition {
            name: TableName::new("payments").unwrap(),
            model: "general-forecast/tabicl-v2-classifier".to_string(),
            columns: vec![Column {
                name: "amount".to_string(),
                kind: ColumnKind::Numeric,
            }],
            classes: 2,
            strategy: StrategySpec::Exact,
            page_size: 64,
            time_bucket: None,
            refit_after: None,
            refit_shift: None,
            fit_rows: None,
            past_cap: PastCapSpec::Warn,
            regions: None,
            series: None,
            refit_cutoffs: "each".to_string(),
            retrieval_categoricals: "codes".to_string(),
            scale_rows: None,
            summary: SummarySpec::Table,
            created: CreationId::parse("0123456789abcdef"),
            times: TimeKind::Integer,
            route_by: None,
            durability: Default::default(),
        };
        let log = Log::create(path, key, &definition, Terms::default()).unwrap();
        let mut header = serde_json::to_value(&definition).unwrap();
        header["strategy"] = strategy;
        let frame = Frame::header(
            json!({"record": "table", "definition": header, "base": 1, "terms": Terms::default()}),
        );
        let sealed = log.seal_after(&frame, 0, &[0; TAG]).unwrap();
        drop(log);
        std::fs::write(path, [file_header(key).as_slice(), &sealed].concat()).unwrap();
    }

    #[test]
    fn a_table_with_a_retired_strategy_is_not_served_and_keeps_its_log() {
        let directory = std::env::temp_dir().join(format!("gf-retired-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("payments.log");
        let key = EncryptionKey::from_hex(&"07".repeat(32)).unwrap();
        log_with_strategy(
            &path,
            &key,
            json!({"kind": "pages", "budget": 4096, "stitch": "joint"}),
        );
        let log = std::fs::read(&path).unwrap();
        let head = std::fs::read(head_path(&path)).unwrap();
        let keys = Keys::only(key);

        let error = Log::open(&path, &keys).err().unwrap();
        assert!(matches!(&error, LogError::RetiredStrategy(kind) if kind == "pages"));
        assert!(!error.damaged());
        assert!(error.to_string().contains("retired pages strategy"));
        // The server starts without the table and leaves its log as it was, so
        // the name stays taken.
        assert!(setup::open_log(&path, &keys).unwrap().is_none());
        assert_eq!(std::fs::read(&path).unwrap(), log);
        assert_eq!(std::fs::read(head_path(&path)).unwrap(), head);
        assert!(!directory.join("payments.log.unreadable").exists());
        std::fs::remove_dir_all(&directory).unwrap();
    }

    fn route(kind: RouteKind, bounds: &[f64]) -> RouteBy {
        RouteBy {
            column: "amount".to_string(),
            kind,
            bounds: bounds.to_vec(),
        }
    }

    #[test]
    fn a_hashed_route_spreads_values_over_partitions_and_never_moves() {
        let route = route(RouteKind::Hashed, &[]);
        let assignment = route.assignment(4);
        let mut counts = [0usize; 4];
        for value in 0..4_000 {
            let partition = assignment.slice_of(route.key(f64::from(value)));
            counts[partition] += 1;
            assert_eq!(partition, assignment.slice_of(route.key(f64::from(value))));
        }
        assert!(
            counts.iter().all(|&count| (800..1_200).contains(&count)),
            "{counts:?}"
        );
    }

    #[test]
    fn an_ordered_route_sends_each_value_past_its_bounds() {
        let route = route(RouteKind::Ordered, &[-1.0, 5.0]);
        let assignment = route.assignment(3);
        assert_eq!(assignment.slices.len(), 3);
        let partitions: Vec<usize> = [-1e9, -1.5, -1.0, 0.0, 4.9, 5.0, 7e12]
            .into_iter()
            .map(|value| assignment.slice_of(route.key(value)))
            .collect();
        assert_eq!(partitions, [0, 0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn a_route_keys_negative_zero_as_zero() {
        for kind in [RouteKind::Hashed, RouteKind::Ordered] {
            let route = route(kind, &[]);
            assert_eq!(route.key(-0.0), route.key(0.0));
            assert_ne!(route.key(-1.0), route.key(1.0));
        }
        // A bound of zero puts negative zero past it, with zero.
        let route = route(RouteKind::Ordered, &[0.0]);
        let assignment = route.assignment(2);
        assert_eq!(assignment.slice_of(route.key(-0.0)), 1);
        assert_eq!(assignment.slice_of(route.key(-1e-300)), 0);
    }

    #[test]
    fn a_route_names_a_numeric_column_and_bounds_in_order() {
        let columns = vec![
            Column {
                name: "amount".to_string(),
                kind: ColumnKind::Numeric,
            },
            Column {
                name: "merchant".to_string(),
                kind: ColumnKind::Categorical,
            },
        ];
        assert_eq!(route(RouteKind::Hashed, &[]).check(&columns), Ok(()));
        assert_eq!(
            route(RouteKind::Ordered, &[0.0, 1.0]).check(&columns),
            Ok(())
        );
        assert!(route(RouteKind::Hashed, &[1.0]).check(&columns).is_err());
        assert!(
            route(RouteKind::Ordered, &[1.0, 1.0])
                .check(&columns)
                .is_err()
        );
        assert!(
            route(RouteKind::Ordered, &[f64::INFINITY])
                .check(&columns)
                .is_err()
        );
        let mut other = route(RouteKind::Hashed, &[]);
        other.column = "merchant".to_string();
        assert!(other.check(&columns).is_err());
        other.column = "missing".to_string();
        assert!(other.check(&columns).is_err());
        // A definition without the field reads as a table without a route.
        let plain: TableDefinition = serde_json::from_value(json!({
            "name": "payments", "model": "tiny", "columns": [], "classes": 0
        }))
        .unwrap();
        assert_eq!(plain.route_by, None);
    }
}
