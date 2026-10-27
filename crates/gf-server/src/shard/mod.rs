// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A shard: one worker, its node, and the logs of the tables placed on it.
//!
//! A shard runs as one task. It takes every queued command, logging each
//! write before applying it, and plans pending predictions into steps for the
//! worker. Up to two steps are in flight: the shard plans and sends the next
//! while the worker computes the current one, then routes each step's
//! predictions back to its requests.
//!
//! Writes that arrive together are committed together: each is checked and
//! staged in its table's log, the logs are synced once, and then each is
//! applied, so a burst of small writes costs one sync per table rather than
//! one each. Writes to a key an earlier write in the batch touched wait for the
//! next batch, so every check sees the rows it would have seen alone. A write
//! that carries a client's id applies once, however often it is sent. The
//! strings of categorical columns take their codes from the table's
//! vocabulary as the write is logged, and new words go into the record, so a
//! second write with new words for the table waits for the next batch.
//!
//! Replicated records were checked by their primary, so a shipment is staged,
//! synced, then applied in order, and a replica's log and rows match its
//! primary's. A log that outgrows twice its live rows is compacted on its own.
//!
//! If the worker stops, or takes longer than the step timeout to answer a
//! step, the shard fails the steps in flight, starts it again, and adds its
//! tables back; their caches and fits start over. While it starts, writes and
//! other log work go on, and predictions wait for it.
//!
//! A prediction whose caller has gone, such as one that timed out, leaves the
//! queue before the next step is planned. Each log's head is saved when its
//! writes go idle, and again when the shard stops.
//!
//! The worker reports which tables' parts of a step failed, so one table's
//! error fails only its own requests. A table that fails three steps running
//! is quarantined: its queries are refused until a write or refit changes it.
//!
//! A table whose log refuses writes, or holds a synced record its rows could
//! not take, is out of service until the server restarts: log and rows no
//! longer agree, so it refuses writes, queries and shipments, and a restart
//! replays the log or sets it aside. Dropping it still works.
//! A step that fails as a whole, because the worker stopped or hung, counts
//! against the tables it fit, since a fit is the likeliest cause.

use std::collections::{HashMap, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::time::Instant;

use gf_core::cache::{Eviction, ModelId, Movement, Tier, TierConfig};
use gf_core::forecasting::{self, HORIZON_PASSES, Neighbors, SeriesSpec, horizon_limit};
use gf_core::node::{
    CompactionSource, EntrySizes, Node, NodeError, RequestId, Scheduling, SummaryScope, TableId,
    TableRows, TableSpec, Task,
};
use gf_core::planner::Strategy;
use gf_core::rows::RowMemory;
use gf_core::table::RowId;
use std::collections::VecDeque;

use gf_core::node::{Action, Step, StepUnit, Targets, UNLABELED};
use gf_core::regions::RegionStats;
use tokio::sync::{mpsc, oneshot};

use crate::budget::Charge;
use crate::definition::{CreationId, Durability, TableDefinition};
use crate::ephemeral::{Ephemeral, EphemeralTables};
use crate::error::ServerError;
use crate::metrics::{Metrics, TableMetrics, Timing};
use crate::name::TableName;
use crate::partition::{self, Given, ModelRows, Offer};
use crate::protocol::{
    ModelProfile, Outcome, Predictions, StrategySpec, WorkerReply, WorkerRequest, movements_json,
    outcome_from, step_frame,
};
use crate::routes::Routes;
use crate::wal::{
    self, Compacted, Compactor, EncryptionKey, Log, LogEntry, PastWrite, Record, Replay, RowSealer,
    Rows, SeriesKey, Terms, decode,
};
use crate::worker::{self, WorkerConnection};
use gf_core::columns::{self, Strings, Vocabulary, Words};

mod commit;
mod compaction;
mod models;
mod replication;
mod restart;
mod steps;

fn profile_steps(steps: u64, what: &str) -> Result<NonZeroUsize, ServerError> {
    usize::try_from(steps)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| ServerError::Internal(format!("the forecaster's profile has no {what}")))
}

/// The request that has a worker serve a table.
fn add_table(id: TableId, definition: &TableDefinition) -> WorkerRequest {
    WorkerRequest::AddTable {
        table: id.0,
        model: definition.model.clone(),
        classes: definition.classes,
        features: definition.model_features(),
        categorical: columns::categorical(&definition.columns),
        page_size: definition.page_size,
        summary: definition.summary,
        series: definition.series.is_some(),
    }
}

/// Checks a forecaster's quantile levels: one per output column, increasing,
/// strictly between 0 and 1.
fn check_quantiles(quantiles: Option<&[f64]>, outputs: u64) -> Result<(), ServerError> {
    let Some(levels) = quantiles else {
        return Err(ServerError::Internal(
            "the forecaster's profile names no quantile levels".to_string(),
        ));
    };
    let increasing = levels.windows(2).all(|pair| pair[0] < pair[1]);
    let inside = levels.iter().all(|&level| level > 0.0 && level < 1.0);
    if levels.len() as u64 != outputs || !increasing || !inside {
        return Err(ServerError::Internal(format!(
            "the forecaster's profile lists {} quantile levels for {outputs} outputs; they \
             must be one per output, increasing and strictly between 0 and 1",
            levels.len()
        )));
    }
    Ok(())
}

/// The levels of a tabular regressor's evenly spaced quantile columns.
fn implied_levels(columns: u64) -> Vec<f64> {
    (0..columns)
        .map(|column| (column + 1) as f64 / (columns + 1) as f64)
        .collect()
}

/// Steps a shard keeps in flight: one computing, and the next ready behind it.
const PIPELINE_DEPTH: usize = 2;

/// Failed steps in a row that quarantine a table.
const FAILURES: u32 = 3;

/// Write ids remembered per table, for retries to find.
const RECENT_WRITES: usize = 4096;

/// Records past the last compaction before a log may be compacted on its own.
const COMPACT_RECORDS: u64 = 1024;

/// How many times a table's live rows a log may take before it is compacted.
const COMPACT_FACTOR: u64 = 2;

/// The most feature columns a table may have.
const MAX_FEATURES: usize = 10_000;

/// A step the worker has been sent, when it was planned and sent, and where
/// its reply will arrive.
struct InFlight {
    step: Step,
    planned: Instant,
    sent: Instant,
    reply: oneshot::Receiver<worker::Reply>,
}

/// A table the worker was told to serve or stop serving without the shard
/// waiting: the worker answers frames in turn, so a step sent after it finds
/// the table added, and its reply comes before that step's.
enum Control {
    Add(TableId),
    Remove(TableId),
}

/// When a step was planned, sent, started once the step before it finished,
/// and finished.
#[derive(Clone, Copy)]
struct StepTimes {
    planned: Instant,
    sent: Instant,
    started: Instant,
    finished: Instant,
}

/// A request's share of its step: the context rows it newly encoded, and
/// its part of the worker's time.
type Share = (f64, f64);

/// A queued prediction request: its reply, trace id, table, query bytes, and
/// when it arrived.
struct Pending {
    reply: Reply<Predictions>,
    trace: String,
    table: TableId,
    bytes: u64,
    arrived: Instant,
}

/// A query a table took, before its caller waits on it: its id, table and
/// query bytes, which `queue_pending` holds against the queue.
struct Submitted {
    request: RequestId,
    table: TableId,
    bytes: u64,
}

/// A change to a table's rows, or to its fitted state.
#[derive(Clone, Debug)]
pub enum Write {
    Insert(Incoming),
    Update(Incoming),
    Upsert(Incoming),
    /// Rows' targets, without features: an update keeping their features
    /// and times.
    Label(Incoming),
    /// With `missing_ok`, keys the table does not hold are skipped.
    Delete {
        keys: Vec<u128>,
        missing_ok: bool,
        /// The client's id for the write, so a retry applies once.
        id: Option<String>,
    },
    /// Refits the table's state and retrieval scaling on its live rows.
    Refit,
    /// Deletes every row.
    Truncate,
    /// Changes how the table chooses contexts from here on.
    Strategy(StrategySpec),
}

impl Write {
    /// The client's id for the write, if it sent one.
    fn write_id(&self) -> Option<&str> {
        match self {
            Self::Insert(incoming)
            | Self::Update(incoming)
            | Self::Upsert(incoming)
            | Self::Label(incoming) => incoming.rows.id.as_deref(),
            Self::Delete { id, .. } => id.as_deref(),
            Self::Refit | Self::Truncate | Self::Strategy(_) => None,
        }
    }

    /// The keys the write touches, before the table has checked them.
    fn keys(&self) -> &[u128] {
        match self {
            Self::Insert(incoming)
            | Self::Update(incoming)
            | Self::Upsert(incoming)
            | Self::Label(incoming) => &incoming.rows.keys,
            Self::Delete { keys, .. } => keys,
            Self::Refit | Self::Truncate | Self::Strategy(_) => &[],
        }
    }

    /// Whether the write carries targets: rows, rather than keys alone.
    fn has_targets(&self) -> bool {
        matches!(
            self,
            Self::Insert(_) | Self::Update(_) | Self::Upsert(_) | Self::Label(_)
        )
    }

    /// Whether the write holds strings for the table's vocabularies to code.
    fn has_strings(&self) -> bool {
        match self {
            Self::Insert(incoming)
            | Self::Update(incoming)
            | Self::Upsert(incoming)
            | Self::Label(incoming) => !incoming.strings.is_empty(),
            Self::Delete { .. } | Self::Refit | Self::Truncate | Self::Strategy(_) => false,
        }
    }
}

/// Rows as a client sent them: numbers in place, and the strings of
/// categorical columns apart, for the table's vocabularies to code.
#[derive(Clone, Debug, PartialEq)]
pub struct Incoming {
    pub rows: Rows,
    pub strings: Strings,
}

impl From<Rows> for Incoming {
    fn from(rows: Rows) -> Self {
        Self {
            rows,
            strings: Vec::new(),
        }
    }
}

type Reply<T> = oneshot::Sender<Result<T, ServerError>>;

/// Where a shipment came from: the source copy's terms and creation id, if
/// the router gave them.
#[derive(Clone, Debug, Default)]
pub struct Lineage {
    pub terms: Option<Terms>,
    pub created: Option<CreationId>,
}

/// One table's log length, terms and creation id, as `LogLengths` reports them.
#[derive(Clone, Debug)]
pub struct LogLength {
    pub table: TableName,
    pub length: u64,
    pub terms: Terms,
    pub created: Option<CreationId>,
}

/// A client write waiting to be committed with the others that arrived with
/// it, holding its share of the request budget until it is answered.
struct Queued {
    table: TableName,
    write: Write,
    fence: Fence,
    charge: Charge,
    reply: Reply<Written>,
}

/// A write staged in its log, to apply once the log is synced.
struct Staged {
    table: TableName,
    id: TableId,
    record: Record,
    log_length: u64,
    charge: Charge,
    reply: Reply<Written>,
}

/// The last writes a table applied, by the client's id, for retries to find.
#[derive(Default)]
struct Recent {
    order: VecDeque<String>,
    written: HashMap<String, Written>,
}

impl Recent {
    fn remember(&mut self, id: String, written: Written) {
        if self.written.insert(id.clone(), written).is_none() {
            self.order.push_back(id);
            if self.order.len() > RECENT_WRITES
                && let Some(oldest) = self.order.pop_front()
            {
                self.written.remove(&oldest);
            }
        }
    }
}

/// This node's write lease: the router that holds it, and when it ends.
///
/// The router that fails tables over renews it, so a node cut off from that
/// router stops taking writes before another copy is promoted. A lease is
/// renewed only for its holder until it ends, so no two routers keep one.
#[derive(Clone, Debug)]
pub struct Lease(Arc<Mutex<Held>>);

#[derive(Debug)]
struct Held {
    holder: Option<String>,
    until: Instant,
}

impl Lease {
    /// A lease that has lapsed until a router takes it.
    #[must_use]
    pub fn lapsed() -> Self {
        Self(Arc::new(Mutex::new(Held {
            holder: None,
            until: Instant::now(),
        })))
    }

    /// Lets the node take writes for `duration` from now, for `holder`.
    ///
    /// # Errors
    ///
    /// The holder of a lease that has not ended, if it is another router.
    pub fn renew(&self, holder: &str, duration: Duration) -> Result<(), String> {
        let mut held = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let now = Instant::now();
        if let Some(current) = Self::other(&held, holder, now) {
            return Err(current);
        }
        *held = Held {
            holder: Some(holder.to_string()),
            until: now + duration,
        };
        Ok(())
    }

    /// Another router holding a lease that has not ended, if one does; a
    /// renewal for `holder` would be refused for it.
    #[must_use]
    pub fn held_by_another(&self, holder: &str) -> Option<String> {
        let held = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        Self::other(&held, holder, Instant::now())
    }

    fn other(held: &Held, holder: &str, now: Instant) -> Option<String> {
        held.holder
            .as_ref()
            .filter(|current| *current != holder && now < held.until)
            .cloned()
    }

    /// Refuses writes once the lease has lapsed.
    ///
    /// # Errors
    ///
    /// `Fenced` if the lease has lapsed.
    pub fn check(&self) -> Result<(), ServerError> {
        if Instant::now() < self.0.lock().unwrap_or_else(PoisonError::into_inner).until {
            Ok(())
        } else {
            Err(ServerError::Fenced(
                "this node's write lease has lapsed; writes resume once a router renews it"
                    .to_string(),
            ))
        }
    }
}

/// What a client write must still hold when it is logged, however long it
/// waited behind other commands.
#[derive(Clone, Debug, Default)]
pub struct Fence {
    /// This node's write lease, if writes need one.
    pub lease: Option<Lease>,
    /// The term the router sent the write under, if it named one.
    pub term: Option<u64>,
}

impl Fence {
    /// What replicated records and promotions hold: nothing.
    const NONE: Self = Self {
        lease: None,
        term: None,
    };

    fn check(&self, table: &str, log: &Log) -> Result<(), ServerError> {
        if let Some(lease) = &self.lease {
            lease.check()?;
            // A router that has not learned the table's term, as after it
            // restarts, could be writing to a primary since replaced.
            if self.term.is_none() {
                return Err(ServerError::Fenced(format!(
                    "a write to table {table:?} names no term; this node takes writes \
                     only from a router that knows the table's term"
                )));
            }
        }
        let current = log.term().term;
        match self.term {
            Some(term) if term != current => Err(ServerError::Fenced(format!(
                "table {table:?} is at term {current}, but the write was sent for term {term}"
            ))),
            _ => Ok(()),
        }
    }
}

/// What a write changed, and the table's log length after it.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Written {
    pub rows: usize,
    pub log_length: u64,
}

/// A table's definition, live rows, log length, and prediction width.
#[derive(Clone, Debug)]
pub struct Description {
    pub definition: TableDefinition,
    pub rows: u64,
    pub log_length: u64,
    /// Columns in each prediction: the table's classes, or the model's quantiles.
    pub outputs: u64,
    /// The quantile level of each prediction column, for a table predicting
    /// values: a forecaster's own levels, or a tabular regressor's evenly
    /// spaced ones.
    pub levels: Option<Vec<f64>>,
    /// The strategy the table's next plan chooses contexts with: its
    /// adaptive strategy resolved against its live rows, or the one it was given.
    pub strategy: StrategySpec,
    /// The most context rows the model was trained on, if it says.
    pub context_rows: Option<u64>,
    /// The most rows the table's state is fit on: its `fit_rows`, or fewer
    /// where the worker's call budget takes fewer at its width; None for a
    /// table that fits each context's own, or holds series.
    pub effective_fit_rows: Option<u64>,
    /// What the table's regions served, if it has them.
    pub regions: Option<RegionStats>,
    /// Each column's words in code order; empty but for categorical columns.
    pub vocabularies: Vec<Vec<String>>,
    /// A classifier's labels in code order, if its targets are strings.
    pub labels: Vec<String>,
    /// How many series the table holds the keys of, for a table of series.
    pub series: Option<u64>,
}

/// What a shard is asked to do.
pub enum Command {
    /// Creates a table: a client's at term 0, or a replica's with its source's
    /// terms; answers the table's quantile levels, if it predicts values.
    Create {
        definition: TableDefinition,
        terms: Terms,
        reply: Reply<Option<Vec<f64>>>,
    },
    /// Rebuilds a table from its log at startup, one record at a time;
    /// answers its quantile levels, if it predicts values.
    Restore {
        replay: Box<Replay>,
        reply: Reply<Option<Vec<f64>>>,
    },
    Write {
        table: TableName,
        write: Write,
        fence: Fence,
        /// What the write holds of the server's request budget, given back
        /// once the shard has answered it.
        charge: Charge,
        reply: Reply<Written>,
    },
    Refit {
        table: TableName,
        fence: Fence,
        reply: Reply<Written>,
    },
    /// Changes how the table chooses contexts from here on, logged like a write.
    SetStrategy {
        table: TableName,
        strategy: StrategySpec,
        fence: Fence,
        reply: Reply<Written>,
    },
    Predict {
        table: TableName,
        features: Vec<f32>,
        /// The strings of categorical columns, which take the codes the table
        /// knows, or NaN.
        strings: Strings,
        /// Each query's time cutoff, if they have them.
        before: Option<Vec<i64>>,
        /// The keys of the rows each query reads, if they were chosen for it.
        given: Option<Vec<Vec<u128>>>,
        /// The request's trace id, for the log line of the step it runs in.
        trace: String,
        reply: Reply<Predictions>,
    },
    /// Forecasts series of a table of series over `horizon` steps, each from
    /// its `before` time, or the step after its latest observed value.
    Forecast {
        table: TableName,
        /// The series, as their keys' 64-bit digests.
        series: Vec<u64>,
        before: Option<Vec<i64>>,
        horizon: NonZeroUsize,
        /// Each series' group, for a table of given groups.
        groups: Option<Vec<Vec<u64>>>,
        trace: String,
        reply: Reply<Predictions>,
    },
    Describe {
        table: TableName,
        reply: Reply<Description>,
    },
    /// The keys of a table's series as clients gave them, in the order the
    /// table first saw them.
    KnownSeries {
        table: TableName,
        reply: Reply<Vec<SeriesKey>>,
    },
    /// A table's log entries from a sequence number, at most `limit` of them,
    /// with their sequence numbers.
    ReadLog {
        table: TableName,
        from: u64,
        limit: usize,
        /// The most sealed bytes past the first entry, as `Log::read` takes them.
        bytes: usize,
        reply: Reply<Vec<(u64, Vec<u8>)>>,
    },
    /// Applies log entries read from another node, with their sequence numbers.
    ///
    /// Entries the log already has are skipped, so shipping is idempotent. The
    /// lineage, if the router gave it, is the source copy's terms and creation
    /// id, which this copy's must match.
    Replicate {
        table: TableName,
        entries: Vec<(u64, Vec<u8>)>,
        lineage: Box<Lineage>,
        reply: Reply<u64>,
    },
    /// Rewrites a table's log as a snapshot of its live rows, so changed and
    /// deleted rows' old values leave the disk; returns the log length. The
    /// snapshot takes a sequence number, so a client's compaction is fenced
    /// as its writes are.
    Compact {
        table: TableName,
        fence: Fence,
        reply: Reply<u64>,
    },
    /// Makes this copy the table's primary under a later term; returns the log length.
    Promote {
        table: TableName,
        term: u64,
        primary: String,
        reply: Reply<u64>,
    },
    /// Stops serving a table and deletes its log, or with `keep` sets the log
    /// aside, as for a copy whose writes diverged from the primary's.
    Release {
        table: TableName,
        keep: bool,
        reply: Reply<()>,
    },
    /// Stops serving a table but leaves its log in place, for another shard
    /// of this server to restore it from.
    Handoff { table: TableName, reply: Reply<()> },
    /// Rewrites a table's log under the current key, keeping every entry and
    /// sequence number; returns the log length.
    Rotate { table: TableName, reply: Reply<u64> },
    /// Every table's log length, terms and creation id.
    LogLengths { reply: Reply<Vec<LogLength>> },
    /// The keys of each query's context, planned as a prediction would be.
    Contexts {
        table: TableName,
        features: Vec<f32>,
        strings: Strings,
        before: Option<Vec<i64>>,
        reply: Reply<Vec<Vec<u128>>>,
    },
    /// Each query's region churn; None where its region has no built context.
    RegionChurn {
        table: TableName,
        features: Vec<f32>,
        strings: Strings,
        reply: Reply<Vec<Option<f64>>>,
    },
    /// The rows a table offers queries for contexts assembled across its
    /// partitions, and the queries themselves, as rows travel between nodes.
    Candidates {
        table: TableName,
        features: Vec<f32>,
        strings: Strings,
        /// This partition's share of a sample, if the strategy samples.
        share: Option<u64>,
        /// The queries' cutoff, which every offered row comes before.
        before: Option<i64>,
        /// The strategy the router assembles by, in place of the table's own.
        strategy: Option<Strategy>,
        reply: Reply<(ModelRows, Vec<Offer>)>,
    },
    /// Predicts queries, the rows without keys, against contexts given to
    /// them from the rows with keys, gathered across the partitions of
    /// `table`, on a table made for the request alone.
    PredictGiven {
        table: TableName,
        rows: ModelRows,
        given: Given,
        trace: String,
        reply: Reply<Predictions>,
    },
    /// Predicts queries against a request's context: from the ephemeral
    /// table its definition names if the shard holds it, else from one made
    /// now with the context's rows. The table stays pinned until the answer.
    Infer {
        context: Box<RequestContext>,
        asked: Asked,
        trace: String,
        reply: Reply<Inferred>,
    },
    /// The models the worker serves, by name.
    Models { reply: Reply<Vec<String>> },
    /// Has the worker load a model under `name` from `spec`, unless it
    /// serves one by that name already; a restarted worker loads it too.
    AddModel {
        name: String,
        spec: String,
        reply: Reply<()>,
    },
    /// Saves every log's head, as a server does before it exits.
    SaveHeads { reply: Reply<()> },
    /// Stops the worker and the shard, as a server does once it has drained:
    /// the worker is told to exit, and killed if it does not; nothing is
    /// started again.
    Shutdown { reply: Reply<()> },
    /// Has the worker sample its own threads for `seconds`, their stacks
    /// written collapsed to `path`, and with `trace` write JAX's trace of
    /// the window into that directory.
    Record {
        seconds: f64,
        path: String,
        trace: Option<String>,
        reply: Reply<()>,
    },
}

struct Entry {
    id: TableId,
    log: Log,
    definition: TableDefinition,
    /// Columns in each prediction.
    outputs: u64,
    /// The quantile level of each prediction column, for a table predicting values.
    levels: Option<Vec<f64>>,
    /// One per column; only categorical columns' hold words.
    vocabularies: Vec<Vocabulary>,
    /// The series the table holds the keys of, for a table of series.
    series: SeriesKeys,
    /// The log length a compaction on the shard's own account waits for, and
    /// how much longer the next wait is: each failed try copies the whole
    /// table, so the waits double.
    compact_after: u64,
    compact_wait: u64,
    metrics: Arc<TableMetrics>,
}

/// What a table keeps of its records besides rows: its columns' words and
/// its series' keys.
#[derive(Clone, Debug, Default)]
struct Learned {
    vocabularies: Vec<Vocabulary>,
    series: SeriesKeys,
}

/// The series a table of series has been given the keys of, in the order it
/// first saw them, with the digests it knows them by.
#[derive(Clone, Debug, Default)]
struct SeriesKeys {
    keys: Vec<SeriesKey>,
    digests: HashSet<u64>,
}

impl SeriesKeys {
    fn keys(&self) -> &[SeriesKey] {
        &self.keys
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    /// The keys among `named` the table does not hold yet, each once and
    /// checked to read back, in the order named.
    fn new_among(&self, named: Vec<SeriesKey>) -> Result<Vec<SeriesKey>, String> {
        let mut fresh = Vec::new();
        let mut seen = HashSet::new();
        for key in named {
            key.check()?;
            if !self.digests.contains(&key.digest) && seen.insert(key.digest) {
                fresh.push(key);
            }
        }
        Ok(fresh)
    }

    /// Follows a record: a snapshot's keys are every key and a truncate
    /// leaves none, and rows add the keys they name that are not held.
    fn track(&mut self, record: &Record) {
        if matches!(record, Record::Snapshot(_) | Record::Truncate) {
            self.keys.clear();
            self.digests.clear();
        }
        for key in record.series() {
            if self.digests.insert(key.digest) {
                self.keys.push(key.clone());
            }
        }
    }
}

/// A table made from rows gathered for given contexts.
struct Scratch {
    /// Which partition's definition it took, which creation, and a hash of its rows.
    content: (TableName, Option<CreationId>, u64),
    /// Each gathered row's id, by its position among the request's rows.
    ids: Vec<RowId>,
    /// The vocabulary each categorical column coded its rows' words with.
    vocabularies: Vec<(usize, Vocabulary)>,
    /// When it was last used, by the shard's count of uses.
    used: u64,
    /// Its partition's metrics.
    metrics: Arc<TableMetrics>,
}

/// Idle scratch tables a shard keeps for requests over the same rows.
const SCRATCH_KEPT: usize = 4;

/// The metrics every ephemeral table reports under: one series for them
/// all, named as no table may be, since each request context is a table.
const EPHEMERAL_METRICS: &str = "ctx-*";

/// How many ephemeral tables a shard keeps, in count and bytes of rows, and
/// how long one may go unused; past either bound the least recently used
/// unpinned tables go first. A table is pinned while a request reads it.
#[derive(Clone, Copy, Debug)]
pub struct EphemeralLimits {
    pub tables: usize,
    pub bytes: u64,
    pub idle: Duration,
}

/// A request's context, as the shard makes its ephemeral table from it.
#[derive(Clone, Debug)]
pub struct RequestContext {
    /// What names the context, and routes it to its shard.
    pub fingerprint: [u8; 32],
    /// The table's definition: `durability = ephemeral`, named for the
    /// context's fingerprint.
    pub definition: TableDefinition,
    pub keys: Vec<u128>,
    /// Row-major features, `definition.features()` per row.
    pub features: Vec<f32>,
    pub targets: Targets,
    /// Each row's time, for a context of series.
    pub times: Option<Vec<Option<i64>>>,
    /// Whether many requests bring this context, as every prediction of a
    /// hosted revision brings its snapshot's rows: its state is fit and
    /// kept, and queries embed against it, as for a durable table, so its
    /// predictions are the ones the same rows bound durably answer, bit for
    /// bit, from the first request on. Otherwise the context is one
    /// request's, fit and decoded in one fused call.
    pub shared: bool,
}

/// What a request asks of its context: predictions of query rows, or
/// forecasts of some of its series.
#[derive(Clone, Debug, PartialEq)]
pub enum Asked {
    /// Row-major query features, `definition.features()` per row.
    Rows(Vec<f32>),
    /// Forecasts of `series` from before each cutoff, over `horizon` steps,
    /// each with the members `groups` names.
    Forecast {
        series: Vec<u64>,
        before: Option<Vec<i64>>,
        horizon: NonZeroUsize,
        groups: Option<Vec<Vec<u64>>>,
    },
}

/// Predictions against a request's context, with the quantile level of each
/// column for a table predicting values.
#[derive(Clone, Debug, PartialEq)]
pub struct Inferred {
    pub predictions: Predictions,
    pub levels: Option<Vec<f64>>,
}

/// Memory, padding and timing settings for a shard.
#[derive(Clone, Copy, Debug)]
pub struct ShardSettings {
    pub device_bytes: u64,
    pub host_bytes: u64,
    /// Memory for the tables' rows; rows past it wait in sealed files.
    pub row_memory_bytes: u64,
    pub disk_bytes: u64,
    pub slot_bytes: u64,
    pub bucket_minimum: usize,
    /// Context bytes one step may newly encode; the device tier's size if None.
    pub step_bytes: Option<u64>,
    /// The order each step takes waiting tables in.
    pub scheduling: Scheduling,
    /// Which cached context a full tier gives up first.
    pub eviction: Eviction,
    /// How long the worker may take to answer a step before it is started again.
    pub step_timeout: Duration,
    /// Bytes of queries that may wait for a step; requests past it are refused.
    pub queue_bytes: u64,
    /// The most bytes of rows one record of a compacted log holds: a log
    /// entry's length is 32 bits, and a record is sealed in memory whole.
    pub snapshot_bytes: u64,
    /// Times each part of planning a step, for the `gf_plan_part_*` metrics.
    pub timers: bool,
    /// The ephemeral tables of request contexts the shard keeps.
    pub ephemeral: EphemeralLimits,
}

/// What wakes a shard: a command, the worker's reply to the oldest step, the
/// worker stopping, the oldest step passing its deadline, or a stopped worker
/// due another start or done starting.
enum Event {
    /// The worker compiled programs that steps deferred tables for.
    Compiled,
    /// The worker's answer to the oldest model it was told to load.
    Added(worker::Reply),
    Command(Option<Box<Command>>),
    /// A log rewritten in the background, ready to install.
    Compacted(Box<Finished>),
    Reply(worker::Reply),
    Stopped,
    Hung,
    Retry,
    Started(Box<Result<WorkerConnection, String>>),
    Idle,
    /// The least recently used ephemeral table has gone unused too long.
    Expired,
}

/// A compaction under way: the writes it holds back, the callers waiting for
/// it, and the number that tells its result from an earlier table's.
#[derive(Default)]
struct Compacting {
    number: u64,
    held: Vec<Queued>,
    replies: Vec<Reply<u64>>,
}

/// A compaction's result, from the thread that wrote it.
struct Finished {
    table: TableName,
    number: u64,
    compacted: Result<Compacted, String>,
}

/// A worker that stopped: when to start it again, how long to wait after
/// that, and the start under way, if one is.
struct Down {
    retry: Instant,
    delay: Duration,
    starting: Option<tokio::task::JoinHandle<Result<WorkerConnection, String>>>,
}

/// One worker, its node, and its tables.
pub struct Shard {
    node: Node,
    worker: WorkerConnection,
    tables: HashMap<TableName, Entry>,
    /// Tables made from rows gathered for given contexts, kept a while after
    /// their requests leave so requests over the same rows reuse their caches.
    scratch: HashMap<TableId, Scratch>,
    /// Counts uses of scratch tables, so the least recent go first.
    scratch_uses: u64,
    /// Request contexts' tables, by name, held in memory alone.
    ephemeral: EphemeralTables,
    /// The routes to the shards, and this shard's place in them, told
    /// which contexts' tables it holds.
    routes: Option<(Arc<Routes>, usize)>,
    /// Each served table's name, by its id in the node.
    names: HashMap<TableId, TableName>,
    profiles: HashMap<(String, usize), ModelProfile>,
    /// How many rows the worker's fits take, by model and feature count:
    /// the most a state alone is fit on, and the most a fit embeds.
    fit_limits: HashMap<(String, usize), (usize, usize)>,
    key: EncryptionKey,
    directory: PathBuf,
    bucket_minimum: NonZeroUsize,
    /// The node's device, host and disk tiers, which the worker's pools match.
    tiers: [TierConfig; 3],
    step_timeout: Duration,
    queue_bytes: u64,
    snapshot_bytes: u64,
    pending: HashMap<RequestId, Pending>,
    /// Bytes of queries in `pending`.
    queued_bytes: u64,
    in_flight: VecDeque<InFlight>,
    /// Tables added to or removed from the worker whose replies are still due.
    controls: VecDeque<(Control, oneshot::Receiver<worker::Reply>)>,
    /// Models the worker is loading, oldest first.
    adding: VecDeque<models::Adding>,
    /// When the worker last finished a step, to time steps that queued behind it.
    finished: Instant,
    /// Failed steps in a row, by table, and the tables quarantined for them.
    failures: HashMap<TableId, u32>,
    quarantined: HashMap<TableId, String>,
    /// Tables out of service until the server restarts, and why.
    broken: HashMap<TableId, String>,
    /// Each table's last writes by client id.
    recent: HashMap<TableId, Recent>,
    /// Set while the worker is stopped and starting again.
    down: Option<Down>,
    /// Whether the server told the shard to shut down, so a worker that
    /// stops is not started again.
    stopping: bool,
    metrics: Arc<Metrics>,
    /// The bytes of each cache tier this shard last added to the gauges.
    cache_reported: [u64; 3],
    /// The bytes of rows in memory this shard last added to its gauge.
    rows_reported: u64,
    /// Tables whose logs are being rewritten off the shard, and where their results come.
    compacting: HashMap<TableName, Compacting>,
    /// Tables whose logs synced since their heads were last saved.
    unsaved_heads: HashSet<TableName>,
    compactions: mpsc::UnboundedReceiver<Finished>,
    compacted: mpsc::UnboundedSender<Finished>,
}

impl Shard {
    /// A shard around a started worker, keeping table logs in `directory`.
    ///
    /// # Errors
    ///
    /// If the memory settings are invalid.
    pub fn new(
        worker: WorkerConnection,
        memory: ShardSettings,
        key: EncryptionKey,
        directory: PathBuf,
        metrics: Arc<Metrics>,
    ) -> Result<Self, String> {
        let slot = NonZeroU64::new(memory.slot_bytes).ok_or("slot_bytes must be positive")?;
        let tier = |bytes: u64| -> Result<TierConfig, String> {
            let slots = u32::try_from(bytes / slot.get())
                .map_err(|_| "too many cache slots".to_string())?;
            Ok(TierConfig {
                slot_bytes: slot,
                slots,
            })
        };
        let tiers = [
            tier(memory.device_bytes)?,
            tier(memory.host_bytes)?,
            tier(memory.disk_bytes)?,
        ];
        let (compacted, compactions) = mpsc::unbounded_channel();
        let mut node = Node::new(tiers[0], tiers[1])
            .and_then(|node| node.with_disk(tiers[2]))
            .map_err(|error| error.to_string())?
            .with_eviction(memory.eviction)
            .with_row_memory(RowMemory::bounded(memory.row_memory_bytes));
        if let Some(bytes) = memory.step_bytes {
            node.set_step_bytes(bytes);
        }
        node.set_scheduling(memory.scheduling);
        node.set_timers(memory.timers);
        Ok(Self {
            node,
            tiers,
            worker,
            tables: HashMap::new(),
            names: HashMap::new(),
            profiles: HashMap::new(),
            fit_limits: HashMap::new(),
            key,
            directory,
            bucket_minimum: NonZeroUsize::new(memory.bucket_minimum)
                .ok_or("bucket_minimum must be positive")?,
            step_timeout: memory.step_timeout,
            queue_bytes: memory.queue_bytes,
            snapshot_bytes: memory.snapshot_bytes,
            pending: HashMap::new(),
            scratch: HashMap::new(),
            scratch_uses: 0,
            ephemeral: EphemeralTables::new(memory.ephemeral),
            routes: None,
            queued_bytes: 0,
            in_flight: VecDeque::new(),
            controls: VecDeque::new(),
            adding: VecDeque::new(),
            finished: Instant::now(),
            failures: HashMap::new(),
            quarantined: HashMap::new(),
            broken: HashMap::new(),
            recent: HashMap::new(),
            down: None,
            stopping: false,
            metrics,
            cache_reported: [0; 3],
            rows_reported: 0,
            compacting: HashMap::new(),
            unsaved_heads: HashSet::new(),
            compactions,
            compacted,
        })
    }

    /// Tells `routes`, as shard `shard` of them, which contexts' tables it
    /// holds and drops.
    #[must_use]
    pub fn with_routes(mut self, routes: Arc<Routes>, shard: usize) -> Self {
        self.routes = Some((routes, shard));
        self
    }

    /// Sizes the worker's pools as the node's tiers.
    ///
    /// # Errors
    ///
    /// If the worker cannot be reached.
    pub async fn allocate(&mut self) -> Result<(), String> {
        let frame = WorkerRequest::Allocate {
            device_slots: self.tiers[0].slots,
            host_slots: self.tiers[1].slots,
            disk_slots: self.tiers[2].slots,
        }
        .frame();
        // The worker compiles the programs earlier workers recorded before
        // it answers, reporting progress after each, so the shard takes no
        // step until they are ready.
        let reply = self
            .worker
            .request_progressing(&frame, self.step_timeout)
            .await?;
        if let Ok(WorkerReply::Warmed {
            compiles,
            seconds,
            compile_cache,
        }) = WorkerReply::of(&reply)
        {
            tracing::info!(compiles, seconds, "worker compiled its recorded programs");
            self.metrics.warm_compiles.add(compiles);
            self.metrics.warm_seconds.add(seconds);
            self.metrics.compile_cache_hits.add(compile_cache.hits);
            self.metrics.compile_cache_misses.add(compile_cache.misses);
        }
        Ok(())
    }

    /// Runs the shard on a thread of its own, so planning never holds up the
    /// runtime that serves Flight; commands go to the returned sender.
    ///
    /// # Errors
    ///
    /// If the thread or its runtime cannot start.
    pub fn spawn(self) -> std::io::Result<mpsc::Sender<Command>> {
        let (sender, receiver) = mpsc::channel(1024);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        std::thread::Builder::new()
            .name("gf-shard".to_string())
            .spawn(move || runtime.block_on(self.run(receiver)))?;
        Ok(sender)
    }

    async fn run(mut self, mut commands: mpsc::Receiver<Command>) {
        loop {
            self.cancel_abandoned();
            self.settle_controls().await;
            while self.in_flight.len() < PIPELINE_DEPTH
                && !self.holds_new_contexts()
                && self.dispatch().await
            {}
            let deadline = self.deadline();
            let expiry = self.ephemeral.expiry();
            let next = self.in_flight.front_mut().map(|step| &mut step.reply);
            let adding = self.adding.front_mut().map(|adding| &mut adding.reply);
            let up = self.down.is_none();
            let stopped = self.worker.pipeline.stopped();
            let compiled = self.worker.pipeline.compiled();
            let retry = self
                .down
                .as_ref()
                .filter(|down| down.starting.is_none())
                .map(|down| down.retry);
            let starting = self.down.as_mut().and_then(|down| down.starting.as_mut());
            let heads_behind = !self.unsaved_heads.is_empty();
            // Biased, so a reply that came in while the shard was busy, such
            // as compacting a log, wins over the deadline it would have met.
            let event = tokio::select! {
                biased;
                reply = async { next.expect("guarded").await }, if next.is_some() => {
                    Event::Reply(reply.unwrap_or_else(|_| Err("the worker stopped".to_string())))
                }
                added = async { adding.expect("guarded").await }, if adding.is_some() => {
                    Event::Added(added.unwrap_or_else(|_| Err("the worker stopped".to_string())))
                }
                () = stopped, if up => Event::Stopped,
                () = compiled, if up => Event::Compiled,
                () = tokio::time::sleep_until(deadline.unwrap_or_else(Instant::now)),
                    if deadline.is_some() => Event::Hung,
                command = commands.recv() => Event::Command(command.map(Box::new)),
                Some(finished) = self.compactions.recv() => Event::Compacted(Box::new(finished)),
                () = tokio::time::sleep_until(retry.unwrap_or_else(Instant::now)),
                    if retry.is_some() => Event::Retry,
                started = async { starting.expect("guarded").await }, if starting.is_some() => {
                    Event::Started(Box::new(
                        started.unwrap_or_else(|error| Err(error.to_string())),
                    ))
                }
                () = std::future::ready(()), if heads_behind => Event::Idle,
                () = tokio::time::sleep_until(expiry.unwrap_or_else(Instant::now)),
                    if expiry.is_some() => Event::Expired,
            };
            match event {
                Event::Command(Some(command)) => {
                    let mut batch = Vec::new();
                    self.queue_or_handle(*command, &mut batch).await;
                    while let Ok(command) = commands.try_recv() {
                        self.queue_or_handle(command, &mut batch).await;
                    }
                    self.commit_batch(batch).await;
                }
                Event::Command(None) => break,
                Event::Stopped | Event::Hung if self.stopping => break,
                Event::Compacted(finished) => self.finish_compaction(*finished).await,
                Event::Reply(reply) => self.finish(reply).await,
                Event::Added(answer) => self.added(answer),
                Event::Stopped => self.restart("the worker stopped").await,
                Event::Hung if self.overdue() => self.restart(&self.hung()).await,
                // The worker reported progress since the deadline was set.
                Event::Hung => {}
                Event::Retry => {
                    let launch = self.worker.launch.clone();
                    let down = self.down.as_mut().expect("retries come while down");
                    down.starting = Some(tokio::spawn(WorkerConnection::spawn(launch)));
                }
                Event::Started(started) => self.started(*started).await,
                Event::Idle => self.save_idle_heads(),
                Event::Expired => self.sweep_ephemeral().await,
                Event::Compiled => {
                    let keys = self.worker.pipeline.take_compiled();
                    self.node.compiled(keys);
                }
            }
            self.flush_controls();
        }
        self.drain().await;
        self.save_heads();
    }

    /// When the oldest step in flight must have answered. A step starts once
    /// the one before it finished, since the worker runs them in turn, and
    /// each compile the worker reports finished during it moves it later.
    /// A model being loaded is timed the same way.
    fn deadline(&self) -> Option<Instant> {
        let progressed = self.worker.pipeline.progressed();
        let step = self.in_flight.front().map(|step| {
            step.sent
                .max(self.finished)
                .max(progressed.unwrap_or(step.sent))
                + self.step_timeout
        });
        [step, self.adding_deadline()].into_iter().flatten().min()
    }

    /// Whether the oldest step in flight is past its deadline.
    fn overdue(&self) -> bool {
        self.deadline()
            .is_some_and(|deadline| deadline <= Instant::now())
    }

    fn hung(&self) -> String {
        let on = self.loading().map_or_else(
            || "a step".to_string(),
            |name| format!("loading model {name:?}"),
        );
        format!(
            "the worker went {} seconds without progress on {on}",
            self.step_timeout.as_secs()
        )
    }

    /// Saves each log's head that lags its log, as a failed write at
    /// compaction leaves it; a failure leaves it at an earlier length, which
    /// only weakens its check until the next save.
    fn save_heads(&mut self) {
        for (table, entry) in &mut self.tables {
            if let Err(error) = entry.log.save_head() {
                tracing::error!(table = table.as_str(), "saving the log's head: {error}");
            }
        }
        self.unsaved_heads.clear();
    }

    fn save_idle_heads(&mut self) {
        let tables: Vec<TableName> = self.unsaved_heads.drain().collect();
        for table in tables {
            if let Some(entry) = self.tables.get_mut(&table)
                && let Err(error) = entry.log.save_head()
            {
                tracing::error!(table = table.as_str(), "saving the log's head: {error}");
            }
        }
    }

    /// Takes queries whose callers have gone, such as ones that timed out,
    /// out of the queue, so they are neither planned nor counted against it.
    fn cancel_abandoned(&mut self) {
        let abandoned: Vec<RequestId> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.reply.is_closed())
            .map(|(&request, _)| request)
            .collect();
        for request in abandoned {
            self.node.cancel(request);
            self.take(request);
        }
    }

    /// Queues a write to commit with the others that arrived with it, or
    /// commits the queued writes and handles any other command, so commands
    /// take effect in the order they arrived.
    /// Adds a write to the batch, or holds it while its table's log is rewritten.
    fn queue(&mut self, batch: &mut Vec<Queued>, queued: Queued) {
        match self.compacting.get_mut(&queued.table) {
            Some(compacting) => compacting.held.push(queued),
            None => batch.push(queued),
        }
    }

    async fn queue_or_handle(&mut self, command: Command, batch: &mut Vec<Queued>) {
        match command {
            Command::Write {
                table,
                write,
                fence,
                charge,
                reply,
            } => self.queue(
                batch,
                Queued {
                    table,
                    write,
                    fence,
                    charge,
                    reply,
                },
            ),
            Command::Refit {
                table,
                fence,
                reply,
            } => self.queue(
                batch,
                Queued {
                    table,
                    write: Write::Refit,
                    fence,
                    charge: Charge::default(),
                    reply,
                },
            ),
            Command::SetStrategy {
                table,
                strategy,
                fence,
                reply,
            } => self.queue(
                batch,
                Queued {
                    table,
                    write: Write::Strategy(strategy),
                    fence,
                    charge: Charge::default(),
                    reply,
                },
            ),
            command => {
                self.commit_batch(std::mem::take(batch)).await;
                self.handle(command).await;
            }
        }
    }

    async fn handle(&mut self, command: Command) {
        match command {
            Command::Create {
                definition,
                terms,
                reply,
            } => {
                let _ = reply.send(self.create(&definition, terms).await);
            }
            Command::Restore { replay, reply } => {
                let _ = reply.send(self.restore(*replay).await);
            }
            Command::SaveHeads { reply } => {
                self.save_heads();
                let _ = reply.send(Ok(()));
            }
            Command::Shutdown { reply } => {
                self.worker.terminate().await;
                self.stopping = true;
                let _ = reply.send(Ok(()));
            }
            Command::Record {
                seconds,
                path,
                trace,
                reply,
            } => {
                let request = WorkerRequest::Record {
                    seconds,
                    path,
                    trace,
                };
                let answered = if self.down.is_some() {
                    Err(ServerError::Internal("the worker is down".to_string()))
                } else {
                    match self.ask_worker(&request).await {
                        Ok(WorkerReply::Ok) => Ok(()),
                        Ok(WorkerReply::Error { message }) => Err(ServerError::Internal(message)),
                        Ok(other) => Err(ServerError::Internal(format!(
                            "the worker answered a record with {other:?}"
                        ))),
                        Err(error) => Err(error),
                    }
                };
                let _ = reply.send(answered);
            }
            Command::Write { reply, .. }
            | Command::Refit { reply, .. }
            | Command::SetStrategy { reply, .. } => {
                let _ = reply.send(Err(ServerError::Internal(
                    "writes are committed in batches".to_string(),
                )));
            }
            Command::Describe { table, reply } => {
                let described = self.entry(&table).and_then(|id| {
                    let entry = &self.tables[&table];
                    Ok(Description {
                        definition: entry.definition.clone(),
                        rows: self.node.len(id)?,
                        log_length: entry.log.length(),
                        outputs: entry.outputs,
                        levels: entry.levels.clone(),
                        strategy: StrategySpec::of(self.node.strategy(id)?),
                        context_rows: self.node.spec(id)?.context_rows.map(NonZeroU64::get),
                        effective_fit_rows: {
                            let spec = self.node.spec(id)?;
                            (spec.summary == SummaryScope::Table && spec.series.is_none())
                                .then(|| spec.effective_fit_rows() as u64)
                        },
                        regions: self.node.region_stats(id)?,
                        vocabularies: entry
                            .vocabularies
                            .iter()
                            .take(entry.definition.columns.len())
                            .map(|vocabulary| vocabulary.words().to_vec())
                            .collect(),
                        labels: entry
                            .vocabularies
                            .get(entry.definition.columns.len())
                            .map(|vocabulary| vocabulary.words().to_vec())
                            .unwrap_or_default(),
                        series: entry
                            .definition
                            .series
                            .is_some()
                            .then(|| entry.series.len() as u64),
                    })
                });
                let _ = reply.send(described);
            }
            Command::KnownSeries { table, reply } => {
                let known = self.entry(&table).and_then(|_| {
                    let entry = &self.tables[&table];
                    if entry.definition.series.is_none() {
                        return Err(ServerError::Invalid(format!(
                            "table {table:?} holds rows, not series"
                        )));
                    }
                    Ok(entry.series.keys().to_vec())
                });
                let _ = reply.send(known);
            }
            Command::Models { reply } => {
                let _ = reply.send(Ok(self.worker.models.clone()));
            }
            Command::AddModel { name, spec, reply } => self.add_model(name, spec, reply).await,
            Command::ReadLog {
                table,
                from,
                limit,
                bytes,
                reply,
            } => {
                let entries = self.entry(&table).and_then(|_| {
                    self.tables[&table]
                        .log
                        .read(from, limit, bytes)
                        .map_err(ServerError::from)
                });
                let _ = reply.send(entries);
            }
            Command::Replicate {
                table,
                entries,
                lineage,
                reply,
            } => {
                let _ = reply.send(self.replicate(&table, entries, *lineage).await);
            }
            Command::Compact {
                table,
                fence,
                reply,
            } => {
                let fenced = match self.tables.get(&table) {
                    Some(entry) => fence.check(&table, &entry.log),
                    None => Ok(()),
                };
                match fenced.and_then(|()| self.start_compaction(&table)) {
                    Ok(()) => self
                        .compacting
                        .get_mut(&table)
                        .expect("just started")
                        .replies
                        .push(reply),
                    Err(error) => {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Promote {
                table,
                term,
                primary,
                reply,
            } => {
                let promoted = self
                    .commit(&table, Record::Term { term, primary }, &Fence::NONE)
                    .await;
                let _ = reply.send(promoted.map(|written| written.log_length));
            }
            Command::Rotate { table, reply } => {
                let rotated = self.entry(&table).and_then(|_| {
                    let log = &mut self.tables.get_mut(&table).expect("found above").log;
                    log.rotate()?;
                    Ok(log.length())
                });
                let _ = reply.send(rotated);
            }
            Command::Release { table, keep, reply } => {
                let _ = reply.send(self.release(&table, keep).await);
            }
            Command::Handoff { table, reply } => {
                let _ = reply.send(self.hand_off(&table).await);
            }
            Command::LogLengths { reply } => {
                let lengths = self
                    .tables
                    .iter()
                    .map(|(name, entry)| LogLength {
                        table: name.clone(),
                        length: entry.log.length(),
                        terms: entry.log.terms().clone(),
                        created: entry.definition.created.clone(),
                    })
                    .collect();
                let _ = reply.send(Ok(lengths));
            }
            Command::Contexts {
                table,
                mut features,
                strings,
                before,
                reply,
            } => {
                let contexts = self
                    .resolve(&table, &mut features, &strings)
                    .and_then(|()| self.usable(&table))
                    .and_then(|id| Ok(self.node.contexts(id, &features, before.as_deref())?));
                let _ = reply.send(contexts);
            }
            Command::RegionChurn {
                table,
                mut features,
                strings,
                reply,
            } => {
                let churn = self
                    .resolve(&table, &mut features, &strings)
                    .and_then(|()| self.usable(&table))
                    .and_then(|id| Ok(self.node.region_churn(id, &features)?));
                let _ = reply.send(churn);
            }
            Command::Candidates {
                table,
                features,
                strings,
                share,
                before,
                strategy,
                reply,
            } => {
                let _ = reply.send(self.offer(&table, features, &strings, share, before, strategy));
            }
            Command::Infer {
                context,
                asked,
                trace,
                reply,
            } => match self.infer(*context, asked).await {
                Ok((submitted, levels)) => {
                    let (answered, answer) = oneshot::channel();
                    self.queue_pending(submitted, answered, trace);
                    // A caller that goes away closes the queued reply, so the
                    // abandoned query leaves the queue as a prediction's does.
                    tokio::spawn(relay(answer, reply, levels));
                    self.sweep_ephemeral().await;
                }
                Err(error) => {
                    let _ = reply.send(Err(error));
                }
            },
            Command::PredictGiven {
                table,
                rows,
                given,
                trace,
                reply,
            } => {
                let submitted = self.submit_given(&table, rows, given).await;
                self.admit(submitted, reply, trace);
            }
            Command::Forecast {
                table,
                series,
                before,
                horizon,
                groups,
                trace,
                reply,
            } => {
                let submitted = self.usable(&table).and_then(|id| {
                    self.submit_forecast(id, &table, series, before, horizon, groups)
                });
                self.admit(submitted, reply, trace);
            }
            Command::Predict {
                table,
                mut features,
                strings,
                before,
                given,
                trace,
                reply,
            } => {
                let resolved = self.resolve(&table, &mut features, &strings);
                let submitted =
                    resolved.and_then(|()| self.submit(&table, features, before, given));
                self.admit(submitted, reply, trace);
            }
        }
    }

    /// The rows a table offers queries, then the queries themselves, as rows
    /// travel between partitions: categorical features as words.
    fn offer(
        &mut self,
        table: &str,
        queries: Vec<f32>,
        strings: &Strings,
        share: Option<u64>,
        before: Option<i64>,
        strategy: Option<Strategy>,
    ) -> Result<(ModelRows, Vec<Offer>), ServerError> {
        let id = self.usable(table)?;
        let mut coded = queries.clone();
        self.resolve(table, &mut coded, strings)?;
        let offered = self.node.candidates(id, &coded, share, before, strategy)?;
        let entry = &self.tables[table];
        let width = entry.definition.features();
        let ids: Vec<RowId> = offered.iter().map(|candidate| candidate.row).collect();
        let held = self.node.rows_of(id, &ids)?;
        let count = queries.len() / width.max(1);
        let mut targets: Vec<Option<f32>> = match &held.targets {
            Targets::Classes(classes) => classes.iter().map(|&class| Some(class as f32)).collect(),
            Targets::Values(values) => values.iter().copied().map(Some).collect(),
        };
        targets.resize(ids.len() + count, None);
        let mut rows = ModelRows {
            keys: held.keys.iter().copied().map(Some).collect(),
            features: held.features,
            strings: Vec::new(),
            targets,
        };
        rows.keys.resize(ids.len() + count, None);
        rows.features.extend(&queries);
        for (column, offset) in categorical_offsets(&entry.definition) {
            let words = entry.vocabularies[column].words();
            let asked = strings.iter().find(|(held, _)| *held == column);
            let mut values: Vec<Option<String>> = (0..ids.len())
                .map(|row| {
                    let code = rows.features[row * width + offset];
                    code.is_finite()
                        .then(|| words.get(code as usize).cloned())
                        .flatten()
                })
                .collect();
            values
                .extend((0..count).map(|query| asked.and_then(|(_, words)| words[query].clone())));
            for row in 0..ids.len() + count {
                rows.features[row * width + offset] = f32::NAN;
            }
            rows.strings.push((column, values));
        }
        let offers = offered
            .iter()
            .map(|candidate| Offer {
                query: candidate.query,
                score: candidate.score,
            })
            .chain((0..count).map(|query| Offer {
                query: Some(query as u32),
                score: 0.0,
            }))
            .collect();
        Ok((rows, offers))
    }

    /// Queues queries against the contexts given them, on a table made from
    /// the given rows alone, with `table`'s definition; the table is dropped
    /// once the request leaves.
    async fn submit_given(
        &mut self,
        table: &str,
        rows: ModelRows,
        given: Given,
    ) -> Result<Submitted, ServerError> {
        self.usable(table)?;
        let definition = self.tables[table].definition.clone();
        let width = definition.features();
        let mut keys = Vec::new();
        let mut context = Vec::new();
        let mut queries = Vec::new();
        for (row, key) in rows.keys.iter().enumerate() {
            let values = &rows.features[row * width..(row + 1) * width];
            match key {
                Some(key) => {
                    keys.push(*key);
                    context.extend_from_slice(values);
                }
                None => queries.extend_from_slice(values),
            }
        }
        if queries.is_empty() {
            return Err(ServerError::Invalid("no query rows".to_string()));
        }
        let mut held_words = Vec::with_capacity(rows.strings.len());
        let mut asked_words = Vec::with_capacity(rows.strings.len());
        for (column, words) in &rows.strings {
            let (held, asked): (Vec<_>, Vec<_>) = rows
                .keys
                .iter()
                .zip(words)
                .partition(|(key, _)| key.is_some());
            held_words.push((
                *column,
                held.into_iter()
                    .map(|(_, word)| word.clone())
                    .collect::<Vec<_>>(),
            ));
            asked_words.push((
                *column,
                asked
                    .into_iter()
                    .map(|(_, word)| word.clone())
                    .collect::<Vec<_>>(),
            ));
        }
        let targets =
            partition::targets(&rows, definition.classes == 0).map_err(ServerError::Invalid)?;
        // A partition dropped and made again is another creation, which
        // may differ in its model, task or columns.
        let content = (
            definition.name.clone(),
            definition.created.clone(),
            content_hash(&keys, &context, &held_words, &targets),
        );
        // An idle table made from these same rows serves again, caches and all.
        let reused = self
            .scratch
            .iter()
            .find(|(id, scratch)| {
                scratch.content == content
                    && !self.pending.values().any(|pending| pending.table == **id)
            })
            .map(|(&id, _)| id);
        let id = match reused {
            Some(id) => id,
            None => {
                let mut vocabularies = Vec::with_capacity(held_words.len());
                for (column, held) in &held_words {
                    let mut vocabulary = Vocabulary::default();
                    let mut new = Vec::new();
                    let codes = vocabulary
                        .assign(held, &mut new)
                        .map_err(ServerError::Invalid)?;
                    vocabulary.extend(new);
                    fill(&mut context, &definition, *column, &codes)?;
                    vocabularies.push((*column, vocabulary));
                }
                let id = self.serve_scratch(&definition).await?;
                match self.node.insert(id, &keys, &context, targets) {
                    Ok(ids) => {
                        self.scratch.insert(
                            id,
                            Scratch {
                                content,
                                ids,
                                vocabularies,
                                used: 0,
                                metrics: self.metrics.table(definition.name.as_str()),
                            },
                        );
                    }
                    Err(error) => {
                        self.unserve(id).await;
                        return Err(error.into());
                    }
                }
                id
            }
        };
        self.scratch_uses += 1;
        let scratch = self.scratch.get_mut(&id).expect("made or found above");
        scratch.used = self.scratch_uses;
        for ((column, asked), (_, vocabulary)) in asked_words.iter().zip(&scratch.vocabularies) {
            fill(&mut queries, &definition, *column, &vocabulary.codes(asked))?;
        }
        let ids = &scratch.ids;
        let contexts = given
            .contexts
            .iter()
            .map(|positions| {
                positions
                    .iter()
                    .map(|&position| {
                        ids.get(position as usize).copied().ok_or_else(|| {
                            ServerError::Invalid(format!(
                                "a context names row {position} of {}",
                                ids.len()
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let request = self.node.submit_given(id, &queries, contexts)?;
        Ok(Submitted {
            request,
            table: id,
            bytes: queries.len() as u64 * 4,
        })
    }

    /// Queues queries against a request context's ephemeral table, making
    /// the table from the context's rows unless the shard holds it already.
    async fn infer(
        &mut self,
        context: RequestContext,
        asked: Asked,
    ) -> Result<(Submitted, Option<Vec<f64>>), ServerError> {
        let RequestContext {
            fingerprint,
            definition,
            keys,
            features,
            targets,
            times,
            shared,
        } = context;
        if definition.durability != Durability::Ephemeral {
            return Err(ServerError::Invalid(
                "a request's context takes an ephemeral table".to_string(),
            ));
        }
        let name = definition.name.clone();
        let id = match self.ephemeral.named(&name) {
            Some(table) if table.definition == definition => table.id,
            Some(_) => {
                return Err(ServerError::Internal(format!(
                    "table {name:?} holds another context"
                )));
            }
            None => {
                self.make_ephemeral(
                    fingerprint,
                    definition,
                    &keys,
                    &features,
                    targets,
                    times.as_deref(),
                    shared,
                )
                .await?
            }
        };
        if let Some((routes, shard)) = &self.routes {
            routes.holds(&fingerprint, *shard);
        }
        let levels = self
            .ephemeral
            .get(id)
            .and_then(|table| table.levels.clone());
        let submitted = match asked {
            Asked::Rows(queries) => self.enqueue(id, name.as_str(), queries, None, None)?,
            Asked::Forecast {
                series,
                before,
                horizon,
                groups,
            } => self.submit_forecast(id, name.as_str(), series, before, horizon, groups)?,
        };
        Ok((submitted, levels))
    }

    /// Serves an ephemeral table in memory, without a log, and adds its rows;
    /// see `RequestContext::shared`.
    #[allow(clippy::too_many_arguments)]
    async fn make_ephemeral(
        &mut self,
        fingerprint: [u8; 32],
        definition: TableDefinition,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
        shared: bool,
    ) -> Result<TableId, ServerError> {
        let (spec, _, levels) = self.spec(&definition, false).await?;
        let id = self.node.create_table(spec)?;
        if !shared {
            self.node.set_single_use(id)?;
        }
        self.send_to_worker(id, &definition);
        let rows = targets.len() as u64;
        if let Err(error) = self.node.insert_at(id, keys, features, targets, times) {
            self.unserve(id).await;
            return Err(error.into());
        }
        let bytes = keys.len() as u64 * 16 + features.len() as u64 * 4 + rows * 4;
        self.ephemeral.insert(Ephemeral::new(
            id,
            fingerprint,
            definition,
            levels,
            bytes,
            self.metrics.table(EPHEMERAL_METRICS),
        ));
        self.metrics.ephemeral_tables.add(1);
        Ok(id)
    }

    /// Drops ephemeral tables, least recently used first, while more are
    /// kept than the limits allow or they went unused past the idle limit;
    /// tables a request reads are pinned.
    async fn sweep_ephemeral(&mut self) {
        for table in self.ephemeral.evict(Instant::now()) {
            self.metrics.ephemeral_tables.sub(1);
            self.dropped(&table);
            self.unserve(table.id).await;
        }
    }

    /// Drops the scratch tables no request needs, past the `SCRATCH_KEPT`
    /// most recently used.
    async fn sweep_scratch(&mut self) {
        let mut idle: Vec<(u64, TableId)> = self
            .scratch
            .iter()
            .filter(|(id, _)| !self.pending.values().any(|pending| pending.table == **id))
            .map(|(&id, scratch)| (scratch.used, id))
            .collect();
        idle.sort_unstable_by(|left, right| right.cmp(left));
        for (_, id) in idle.into_iter().skip(SCRATCH_KEPT) {
            self.scratch.remove(&id);
            self.unserve(id).await;
        }
    }

    /// Writes the codes of queries' strings into their features: the codes
    /// the table's vocabularies hold, and NaN for words they do not.
    fn resolve(
        &self,
        table: &str,
        features: &mut [f32],
        strings: &Strings,
    ) -> Result<(), ServerError> {
        self.entry(table)?;
        let entry = &self.tables[table];
        for (column, values) in strings {
            let vocabulary = entry
                .vocabularies
                .get(*column)
                .ok_or_else(|| ServerError::Internal(format!("no column {column} to code")))?;
            fill(
                features,
                &entry.definition,
                *column,
                &vocabulary.codes(values),
            )?;
        }
        Ok(())
    }

    /// Codes a write's strings: known words take their codes, and new words
    /// the next ones, which the record carries for replay and replicas.
    fn encode(&self, table: &str, incoming: Incoming) -> Result<Rows, ServerError> {
        let Incoming { mut rows, strings } = incoming;
        let entry = &self.tables[table];
        // String labels code against the vocabulary after the columns'.
        let target = entry.definition.columns.len();
        let labelled = strings.iter().any(|(column, _)| *column == target);
        self.check_labels(table, labelled)?;
        if !rows.series.is_empty() {
            if entry.definition.series.is_none() {
                return Err(ServerError::Invalid(format!(
                    "table {table:?} holds rows, not series, so its writes name no series keys"
                )));
            }
            // The record carries the keys new to the table. A write staged
            // earlier in the batch may carry a key again, since it applies
            // after this one is prepared; replay takes each key once.
            rows.series = entry
                .series
                .new_among(std::mem::take(&mut rows.series))
                .map_err(ServerError::Invalid)?;
        }
        for (column, values) in strings {
            let vocabulary = entry
                .vocabularies
                .get(column)
                .ok_or_else(|| ServerError::Internal(format!("no column {column} to code")))?;
            let mut added = Vec::new();
            let codes = vocabulary.assign(&values, &mut added).map_err(|error| {
                let name = entry
                    .definition
                    .columns
                    .get(column)
                    .map_or("target", |column| column.name.as_str());
                ServerError::Invalid(format!("column {name:?}: {error}"))
            })?;
            if column == target {
                let classes = entry.definition.classes;
                if vocabulary.words().len() + added.len() > classes as usize {
                    let held: Vec<&String> = vocabulary.words().iter().chain(&added).collect();
                    return Err(ServerError::Invalid(format!(
                        "table {table:?} has {classes} classes, but its labels would be {held:?}"
                    )));
                }
                if let Targets::Classes(labels) = &mut rows.targets {
                    // Codes are whole numbers below the classes, so they fit;
                    // a missing label is one not known yet.
                    *labels = codes
                        .iter()
                        .map(|&code| {
                            if code.is_finite() {
                                code as i32
                            } else {
                                UNLABELED
                            }
                        })
                        .collect();
                }
            } else {
                fill(&mut rows.features, &entry.definition, column, &codes)?;
            }
            if !added.is_empty() {
                rows.words.push(Words {
                    column,
                    values: added,
                });
            }
        }
        Ok(rows)
    }

    /// Whether a table's targets are known to be labels or codes: it holds
    /// rows, or labels.
    fn settled(&self, table: &str) -> bool {
        let entry = &self.tables[table];
        let labelled = entry
            .vocabularies
            .get(entry.definition.columns.len())
            .is_some_and(|labels| !labels.words().is_empty());
        labelled || self.node.len(entry.id).is_ok_and(|rows| rows > 0)
    }

    /// Refuses targets given as labels to a table whose classes are codes,
    /// and as codes to one whose classes are labels.
    fn check_labels(&self, table: &str, labelled: bool) -> Result<(), ServerError> {
        let entry = &self.tables[table];
        let labels = entry
            .vocabularies
            .get(entry.definition.columns.len())
            .map_or(&[][..], Vocabulary::words);
        if !labelled && !labels.is_empty() {
            return Err(ServerError::Invalid(format!(
                "table {table:?}'s classes are labels {labels:?}; give its targets as labels too"
            )));
        }
        if labelled && labels.is_empty() && self.node.len(entry.id)? > 0 {
            return Err(ServerError::Invalid(format!(
                "table {table:?}'s classes are codes 0 .. {}; give its targets as codes too",
                entry.definition.classes.saturating_sub(1)
            )));
        }
        Ok(())
    }

    /// Makes a table's remembered writes those a snapshot carries.
    fn restore_writes(&mut self, id: TableId, writes: &[PastWrite]) {
        let mut recent = Recent::default();
        for write in writes {
            let written = Written {
                rows: usize::try_from(write.rows).unwrap_or(usize::MAX),
                log_length: write.log_length,
            };
            recent.remember(write.id.clone(), written);
        }
        self.recent.insert(id, recent);
    }

    /// Adds a record's words to the table's vocabularies, or with `reset`
    /// makes them exactly those words, as a snapshot's are.
    fn learn(&mut self, id: TableId, words: &[Words], reset: bool) {
        if let Some(entry) = self
            .names
            .get(&id)
            .and_then(|name| self.tables.get_mut(name))
        {
            learn(&mut entry.vocabularies, words, reset);
        }
    }

    /// Queues a request's queries with the node, unless its table is
    /// quarantined or the queue is full.
    fn submit(
        &mut self,
        table: &str,
        features: Vec<f32>,
        before: Option<Vec<i64>>,
        given: Option<Vec<Vec<u128>>>,
    ) -> Result<Submitted, ServerError> {
        let id = self.usable(table)?;
        self.enqueue(id, table, features, before, given)
    }

    /// Queues queries of a table the node holds, within the queue's budget.
    fn enqueue(
        &mut self,
        id: TableId,
        table: &str,
        features: Vec<f32>,
        before: Option<Vec<i64>>,
        given: Option<Vec<Vec<u128>>>,
    ) -> Result<Submitted, ServerError> {
        if let Some(reason) = self.quarantined.get(&id) {
            return Err(ServerError::Invalid(format!(
                "table {table:?} takes no queries: it failed {FAILURES} steps running \
                 ({reason}); a write or refit lets it try again"
            )));
        }
        // Queries and the 16-byte keys of any given contexts.
        let keys = given
            .as_ref()
            .map_or(0, |contexts| contexts.iter().map(Vec::len).sum());
        let bytes = features.len() as u64 * 4 + keys as u64 * 16;
        if self.queued_bytes + bytes > self.queue_bytes {
            self.cancel_abandoned();
        }
        if self.queued_bytes + bytes > self.queue_bytes {
            return Err(ServerError::Busy(format!(
                "{} MiB of queries are waiting already; retry shortly",
                self.queued_bytes >> 20
            )));
        }
        let request = match given {
            Some(contexts) => self.node.submit_keys(id, &features, &contexts)?,
            None => self.node.submit_before(id, &features, before.as_deref())?,
        };
        Ok(Submitted {
            request,
            table: id,
            bytes,
        })
    }

    /// Queues a forecast of series, as `submit` queues queries.
    /// The definition of a table the shard serves, durable or ephemeral.
    fn definition(&self, id: TableId, table: &str) -> Option<&TableDefinition> {
        self.ephemeral
            .get(id)
            .map(|ephemeral| &ephemeral.definition)
            .or_else(|| self.tables.get(table).map(|entry| &entry.definition))
    }

    fn submit_forecast(
        &mut self,
        id: TableId,
        table: &str,
        series: Vec<u64>,
        before: Option<Vec<i64>>,
        horizon: NonZeroUsize,
        groups: Option<Vec<Vec<u64>>>,
    ) -> Result<Submitted, ServerError> {
        if let Some(reason) = self.quarantined.get(&id) {
            return Err(ServerError::Invalid(format!(
                "table {table:?} takes no forecasts: it failed {FAILURES} steps running \
                 ({reason}); a write lets it try again"
            )));
        }
        let definition = self
            .definition(id, table)
            .ok_or_else(|| ServerError::Internal(format!("table {table:?} has no definition")))?;
        let key = (definition.model.clone(), definition.model_features());
        if let Some(profile) = self.profiles.get(&key) {
            let limit = horizon_limit(
                profile_steps(profile.output_patch, "output patch")?,
                profile
                    .pass_limit
                    .map(|steps| profile_steps(steps, "pass limit"))
                    .transpose()?,
            )
            .get();
            if horizon.get() > limit {
                return Err(ServerError::Invalid(format!(
                    "a horizon of {horizon} steps is past the {limit} the model {:?} forecasts \
                     in {HORIZON_PASSES} passes",
                    definition.model
                )));
            }
        }
        // Each forecast's answer: a value per step, while it waits.
        let bytes = (series.len() as u64).saturating_mul(horizon.get() as u64) * 8;
        if self.queued_bytes + bytes > self.queue_bytes {
            self.cancel_abandoned();
        }
        if self.queued_bytes + bytes > self.queue_bytes {
            return Err(ServerError::Busy(format!(
                "{} MiB of queries are waiting already; retry shortly",
                self.queued_bytes >> 20
            )));
        }
        let request = self.node.submit_forecast(
            id,
            &series,
            before.as_deref(),
            horizon,
            groups.as_deref(),
        )?;
        Ok(Submitted {
            request,
            table: id,
            bytes,
        })
    }

    /// Brings the memory gauges up to date: the cache and row gauges, which
    /// shards share, so each adds only how its own changed, and each table's
    /// index.
    fn report_cache(&mut self) {
        for entry in self.tables.values() {
            if let Ok(bytes) = self.node.index_bytes(entry.id) {
                entry.metrics.index_bytes.set(bytes);
            }
            if let Ok(Some(stats)) = self.node.region_stats(entry.id) {
                let metrics = &entry.metrics;
                metrics.region_hits.set(stats.hits);
                metrics.region_stale.set(stats.stale);
                metrics.region_misses.set(stats.misses);
                metrics.region_fresh.set(stats.fresh);
                metrics.rebuilds.set(stats.rebuilds);
                metrics.rebuilt_rows.set(stats.rebuilt_rows);
                let micros = u64::try_from(stats.overdue.as_micros()).unwrap_or(u64::MAX);
                metrics.rebuild_overdue_micros.set(micros);
            }
        }
        for (index, tier) in [Tier::Device, Tier::Host, Tier::Disk]
            .into_iter()
            .enumerate()
        {
            let bytes = self.node.cache_bytes(tier);
            let gauge = &self.metrics.cache_bytes[index];
            let reported = &mut self.cache_reported[index];
            gauge.add(bytes);
            gauge.sub(*reported);
            *reported = bytes;
        }
        let rows = self.node.row_bytes();
        self.metrics.row_bytes.add(rows);
        self.metrics.row_bytes.sub(self.rows_reported);
        self.rows_reported = rows;
    }

    /// Takes a request out of the queue, for its answer.
    fn take(&mut self, request: RequestId) -> Option<Pending> {
        let pending = self.pending.remove(&request)?;
        self.queued_bytes -= pending.bytes;
        self.ephemeral.unpin(pending.table, Instant::now());
        Some(pending)
    }

    /// Queues a request for its answer, pinning the ephemeral table it reads.
    fn queue_pending(&mut self, submitted: Submitted, reply: Reply<Predictions>, trace: String) {
        let Submitted {
            request,
            table,
            bytes,
        } = submitted;
        self.queued_bytes += bytes;
        self.ephemeral.pin(table);
        self.pending.insert(
            request,
            Pending {
                reply,
                trace,
                table,
                bytes,
                arrived: Instant::now(),
            },
        );
    }

    /// Queues a query a table took, or answers why it did not take it.
    fn admit(
        &mut self,
        submitted: Result<Submitted, ServerError>,
        reply: Reply<Predictions>,
        trace: String,
    ) {
        match submitted {
            Ok(submitted) => self.queue_pending(submitted, reply, trace),
            Err(error) => {
                let _ = reply.send(Err(error));
            }
        }
    }

    fn entry(&self, table: &str) -> Result<TableId, ServerError> {
        self.tables
            .get(table)
            .map(|entry| entry.id)
            .ok_or_else(|| ServerError::NotFound(format!("table {table:?} does not exist")))
    }

    /// A table to write to, predict from or ship to, unless it is out of service.
    fn usable(&self, table: &str) -> Result<TableId, ServerError> {
        let id = self.entry(table)?;
        match self.broken.get(&id) {
            Some(reason) => Err(ServerError::Internal(format!(
                "table {table:?} is out of service until the server restarts: {reason}"
            ))),
            None => Ok(id),
        }
    }

    /// Takes a table out of service until the server restarts.
    fn break_table(&mut self, id: TableId, reason: String) {
        if self.broken.contains_key(&id) {
            return;
        }
        let name = self.names.get(&id).map_or("", |name| name.as_str());
        tracing::error!(table = name, "taking the table out of service: {reason}");
        self.metrics.tables_broken.add(1);
        self.broken.insert(id, reason);
    }

    /// Takes a table out of service if its log refuses writes now.
    fn check_poisoned(&mut self, table: &str) {
        let Some(entry) = self.tables.get(table) else {
            return;
        };
        if let Some(reason) = entry.log.poisoned() {
            let reason = format!("its log takes no more writes ({reason})");
            self.break_table(entry.id, reason);
        }
    }

    /// What the worker measures about a model for a feature count, asked once.
    async fn profile(&mut self, model: &str, features: usize) -> Result<ModelProfile, ServerError> {
        let key = (model.to_string(), features);
        if let Some(profile) = self.profiles.get(&key) {
            return Ok(profile.clone());
        }
        let request = WorkerRequest::Profile {
            model: model.to_string(),
            features,
        };
        let reply = self.ask_worker(&request).await?;
        let WorkerReply::Profile(profile) = reply else {
            return Err(ServerError::Internal(format!(
                "the worker answered a profile with {reply:?}"
            )));
        };
        self.profiles.insert(key, profile.clone());
        Ok(profile)
    }

    /// Asks the worker how many rows the fits of tables waiting for them take,
    /// once per model and feature count, and sets them.
    ///
    /// Returns whether any table waited. A worker that cannot say leaves the
    /// table fitting on no more rows than it could before.
    async fn measure_fits(&mut self) -> bool {
        let tables = self.node.take_unmeasured();
        for &table in &tables {
            // A scratch table fits as its partition's table does; a request
            // context is an ephemeral table, which `tables` does not hold.
            let Some(name) = self.name_of(table) else {
                continue;
            };
            let Some(definition) = self.definition(table, name) else {
                continue;
            };
            let key = (definition.model.clone(), definition.model_features());
            let limits = if let Some(&limits) = self.fit_limits.get(&key) {
                limits
            } else {
                let request = WorkerRequest::FitLimits {
                    model: key.0.clone(),
                    features: key.1,
                };
                let rows = |rows: Option<u64>| rows.map_or(usize::MAX, |rows| rows as usize);
                match self.ask_worker(&request).await {
                    Ok(WorkerReply::FitLimits {
                        state_rows,
                        fused_rows,
                    }) => {
                        let limits = (rows(state_rows), rows(fused_rows));
                        self.fit_limits.insert(key, limits);
                        limits
                    }
                    reply => {
                        tracing::warn!(
                            model = key.0.as_str(),
                            "the worker could not measure the model's fits ({reply:?}); \
                             they take no more rows than before"
                        );
                        let bound = self
                            .node
                            .spec(table)
                            .map_or(usize::MAX, |spec| spec.sizes.fit_bound);
                        (bound, bound)
                    }
                }
            };
            let _ = self.node.set_fit_limits(table, limits.0, limits.1);
        }
        !tables.is_empty()
    }

    /// Sends the worker a request and reads its reply.
    async fn ask_worker(&mut self, request: &WorkerRequest) -> Result<WorkerReply, ServerError> {
        let frame = self
            .worker
            .request_within(&request.frame(), self.step_timeout)
            .await
            .map_err(ServerError::Internal)?;
        WorkerReply::of(&frame).map_err(|error| ServerError::Internal(error.0))
    }

    /// Asks the worker whether its model can fit a table of this task and
    /// width, so a table it would refuse is refused before it exists.
    async fn check_model(&mut self, definition: &TableDefinition) -> Result<(), ServerError> {
        let request = WorkerRequest::Check {
            model: definition.model.clone(),
            classes: definition.classes,
            features: definition.model_features(),
            series: definition
                .series
                .as_ref()
                .map_or(1, |series| match series.group {
                    crate::definition::GroupSpec::Alone => 1,
                    crate::definition::GroupSpec::Random { size, .. }
                    | crate::definition::GroupSpec::Given { size } => {
                        usize::try_from(size).unwrap_or(usize::MAX)
                    }
                }),
        };
        match self.ask_worker(&request).await? {
            WorkerReply::Ok => Ok(()),
            WorkerReply::Invalid { message } => Err(ServerError::Invalid(format!(
                "model {:?} cannot serve this table: {message}",
                definition.model
            ))),
            reply => Err(ServerError::Internal(format!(
                "the worker answered a check with {reply:?}"
            ))),
        }
    }

    /// Checks a table against its model and serves it on the node and the
    /// worker, without a log yet; returns its id and prediction width.
    async fn serve(
        &mut self,
        definition: &TableDefinition,
        check: bool,
    ) -> Result<(TableId, u64, Option<Vec<f64>>), ServerError> {
        let name = &definition.name;
        if self.tables.contains_key(name) {
            return Err(ServerError::Exists(format!("table {name:?} exists")));
        }
        let (spec, outputs, levels) = self.spec(definition, check).await?;
        let rows = self.directory.join(format!("{}.rows", definition.name));
        let creation = definition
            .created
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let sealer = Arc::new(RowSealer::new(&self.key, &definition.name, &creation));
        let row_ids = definition.config().map_err(ServerError::Invalid)?.row_ids;
        let id = self.node.create_table_on_disk(spec, &rows, sealer)?;
        if let Some(row_ids) = row_ids {
            self.node.set_row_ids(id, row_ids)?;
        }
        self.add_to_worker(id, definition).await?;
        Ok((id, outputs, levels))
    }

    /// Serves a table made for one request alone, in memory and without a log.
    async fn serve_scratch(
        &mut self,
        definition: &TableDefinition,
    ) -> Result<TableId, ServerError> {
        let (spec, _, _) = self.spec(definition, false).await?;
        let id = self.node.create_table(spec)?;
        self.add_to_worker(id, definition).await?;
        Ok(id)
    }

    /// Has the worker serve a table the node serves, or takes it off the node.
    async fn add_to_worker(
        &mut self,
        id: TableId,
        definition: &TableDefinition,
    ) -> Result<(), ServerError> {
        if let Err(error) = self
            .worker
            .request_within(&add_table(id, definition).frame(), self.step_timeout)
            .await
        {
            let _ = self.node.remove_table(id);
            return Err(ServerError::Internal(error));
        }
        Ok(())
    }

    /// Has the worker serve a table the node serves without waiting for its
    /// answer, so tables made while steps are in flight do not queue behind
    /// them, and tables made in one tick go to the worker in one frame;
    /// `settle_controls` reads the answer.
    fn send_to_worker(&mut self, id: TableId, definition: &TableDefinition) {
        let reply = self.worker.pipeline.control(add_table(id, definition));
        self.controls.push_back((Control::Add(id), reply));
    }

    /// Sends the worker the controls this tick queued, in one frame; if the
    /// worker stopped, their replies fail and the restart serves the tables
    /// again.
    fn flush_controls(&mut self) {
        if let Err(error) = self.worker.pipeline.flush() {
            tracing::error!("sending tables and tiers' changes to the worker: {error}");
        }
    }

    /// Reads the worker's answers to tables added and removed, in the order
    /// sent, as far as they came. A table the worker refused to add leaves
    /// the shard, failing the requests that wait on it.
    async fn settle_controls(&mut self) {
        while let Some((_, reply)) = self.controls.front_mut() {
            let answer = match reply.try_recv() {
                Ok(answer) => answer,
                Err(oneshot::error::TryRecvError::Empty) => break,
                Err(oneshot::error::TryRecvError::Closed) => Err("the worker stopped".to_string()),
            };
            let (control, _) = self.controls.pop_front().expect("found above");
            let Err(error) = answer else {
                continue;
            };
            match control {
                Control::Add(id) => {
                    tracing::error!(table = id.0, "adding a table to the worker: {error}");
                    self.drop_unserved(id, &ServerError::Internal(error)).await;
                }
                Control::Remove(id) => {
                    tracing::error!(table = id.0, "removing a table from the worker: {error}");
                }
            }
        }
    }

    /// Tells the routes a context's table left this shard.
    fn dropped(&self, table: &Ephemeral) {
        if let Some((routes, shard)) = &self.routes {
            routes.dropped(&table.fingerprint, *shard);
        }
    }

    /// Takes an ephemeral table the worker could not serve off the shard,
    /// failing the requests that wait on it.
    async fn drop_unserved(&mut self, id: TableId, error: &ServerError) {
        if let Some(table) = self.ephemeral.remove(id) {
            self.metrics.ephemeral_tables.sub(1);
            self.dropped(&table);
        }
        let waiting: Vec<RequestId> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.table == id)
            .map(|(&request, _)| request)
            .collect();
        for request in waiting {
            self.node.cancel(request);
            if let Some(pending) = self.take(request) {
                let _ = pending.reply.send(Err(error.clone()));
            }
        }
        if let Ok(dropped) = self.node.remove_table(id) {
            self.send_drops(&dropped).await;
        }
    }

    /// A table's spec on the node, checked against its model, with its
    /// prediction width.
    async fn spec(
        &mut self,
        definition: &TableDefinition,
        check: bool,
    ) -> Result<(TableSpec, u64, Option<Vec<f64>>), ServerError> {
        self.worker_up()?;
        let model = self
            .worker
            .models
            .iter()
            .position(|candidate| *candidate == definition.model)
            .ok_or_else(|| {
                ServerError::Invalid(format!("model {:?} is not served", definition.model))
            })?;
        let config = definition.config().map_err(ServerError::Invalid)?;
        let settings = definition.series_settings().map_err(ServerError::Invalid)?;
        if let Some(roles) = &definition.roles {
            roles
                .check(definition.classes)
                .map_err(ServerError::Invalid)?;
        }
        if let Some(route) = &definition.route_by {
            route
                .check(&definition.columns)
                .map_err(ServerError::Invalid)?;
        }
        // A table of series may hold targets alone, as a local one may.
        if definition.series.is_none() || !definition.columns.is_empty() {
            columns::check(&definition.columns, MAX_FEATURES).map_err(ServerError::Invalid)?;
        }
        if definition.features() > MAX_FEATURES {
            return Err(ServerError::Invalid(format!(
                "a table's columns become at most {MAX_FEATURES} model features, \
                 each datetime column {}",
                columns::DATETIME_PARTS
            )));
        }
        let profile = self
            .profile(&definition.model, definition.model_features())
            .await?;
        let ModelProfile {
            bytes_per_row,
            representation_bytes: bytes_per_representation,
            regression,
            class_slots,
            outputs,
            context_rows,
            state_bytes,
            categorical_state_bytes,
            column_summary,
            unlabeled_context,
            fit_bound,
            ..
        } = profile;
        let served = &definition.model;
        let own = config.summary == SummaryScope::Context;
        if own && !column_summary {
            return Err(ServerError::Invalid(format!(
                "model {served:?} has no column summary to fit per context; \
                 create the table with the table summary"
            )));
        }
        match (settings.is_some(), profile.series) {
            (true, false) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} is not a forecaster; create the table without series"
                )));
            }
            (false, true) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} is a forecaster; create a table of series with it"
                )));
            }
            _ => {}
        }
        if settings.is_some() && config.task != Task::Values {
            return Err(ServerError::Invalid(
                "a table of series predicts values; create it without classes".to_string(),
            ));
        }
        match (config.task, regression) {
            (Task::Classes(_), true) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} predicts values; create the table without classes"
                )));
            }
            (Task::Values, false) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} predicts classes; give the table its number of classes"
                )));
            }
            (Task::Classes(1), false) => {
                return Err(ServerError::Invalid(
                    "a classifier tells at least 2 classes apart, not 1 class; a regressor's \
                     table takes none"
                        .to_string(),
                ));
            }
            (Task::Classes(classes), false) if u64::from(classes) > class_slots => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} serves at most {class_slots} classes, not {classes}"
                )));
            }
            _ => {}
        }
        if check {
            self.check_model(definition).await?;
        }
        if profile.series {
            check_quantiles(profile.quantiles.as_deref(), outputs)?;
        }
        let series = settings
            .map(|settings| -> Result<SeriesSpec, ServerError> {
                let max_rows = profile
                    .max_rows
                    .map(|rows| profile_steps(rows, "positive cap on rows"))
                    .transpose()?;
                let per_member = 1 + definition.model_features();
                let size = settings.group.size();
                if let Some(cap) = max_rows
                    && size > 1
                    && size.saturating_mul(per_member) > cap.get()
                {
                    return Err(ServerError::Invalid(format!(
                        "a group of {size} series takes {} rows with the covariates, past \
                         the model's {cap} per group",
                        size * per_member
                    )));
                }
                Ok(SeriesSpec {
                    step: settings.step,
                    patch: profile_steps(profile.patch, "patch")?,
                    context: profile_steps(profile.context_limit, "context limit")?,
                    max_horizon: horizon_limit(
                        profile_steps(profile.output_patch, "output patch")?,
                        profile
                            .pass_limit
                            .map(|steps| profile_steps(steps, "pass limit"))
                            .transpose()?,
                    ),
                    max_rows,
                    known: settings.known,
                    group: settings.group,
                    neighbors: Neighbors::Targets,
                    reusable: profile.reusable,
                    extendable: profile.extendable,
                    patch_bytes: profile.patch_bytes,
                })
            })
            .transpose()?;
        // A table of series plans its own units: no retrieval, regions or pages.
        let spec = TableSpec {
            model: ModelId(u32::try_from(model).expect("few models")),
            features: definition.features(),
            retrieval: columns::retrieval(&definition.columns),
            categorical: columns::categorical(&definition.columns),
            categories: config.categories,
            unlabeled_context,
            task: config.task,
            page_size: config.page_size,
            strategy: if series.is_some() {
                Strategy::Exact
            } else {
                config.strategy
            },
            time_bucket: config.time_bucket,
            refit: config.refit,
            fit_rows: config.fit_rows,
            context_rows: context_rows.and_then(NonZeroU64::new),
            past_cap: config.past_cap,
            regions: if series.is_some() {
                None
            } else {
                config.regions
            },
            scale_rows: config.scale_rows,
            summary: config.summary,
            sizes: EntrySizes {
                bytes_per_row,
                bytes_per_representation,
                bucket_minimum: self.bucket_minimum,
                state_bytes: if !own {
                    0
                } else if columns::categorical(&definition.columns).is_empty() {
                    state_bytes
                } else {
                    categorical_state_bytes
                },
                state_rows: usize::MAX,
                fused_rows: usize::MAX,
                fit_bound: fit_bound.map_or(usize::MAX, |rows| rows as usize),
            },
            series,
        };
        let outputs = if regression {
            outputs
        } else {
            u64::from(definition.classes)
        };
        // The level of each prediction column: a forecaster's own, a tabular
        // regressor's evenly spaced, and none for a classifier.
        let levels = if !regression {
            None
        } else if let Some(quantiles) = profile.quantiles {
            Some(quantiles)
        } else {
            Some(implied_levels(outputs))
        };
        Ok((spec, outputs, levels))
    }

    /// Stops serving a table on the node and the worker, as `serve` began.
    async fn unserve(&mut self, id: TableId) {
        self.recent.remove(&id);
        self.lift_quarantine(id);
        if self.broken.remove(&id).is_some() {
            self.metrics.tables_broken.sub(1);
        }
        if let Ok(dropped) = self.node.remove_table(id) {
            self.send_drops(&dropped).await;
        }
        if self.down.is_some() {
            // The worker starts again with only the tables the shard still serves.
            return;
        }
        let reply = self
            .worker
            .pipeline
            .control(WorkerRequest::RemoveTable { table: id.0 });
        self.controls.push_back((Control::Remove(id), reply));
    }

    /// Serves a new table with a new log under these terms; returns its
    /// quantile levels, if it predicts values.
    async fn create(
        &mut self,
        definition: &TableDefinition,
        terms: Terms,
    ) -> Result<Option<Vec<f64>>, ServerError> {
        let (id, outputs, levels) = self.serve(definition, true).await?;
        let path = self.log_path(&definition.name);
        let log = match Log::create(&path, &self.key, definition, terms) {
            Ok(log) => log,
            Err(error) => {
                self.unserve(id).await;
                return Err(error.into());
            }
        };
        self.add_entry(definition, id, outputs, levels.clone(), log, None);
        Ok(levels)
    }

    /// Serves a table from its log at startup, applying each record as it is
    /// read. A table that fails to restore is served nowhere, and its log
    /// stays, or is set aside if it is damaged.
    async fn restore(&mut self, mut replay: Replay) -> Result<Option<Vec<f64>>, ServerError> {
        let mut definition = replay.definition().clone();
        let (id, outputs, levels) = self.serve(&definition, false).await?;
        let path = self.log_path(&definition.name);
        let mut vocabularies = vec![Vocabulary::default(); vocabularies_of(&definition)];
        let mut series = SeriesKeys::default();
        let mut replayed = Ok(());
        while let Some(record) = replay.next() {
            let record = match record {
                Ok(record) => record,
                Err(error) => {
                    replayed = Err(error);
                    break;
                }
            };
            if let Record::Strategy(strategy) = &record {
                definition.strategy = strategy.clone();
            }
            if let Some(strategy) = snapshot_strategy(&record) {
                take_strategy(&mut definition, strategy);
            }
            learn(
                &mut vocabularies,
                record.words(),
                matches!(record, Record::Snapshot(_)),
            );
            series.track(&record);
            // The worker starts empty, so a refit's drops have nothing to drop.
            if let Err(error) = self.apply_logged(id, record, replay.length()) {
                self.unserve(id).await;
                return Err(error);
            }
        }
        match replayed.and_then(|()| replay.finish()) {
            Ok(log) => {
                self.unsaved_heads.insert(definition.name.clone());
                self.add_entry(
                    &definition,
                    id,
                    outputs,
                    levels.clone(),
                    log,
                    Some(Learned {
                        vocabularies,
                        series,
                    }),
                );
                Ok(levels)
            }
            Err(error) => {
                self.unserve(id).await;
                if error.damaged() {
                    wal::set_aside_damaged(&path, &error);
                }
                Err(error.into())
            }
        }
    }

    fn log_path(&self, table: &TableName) -> PathBuf {
        self.directory.join(format!("{table}.log"))
    }

    /// Records a table the node and worker serve, with its log.
    fn add_entry(
        &mut self,
        definition: &TableDefinition,
        id: TableId,
        outputs: u64,
        levels: Option<Vec<f64>>,
        log: Log,
        learned: Option<Learned>,
    ) {
        let Learned {
            vocabularies,
            series,
        } = learned.unwrap_or_else(|| Learned {
            vocabularies: vec![Vocabulary::default(); vocabularies_of(definition)],
            series: SeriesKeys::default(),
        });
        self.names.insert(id, definition.name.clone());
        self.tables.insert(
            definition.name.clone(),
            Entry {
                id,
                log,
                definition: definition.clone(),
                outputs,
                levels,
                vocabularies,
                series,
                compact_after: 0,
                compact_wait: COMPACT_RECORDS,
                metrics: self.metrics.table(definition.name.as_str()),
            },
        );
    }

    /// A served table's name, a scratch table's partition's, or an
    /// ephemeral table's.
    fn name_of(&self, id: TableId) -> Option<&TableName> {
        self.names
            .get(&id)
            .or_else(|| self.scratch.get(&id).map(|scratch| &scratch.content.0))
            .or_else(|| self.ephemeral_of(id).map(|table| &table.definition.name))
    }

    fn ephemeral_of(&self, id: TableId) -> Option<&Ephemeral> {
        self.ephemeral.get(id)
    }

    /// A table's metrics by its id in the node; a scratch table's are its partition's.
    fn table_metrics(&self, id: TableId) -> Option<&TableMetrics> {
        match self.names.get(&id) {
            Some(name) => self.tables.get(name).map(|entry| &*entry.metrics),
            None => self
                .scratch
                .get(&id)
                .map(|scratch| &*scratch.metrics)
                .or_else(|| self.ephemeral_of(id).map(|table| &*table.metrics)),
        }
    }

    /// Stops serving a table: its log goes first, so a failure changes
    /// nothing, then the node forgets it, then the worker.
    async fn release(&mut self, table: &TableName, keep: bool) -> Result<(), ServerError> {
        if let Some(compacting) = self.compacting.remove(table) {
            let gone = ServerError::NotFound(format!("table {table:?} was dropped"));
            for queued in compacting.held {
                let _ = queued.reply.send(Err(gone.clone()));
            }
            for reply in compacting.replies {
                let _ = reply.send(Err(gone.clone()));
            }
        }
        let id = self.entry(table)?;
        self.settle().await;
        let term = self.tables[table].log.term().term;
        let path = self.log_path(table);
        let removed = if keep {
            // Not a .log, so restarts leave it alone.
            wal::set_aside(&path, &format!("diverged-{term}")).map(drop)
        } else {
            wal::remove(&path)
        };
        removed.map_err(|error| ServerError::Internal(format!("removing the log: {error}")))?;
        self.tables.remove(table);
        self.unsaved_heads.remove(table);
        self.names.remove(&id);
        self.unserve(id).await;
        Ok(())
    }

    /// Runs every pending and in-flight step to its end, so a table about to
    /// leave has no entries held and no requests waiting.
    async fn settle(&mut self) {
        loop {
            while self.dispatch().await {}
            if self.in_flight.is_empty() {
                break;
            }
            self.drain().await;
        }
    }

    /// Stops serving a table whose log another shard will restore: its
    /// requests finish here first, and its log stays as it is.
    async fn hand_off(&mut self, table: &TableName) -> Result<(), ServerError> {
        if self.compacting.contains_key(table) {
            return Err(ServerError::Busy(format!("table {table:?} is compacting")));
        }
        let id = self.entry(table)?;
        self.settle().await;
        let entry = self.tables.remove(table).expect("found above");
        if let Err(error) = entry.log.close() {
            tracing::error!(table = table.as_str(), "closing the log's head: {error}");
        }
        self.unsaved_heads.remove(table);
        self.names.remove(&id);
        self.unserve(id).await;
        Ok(())
    }

    /// Whether the next step waits for the one in flight: new contexts
    /// that arrive while it runs go together in the step after it, side by
    /// side, rather than each in a step of its own behind it.
    fn holds_new_contexts(&self) -> bool {
        !self.in_flight.is_empty() && self.node.only_new_contexts_wait()
    }
}

/// Passes a request context's predictions on with its table's levels, or
/// stops waiting once the caller goes away, which closes the queued reply.
async fn relay(
    answer: oneshot::Receiver<Result<Predictions, ServerError>>,
    reply: Reply<Inferred>,
    levels: Option<Vec<f64>>,
) {
    let mut reply = reply;
    tokio::select! {
        answered = answer => {
            let answered = answered
                .unwrap_or_else(|_| Err(ServerError::Unavailable("the shard has stopped".to_string())));
            let _ = reply.send(answered.map(|predictions| Inferred { predictions, levels }));
        }
        () = reply.closed() => {}
    }
}

/// Why a table went out of service after a synced record failed to apply.
/// A table's rows in parts of at most `per_record` rows, in order; one part,
/// of no rows, for an empty table.
/// Applies a compaction's record to a table: one with a layout rebuilds the
/// table exactly, one without, from a log compacted before layouts were kept,
/// makes the table hold its rows.
/// The strategy a snapshot's layout carries, if it is one with a layout.
fn snapshot_strategy(record: &Record) -> Option<Strategy> {
    match record {
        Record::Snapshot(rows) => rows.table.as_ref().and_then(|table| table.strategy),
        _ => None,
    }
}

/// Gives the definition the strategy a snapshot carried, keeping the
/// definition's own spelling of it when it is the same strategy.
fn take_strategy(definition: &mut TableDefinition, strategy: Strategy) {
    if definition.strategy.strategy() != strategy {
        definition.strategy = StrategySpec::of(strategy);
    }
}

fn unapplied(error: &ServerError) -> String {
    format!("its log holds a record its rows could not take ({error})")
}

/// How many vocabularies a table keeps: one per column, and one for labels.
fn vocabularies_of(definition: &TableDefinition) -> usize {
    definition.columns.len() + 1
}

/// Adds words to vocabularies, or with `reset` makes them exactly those words.
fn learn(vocabularies: &mut [Vocabulary], words: &[Words], reset: bool) {
    if reset {
        for vocabulary in vocabularies.iter_mut() {
            vocabulary.clear();
        }
    }
    for added in words {
        if let Some(vocabulary) = vocabularies.get_mut(added.column) {
            vocabulary.extend(added.values.iter().cloned());
        }
    }
}

/// A hash of a scratch table's rows: keys, features, words and targets.
fn content_hash(
    keys: &[u128],
    features: &[f32],
    words: &[(usize, Vec<Option<String>>)],
    targets: &Targets,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    keys.hash(&mut hasher);
    for value in features {
        value.to_bits().hash(&mut hasher);
    }
    words.hash(&mut hasher);
    match targets {
        Targets::Classes(classes) => classes.hash(&mut hasher),
        Targets::Values(values) => {
            for value in values {
                value.to_bits().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Each categorical column with the model feature it codes into.
fn categorical_offsets(definition: &TableDefinition) -> Vec<(usize, usize)> {
    definition
        .columns
        .iter()
        .zip(columns::offsets(&definition.columns))
        .enumerate()
        .filter(|(_, (column, _))| column.kind == columns::ColumnKind::Categorical)
        .map(|(index, (_, offset))| (index, offset))
        .collect()
}

/// Writes a categorical column's codes into its model feature of row-major features.
fn fill(
    features: &mut [f32],
    definition: &TableDefinition,
    column: usize,
    codes: &[f32],
) -> Result<(), ServerError> {
    let width = definition.features();
    let offset = columns::offsets(&definition.columns)[column];
    if codes.len() * width != features.len() {
        return Err(ServerError::Internal(format!(
            "column {column} has {} values for {} rows",
            codes.len(),
            features.len() / width.max(1)
        )));
    }
    for (row, code) in codes.iter().enumerate() {
        features[row * width + offset] = *code;
    }
    Ok(())
}
