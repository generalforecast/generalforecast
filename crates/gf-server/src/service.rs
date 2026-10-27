// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The Arrow Flight API.
//!
//! - Actions: `create_table`, `describe`, `delete`, `refit`, `compact`, `drop_table`,
//!   `list_tables`, `list_models` and `health`, with JSON bodies.
//! - `DoPut` to `insert/<table>`, `update/<table>` or `upsert/<table>`: batches
//!   with a `key` column of 128-bit keys (16 little-endian bytes each, or
//!   uint64 or int64 integers), float32 feature columns, a `target` column
//!   (int32 or float32), and optionally a `time` column (int64). `DoPut` to
//!   `delete/<table>` takes batches with a `key` column alone; with the
//!   `gf-missing-ok` metadata, keys the table lacks are skipped.
//! - `DoExchange` on `predict/<table>`: float32 feature batches in, predictions
//!   out; an optional `before` column (int64) limits each query to earlier rows.
//!
//! Routers keep replicas with three more calls:
//!
//! - `DoGet` with a ticket `{"table", "from"}` streams the table's log entries
//!   from that sequence number, each in parts of at most 1 MiB, one per message
//!   body, with metadata `{"sequence": n, "last": bool}`. A compacted log starts
//!   again at its snapshot.
//! - `DoPut` to `replicate/<table>` applies such a stream. Entry 0, the
//!   definition, creates the table, and a snapshot compacts the replica's log.
//! - Actions `log_lengths` (every table's log length and term), `release` (drop
//!   a table whose copies live elsewhere, or set aside a diverged one),
//!   `promote` (make this copy the primary under a later term) and `lease`
//!   (let this node take writes for some seconds more).
//!
//! Logs hold raw rows, so only replication tokens may make these calls, and
//! `replicate/<table>` also needs the `gf-shipment` mark that routers set only
//! on their own shipments.
//!
//! Every call needs `authorization: Bearer <token>` with a configured token, and
//! reaches only the tables that token's scope names.
//!
//! A write is logged only while this node holds its lease, if writes need one,
//! and while its table is at the term a router named in `gf-term`, if any.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use arrow_array::builder::{FixedSizeBinaryBuilder, ListBuilder};
use arrow_array::{
    Array, ArrayRef, FixedSizeBinaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt64Array,
};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use arrow_schema::{DataType, Field, Schema};
use futures::stream::Peekable;
use futures::{Stream, StreamExt, TryStreamExt, stream};
use gf_core::node::{Targets, Task, UNLABELED};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{OwnedRwLockReadGuard, RwLock, mpsc, oneshot, watch};
use tonic::service::interceptor::InterceptedService;
use tonic::{Request, Response, Status, Streaming};

use crate::action::{
    Answer, CopyLength, DeleteAction, FlightAction, LeaseAnswer, LogLengths, PutCommand,
    check_write_id, profile_directory, unix_ms,
};
use crate::auth::{
    CREATED, MISSING_OK, QUANTILES, SHIPMENT, Scope, TERM, TERMS, Tokens, WRITE_ID, bearer,
};
use crate::budget::Charge;
/// Where the budget was before it served the Models API too, kept for the
/// crates that embed this server.
pub use crate::budget::RequestBudget;
use crate::config::{Assign, PlacementConfig};
use crate::error::{ServerError, exists};
use crate::metrics::{Histogram, Metrics};
use crate::name::TableName;
use crate::partition;
use crate::protocol::{Predictions, StrategySpec};
use crate::quantiles::{self, quantile_values};
use crate::shard::{Command, Description, Fence, Incoming, Lease, Lineage, LogLength, Write};
use crate::trace::{Trace, traced};
use crate::wal::{
    CreationId, Durability, GivenKey, LogEntry, Rows, SeriesKey, TableDefinition, Terms, decode,
};
use gf_core::columns::{self, Cells, Column, ColumnKind, Strings};

type Boxed<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// The largest message either end of a Flight call takes. Clients split
/// writes into batches below it, and log entries travel in parts.
pub const MESSAGE_BYTES: usize = 64 << 20;

/// The most bytes of a log entry in one message.
const ENTRY_PART: usize = 1 << 20;

/// The most bytes of one log entry, as the log's length prefix allows.
const ENTRY_BYTES: usize = u32::MAX as usize;

/// The most bytes of Arrow data one write or prediction request carries.
const REQUEST_BYTES: usize = 256 << 20;

/// Where a table lives, its columns, and how its predictions are labeled.
#[derive(Clone, Debug)]
pub struct Placement {
    pub shard: usize,
    pub task: Task,
    pub columns: Arc<Vec<Column>>,
    /// The table's creation id, which every write's answer carries, so a
    /// router always knows which creation of the name a copy holds.
    pub created: Option<CreationId>,
    /// The quantile level of each prediction column, for a table predicting values.
    pub levels: Option<Arc<Vec<f64>>>,
    /// The grid's step and default horizon, for a table of series.
    pub series: Option<SeriesPlacement>,
}

/// What the service needs of a table of series to take and answer forecasts.
#[derive(Clone, Copy, Debug)]
pub struct SeriesPlacement {
    /// The grid's step, in the table's time unit.
    pub step: i64,
    /// Steps a forecast covers when its call names none.
    pub horizon: Option<usize>,
}

impl SeriesPlacement {
    /// The settings of a table's series definition, if it has one.
    #[must_use]
    pub fn of(definition: &TableDefinition) -> Option<Self> {
        definition.series.as_deref().map(|series| Self {
            step: i64::try_from(series.step).unwrap_or(i64::MAX),
            horizon: series
                .horizon
                .and_then(|horizon| usize::try_from(horizon).ok()),
        })
    }
}

/// Marks a request made with a replication token.
#[derive(Clone, Copy, Debug)]
struct Replication;

/// Refuses a replication call made without a replication token.
fn replication<T>(request: &Request<T>) -> Result<(), Status> {
    if request.extensions().get::<Replication>().is_some() {
        Ok(())
    } else {
        Err(Status::permission_denied(
            "reading or applying table logs needs a replication token",
        ))
    }
}

/// Refuses log entries that are not a router's shipment: routers forward
/// client writes under a replication token too, but never with this mark.
fn shipment<T>(request: &Request<T>) -> Result<(), Status> {
    replication(request)?;
    if request.metadata().contains_key(SHIPMENT) {
        Ok(())
    } else {
        Err(Status::permission_denied(
            "log entries are applied only from a router's shipment",
        ))
    }
}

/// Tables being created: the shard each goes to, and what changes once it is
/// placed or has failed. Their names are taken meanwhile.
type Creating = HashMap<TableName, (usize, watch::Receiver<()>)>;

/// A table name taken while its creation runs. Dropping it gives the name back
/// and wakes the creations waiting on it, however the creation ended.
struct Reservation {
    creating: Arc<Mutex<Creating>>,
    name: TableName,
    _placed: watch::Sender<()>,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.creating
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.name);
    }
}

/// The Flight service over a server's shards.
#[derive(Clone)]
pub struct Service {
    shards: Arc<Vec<mpsc::Sender<Command>>>,
    placement: Arc<RwLock<HashMap<TableName, Placement>>>,
    creating: Arc<Mutex<Creating>>,
    metrics: Arc<Metrics>,
    /// This node's write lease, if writes need one.
    lease: Option<Lease>,
    budget: RequestBudget,
    max_tables: usize,
    /// One lock per table: calls hold it to read while they reach the
    /// table's shard, and a move holds it to write.
    routes: Arc<Mutex<HashMap<String, Arc<RwLock<()>>>>>,
    /// What placing tables by load and moving them needs, if the server does.
    mover: Option<Arc<Mover>>,
    /// Where `profile_workers` writes, if it is allowed.
    profiles: Option<PathBuf>,
    /// Times each part of a request, for the `gf_request_*_seconds` metrics.
    timers: bool,
}

/// What a server that places tables by load keeps between rounds.
pub struct Mover {
    tables: PathBuf,
    keys: crate::wal::Keys,
    churn_budget: f64,
    imbalance_floor: f64,
    /// Each shard's worker seconds a second over the last round.
    shard_loads: Mutex<Vec<f64>>,
    /// Each table's worker seconds at the last round.
    seen: Mutex<HashMap<String, f64>>,
    /// The shard of each table placed since the last round.
    fresh: Mutex<HashMap<String, usize>>,
    /// When the last round ran, or the server started.
    last_round: Mutex<Instant>,
}

impl Mover {
    /// Drops what the mover remembers of a table.
    fn forget(&self, table: &str) {
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(table);
        self.fresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(table);
    }
}

/// What a round of moving tables did.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Round {
    pub moved: Vec<(String, usize, usize)>,
    pub imbalance_before: f64,
    pub imbalance_after: f64,
    pub shard_loads: Vec<f64>,
    pub seconds: f64,
}

impl Service {
    /// A service over shards, with tables already placed by a restore.
    #[must_use]
    pub fn new(
        shards: Vec<mpsc::Sender<Command>>,
        placement: HashMap<TableName, Placement>,
        metrics: Arc<Metrics>,
    ) -> Self {
        metrics.tables.set(placement.len() as u64);
        Self {
            shards: Arc::new(shards),
            placement: Arc::new(RwLock::new(placement)),
            creating: Arc::default(),
            metrics,
            lease: None,
            budget: RequestBudget::new(u64::MAX),
            max_tables: usize::MAX,
            routes: Arc::default(),
            mover: None,
            profiles: None,
            timers: false,
        }
    }

    /// Places tables as `config` says; with `assign = "load"`, moves tables
    /// between shards by restoring their logs, kept under `tables`, with `keys`.
    #[must_use]
    pub fn with_placement(
        mut self,
        config: &PlacementConfig,
        tables: PathBuf,
        keys: crate::wal::Keys,
    ) -> Self {
        if config.assign == Assign::Load {
            self.mover = Some(Arc::new(Mover {
                tables,
                keys,
                churn_budget: config.churn_budget,
                imbalance_floor: config.imbalance_floor,
                shard_loads: Mutex::new(vec![0.0; self.shards.len()]),
                seen: Mutex::default(),
                fresh: Mutex::default(),
                last_round: Mutex::new(Instant::now()),
            }));
        }
        self
    }

    fn route(&self, table: &str) -> Arc<RwLock<()>> {
        self.routes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(table.to_string())
            .or_default()
            .clone()
    }

    /// A table's placement, held so the table cannot move until the guard
    /// drops; a call keeps it until its shard has answered, and takes it
    /// after it has read its rows, so a slow upload holds no move up.
    async fn routed(&self, table: &str) -> Result<(Placement, OwnedRwLockReadGuard<()>), Status> {
        let guard = self.locked(table, |lock| lock.read_owned()).await?;
        Ok((self.placement(table).await?, guard))
    }

    /// The table's lock, taken with `take`, once it is the lock the table
    /// has: a drop that ran meanwhile took the lock out of the map, and a
    /// table created since has its own.
    async fn locked<G, F>(
        &self,
        table: &str,
        take: impl Fn(Arc<RwLock<()>>) -> F,
    ) -> Result<G, Status>
    where
        F: Future<Output = G>,
    {
        loop {
            let lock = {
                // Looked up while the placement names the table, so a drop,
                // which forgets the placement before the lock, cannot leave
                // a fresh lock behind for a name no table has.
                let placement = self.placement.read().await;
                if !placement.contains_key(table) {
                    return Err(Status::not_found(format!("table {table:?} does not exist")));
                }
                self.route(table)
            };
            let guard = take(lock.clone()).await;
            let routes = self.routes.lock().unwrap_or_else(PoisonError::into_inner);
            match routes.get(table) {
                Some(current) if Arc::ptr_eq(current, &lock) => return Ok(guard),
                Some(_) => {}
                None => return Err(Status::not_found(format!("table {table:?} does not exist"))),
            }
        }
    }

    /// Moves a table to shard `to`: its calls wait, its shard finishes what
    /// it holds and lets go of the table, and `to` restores it from its log.
    ///
    /// # Errors
    ///
    /// If the server does not move tables, the table or shard does not
    /// exist, or either shard refuses.
    pub async fn move_table(&self, table: &TableName, to: usize) -> Result<f64, Status> {
        let mover = self
            .mover
            .clone()
            .ok_or_else(|| invalid("tables move only with placement.assign = \"load\""))?;
        if to >= self.shards.len() {
            return Err(invalid(format!(
                "the server has {} shards",
                self.shards.len()
            )));
        }
        let started = Instant::now();
        let _moving = self.locked(table, |lock| lock.write_owned()).await?;
        let from = self.placement(table).await?;
        if from.shard == to {
            return Ok(0.0);
        }
        if from.series.is_some() {
            return Err(invalid("a table of series does not move yet"));
        }
        let name = table.clone();
        self.ask(from.shard, |reply| Command::Handoff { table: name, reply })
            .await?;
        let path = mover.tables.join(format!("{table}.log"));
        let restored = async {
            let replay = crate::setup::open_log(&path, &mover.keys)
                .map_err(|error| Status::internal(format!("opening the log: {error}")))?
                .ok_or_else(|| Status::internal("the log did not open"))?;
            self.ask(to, |reply| Command::Restore {
                replay: Box::new(replay),
                reply,
            })
            .await
        };
        let (shard, levels) = match restored.await {
            Ok(levels) => (to, levels),
            Err(error) => {
                // Serve it where it was rather than nowhere.
                tracing::error!(table = table.as_str(), to, "moving a table: {error}");
                let back = async {
                    let replay = crate::setup::open_log(&path, &mover.keys)
                        .map_err(|error| Status::internal(format!("reopening the log: {error}")))?
                        .ok_or_else(|| Status::internal("the log did not reopen"))?;
                    self.ask(from.shard, |reply| Command::Restore {
                        replay: Box::new(replay),
                        reply,
                    })
                    .await
                };
                match back.await {
                    Ok(levels) => (from.shard, levels),
                    Err(error) => {
                        // No shard serves it now. Its log stays, so a restart
                        // brings it back, and a creation of its name fails on
                        // the log rather than replacing it.
                        tracing::error!(
                            table = table.as_str(),
                            from = from.shard,
                            "a moved table is out of service until the server restarts: {error}"
                        );
                        let mut placement = self.placement.write().await;
                        placement.remove(table);
                        self.metrics.tables.set(placement.len() as u64);
                        drop(placement);
                        self.metrics.forget(table);
                        mover.forget(table);
                        return Err(Status::internal(format!(
                            "table {table:?} could not move and is out of service until the \
                             server restarts: {error}"
                        )));
                    }
                }
            }
        };
        if let Some(placed) = self.placement.write().await.get_mut(table) {
            placed.shard = shard;
            placed.levels = levels.map(Arc::new);
        }
        let seconds = started.elapsed().as_secs_f64();
        self.metrics.moves.add(u64::from(shard == to));
        self.metrics.move_seconds.observe(started.elapsed());
        tracing::info!(
            table = table.as_str(),
            from = from.shard,
            to = shard,
            seconds,
            "moved a table"
        );
        if shard == to {
            Ok(seconds)
        } else {
            Err(Status::internal("the table could not move and stayed"))
        }
    }

    /// One round: each table's worker seconds since the last round are its
    /// load, `gf-assign` chooses moves within the churn budget, and they run
    /// one at a time.
    ///
    /// # Errors
    ///
    /// If the server does not move tables.
    pub async fn rebalance_round(&self) -> Result<Round, Status> {
        let mover = self
            .mover
            .clone()
            .ok_or_else(|| invalid("tables move only with placement.assign = \"load\""))?;
        let started = Instant::now();
        let elapsed = {
            let mut last = mover
                .last_round
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            std::mem::replace(&mut *last, started)
                .elapsed()
                .as_secs_f64()
        };
        let gpus = self.shards.len();
        let mut tables: Vec<(TableName, usize)> = self
            .placement
            .read()
            .await
            .iter()
            .filter(|(_, placed)| placed.series.is_none())
            .map(|(name, placed)| (name.clone(), placed.shard))
            .collect();
        tables.sort_by(|a, b| a.0.cmp(&b.0));
        // A move costs the rows the destination replays, which the shard
        // holding the table counts; a table that cannot say sits this round out.
        let mut placed = Vec::with_capacity(tables.len());
        for (name, shard) in tables {
            let table = name.clone();
            match self
                .ask(shard, |reply| Command::Describe { table, reply })
                .await
            {
                Ok(described) => placed.push((name, shard, described.rows)),
                Err(error) => tracing::warn!(table = name.as_str(), "not measured: {error}"),
            }
        }
        let mut loads = Vec::with_capacity(placed.len());
        {
            let mut seen = mover.seen.lock().unwrap_or_else(PoisonError::into_inner);
            for (name, _, rows) in &placed {
                let now = self
                    .metrics
                    .existing_table(name)
                    .map_or(0.0, |counts| counts.gpu_seconds.get());
                let before = seen.insert(name.to_string(), now).unwrap_or(0.0);
                let load = if now >= before { now - before } else { now } / elapsed.max(1e-9);
                loads.push(gf_assign::SliceLoad {
                    load,
                    rows: *rows as f64,
                    median: None,
                });
            }
        }
        let mut shard_loads = vec![0.0; gpus];
        for ((_, shard, _), load) in placed.iter().zip(&loads) {
            shard_loads[*shard] += load.load;
        }
        *mover
            .shard_loads
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = shard_loads.clone();
        mover
            .fresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        let imbalance = |g: &[f64]| {
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            if mean > 0.0 {
                g.iter().cloned().fold(0.0, f64::max) / mean
            } else {
                1.0
            }
        };
        let imbalance_before = imbalance(&shard_loads);
        let mut round = Round {
            moved: Vec::new(),
            imbalance_before,
            imbalance_after: imbalance_before,
            shard_loads,
            seconds: 0.0,
        };
        if placed.is_empty() || gpus < 2 {
            return Ok(round);
        }
        // Each table is one slice; tables are never split, merged or copied.
        let assignment = gf_assign::Assignment {
            generation: 0,
            slices: placed
                .iter()
                .enumerate()
                .map(|(index, (_, shard, _))| gf_assign::Slice {
                    start: index as u64,
                    holders: vec![*shard],
                })
                .collect(),
        };
        let policy = gf_assign::Policy {
            mode: gf_assign::Mode::Free,
            churn_budget: mover.churn_budget,
            imbalance_floor: mover.imbalance_floor,
            min_slices_per_gpu: usize::MAX,
            max_copies: 1,
            ..gf_assign::Policy::default()
        };
        let next = gf_assign::rebalance(&assignment, &loads, gpus, &policy);
        for ((name, from, _), slice) in placed.iter().zip(&next.assignment.slices) {
            let to = slice.holders[0];
            if to != *from {
                match self.move_table(name, to).await {
                    Ok(_) => round.moved.push((name.to_string(), *from, to)),
                    Err(error) => tracing::warn!(table = name.as_str(), "not moved: {error}"),
                }
            }
        }
        // The imbalance left by the moves that ran, not the ones planned.
        let mut after = vec![0.0; gpus];
        for ((name, from, _), load) in placed.iter().zip(&loads) {
            let shard = round
                .moved
                .iter()
                .find(|(moved, _, _)| moved == name.as_str())
                .map_or(*from, |&(_, _, to)| to);
            after[shard] += load.load;
        }
        round.imbalance_after = imbalance(&after);
        round.seconds = started.elapsed().as_secs_f64();
        self.metrics
            .placement_imbalance
            .set((round.imbalance_after * 1000.0) as u64);
        Ok(round)
    }

    /// Runs a round every `every` until the server stops.
    pub async fn rebalance(self, every: Duration) {
        let mut ticks = tokio::time::interval(every);
        ticks.tick().await;
        loop {
            ticks.tick().await;
            match self.rebalance_round().await {
                Ok(round) if !round.moved.is_empty() => {
                    tracing::info!(
                        moved = round.moved.len(),
                        imbalance = round.imbalance_after,
                        "rebalanced"
                    );
                }
                Ok(_) => {}
                Err(error) => tracing::warn!("rebalancing: {error}"),
            }
        }
    }

    /// Refuses creations past `tables` tables.
    #[must_use]
    pub fn with_max_tables(mut self, tables: usize) -> Self {
        self.max_tables = tables;
        self
    }

    /// Holds at most `bytes` of client writes and queries in memory at once.
    #[must_use]
    pub fn with_request_budget(mut self, bytes: u64) -> Self {
        self.budget = RequestBudget::new(bytes);
        self
    }

    /// Without a directory, requests aren't timed and `profile_workers` is refused.
    #[must_use]
    pub fn with_profiling(mut self, directory: Option<PathBuf>) -> Self {
        self.timers = directory.is_some();
        self.profiles = directory;
        self
    }

    /// Laps of a request's parts from `start`, timed if the server times requests.
    fn laps(&self, start: impl FnOnce() -> Instant) -> Laps {
        Laps(self.timers.then(start))
    }

    /// Takes client writes only while a router keeps this node's lease.
    #[must_use]
    pub fn requiring_leases(mut self) -> Self {
        self.lease = Some(Lease::lapsed());
        self
    }

    /// Refuses a client write early once the lease has lapsed; the shard
    /// checks again just before logging it.
    fn leased(&self) -> Result<(), Status> {
        self.lease
            .as_ref()
            .map_or(Ok(()), Lease::check)
            .map_err(ServerError::status)
    }

    /// What a client write must still hold when it is logged: the lease, and
    /// the term a router sent it under.
    fn fence<T>(&self, request: &Request<T>) -> Result<Fence, Status> {
        let term = match request.metadata().get(TERM) {
            Some(term) => Some(
                term.to_str()
                    .ok()
                    .and_then(|term| term.parse().ok())
                    .ok_or_else(|| invalid(format!("{TERM} must be a term number")))?,
            ),
            None => None,
        };
        Ok(Fence {
            lease: self.lease.clone(),
            term,
        })
    }

    /// The service behind a bearer-token check: each token reaches the tables
    /// its scope names, and a replication token reaches their logs too.
    #[must_use]
    pub fn into_server(
        self,
        tokens: Tokens,
        replication_tokens: Tokens,
    ) -> InterceptedService<FlightServiceServer<Self>, impl tonic::service::Interceptor + Clone>
    {
        let tokens = Arc::new(tokens);
        let replication_tokens = Arc::new(replication_tokens);
        let server = FlightServiceServer::new(self)
            .max_decoding_message_size(MESSAGE_BYTES)
            .max_encoding_message_size(MESSAGE_BYTES);
        InterceptedService::new(server, move |mut request: Request<()>| {
            let unauthenticated = || Status::unauthenticated("a valid bearer token is required");
            let token = bearer(&request).ok_or_else(unauthenticated)?;
            let (scope, replicates) = if let Some(scope) = replication_tokens.scope(token) {
                (scope.clone(), true)
            } else if let Some(scope) = tokens.scope(token) {
                (scope.clone(), false)
            } else {
                return Err(unauthenticated());
            };
            if replicates {
                request.extensions_mut().insert(Replication);
            }
            request.extensions_mut().insert(scope);
            Ok(request)
        })
    }

    async fn placement(&self, table: &str) -> Result<Placement, Status> {
        self.placement
            .read()
            .await
            .get(table)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("table {table:?} does not exist")))
    }

    /// Asks the shard serving `table`, holding the table's route until it
    /// answers so the command never crosses a move; answers what the shard
    /// said and where the table was.
    async fn at_table<T>(
        &self,
        table: TableName,
        command: impl FnOnce(TableName, oneshot::Sender<Result<T, ServerError>>) -> Command,
    ) -> Result<(T, Placement), Status> {
        let (placement, _routed) = self.routed(&table).await?;
        let answer = self
            .ask(placement.shard, |reply| command(table, reply))
            .await?;
        Ok((answer, placement))
    }

    async fn ask<T>(
        &self,
        shard: usize,
        command: impl FnOnce(oneshot::Sender<Result<T, ServerError>>) -> Command,
    ) -> Result<T, Status> {
        let (reply, answer) = oneshot::channel();
        self.shards[shard]
            .send(command(reply))
            .await
            .map_err(|_| Status::unavailable("the shard has stopped"))?;
        answer
            .await
            .map_err(|_| Status::unavailable("the shard has stopped"))?
            .map_err(ServerError::status)
    }

    async fn create(&self, definition: TableDefinition, terms: Terms) -> Result<(), Status> {
        if definition.durability != Durability::Durable {
            return Err(invalid(
                "ephemeral tables hold request contexts, which inference requests make",
            ));
        }
        let (name, task, columns, creation) = (
            definition.name.clone(),
            Task::of_classes(definition.classes),
            Arc::new(definition.columns.clone()),
            definition.created.clone(),
        );
        let series = SeriesPlacement::of(&definition);
        // The name is taken while the shard creates the table, without holding
        // the placement lock across the worker's round trip; another creation
        // of the name waits for this one, then finds the table there or not.
        let (shard, reservation) = loop {
            let mut waiting = {
                let placement = self.placement.read().await;
                let mut creating = self.creating.lock().unwrap_or_else(PoisonError::into_inner);
                if placement.contains_key(&name) {
                    return Err(exists(format!("table {:?} exists", name.as_str())));
                }
                // Logs are files named for their tables, which a filesystem
                // that ignores case would take for one another.
                if let Some(other) = placement
                    .keys()
                    .chain(creating.keys())
                    .find(|other| other.as_str().eq_ignore_ascii_case(name.as_str()))
                    .filter(|other| **other != name)
                {
                    return Err(exists(format!(
                        "table {:?} exists, and names may not differ only by case",
                        other.as_str()
                    )));
                }
                if let Some((_, placed)) = creating.get(&name) {
                    placed.clone()
                } else if placement.len() + creating.len() >= self.max_tables {
                    return Err(Status::resource_exhausted(format!(
                        "the server holds its most tables, {}; drop some, or raise max_tables",
                        self.max_tables
                    )));
                } else {
                    let mut counts = vec![0usize; self.shards.len()];
                    for &shard in placement
                        .values()
                        .map(|placed| &placed.shard)
                        .chain(creating.values().map(|(shard, _)| shard))
                    {
                        counts[shard] += 1;
                    }
                    let loads = self.mover.as_ref().map(|mover| {
                        mover
                            .shard_loads
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .clone()
                    });
                    // A table placed since the last round counts as one of mean
                    // load, so creations in a burst spread out.
                    let mut fresh = vec![0usize; self.shards.len()];
                    if let Some(mover) = &self.mover {
                        for &shard in mover
                            .fresh
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .values()
                        {
                            fresh[shard] += 1;
                        }
                    }
                    // The mean load of a table the last round measured.
                    let measured = counts
                        .iter()
                        .sum::<usize>()
                        .saturating_sub(fresh.iter().sum::<usize>());
                    let per_table = loads
                        .as_ref()
                        .map_or(0.0, |l| l.iter().sum::<f64>() / measured.max(1) as f64);
                    let load = |index: usize| {
                        loads.as_ref().map_or(0.0, |l| l[index]) + per_table * fresh[index] as f64
                    };
                    let shard = (0..counts.len())
                        .min_by(|&a, &b| {
                            load(a)
                                .total_cmp(&load(b))
                                .then((counts[a], a).cmp(&(counts[b], b)))
                        })
                        .expect("servers have a shard");
                    if let Some(mover) = &self.mover {
                        mover
                            .fresh
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .insert(name.to_string(), shard);
                    }
                    let (placed, waiting) = watch::channel(());
                    creating.insert(name.clone(), (shard, waiting));
                    let reservation = Reservation {
                        creating: self.creating.clone(),
                        name: name.clone(),
                        _placed: placed,
                    };
                    break (shard, reservation);
                }
            };
            // Ends once the other creation gives the name back.
            let _ = waiting.changed().await;
        };
        // Apart from this call, so a client that goes away cannot leave a table
        // its shard serves but the placement never names.
        let service = self.clone();
        tokio::spawn(async move {
            // Calls for the name wait until the table is placed or is not.
            let _routed = service.route(&name).write_owned().await;
            let created = service
                .ask(shard, |reply| Command::Create {
                    definition,
                    terms,
                    reply,
                })
                .await;
            let mut placement = service.placement.write().await;
            let levels = match created {
                Ok(levels) => levels,
                Err(error) => {
                    if let Some(mover) = &service.mover {
                        mover.forget(&name);
                    }
                    service
                        .routes
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .remove(name.as_str());
                    return Err(error);
                }
            };
            placement.insert(
                name,
                Placement {
                    shard,
                    task,
                    columns,
                    created: creation,
                    levels: levels.map(Arc::new),
                    series,
                },
            );
            service.metrics.tables.add(1);
            drop(reservation);
            Ok(())
        })
        .await
        .map_err(|error| Status::internal(format!("creating the table: {error}")))?
    }
}

impl Service {
    /// Applies a stream of log entries from another node; the first message names the table.
    async fn replicate(
        &self,
        table: TableName,
        mut stream: Peekable<Streaming<FlightData>>,
        lineage: Lineage,
    ) -> Result<u64, Status> {
        #[derive(Deserialize)]
        struct Sequenced {
            sequence: u64,
            #[serde(default = "last")]
            last: bool,
        }
        fn last() -> bool {
            true
        }
        let mut length = None;
        let mut batch: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut batch_bytes = 0;
        // The parts of an entry read so far, until its last arrives.
        let mut partial: Option<(u64, Vec<u8>)> = None;
        // What the batch and the partial entry hold of the server's budget.
        let mut charge = self.budget.charge();
        loop {
            let next = stream.next().await.transpose()?;
            let mut entry = None;
            if let Some(message) = &next {
                let Sequenced { sequence, last } = serde_json::from_slice(&message.app_metadata)
                    .map_err(|error| invalid(format!("bad replication metadata: {error}")))?;
                let held = partial.as_ref().map_or(0, |(_, bytes)| bytes.len());
                if (held + message.data_body.len()) as u64 > self.budget.limit() {
                    return Err(Status::resource_exhausted(format!(
                        "a log entry is larger than the {} MiB of requests this node holds \
                         at once; raise its request_bytes_in_flight",
                        self.budget.limit() >> 20
                    )));
                }
                charge.grow(message.data_body.len() as u64)?;
                let bytes = match partial.take() {
                    Some((earlier, mut bytes)) if earlier == sequence => {
                        bytes.extend_from_slice(&message.data_body);
                        bytes
                    }
                    Some(_) => return Err(invalid("a log entry ended before its last part")),
                    None => message.data_body.to_vec(),
                };
                if bytes.len() > ENTRY_BYTES {
                    return Err(invalid("a log entry is larger than a log can hold"));
                }
                if last {
                    entry = Some((sequence, bytes));
                } else {
                    partial = Some((sequence, bytes));
                }
            } else if partial.is_some() {
                return Err(invalid("the stream ended inside a log entry"));
            }
            if let Some((sequence, bytes)) = entry {
                if sequence == 0 {
                    let LogEntry::Definition(definition, terms) =
                        decode(&bytes).map_err(|error| invalid(error.to_string()))?
                    else {
                        return Err(invalid("entry 0 of a log is its definition"));
                    };
                    if definition.name != table {
                        return Err(invalid("the definition names another table"));
                    }
                    if lineage
                        .created
                        .as_ref()
                        .is_some_and(|created| definition.created.as_ref() != Some(created))
                    {
                        return Err(invalid("the definition is not the shipped copy's"));
                    }
                    match self.create(definition, terms).await {
                        Ok(()) => {}
                        Err(error) if error.code() == tonic::Code::AlreadyExists => {}
                        Err(error) => return Err(error),
                    }
                }
                // The shard takes a later term from an existing copy's definition.
                batch_bytes += bytes.len();
                batch.push((sequence, bytes));
            }
            let full = batch.len() >= REPLICATE_BATCH || batch_bytes >= REPLICATE_BATCH_BYTES;
            if !batch.is_empty() && (next.is_none() || full) {
                let (placed, _routed) = self.routed(&table).await?;
                let shard = placed.shard;
                let entries = std::mem::take(&mut batch);
                batch_bytes = 0;
                let name = table.clone();
                let lineage = Box::new(lineage.clone());
                length = Some(
                    self.ask(shard, |reply| Command::Replicate {
                        table: name,
                        entries,
                        lineage,
                        reply,
                    })
                    .await?,
                );
                // The shard has the batch now; the partial entry stays charged.
                charge = self.budget.charge();
                if let Some((_, bytes)) = &partial {
                    charge.grow(bytes.len() as u64)?;
                }
            }
            if next.is_none() {
                break;
            }
        }
        match length {
            Some(length) => Ok(length),
            None => {
                let (placed, _routed) = self.routed(&table).await?;
                let shard = placed.shard;
                let described = self
                    .ask(shard, |reply| Command::Describe { table, reply })
                    .await?;
                Ok(described.log_length)
            }
        }
    }

    /// Stops serving a table here and deletes its log.
    ///
    /// The table leaves the placement first, so calls for it find nothing
    /// while the shard drains it, and comes back if the shard cannot. Its
    /// name stays taken meanwhile, so a creation of it waits until the old
    /// table's log is gone rather than racing it on another shard, and its
    /// route is held, so a move of it finishes first or waits.
    async fn release(&self, table: TableName, keep: bool) -> Result<(), Status> {
        let routed = self.route(&table).write_owned().await;
        let (placed, reservation) = {
            let mut placement = self.placement.write().await;
            let Some(placed) = placement.remove(&table) else {
                self.routes
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(table.as_str());
                return Err(Status::not_found(format!("table {table:?} does not exist")));
            };
            let (signal, waiting) = watch::channel(());
            self.creating
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(table.clone(), (placed.shard, waiting));
            let reservation = Reservation {
                creating: self.creating.clone(),
                name: table.clone(),
                _placed: signal,
            };
            (placed, reservation)
        };
        // Apart from this call, so a client that goes away cannot leave a
        // table its shard serves but the placement no longer names.
        let service = self.clone();
        tokio::spawn(async move {
            let name = table.clone();
            let released = service
                .ask(placed.shard, |reply| Command::Release {
                    table: name,
                    keep,
                    reply,
                })
                .await;
            let mut placement = service.placement.write().await;
            if let Err(error) = released {
                placement.insert(table, placed);
                return Err(error);
            }
            service.metrics.tables.set(placement.len() as u64);
            service.metrics.forget(&table);
            if let Some(mover) = &service.mover {
                mover.forget(&table);
            }
            service
                .routes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(table.as_str());
            drop(reservation);
            drop(routed);
            Ok(())
        })
        .await
        .map_err(|error| Status::internal(format!("dropping the table: {error}")))?
    }
}

fn invalid(message: impl Into<String>) -> Status {
    Status::invalid_argument(message.into())
}

/// The command, table and body an exchange's descriptor path names: a
/// prediction's `predict/payments`, or a partition's call with its JSON body
/// third.
fn exchange_path(
    descriptor: Option<&FlightDescriptor>,
) -> Result<(String, TableName, Vec<u8>), Status> {
    let mut parts = descriptor
        .map(|descriptor| descriptor.path.clone())
        .unwrap_or_default();
    let body = if parts.len() == 3 {
        parts.pop().map(String::into_bytes).unwrap_or_default()
    } else {
        Vec::new()
    };
    let named = FlightDescriptor::new_path(parts);
    let (command, table) = path(Some(&named))?;
    Ok((command, table, body))
}

/// The command and table a descriptor path names, such as `insert/payments`.
fn path(descriptor: Option<&FlightDescriptor>) -> Result<(String, TableName), Status> {
    let parts = descriptor
        .map(|descriptor| descriptor.path.clone())
        .unwrap_or_default();
    match parts.as_slice() {
        [command, table] => {
            let table =
                TableName::new(table.as_str()).map_err(|error| invalid(error.to_string()))?;
            Ok((command.clone(), table))
        }
        _ => Err(invalid("the descriptor path must be [command, table]")),
    }
}

/// The descriptor a Flight stream's first message carries, without reading past it.
async fn descriptor(stream: &mut Peekable<Streaming<FlightData>>) -> Option<FlightDescriptor> {
    match Pin::new(stream).peek().await {
        Some(Ok(first)) => first.flight_descriptor.clone(),
        _ => None,
    }
}

/// The record batches in a Flight stream, charged to `charge` as they
/// arrive, and the charge, which the call holds until the shard has answered
/// what the batches became.
///
/// `DoExchange` clients may open with a message that holds only the descriptor,
/// so messages without an Arrow header are skipped. A stream past
/// `REQUEST_BYTES` on the wire or once decoded, or past the server's budget,
/// is refused as it arrives.
async fn batches(
    stream: Peekable<Streaming<FlightData>>,
    charge: Charge,
) -> Result<(Vec<RecordBatch>, Charge), Status> {
    let mut received = 0;
    let mut decoded = 0;
    // Shared by the wire and the decoder, whose streams outlive no call.
    let charge = Arc::new(std::sync::Mutex::new(charge));
    let wire_charge = Arc::clone(&charge);
    let decoded_charge = Arc::clone(&charge);
    let messages = stream
        .try_filter(|message| futures::future::ready(!message.data_header.is_empty()))
        .and_then(move |message| {
            received += message.data_body.len();
            let taken = if received > REQUEST_BYTES {
                Err(invalid(format!(
                    "a request may carry at most {} MiB; split it",
                    REQUEST_BYTES >> 20
                )))
            } else {
                wire_charge
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .grow(message.data_body.len() as u64)
            };
            futures::future::ready(taken.map(|()| message))
        })
        .map_err(|status| FlightError::Tonic(Box::new(status)));
    let batches: Vec<RecordBatch> = FlightRecordBatchStream::new_from_flight_data(messages)
        .and_then(move |batch| {
            // The wire bounds nothing once decoded: a column of Arrow's null
            // type takes no bytes however many rows its header claims.
            let bytes = decoded_bytes(&batch);
            decoded += bytes;
            let taken = if decoded > REQUEST_BYTES {
                Err(invalid(format!(
                    "a request's rows may take at most {} MiB in memory; split it",
                    REQUEST_BYTES >> 20
                )))
            } else {
                decoded_charge
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .grow(bytes as u64)
            };
            futures::future::ready(
                taken
                    .map(|()| batch)
                    .map_err(|status| FlightError::Tonic(Box::new(status))),
            )
        })
        .try_collect()
        .await
        .map_err(|error| match error {
            FlightError::Tonic(status) => *status,
            error => invalid(format!("bad Arrow data: {error}")),
        })?;
    let charge = Arc::into_inner(charge)
        .expect("the streams that shared the charge are done")
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    Ok((batches, charge))
}

/// A query exchange's rows: its batches, held against `charge` until the
/// shard has answered them, and their features and words for `columns`.
///
/// # Errors
///
/// As `batches` and `features`, and `INVALID_ARGUMENT` for no rows at all.
async fn query_rows(
    stream: Peekable<Streaming<FlightData>>,
    columns: &[Column],
    charge: Charge,
) -> Result<(Vec<RecordBatch>, Charge, Vec<f32>, Strings), Status> {
    let (batches, charge) = batches(stream, charge).await?;
    let (features, strings, rows) = features(&batches, columns)?;
    if rows == 0 {
        return Err(invalid("no query rows"));
    }
    Ok((batches, charge, features, strings))
}

/// One batch as a Flight response.
fn encoded(batch: RecordBatch) -> Response<Boxed<FlightData>> {
    let encoded = FlightDataEncoderBuilder::new()
        .build(stream::iter([Ok(batch)]))
        .map_err(|error| Status::internal(error.to_string()));
    Response::new(Box::pin(encoded))
}

/// Bytes a batch takes once decoded for the shard, beyond the Arrow data it
/// arrived as, which the wire charge covers: four per cell, with each string
/// of a text column in an allocation of its own.
fn decoded_bytes(batch: &RecordBatch) -> usize {
    let rows = batch.num_rows();
    batch
        .columns()
        .iter()
        .map(|column| {
            let per_row = match column.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Dictionary(..) => 32,
                DataType::FixedSizeBinary(_) => 16,
                _ => 4,
            };
            rows * per_row
        })
        .sum()
}

fn float_column(column: &dyn Array, name: &str) -> Result<Vec<f32>, Status> {
    let values = column
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid(format!("column {name:?} must be float32")))?;
    Ok((0..values.len())
        .map(|index| {
            if values.is_null(index) {
                f32::NAN
            } else {
                values.value(index)
            }
        })
        .collect())
}

/// Row-major features from the batches' feature columns, by name and kind:
/// numbers in place, and the strings of categorical columns apart, for the
/// shard to code. Columns the table does not have are left alone.
fn features(
    batches: &[RecordBatch],
    columns: &[Column],
) -> Result<(Vec<f32>, Strings, usize), Status> {
    let mut rows = Vec::new();
    let mut strings: BTreeMap<usize, Vec<Option<String>>> = BTreeMap::new();
    let mut count = 0;
    for batch in batches {
        let mut decoded = Vec::with_capacity(columns.len());
        for column in columns {
            let array = batch
                .column_by_name(&column.name)
                .ok_or_else(|| invalid(format!("missing column {:?}", column.name)))?;
            decoded.push(columns::decode(array, column).map_err(invalid)?);
        }
        for row in 0..batch.num_rows() {
            for cells in &decoded {
                cells.extend_row(row, &mut rows);
            }
        }
        for (index, cells) in decoded.into_iter().enumerate() {
            if let Cells::Strings(values) = cells {
                strings.entry(index).or_default().extend(values);
            }
        }
        count += batch.num_rows();
    }
    for (index, values) in &strings {
        if values.len() != count {
            return Err(invalid(format!(
                "column {:?} must hold strings in every batch or in none",
                columns[*index].name
            )));
        }
    }
    Ok((rows, strings.into_iter().collect(), count))
}

/// A column that must be there, without nulls.
fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array, Status> {
    let column = batch
        .column_by_name(name)
        .map(AsRef::as_ref)
        .ok_or_else(|| invalid(format!("missing column {name:?}")))?;
    if column.null_count() > 0 {
        return Err(invalid(format!("column {name:?} has nulls")));
    }
    Ok(column)
}

/// A batch's 128-bit keys; see `partition::keys_of`.
fn batch_keys(batch: &RecordBatch) -> Result<Vec<u128>, Status> {
    if let Some(key) = batch.column_by_name("key")
        && key.null_count() > 0
    {
        return Err(invalid("column \"key\" has nulls"));
    }
    partition::keys_of(batch).map_err(invalid)
}

/// Each query's given context from the batches' "context" column, a list of
/// 16-byte keys per query, if they have one: in every batch, or in none.
fn given_keys(batches: &[RecordBatch]) -> Result<Option<Vec<Vec<u128>>>, Status> {
    let having = batches
        .iter()
        .filter(|batch| batch.column_by_name("context").is_some())
        .count();
    if having == 0 {
        return Ok(None);
    }
    if having != batches.len() {
        return Err(invalid(
            "column \"context\" must be in every batch or in none",
        ));
    }
    let mut contexts = Vec::new();
    for batch in batches {
        let lists = batch
            .column_by_name("context")
            .and_then(|column| column.as_any().downcast_ref::<arrow_array::ListArray>())
            .ok_or_else(|| invalid("column \"context\" holds a list of 16-byte keys per query"))?;
        for index in 0..lists.len() {
            let list = lists.value(index);
            let keys = list
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .filter(|keys| keys.value_length() == 16 && keys.null_count() == 0)
                .ok_or_else(|| {
                    invalid("column \"context\" holds a list of 16-byte keys per query")
                })?;
            contexts.push(
                (0..keys.len())
                    .map(|key| u128::from_le_bytes(keys.value(key).try_into().expect("16 bytes")))
                    .collect(),
            );
        }
    }
    Ok(Some(contexts))
}

/// Each series' group of a forecast request, from a `group` column of lists
/// of series digests, or None without one.
fn groups_column(batches: &[RecordBatch]) -> Result<Option<Vec<Vec<u64>>>, Status> {
    let having = batches
        .iter()
        .filter(|batch| batch.column_by_name("group").is_some())
        .count();
    if having == 0 {
        return Ok(None);
    }
    if having != batches.len() {
        return Err(invalid(
            "column \"group\" must be in every batch or in none",
        ));
    }
    let mut groups = Vec::new();
    for batch in batches {
        let lists = batch
            .column_by_name("group")
            .and_then(|column| column.as_any().downcast_ref::<arrow_array::ListArray>())
            .ok_or_else(|| invalid("column \"group\" holds a list of series per forecast"))?;
        for index in 0..lists.len() {
            let list = lists.value(index);
            let any = list.as_any();
            let members: Vec<u64> = if let Some(unsigned) = any.downcast_ref::<UInt64Array>() {
                unsigned.values().iter().copied().collect()
            } else if let Some(signed) = any.downcast_ref::<Int64Array>() {
                signed.values().iter().map(|&key| key as u64).collect()
            } else {
                return Err(invalid(
                    "column \"group\" holds a list of uint64 or int64 series per forecast",
                ));
            };
            groups.push(members);
        }
    }
    Ok(Some(groups))
}

/// The keys of delete batches.
fn keys(batches: &[RecordBatch]) -> Result<Vec<u128>, Status> {
    let mut keys = Vec::new();
    for batch in batches {
        keys.extend(batch_keys(batch)?);
    }
    Ok(keys)
}

/// Keys, features and targets from insert or update batches, or keys and
/// targets alone from label batches. A target not known yet is NaN, or null
/// or -1 for a class.
fn changes(
    batches: &[RecordBatch],
    placement: &Placement,
    id: Option<String>,
    labels_only: bool,
) -> Result<Incoming, Status> {
    let mut keys = Vec::new();
    let mut targets = match placement.task {
        Task::Values => Targets::Values(Vec::new()),
        Task::Classes(_) => Targets::Classes(Vec::new()),
    };
    // String labels, which the table's label vocabulary codes after its columns'.
    let mut labels_given: Vec<Option<String>> = Vec::new();
    for batch in batches {
        keys.extend(batch_keys(batch)?);
        // A null target is one not known yet.
        let target = batch
            .column_by_name("target")
            .map(AsRef::as_ref)
            .ok_or_else(|| invalid("missing column \"target\""))?;
        match &mut targets {
            Targets::Classes(codes) => {
                if let Some(given) = target.as_any().downcast_ref::<Int32Array>() {
                    codes.extend(given.iter().map(|code| code.unwrap_or(UNLABELED)));
                    continue;
                }
                let label = Column {
                    name: "target".to_string(),
                    kind: ColumnKind::Categorical,
                };
                let array = batch.column_by_name("target").expect("found above");
                let Ok(Cells::Strings(strings)) = columns::decode(array, &label) else {
                    return Err(invalid(
                        "column \"target\" of a classification table holds int32 class \
                         codes or string labels",
                    ));
                };
                codes.extend(std::iter::repeat_n(0, strings.len()));
                labels_given.extend(strings);
            }
            Targets::Values(values) => {
                let targets = float_column(target, "target")?;
                // A series table stores a step without a target as NaN, and
                // any table a row whose label is not known yet.
                if targets.iter().any(|value| value.is_infinite()) {
                    return Err(invalid(if placement.series.is_some() {
                        "targets of a series table are finite or NaN, for a step without one"
                    } else {
                        "regression targets are finite, or NaN for rows not labeled yet"
                    }));
                }
                values.extend(targets);
            }
        }
    }
    let (features, mut strings, rows) = if labels_only {
        (Vec::new(), Vec::new(), keys.len())
    } else {
        features(batches, &placement.columns)?
    };
    if rows != keys.len() {
        return Err(invalid("every row needs a key"));
    }
    if !labels_given.is_empty() {
        if labels_given.len() != rows {
            return Err(invalid(
                "column \"target\" must hold labels in every batch or in none",
            ));
        }
        strings.push((placement.columns.len(), labels_given));
    }
    let series = if placement.series.is_some() && !labels_only {
        series_keys(batches, &keys)?
    } else {
        Vec::new()
    };
    Ok(Incoming {
        rows: Rows {
            keys,
            features,
            targets,
            times: times(batches, "time")?.map(|times| times.into_iter().map(Some).collect()),
            id,
            words: Vec::new(),
            series,
            writes: Vec::new(),
            layout: None,
            table: None,
        },
        strings,
    })
}

/// The series the observations of a write to a table of series belong to,
/// each once: its digest, the high half of its observations' row keys, and
/// its key as the client gave it, which column `"series_key"` holds per
/// observation as the JSON of a `GivenKey`, plain or dictionary-encoded text.
/// A key that does not digest to its observations' series is refused.
fn series_keys(batches: &[RecordBatch], keys: &[u128]) -> Result<Vec<SeriesKey>, Status> {
    let mut named = Vec::new();
    let mut seen = HashSet::new();
    let mut row = 0;
    for batch in batches {
        let column = column(batch, "series_key")?;
        if !matches!(
            column.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Dictionary(..)
        ) {
            return Err(invalid(
                "column \"series_key\" holds each observation's series key as text",
            ));
        }
        let texts = arrow_cast::cast(column, &DataType::Utf8)
            .map_err(|error| invalid(format!("column \"series_key\": {error}")))?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast to utf8");
        for index in 0..texts.len() {
            let Some(key) = keys.get(row) else {
                return Err(invalid("every observation needs a key"));
            };
            row += 1;
            let digest = (key >> 64) as u64;
            if !seen.insert(digest) {
                continue;
            }
            if texts.is_null(index) {
                return Err(invalid("a series key cannot be missing"));
            }
            let given: GivenKey = serde_json::from_str(texts.value(index)).map_err(|error| {
                invalid(format!(
                    "column \"series_key\" holds each series' key as the JSON of its kind and \
                     value: {error}"
                ))
            })?;
            let series = SeriesKey { digest, key: given };
            series
                .check()
                .map_err(|error| invalid(format!("column \"series_key\": {error}")))?;
            named.push(series);
        }
    }
    if row != keys.len() {
        return Err(invalid("every observation needs a series key"));
    }
    Ok(named)
}

/// The series a forecast batch names, by their keys' 64-bit digests: a uint64
/// or int64 column, or 16-byte keys as `generalforecast.keys.series_keys` digests them.
fn series_column(batches: &[RecordBatch]) -> Result<Vec<u64>, Status> {
    let mut keys = Vec::new();
    for batch in batches {
        let column = column(batch, "series")?;
        let any = column.as_any();
        if let Some(unsigned) = any.downcast_ref::<UInt64Array>() {
            keys.extend(unsigned.values().iter().copied());
        } else if let Some(signed) = any.downcast_ref::<Int64Array>() {
            keys.extend(signed.values().iter().map(|&key| key as u64));
        } else if let Some(wide) = any.downcast_ref::<FixedSizeBinaryArray>()
            && wide.value_length() == 16
        {
            keys.extend((0..wide.len()).map(|index| {
                u64::from_le_bytes(wide.value(index)[..8].try_into().expect("8 bytes"))
            }));
        } else {
            return Err(invalid(
                "column \"series\" must be uint64, int64 or fixed_size_binary(16)",
            ));
        }
    }
    Ok(keys)
}

/// An optional int64 column's values across batches: in every batch, or in
/// none, and never null.
fn times(batches: &[RecordBatch], name: &str) -> Result<Option<Vec<i64>>, Status> {
    let having = batches
        .iter()
        .filter(|batch| batch.column_by_name(name).is_some())
        .count();
    if having == 0 {
        return Ok(None);
    }
    if having < batches.len() {
        return Err(invalid(format!(
            "column {name:?} must be in every batch or in none"
        )));
    }
    let mut values = Vec::new();
    for batch in batches {
        let times = column(batch, name)?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| invalid(format!("column {name:?} must be int64")))?;
        if times.null_count() > 0 {
            return Err(invalid(format!("column {name:?} must not hold nulls")));
        }
        values.extend(times.values().iter().copied());
    }
    Ok(Some(values))
}

/// Predictions as a batch: `p0..` class probabilities, or `q001..` quantiles.
/// The quantile levels a client asked a regressor's predictions to keep, if any.
fn quantile_levels<T>(request: &Request<T>) -> Result<Option<Vec<f64>>, Status> {
    let Some(given) = request.metadata().get(QUANTILES) else {
        return Ok(None);
    };
    given
        .to_str()
        .map_err(|_| invalid(format!("{QUANTILES} is a comma-separated list")))
        .and_then(|given| {
            quantiles::parse_levels(given).map_err(|error| invalid(format!("{QUANTILES}: {error}")))
        })
        .map(Some)
}

/// The quantile columns of predictions at `levels`, as `quantile_values`
/// computes them, named `q<level>`.
fn quantile_columns(
    values: &[f32],
    known: &[f64],
    levels: &[f64],
) -> Result<Vec<(Field, ArrayRef)>, Status> {
    let picked = quantile_values(values, known, levels)
        .ok_or_else(|| Status::internal("a regressor's predictions have no columns"))?;
    Ok(levels
        .iter()
        .zip(picked)
        .map(|(&level, picked)| {
            (
                Field::new(format!("q{level}"), DataType::Float32, false),
                Arc::new(Float32Array::from(picked)) as ArrayRef,
            )
        })
        .collect())
}

/// The table's quantile levels for its prediction columns: its own, or the
/// evenly spaced ones an earlier placement implies.
fn known_levels(placement: &Placement, columns: usize) -> Vec<f64> {
    match &placement.levels {
        Some(levels) if levels.len() == columns => levels.to_vec(),
        _ => (0..columns)
            .map(|column| (column + 1) as f64 / (columns + 1) as f64)
            .collect(),
    }
}

/// Predictions at quantile levels; columns `q<level>`.
fn quantile_batch(
    predictions: &Predictions,
    levels: &[f64],
    placement: &Placement,
) -> Result<RecordBatch, Status> {
    let known = known_levels(placement, predictions.columns);
    let (fields, arrays): (Vec<Field>, Vec<Arc<dyn Array>>) =
        quantile_columns(&predictions.values, &known, levels)?
            .into_iter()
            .unzip();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|error| Status::internal(error.to_string()))
}

/// Forecasts as a long batch: each series' digest and the time of each step,
/// then one `q<level>` column per level, the table's own without `levels`.
fn forecasted(
    predictions: &Predictions,
    series: &[u64],
    step: i64,
    horizon: NonZeroUsize,
    levels: Option<&[f64]>,
    placement: &Placement,
) -> Result<RecordBatch, Status> {
    let horizon = horizon.get();
    if predictions.horizon != Some(horizon)
        || predictions.rows != series.len()
        || predictions.cutoffs.len() != series.len()
    {
        return Err(Status::internal(
            "a forecast per series over the horizon asked was expected",
        ));
    }
    let known = known_levels(placement, predictions.columns);
    if placement
        .levels
        .as_ref()
        .is_some_and(|levels| levels.len() != predictions.columns)
    {
        return Err(Status::internal(
            "the forecast has a column per quantile level of the table",
        ));
    }
    let mut fields = vec![
        Field::new("series", DataType::UInt64, false),
        Field::new("time", DataType::Int64, false),
    ];
    let mut arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(UInt64Array::from(
            series
                .iter()
                .flat_map(|&key| std::iter::repeat_n(key, horizon))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            predictions
                .cutoffs
                .iter()
                .flat_map(|&cutoff| {
                    (0..horizon as i64).map(move |ahead| cutoff.saturating_add(ahead * step))
                })
                .collect::<Vec<_>>(),
        )),
    ];
    let columns = quantile_columns(&predictions.values, &known, levels.unwrap_or(&known))?;
    for (field, array) in columns {
        fields.push(field);
        arrays.push(array);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|error| Status::internal(error.to_string()))
}

fn prediction_batch(predictions: &Predictions, task: Task) -> Result<RecordBatch, Status> {
    let names: Vec<String> = (0..predictions.columns)
        .map(|index| match task {
            Task::Values => format!("q{:03}", index + 1),
            Task::Classes(_) => format!("p{index}"),
        })
        .collect();
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|name| Field::new(name, DataType::Float32, false))
            .collect::<Vec<_>>(),
    ));
    let columns: Vec<Arc<dyn Array>> = (0..predictions.columns)
        .map(|column| {
            let values: Vec<f32> = (0..predictions.rows)
                .map(|row| predictions.values[row * predictions.columns + column])
                .collect();
            Arc::new(Float32Array::from(values)) as Arc<dyn Array>
        })
        .collect();
    RecordBatch::try_new(schema, columns).map_err(|error| Status::internal(error.to_string()))
}

/// The schema metadata that carries a prediction's share of its step's
/// worker time, in seconds, so a client can attribute cost per request.
pub const GPU_SECONDS: &str = "generalforecast.gpu_seconds";

/// The schema metadata that carries the epoch of the table's state that
/// answered, which rises with each refit, so a stream can mark refits.
pub const EPOCH: &str = "generalforecast.epoch";

/// The header a forecast's horizon travels in: steps, at least one.
pub const HORIZON: &str = "gf-horizon";

/// The horizon a forecast call asks for, if it names one.
fn horizon_of<T>(request: &Request<T>) -> Result<Option<usize>, Status> {
    let Some(given) = request.metadata().get(HORIZON) else {
        return Ok(None);
    };
    given
        .to_str()
        .ok()
        .and_then(|steps| steps.trim().parse::<usize>().ok())
        .filter(|&steps| steps > 0)
        .map(Some)
        .ok_or_else(|| invalid(format!("{HORIZON} is a whole number of steps, at least 1")))
}

/// A batch as a Flight response, with the predictions' worker time and
/// epoch in the schema's metadata. The reply streams, and if `laps` are
/// timed, the time since their last lap plus the time spent encoding each
/// message goes to `request_encode_seconds` once the stream ends or is
/// dropped.
fn answered(
    batch: RecordBatch,
    predictions: &Predictions,
    mut laps: Laps,
    metrics: Arc<Metrics>,
) -> Result<Response<Boxed<FlightData>>, Status> {
    let schema = Schema::clone(&batch.schema()).with_metadata(HashMap::from([
        (GPU_SECONDS.to_string(), predictions.gpu_seconds.to_string()),
        (EPOCH.to_string(), predictions.epoch.to_string()),
    ]));
    let batch = batch
        .with_schema(Arc::new(schema))
        .map_err(|error| Status::internal(error.to_string()))?;
    let encoder = FlightDataEncoderBuilder::new()
        .build(stream::iter([Ok(batch)]))
        .map_err(|error| Status::internal(error.to_string()))
        .boxed();
    if !laps.timed() {
        return Ok(Response::new(encoder));
    }
    let timed = Timed::new(encoder, laps.lap(), move |spent| {
        metrics.request_encode_seconds.observe(spent);
    });
    Ok(Response::new(Box::pin(timed)))
}

/// The time between a request's parts, read only when the server times
/// requests: untimed laps read no clock and observe nothing.
#[derive(Clone, Copy)]
struct Laps(Option<Instant>);

impl Laps {
    fn timed(self) -> bool {
        self.0.is_some()
    }

    /// The time since the last lap, which starts the next; zero untimed.
    fn lap(&mut self) -> Duration {
        let Some(last) = &mut self.0 else {
            return Duration::ZERO;
        };
        let now = Instant::now();
        let spent = now - *last;
        *last = now;
        spent
    }

    /// Counts `spent` in `histogram` if timed.
    fn observe(self, histogram: &Histogram, spent: Duration) {
        if self.timed() {
            histogram.observe(spent);
        }
    }
}

/// A stream that sums the time its inner stream takes to produce each item
/// and hands the total to `done` once, when the stream ends or is dropped.
struct Timed<S, F: FnOnce(Duration)> {
    stream: S,
    spent: Duration,
    done: Option<F>,
}

impl<S, F: FnOnce(Duration)> Timed<S, F> {
    fn new(stream: S, spent: Duration, done: F) -> Self {
        Self {
            stream,
            spent,
            done: Some(done),
        }
    }

    fn finish(&mut self) {
        if let Some(done) = self.done.take() {
            done(self.spent);
        }
    }
}

impl<S: Stream + Unpin, F: FnOnce(Duration) + Unpin> Stream for Timed<S, F> {
    type Item = S::Item;

    fn poll_next(
        mut self: Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<S::Item>> {
        let polling = Instant::now();
        let polled = self.stream.poll_next_unpin(context);
        if let std::task::Poll::Ready(item) = &polled {
            self.spent += polling.elapsed();
            if item.is_none() {
                self.finish();
            }
        }
        polled
    }
}

impl<S, F: FnOnce(Duration)> Drop for Timed<S, F> {
    fn drop(&mut self) {
        self.finish();
    }
}

#[derive(Deserialize)]
struct LogTicket {
    table: TableName,
    from: u64,
}

/// Entries per command when applying a replicated stream, so a whole log never sits in memory.
const REPLICATE_BATCH: usize = 64;

/// Bytes of entries per command when applying a replicated stream.
const REPLICATE_BATCH_BYTES: usize = 64 << 20;

/// Entries per command when reading a log for a replica, so a whole log never sits in memory.
const READ_BATCH: usize = 64;

/// Sealed bytes of entries past the first per command when reading a log
/// for a replica, so a page of snapshot records never sits in memory at once.
const READ_BATCH_BYTES: usize = 64 << 20;

/// A client's id for a write, from the `gf-write-id` metadata.
fn write_id<T>(request: &Request<T>) -> Result<Option<String>, Status> {
    match request.metadata().get(WRITE_ID) {
        Some(id) => {
            let id = id
                .to_str()
                .map_err(|_| invalid(format!("{WRITE_ID} must be ASCII")))?;
            check_write_id(id).map_err(|error| invalid(format!("{WRITE_ID}: {error}")))?;
            Ok(Some(id.to_string()))
        }
        None => Ok(None),
    }
}

/// Metadata a router sets on a shipment: the source copy's terms and creation id.
fn lineage<T>(request: &Request<T>) -> Result<Lineage, Status> {
    let terms = match request.metadata().get(TERMS) {
        Some(terms) => Some(
            terms
                .to_str()
                .ok()
                .and_then(|terms| serde_json::from_str(terms).ok())
                .ok_or_else(|| invalid(format!("{TERMS} must be a JSON list of terms")))?,
        ),
        None => None,
    };
    let created = request
        .metadata()
        .get(CREATED)
        .and_then(|created| created.to_str().ok())
        .and_then(CreationId::parse);
    Ok(Lineage { terms, created })
}

/// A new creation id, by this node's clock.
fn creation_id() -> Result<CreationId, Status> {
    let mut bytes = [0u8; 2];
    getrandom::fill(&mut bytes).map_err(|error| Status::internal(error.to_string()))?;
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| Status::internal(error.to_string()))?
        .as_millis();
    Ok(CreationId::new(millis, bytes))
}

/// What a write changed, and the creation of the table it changed.
fn answer(written: &crate::shard::Written, placement: &Placement) -> serde_json::Value {
    json!(Answer {
        rows: Some(written.rows),
        log_length: Some(written.log_length),
        created: placement.created.clone(),
    })
}

fn results(values: Vec<serde_json::Value>) -> Boxed<arrow_flight::Result> {
    let items = values.into_iter().map(|value| {
        Ok(arrow_flight::Result {
            body: value.to_string().into(),
        })
    });
    Box::pin(stream::iter(items.collect::<Vec<_>>()))
}

impl Service {
    async fn put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<PutResult>>, Status> {
        let replicating = shipment(&request);
        let fence = self.fence(&request)?;
        let scope = Scope::of(&request);
        let id = write_id(&request)?;
        let lineage = lineage(&request)?;
        let missing_ok = request.metadata().contains_key(MISSING_OK);
        let mut stream = request.into_inner().peekable();
        let (command, table) = path(descriptor(&mut stream).await.as_ref())?;
        scope.check(&table)?;
        let command: PutCommand = command.parse()?;
        if command == PutCommand::Replicate {
            replicating?;
            if lineage.created.is_none() {
                return Err(invalid(format!(
                    "a shipment names its source copy's creation in {CREATED}"
                )));
            }
            let log_length = self.replicate(table, stream, lineage).await?;
            let result = PutResult {
                app_metadata: json!({"log_length": log_length}).to_string().into(),
            };
            return Ok(Response::new(Box::pin(stream::iter([Ok(result)]))));
        }
        // Everything that can refuse the call goes before its rows are read.
        self.leased()?;
        let mut laps = self.laps(Instant::now);
        let placement = self.placement(&table).await?;
        let mut route = laps.lap();
        let (batches, charge) = batches(stream, self.budget.charge()).await?;
        let write = match command {
            PutCommand::Insert => Write::Insert(changes(&batches, &placement, id, false)?),
            PutCommand::Update => Write::Update(changes(&batches, &placement, id, false)?),
            PutCommand::Upsert => Write::Upsert(changes(&batches, &placement, id, false)?),
            PutCommand::Label => Write::Label(changes(&batches, &placement, id, true)?),
            PutCommand::Delete => Write::Delete {
                keys: keys(&batches)?,
                missing_ok,
                id,
            },
            PutCommand::Replicate => unreachable!("shipments are applied above"),
        };
        let decode = laps.lap();
        // The shard command must not cross a move; the upload above may. The
        // table's counts are taken under its route, which a drop waits for,
        // so a drop never finds them made again behind it.
        let (placement, _routed) = self.routed(&table).await?;
        let counts = self.metrics.table(&table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let written = self
            .ask(placement.shard, |reply| Command::Write {
                table,
                write,
                fence,
                charge,
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.rows_written.add(written.rows as u64);
        let result = PutResult {
            app_metadata: answer(&written, &placement).to_string().into(),
        };
        let encode = laps.lap();
        laps.observe(&self.metrics.request_encode_seconds, encode);
        Ok(Response::new(Box::pin(stream::iter([Ok(result)]))))
    }

    async fn exchange(
        &self,
        request: Request<Streaming<FlightData>>,
        trace: &Trace,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let started = Instant::now();
        let scope = Scope::of(&request);
        let levels = quantile_levels(&request)?;
        let horizon = horizon_of(&request)?;
        let internal = replication(&request);
        let mut stream = request.into_inner().peekable();
        let descriptor = descriptor(&mut stream).await;
        let (command, table, body) = exchange_path(descriptor.as_ref())?;
        scope.check(&table)?;
        match command.as_str() {
            "predict" => {}
            "forecast" => {
                return self
                    .forecast(&table, stream, horizon, levels.as_deref(), trace)
                    .await;
            }
            "contexts" => return self.contexts(&table, stream).await,
            "region_churn" => return self.region_churn(&table, stream).await,
            // Rows of a partitioned table travel between nodes only for routers.
            "candidates" => {
                internal?;
                return self.offer(&table, stream, &body).await;
            }
            "predict_given" => {
                internal?;
                return self
                    .predict_given(&table, stream, &body, levels.as_deref(), trace)
                    .await;
            }
            _ => return Err(invalid(format!("unknown exchange {command:?}"))),
        }
        let placement = self.placement(&table).await?;
        if placement.series.is_some() {
            return Err(invalid(
                "a table of series takes forecasts, not queries; use the forecast exchange",
            ));
        }
        if levels.is_some() && placement.task != Task::Values {
            return Err(invalid("quantiles are a regressor's predictions"));
        }
        let mut laps = self.laps(|| started);
        let mut route = laps.lap();
        let (batches, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        let before = times(&batches, "before")?;
        let given = given_keys(&batches)?;
        if before.is_some() && given.is_some() {
            return Err(invalid(
                "a query reads either its given rows or rows before its cutoff, not both",
            ));
        }
        let decode = laps.lap();
        let (placement, _routed) = self.routed(&table).await?;
        let counts = self.metrics.table(&table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let predictions = self
            .ask(placement.shard, |reply| Command::Predict {
                table,
                features,
                strings,
                before,
                given,
                trace: trace.id().to_string(),
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.predictions.add(predictions.rows as u64);
        counts.predict_seconds.observe(started.elapsed());
        self.predicted(&predictions, levels.as_deref(), &placement, laps)
            .await
    }

    /// Forecasts the series a batch names, each from its `before` time if
    /// the batch has one, over the horizon the call or the table gives.
    async fn forecast(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
        horizon: Option<usize>,
        levels: Option<&[f64]>,
        trace: &Trace,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let started = Instant::now();
        let placement = self.placement(table).await?;
        let Some(series) = placement.series else {
            return Err(invalid(
                "the table holds rows to predict from, not series; use the predict exchange",
            ));
        };
        let horizon = horizon
            .or(series.horizon)
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| {
                invalid(format!(
                    "give a horizon of at least one step in the {HORIZON} header, or a \
                     default when creating the table"
                ))
            })?;
        let mut laps = self.laps(|| started);
        let mut route = laps.lap();
        let (batches, _charge) = batches(stream, self.budget.charge()).await?;
        let keys = series_column(&batches)?;
        if keys.is_empty() {
            return Err(invalid("no series to forecast"));
        }
        let before = times(&batches, "before")?;
        let groups = groups_column(&batches)?;
        let decode = laps.lap();
        let (placement, _routed) = self.routed(table).await?;
        let counts = self.metrics.table(table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let predictions = self
            .ask(placement.shard, |reply| Command::Forecast {
                table: table.clone(),
                series: keys.clone(),
                before,
                horizon,
                groups,
                trace: trace.id().to_string(),
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.predictions.add(predictions.rows as u64);
        counts.predict_seconds.observe(started.elapsed());
        let batch = forecasted(
            &predictions,
            &keys,
            series.step,
            horizon,
            levels,
            &placement,
        )?;
        answered(batch, &predictions, laps, Arc::clone(&self.metrics))
    }

    /// The keys of each query's context, as a prediction of them would read
    /// it: one row per query, its keys a list of 16-byte values.
    async fn contexts(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let placement = self.placement(table).await?;
        let (batches, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        let before = times(&batches, "before")?;
        let (placement, _routed) = self.routed(table).await?;
        let contexts = self
            .ask(placement.shard, |reply| Command::Contexts {
                table: table.clone(),
                features,
                strings,
                before,
                reply,
            })
            .await?;
        let mut keys = ListBuilder::new(FixedSizeBinaryBuilder::new(16));
        for context in contexts {
            for key in context {
                keys.values()
                    .append_value(key.to_le_bytes())
                    .map_err(|error| Status::internal(error.to_string()))?;
            }
            keys.append(true);
        }
        let batch = RecordBatch::try_from_iter([("keys", Arc::new(keys.finish()) as ArrayRef)])
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(encoded(batch))
    }

    /// Each query's region churn, as `Rebuild::Churn` compares it: one row per
    /// query, NaN where its region has no built context.
    async fn region_churn(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let placement = self.placement(table).await?;
        let (_, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        let (placement, _routed) = self.routed(table).await?;
        let churn = self
            .ask(placement.shard, |reply| Command::RegionChurn {
                table: table.clone(),
                features,
                strings,
                reply,
            })
            .await?;
        let churn: Float64Array = churn
            .into_iter()
            .map(|churn| Some(churn.unwrap_or(f64::NAN)))
            .collect();
        let batch = RecordBatch::try_from_iter([("churn", Arc::new(churn) as ArrayRef)])
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(encoded(batch))
    }

    /// The rows a partition offers queries, and the queries, as rows travel
    /// between partitions; the body may name the partition's share of a sample.
    async fn offer(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
        body: &[u8],
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        #[derive(Default, serde::Deserialize)]
        struct Share {
            share: Option<u64>,
            #[serde(default)]
            strategy: Option<StrategySpec>,
        }
        let share: Share = if body.is_empty() {
            Share::default()
        } else {
            serde_json::from_slice(body).map_err(|error| invalid(error.to_string()))?
        };
        let placement = self.placement(table).await?;
        let (batches, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        // A router asks for one cutoff's queries at a time.
        let before = match times(&batches, "before")? {
            None => None,
            Some(cutoffs) if cutoffs.windows(2).all(|pair| pair[0] == pair[1]) => {
                cutoffs.first().copied()
            }
            Some(_) => return Err(invalid("a partition's candidates take one cutoff")),
        };
        let (placement, _routed) = self.routed(table).await?;
        let (offered, offers) = self
            .ask(placement.shard, |reply| Command::Candidates {
                table: table.clone(),
                features,
                strings,
                share: share.share,
                before,
                strategy: share.strategy.map(|strategy| strategy.strategy()),
                reply,
            })
            .await?;
        let batch = partition::rows_batch(
            &offered,
            &placement.columns,
            placement.task == Task::Values,
            Some(&offers),
        )
        .map_err(|error| Status::internal(error.to_string()))?;
        Ok(encoded(batch))
    }

    /// Predicts the rows without keys against contexts the body gives them
    /// from the rows with keys, gathered across a partitioned table.
    async fn predict_given(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
        body: &[u8],
        levels: Option<&[f64]>,
        trace: &Trace,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let started = Instant::now();
        let given: partition::Given =
            serde_json::from_slice(body).map_err(|error| invalid(error.to_string()))?;
        let placement = self.placement(table).await?;
        if levels.is_some() && placement.task != Task::Values {
            return Err(invalid("quantiles are a regressor's predictions"));
        }
        let mut laps = self.laps(|| started);
        let mut route = laps.lap();
        let (batches, _charge) = batches(stream, self.budget.charge()).await?;
        let rows = partition::read_rows(&batches, &placement.columns).map_err(invalid)?;
        let decode = laps.lap();
        let (placement, _routed) = self.routed(table).await?;
        let counts = self.metrics.table(table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let predictions = self
            .ask(placement.shard, |reply| Command::PredictGiven {
                table: table.clone(),
                rows,
                given,
                trace: trace.id().to_string(),
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.predictions.add(predictions.rows as u64);
        counts.predict_seconds.observe(started.elapsed());
        self.predicted(&predictions, levels, &placement, laps).await
    }

    /// Predictions as a Flight response: class probabilities or quantiles,
    /// with their worker time in the schema's metadata.
    async fn predicted(
        &self,
        predictions: &Predictions,
        levels: Option<&[f64]>,
        placement: &Placement,
        laps: Laps,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let batch = match levels {
            Some(levels) => quantile_batch(predictions, levels, placement)?,
            None => prediction_batch(predictions, placement.task)?,
        };
        answered(batch, predictions, laps, Arc::clone(&self.metrics))
    }

    async fn act(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Boxed<arrow_flight::Result>>, Status> {
        let parsed = FlightAction::parse(request.get_ref())?;
        if parsed.needs_replication() {
            replication(&request)?;
        }
        if parsed.needs_lease() {
            self.leased()?;
        }
        let fence = self.fence(&request)?;
        let scope = Scope::of(&request);
        parsed.check_scope(&scope)?;
        let reply = match parsed {
            FlightAction::Health => results(vec![json!("ok")]),
            FlightAction::ListTables => {
                let mut names: Vec<TableName> = self
                    .placement
                    .read()
                    .await
                    .keys()
                    .filter(|name| scope.allows(name))
                    .cloned()
                    .collect();
                names.sort();
                results(vec![json!(names)])
            }
            FlightAction::ListModels => {
                let mut names = std::collections::BTreeSet::new();
                for shard in 0..self.shards.len() {
                    names.extend(self.ask(shard, |reply| Command::Models { reply }).await?);
                }
                results(vec![json!(names)])
            }
            FlightAction::Locate(_) => {
                return Err(Status::unimplemented("locate is a router's action"));
            }
            FlightAction::Placements => {
                let mut tables: Vec<(TableName, usize)> = self
                    .placement
                    .read()
                    .await
                    .iter()
                    .filter(|(name, _)| scope.allows(name))
                    .map(|(name, placed)| (name.clone(), placed.shard))
                    .collect();
                tables.sort();
                let loads = self.mover.as_ref().map(|mover| {
                    mover
                        .shard_loads
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .clone()
                });
                results(vec![json!({
                    "shards": self.shards.len(),
                    "tables": tables.into_iter().collect::<BTreeMap<_, _>>(),
                    "shard_loads": loads,
                    "moves": self.metrics.moves.get(),
                })])
            }
            FlightAction::MoveTable { table, shard } => {
                let seconds = self.move_table(&table, shard).await?;
                results(vec![json!({"seconds": seconds})])
            }
            FlightAction::Rebalance => {
                let round = self.rebalance_round().await?;
                results(vec![json!(round)])
            }
            FlightAction::ProfileWorkers {
                seconds,
                directory,
                trace,
            } => {
                let Some(root) = &self.profiles else {
                    return Err(Status::failed_precondition(
                        "profile_workers needs [profiling] in the server's configuration",
                    ));
                };
                let directory = profile_directory(root, &directory)?;
                let mut files = Vec::new();
                let mut traces = Vec::new();
                for shard in 0..self.shards.len() {
                    let path = directory.join(format!("worker-{shard}.txt"));
                    let path = path.to_string_lossy().into_owned();
                    let trace = trace.then(|| {
                        directory
                            .join(format!("worker-{shard}"))
                            .to_string_lossy()
                            .into_owned()
                    });
                    self.ask(shard, |reply| Command::Record {
                        seconds,
                        path: path.clone(),
                        trace: trace.clone(),
                        reply,
                    })
                    .await?;
                    files.push(path);
                    traces.extend(trace);
                }
                results(vec![
                    json!({"seconds": seconds, "files": files, "traces": traces}),
                ])
            }
            FlightAction::CreateTable(definition) => {
                let split = serde_json::from_slice::<serde_json::Value>(&request.get_ref().body)
                    .ok()
                    .and_then(|body| body.get("partitions").and_then(serde_json::Value::as_u64))
                    .is_some_and(|partitions| partitions != 1);
                if split {
                    return Err(invalid(
                        "a table splits into partitions only through a router",
                    ));
                }
                if crate::engine::is_bound(&definition.name) {
                    return Err(invalid(format!(
                        "table names starting with {:?} are the Models API's own",
                        crate::engine::BOUND_PREFIX
                    )));
                }
                let mut definition = *definition;
                let created = creation_id()?;
                definition.created = Some(created.clone());
                let name = definition.name.clone();
                self.create(definition, Terms::default()).await?;
                let answer = Answer {
                    rows: None,
                    log_length: Some(1),
                    created: Some(created),
                };
                let mut answer = json!(answer);
                answer["table"] = json!(name);
                results(vec![answer])
            }
            FlightAction::Delete(DeleteAction {
                table,
                keys,
                missing_ok,
                write_id,
            }) => {
                // The keys are held until the shard answers, as a put's rows are.
                let mut charge = self.budget.charge();
                charge.grow((keys.len() * std::mem::size_of::<u128>()) as u64)?;
                let name = table.clone();
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::Write {
                        table,
                        write: Write::Delete {
                            keys: keys.into_iter().map(|key| key as u128).collect(),
                            missing_ok,
                            id: write_id,
                        },
                        fence,
                        charge,
                        reply,
                    })
                    .await?;
                if let Some(counts) = self.metrics.existing_table(&name) {
                    counts.rows_written.add(written.rows as u64);
                }
                results(vec![answer(&written, &placement)])
            }
            FlightAction::Describe { table, words } => {
                let (
                    Description {
                        definition,
                        rows,
                        log_length,
                        outputs,
                        levels,
                        strategy,
                        context_rows,
                        effective_fit_rows,
                        regions,
                        vocabularies,
                        labels,
                        series,
                    },
                    _,
                ) = self
                    .at_table(table, |table, reply| Command::Describe { table, reply })
                    .await?;
                let words: serde_json::Map<String, serde_json::Value> = definition
                    .columns
                    .iter()
                    .zip(vocabularies)
                    .filter(|(column, _)| words && column.kind == ColumnKind::Categorical)
                    .map(|(column, words)| (column.name.clone(), json!(words)))
                    .collect();
                let mut described = serde_json::to_value(definition)
                    .map_err(|error| Status::internal(error.to_string()))?;
                described["vocabularies"] = json!(words);
                described["labels"] = json!(labels);
                described["rows"] = json!(rows);
                described["log_length"] = json!(log_length);
                described["outputs"] = json!(outputs);
                described["quantile_levels"] = json!(levels);
                described["context_rows"] = json!(context_rows);
                described["resolved_strategy"] = json!(strategy);
                if let Some(rows) = effective_fit_rows {
                    described["effective_fit_rows"] = json!(rows);
                }
                if let Some(count) = series {
                    described["series_count"] = json!(count);
                }
                described["region_stats"] = json!(regions.map(|stats| json!({
                    "regions": stats.regions,
                    "mean_rows": stats.mean_rows,
                    "levels": stats.levels,
                    "hits": stats.hits,
                    "stale": stats.stale,
                    "misses": stats.misses,
                    "fresh": stats.fresh,
                    "rebuilds": stats.rebuilds,
                    "rebuilt_rows": stats.rebuilt_rows,
                    "overdue_seconds": stats.overdue.as_secs_f64(),
                })));
                results(vec![described])
            }
            FlightAction::KnownSeries(table) => {
                let (keys, _) = self
                    .at_table(table, |table, reply| Command::KnownSeries { table, reply })
                    .await?;
                results(vec![json!(keys)])
            }
            FlightAction::Truncate(table) => {
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::Write {
                        table,
                        write: Write::Truncate,
                        fence,
                        charge: Charge::default(),
                        reply,
                    })
                    .await?;
                results(vec![answer(&written, &placement)])
            }
            FlightAction::Refit(table) => {
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::Refit {
                        table,
                        fence,
                        reply,
                    })
                    .await?;
                results(vec![answer(&written, &placement)])
            }
            FlightAction::SetStrategy { table, strategy } => {
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::SetStrategy {
                        table,
                        strategy,
                        fence,
                        reply,
                    })
                    .await?;
                results(vec![answer(&written, &placement)])
            }
            FlightAction::LogLengths => {
                let mut lengths = LogLengths::new();
                for shard in 0..self.shards.len() {
                    for LogLength {
                        table,
                        length,
                        terms,
                        created,
                    } in self
                        .ask(shard, |reply| Command::LogLengths { reply })
                        .await?
                    {
                        if scope.allows(&table) {
                            let copy = CopyLength {
                                length,
                                terms,
                                created,
                            };
                            lengths.insert(table.into(), copy);
                        }
                    }
                }
                results(vec![json!(lengths)])
            }
            FlightAction::Compact(table) => {
                let (log_length, _) = self
                    .at_table(table, |table, reply| Command::Compact {
                        table,
                        fence,
                        reply,
                    })
                    .await?;
                results(vec![json!({"log_length": log_length})])
            }
            FlightAction::DropTable(table) => {
                self.release(table, false).await?;
                results(vec![json!({})])
            }
            FlightAction::Release { table, keep } => {
                self.release(table, keep).await?;
                results(vec![json!({})])
            }
            FlightAction::Promote {
                table,
                term,
                primary,
            } => {
                let (log_length, _) = self
                    .at_table(table, |table, reply| Command::Promote {
                        table,
                        term,
                        primary,
                        reply,
                    })
                    .await?;
                results(vec![json!({"log_length": log_length})])
            }
            FlightAction::Lease(request) => {
                let mut duration = Duration::from_secs_f64(request.seconds.clamp(0.0, 3600.0));
                if let Some(sent) = request.sent_unix_ms {
                    // The lease ends as long after the router sent it, by its clock.
                    let age = Duration::from_millis(unix_ms().saturating_sub(sent));
                    duration = duration.checked_sub(age).ok_or_else(|| {
                        Status::failed_precondition(
                            "a lease request arrived after the lease would have ended; \
                             the node was slow to read it, or its clock and the router's disagree",
                        )
                    })?;
                }
                let lease = self.lease.as_ref();
                let other = lease.and_then(|lease| lease.held_by_another(&request.holder));
                let renewed = match (lease, request.probe) {
                    (Some(lease), false) => lease.renew(&request.holder, duration),
                    _ => other.map_or(Ok(()), Err),
                };
                let answer = LeaseAnswer {
                    granted: renewed.is_ok(),
                    holder: renewed.err(),
                    enforced: lease.is_some(),
                };
                results(vec![json!(answer)])
            }
        };
        Ok(Response::new(reply))
    }
}

#[tonic::async_trait]
impl FlightService for Service {
    type HandshakeStream = Boxed<HandshakeResponse>;
    type ListFlightsStream = Boxed<FlightInfo>;
    type DoGetStream = Boxed<FlightData>;
    type DoPutStream = Boxed<PutResult>;
    type DoActionStream = Boxed<arrow_flight::Result>;
    type ListActionsStream = Boxed<ActionType>;
    type DoExchangeStream = Boxed<FlightData>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented(
            "authenticate with a bearer token on each call",
        ))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("use the list_tables action"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        replication(&request)?;
        let scope = Scope::of(&request);
        let LogTicket { table, from } = serde_json::from_slice(&request.into_inner().ticket)
            .map_err(|error| invalid(format!("bad log ticket: {error}")))?;
        scope.check(&table)?;
        let (placed, _routed) = self.routed(&table).await?;
        let shard = placed.shard;
        // Entries come a page at a time, so a long log never sits in memory at once.
        let service = self.clone();
        let pages = stream::try_unfold(Some(from), move |next| {
            let (service, table) = (service.clone(), table.clone());
            async move {
                let Some(from) = next else {
                    return Ok::<_, Status>(None);
                };
                let entries = service
                    .ask(shard, |reply| Command::ReadLog {
                        table,
                        from,
                        limit: READ_BATCH,
                        bytes: READ_BATCH_BYTES,
                        reply,
                    })
                    .await?;
                // A page cut short by its bounds has more behind it; an empty one ends the log.
                let next = entries.last().map(|(sequence, _)| sequence + 1);
                let messages: Vec<Result<FlightData, Status>> = entries
                    .into_iter()
                    .flat_map(|(sequence, entry)| {
                        let entry = bytes::Bytes::from(entry);
                        let parts = entry.len().div_ceil(ENTRY_PART).max(1);
                        (0..parts)
                            .map(|part| {
                                let end = ((part + 1) * ENTRY_PART).min(entry.len());
                                let last = part + 1 == parts;
                                Ok(FlightData {
                                    app_metadata: json!({"sequence": sequence, "last": last})
                                        .to_string()
                                        .into(),
                                    data_body: entry.slice(part * ENTRY_PART..end),
                                    ..Default::default()
                                })
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                Ok(Some((stream::iter(messages), next)))
            }
        })
        .try_flatten();
        Ok(Response::new(Box::pin(pages)))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let trace = Trace::of(&request);
        traced("put", &trace, self.put(request)).await
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let trace = Trace::of(&request);
        traced("action", &trace, self.act(request)).await
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = [
            "create_table",
            "describe",
            "known_series",
            "delete",
            "truncate",
            "refit",
            "set_strategy",
            "compact",
            "drop_table",
            "list_tables",
            "list_models",
            "health",
            "log_lengths",
            "release",
            "promote",
            "lease",
            "placements",
            "move_table",
            "rebalance",
            "profile_workers",
        ]
        .into_iter()
        .map(|name| {
            Ok(ActionType {
                r#type: name.to_string(),
                description: String::new(),
            })
        });
        Ok(Response::new(Box::pin(stream::iter(
            actions.collect::<Vec<_>>(),
        ))))
    }

    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        let trace = Trace::of(&request);
        traced("predict", &trace, self.exchange(request, &trace)).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use futures::{StreamExt, executor, stream};

    use super::Timed;

    /// The times a `Timed` stream recorded.
    type Recorded = Arc<Mutex<Vec<Duration>>>;

    /// A stream of three items that counts how many have been pulled, and
    /// the encode times its `Timed` wrapper recorded.
    fn timed() -> (
        impl futures::Stream<Item = Result<usize, String>> + Unpin,
        Arc<AtomicUsize>,
        Recorded,
    ) {
        let pulled = Arc::new(AtomicUsize::new(0));
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let counter = Arc::clone(&pulled);
        let items = stream::iter([Ok(0), Err("bad".to_string()), Ok(2)])
            .inspect(move |_| _ = counter.fetch_add(1, Ordering::SeqCst));
        let sink = Arc::clone(&recorded);
        let stream = Timed::new(items, Duration::from_secs(1), move |spent| {
            sink.lock().unwrap().push(spent);
        });
        (stream, pulled, recorded)
    }

    #[test]
    fn timed_streams_items_in_order_and_records_once_at_the_end() {
        let (mut stream, pulled, recorded) = timed();
        assert_eq!(pulled.load(Ordering::SeqCst), 0);

        let mut seen = Vec::new();
        while let Some(item) = executor::block_on(stream.next()) {
            assert_eq!(pulled.load(Ordering::SeqCst), seen.len() + 1);
            assert!(recorded.lock().unwrap().is_empty());
            seen.push(item);
        }
        assert_eq!(seen, [Ok(0), Err("bad".to_string()), Ok(2)]);
        assert_eq!(recorded.lock().unwrap().len(), 1);
        assert!(recorded.lock().unwrap()[0] >= Duration::from_secs(1));

        drop(stream);
        assert_eq!(recorded.lock().unwrap().len(), 1);
    }

    #[test]
    fn timed_records_once_when_dropped_early() {
        let (mut stream, pulled, recorded) = timed();
        assert_eq!(executor::block_on(stream.next()), Some(Ok(0)));
        assert!(recorded.lock().unwrap().is_empty());

        drop(stream);
        assert_eq!(pulled.load(Ordering::SeqCst), 1);
        assert_eq!(recorded.lock().unwrap().len(), 1);
    }
}
