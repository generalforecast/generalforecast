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
use crate::definition::{CreationId, Durability, TableDefinition};
use crate::error::ServerError;
/// The Flight status for a name taken, where the router finds it.
pub use crate::error::exists;
use crate::metrics::{Histogram, Metrics};
use crate::name::TableName;
use crate::partition;
use crate::protocol::{Predictions, StrategySpec};
use crate::quantiles::{self, quantile_values};
use crate::shard::{
    Command, Description, Fence, Incoming, KnownFuture, Lease, Lineage, LogLength, Write,
};
use crate::trace::{Trace, traced};
use crate::wal::{GivenKey, LogEntry, Rows, SeriesKey, Terms, decode};
use gf_core::columns::{self, Cells, Column, ColumnKind, Strings};

mod calls;
mod frames;
mod mover;
mod replicas;

use frames::{Laps, invalid};

pub use mover::{Mover, Round};

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

/// The schema metadata that carries a prediction's share of its step's
/// worker time, in seconds, so a client can attribute cost per request.
pub const GPU_SECONDS: &str = "generalforecast.gpu_seconds";

/// The schema metadata that carries the epoch of the table's state that
/// answered, which rises with each refit, so a stream can mark refits.
pub const EPOCH: &str = "generalforecast.epoch";

/// The header a forecast's horizon travels in: steps, at least one.
pub const HORIZON: &str = "gf-horizon";

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
