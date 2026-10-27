// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The router's Flight service: each call goes to the nodes that hold its table.
//!
//! Arrow streams pass through without being decoded. A table keeps `replicas`
//! copies on the nodes that rank highest for it. Writes go to the first, its
//! primary, and reach the other replicas as log entries before the client hears
//! back; a write that changes rows is acknowledged only once a replica holds
//! it. Reads go to the first copy that is up and current, and move on if a
//! node cannot be reached. A background check marks nodes up or down, ships log
//! entries to replicas that are behind or new, and releases copies no replica
//! needs.
//!
//! A table's writer changes only under a new term, so a writer that returns
//! can never merge writes it took unacknowledged with its successor's: its
//! copy diverges from the new term and is set aside. Its first write names
//! the table's top-ranked node its primary, under term 1. When nodes are
//! added or removed and the primary is no longer among the nodes the table
//! ranks first, a router hands the table off: the new node catches up, then
//! the primary itself begins a term naming it, which fences the primary
//! against writes of the old term from any router. A primary that left the
//! configuration takes no writes until it returns or is failed over.
//!
//! The router with `failover_seconds` alone renews each node's write lease,
//! under its name, and a node renews a lease only for the router holding it
//! until it ends, so no two routers keep one. Once a primary has been out of
//! reach for `failover_seconds`, its lease has lapsed and it takes no more
//! writes, so the router promotes the first replica that holds every
//! acknowledged write, under a later term. Since a write is acknowledged once
//! a replica holds it, that needs every other replica up. A router that finds
//! another holding leases stands down, since only one may fail over; it asks
//! again from time to time, and takes over once no other holds a lease.
//!
//! Every router sends writes to the primary the latest term names, under that
//! term, and leaves out copies whose writes diverged from it, which the
//! failing-over router sets aside. A router learns every table's copies before
//! it serves, and refuses writes to a table whose term it does not know, so a
//! router that restarts never writes to a primary since replaced.
//!
//! Dropping a table needs every node up, so no copy survives to be repaired
//! back, and waits for the table's writes and shipments in flight to finish.
//! Each table has a lifecycle lock of its own, and a shipment holds it only for
//! itself, so a drop waits for one of its table's shipments, and holds up only
//! that table's writes. A shipment for a write takes at most the node's call
//! timeout; one for a repair, `SHIP_TIMEOUT`, counted from its first request.

mod copies;
mod failover;
mod forward;
mod leases;
mod moves;
mod node;
mod partitioned;
mod replicas;

use std::collections::{BTreeSet, HashMap};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Duration;

use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::{Action, PutResult};
use futures::Stream;
use gf_server::action::{Answer, FlightAction};
use gf_server::auth::{Scope, bearer, digest};
use gf_server::metrics::Counter;
use gf_server::service::MESSAGE_BYTES;
use gf_server::trace::Trace;
use gf_server::wal::{CreationId, TableDefinition};
use serde_json::{Value, json};
use tokio::time::Instant;
use tonic::metadata::MetadataMap;
use tonic::service::interceptor::InterceptedService;
use tonic::{Code, Request, Status};

use crate::quota::Quotas;
use crate::replication::TableCopy;
use copies::{Copies, Seen};
use failover::Failover;
pub use node::{Cluster, Node, Timeouts};

type Boxed<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

tokio::task_local! {
    /// The trace of the client call being forwarded, which node calls carry on.
    static TRACE: Trace;
}

/// How long a node may take to answer a check before it counts as down.
const CHECK_TIMEOUT: Duration = Duration::from_secs(5);

/// How long one shipment of log entries may take, however long the log.
const SHIP_TIMEOUT: Duration = Duration::from_secs(600);

/// Tables a check repairs at once, so one slow shipment holds up no others.
const REPAIRS: usize = 8;

/// A table name held while it is made; the name's lock is forgotten once
/// no create holds or waits for it.
struct Creating {
    held: Option<tokio::sync::OwnedMutexGuard<()>>,
    names: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    table: String,
}

impl Drop for Creating {
    fn drop(&mut self) {
        drop(self.held.take());
        let mut names = self.names.lock().unwrap_or_else(PoisonError::into_inner);
        if names
            .get(&self.table)
            .is_some_and(|lock| Arc::strong_count(lock) == 1)
        {
            names.remove(&self.table);
        }
    }
}

/// The client tokens by digest, the tables each reaches, and their quotas.
struct Clients {
    tokens: HashMap<[u8; 32], Scope>,
    quotas: Quotas,
}

/// The router's Flight service.
#[derive(Clone)]
pub struct Router {
    cluster: Arc<RwLock<Arc<Cluster>>>,
    copies: Arc<Mutex<Copies>>,
    /// The creation each table's name serves, by table: the one this router
    /// created or chose, kept while a copy of it remains.
    creations: Arc<Mutex<HashMap<String, CreationId>>>,
    /// The longest log seen on a replica of each table's creation, so every
    /// acknowledged write; writes on the primary alone were never acknowledged.
    seen: Arc<Mutex<HashMap<String, Seen>>>,
    /// Per table, held shared by writes and shipments, and exclusively by drops.
    lifecycles: Arc<Mutex<HashMap<String, Arc<tokio::sync::RwLock<()>>>>>,
    /// Tables a repair is running for, so a check starts at most one each.
    repairing: Arc<Mutex<BTreeSet<String>>>,
    /// Repairs that may run at once.
    repairs: Arc<tokio::sync::Semaphore>,
    /// Replicas promoted for lost primaries, and tables handed to new nodes.
    promotions: Arc<Counter>,
    moves: Arc<Counter>,
    failover: Option<Failover>,
    clients: Arc<RwLock<Arc<Clients>>>,
    /// Each partitioned table's definition, as its first partition described
    /// it, with the copy of that partition it was read from: the strategy
    /// its predictions assemble by, and its route if it has one.
    definitions: Arc<Mutex<HashMap<String, (TableCopy, TableDefinition)>>>,
    /// Per table name, held while it is made, so two creates of one name
    /// through this router never both pass the check that it is free.
    creating: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl Router {
    /// A router over a cluster.
    #[must_use]
    pub fn new(cluster: Cluster) -> Self {
        Self {
            cluster: Arc::new(RwLock::new(Arc::new(cluster))),
            copies: Arc::default(),
            creations: Arc::default(),
            seen: Arc::default(),
            lifecycles: Arc::default(),
            repairing: Arc::default(),
            repairs: Arc::new(tokio::sync::Semaphore::new(REPAIRS)),
            promotions: Arc::default(),
            moves: Arc::default(),
            failover: None,
            clients: Arc::new(RwLock::new(Arc::new(Clients {
                tokens: HashMap::new(),
                quotas: Quotas::new(HashMap::new()),
            }))),
            definitions: Arc::default(),
            creating: Arc::default(),
        }
    }

    /// Sets the client tokens, by digest, each reaching the tables its scope
    /// names, and their quotas, as at start and when the configuration is reloaded.
    /// A token's bucket goes on from what it held, so a reload refills none.
    pub fn set_clients(&self, tokens: HashMap<[u8; 32], Scope>, quotas: Quotas) {
        let mut clients = self.clients.write().unwrap_or_else(PoisonError::into_inner);
        let quotas = quotas.continuing(&clients.quotas);
        *clients = Arc::new(Clients { tokens, quotas });
    }

    /// Promotes a replica once a primary has been out of reach for `after`,
    /// renewing every node's write lease for `lease`, shorter than `after`,
    /// under the name `holder`. Only one router in a cluster may do this.
    #[must_use]
    pub fn with_failover(mut self, after: Duration, lease: Duration, holder: String) -> Self {
        self.failover = Some(Failover {
            after,
            lease,
            holder,
            leased: Arc::default(),
            since: Arc::new(Mutex::new(Instant::now())),
            contested: Arc::default(),
        });
        self
    }

    /// Learns every table's copies, as a check does: a router serves only once
    /// it knows each table's term, which every write names.
    pub async fn prepare(&self) {
        self.check().await;
    }

    /// Replaces the cluster, as when the configuration is reloaded.
    pub fn reconfigure(&self, cluster: Cluster) {
        *self.cluster.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(cluster);
        if let Some(failover) = &self.failover {
            *failover
                .since
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Instant::now();
        }
    }

    /// The router's metrics, in Prometheus's text format: whether each node
    /// answered the last check, whether this router stands down, repairs
    /// running, tables, replicas behind their table's longest copy, and the
    /// promotions and moves made.
    #[must_use]
    pub fn metrics(&self) -> String {
        use std::fmt::Write;
        let cluster = self.cluster();
        let tables: Vec<String> = self
            .copies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .names()
            .cloned()
            .collect();
        let behind: usize = tables
            .iter()
            .map(|table| {
                let (lengths, _) = self.view(&cluster, table);
                let longest = lengths.iter().flatten().max().copied();
                self.desired(&cluster, table)
                    .into_iter()
                    .filter(|&node| lengths[node] < longest)
                    .count()
            })
            .sum();
        let standing_down = self
            .failover
            .as_ref()
            .is_some_and(|failover| failover.contested.load(Ordering::Relaxed));
        let repairing = self
            .repairing
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len();
        let mut text = String::new();
        let _ = writeln!(
            text,
            "# HELP gf_router_node_up Whether the node answered the last check.\n\
             # TYPE gf_router_node_up gauge"
        );
        for node in &cluster.nodes {
            let up = u8::from(node.checked.load(Ordering::Relaxed));
            let _ = writeln!(text, "gf_router_node_up{{node={:?}}} {up}", node.name);
        }
        let values: [(&str, &str, &str, u64); 6] = [
            (
                "gf_router_standing_down",
                "gauge",
                "Whether another router fails tables over instead of this one.",
                u64::from(standing_down),
            ),
            (
                "gf_router_repairs_running",
                "gauge",
                "Tables whose copies are being repaired.",
                repairing as u64,
            ),
            (
                "gf_router_tables",
                "gauge",
                "Tables the last check found.",
                tables.len() as u64,
            ),
            (
                "gf_router_replicas_behind",
                "gauge",
                "Replicas whose log is shorter than their table's longest copy.",
                behind as u64,
            ),
            (
                "gf_router_promotions_total",
                "counter",
                "Replicas promoted for primaries out of reach.",
                self.promotions.get(),
            ),
            (
                "gf_router_moves_total",
                "counter",
                "Tables whose writes moved to the node they rank first.",
                self.moves.get(),
            ),
        ];
        for (name, kind, help, value) in values {
            let _ = writeln!(
                text,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
            );
        }
        text
    }

    fn cluster(&self) -> Arc<Cluster> {
        self.cluster
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The service behind client token and quota checks, as `set_clients` last set them.
    #[must_use]
    pub fn into_server(
        self,
    ) -> InterceptedService<FlightServiceServer<Self>, impl tonic::service::Interceptor + Clone>
    {
        let clients = self.clients.clone();
        let server = FlightServiceServer::new(self)
            .max_decoding_message_size(MESSAGE_BYTES)
            .max_encoding_message_size(MESSAGE_BYTES);
        InterceptedService::new(server, move |mut request: Request<()>| {
            let clients = clients
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            let token = bearer(&request)
                .map(digest)
                .ok_or_else(|| Status::unauthenticated("a valid bearer token is required"))?;
            let Some(scope) = clients.tokens.get(&token).cloned() else {
                return Err(Status::unauthenticated("a valid bearer token is required"));
            };
            clients
                .quotas
                .check(&token)
                .map_err(Status::resource_exhausted)?;
            request.extensions_mut().insert(scope);
            Ok(request)
        })
    }

    /// A table's lifecycle lock.
    fn lifecycle(&self, table: &str) -> Arc<tokio::sync::RwLock<()>> {
        self.lifecycles
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(table.to_string())
            .or_default()
            .clone()
    }

    /// Forgets a dropped table's lifecycle lock, unless a call still holds it.
    fn forget_lifecycle(&self, table: &str) {
        let mut lifecycles = self
            .lifecycles
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if lifecycles
            .get(table)
            .is_some_and(|lock| Arc::strong_count(lock) == 1)
        {
            lifecycles.remove(table);
        }
    }

    /// Holds the name `table` while a create of it runs; another create of
    /// the name waits for it, then finds the name taken or free.
    async fn creating(&self, table: &str) -> Creating {
        let lock = self
            .creating
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(table.to_string())
            .or_default()
            .clone();
        Creating {
            held: Some(lock.lock_owned().await),
            names: self.creating.clone(),
            table: table.to_string(),
        }
    }

    /// Calls a write action on the table's primary, then replicates it.
    async fn write_action(
        &self,
        table: &str,
        action: &Action,
        parsed: &FlightAction,
    ) -> Result<Vec<Value>, Status> {
        // A new table has no term yet; it begins at term 0 on its primary.
        let creates = matches!(parsed, FlightAction::CreateTable(_));
        if !creates {
            self.name_writer(table).await?;
        }
        let _writing = self.lifecycle(table).read_owned().await;
        let cluster = self.cluster();
        let term = if creates {
            None
        } else {
            Some(self.known_term(&cluster, table)?)
        };
        let primary = self.primary(&cluster, table).await?;
        let node = &cluster.nodes[primary];
        let request = node.write(action.clone(), term, &MetadataMap::new());
        // Deletes wait for no fit; the other actions may run long.
        let rows = parsed.changes_rows();
        let timeout = if rows {
            node.write_timeout
        } else {
            node.timeout
        };
        let results = node
            .call_within(request, timeout)
            .await
            .map_err(|status| write_failed(table, node, status))?;
        let answer = results.first().map(Answer::of).unwrap_or_default();
        if let Some(length) = answer.log_length {
            // Deletes change rows, and a create makes the table: a replica
            // must hold each before it is acknowledged, or a primary lost
            // before the next check takes it along. A refit or a compaction
            // loses nothing with its primary, so its replicas catch up later.
            let confirm = rows || creates;
            self.replicate(&cluster, table, primary, length, confirm, answer.created)
                .await?;
        }
        Ok(results)
    }

    /// Calls a read action on the first reachable copy of the table.
    async fn read_action(&self, table: &str, action: &Action) -> Result<Vec<Value>, Status> {
        let cluster = self.cluster();
        let body: Value = serde_json::from_slice(&action.body).unwrap_or(Value::Null);
        let mut failure = None;
        for index in self.read_order(&cluster, table) {
            match cluster.nodes[index]
                .action(&action.r#type, body.clone())
                .await
            {
                Err(status) if status.code() == Code::Unavailable => failure = Some(status),
                answer => return answer,
            }
        }
        Err(no_copy(table, failure))
    }

    /// Drops a table's copy on every node.
    async fn drop_table(&self, table: &str) -> Result<Vec<Value>, Status> {
        // A table no copy holds is not worth stopping every write and repair for.
        let known = self
            .copies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(table);
        if !known {
            let describe = action("describe", &json!({"table": table}));
            if let Err(status) = self.read_action(table, &describe).await
                && status.code() == Code::NotFound
            {
                return Err(status);
            }
        }
        let dropping = self.lifecycle(table).write_owned().await;
        let cluster = self.cluster();
        let health = futures::future::join_all(cluster.nodes.iter().map(|node| async move {
            tokio::time::timeout(CHECK_TIMEOUT, node.action("health", json!({})))
                .await
                .unwrap_or_else(|_| Err(Status::unavailable("timed out")))
        }))
        .await;
        for (node, answer) in cluster.nodes.iter().zip(health) {
            if let Err(status) = answer {
                node.note(&status);
                return Err(Status::unavailable(format!(
                    "dropping table {table:?} needs every node up, and {} is unreachable",
                    node.name
                )));
            }
        }
        let answers = futures::future::join_all(
            cluster
                .nodes
                .iter()
                .map(|node| node.action("drop_table", json!({"table": table}))),
        )
        .await;
        self.copies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drop_table(table);
        self.creations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(table);
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(table);
        let mut dropped = false;
        for (node, answer) in cluster.nodes.iter().zip(answers) {
            match answer {
                Ok(_) => dropped = true,
                Err(status) if status.code() == Code::NotFound => {}
                // Copies left behind are dropped when the drop is retried.
                Err(status) => {
                    node.note(&status);
                    return Err(status);
                }
            }
        }
        drop(dropping);
        self.forget_lifecycle(table);
        if dropped {
            Ok(vec![json!({})])
        } else {
            Err(Status::not_found(format!("table {table:?} does not exist")))
        }
    }
}

fn action(kind: &str, body: &Value) -> Action {
    Action {
        r#type: kind.to_string(),
        body: body.to_string().into(),
    }
}

/// What a node answered a `DoPut` with, or an empty answer if it is not one.
fn put_answer(results: &[PutResult]) -> Answer {
    results
        .first()
        .and_then(|result| serde_json::from_slice(&result.app_metadata).ok())
        .unwrap_or_default()
}

fn unavailable(table: &str, node: &Node, reason: &str) -> Status {
    Status::unavailable(format!(
        "table {table:?} is unavailable for writes: its primary {} is unreachable ({reason}); \
         writes resume when it returns, or once the router with failover_seconds promotes \
         a replica",
        node.name
    ))
}

fn write_failed(table: &str, node: &Node, status: Status) -> Status {
    if status.code() == Code::Unavailable {
        unavailable(table, node, status.message())
    } else {
        status
    }
}

fn no_copy(table: &str, failure: Option<Status>) -> Status {
    let reason = failure.map_or_else(
        || "no node holding it is up".to_string(),
        |status| status.message().to_string(),
    );
    Status::unavailable(format!("table {table:?} is unavailable: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_drop_holds_up_only_its_own_tables_calls() {
        let router = Router::new(Cluster::new(Vec::new(), 1));
        let dropping = router.lifecycle("a").write_owned().await;

        // Another table's writes go ahead, and the dropped table's wait.
        assert!(router.lifecycle("b").try_read_owned().is_ok());
        assert!(router.lifecycle("a").try_read_owned().is_err());

        let waiting = router.lifecycle("a");
        drop(dropping);
        router.forget_lifecycle("a");
        assert!(router.lifecycles.lock().unwrap().contains_key("a"));
        drop(waiting);
        router.forget_lifecycle("a");
        assert!(!router.lifecycles.lock().unwrap().contains_key("a"));
    }
}
