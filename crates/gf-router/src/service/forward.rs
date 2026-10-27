// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Forwarding clients' calls: their streams to the nodes that hold their
//! tables, and their actions to the router's own handling.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use arrow_flight::flight_service_server::FlightService;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::{Stream, StreamExt, TryStreamExt, stream};
use gf_server::action::{DeleteAction, FlightAction, PutCommand};
use gf_server::auth::{QUANTILES, Scope};
use gf_server::name::TableName;
use gf_server::service::HORIZON;
use gf_server::trace::{Trace, traced};
use serde_json::{Value, json};
use tokio::sync::{oneshot, watch};
use tokio::time::Instant;
use tonic::{Code, Request, Response, Status, Streaming};

use super::{Boxed, Router, TRACE, no_copy, put_answer, write_failed};
use crate::partitions;

/// A node's answer stream, cut off with `DEADLINE_EXCEEDED` at `deadline`, so a
/// node that stops partway never holds the client's call open.
pub(super) fn within(
    stream: Streaming<FlightData>,
    deadline: tokio::time::Instant,
    node: String,
) -> Boxed<FlightData> {
    Box::pin(stream::unfold(Some(stream), move |state| {
        let node = node.clone();
        async move {
            let mut stream = state?;
            match tokio::time::timeout_at(deadline, stream.message()).await {
                Ok(Ok(Some(message))) => Some((Ok(message), Some(stream))),
                Ok(Ok(None)) => None,
                Ok(Err(status)) => Some((Err(status), None)),
                Err(_) => Some((
                    Err(Status::deadline_exceeded(format!(
                        "{node} did not finish its answer in time"
                    ))),
                    None,
                )),
            }
        }
    }))
}

fn results(values: Vec<Value>) -> Boxed<arrow_flight::Result> {
    let items = values.into_iter().map(|value| {
        Ok(arrow_flight::Result {
            body: value.to_string().into(),
        })
    });
    Box::pin(stream::iter(items.collect::<Vec<_>>()))
}

/// A client's stream, forwardable to one node after another until one is reached.
///
/// The first message names the command and table; only the commands a call
/// allows pass, so a client cannot reach the node calls the router itself
/// makes, such as `replicate`. If the client's stream fails partway, the
/// forwarded stream never ends, so a node cannot mistake a cut stream for a
/// complete one; the error arrives on the receiver instead and the call to the
/// node is dropped.
pub(super) struct Forward {
    pub(super) command: String,
    pub(super) table: TableName,
    pub(super) first: FlightData,
    pub(super) rest: Arc<tokio::sync::Mutex<Streaming<FlightData>>>,
    pub(super) failed: Arc<Mutex<Option<oneshot::Sender<Status>>>>,
    /// Whether a node has read past the first message, after which the stream cannot be replayed.
    pub(super) started: Arc<AtomicBool>,
    /// When a message last passed to the node, or the stream ended.
    pub(super) progress: Arc<watch::Sender<Instant>>,
}

impl Forward {
    pub(super) async fn new(
        mut stream: Streaming<FlightData>,
        accepts: fn(&str) -> bool,
    ) -> Result<(Self, oneshot::Receiver<Status>), Status> {
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("an empty stream"))?;
        let path = first
            .flight_descriptor
            .as_ref()
            .map(|descriptor| descriptor.path.clone())
            .unwrap_or_default();
        let [command, table] = path.as_slice() else {
            return Err(Status::invalid_argument(
                "the descriptor path must be [command, table]",
            ));
        };
        if !accepts(command) {
            return Err(Status::invalid_argument(format!(
                "unknown command {command:?}"
            )));
        }
        let table = TableName::new(table.as_str())
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let (failed, failure) = oneshot::channel();
        let forward = Self {
            command: command.clone(),
            table,
            first,
            rest: Arc::new(tokio::sync::Mutex::new(stream)),
            failed: Arc::new(Mutex::new(Some(failed))),
            started: Arc::default(),
            progress: Arc::new(watch::Sender::new(Instant::now())),
        };
        Ok((forward, failure))
    }

    pub(super) fn started(&self) -> bool {
        self.started.load(Ordering::Relaxed)
    }

    /// Resolves once the write has made no progress for `timeout`: no message
    /// of the client's passed to the node, or, once all had, the node did not
    /// answer. A node that stops reading partway holds the client no longer.
    pub(super) async fn stalled(&self, timeout: Duration) {
        let mut progress = self.progress.subscribe();
        loop {
            let last = *progress.borrow_and_update();
            tokio::select! {
                changed = progress.changed() => {
                    if changed.is_err() {
                        std::future::pending::<()>().await;
                    }
                }
                () = tokio::time::sleep_until(last + timeout) => return,
            }
        }
    }

    /// The stream to send one node: the first message again, then the rest of the client's.
    pub(super) fn messages(&self) -> impl Stream<Item = FlightData> + Send + 'static {
        let state = (
            self.rest.clone(),
            self.failed.clone(),
            self.started.clone(),
            self.progress.clone(),
        );
        let rest = stream::unfold(state, |(rest, failed, started, progress)| async move {
            started.store(true, Ordering::Relaxed);
            let message = rest.lock().await.message().await;
            progress.send_replace(Instant::now());
            match message {
                Ok(Some(message)) => Some((message, (rest, failed, started, progress))),
                Ok(None) => None,
                Err(status) => {
                    let sender = failed.lock().unwrap_or_else(PoisonError::into_inner).take();
                    if let Some(sender) = sender {
                        let _ = sender.send(status);
                    }
                    std::future::pending().await
                }
            }
        });
        stream::once(std::future::ready(self.first.clone())).chain(rest)
    }
}

impl Router {
    async fn put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<PutResult>>, Status> {
        let scope = Scope::of(&request);
        let metadata = request.metadata().clone();
        let client_write = |command: &str| {
            command
                .parse::<PutCommand>()
                .is_ok_and(PutCommand::is_client_write)
        };
        let (forward, failure) = Forward::new(request.into_inner(), client_write).await?;
        scope.check(&forward.table)?;
        let count = self.partition_count(&forward.table);
        if count > 0 {
            return self
                .put_partitioned(&forward, failure, &metadata, count)
                .await;
        }
        self.name_writer(&forward.table).await?;
        let _writing = self.lifecycle(&forward.table).read_owned().await;
        let table = forward.table.clone();
        let cluster = self.cluster();
        let term = self.known_term(&cluster, &table)?;
        let primary = self.primary(&cluster, &table).await?;
        let node = &cluster.nodes[primary];
        let request = node.write(forward.messages(), Some(term), &metadata);
        let call = async {
            node.client
                .clone()
                .do_put(request)
                .await?
                .into_inner()
                .try_collect::<Vec<PutResult>>()
                .await
        };
        let written = tokio::select! {
            written = call => written
                .inspect_err(|status| node.note(status))
                .map_err(|status| write_failed(&table, node, status))?,
            Ok(status) = failure => return Err(status),
            () = forward.stalled(node.write_timeout) => {
                return Err(Status::unavailable(format!(
                    "a write to {} made no progress for {} seconds; it applies at most once \
                     if retried",
                    node.name,
                    node.write_timeout.as_secs_f64()
                )));
            }
        };
        let answer = put_answer(&written);
        if let Some(length) = answer.log_length {
            self.replicate(&cluster, &table, primary, length, true, answer.created)
                .await?;
        }
        let written: Vec<Result<PutResult, Status>> = written.into_iter().map(Ok).collect();
        Ok(Response::new(Box::pin(stream::iter(written))))
    }

    async fn exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let scope = Scope::of(&request);
        let quantiles = request.metadata().get(QUANTILES).cloned();
        let horizon = request.metadata().get(HORIZON).cloned();
        let (forward, mut failure) = Forward::new(request.into_inner(), |command| {
            ["predict", "contexts", "region_churn", "forecast"].contains(&command)
        })
        .await?;
        scope.check(&forward.table)?;
        let count = self.partition_count(&forward.table);
        if count > 0 && forward.command == "contexts" {
            return Err(Status::invalid_argument(
                "a partitioned table's contexts are gathered for each prediction alone",
            ));
        }
        if count > 0 && forward.command == "region_churn" {
            return Err(Status::invalid_argument(
                "a partitioned table has no regions",
            ));
        }
        if count > 0 && forward.command == "forecast" {
            return Err(Status::invalid_argument(
                "a partitioned table holds rows to predict from, not series",
            ));
        }
        if count > 0 {
            return match self.route_of(&forward.table, count).await? {
                Some(route) => {
                    self.predict_routed(&forward, failure, quantiles, count, &route)
                        .await
                }
                None => {
                    self.predict_partitioned(&forward, failure, quantiles, count)
                        .await
                }
            };
        }
        let cluster = self.cluster();
        let mut last = None;
        for index in self.read_order(&cluster, &forward.table) {
            let node = &cluster.nodes[index];
            let deadline = tokio::time::Instant::now() + node.timeout;
            let mut client = node.client.clone();
            let mut request = node.request(forward.messages());
            if let Some(quantiles) = &quantiles {
                request.metadata_mut().insert(QUANTILES, quantiles.clone());
            }
            if let Some(horizon) = &horizon {
                request.metadata_mut().insert(HORIZON, horizon.clone());
            }
            let call = client.do_exchange(request);
            tokio::select! {
                response = call => match response {
                    Ok(response) => {
                        let answer = within(response.into_inner(), deadline, node.name.clone());
                        return Ok(Response::new(answer));
                    }
                    // Nothing past the first message was sent, so the next copy can take it.
                    Err(status) if status.code() == Code::Unavailable && !forward.started() => {
                        node.note(&status);
                        last = Some(status);
                    }
                    Err(status) => return Err(status),
                },
                Ok(status) = &mut failure => return Err(status),
                () = tokio::time::sleep_until(deadline) => {
                    return Err(Status::unavailable(format!("{} timed out", node.name)));
                }
            }
        }
        Err(no_copy(&forward.table, last))
    }

    async fn act(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Boxed<arrow_flight::Result>>, Status> {
        let scope = Scope::of(&request);
        let action = request.into_inner();
        let parsed = FlightAction::parse(&action)?;
        // Routers take the replication actions from no one: they make them.
        if parsed.needs_replication() {
            return Err(Status::unimplemented(format!(
                "unknown action {:?}",
                parsed.name()
            )));
        }
        parsed.check_scope(&scope)?;
        if let Some(answer) = self.act_partitioned(&action, &parsed).await? {
            return Ok(Response::new(results(answer)));
        }
        let answer = match &parsed {
            FlightAction::Health => vec![json!("ok")],
            FlightAction::Placements
            | FlightAction::MoveTable { .. }
            | FlightAction::Rebalance
            | FlightAction::ProfileWorkers { .. } => {
                return Err(Status::unimplemented(format!(
                    "{} acts on one server's shards; call the server",
                    parsed.name()
                )));
            }
            FlightAction::ListModels => {
                let cluster = self.cluster();
                let mut last = None;
                for node in cluster.nodes.iter().filter(|node| node.is_up()) {
                    match node.action("list_models", json!({})).await {
                        Ok(answer) => return Ok(Response::new(results(answer))),
                        Err(status) => last = Some(status),
                    }
                }
                return Err(last.unwrap_or_else(|| Status::unavailable("no node is reachable")));
            }
            FlightAction::ListTables => {
                let cluster = self.cluster();
                let answers = futures::future::join_all(
                    cluster
                        .nodes
                        .iter()
                        .filter(|node| node.is_up())
                        .map(|node| node.action("list_tables", json!({}))),
                )
                .await;
                if answers.iter().all(Result::is_err) {
                    return Err(Status::unavailable("no node is reachable"));
                }
                let names: BTreeSet<String> = answers
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter_map(|listed| serde_json::from_value::<Vec<String>>(listed).ok())
                    .flatten()
                    // A partitioned table lists once, by its own name.
                    .map(|name| {
                        partitions::parse(&name)
                            .map_or(name.clone(), |(table, _, _)| table.to_string())
                    })
                    .filter(|name| scope.allows(name))
                    .collect();
                vec![json!(names)]
            }
            FlightAction::Locate(table) => {
                let cluster = self.cluster();
                let names: Vec<&str> = self
                    .desired(&cluster, table)
                    .into_iter()
                    .map(|index| cluster.nodes[index].name.as_str())
                    .collect();
                vec![json!({"nodes": names})]
            }
            FlightAction::Describe { table, .. } => self.read_action(table, &action).await?,
            FlightAction::CreateTable(definition) => {
                self.write_action(&definition.name, &action, &parsed)
                    .await?
            }
            FlightAction::Delete(DeleteAction { table, .. })
            | FlightAction::Truncate(table)
            | FlightAction::Refit(table)
            | FlightAction::SetStrategy { table, .. }
            | FlightAction::Compact(table) => self.write_action(table, &action, &parsed).await?,
            FlightAction::DropTable(table) => self.drop_table(table).await?,
            FlightAction::LogLengths
            | FlightAction::Release { .. }
            | FlightAction::Promote { .. }
            | FlightAction::Lease(_) => unreachable!("refused above"),
        };
        Ok(Response::new(results(answer)))
    }
}

#[tonic::async_trait]
impl FlightService for Router {
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
        _request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        Err(Status::unimplemented("predict with DoExchange"))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let trace = Trace::of(&request).child();
        TRACE
            .scope(trace.clone(), traced("put", &trace, self.put(request)))
            .await
    }

    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        let trace = Trace::of(&request).child();
        TRACE
            .scope(
                trace.clone(),
                traced("predict", &trace, self.exchange(request)),
            )
            .await
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let trace = Trace::of(&request).child();
        TRACE
            .scope(trace.clone(), traced("action", &trace, self.act(request)))
            .await
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = [
            "create_table",
            "describe",
            "delete",
            "truncate",
            "refit",
            "set_strategy",
            "compact",
            "drop_table",
            "list_tables",
            "list_models",
            "locate",
            "health",
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
}
