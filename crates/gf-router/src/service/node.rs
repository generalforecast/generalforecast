// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The nodes a router forwards to, and the calls it makes of them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arrow_flight::Action;
use arrow_flight::flight_service_client::FlightServiceClient;
use futures::TryStreamExt;
use gf_server::auth::{CREATED, MISSING_OK, SHIPMENT, TERM, TERMS, WRITE_ID};
use gf_server::service::MESSAGE_BYTES;
use serde_json::Value;
use tonic::metadata::{Ascii, MetadataMap, MetadataValue};
use tonic::transport::Channel;
use tonic::{Code, Request, Status};

use super::{CHECK_TIMEOUT, TRACE, action};
use crate::placement::Placement;
use crate::replication::TableCopy;

/// A node the router forwards to, the authorization header it presents there, and whether it is up.
pub struct Node {
    pub name: String,
    pub(super) client: FlightServiceClient<Channel>,
    pub(super) authorization: MetadataValue<Ascii>,
    /// How long a call to the node may take, whatever it is.
    pub(super) timeout: Duration,
    /// How long a write of rows or keys may take, and its shipment to replicas.
    pub(super) write_timeout: Duration,
    pub(super) up: AtomicBool,
    /// Whether the last check reached the node, to log when that changes.
    pub(super) checked: AtomicBool,
}

/// How long calls to a node may take.
#[derive(Clone, Copy, Debug)]
pub struct Timeouts {
    /// Any call, including a prediction that waits for a fit.
    pub call: Duration,
    /// A write of rows or keys, and again its shipment to replicas.
    pub write: Duration,
}

impl Node {
    /// A node, assumed up until a call or check fails, whose calls take at most `timeouts` say.
    #[must_use]
    pub fn new(
        name: String,
        client: FlightServiceClient<Channel>,
        authorization: MetadataValue<Ascii>,
        timeouts: Timeouts,
    ) -> Self {
        Self {
            name,
            client: client
                .max_decoding_message_size(MESSAGE_BYTES)
                .max_encoding_message_size(MESSAGE_BYTES),
            authorization,
            timeout: timeouts.call,
            write_timeout: timeouts.write,
            up: AtomicBool::new(true),
            checked: AtomicBool::new(true),
        }
    }

    pub(super) fn is_up(&self) -> bool {
        self.up.load(Ordering::Relaxed)
    }

    /// Marks the node down if a call failed because it could not be reached:
    /// a transport failure, which carries its cause, rather than an answer.
    pub(super) fn note(&self, status: &Status) {
        if status.code() == Code::Unavailable && std::error::Error::source(status).is_some() {
            self.down();
        }
    }

    pub(super) fn down(&self) {
        self.up.store(false, Ordering::Relaxed);
    }

    /// A request for this node: the router's token replaces the client's, and
    /// the client call's trace goes on.
    pub(super) fn request<T>(&self, message: T) -> Request<T> {
        let mut request = Request::new(message);
        request
            .metadata_mut()
            .insert("authorization", self.authorization.clone());
        if let Ok(Ok(traceparent)) = TRACE.try_with(|trace| trace.header().parse()) {
            request.metadata_mut().insert("traceparent", traceparent);
        }
        request
    }

    /// A request carrying log entries from another node, marked so the node
    /// applies them, with the source copy's lineage for the node to check.
    pub(super) fn shipment<T>(&self, message: T, source: Option<&TableCopy>) -> Request<T> {
        let mut request = self.request(message);
        request
            .metadata_mut()
            .insert(SHIPMENT, MetadataValue::from_static("1"));
        if let Some(source) = source {
            let terms = serde_json::to_string(&source.terms).unwrap_or_default();
            if let Ok(terms) = terms.parse() {
                request.metadata_mut().insert(TERMS, terms);
            }
            if let Some(created) = &source.created
                && let Ok(created) = created.as_str().parse()
            {
                request.metadata_mut().insert(CREATED, created);
            }
        }
        request
    }

    /// A request for a write the router sends under `term`, which the node
    /// logs only while the table is still at that term, carrying on the
    /// client's metadata that writes take: its write id, so a retry applies
    /// once, and whether a delete skips missing keys.
    pub(super) fn write<T>(
        &self,
        message: T,
        term: Option<u64>,
        client: &MetadataMap,
    ) -> Request<T> {
        let mut request = self.request(message);
        if let Some(term) = term {
            request.metadata_mut().insert(TERM, term.into());
        }
        for name in [WRITE_ID, MISSING_OK] {
            if let Some(value) = client.get(name) {
                request.metadata_mut().insert(name, value.clone());
            }
        }
        request
    }

    /// Calls an action and parses each result as JSON.
    ///
    /// # Errors
    ///
    /// With the node's status.
    pub async fn action(&self, kind: &str, body: Value) -> Result<Vec<Value>, Status> {
        self.call(self.request(action(kind, &body))).await
    }

    /// Calls a control action, such as a promotion or a release, within the
    /// check's timeout, so one slow node holds up no other table's repair.
    pub(super) async fn control(&self, kind: &str, body: Value) -> Result<Vec<Value>, Status> {
        self.call_within(self.request(action(kind, &body)), CHECK_TIMEOUT)
            .await
    }

    /// Calls an action request and parses each result as JSON, waiting at most
    /// the node's timeout; whether the node is up is the checks' to say, since
    /// a long compaction may still finish.
    pub(super) async fn call(&self, request: Request<Action>) -> Result<Vec<Value>, Status> {
        self.call_within(request, self.timeout).await
    }

    pub(super) async fn call_within(
        &self,
        request: Request<Action>,
        timeout: Duration,
    ) -> Result<Vec<Value>, Status> {
        let mut client = self.client.clone();
        let call = async {
            client
                .do_action(request)
                .await?
                .into_inner()
                .try_collect::<Vec<arrow_flight::Result>>()
                .await
        };
        let results = match tokio::time::timeout(timeout, call).await {
            Ok(results) => results.inspect_err(|status| self.note(status))?,
            Err(_) => return Err(Status::unavailable(format!("{} timed out", self.name))),
        };
        results
            .iter()
            .map(|result| {
                serde_json::from_slice(&result.body)
                    .map_err(|error| Status::internal(format!("{}: {error}", self.name)))
            })
            .collect()
    }
}

/// The nodes tables are placed on, and how many copies each table keeps.
pub struct Cluster {
    pub(super) nodes: Vec<Node>,
    pub(super) placement: Placement,
    pub(super) replicas: usize,
}

impl Cluster {
    /// A cluster over nodes, in configuration order.
    #[must_use]
    pub fn new(nodes: Vec<Node>, replicas: usize) -> Self {
        let placement = Placement::new(nodes.iter().map(|node| node.name.clone()).collect());
        Self {
            nodes,
            placement,
            replicas: replicas.max(1),
        }
    }

    /// The nodes that should hold a table, primary first.
    pub(super) fn replicas(&self, table: &str) -> Vec<usize> {
        let mut ranked = self.placement.ranked(table);
        ranked.truncate(self.replicas);
        ranked
    }

    pub(super) fn up(&self) -> Vec<bool> {
        self.nodes.iter().map(Node::is_up).collect()
    }

    pub(super) fn index(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|node| node.name == name)
    }
}
