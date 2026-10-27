// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Keeping each table's replicas: shipping log entries to copies that are
//! behind, the background check of every node, and the repairs it starts.

use std::sync::PoisonError;
use std::sync::atomic::Ordering;
use std::time::Duration;

use arrow_flight::{FlightData, FlightDescriptor, PutResult, Ticket};
use futures::{Stream, StreamExt, TryStreamExt, stream};
use gf_server::action::CopyLength;
use gf_server::wal::CreationId;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tonic::{Code, Status, Streaming};

use super::copies::Tables;
use super::{CHECK_TIMEOUT, Cluster, Node, Router, SHIP_TIMEOUT, put_answer, unavailable};
use crate::replication::{Repair, TableCopy, repairs};

impl Router {
    /// Ships a table's log entries from sequence `from` on, within `timeout`,
    /// and returns the target's log length.
    ///
    /// If reading from the source fails, the shipment fails too. The target
    /// may keep a prefix of the entries, which is still a valid copy.
    pub(super) async fn ship(
        &self,
        table: &str,
        source: &Node,
        target: &Node,
        from: u64,
        timeout: Duration,
    ) -> Result<u64, Status> {
        let shipment = self.ship_entries(table, source, target, from);
        tokio::time::timeout(timeout, shipment)
            .await
            .unwrap_or_else(|_| {
                Err(Status::deadline_exceeded(format!(
                    "shipping {table:?} from {} to {} took over {} seconds",
                    source.name,
                    target.name,
                    timeout.as_secs()
                )))
            })
    }

    pub(super) async fn ship_entries(
        &self,
        table: &str,
        source: &Node,
        target: &Node,
        from: u64,
    ) -> Result<u64, Status> {
        // The target checks the creation, so a copy of another table of the
        // name is never shipped over this one's.
        if self
            .copy(table, source)
            .is_none_or(|copy| copy.created.is_none())
        {
            return Err(Status::failed_precondition(format!(
                "{table:?} on {} is of no known creation yet; a check will learn it",
                source.name
            )));
        }
        let ticket = Ticket {
            ticket: json!({"table": table, "from": from}).to_string().into(),
        };
        let mut entries = source
            .client
            .clone()
            .do_get(source.request(ticket))
            .await
            .inspect_err(|status| source.note(status))?
            .into_inner();
        let Some(mut first) = entries
            .message()
            .await
            .inspect_err(|status| source.note(status))?
        else {
            return Ok(from);
        };
        first.flight_descriptor = Some(FlightDescriptor::new_path(vec![
            "replicate".to_string(),
            table.to_string(),
        ]));
        let (rest, failure) = unbroken(entries);
        let lineage = self.copy(table, source);
        let put = async {
            let messages = stream::once(std::future::ready(first)).chain(rest);
            target
                .client
                .clone()
                .do_put(target.shipment(messages, lineage.as_ref()))
                .await?
                .into_inner()
                .try_collect::<Vec<PutResult>>()
                .await
        };
        let results = tokio::select! {
            results = put => results.inspect_err(|status| target.note(status))?,
            Ok(status) = failure => {
                source.note(&status);
                return Err(status);
            }
        };
        let length = put_answer(&results).log_length.ok_or_else(|| {
            Status::internal(format!("{} did not report a log length", target.name))
        })?;
        self.record(table, target, length, None);
        Ok(length)
    }

    /// The primary to write to, first brought up to the longest copy if a move left it behind.
    pub(super) async fn primary(&self, cluster: &Cluster, table: &str) -> Result<usize, Status> {
        let Some(primary) = self.writer(cluster, table) else {
            let name = self.named_primary(cluster, table).unwrap_or_default();
            return Err(Status::unavailable(format!(
                "table {table:?} is unavailable for writes: its primary {name} left the \
                 configuration; add it back, or let the router with failover_seconds \
                 promote a replica"
            )));
        };
        let node = &cluster.nodes[primary];
        if !node.is_up() {
            return Err(unavailable(table, node, "it is down"));
        }
        let (lengths, _) = self.view(cluster, table);
        if let Some(longest) = lengths.iter().flatten().copied().max()
            && lengths[primary] != Some(longest)
        {
            let source = lengths
                .iter()
                .position(|length| *length == Some(longest))
                .expect("the longest copy exists");
            let from = lengths[primary].unwrap_or(0);
            let shipped = self
                .ship(
                    table,
                    &cluster.nodes[source],
                    node,
                    from,
                    node.write_timeout,
                )
                .await?;
            if shipped < longest {
                return Err(unavailable(
                    table,
                    node,
                    &format!("it holds {shipped} of {longest} log entries after catching up"),
                ));
            }
        }
        Ok(primary)
    }

    /// Records the primary's log length after a write and ships the write to
    /// the other replicas, all at once.
    ///
    /// A replica that cannot take it is caught up by the background check.
    /// With `confirm`, a table with replicas needs one of them to hold the
    /// write, so an acknowledged write is never on its primary alone.
    ///
    /// # Errors
    ///
    /// `UNAVAILABLE` if the write needed confirming and no replica holds it.
    pub(super) async fn replicate(
        &self,
        cluster: &Cluster,
        table: &str,
        primary: usize,
        length: u64,
        confirm: bool,
        created: Option<CreationId>,
    ) -> Result<(), Status> {
        let source = &cluster.nodes[primary];
        self.record(table, source, length, created);
        // A diverged copy counts as down, so nothing ships to it until it is set aside.
        let (lengths, up) = self.view(cluster, table);
        // The replicas promotion looks at, so a write one confirms is never
        // on a copy failing over would pass by.
        let replicas: Vec<usize> = self
            .desired(cluster, table)
            .into_iter()
            .filter(|&replica| replica != primary)
            .collect();
        let reachable: Vec<usize> = replicas
            .iter()
            .copied()
            .filter(|&replica| up[replica])
            .collect();
        let shipments = futures::future::join_all(reachable.iter().map(|&replica| {
            let target = &cluster.nodes[replica];
            let from = lengths[replica].unwrap_or(0);
            async move {
                let shipped = self
                    .ship(table, source, target, from, target.write_timeout)
                    .await;
                // The replica's log is shorter than recorded, or gone: start
                // over from its beginning, which is idempotent. A fenced copy
                // answers with failed precondition and would refuse that too.
                let restart = [Code::OutOfRange, Code::NotFound];
                if matches!(&shipped, Err(status) if restart.contains(&status.code())) {
                    return self
                        .ship(table, source, target, 0, target.write_timeout)
                        .await;
                }
                shipped
            }
        }))
        .await;
        let mut confirmed = false;
        let copies = self.copies(cluster, table);
        let created = self.creation(cluster, table, &copies);
        for (&replica, shipped) in reachable.iter().zip(shipments) {
            match shipped {
                Ok(shipped) => {
                    self.acknowledge(table, created.clone(), shipped);
                    confirmed |= shipped >= length;
                }
                Err(status) => tracing::warn!(
                    table,
                    replica = cluster.nodes[replica].name,
                    "replicating a write: {status}"
                ),
            }
        }
        if confirm && !replicas.is_empty() && !confirmed {
            return Err(Status::unavailable(format!(
                "table {table:?}: the write reached its primary {} but no replica holds it, \
                 so it is not acknowledged; it may still apply once a replica catches up",
                source.name
            )));
        }
        Ok(())
    }

    /// Checks every node, then repairs every table's copies, and with leases
    /// renews them on a loop of its own, so a slow repair never lets them
    /// lapse. Runs until the process exits.
    pub async fn maintain(self, interval: Duration) {
        if let Some(failover) = self.failover.clone() {
            tokio::spawn(self.clone().renew(interval, failover));
        }
        let mut ticks = tokio::time::interval(interval);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticks.tick().await;
            self.check().await;
        }
    }

    /// Learns every node's copies, and whether it is up, then fails tables
    /// over and starts repairs.
    pub(super) async fn check(&self) {
        let cluster = self.cluster();
        let since = self
            .copies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stamp();
        let answers = futures::future::join_all(cluster.nodes.iter().map(|node| async move {
            tokio::time::timeout(CHECK_TIMEOUT, node.action("log_lengths", json!({})))
                .await
                .unwrap_or_else(|_| Err(Status::unavailable("timed out")))
        }))
        .await;
        let up: Vec<bool> = answers.iter().map(Result::is_ok).collect();
        let mut answered = Tables::new();
        for (node, answer) in cluster.nodes.iter().zip(answers) {
            let answer = match answer {
                Ok(answer) => {
                    if !node.checked.swap(true, Ordering::Relaxed) {
                        tracing::info!(node = node.name, "the node is up");
                    }
                    answer
                }
                Err(status) => {
                    // Said once, when the node goes down, with why, such as a
                    // token the node refuses.
                    if node.checked.swap(false, Ordering::Relaxed) {
                        tracing::warn!(
                            node = node.name,
                            code = ?status.code(),
                            "the node is down: {}",
                            status.message()
                        );
                    }
                    continue;
                }
            };
            let tables = answer
                .first()
                .and_then(Value::as_object)
                .into_iter()
                .flatten();
            // Each copy apart, so one this router cannot read leaves the others.
            for (table, copy) in tables {
                if let Ok(CopyLength {
                    length,
                    terms,
                    created,
                }) = CopyLength::deserialize(copy)
                {
                    answered.entry(table.clone()).or_default().insert(
                        node.name.clone(),
                        TableCopy {
                            length,
                            terms,
                            created,
                        },
                    );
                }
            }
        }
        let tables: Vec<String> = {
            let mut copies = self.copies.lock().unwrap_or_else(PoisonError::into_inner);
            copies.install(answered, since);
            self.creations
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|table, _| copies.contains(table));
            self.definitions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|table, _| !copies.partition_counts(table).is_empty());
            copies.names().cloned().collect()
        };
        // Only now, so leases never see a returning node up with its copies unknown.
        for (node, up) in cluster.nodes.iter().zip(up) {
            node.up.store(up, Ordering::Relaxed);
        }
        for table in &tables {
            // Diverged copies and copies of an earlier creation hold no
            // acknowledged write of this one, so they count for nothing.
            let writer = self.writer(&cluster, table);
            let (lengths, _) = self.view(&cluster, table);
            // Every copy but the writer's counts, even one no longer a replica:
            // after replicas change, an old one may hold writes the new ones
            // are still catching up on.
            let longest = lengths
                .iter()
                .enumerate()
                .filter(|&(node, _)| Some(node) != writer)
                .filter_map(|(_, length)| *length)
                .max();
            if let Some(longest) = longest {
                let copies = self.copies(&cluster, table);
                let created = self.creation(&cluster, table, &copies);
                self.acknowledge(table, created, longest);
            }
        }
        if let Some(failover) = &self.failover
            && !failover.contested.load(Ordering::Relaxed)
        {
            self.fail_over(&cluster, &tables, failover).await;
        }
        // Apart from the check, so a long shipment holds up neither the next
        // check nor failing over; one repair runs per table at a time.
        for table in tables {
            let started = self
                .repairing
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(table.clone());
            if !started {
                continue;
            }
            let (router, cluster) = (self.clone(), cluster.clone());
            tokio::spawn(async move {
                if let Ok(_permit) = router.repairs.clone().acquire_owned().await {
                    router.repair(&cluster, table.clone()).await;
                }
                router
                    .repairing
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(&table);
            });
        }
    }

    /// Ships to a table's replicas that are behind, and releases copies no
    /// replica needs, one at a time.
    pub(super) async fn repair(&self, cluster: &Cluster, table: String) {
        if self.moves()
            && let Err(status) = self.move_writer(cluster, &table).await
        {
            tracing::warn!(table, "moving writes: {status}");
        }
        let (lengths, up) = self.view(cluster, &table);
        let planned = repairs(&self.desired(cluster, &table), &lengths, &up);
        for repair in planned {
            // Held per shipment, so a drop waits for one, not the whole pass.
            let _repairing = self.lifecycle(&table).read_owned().await;
            let repaired = match repair {
                Repair::Ship {
                    source,
                    target,
                    from,
                } => self
                    .ship(
                        &table,
                        &cluster.nodes[source],
                        &cluster.nodes[target],
                        from,
                        SHIP_TIMEOUT,
                    )
                    .await
                    .map(drop),
                Repair::Release { node } => {
                    let node = &cluster.nodes[node];
                    tracing::info!(table, node = node.name, "releasing a copy no replica needs");
                    node.control("release", json!({"table": table}))
                        .await
                        .map(drop)
                }
            };
            if let Err(status) = repaired {
                tracing::warn!(table, "repairing copies: {status}");
            }
        }
    }
}

/// A node's stream of log entries that never ends early: an error goes to the
/// receiver instead, so the target never takes a cut stream for a whole one.
fn unbroken(
    entries: Streaming<FlightData>,
) -> (
    impl Stream<Item = FlightData> + Send + 'static,
    oneshot::Receiver<Status>,
) {
    let (failed, failure) = oneshot::channel();
    let stream = stream::unfold(
        (entries, Some(failed)),
        |(mut entries, mut failed)| async move {
            match entries.message().await {
                Ok(Some(entry)) => Some((entry, (entries, failed))),
                Ok(None) => None,
                Err(status) => {
                    if let Some(failed) = failed.take() {
                        let _ = failed.send(status);
                    }
                    std::future::pending().await
                }
            }
        },
    );
    (stream, failure)
}
