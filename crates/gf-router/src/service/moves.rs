// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Who takes a table's writes: naming its first writer, beginning terms, and
//! handing the writes to the node the table now ranks first.

use std::sync::PoisonError;
use std::sync::atomic::Ordering;

use gf_server::action::Answer;
use gf_server::wal::Term;
use serde_json::json;
use tonic::{Code, Status};

use super::{Cluster, Router, SHIP_TIMEOUT};

impl Router {
    /// Begins `term` on `node`'s copy of a table, naming `primary`, and
    /// records it; returns the copy's log length after it.
    pub(super) async fn promote(
        &self,
        cluster: &Cluster,
        table: &str,
        node: usize,
        term: u64,
        primary: usize,
    ) -> Result<u64, Status> {
        let (target, named) = (&cluster.nodes[node], &cluster.nodes[primary]);
        let promote = json!({"table": table, "term": term, "primary": named.name});
        let answer = target.control("promote", promote).await?;
        let length = answer
            .first()
            .and_then(|answer| Answer::of(answer).log_length)
            .ok_or_else(|| {
                Status::internal(format!("{} did not report a log length", target.name))
            })?;
        let mut copies = self.copies.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(copy) = copies
            .get_mut(table)
            .and_then(|copies| copies.get_mut(&target.name))
        {
            copy.terms.begin(Term {
                term,
                primary: named.name.clone(),
                start: copy.length,
            });
            copy.length = length;
        }
        Ok(length)
    }

    /// Names a table's writer under term 1 before its first write, so the
    /// writer only ever changes under a new term: at term 0 it is whichever
    /// node ranks first, which a change of configuration moves.
    pub(super) async fn name_writer(&self, table: &str) -> Result<(), Status> {
        if self.term(&self.cluster(), table) != Some(0) {
            return Ok(());
        }
        let _naming = self.lifecycle(table).write_owned().await;
        let cluster = self.cluster();
        if self.term(&cluster, table) != Some(0) {
            return Ok(());
        }
        // Brought up to the longest copy first, so it holds every acknowledged write.
        let primary = self.primary(&cluster, table).await?;
        match self.promote(&cluster, table, primary, 1, primary).await {
            Ok(_) => Ok(()),
            // Another router named it meanwhile; a check learns of it.
            Err(status) if status.code() == Code::OutOfRange => Err(Status::unavailable(format!(
                "table {table:?} took a new term elsewhere; retry shortly"
            ))),
            Err(status) => Err(status),
        }
    }

    /// Whether this router hands tables off to the nodes they now rank first:
    /// unless it stands down for another router that fails tables over.
    pub(super) fn moves(&self) -> bool {
        self.failover
            .as_ref()
            .is_none_or(|failover| !failover.contested.load(Ordering::Relaxed))
    }

    /// Hands a table's writes to the node it now ranks first, once its writer
    /// is no longer among its top-ranked nodes, as after nodes are added.
    ///
    /// The target catches up while writes go on; then, holding the table's
    /// writes back, it catches up the rest, and the writer begins a new term
    /// naming the target, which fences it: it takes no more writes of the
    /// old term from any router. The target then takes the term from it.
    pub(super) async fn move_writer(&self, cluster: &Cluster, table: &str) -> Result<(), Status> {
        let ranked = cluster.replicas(table);
        let Some(writer) = self.writer(cluster, table) else {
            return Ok(());
        };
        let term = self.term(cluster, table).unwrap_or(0);
        let target = ranked[0];
        let (from, to) = (&cluster.nodes[writer], &cluster.nodes[target]);
        if term == 0 || ranked.contains(&writer) || !from.is_up() || !to.is_up() {
            return Ok(());
        }
        {
            let _shipping = self.lifecycle(table).read_owned().await;
            let (lengths, _) = self.view(cluster, table);
            let have = lengths[target].unwrap_or(0);
            if lengths[writer].is_some_and(|length| have < length) {
                self.ship(table, from, to, have, SHIP_TIMEOUT).await?;
            }
        }
        let _moving = self.lifecycle(table).write_owned().await;
        if self.writer(cluster, table) != Some(writer) || self.term(cluster, table) != Some(term) {
            return Ok(());
        }
        let (lengths, _) = self.view(cluster, table);
        let have = lengths[target].unwrap_or(0);
        let length = lengths[writer].unwrap_or(0);
        if have < length {
            self.ship(table, from, to, have, from.write_timeout).await?;
        }
        let fenced = self
            .promote(cluster, table, writer, term + 1, target)
            .await?;
        let shipped = self
            .ship(table, from, to, length, from.write_timeout)
            .await?;
        if shipped < fenced {
            // Writes wait until a check catches the target up.
            return Err(Status::unavailable(format!(
                "{} holds {shipped} of {fenced} log entries after taking {table:?}",
                to.name
            )));
        }
        self.moves.add(1);
        tracing::info!(
            table,
            from = from.name,
            to = to.name,
            term = term + 1,
            "moved writes"
        );
        Ok(())
    }
}
