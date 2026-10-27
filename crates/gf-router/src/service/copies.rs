// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! What the router saw of each table's copies, and what follows from it: each
//! table's creation, writer, term, and the nodes that should hold it.

use std::collections::HashMap;
use std::sync::PoisonError;

use gf_server::wal::{CreationId, Terms};
use tonic::Status;

use super::{Cluster, Node, Router};
use crate::replication::{TableCopy, creation, current, diverged, read_order, same_creation};

/// The longest log seen on a replica of one creation of a table.
#[derive(Clone, Debug, Default)]
pub(super) struct Seen {
    pub(super) created: Option<CreationId>,
    pub(super) length: u64,
}

/// What the router saw of each table's copies, by table and then node name.
pub(super) type Copies = HashMap<String, HashMap<String, TableCopy>>;

impl Router {
    /// What the router saw of a table's copies, by node index.
    pub(super) fn copies(&self, cluster: &Cluster, table: &str) -> Vec<Option<TableCopy>> {
        let copies = self.copies.lock().unwrap_or_else(PoisonError::into_inner);
        let copies = copies.get(table);
        cluster
            .nodes
            .iter()
            .map(|node| copies.and_then(|copies| copies.get(&node.name).cloned()))
            .collect()
    }

    /// The creation of the table this name serves: the one the router
    /// created or chose before, while a copy of it remains, and otherwise the
    /// one the copies vote for, which it keeps from then on. So a copy of an
    /// earlier creation that turns up on a node that missed the drop cannot
    /// take the name back, whatever its node's clock said.
    pub(super) fn creation(
        &self,
        cluster: &Cluster,
        table: &str,
        copies: &[Option<TableCopy>],
    ) -> Option<CreationId> {
        let mut creations = self
            .creations
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let present: Vec<&CreationId> = copies
            .iter()
            .flatten()
            .filter_map(|copy| copy.created.as_ref())
            .collect();
        if let Some(known) = creations.get(table)
            && (present.is_empty() || present.contains(&known))
        {
            return Some(known.clone());
        }
        let chosen = creation(copies, &cluster.replicas(table));
        if let Some(chosen) = &chosen {
            creations.insert(table.to_string(), chosen.clone());
        }
        chosen
    }

    /// What the router saw of a table's copies of its creation, by node index;
    /// copies of an earlier table of the name are left out.
    pub(super) fn kept(&self, cluster: &Cluster, table: &str) -> Vec<Option<TableCopy>> {
        let copies = self.copies(cluster, table);
        same_creation(&copies, self.creation(cluster, table, &copies).as_ref())
    }

    /// A table's log lengths, and which nodes count as up for it, by node index.
    ///
    /// A copy whose writes diverged counts as absent and its node as down, so
    /// nothing ships to it, from it, or reads it until it is set aside.
    pub(super) fn view(&self, cluster: &Cluster, table: &str) -> (Vec<Option<u64>>, Vec<bool>) {
        let copies = self.copies(cluster, table);
        let creation = self.creation(cluster, table, &copies);
        let mut up = cluster.up();
        let lengths = copies
            .iter()
            .zip(diverged(&copies, creation.as_ref()))
            .zip(&mut up)
            .map(|((copy, diverged), up)| {
                *up &= !diverged;
                copy.as_ref().filter(|_| !diverged).map(|copy| copy.length)
            })
            .collect();
        (lengths, up)
    }

    /// Records a copy's log length after a write or a shipment, which leaves it
    /// at the table's latest terms and creation; `created` is the creation a
    /// create answered with, if this write was one, which the name serves
    /// from here on.
    pub(super) fn record(
        &self,
        table: &str,
        node: &Node,
        length: u64,
        created: Option<CreationId>,
    ) {
        let known = {
            let mut creations = self
                .creations
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(created) = &created {
                creations.insert(table.to_string(), created.clone());
            }
            creations.get(table).cloned()
        };
        let mut copies = self.copies.lock().unwrap_or_else(PoisonError::into_inner);
        let copies = copies.entry(table.to_string()).or_default();
        // The terms come from the table's own copies: one of an earlier
        // creation, left on a node that missed the drop, has terms of its own.
        let (terms, seen) = copies
            .values()
            .filter(|copy| known.is_none() || copy.created.is_none() || copy.created == known)
            .max_by_key(|copy| (copy.term().term, copy.length))
            .map_or_else(
                || (Terms::default(), None),
                |copy| (copy.terms.clone(), copy.created.clone()),
            );
        let created = created.or(seen).or(known);
        copies.insert(
            node.name.clone(),
            TableCopy {
                length,
                terms,
                created,
            },
        );
    }

    /// What the router saw of one node's copy of a table.
    pub(super) fn copy(&self, table: &str, node: &Node) -> Option<TableCopy> {
        self.copies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(table)
            .and_then(|copies| copies.get(&node.name).cloned())
    }

    /// Raises the longest log seen on a replica of the table's creation; a
    /// new creation of the name starts again.
    pub(super) fn acknowledge(&self, table: &str, created: Option<CreationId>, length: u64) {
        let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = seen.entry(table.to_string()).or_default();
        if entry.created != created {
            *entry = Seen { created, length };
        } else {
            entry.length = entry.length.max(length);
        }
    }

    /// The longest log seen on a replica of the table's current creation.
    pub(super) fn acknowledged(&self, cluster: &Cluster, table: &str) -> u64 {
        let copies = self.copies(cluster, table);
        let created = self.creation(cluster, table, &copies);
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(table)
            .filter(|seen| seen.created == created)
            .map_or(0, |seen| seen.length)
    }

    pub(super) fn forget(&self, table: &str, node: &Node) {
        if let Some(copies) = self
            .copies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_mut(table)
        {
            copies.remove(&node.name);
        }
    }

    /// The node that takes a table's writes: the primary its latest term
    /// named, or at term 0, before its first write named one, its top-ranked
    /// node. None if the named primary left the configuration: the writer
    /// changes only under a new term, or a primary that returned could merge
    /// its unacknowledged writes with its successor's.
    pub(super) fn writer(&self, cluster: &Cluster, table: &str) -> Option<usize> {
        let copies = self.kept(cluster, table);
        match current(&copies).map(TableCopy::term) {
            Some(term) if term.term > 0 => cluster.index(&term.primary),
            _ => Some(cluster.replicas(table)[0]),
        }
    }

    /// The primary the table's latest term names, by name, if it names one.
    pub(super) fn named_primary(&self, cluster: &Cluster, table: &str) -> Option<String> {
        current(&self.kept(cluster, table))
            .map(TableCopy::term)
            .filter(|term| term.term > 0)
            .map(|term| term.primary.clone())
    }

    /// The table's latest term as the router saw it, if it has seen a copy.
    pub(super) fn term(&self, cluster: &Cluster, table: &str) -> Option<u64> {
        current(&self.kept(cluster, table)).map(|copy| copy.term().term)
    }

    /// The table's latest term, which every write names.
    ///
    /// # Errors
    ///
    /// `UNAVAILABLE` if the router has seen no copy of the table: it may have
    /// moved to a primary this router has not learned of.
    pub(super) fn known_term(&self, cluster: &Cluster, table: &str) -> Result<u64, Status> {
        self.term(cluster, table).ok_or_else(|| {
            Status::unavailable(format!(
                "the router has not learned table {table:?}'s copies yet; retry shortly"
            ))
        })
    }

    /// The nodes that should hold a table, primary first: its replicas, and its
    /// primary wherever a promotion put it.
    pub(super) fn desired(&self, cluster: &Cluster, table: &str) -> Vec<usize> {
        let mut desired = cluster.replicas(table);
        if let Some(writer) = self.writer(cluster, table) {
            desired.retain(|&node| node != writer);
            desired.insert(0, writer);
            // A primary outside the top-ranked nodes, until it hands the
            // table off, takes a place.
            desired.truncate(cluster.replicas);
        }
        desired
    }

    pub(super) fn read_order(&self, cluster: &Cluster, table: &str) -> Vec<usize> {
        let (lengths, up) = self.view(cluster, table);
        read_order(&self.desired(cluster, table), &lengths, &up)
    }
}
