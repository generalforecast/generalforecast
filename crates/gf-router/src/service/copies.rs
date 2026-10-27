// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! What the router saw of each table's copies, and what follows from it: each
//! table's creation, writer, term, and the nodes that should hold it.
//!
//! A check learns every copy from the nodes, while writes, shipments,
//! promotions and drops record what they changed as they go. A check's
//! answers are gathered before it takes their place, so the changes recorded
//! since it began stand over them: a check never takes back a term a write
//! began meanwhile, nor brings back a table dropped meanwhile.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::PoisonError;

use gf_server::wal::{CreationId, Terms};
use tonic::Status;

use super::{Cluster, Node, Router};
use crate::partitions;
use crate::replication::{TableCopy, creation, current, diverged, read_order, same_creation};

/// The longest log seen on a replica of one creation of a table.
#[derive(Clone, Debug, Default)]
pub(super) struct Seen {
    pub(super) created: Option<CreationId>,
    pub(super) length: u64,
}

/// Each table's copies, by table and then node name.
pub(super) type Tables = HashMap<String, HashMap<String, TableCopy>>;

/// What the router saw of each table's copies, and the changes it made to
/// them since its last check began.
#[derive(Debug, Default)]
pub(super) struct Copies {
    tables: Tables,
    /// For each table split into partitions, how many partitions of each
    /// count its partitions' names say there are.
    partitions: HashMap<String, BTreeMap<usize, usize>>,
    /// Counts the changes the router makes, each stamped with the count.
    stamp: u64,
    /// The latest change to each table's copy on a node, by table and node
    /// name, or to a whole table, by its name alone.
    changed: HashMap<(String, Option<String>), u64>,
    /// The tables each node out of reach held when last answered, by node
    /// name: its copies are left out of what the router saw, which only the
    /// nodes that answer say, but its names stay taken.
    unreachable: HashMap<String, BTreeSet<String>>,
}

impl Copies {
    /// A table's copies, by node name.
    pub(super) fn table(&self, table: &str) -> Option<&HashMap<String, TableCopy>> {
        self.tables.get(table)
    }

    /// One node's copy of a table.
    pub(super) fn get(&self, table: &str, node: &str) -> Option<&TableCopy> {
        self.tables.get(table)?.get(node)
    }

    /// Whether any node holds a copy of the table.
    pub(super) fn contains(&self, table: &str) -> bool {
        self.tables.contains_key(table)
    }

    /// Every table with a copy.
    pub(super) fn names(&self) -> impl Iterator<Item = &String> {
        self.tables.keys()
    }

    /// The partition counts the partitions of a table split into partitions
    /// name, fewest first; empty for a table not split. More than one is a
    /// table made twice at once through two routers.
    pub(super) fn partition_counts(&self, table: &str) -> Vec<usize> {
        self.partitions
            .get(table)
            .map(|counts| counts.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Whether a node out of reach held a table under the name when last
    /// answered, whole or as a partition.
    pub(super) fn held_out_of_reach(&self, table: &str) -> bool {
        self.unreachable.values().flatten().any(|name| {
            name == table || partitions::parse(name).is_some_and(|(owner, _, _)| owner == table)
        })
    }

    /// The stamp a check begins at: changes after it stand over its answers.
    pub(super) fn stamp(&self) -> u64 {
        self.stamp
    }

    fn change(&mut self, table: &str, node: Option<&str>) {
        self.stamp += 1;
        self.changed
            .insert((table.to_string(), node.map(str::to_string)), self.stamp);
    }

    /// Sets a node's copy of a table.
    pub(super) fn set(&mut self, table: &str, node: &str, copy: TableCopy) {
        self.change(table, Some(node));
        if !self.tables.contains_key(table) {
            index(&mut self.partitions, table, 1);
        }
        self.tables
            .entry(table.to_string())
            .or_default()
            .insert(node.to_string(), copy);
    }

    /// Changes a node's copy of a table, if the router saw one.
    pub(super) fn update(&mut self, table: &str, node: &str, change: impl FnOnce(&mut TableCopy)) {
        if let Some(copy) = self
            .tables
            .get_mut(table)
            .and_then(|copies| copies.get_mut(node))
        {
            change(copy);
            self.change(table, Some(node));
        }
    }

    /// Forgets a node's copy of a table, as once it is set aside.
    pub(super) fn forget(&mut self, table: &str, node: &str) {
        self.change(table, Some(node));
        if let Some(copies) = self.tables.get_mut(table) {
            copies.remove(node);
            if copies.is_empty() {
                self.remove(table);
            }
        }
    }

    /// Forgets every copy of a dropped table, which a drop makes on every
    /// node, so no node out of reach still holds it.
    pub(super) fn drop_table(&mut self, table: &str) {
        self.change(table, None);
        self.remove(table);
        for held in self.unreachable.values_mut() {
            held.remove(table);
        }
    }

    fn remove(&mut self, table: &str) {
        if self.tables.remove(table).is_some() {
            index(&mut self.partitions, table, -1);
        }
    }

    /// Takes a check's answers, begun at `since`, in place of what the
    /// router saw, but for what it changed itself since then. The nodes in
    /// `unanswered` keep the names of the tables they held, until they answer.
    pub(super) fn install(&mut self, mut answered: Tables, since: u64, unanswered: &[&str]) {
        let mut unreachable: HashMap<String, BTreeSet<String>> = HashMap::new();
        for &node in unanswered {
            let mut held = self.unreachable.remove(node).unwrap_or_default();
            held.extend(
                self.tables
                    .iter()
                    .filter(|(_, copies)| copies.contains_key(node))
                    .map(|(table, _)| table.clone()),
            );
            unreachable.insert(node.to_string(), held);
        }
        self.unreachable = unreachable;
        for ((table, node), &stamp) in &self.changed {
            if stamp <= since {
                continue;
            }
            match node {
                Some(node) => match self.get(table, node) {
                    Some(copy) => {
                        answered
                            .entry(table.clone())
                            .or_default()
                            .insert(node.clone(), copy.clone());
                    }
                    None => {
                        if let Some(copies) = answered.get_mut(table) {
                            copies.remove(node);
                        }
                    }
                },
                None => match self.tables.get(table) {
                    Some(copies) => {
                        answered.insert(table.clone(), copies.clone());
                    }
                    None => {
                        answered.remove(table);
                    }
                },
            }
        }
        answered.retain(|_, copies| !copies.is_empty());
        self.changed.retain(|_, stamp| *stamp > since);
        self.partitions.clear();
        for table in answered.keys() {
            index(&mut self.partitions, table, 1);
        }
        self.tables = answered;
    }
}

/// Counts a partition's table in or out of the partition index.
fn index(partitions: &mut HashMap<String, BTreeMap<usize, usize>>, table: &str, by: isize) {
    let Some((owner, _, count)) = partitions::parse(table) else {
        return;
    };
    let counts = partitions.entry(owner.to_string()).or_default();
    let held = counts.entry(count).or_default();
    *held = held.saturating_add_signed(by);
    if *held == 0 {
        counts.remove(&count);
    }
    if counts.is_empty() {
        partitions.remove(owner);
    }
}

impl Router {
    /// What the router saw of a table's copies, by node index.
    pub(super) fn copies(&self, cluster: &Cluster, table: &str) -> Vec<Option<TableCopy>> {
        let copies = self.copies.lock().unwrap_or_else(PoisonError::into_inner);
        let copies = copies.table(table);
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
        // The terms come from the table's own copies: one of an earlier
        // creation, left on a node that missed the drop, has terms of its own.
        let (terms, seen) = copies
            .table(table)
            .into_iter()
            .flat_map(HashMap::values)
            .filter(|copy| known.is_none() || copy.created.is_none() || copy.created == known)
            .max_by_key(|copy| (copy.term().term, copy.length))
            .map_or_else(
                || (Terms::default(), None),
                |copy| (copy.terms.clone(), copy.created.clone()),
            );
        let created = created.or(seen).or(known);
        copies.set(
            table,
            &node.name,
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
            .get(table, &node.name)
            .cloned()
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

    /// Forgets a node's copy of a table, once it is set aside.
    pub(super) fn forget(&self, table: &str, node: &Node) {
        self.copies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .forget(table, &node.name);
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

#[cfg(test)]
mod tests {
    use super::*;
    use gf_server::wal::Term;

    fn copy(length: u64, term: u64) -> TableCopy {
        let mut terms = Terms::default();
        if term > 0 {
            terms.begin(Term {
                term,
                primary: "a".to_string(),
                start: 0,
            });
        }
        TableCopy {
            length,
            terms,
            created: CreationId::parse("one"),
        }
    }

    fn answered(entries: &[(&str, &str, TableCopy)]) -> Tables {
        let mut tables = Tables::new();
        for (table, node, copy) in entries {
            tables
                .entry(table.to_string())
                .or_default()
                .insert(node.to_string(), copy.clone());
        }
        tables
    }

    #[test]
    fn a_check_takes_back_nothing_recorded_since_it_began() {
        let mut copies = Copies::default();
        copies.set("rows", "a", copy(1, 0));
        let since = copies.stamp();
        // While the check waits on the nodes, a write begins term 1 and grows the log.
        copies.update("rows", "a", |copy| *copy = self::copy(2, 1));
        copies.set("rows", "b", copy(2, 1));

        copies.install(answered(&[("rows", "a", copy(1, 0))]), since, &[]);

        assert_eq!(copies.get("rows", "a"), Some(&copy(2, 1)));
        assert_eq!(copies.get("rows", "b"), Some(&copy(2, 1)));
        // The next check's answers, begun after the write, stand.
        let since = copies.stamp();
        copies.install(answered(&[("rows", "a", copy(5, 1))]), since, &[]);
        assert_eq!(copies.get("rows", "a"), Some(&copy(5, 1)));
        assert_eq!(copies.get("rows", "b"), None);
    }

    #[test]
    fn a_check_never_brings_back_a_table_dropped_since_it_began() {
        let mut copies = Copies::default();
        copies.set("gone", "a", copy(3, 1));
        copies.set("kept", "a", copy(3, 1));
        let since = copies.stamp();
        copies.drop_table("gone");
        copies.forget("kept", "a");

        let stale = [("gone", "a", copy(3, 1)), ("kept", "a", copy(3, 1))];
        copies.install(answered(&stale), since, &[]);

        assert!(!copies.contains("gone"));
        assert!(!copies.contains("kept"));
        // Made again after the drop, the table's new copy stands.
        let since = copies.stamp();
        copies.drop_table("gone");
        copies.set("gone", "b", copy(1, 0));
        copies.install(answered(&stale), since, &[]);
        assert_eq!(copies.table("gone").map(HashMap::len), Some(1));
        assert_eq!(copies.get("gone", "b"), Some(&copy(1, 0)));
    }

    #[test]
    fn partition_counts_follow_the_partitions_held() {
        let mut copies = Copies::default();
        let name = |index, count| partitions::name("split", index, count);
        copies.set(&name(0, 2), "a", copy(1, 0));
        copies.set(&name(1, 2), "a", copy(1, 0));
        copies.set(&name(1, 2), "b", copy(1, 0));
        assert_eq!(copies.partition_counts("split"), [2]);
        assert!(copies.partition_counts("other").is_empty());

        copies.drop_table(&name(0, 2));
        assert_eq!(copies.partition_counts("split"), [2]);
        copies.forget(&name(1, 2), "a");
        copies.forget(&name(1, 2), "b");
        assert!(copies.partition_counts("split").is_empty());

        let since = copies.stamp();
        copies.install(
            answered(&[
                (&name(0, 3), "a", copy(1, 0)),
                (&name(0, 5), "a", copy(1, 0)),
            ]),
            since,
            &[],
        );
        assert_eq!(copies.partition_counts("split"), [3, 5]);
    }

    #[test]
    fn a_node_out_of_reach_keeps_its_names_taken_until_it_answers() {
        let mut copies = Copies::default();
        copies.set("rows", "a", copy(1, 0));
        copies.set(&partitions::name("split", 0, 2), "b", copy(1, 0));

        // b goes down: its copies leave the view, its names stay taken.
        let since = copies.stamp();
        copies.install(answered(&[("rows", "a", copy(1, 0))]), since, &["b"]);
        assert!(copies.partition_counts("split").is_empty());
        assert!(copies.held_out_of_reach("split"));
        assert!(!copies.held_out_of_reach("rows"));
        // Still down at the next check, it holds them still.
        let since = copies.stamp();
        copies.install(answered(&[("rows", "a", copy(1, 0))]), since, &["b"]);
        assert!(copies.held_out_of_reach("split"));

        // Back, it says for itself what it holds.
        let since = copies.stamp();
        copies.install(answered(&[("rows", "a", copy(1, 0))]), since, &[]);
        assert!(!copies.held_out_of_reach("split"));
    }
}
