//! Where a table's copies should be shipped, released and read, and which may
//! take over its writes.
//!
//! These are pure decisions over what the router last saw: the replicas a
//! table should have (its top-ranked nodes, primary first), each node's log
//! length for it (`None` without a copy), and which nodes are up. Logs are
//! prefixes of the primary's, so the longest log is the most current copy;
//! copies whose writes diverged from the current term are left out first.

use std::collections::BTreeMap;

use gf_server::wal::{CreationId, Term, Terms};

/// What the router saw of one copy of a table: its log length, terms and creation id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableCopy {
    pub length: u64,
    /// Every term the copy has seen, oldest first.
    pub terms: Terms,
    /// The creation the copy belongs to, if known yet; a copy of an earlier
    /// table of the same name, left on a node that was down when it was
    /// dropped, differs.
    pub created: Option<CreationId>,
}

impl TableCopy {
    /// The copy's latest term.
    #[must_use]
    pub fn term(&self) -> &Term {
        self.terms.latest()
    }
}

/// The copy at the latest term, longest first: its terms say who takes writes.
#[must_use]
pub fn current(copies: &[Option<TableCopy>]) -> Option<&TableCopy> {
    copies
        .iter()
        .flatten()
        .max_by_key(|copy| (copy.term().term, copy.length))
}

/// The creation the table's copies belong to: the one most of its ranked
/// `replicas` hold, and among equals the latest, since creation ids begin
/// with the creating node's clock. A copy of an earlier table of the same
/// name survives only on a node that missed its drop, so the replicas outvote
/// it however their clocks compare. A copy recorded from a write before any
/// check knows no creation yet.
#[must_use]
pub fn creation(copies: &[Option<TableCopy>], replicas: &[usize]) -> Option<CreationId> {
    let mut votes: BTreeMap<&CreationId, usize> = BTreeMap::new();
    for (node, copy) in copies.iter().enumerate() {
        if let Some(created) = copy.as_ref().and_then(|copy| copy.created.as_ref()) {
            *votes.entry(created).or_default() += usize::from(replicas.contains(&node));
        }
    }
    votes
        .into_iter()
        .max_by_key(|&(created, count)| (count, created.clone()))
        .map(|(created, _)| created.clone())
}

/// Whether a copy is of the table's creation, or of none known yet.
fn of_creation(copy: &TableCopy, creation: Option<&CreationId>) -> bool {
    copy.created.is_none()
        || creation.is_none_or(|creation| copy.created.as_ref() == Some(creation))
}

/// The copies of the table's `creation`, with those of another blanked out.
#[must_use]
pub fn same_creation(
    copies: &[Option<TableCopy>],
    creation: Option<&CreationId>,
) -> Vec<Option<TableCopy>> {
    copies
        .iter()
        .map(|copy| copy.clone().filter(|copy| of_creation(copy, creation)))
        .collect()
}

/// Copies holding writes the current term's log lacks, or of another creation
/// than the table's `creation`.
///
/// A copy at an older term follows the current log only if that log passed
/// through the copy's term, and the copy ends before the next term began;
/// anything after that was written to a primary that has since been replaced.
/// A copy at the current term number but begun by another promotion diverged
/// too, and so did a copy of an earlier table of the same name.
#[must_use]
pub fn diverged(copies: &[Option<TableCopy>], creation: Option<&CreationId>) -> Vec<bool> {
    let same = same_creation(copies, creation);
    let Some(current) = current(&same) else {
        return vec![false; copies.len()];
    };
    copies
        .iter()
        .map(|copy| {
            copy.as_ref().is_some_and(|copy| {
                if !of_creation(copy, creation) {
                    return true;
                }
                let term = copy.term();
                // Two routers may each have promoted a copy at one term number.
                if term.term == current.term().term {
                    return term != current.term();
                }
                if term.term > current.term().term {
                    return false;
                }
                let next = current.terms.iter().find(|later| later.term > term.term);
                !current.terms.contains(term) || next.is_some_and(|next| copy.length > next.start)
            })
        })
        .collect()
}

/// The replica to promote when the primary is lost: the longest that is up and
/// at the current term, the first in rank order among equals, if it holds every
/// write the router has seen acknowledged. None otherwise, so none is promoted.
///
/// An acknowledged write is on its primary and at least one replica. So every
/// replica but the primary must be up at the current term: one of them then
/// holds each acknowledged write, and the router sees it even after a restart.
/// Copies at one term are prefixes of one log, so the longest holds them all.
#[must_use]
pub fn promotion(
    desired: &[usize],
    copies: &[Option<TableCopy>],
    up: &[bool],
    primary: usize,
    seen: u64,
) -> Option<usize> {
    let term = current(copies)?.term().term;
    let current = |node: usize| {
        up[node]
            && copies[node]
                .as_ref()
                .is_some_and(|copy| copy.term().term == term)
    };
    let reachable = desired.iter().filter(|&&node| current(node)).count();
    if reachable + 1 < desired.len() {
        return None;
    }
    let length = |node: usize| copies[node].as_ref().map_or(0, |copy| copy.length);
    desired
        .iter()
        .copied()
        .filter(|&node| node != primary && current(node))
        .fold(None, |best: Option<usize>, node| match best {
            Some(best) if length(best) >= length(node) => Some(best),
            _ => Some(node),
        })
        .filter(|&node| length(node) >= seen)
}

/// A change that brings a table's copies to where they should be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repair {
    /// Ship log entries from sequence `from` on, from one node to another.
    Ship {
        source: usize,
        target: usize,
        from: u64,
    },
    /// Drop a copy that no replica needs.
    Release { node: usize },
}

/// The most current copy, preferring replicas in rank order.
fn longest(desired: &[usize], lengths: &[Option<u64>]) -> Option<(usize, u64)> {
    let longest = lengths.iter().flatten().copied().max()?;
    let source = desired
        .iter()
        .copied()
        .chain(0..lengths.len())
        .find(|&node| lengths[node] == Some(longest))?;
    Some((source, longest))
}

/// Ships to every replica that is up and behind; once all are current, releases other copies.
#[must_use]
pub fn repairs(desired: &[usize], lengths: &[Option<u64>], up: &[bool]) -> Vec<Repair> {
    let Some((source, longest)) = longest(desired, lengths) else {
        return Vec::new();
    };
    let ships: Vec<Repair> = desired
        .iter()
        .copied()
        .filter(|&node| up[node] && lengths[node] != Some(longest))
        .map(|target| Repair::Ship {
            source,
            target,
            from: lengths[target].unwrap_or(0),
        })
        .collect();
    if !ships.is_empty() || desired.iter().any(|&node| lengths[node] != Some(longest)) {
        return ships;
    }
    (0..lengths.len())
        .filter(|node| up[*node] && lengths[*node].is_some() && !desired.contains(node))
        .map(|node| Repair::Release { node })
        .collect()
}

/// The nodes to try for a read: current replicas in rank order, then any other current copy.
///
/// With nothing known about the table yet, the replicas that are up, primary first.
#[must_use]
pub fn read_order(desired: &[usize], lengths: &[Option<u64>], up: &[bool]) -> Vec<usize> {
    let Some((_, longest)) = longest(desired, lengths) else {
        return desired.iter().copied().filter(|&node| up[node]).collect();
    };
    let current = |node: &usize| up[*node] && lengths[*node] == Some(longest);
    let others = (0..lengths.len()).filter(|node| !desired.contains(node));
    desired
        .iter()
        .copied()
        .chain(others)
        .filter(current)
        .collect()
}
