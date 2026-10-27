// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rendezvous hashing: every table ranks every node, and the top node serves it.
//!
//! Adding a node moves only the tables it now ranks first, and removing one
//! moves only the tables it served, so a node set change disturbs as little as
//! possible. The hash is fixed, so every router agrees on the placement.

use gf_assign::key::hashed;

/// The nodes tables are placed on, in configuration order.
#[derive(Clone, Debug)]
pub struct Placement {
    nodes: Vec<String>,
}

impl Placement {
    /// A placement over named nodes.
    #[must_use]
    pub fn new(nodes: Vec<String>) -> Self {
        Self { nodes }
    }

    /// Node indices for a table, from the one that serves it down.
    #[must_use]
    pub fn ranked(&self, table: &str) -> Vec<usize> {
        let mut scores: Vec<(u64, usize)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (hashed(format!("{node}\0{table}").as_bytes()), index))
            .collect();
        scores.sort_unstable_by(|left, right| right.cmp(left));
        scores.into_iter().map(|(_, index)| index).collect()
    }

    /// The node that serves a table.
    ///
    /// # Panics
    ///
    /// If there are no nodes.
    #[must_use]
    pub fn primary(&self, table: &str) -> usize {
        self.ranked(table)[0]
    }
}
