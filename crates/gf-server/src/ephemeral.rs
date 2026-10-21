// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A shard's ephemeral tables: request contexts kept in memory alone, found
//! by name or id, with the unpinned ones in order of last use, so eviction
//! and expiry take the least recently used without scanning the others.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use gf_core::node::TableId;
use tokio::time::Instant;

use crate::metrics::TableMetrics;
use crate::name::TableName;
use crate::shard::EphemeralLimits;
use crate::wal::TableDefinition;

/// A table holding one request context's rows in memory alone, named
/// `ctx-<fingerprint>` for the context, so requests with the same context
/// read the same table and reuse its fits and encoded contexts.
pub struct Ephemeral {
    pub id: TableId,
    /// What names its context, and routes requests to its shard.
    pub fingerprint: [u8; 32],
    pub definition: TableDefinition,
    /// The quantile level of each prediction column, for a table predicting values.
    pub levels: Option<Vec<f64>>,
    /// The bytes of its rows: keys, features and targets.
    pub bytes: u64,
    pub metrics: Arc<TableMetrics>,
    /// When it was made, or last stopped being read.
    used: Instant,
    /// The requests queued on it; while any are, it is pinned.
    pins: usize,
}

impl Ephemeral {
    #[must_use]
    pub fn new(
        id: TableId,
        fingerprint: [u8; 32],
        definition: TableDefinition,
        levels: Option<Vec<f64>>,
        bytes: u64,
        metrics: Arc<TableMetrics>,
    ) -> Self {
        Self {
            id,
            fingerprint,
            definition,
            levels,
            bytes,
            metrics,
            used: Instant::now(),
            pins: 0,
        }
    }
}

/// The ephemeral tables of a shard, within its `EphemeralLimits`.
///
/// A table is pinned while requests are queued on it, and counts as used
/// when the last of them leaves; only unpinned tables are ordered for
/// eviction, so pinning, eviction and the next expiry cost a logarithm of
/// the tables kept, not a pass over them.
pub struct EphemeralTables {
    limits: EphemeralLimits,
    tables: HashMap<TableId, Ephemeral>,
    names: HashMap<TableName, TableId>,
    /// The unpinned tables, least recently used first.
    unpinned: BTreeSet<(Instant, TableId)>,
    bytes: u64,
}

impl EphemeralTables {
    #[must_use]
    pub fn new(limits: EphemeralLimits) -> Self {
        Self {
            limits,
            tables: HashMap::new(),
            names: HashMap::new(),
            unpinned: BTreeSet::new(),
            bytes: 0,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    #[must_use]
    pub fn named(&self, name: &TableName) -> Option<&Ephemeral> {
        self.names.get(name).map(|id| &self.tables[id])
    }

    #[must_use]
    pub fn get(&self, id: TableId) -> Option<&Ephemeral> {
        self.tables.get(&id)
    }

    /// Keeps a table, unpinned and used now; its name must be new.
    pub fn insert(&mut self, table: Ephemeral) {
        debug_assert!(!self.names.contains_key(&table.definition.name));
        self.bytes += table.bytes;
        self.unpinned.insert((table.used, table.id));
        self.names.insert(table.definition.name.clone(), table.id);
        self.tables.insert(table.id, table);
    }

    /// Pins a table for a request queued on it; other tables are not ephemeral.
    pub fn pin(&mut self, id: TableId) {
        if let Some(table) = self.tables.get_mut(&id) {
            if table.pins == 0 {
                self.unpinned.remove(&(table.used, id));
            }
            table.pins += 1;
        }
    }

    /// Unpins a table for a request that left its queue; the last one to
    /// leave marks it used at `now`.
    pub fn unpin(&mut self, id: TableId, now: Instant) {
        if let Some(table) = self.tables.get_mut(&id)
            && table.pins > 0
        {
            table.pins -= 1;
            if table.pins == 0 {
                table.used = now;
                self.unpinned.insert((now, id));
            }
        }
    }

    pub fn remove(&mut self, id: TableId) -> Option<Ephemeral> {
        let table = self.tables.remove(&id)?;
        self.names.remove(&table.definition.name);
        self.unpinned.remove(&(table.used, id));
        self.bytes -= table.bytes;
        Some(table)
    }

    /// Takes every table, as when the worker starts again without them.
    pub fn drain(&mut self) -> Vec<Ephemeral> {
        self.names.clear();
        self.unpinned.clear();
        self.bytes = 0;
        self.tables.drain().map(|(_, table)| table).collect()
    }

    /// When the least recently used unpinned table goes idle, if there is one.
    #[must_use]
    pub fn expiry(&self) -> Option<Instant> {
        self.unpinned
            .first()
            .map(|(used, _)| *used + self.limits.idle)
    }

    /// Takes the unpinned tables to evict at `now`, least recently used
    /// first: while more tables or bytes are kept than the limits allow, and
    /// any that went unused for the idle limit.
    pub fn evict(&mut self, now: Instant) -> Vec<Ephemeral> {
        let mut evicted = Vec::new();
        while let Some(&(used, id)) = self.unpinned.first() {
            let over = self.tables.len() > self.limits.tables || self.bytes > self.limits.bytes;
            if !over && now.duration_since(used) < self.limits.idle {
                break;
            }
            evicted.extend(self.remove(id));
        }
        evicted
    }
}
