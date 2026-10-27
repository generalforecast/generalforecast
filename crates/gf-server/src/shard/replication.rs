// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Applying shipments: log entries another copy shipped, staged and synced in
//! order, with snapshots that skip ahead compacting the log.

use super::compaction::apply_compacted;
use super::*;

impl Shard {
    pub(super) async fn replicate(
        &mut self,
        table: &str,
        entries: Vec<(u64, Vec<u8>)>,
        Lineage { terms, created }: Lineage,
    ) -> Result<u64, ServerError> {
        let id = self.usable(table)?;
        if self.compacting.contains_key(table) {
            return Err(ServerError::Busy(format!(
                "table {table:?} is compacting its log; ship again shortly"
            )));
        }
        let entry = &self.tables[table];
        if created.is_some_and(|created| entry.definition.created.as_ref() != Some(&created)) {
            return Err(ServerError::Invalid(format!(
                "the shipped copy is of another table named {table:?}"
            )));
        }
        if let Some(terms) = terms
            && !terms.starts_with(entry.log.terms())
        {
            return Err(ServerError::Invalid(format!(
                "this copy of table {table:?} has terms the shipped copy's log never passed through"
            )));
        }
        let mut staged = Vec::new();
        let shipped = self.stage_shipment(table, id, entries, &mut staged).await;
        // Records staged before a failure are whole and run on from the log,
        // so they apply, unless the failure was the log's and cut them off.
        let applied = if self.tables[table].log.has_staged() {
            self.apply_staged(id, staged).await
        } else {
            Ok(())
        };
        if let Some(entry) = self.tables.get_mut(table) {
            entry.log.discard_staged();
        }
        shipped.and(applied)?;
        Ok(self.tables[table].log.length())
    }

    /// Stages a shipment's records in order, applying snapshots as they come.
    pub(super) async fn stage_shipment(
        &mut self,
        table: &str,
        id: TableId,
        entries: Vec<(u64, Vec<u8>)>,
        staged: &mut Vec<(Record, u64)>,
    ) -> Result<(), ServerError> {
        for (sequence, bytes) in entries {
            if sequence == 0 {
                // A source compacted past its term record carries the term here.
                if let LogEntry::Definition(_, terms) = decode(&bytes)? {
                    let entry = self.tables.get_mut(table).expect("checked above");
                    entry.log.adopt(terms);
                }
                continue;
            }
            let length = self.tables[table].log.length();
            if sequence < length {
                continue;
            }
            match decode(&bytes)? {
                // A snapshot may skip ahead: it replaces everything before it.
                // The log takes it first, so a failure there changes nothing;
                // rows the log then holds but the node cannot take break the
                // table, as any logged record does.
                LogEntry::Record(snapshot @ Record::Snapshot(_)) => {
                    self.apply_staged(id, std::mem::take(staged)).await?;
                    let entry = self.tables.get_mut(table).expect("checked above");
                    if let Some(strategy) = snapshot_strategy(&snapshot) {
                        take_strategy(&mut entry.definition, strategy);
                    }
                    entry.log.compact(
                        &entry.definition,
                        sequence,
                        std::slice::from_ref(&snapshot),
                    )?;
                    self.unsaved_heads.insert(self.names[&id].clone());
                    self.track_series(id, &snapshot);
                    let Record::Snapshot(rows) = snapshot else {
                        unreachable!("matched a snapshot above");
                    };
                    self.learn(id, &rows.words, true);
                    self.restore_writes(id, &rows.writes);
                    apply_compacted(&mut self.node, id, rows)
                        .map_err(ServerError::from)
                        .inspect_err(|error| self.break_table(id, unapplied(error)))?;
                    let purged = self.node.take_drops();
                    self.send_drops(&purged).await;
                }
                _ if sequence > length => {
                    return Err(ServerError::Gap(format!(
                        "table {table:?} has {length} log entries, so entry {sequence} cannot apply yet"
                    )));
                }
                LogEntry::Record(record) => {
                    let log = &mut self.tables.get_mut(table).expect("checked above").log;
                    log.stage(&record)?;
                    staged.push((record, log.length()));
                }
                LogEntry::Definition(..) => {
                    return Err(ServerError::Invalid(
                        "a table's definition can only be its first log entry".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Syncs a shipment's staged records, then applies them in order: the
    /// primary checked them, so they apply as they did there.
    pub(super) async fn apply_staged(
        &mut self,
        id: TableId,
        staged: Vec<(Record, u64)>,
    ) -> Result<(), ServerError> {
        if staged.is_empty() {
            return Ok(());
        }
        let name = self.names[&id].clone();
        let synced = self.tables.get_mut(&name).expect("found above").log.sync();
        self.check_poisoned(&name);
        synced?;
        self.unsaved_heads.insert(name);
        for (record, log_length) in staged {
            let (_, dropped) = self
                .apply_record(id, record, log_length)
                .inspect_err(|error| self.break_table(id, unapplied(error)))?;
            self.send_drops(&dropped).await;
        }
        let purged = self.node.take_drops();
        self.send_drops(&purged).await;
        self.lift_quarantine(id);
        Ok(())
    }
}
