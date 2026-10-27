// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Compacting a table's log in the background, and installing the rewrite
//! once it is done.

use super::*;

impl Shard {
    /// Rewrites a table's log as a snapshot of its live rows, in key order,
    /// then renumbers the rows as replicas and restarts that read the snapshot
    /// will, so this copy keeps the same row ids, pages and samples as theirs.
    /// The log goes first, so a rewrite that fails leaves the rows as they were.
    ///
    /// Rows past `snapshot_bytes` follow the snapshot as inserts, in key order
    /// still, so a table of any size compacts into records that each fit a
    /// log entry and seal in memory one at a time.
    pub(super) fn start_compaction(&mut self, table: &str) -> Result<(), ServerError> {
        if self.compacting.contains_key(table) {
            return Ok(());
        }
        let id = self.usable(table)?;
        let mut source = self.node.compaction_source(id)?;
        let (name, entry) = self
            .tables
            .get_key_value(table)
            .expect("usable tables are served");
        // The strategy as given, which a clamp may have cut in the node's.
        source.table.strategy = Some(entry.definition.strategy.strategy());
        let compactor = entry
            .log
            .begin_compaction(&entry.definition, entry.log.length())?;
        let number = compactor.number();
        let row_bytes = (entry.definition.features() * 4 + 48) as u64;
        let per_record = usize::try_from(self.snapshot_bytes / row_bytes)
            .unwrap_or(usize::MAX)
            .max(1);
        let words = entry
            .vocabularies
            .iter()
            .enumerate()
            .filter(|(_, vocabulary)| !vocabulary.words().is_empty())
            .map(|(column, vocabulary)| Words {
                column,
                values: vocabulary.words().to_vec(),
            })
            .collect();
        let series = entry.series.keys().to_vec();
        let writes = self.recent.get(&id).map_or_else(Vec::new, |recent| {
            recent
                .order
                .iter()
                .map(|write| {
                    let written = recent.written[write];
                    PastWrite {
                        id: write.clone(),
                        rows: written.rows as u64,
                        log_length: written.log_length,
                    }
                })
                .collect()
        });
        let name = name.clone();
        let sender = self.compacted.clone();
        let table = name.clone();
        tokio::task::spawn_blocking(move || {
            let compacted = write_compaction(compactor, &source, words, series, writes, per_record);
            let _ = sender.send(Finished {
                table,
                number,
                compacted,
            });
        });
        self.compacting.insert(
            name,
            Compacting {
                number,
                ..Compacting::default()
            },
        );
        Ok(())
    }

    /// Installs a log rewritten in the background, answers those waiting for
    /// it, and commits the writes it held back.
    pub(super) async fn finish_compaction(&mut self, finished: Finished) {
        let Finished {
            table,
            number,
            compacted,
        } = finished;
        if self
            .compacting
            .get(&table)
            .is_none_or(|compacting| compacting.number != number)
        {
            // An earlier table of the name, since dropped.
            if let Ok(compacted) = compacted {
                compacted.discard();
            }
            return;
        }
        let compacting = self.compacting.remove(&table).expect("checked above");
        let installed = compacted
            .map_err(ServerError::Internal)
            .and_then(|compacted| {
                let entry = self
                    .tables
                    .get_mut(&table)
                    .expect("compacting tables are served");
                entry.log.install_compaction(compacted)?;
                Ok(entry.log.length())
            });
        if self.tables.contains_key(&table) {
            self.unsaved_heads.insert(table.clone());
        }
        match &installed {
            Ok(length) => {
                let entry = self
                    .tables
                    .get_mut(&table)
                    .expect("compacting tables are served");
                entry.compact_wait = COMPACT_RECORDS;
                tracing::info!(table = table.as_str(), length, "compacted a log");
            }
            Err(error) => self.compaction_failed(&table, &error.to_string()),
        }
        for reply in compacting.replies {
            let _ = reply.send(installed.clone());
        }
        self.commit_batch(compacting.held).await;
    }

    /// Waits longer before compacting a table again after a compaction failed.
    pub(super) fn compaction_failed(&mut self, table: &str, error: &str) {
        let Some(entry) = self.tables.get_mut(table) else {
            return;
        };
        entry.compact_after = entry.log.length() + entry.compact_wait;
        entry.compact_wait = entry.compact_wait.saturating_mul(2);
        tracing::error!(
            table,
            "compacting a log: {error}; trying again after {} more records",
            entry.compact_after - entry.log.length()
        );
    }
}

pub(super) fn apply_compacted(node: &mut Node, id: TableId, rows: Rows) -> Result<(), NodeError> {
    let count = rows.keys.len();
    let table = TableRows {
        keys: rows.keys,
        features: rows.features,
        targets: rows.targets,
        times: rows.times.unwrap_or_else(|| vec![None; count]),
    };
    match rows.layout {
        Some(layout) => node.restore(id, &table, &layout, rows.table.as_deref()),
        None => node.replace_rows(
            id,
            &table.keys,
            &table.features,
            table.targets,
            Some(&table.times),
        ),
    }
}

/// Writes a table's compaction, part by part, from rows as they were when
/// it began: the snapshot first, with the table's layout, its words, its
/// series' keys and its clients' recent writes, then the rest as inserts
/// that continue it.
fn write_compaction(
    mut compactor: Compactor,
    source: &CompactionSource,
    mut words: Vec<Words>,
    mut series: Vec<SeriesKey>,
    mut writes: Vec<PastWrite>,
    per_record: usize,
) -> Result<Compacted, String> {
    let total = source.layout.ids.len();
    let mut start = 0;
    loop {
        let end = (start + per_record).min(total);
        let (part, layout) = source.part(start, end).map_err(|error| error.to_string())?;
        let first = start == 0;
        let rows = Rows {
            keys: part.keys,
            features: part.features,
            targets: part.targets,
            times: Some(part.times),
            id: None,
            words: std::mem::take(&mut words),
            series: std::mem::take(&mut series),
            writes: std::mem::take(&mut writes),
            layout: Some(layout),
            table: first.then(|| Box::new(source.table.clone())),
        };
        let record = if first {
            Record::Snapshot(rows)
        } else {
            Record::Insert(rows)
        };
        compactor
            .append(&record)
            .map_err(|error| error.to_string())?;
        start = end;
        if start >= total {
            return compactor.finish().map_err(|error| error.to_string());
        }
    }
}
