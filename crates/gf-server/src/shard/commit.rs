// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Committing writes: each batch staged in its tables' logs, synced once,
//! then applied, and the records a log holds applied to the node.

use super::compaction::apply_compacted;
use super::*;

impl Shard {
    /// Applies a logged record to the node and remembers its write id, with
    /// `log_length` the log's length once the record was logged. Returns what
    /// the executor is to drop: a refit's entries nothing reads any more.
    pub(super) fn apply_record(
        &mut self,
        id: TableId,
        record: Record,
        log_length: u64,
    ) -> Result<(Written, Vec<Movement>), ServerError> {
        self.learn(id, record.words(), matches!(record, Record::Snapshot(_)));
        self.track_series(id, &record);
        self.apply_logged(id, record, log_length)
    }

    /// Follows a record's series keys in its table's.
    pub(super) fn track_series(&mut self, id: TableId, record: &Record) {
        if let Some(entry) = self
            .names
            .get(&id)
            .and_then(|name| self.tables.get_mut(name))
        {
            entry.series.track(record);
        }
    }

    /// Applies a logged record as `apply_record` does, but for the words it
    /// adds, which the caller takes.
    pub(super) fn apply_logged(
        &mut self,
        id: TableId,
        record: Record,
        log_length: u64,
    ) -> Result<(Written, Vec<Movement>), ServerError> {
        let write_id = record.write_id().map(str::to_string);
        if let Record::Snapshot(rows) = &record {
            self.restore_writes(id, &rows.writes);
        }
        let mut dropped = Vec::new();
        let rows = match record {
            Record::Refit => {
                dropped = self.node.refit(id)?;
                0
            }
            Record::Strategy(strategy) => {
                self.node.set_strategy(id, strategy.strategy())?;
                if let Some(entry) = self
                    .names
                    .get(&id)
                    .and_then(|name| self.tables.get_mut(name))
                {
                    entry.definition.strategy = strategy;
                }
                0
            }
            Record::Term { .. } => 0,
            record => {
                let carried = snapshot_strategy(&record);
                let rows = self.change(id, record)?;
                if let Some(strategy) = carried
                    && let Some(entry) = self
                        .names
                        .get(&id)
                        .and_then(|name| self.tables.get_mut(name))
                {
                    take_strategy(&mut entry.definition, strategy);
                }
                rows
            }
        };
        let written = Written { rows, log_length };
        if let Some(write_id) = write_id {
            self.recent
                .entry(id)
                .or_default()
                .remember(write_id, written);
        }
        Ok((written, dropped))
    }

    /// Applies a record that changes rows; returns how many rows it changed.
    pub(super) fn change(&mut self, id: TableId, record: Record) -> Result<usize, NodeError> {
        let node = &mut self.node;
        match record {
            Record::Insert(rows) if rows.layout.is_some() => {
                let count = rows.keys.len();
                apply_compacted(node, id, rows)?;
                Ok(count)
            }
            Record::Insert(rows) => {
                let times = rows.times.as_deref();
                Ok(node
                    .insert_at(id, &rows.keys, &rows.features, rows.targets, times)?
                    .len())
            }
            Record::Update(rows) => {
                let times = rows.times.as_deref();
                Ok(node
                    .update_at(id, &rows.keys, &rows.features, rows.targets, times)?
                    .len())
            }
            Record::Upsert(rows) => {
                let times = rows.times.as_deref();
                Ok(node
                    .upsert_at(id, &rows.keys, &rows.features, rows.targets, times)?
                    .len())
            }
            Record::Delete { keys, .. } => Ok(node.delete(id, &keys)?.len()),
            Record::Truncate => Ok(node.truncate(id)?.len()),
            Record::Snapshot(rows) => {
                let count = rows.keys.len();
                apply_compacted(node, id, rows)?;
                Ok(count)
            }
            Record::Refit | Record::Strategy(_) | Record::Term { .. } => {
                unreachable!("refits, strategies and terms change no rows")
            }
        }
    }

    /// Commits queued writes: each is checked and staged in its table's log,
    /// the logs are synced once, and then each is applied. Writes to a key an
    /// earlier write in the batch touched, and those after a log that refused
    /// a write, go into the next batch.
    pub(super) async fn commit_batch(&mut self, mut batch: Vec<Queued>) {
        if batch.is_empty() {
            return;
        }
        while !batch.is_empty() {
            batch = self.commit_some(batch).await;
        }
        self.report_cache();
    }

    /// Commits a prefix of the batch and returns the rest.
    async fn commit_some(&mut self, batch: Vec<Queued>) -> Vec<Queued> {
        let mut staged: Vec<Staged> = Vec::new();
        let mut touched: HashMap<TableName, HashSet<u128>> = HashMap::new();
        // Tables whose vocabularies a staged write extends; a later write with
        // strings for one would code them against words not yet applied.
        let mut extended: HashSet<TableName> = HashSet::new();
        // Tables a staged truncate empties; a later write to one checks against
        // rows that are about to go, so it waits for the next batch.
        let mut truncated: HashSet<TableName> = HashSet::new();
        // Tables with neither rows nor labels yet, whose first write in the
        // batch says whether their targets are labels or codes; a later one
        // waits, so it is checked against what that write decided.
        let mut unsettled: HashSet<TableName> = HashSet::new();
        let mut rest = Vec::new();
        let mut queued = batch.into_iter();
        for next in queued.by_ref() {
            let conflicts = touched
                .get(&next.table)
                .is_some_and(|keys| next.write.keys().iter().any(|key| keys.contains(key)));
            if conflicts
                || (next.write.has_strings() && extended.contains(&next.table))
                || (next.write.has_targets() && unsettled.contains(&next.table))
                || truncated.contains(&next.table)
            {
                rest.push(next);
                break;
            }
            let Queued {
                table,
                write,
                fence,
                charge,
                reply,
            } = next;
            let prepared = self.prepare(&table, write);
            let (id, record) = match prepared {
                Ok(Ok((id, record))) => (id, record),
                // A write the table already applied, or one that is refused.
                Ok(Err(written)) => {
                    let _ = reply.send(Ok(written));
                    continue;
                }
                Err(error) => {
                    self.metrics.requests_failed.add(1);
                    let _ = reply.send(Err(error));
                    continue;
                }
            };
            let checked = self.check(id, &table, &record).and_then(|()| {
                let log = &self.tables[&table].log;
                fence.check(&table, log)
            });
            if let Err(error) = checked {
                self.metrics.requests_failed.add(1);
                let _ = reply.send(Err(error));
                continue;
            }
            let log = &mut self.tables.get_mut(&table).expect("checked").log;
            match log.stage(&record) {
                Ok(()) => {
                    let log_length = log.length();
                    touched
                        .entry(table.clone())
                        .or_default()
                        .extend(record.keys());
                    if !record.words().is_empty() {
                        extended.insert(table.clone());
                    }
                    if matches!(record, Record::Truncate) {
                        truncated.insert(table.clone());
                    }
                    if !record.keys().is_empty() && !self.settled(&table) {
                        unsettled.insert(table.clone());
                    }
                    staged.push(Staged {
                        table,
                        id,
                        record,
                        log_length,
                        charge,
                        reply,
                    });
                }
                Err(error) => {
                    // A write the log refused cut its staged ones back off too.
                    self.check_poisoned(&table);
                    self.metrics.requests_failed.add(1);
                    let _ = reply.send(Err(error.into()));
                    let (failed, kept): (Vec<Staged>, Vec<Staged>) =
                        staged.into_iter().partition(|staged| staged.table == table);
                    staged = kept;
                    for item in failed {
                        self.metrics.requests_failed.add(1);
                        let _ = item.reply.send(Err(ServerError::Internal(
                            "the log refused a later write in the batch".to_string(),
                        )));
                    }
                    break;
                }
            }
        }
        rest.extend(queued);
        // One sync per log, all at once, then every staged write applies in order.
        let logs: HashSet<&str> = staged.iter().map(|item| item.table.as_str()).collect();
        let synced = sync_logs(&mut self.tables, &logs);
        self.unsaved_heads.extend(synced.keys().cloned());
        for table in synced.keys() {
            self.check_poisoned(table);
        }
        let mut applied = Vec::new();
        for Staged {
            table,
            id,
            record,
            log_length,
            charge,
            reply,
        } in staged
        {
            if let Err(error) = &synced[&table] {
                self.metrics.requests_failed.add(1);
                drop(charge);
                let _ = reply.send(Err(error.clone()));
                continue;
            }
            // Records after one the rows could not take wait for the replay.
            if let Err(error) = self.usable(&table) {
                self.metrics.requests_failed.add(1);
                drop(charge);
                let _ = reply.send(Err(error));
                continue;
            }
            let written = match self.apply_record(id, record, log_length) {
                Ok((written, dropped)) => {
                    self.metrics.rows_written.add(written.rows as u64);
                    self.send_drops(&dropped).await;
                    Ok(written)
                }
                Err(error) => {
                    self.metrics.requests_failed.add(1);
                    self.break_table(id, unapplied(&error));
                    Err(error)
                }
            };
            drop(charge);
            let _ = reply.send(written);
            applied.push((table, id));
        }
        // No cache keeps a row's old values once it changes.
        let purged = self.node.take_drops();
        self.send_drops(&purged).await;
        for (table, id) in applied {
            self.lift_quarantine(id);
            self.compact_if_large(&table).await;
        }
        rest
    }

    /// Resolves a write into the record to log, or the answer for one the
    /// table already applied under its client id.
    fn prepare(
        &mut self,
        table: &str,
        write: Write,
    ) -> Result<Result<(TableId, Record), Written>, ServerError> {
        let id = self.usable(table)?;
        // A retry answers as the write did, before its keys are filtered, so
        // a delete with missing_ok reports the rows it deleted the first time.
        if let Some(write_id) = write.write_id()
            && let Some(written) = self
                .recent
                .get(&id)
                .and_then(|recent| recent.written.get(write_id))
        {
            return Ok(Err(*written));
        }
        let record = match write {
            Write::Insert(incoming) => Record::Insert(self.encode(table, incoming)?),
            Write::Update(incoming) => Record::Update(self.encode(table, incoming)?),
            Write::Upsert(incoming) => Record::Upsert(self.encode(table, incoming)?),
            Write::Label(mut incoming) => {
                // Logged as an update with the rows' features as they are.
                incoming.rows.features = self.node.features(id, &incoming.rows.keys)?;
                Record::Update(self.encode(table, incoming)?)
            }
            Write::Delete {
                keys,
                missing_ok,
                id: write_id,
            } => {
                let keys = if missing_ok {
                    self.node.present(id, &keys)?
                } else {
                    keys
                };
                if keys.is_empty() {
                    let log_length = self.tables[table].log.length();
                    return Ok(Err(Written {
                        rows: 0,
                        log_length,
                    }));
                }
                Record::Delete { keys, id: write_id }
            }
            Write::Refit => Record::Refit,
            Write::Truncate => Record::Truncate,
            Write::Strategy(strategy) => Record::Strategy(strategy),
        };
        Ok(Ok((id, record)))
    }

    /// Checks a record against the node before it is logged, so the log and
    /// the node never disagree.
    pub(super) fn check(
        &self,
        id: TableId,
        table: &str,
        record: &Record,
    ) -> Result<(), ServerError> {
        match record {
            Record::Insert(rows) => {
                self.node.check_insert(
                    id,
                    &rows.keys,
                    &rows.features,
                    &rows.targets,
                    rows.times.as_deref(),
                )?;
            }
            Record::Update(rows) => {
                self.node.check_update(
                    id,
                    &rows.keys,
                    &rows.features,
                    &rows.targets,
                    rows.times.as_deref(),
                )?;
            }
            Record::Upsert(rows) => {
                self.node.check_upsert(
                    id,
                    &rows.keys,
                    &rows.features,
                    &rows.targets,
                    rows.times.as_deref(),
                )?;
            }
            Record::Delete { keys, .. } => self.node.check_delete(id, keys)?,
            Record::Refit | Record::Truncate => {}
            Record::Strategy(strategy) => {
                if self.tables[table].definition.series.is_some() {
                    return Err(ServerError::Invalid(
                        "a table of series takes no strategy: its groups are its contexts"
                            .to_string(),
                    ));
                }
                strategy.strategy().check().map_err(NodeError::Strategy)?;
            }
            Record::Term { term, .. } => {
                let current = self.tables[table].log.term().term;
                if *term <= current {
                    return Err(ServerError::Gap(format!(
                        "table {table:?} is already at term {current}"
                    )));
                }
            }
            Record::Snapshot(_) => {
                return Err(ServerError::Invalid(
                    "snapshots come from compaction, not writes".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Changed rows or a refit give a quarantined table another try.
    pub(super) fn lift_quarantine(&mut self, id: TableId) {
        self.failures.remove(&id);
        if self.quarantined.remove(&id).is_some() {
            self.metrics.tables_quarantined.sub(1);
        }
    }

    /// Compacts a log that has grown past `COMPACT_FACTOR` times its live
    /// rows, once enough records have followed its last compaction, or its
    /// last failed try.
    async fn compact_if_large(&mut self, table: &str) {
        let Some(entry) = self.tables.get(table) else {
            return;
        };
        let Ok(rows) = self.node.len(entry.id) else {
            return;
        };
        let row_bytes = (entry.definition.features() * 4 + 24) as u64;
        let live = (rows * row_bytes).max(1 << 20);
        let length = entry.log.length();
        let since = length.saturating_sub(entry.log.base());
        if since < COMPACT_RECORDS
            || entry.log.bytes() <= COMPACT_FACTOR * live
            || length < entry.compact_after
        {
            return;
        }
        if let Err(error) = self.start_compaction(table) {
            self.compaction_failed(table, &error.to_string());
        }
    }

    /// Commits one record alone: checked, logged and synced, then applied.
    pub(super) async fn commit(
        &mut self,
        table: &str,
        record: Record,
        fence: &Fence,
    ) -> Result<Written, ServerError> {
        let id = self.usable(table)?;
        self.check(id, table, &record)?;
        let log = &mut self.tables.get_mut(table).expect("checked above").log;
        fence.check(table, log)?;
        let appended = log.append(&record);
        self.check_poisoned(table);
        appended?;
        self.unsaved_heads.insert(self.names[&id].clone());
        let log_length = self.tables[table].log.length();
        let (written, dropped) = self
            .apply_record(id, record, log_length)
            .inspect_err(|error| self.break_table(id, unapplied(error)))?;
        self.send_drops(&dropped).await;
        let purged = self.node.take_drops();
        self.send_drops(&purged).await;
        self.lift_quarantine(id);
        Ok(written)
    }
}

/// Syncs the logs of `names`, each on a thread of its own when there are
/// several, so a batch across tables waits for the slowest sync, not their sum.
fn sync_logs(
    tables: &mut HashMap<TableName, Entry>,
    names: &HashSet<&str>,
) -> HashMap<TableName, Result<(), ServerError>> {
    let mut logs: Vec<(&TableName, &mut Log)> = tables
        .iter_mut()
        .filter(|(name, _)| names.contains(name.as_str()))
        .map(|(name, entry)| (name, &mut entry.log))
        .collect();
    if let [(name, log)] = logs.as_mut_slice() {
        return HashMap::from([((*name).clone(), log.sync().map_err(ServerError::from))]);
    }
    std::thread::scope(|scope| {
        let syncing: Vec<_> = logs
            .into_iter()
            .map(|(name, log)| {
                (
                    name,
                    scope.spawn(move || log.sync().map_err(ServerError::from)),
                )
            })
            .collect();
        syncing
            .into_iter()
            .map(|(name, sync)| {
                let synced = sync.join().unwrap_or_else(|_| {
                    Err(ServerError::Internal(
                        "syncing the log panicked".to_string(),
                    ))
                });
                (name.clone(), synced)
            })
            .collect()
    })
}
