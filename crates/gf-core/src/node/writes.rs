// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Writes to a table's rows, and the cached entries they make stale.

use super::*;

/// Some of a batch's rows, picked by index.
pub(super) struct Picked {
    keys: Vec<u128>,
    features: Vec<f32>,
    targets: Targets,
    times: Option<Vec<Option<i64>>>,
}

impl Picked {
    pub(super) fn new(
        keys: &[u128],
        features: &[f32],
        width: usize,
        targets: &Targets,
        times: Option<&[Option<i64>]>,
        indices: &[usize],
    ) -> Self {
        Self {
            keys: indices.iter().map(|&index| keys[index]).collect(),
            features: indices
                .iter()
                .flat_map(|&index| &features[index * width..(index + 1) * width])
                .copied()
                .collect(),
            targets: targets.pick(indices),
            times: times.map(|times| indices.iter().map(|&index| times[index]).collect()),
        }
    }

    /// Inserts the rows, which the whole batch's checks passed.
    pub(super) fn insert(&self, node: &mut Node, table: TableId) -> Result<Vec<RowId>, NodeError> {
        let times = self.times.as_deref();
        node.apply_insert(table, &self.keys, &self.features, &self.targets, times)
    }

    /// Updates the rows, which the whole batch's checks passed.
    pub(super) fn update(&self, node: &mut Node, table: TableId) -> Result<Vec<RowId>, NodeError> {
        let times = self.times.as_deref();
        node.apply_update(table, &self.keys, &self.features, &self.targets, times)
    }
}

/// Checks that `times`, if given, has one entry per key.
pub(super) fn check_times(keys: &[u128], times: Option<&[Option<i64>]>) -> Result<(), NodeError> {
    match times {
        Some(times) if times.len() != keys.len() => Err(NodeError::Count {
            what: Counted::Times,
            expected: keys.len(),
            got: times.len(),
        }),
        _ => Ok(()),
    }
}

/// Checks that row-major `features` holds one row of `width` per key.
pub(super) fn check_rows(keys: usize, features: &[f32], width: usize) -> Result<(), NodeError> {
    if features.len() == keys * width {
        return Ok(());
    }
    if keys > 0 && features.len().is_multiple_of(keys) {
        return Err(NodeError::Width {
            expected: width,
            got: features.len() / keys,
        });
    }
    Err(NodeError::Count {
        what: Counted::Rows,
        expected: keys,
        got: features.len() / width.max(1),
    })
}

impl Node {
    /// Drops every cached context of a table that holds one of `rows`; entries
    /// a running step reads are dropped when it completes. A page's entry
    /// stays, with the rows marked stale, as a source for the page's next
    /// version, which copies the rest of its rows instead of embedding them.
    pub(super) fn purge(&mut self, table: TableId, rows: &[RowId]) {
        let changed: Context = rows.iter().copied().collect();
        let epoch = self.tables.get(&table).map(|state| state.epoch);
        for key in self.cache.holding(table, &changed) {
            // A page's entry is a source while it holds a row no write touched.
            if key.kind == Kind::Rows
                && Some(key.epoch) == epoch
                && !self.flight.doomed.contains(&key)
            {
                let fresh = match (self.cache.rows(&key), self.cache.stale(&key)) {
                    (Some(held), Some(stale)) => held
                        .iter()
                        .any(|row| !stale.contains(row) && !changed.contains(row)),
                    _ => false,
                };
                if fresh {
                    let _ = self.cache.mark_stale(&key, &changed);
                    continue;
                }
            }
            let dropped = self.drop_or_doom(&key);
            self.drops.extend(dropped);
        }
    }

    /// Drops a stale entry now, or once the steps reading it complete.
    pub(super) fn drop_or_doom(&mut self, key: &CacheKey) -> Option<Movement> {
        if self.cache.pinned(key) {
            self.flight.doomed.insert(*key);
            return None;
        }
        self.flight.doomed.remove(key);
        self.cache.remove(key)
    }

    /// Makes a table hold exactly these rows, as a snapshot says: deletes the
    /// others, updates rows whose features, target or time differ, and inserts
    /// new ones, then renumbers the rows as `renumber` does, so every copy built
    /// from the snapshot holds the same ids and pages.
    ///
    /// Drops of the entries the renumbering forgets are left for `take_drops`.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn replace_rows(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        let state = self.checked_batch(table, keys, features, &targets, times, true)?;
        let width = state.spec.features;
        let wanted: HashSet<u128> = keys.iter().copied().collect();
        let (mut updates, mut inserts) = (Vec::new(), Vec::new());
        for (index, &key) in keys.iter().enumerate() {
            let values = &features[index * width..(index + 1) * width];
            match state.table.row(RowKey(key)) {
                None => inserts.push(index),
                Some(row) => {
                    let stored = state.rows.rows(&[row])?.remove(0);
                    let unchanged = stored
                        .features
                        .iter()
                        .zip(values)
                        .all(|(old, new)| old.to_bits() == new.to_bits())
                        && stored.target == targets.bits()[index]
                        && times.is_none_or(|times| state.times.time(row) == times[index]);
                    if !unchanged {
                        updates.push(index);
                    }
                }
            }
        }
        if inserts.len() as u64 > state.ids_left() {
            return Err(NodeError::Table(TableError::Full));
        }
        // In key order: the order of deletes decides which pages merge.
        let mut deletes: Vec<u128> = state
            .table
            .keys()
            .map(|(key, _)| key.0)
            .filter(|key| !wanted.contains(key))
            .collect();
        deletes.sort_unstable();
        let updates = Picked::new(keys, features, width, &targets, times, &updates);
        let inserts = Picked::new(keys, features, width, &targets, times, &inserts);
        self.apply_delete(table, &deletes)?;
        updates.update(self, table)?;
        inserts.insert(self, table)?;
        self.renumber(table)
    }

    /// Renumbers a table's rows densely in key order and places them again,
    /// as copies of it built from a snapshot of its log do, so that from here
    /// on its row ids, pages and samples match theirs. Its cached entries hold
    /// the old ids, so they are dropped, or once read by the steps in flight;
    /// the drops are left for `take_drops`. Its fitted state is kept.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn renumber(&mut self, table: TableId) -> Result<(), NodeError> {
        self.state(table)?.renumber()?;
        for key in self.cache.keys_of(table) {
            let dropped = self.drop_or_doom(&key);
            self.drops.extend(dropped);
        }
        Ok(())
    }

    /// Adds rows, replacing the rows of keys the table already holds, at `times`
    /// if given; returns the rows' ids in the order of `keys`.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn upsert_at(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        self.check_upsert(table, keys, features, &targets, times)?;
        let state = &self.tables[&table];
        let (known, new): (Vec<usize>, Vec<usize>) =
            (0..keys.len()).partition(|&index| state.table.row(RowKey(keys[index])).is_some());
        let width = state.spec.features;
        let updates = Picked::new(keys, features, width, &targets, times, &known);
        let inserts = Picked::new(keys, features, width, &targets, times, &new);
        updates.update(self, table)?;
        inserts.insert(self, table)?;
        let state = &self.tables[&table];
        Ok(keys
            .iter()
            .map(|&key| state.table.row(RowKey(key)).expect("just written"))
            .collect())
    }

    /// The keys among `keys` that a table holds, in order.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn present(&self, table: TableId, keys: &[u128]) -> Result<Vec<u128>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        Ok(keys
            .iter()
            .copied()
            .filter(|&key| state.table.row(RowKey(key)).is_some())
            .collect())
    }

    /// The table a batch writes to, once the batch's shape, keys, targets
    /// and times are checked against it; with `replacing`, a time a series
    /// already holds may be written over, as an update or upsert does.
    pub(super) fn checked_batch(
        &self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
        replacing: bool,
    ) -> Result<&TableState, NodeError> {
        check_times(keys, times)?;
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.check(keys, features, targets)?;
        state.check_series(keys, times, replacing)?;
        Ok(state)
    }

    /// Checks an insert without applying it, so it can be logged before it is applied.
    ///
    /// # Errors
    ///
    /// The error `insert` would return.
    pub fn check_insert(
        &self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        let state = self.checked_batch(table, keys, features, targets, times, false)?;
        if let Some(&key) = keys
            .iter()
            .find(|&&key| state.table.row(RowKey(key)).is_some())
        {
            return Err(NodeError::Table(TableError::DuplicateKey(RowKey(key))));
        }
        if keys.len() as u64 > state.ids_left() {
            return Err(NodeError::Table(TableError::Full));
        }
        Ok(())
    }

    /// Checks an update without applying it.
    ///
    /// # Errors
    ///
    /// The error `update` would return.
    pub fn check_update(
        &self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        let state = self.checked_batch(table, keys, features, targets, times, true)?;
        match keys
            .iter()
            .find(|&&key| state.table.row(RowKey(key)).is_none())
        {
            Some(&key) => Err(NodeError::UnknownKey(RowKey(key))),
            None => Ok(()),
        }
    }

    /// Checks an upsert without applying it.
    ///
    /// # Errors
    ///
    /// The error `upsert_at` would return.
    pub fn check_upsert(
        &self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<(), NodeError> {
        let state = self.checked_batch(table, keys, features, targets, times, true)?;
        let new = keys
            .iter()
            .filter(|&&key| state.table.row(RowKey(key)).is_none())
            .count();
        if new as u64 > state.ids_left() {
            return Err(NodeError::Table(TableError::Full));
        }
        Ok(())
    }

    /// Checks a delete without applying it.
    ///
    /// # Errors
    ///
    /// The error `delete` would return.
    pub fn check_delete(&self, table: TableId, keys: &[u128]) -> Result<(), NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let mut seen = BTreeSet::new();
        for &key in keys {
            if !seen.insert(key) {
                return Err(NodeError::DuplicateKey(RowKey(key)));
            }
            if state.table.row(RowKey(key)).is_none() {
                return Err(NodeError::UnknownKey(RowKey(key)));
            }
        }
        Ok(())
    }

    /// Adds labeled rows without times. The first batch sets the table's retrieval scaling.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing is inserted then.
    pub fn insert(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
    ) -> Result<Vec<RowId>, NodeError> {
        self.insert_at(table, keys, features, targets, None)
    }

    /// Adds labeled rows, each at its entry of `times` if given. A row without
    /// a time is before every cutoff.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing is inserted then.
    pub fn insert_at(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        self.check_insert(table, keys, features, &targets, times)?;
        self.apply_insert(table, keys, features, &targets, times)
    }

    /// Adds rows that `check_insert` and `check_times` passed.
    pub(super) fn apply_insert(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        // Only once the write is checked, so every copy applying the same
        // writes renumbers at the same one.
        if self.tables.get(&table).is_some_and(TableState::sparse) {
            self.renumber(table)?;
        }
        let state = self.state(table)?;
        let width = state.spec.features;
        let labeled = state.labeled(targets);
        // A table of series places no rows in pages, so it fits no scaling.
        let paged = state.spec.series.is_none();
        if paged && state.standardizer.is_none() {
            // Labeled rows set the scaling, unless the batch has none.
            let at: Vec<usize> = (0..keys.len()).filter(|&row| labeled[row]).collect();
            // Labeled rows it was fit on, so the first labeled ones fit it again.
            state.scaled_rows = at.len() as u64;
            let standardizer = if at.is_empty() {
                Standardizer::fit(features, targets, &state.spec)
            } else {
                let fit = Picked::new(keys, features, width, targets, None, &at);
                Standardizer::fit(&fit.features, &fit.targets, &state.spec)
            };
            state.standardizer = Some(standardizer);
        }
        let mut rows = Vec::with_capacity(keys.len());
        let mut placed = Vec::with_capacity(keys.len());
        state.rows.push(features, &targets.bits(), keys)?;
        for (offset, &key) in keys.iter().enumerate() {
            let values = &features[offset * width..(offset + 1) * width];
            let row = state.table.apply(Change::Insert(RowKey(key)))?;
            let time = times.and_then(|times| times[offset]);
            if paged {
                placed.push((row, state.spec.bucket(time), state.vector(values)?));
            }
            state.times.set(row, time);
            if let (Some(index), Some(time)) = (&mut state.series, time) {
                index
                    .insert(row, RowKey(key), time, targets.observed(offset))
                    .expect("series rows were checked before the write");
            }
            if state.serves(labeled[offset]) {
                let print = state.print_of(row);
                state.print(time, print, true);
            }
            if !labeled[offset] {
                state.unlabeled.insert(row.0);
            }
            if state.serves(labeled[offset])
                && let Some(ranking) = &mut state.ranking
            {
                ranking.insert(row, key);
            }
            if labeled[offset] {
                state.columns.add(values, 1.0);
            }
            rows.push(row);
        }
        if let (Some(regions), Some(spec)) = (&mut state.regions, state.spec.regions) {
            let now = Instant::now();
            for ((_, _, vector), &labeled) in placed.iter().zip(&labeled) {
                if labeled || state.spec.unlabeled_context {
                    regions.change(vector, spec.rebuild, now);
                }
            }
        }
        let scaled = Scaled {
            rows: &state.rows,
            standardizer: state.standardizer.as_ref(),
        };
        state.index.insert_many(&placed, &scaled)?;
        state.changed += labeled
            .iter()
            .filter(|&&labeled| state.serves(labeled))
            .count() as u64;
        self.rescale_if_due(table)?;
        Ok(rows)
    }

    /// Fits a table's retrieval scaling again once the labeled rows that set
    /// it are half the table's labeled rows or fewer, which depends only on
    /// the writes, so every copy does the same. Once it was fit on `fit_rows`
    /// rows it stays, so a large table is not cut into pages anew each time it
    /// doubles.
    pub(super) fn rescale_if_due(&mut self, table: TableId) -> Result<(), NodeError> {
        let state = self.state(table)?;
        let labeled = state.table.len() - state.unlabeled.len();
        if state.spec.series.is_none()
            && state.spec.retrieval == 0
            && labeled >= 2 * state.scaled_rows.max(1)
            && state.scaled_rows < state.spec.fit_rows.get() as u64
            && state
                .spec
                .scale_rows
                .is_none_or(|rows| state.scaled_rows < rows.get())
        {
            state.rescale()?;
            // Pages are cut anew, so the old pages' representations would
            // only hold slots until evicted.
            for key in self.cache.keys_of(table) {
                if key.kind == Kind::Rows {
                    let dropped = self.drop_or_doom(&key);
                    self.drops.extend(dropped);
                }
            }
        }
        Ok(())
    }

    /// Changes existing rows: new features and targets, and a new version.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn update(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
    ) -> Result<Vec<RowId>, NodeError> {
        self.update_at(table, keys, features, targets, None)
    }

    /// Changes existing rows, and their times to the entries of `times` if given.
    ///
    /// # Errors
    ///
    /// If the shapes, keys or targets are wrong. Nothing changes then.
    pub fn update_at(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        self.check_update(table, keys, features, &targets, times)?;
        self.apply_update(table, keys, features, &targets, times)
    }

    /// Changes rows that `check_update` and `check_times` passed.
    pub(super) fn apply_update(
        &mut self,
        table: TableId,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
        times: Option<&[Option<i64>]>,
    ) -> Result<Vec<RowId>, NodeError> {
        let state = self.state(table)?;
        let width = state.spec.features;
        let labeled = state.labeled(targets);
        let mut rows = Vec::with_capacity(keys.len());
        let mut changed = 0;
        for (offset, &key) in keys.iter().enumerate() {
            let values = &features[offset * width..(offset + 1) * width];
            let before = state.table.row(RowKey(key)).map(|row| state.print_of(row));
            let row = state.table.apply(Change::Update(RowKey(key)))?;
            let was = !state.unlabeled.contains(row.0);
            let old = state.row(row)?;
            let old_time = state.times.time(row);
            let time = times.map_or(old_time, |times| times[offset]);
            // Times' fingerprints sum the rows plans read.
            if let Some(before) = before.filter(|_| state.serves(was)) {
                state.print(old_time, before, false);
            }
            if state.serves(labeled[offset]) {
                let after = state.print_of(row);
                state.print(time, after, true);
            }
            let moved = old
                .iter()
                .zip(values)
                .any(|(old, new)| old.to_bits() != new.to_bits())
                || state.spec.bucket(time) != state.spec.bucket(old_time);
            // A row whose features and window are unchanged keeps its page; a
            // table of series has none.
            let paged = state.spec.series.is_none();
            if moved && paged {
                let scaled = Scaled {
                    rows: &state.rows,
                    standardizer: state.standardizer.as_ref(),
                };
                state.index.remove(row, &scaled)?;
            }
            let target = targets.bits()[offset];
            state.rows.set(&[row], values, &[target], None)?;
            if let Some(index) = &mut state.series {
                index.observe(RowKey(key), targets.observed(offset));
            }
            if moved && paged {
                let bucket = state.spec.bucket(time);
                let vector = state.vector(values)?;
                let scaled = Scaled {
                    rows: &state.rows,
                    standardizer: state.standardizer.as_ref(),
                };
                state.index.insert(row, bucket, &vector, &scaled)?;
            }
            let is = labeled[offset];
            if is {
                state.unlabeled.remove(row.0);
            } else {
                state.unlabeled.insert(row.0);
            }
            if let Some(ranking) = &mut state.ranking {
                match (
                    state.spec.unlabeled_context || was,
                    state.spec.unlabeled_context || is,
                ) {
                    (false, true) => ranking.insert(row, key),
                    (true, false) => ranking.remove(row, key),
                    _ => {}
                }
            }
            if was {
                state.columns.add(&old, -1.0);
            }
            if is {
                state.columns.add(values, 1.0);
            }
            state.times.set(row, time);
            // A row a plan reads before or after counts toward a refit and its
            // region's rebuild: a label is when an unlabeled row enters
            // contexts, or one update to it for a model that reads it unlabeled.
            if state.serves(was) || state.serves(is) {
                state.note_change(values)?;
                changed += 1;
            }
            rows.push(row);
        }
        state.changed += changed;
        self.purge(table, &rows);
        // Labels that arrive may make enough labeled rows to scale again.
        self.rescale_if_due(table)?;
        Ok(rows)
    }

    /// Sets rows' targets, keeping their features and times, such as labels
    /// that arrive after their rows.
    ///
    /// # Errors
    ///
    /// As `update`. Nothing changes then.
    pub fn label(
        &mut self,
        table: TableId,
        keys: &[u128],
        targets: Targets,
    ) -> Result<Vec<RowId>, NodeError> {
        let features = self.features(table, keys)?;
        self.update_at(table, keys, &features, targets, None)
    }

    /// Rows' features by key, row-major.
    ///
    /// # Errors
    ///
    /// If the table or a key does not exist.
    pub fn features(&self, table: TableId, keys: &[u128]) -> Result<Vec<f32>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let mut features = Vec::with_capacity(keys.len() * state.spec.features);
        for &key in keys {
            let row = state
                .table
                .row(RowKey(key))
                .ok_or(NodeError::UnknownKey(RowKey(key)))?;
            features.extend(state.row(row)?);
        }
        Ok(features)
    }

    /// Deletes rows.
    ///
    /// # Errors
    ///
    /// If a key does not exist or repeats. Nothing changes then.
    pub fn delete(&mut self, table: TableId, keys: &[u128]) -> Result<Vec<RowId>, NodeError> {
        self.check_delete(table, keys)?;
        self.apply_delete(table, keys)
    }

    /// Deletes rows that `check_delete` passed.
    pub(super) fn apply_delete(
        &mut self,
        table: TableId,
        keys: &[u128],
    ) -> Result<Vec<RowId>, NodeError> {
        let state = self.state(table)?;
        let mut rows = Vec::with_capacity(keys.len());
        let mut changed = 0;
        for &key in keys {
            let before = state.table.row(RowKey(key)).map(|row| state.print_of(row));
            let row = state.table.apply(Change::Delete(RowKey(key)))?;
            if state.spec.series.is_none() {
                let scaled = Scaled {
                    rows: &state.rows,
                    standardizer: state.standardizer.as_ref(),
                };
                state.index.remove(row, &scaled)?;
            }
            let time = state.times.time(row);
            state.times.remove(row);
            if let Some(index) = &mut state.series {
                index.remove(RowKey(key));
            }
            let labeled = !state.unlabeled.remove(row.0);
            if let Some(print) = before.filter(|_| state.serves(labeled)) {
                state.print(time, print, false);
            }
            if let Some(ranking) = &mut state.ranking {
                ranking.remove(row, key);
            }
            let values = state.row(row)?;
            if labeled {
                state.columns.add(&values, -1.0);
            }
            if state.serves(labeled) {
                state.note_change(&values)?;
                changed += 1;
            }
            rows.push(row);
        }
        state.changed += changed;
        self.purge(table, &rows);
        Ok(rows)
    }

    /// Deletes every row, in key order, as `delete` deletes them.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn truncate(&mut self, table: TableId) -> Result<Vec<RowId>, NodeError> {
        let mut keys: Vec<u128> = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .table
            .keys()
            .map(|(key, _)| key.0)
            .collect();
        keys.sort_unstable();
        self.delete(table, &keys)
    }

    /// Refits a table, as asked: renumbers its rows, fits its retrieval
    /// scaling on them, places every row again, and starts a new epoch.
    ///
    /// Only these refits renumber and rescale, not those a refit policy
    /// starts, since they are what table logs record: copies that replay a log
    /// place rows alike.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn refit(&mut self, table: TableId) -> Result<Vec<Movement>, NodeError> {
        self.state(table)?.renumber()?;
        self.advance_epoch(table)
    }

    /// Starts a new epoch for a table after its state is to be refit.
    ///
    /// Returns drops of its old entries that nothing reads; the rest are dropped
    /// when the steps reading them complete.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn advance_epoch(&mut self, table: TableId) -> Result<Vec<Movement>, NodeError> {
        let state = self.state(table)?;
        state.epoch.0 += 1;
        // The old epoch's kept units are dropped, so none is left to extend.
        state.recent.clear();
        state.writing.clear();
        let (model, epoch) = (state.spec.model, state.epoch);
        self.advanced.push(table);
        Ok(self.cache.drop_epochs_before(model, table, epoch))
    }
}
