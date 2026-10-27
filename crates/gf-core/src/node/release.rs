// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Completing or failing a step, and releasing what it held.

use super::*;

/// The tables a step fits a state for: by a fit, or by a fused unit.
pub(super) fn fitted_tables(step: &Step) -> impl Iterator<Item = TableId> + '_ {
    let fused = step
        .units
        .iter()
        .filter(|unit| unit.placed == Placed::Fuse)
        .map(|unit| unit.table);
    step.fits.iter().map(|fit| fit.table).chain(fused)
}

/// The entries a step encodes and keeps: its contexts and its pages.
pub(super) fn encoded(step: &Step) -> impl Iterator<Item = CacheKey> + '_ {
    let units = step.units.iter().map(|unit| (unit.key, &unit.placed));
    let pages = step.pages.iter().map(|page| (page.key, &page.placed));
    let series = step
        .series_units
        .iter()
        .map(|unit| (unit.key, &unit.placed));
    units
        .chain(pages)
        .chain(series)
        .filter(|(_, placed)| matches!(placed, Placed::Encode(_) | Placed::Extend { .. }))
        .map(|(key, _)| key)
}

/// Which tables' parts of a step the worker failed to run.
pub(super) enum Failed<'a> {
    None,
    Tables(&'a HashSet<TableId>),
    All,
}

impl Failed<'_> {
    pub(super) fn includes(&self, table: TableId) -> bool {
        match self {
            Failed::None => false,
            Failed::Tables(tables) => tables.contains(&table),
            Failed::All => true,
        }
    }
}

impl Node {
    /// Gives back everything a step that failed to plan took, as its ledger
    /// records it: its pins, the entries it admitted, which hold nothing, and
    /// those it moved between tiers, which the executor never moved, so later
    /// steps encode them again; the fits it marked, so they are planned again;
    /// and the region rebuilds it claimed. Returns the drops for the executor,
    /// the step's own among them, since it never sees the step.
    pub(super) fn unwind(&mut self, ledger: Ledger, step: Step) -> Vec<Movement> {
        let Ledger {
            pins,
            admitted,
            fits,
            rebuilds,
        } = ledger;
        for key in &pins {
            let _ = self.cache.unpin(key);
        }
        let (mut dropped, moved): (Vec<Movement>, Vec<Movement>) = step
            .movements
            .into_iter()
            .partition(|movement| movement.to.is_none());
        // Stale entries only this step still read go now.
        let stale: Vec<CacheKey> = pins
            .iter()
            .filter(|key| self.flight.doomed.contains(key))
            .copied()
            .collect();
        for key in admitted
            .iter()
            .chain(moved.iter().map(|movement| &movement.key))
            .chain(&stale)
        {
            dropped.extend(self.drop_or_doom(key));
        }
        for mark in fits {
            if let Some(state) = self.tables.get_mut(&mark.table) {
                state.fitted = mark.fitted;
                state.horizon = mark.horizon;
                state.fitted_latest = mark.fitted_latest;
            }
        }
        for (key, table, region) in rebuilds {
            if let Some(claimed) = self.rebuilding.get_mut(&key) {
                claimed.retain(|rebuilt| (rebuilt.table, rebuilt.region) != (table, region));
                if claimed.is_empty() {
                    self.rebuilding.remove(&key);
                }
            }
            self.rebuilt_regions.remove(&(table, region));
        }
        for unit in &step.series_units {
            if let Some(state) = self.tables.get_mut(&unit.table) {
                state.writing.remove(&unit.key);
            }
        }
        dropped
    }

    /// Releases a finished step's entries; returns drops of entries from ended epochs.
    pub fn complete(&mut self, step: &Step) -> Vec<Movement> {
        self.release(step, &Failed::None)
    }

    /// Completes a step that the worker ran for every table except `failed`.
    /// The failed tables' parts are undone as `fail` undoes a whole step,
    /// except that their tier moves stand, since the worker made them before
    /// any table's part ran. Returns drops as `fail` does.
    pub fn complete_except(&mut self, step: &Step, failed: &HashSet<TableId>) -> Vec<Movement> {
        if failed.is_empty() {
            return self.complete(step);
        }
        self.unfit(step, |table| failed.contains(&table));
        self.release(step, &Failed::Tables(failed))
    }

    /// Completes a step as `complete_except` does, except that the tables
    /// `deferred` names wait for a program the worker compiles beside its
    /// steps, by its key. Their parts are undone as a failed part's, but
    /// their query requests queue again, and steps leave them out until
    /// `compiled` names the key, unless it already did. A step answers its
    /// forecasts whatever the worker does with their table's part.
    pub fn complete_deferring(
        &mut self,
        step: &Step,
        failed: &HashSet<TableId>,
        deferred: &[(TableId, String)],
    ) -> Vec<Movement> {
        let mut undone = failed.clone();
        undone.extend(deferred.iter().map(|(table, _)| *table));
        let requests = step_requests(step);
        let mut requeued = Vec::with_capacity(deferred.len());
        for (table, _) in deferred {
            if let Some(state) = self.tables.get_mut(table) {
                requeued.push(state.scheduled.take(|request| requests.contains(&request)));
            }
        }
        let dropped = self.complete_except(step, &undone);
        for ((table, key), requests) in deferred.iter().zip(requeued) {
            let compiled = self.compiled.contains(key);
            if let Some(state) = self.tables.get_mut(table) {
                state.queue_again(requests);
                if !compiled {
                    state.compiling = Some(key.clone());
                }
            }
        }
        dropped
    }

    /// Lets steps take again the tables waiting for the programs `keys` name.
    pub fn compiled(&mut self, keys: Vec<String>) {
        for state in self.tables.values_mut() {
            if state
                .compiling
                .as_ref()
                .is_some_and(|key| keys.contains(key))
            {
                state.compiling = None;
            }
        }
        self.compiled.extend(keys);
    }

    /// Undoes a step the worker failed to run. The entries it was to encode,
    /// which the worker may hold only in part, are forgotten, and so are the
    /// entries it moved between tiers, which the worker may not have moved, so
    /// a later step encodes them again; the rest are released as `complete`
    /// releases them, and fits it held are planned again. Returns drops for the
    /// forgotten entries and those from ended epochs.
    pub fn fail(&mut self, step: &Step) -> Vec<Movement> {
        self.unfit(step, |_| true);
        self.release(step, &Failed::All)
    }

    /// Plans again the fits a step held for the tables `failed` names.
    pub(super) fn unfit(&mut self, step: &Step, failed: impl Fn(TableId) -> bool) {
        let fits = step.fits.iter().map(|fit| (fit.table, fit.epoch));
        let fused = step
            .units
            .iter()
            .filter(|unit| unit.placed == Placed::Fuse)
            .map(|unit| (unit.table, unit.key.epoch));
        for (table, epoch) in fits.chain(fused).filter(|(table, _)| failed(*table)) {
            if let Some(state) = self.tables.get_mut(&table)
                && state.fitted == Some(epoch)
            {
                state.fitted = None;
            }
        }
    }

    /// Releases a step's pins. An entry that is stale, because the step failed
    /// to encode it, a change purged it, or its epoch ended, is dropped once no
    /// step reads it any more.
    pub(super) fn release(&mut self, step: &Step, failed: &Failed) -> Vec<Movement> {
        self.flight.release(step);
        let kept = step
            .units
            .iter()
            .map(|unit| (unit.table, unit.key, unit.placed.action()))
            .chain(
                step.pages
                    .iter()
                    .map(|page| (page.table, page.key, page.placed.action())),
            )
            .chain(
                step.series_units
                    .iter()
                    .map(|unit| (unit.table, unit.key, unit.placed.action())),
            );
        let mut dropped = Vec::new();
        // A page's sources were pinned for the step too. One a kept page
        // copied from has served its purpose; one a transient page or a
        // failed part copied from stays for the page's next version.
        for page in &step.pages {
            for (source, _) in &page.copies {
                let _ = self.cache.unpin(source);
                let current = self.tables.get(&page.table).map(|state| state.epoch);
                let served = page.placed.action() == Action::Encode && !failed.includes(page.table);
                if served
                    || self.flight.doomed.contains(source)
                    || current.is_none_or(|epoch| source.epoch < epoch)
                {
                    dropped.extend(self.drop_or_doom(source));
                }
            }
        }
        // An extension's source was pinned for the step too, and is dropped
        // like any entry the step read if it was doomed or its epoch ended; a
        // kept unit the step wrote is now one a later unit of its group may
        // extend.
        for unit in &step.series_units {
            if let Some(source) = unit.source {
                let _ = self.cache.unpin(&source);
                let current = self.tables.get(&unit.table).map(|state| state.epoch);
                if self.flight.doomed.contains(&source)
                    || current.is_none_or(|epoch| source.epoch < epoch)
                {
                    dropped.extend(self.drop_or_doom(&source));
                }
            }
            let Some(state) = self.tables.get_mut(&unit.table) else {
                continue;
            };
            let Some((group, recent)) = state.writing.remove(&unit.key) else {
                continue;
            };
            let written = !failed.includes(unit.table)
                && !self.flight.doomed.contains(&unit.key)
                && unit.key.epoch >= state.epoch;
            // A group extends forward, so an earlier cutoff written after a
            // later one, in this step or an earlier, leaves the later as the
            // group's latest while its entry lasts.
            let latest = match state.recent.get(&group) {
                Some(prev) => {
                    prev.cutoff <= recent.cutoff
                        || !self.cache.contains(&prev.key)
                        || self.flight.doomed.contains(&prev.key)
                        || prev.key.epoch < state.epoch
                }
                None => true,
            };
            if written && latest {
                state.recent.insert(group, recent);
                let held = state.recent.len();
                if held > RECENT_PRUNE && held >= 2 * state.recent_pruned {
                    let cache = &self.cache;
                    state.recent.retain(|_, recent| cache.contains(&recent.key));
                    state.recent_pruned = state.recent.len();
                }
            }
        }
        for (table, key, action) in kept {
            if matches!(action, Action::Transient | Action::Fuse) {
                continue;
            }
            let _ = self.cache.unpin(&key);
            let current = self.tables.get(&table).map(|state| state.epoch);
            // A removed table's entries are stale too, and so is an entry a
            // failed step was to write, whether encoded or extended.
            let written = matches!(action, Action::Encode | Action::Extend);
            let stale = (failed.includes(table) && written)
                || self.flight.doomed.contains(&key)
                || current.is_none_or(|epoch| key.epoch < epoch);
            if stale {
                dropped.extend(self.drop_or_doom(&key));
            }
        }
        let now = Instant::now();
        for unit in &step.units {
            if self.rebuilding.contains_key(&unit.key) {
                if !failed.includes(unit.table) && self.cache.contains(&unit.key) {
                    dropped.extend(self.finish_rebuild(unit.key, now));
                } else {
                    for rebuilt in self.rebuilding.remove(&unit.key).unwrap_or_default() {
                        self.rebuilt_regions
                            .remove(&(rebuilt.table, rebuilt.region));
                    }
                }
            }
        }
        let requests = step_requests(step);
        for table in step_tables(step) {
            if let Some(state) = self.tables.get_mut(&table) {
                state.scheduled.take(|request| requests.contains(&request));
            }
        }
        if matches!(failed, Failed::All) {
            // The worker may not have moved these, so they are encoded again.
            for movement in step
                .movements
                .iter()
                .filter(|movement| movement.to.is_some())
            {
                dropped.extend(self.drop_or_doom(&movement.key));
            }
        }
        dropped
    }
}

/// The requests a step answers.
fn step_requests(step: &Step) -> HashSet<RequestId> {
    step.queries
        .iter()
        .map(|batch| batch.request)
        .chain(step.forecasts.iter().map(|forecast| forecast.request))
        .collect()
}

/// The tables whose requests a step answers.
fn step_tables(step: &Step) -> BTreeSet<TableId> {
    step.queries
        .iter()
        .map(|batch| batch.table)
        .chain(step.forecasts.iter().map(|forecast| forecast.table))
        .collect()
}
