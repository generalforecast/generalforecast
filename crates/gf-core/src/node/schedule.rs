// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Planning pending requests and due region rebuilds into steps.

use super::*;

/// Adds one table's pending requests to a step: its fit if due, its units and
/// the pages they need, and its queries. The requests stay pending until the
/// whole step is planned.
#[allow(clippy::too_many_arguments)]
pub(super) fn schedule_table(
    placing: &mut Placing,
    id: TableId,
    state: &mut TableState,
    snapshot: &Snapshot,
    planned: Plan,
    keys: Vec<CacheKey>,
    step: &mut Step,
    parts: &mut PlanParts,
) -> Result<(), NodeError> {
    let since = parts.start();
    let mut placed: BTreeMap<PageId, StepPage> = BTreeMap::new();
    let wanted = state.horizon_of(state.pending[0].2.as_deref());
    // A table that summarizes each context fits each unit on its own rows
    // instead: the worker gets no fit and no pages.
    let own = state.spec.summary == SummaryScope::Context;
    let fuse = !own && wanted.is_none() && fuses(state, snapshot, &planned);
    let fits = step.fits.len();
    placing.ledger.fits.push(FitMark {
        table: id,
        fitted: state.fitted,
        horizon: state.horizon,
        fitted_latest: state.fitted_latest,
        fitted_columns: state.fitted_columns.clone(),
        changed: state.changed,
    });
    if state.fitted != Some(state.epoch) {
        state.horizon = wanted;
    }
    if own {
        state.fitted = Some(state.epoch);
    } else if fuse {
        // The unit fits the state on its rows, every row by key, in the call
        // that encodes and decodes it, so the step has no fit and no pages.
        let rows: Vec<RowId> = snapshot.rows().iter().collect();
        let rows = by_key(&state.table, &rows);
        let (features, _) = state.inputs(&rows)?;
        fitted_on(state, &rows, &features);
        state.fitted_latest = rows.iter().filter_map(|&row| state.times.time(row)).max();
    } else if let Some(horizon) = wanted.filter(|_| state.fitted != Some(state.epoch)) {
        // Fit only on rows before the cutoffs, so no query's state saw a later row.
        let before = &state.times.before(horizon) & &snapshot.rows();
        fit_sample(id, state, &before, step)?;
    } else if state.fitted != Some(state.epoch)
        && snapshot.rows().len()
            > state
                .spec
                .effective_fit_rows()
                .min(state.spec.sizes.fused_rows) as u64
    {
        // Past the rows a fit embeds with its state, it computes the state
        // alone, on a sample past the rows its call takes, and each page is
        // embedded once a plan reads it, so a fit never holds every row's
        // representation.
        fit_sample(id, state, &snapshot.rows(), step)?;
    } else if state.fitted != Some(state.epoch) {
        let pages: Vec<PageId> = state
            .index
            .pages()
            .filter(|&page| {
                state
                    .index
                    .rows(page)
                    .is_some_and(|rows| rows.iter().any(|row| snapshot.version(row).is_some()))
            })
            .collect();
        let first = step.pages.len();
        place_pages(placing, id, state, snapshot, &pages, true, step)?;
        let indices: Vec<StepPage> = (first..step.pages.len()).map(StepPage).collect();
        for (&page, &index) in pages.iter().zip(&indices) {
            placed.insert(page, index);
        }
        let rows: Vec<RowId> = indices
            .iter()
            .flat_map(|&index| step.page(index).rows.clone())
            .collect();
        let (features, targets) = state.inputs(&rows)?;
        fitted_on(state, &rows, &features);
        step.fits.push(Fit {
            table: id,
            epoch: state.epoch,
            features,
            targets,
            rows,
            pages: indices,
        });
    }
    if let Some(fit) = step.fits.get(fits) {
        state.fitted_latest = fit
            .rows
            .iter()
            .filter_map(|&row| state.times.time(row))
            .max();
        parts.add("fit", since);
    }

    let width = state.spec.features;
    let first_unit = step.units.len();
    place_units(
        placing,
        id,
        state,
        snapshot,
        planned.units,
        keys,
        &placed,
        fuse,
        Unkept::Compute,
        step,
        parts,
    )?;

    let since = parts.start();
    let mut queries = planned.queries.into_iter();
    for (request, features, _) in &state.pending {
        let count = features.len() / width;
        let units = queries
            .by_ref()
            .take(count)
            .map(|unit| StepUnit(first_unit + unit.0))
            .collect();
        step.queries.push(QueryBatch {
            request: *request,
            table: id,
            epoch: state.epoch,
            features: state.spec.model_values(features),
            units,
        });
    }
    parts.add("assemble", since);
    Ok(())
}

/// Whether a table's step fuses its one unit: the table holds a single-use
/// context not fit this epoch, its plan reads one context of every row, and
/// the context and the queries waiting stay within the rows a fit embeds
/// with its state.
pub(super) fn fuses(state: &TableState, snapshot: &Snapshot, planned: &Plan) -> bool {
    let most = state
        .spec
        .effective_fit_rows()
        .min(state.spec.sizes.fused_rows) as u64;
    let width = state.spec.features.max(1);
    let queries: usize = state
        .pending
        .iter()
        .map(|(_, features, _)| features.len() / width)
        .sum();
    state.single_use
        && state.fitted != Some(state.epoch)
        && queries as u64 <= most
        && matches!(planned.units.as_slice(), [unit]
            if unit.rows.len() <= most && unit.rows == snapshot.rows())
}

/// Records a fit of a table's state on `rows`, every live row, whose model
/// values are `features`: their column summary and moments.
pub(super) fn fitted_on(state: &mut TableState, rows: &[RowId], features: &[f32]) {
    let width = state.spec.model_features();
    let labeled = state.labeled_values(rows, features);
    let (means, deviations) = moments(&labeled, width);
    state.columns = Columns::of(&labeled, width);
    let hidden = state.hidden();
    state.fitted_columns = Some(Fitted {
        rows: rows.iter().filter(|row| !hidden.contains(row.0)).count() as u64,
        means,
        deviations,
    });
    state.changed = 0;
    state.fitted = Some(state.epoch);
}

/// Fits a table's state alone on a seeded sample of at most the effective
/// `fit_rows` of `from`, every row of it if it has no more: its live rows
/// when they outgrow the rows a fit embeds with its state, or those before a
/// request's cutoffs.
///
/// The sample's representations fill no pages: each page is embedded when a
/// plan first reads it. Column sums keep running over every live row, for
/// drift, while the fitted moments come from the sample.
pub(super) fn fit_sample(
    id: TableId,
    state: &mut TableState,
    from: &Context,
    step: &mut Step,
) -> Result<(), NodeError> {
    let sample: Vec<RowId> = smallest(
        FIT_SEED,
        from,
        state.spec.effective_fit_rows(),
        &state.table,
    )
    .iter()
    .collect();
    // The fit reads the sample by key, like a context, so its moments and
    // the model's fit do not depend on the order rows were written in.
    let rows = by_key(&state.table, &sample);
    let (features, targets) = state.inputs(&rows)?;
    let labeled = state.labeled_values(&rows, &features);
    let (means, deviations) = moments(&labeled, state.spec.model_features());
    state.fitted_columns = Some(Fitted {
        rows: (from.bitmap() - state.hidden()).len(),
        means,
        deviations,
    });
    state.changed = 0;
    step.fits.push(Fit {
        table: id,
        epoch: state.epoch,
        features,
        targets,
        rows,
        pages: Vec::new(),
    });
    state.fitted = Some(state.epoch);
    Ok(())
}

/// The context bytes a step may newly encode, and those it has so far.
pub(super) struct Budget {
    limit: u64,
    used: u64,
    slot_bytes: u64,
}

impl Budget {
    /// Whether the step has spent it all.
    fn spent(&self) -> bool {
        self.used >= self.limit
    }

    /// Takes `cost` if it fits what is left, or if the step has nothing yet
    /// (`first`), so a part larger than the budget still runs alone; returns
    /// whether it took it.
    fn take(&mut self, cost: u64, first: bool) -> bool {
        let used = self.used.saturating_add(cost);
        if !first && used > self.limit {
            return false;
        }
        self.used = used;
        true
    }
}

impl Node {
    /// A new step's budget: `step_bytes`, the device tier's size by default,
    /// or no limit without a device tier, since then nothing is kept.
    fn budget(&self) -> Budget {
        let limit = self.step_bytes.unwrap_or(match self.cache.device_bytes() {
            0 => u64::MAX,
            bytes => bytes,
        });
        Budget {
            limit,
            used: 0,
            slot_bytes: self.cache.slot_bytes(),
        }
    }

    /// Plans pending requests into one step, or None if none can run yet.
    ///
    /// Requests whose queries would read no rows are refused instead; see
    /// `take_refused`. Tables are taken in turn from the one after the last
    /// step's first. A table's requests join the step if the contexts they
    /// newly encode fit the remaining budget; the step's first table always
    /// joins, so a table larger than the budget still runs, alone. A table
    /// whose requests would read what a step in flight encodes or fits waits
    /// for that step, since if it fails the worker never holds them.
    ///
    /// A table whose planning fails, as when its rows cannot be read, has
    /// every request it holds refused with that error, see `take_refused`,
    /// and the step is planned again without it, so one table's failure
    /// never fails another's requests nor comes back at every schedule.
    ///
    /// # Errors
    ///
    /// If the cache refuses an operation for a reason other than room while
    /// rebuilding regions, which no request asked for. The requests stay
    /// queued, and nothing of the step is left pinned or admitted.
    pub fn schedule(&mut self) -> Result<Option<Step>, NodeError> {
        self.parts = PlanParts::new(self.parts.timed);
        let since = self.parts.start();
        // A table being restored is judged once its last row is back.
        for (&id, state) in &mut self.tables {
            if !state.pending.is_empty() && state.restoring.is_none() {
                self.refused.extend(state.take_empty(id));
            }
        }
        let mut waiting: Vec<TableId> = self
            .tables
            .range(self.turn..)
            .chain(self.tables.range(..self.turn))
            // A table being restored takes requests once its last row is back.
            .filter(|(_, state)| {
                (!state.pending.is_empty() || !state.forecasts.is_empty())
                    && state.restoring.is_none()
                    && state.compiling.is_none()
            })
            .map(|(&id, _)| id)
            .collect();
        // A table whose fit would read more rows than it may before the
        // worker measures its limits waits for them.
        waiting.retain(|id| {
            let state = &self.tables[id];
            let held = state.unmeasured_fit();
            if held && self.unmeasured.insert(*id) {
                self.untold.push(*id);
            }
            !held
        });
        if self.scheduling == Scheduling::Arrival {
            waiting.sort_by_key(|id| self.tables[id].first_request());
        }
        // Region contexts due a rebuild take a step of their own once no
        // request waits, or once one has waited out its table's
        // `RebuildWait`, so requests keep reading the old contexts meanwhile.
        let now = Instant::now();
        let due = self.oldest_due(now);
        for (id, state) in &mut self.tables {
            if !due.iter().any(|(table, _)| table == id) {
                state.rebuild_waited = 0;
            }
        }
        let waited = due.iter().any(|(id, due)| {
            let state = &self.tables[id];
            state
                .spec
                .regions
                .is_some_and(|spec| spec.wait.over(state.rebuild_waited, *due, now))
        });
        self.parts.add("checks", since);
        if !due.is_empty()
            && (waiting.is_empty() || waited)
            && let Some(step) = self.rebuild_step(now)?
        {
            for state in self.tables.values_mut() {
                state.rebuild_waited = 0;
            }
            return Ok(Some(step));
        }
        let mut step = Step::default();
        let mut scheduled = Vec::new();
        let mut ledger = Ledger::default();
        let mut planning = None;
        let planned = self.plan_step(
            &waiting,
            &mut step,
            &mut scheduled,
            &mut ledger,
            now,
            &mut planning,
        );
        if let Err(error) = planned {
            let dropped = self.unwind(ledger, step);
            self.drops.extend(dropped);
            let Some(state) = planning.and_then(|id| self.tables.get_mut(&id)) else {
                return Err(error);
            };
            let refused = state.take_requests();
            self.refused
                .extend(refused.into_iter().map(|request| (request, error)));
            return self.schedule();
        }
        let Some(&first) = scheduled.first() else {
            return Ok(None);
        };
        let since = self.parts.start();
        for &id in &scheduled {
            if let Some(state) = self.tables.get_mut(&id) {
                state.gave = state.given.is_some();
                state.schedule_pending();
                state.promote();
                state.generation += 1;
            }
        }
        self.flight.hold(&step);
        self.turn = TableId(first.0 + 1);
        for (id, _) in &due {
            if let Some(state) = self.tables.get_mut(id) {
                state.rebuild_waited += 1;
            }
        }
        self.parts.add("assemble", since);
        Ok(Some(step))
    }

    /// Each table with a region context due a rebuild, among those of current
    /// states no step rebuilds yet, and when its longest due fell due.
    pub(super) fn oldest_due(&mut self, now: Instant) -> Vec<(TableId, Instant)> {
        let (cache, flight, rebuilt) = (&self.cache, &self.flight, &self.rebuilt_regions);
        self.tables
            .iter_mut()
            .filter_map(|(&id, state)| {
                state.age_regions(now);
                due_rebuilds(id, state, cache, flight, rebuilt, now)
                    .into_iter()
                    .map(|(_, due)| due)
                    .min()
                    .map(|due| (id, due))
            })
            .collect()
    }

    /// A step that only builds again region contexts due a rebuild, around
    /// their centers, as many as the step budget allows; None if none can be.
    ///
    /// # Errors
    ///
    /// As `schedule`.
    pub(super) fn rebuild_step(&mut self, now: Instant) -> Result<Option<Step>, NodeError> {
        let mut step = Step::default();
        let mut ledger = Ledger::default();
        if let Err(error) = self.plan_rebuilds(&mut step, &mut ledger, now) {
            let dropped = self.unwind(ledger, step);
            self.drops.extend(dropped);
            return Err(error);
        }
        let since = self.parts.start();
        // Contexts the cache held already replace their regions' at once.
        let held: Vec<CacheKey> = self
            .rebuilding
            .keys()
            .filter(|key| {
                self.cache.contains(key) && !step.units.iter().any(|unit| unit.key == **key)
            })
            .filter(|key| !self.flight.encoding.contains(key))
            .copied()
            .collect();
        for key in held {
            let dropped = self.finish_rebuild(key, now);
            self.drops.extend(dropped);
        }
        if step.units.is_empty() {
            self.parts.add("assemble", since);
            return Ok(None);
        }
        self.flight.hold(&step);
        self.parts.add("assemble", since);
        Ok(Some(step))
    }

    /// Adds the due region rebuilds of each table to `step` while the budget
    /// allows, recording what it takes in `ledger`.
    fn plan_rebuilds(
        &mut self,
        step: &mut Step,
        ledger: &mut Ledger,
        now: Instant,
    ) -> Result<(), NodeError> {
        let mut budget = self.budget();
        let slot_bytes = budget.slot_bytes;
        let tables: Vec<TableId> = self
            .tables
            .range(self.turn..)
            .chain(self.tables.range(..self.turn))
            .map(|(&id, _)| id)
            .collect();
        for id in tables {
            if (!step.units.is_empty() && budget.spent()) || self.flight.fitting.contains(&id) {
                continue;
            }
            let state = self.tables.get_mut(&id).expect("listed above");
            let epoch = state.epoch;
            if state.restoring.is_some() || state.refit_for(None) {
                continue;
            }
            let since = self.parts.start();
            let due: Vec<(usize, Instant)> = due_rebuilds(
                id,
                state,
                &self.cache,
                &self.flight,
                &self.rebuilt_regions,
                now,
            );
            let Some(regions) = state.regions.as_ref().filter(|_| !due.is_empty()) else {
                self.parts.add("regions", since);
                continue;
            };
            let generation = regions.generation;
            let due: Vec<(usize, Instant, u64)> = due
                .into_iter()
                .filter_map(|(region, due)| {
                    Some((region, due, regions.built[region].as_ref()?.changes))
                })
                .collect();
            let centers: Vec<Vec<f32>> = due
                .iter()
                .map(|&(region, _, _)| regions.centers[region].clone())
                .collect();
            self.parts.add("regions", since);
            let since = self.parts.start();
            let snapshot = state.served();
            self.parts.add("snapshot", since);
            let since = self.parts.start();
            state.prepare_cores(epoch);
            let mut planned =
                state.plan_around(&snapshot, &centers, &[], &[], &vec![None; centers.len()])?;
            self.parts.add("retrieve", since);
            let since = self.parts.start();
            let mut kept = Vec::new();
            let mut keys = Vec::new();
            // Regions join while the budget lasts; the step's first always does.
            for (&(region, due, seen), unit) in due.iter().zip(&planned.queries) {
                let unit = &planned.units[unit.0];
                let key = state.spec.entry_keys(id, epoch).context(unit.key);
                if self.flight.encoding.contains(&key) {
                    continue;
                }
                let encodes = !self.rebuilding.contains_key(&key)
                    && !keys.contains(&key)
                    && !self.cache.contains(&key);
                let cost = if encodes {
                    state.spec.sizes.bytes(unit.rows.len() as usize, slot_bytes)
                } else {
                    0
                };
                if !budget.take(cost, step.units.is_empty() && keys.is_empty()) {
                    break;
                }
                self.rebuilding.entry(key).or_default().push(Rebuilding {
                    table: id,
                    region,
                    epoch,
                    generation,
                    rows: unit.rows.clone(),
                    seen,
                    due,
                });
                self.rebuilt_regions.insert((id, region));
                ledger.rebuilds.push((key, id, region));
                if encodes {
                    keys.push(key);
                    kept.push(unit.clone());
                }
            }
            planned.units = kept;
            self.parts.add("checks", since);
            let mut placing = Placing {
                cache: &mut self.cache,
                flight: &self.flight,
                ledger: &mut *ledger,
            };
            // A rebuild the cache cannot keep would be computed for nothing,
            // so it waits for room, its region's built context serving on.
            let skipped = place_units(
                &mut placing,
                id,
                state,
                &snapshot,
                planned.units,
                keys,
                &BTreeMap::new(),
                false,
                Unkept::Skip,
                step,
                &mut self.parts,
            )?;
            for key in skipped {
                for rebuilt in self.rebuilding.remove(&key).unwrap_or_default() {
                    self.rebuilt_regions
                        .remove(&(rebuilt.table, rebuilt.region));
                }
                ledger.rebuilds.retain(|(claimed, _, _)| *claimed != key);
            }
        }
        Ok(())
    }

    /// Makes the contexts rebuilt as `key` their regions' built ones, and
    /// drops the entries they replace; returns the drops.
    pub(super) fn finish_rebuild(&mut self, key: CacheKey, now: Instant) -> Vec<Movement> {
        let mut replaced = Vec::new();
        for rebuilt in self.rebuilding.remove(&key).unwrap_or_default() {
            self.rebuilt_regions
                .remove(&(rebuilt.table, rebuilt.region));
            let Some(state) = self.tables.get_mut(&rebuilt.table) else {
                continue;
            };
            // Regions drawn again since the rebuild was planned are other regions.
            let Some(regions) = state.regions.as_mut().filter(|regions| {
                regions.epoch == rebuilt.epoch && regions.generation == rebuilt.generation
            }) else {
                continue;
            };
            let rows = rebuilt.rows.len();
            let old = regions.built[rebuilt.region].clone();
            let times = state.times.times_of(&rebuilt.rows);
            regions.rebuilt(
                rebuilt.region,
                (rebuilt.rows, key.rows, times),
                rebuilt.seen,
                state
                    .spec
                    .regions
                    .map_or(Rebuild::Never, |spec| spec.rebuild),
                now,
            );
            if let Some(old) = old.filter(|old| old.key != key.rows) {
                replaced.push(CacheKey {
                    rows: old.key,
                    ..key
                });
            }
            let stats = &mut state.region_stats;
            stats.rebuilds += 1;
            stats.rebuilt_rows += rows;
            stats.overdue += now.saturating_duration_since(rebuilt.due);
        }
        // An entry another region still reads stays.
        replaced.retain(|old| {
            !self.tables.get(&old.table).is_some_and(|state| {
                state.regions.as_ref().is_some_and(|regions| {
                    regions
                        .built
                        .iter()
                        .flatten()
                        .any(|built| built.key == old.rows)
                })
            })
        });
        replaced
            .into_iter()
            .flat_map(|old| self.drop_or_doom(&old))
            .collect()
    }

    /// Adds each waiting table's requests to the step while the budget
    /// allows; `planning` names the table an error is the planning of.
    pub(super) fn plan_step(
        &mut self,
        waiting: &[TableId],
        step: &mut Step,
        scheduled: &mut Vec<TableId>,
        ledger: &mut Ledger,
        now: Instant,
        planning: &mut Option<TableId>,
    ) -> Result<(), NodeError> {
        let mut budget = self.budget();
        let slot_bytes = budget.slot_bytes;
        for &id in waiting {
            // The table an error below is the planning of.
            *planning = Some(id);
            // Planning costs time, so tables wait without it once the budget is
            // spent, or while a step in flight fits them.
            if (!scheduled.is_empty() && budget.spent()) || self.flight.fitting.contains(&id) {
                continue;
            }
            let Some(state) = self.tables.get_mut(&id) else {
                continue;
            };
            if state.spec.series.is_some() {
                let since = self.parts.start();
                let planned = self.plan_series(id, &mut budget, step, scheduled, ledger);
                self.parts.add("series", since);
                planned?;
                continue;
            }
            // One state serves a step: requests whose cutoffs need a state fit
            // on other rows wait until these are served. Units fit on their
            // own rows share no state, so serve every cutoff together.
            let since = self.parts.start();
            let wanted = state.horizon_of(state.pending[0].2.as_deref());
            let mixed = state.spec.summary == SummaryScope::Table
                && state
                    .pending
                    .iter()
                    .any(|(_, _, cutoffs)| state.horizon_of(cutoffs.as_deref()) != wanted);
            if mixed {
                let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut state.pending)
                    .into_iter()
                    .partition(|(_, _, cutoffs)| state.horizon_of(cutoffs.as_deref()) == wanted);
                state.pending = now;
                state.later.extend(later);
                state.generation += 1;
            }
            // A refit the policy asks for, or one for other rows, starts a new
            // epoch, but only once the table joins the step, so a table left
            // waiting keeps its cache.
            let refit = state.refit_for(wanted);
            let epoch = Epoch(state.epoch.0 + u64::from(refit));
            self.parts.add("checks", since);
            let since = self.parts.start();
            let snapshot = state.served();
            self.parts.add("snapshot", since);
            let planned = loop {
                // A table the budget left waiting keeps its plan while nothing changes.
                let planned = match state.deferred.take() {
                    Some((generation, planned)) if generation == state.generation => planned,
                    _ if state.given.is_some() => {
                        let since = self.parts.start();
                        let given = state.given.as_ref().expect("matched above");
                        let planned = plan_given(&snapshot, &state.given_rows(given));
                        self.parts.add("retrieve", since);
                        planned?
                    }
                    _ => {
                        let keys = state.spec.entry_keys(id, epoch);
                        let cached = |rows| self.flight.holds(&self.cache, &keys.context(rows));
                        state.plan_pending(epoch, &snapshot, &cached, now, &mut self.parts)?
                    }
                };
                // Requests a context of unlabeled rows alone would answer are
                // refused, and the rest planned again.
                let blind = state.blind(&planned);
                if blind.is_empty() {
                    break Some(planned);
                }
                state.drop_pending(&blind);
                state.generation += 1;
                self.refused.extend(
                    blind
                        .into_iter()
                        .map(|request| (request, NodeError::EmptyContext(id))),
                );
                if state.pending.is_empty() {
                    state.promote();
                    if state.pending.is_empty() {
                        break None;
                    }
                }
            };
            let Some(planned) = planned else {
                continue;
            };
            let since = self.parts.start();
            if let (PastCap::Warn, Some(cap)) = (state.spec.past_cap, state.spec.context_rows)
                && !state.past_cap
                && planned.units.iter().any(|unit| unit.rows.len() > cap.get())
            {
                state.past_cap = true;
                self.past_cap.push(id);
            }
            let entry_keys = state.spec.entry_keys(id, epoch);
            let keys: Vec<CacheKey> = planned
                .units
                .iter()
                .map(|unit| entry_keys.context(unit.key))
                .collect();
            // Contexts a step in flight encodes, or, for those not cached
            // yet, pages it embeds, are read only once it completes.
            let waits = keys.iter().any(|key| self.flight.encoding.contains(key))
                || (keys.iter().any(|key| !self.cache.contains(key))
                    && self
                        .flight
                        .encoding
                        .iter()
                        .any(|key| key.table == id && key.kind == Kind::Rows));
            self.parts.add("checks", since);
            if waits {
                state.deferred = Some((state.generation, planned));
                continue;
            }
            // A context the step cannot read as it is, cached or not, is
            // computed again.
            let cost: u64 = planned
                .units
                .iter()
                .zip(&keys)
                .filter(|(_, key)| !self.flight.reuses(&self.cache, key))
                .map(|(unit, _)| state.spec.sizes.bytes(unit.rows.len() as usize, slot_bytes))
                .sum();
            if !budget.take(cost, scheduled.is_empty()) {
                state.deferred = Some((state.generation, planned));
                continue;
            }
            let state = self.tables.get_mut(&id).expect("planned above");
            let counts = std::mem::take(&mut state.planned_counts);
            state.region_stats.count(&counts);
            if refit {
                step.movements.extend(self.advance_epoch(id)?);
            }
            let state = self.tables.get_mut(&id).expect("planned above");
            let mut placing = Placing {
                cache: &mut self.cache,
                flight: &self.flight,
                ledger: &mut *ledger,
            };
            schedule_table(
                &mut placing,
                id,
                state,
                &snapshot,
                planned,
                keys,
                step,
                &mut self.parts,
            )?;
            scheduled.push(id);
        }
        Ok(())
    }

    /// Adds a series table's forecast requests to the step while the budget
    /// allows: its groups as units to encode, reuse or forecast whole, and a
    /// forecast per series.
    pub(super) fn plan_series(
        &mut self,
        id: TableId,
        budget: &mut Budget,
        step: &mut Step,
        scheduled: &mut Vec<TableId>,
        ledger: &mut Ledger,
    ) -> Result<(), NodeError> {
        let slot_bytes = budget.slot_bytes;
        let state = self.tables.get_mut(&id).expect("waiting tables exist");
        let spec = state
            .spec
            .series
            .clone()
            .expect("a series table has a spec");
        let index = state.series.as_ref().expect("a series table has an index");
        let snapshot = state.table.snapshot();
        let requests: Vec<SeriesRequest> = state
            .forecasts
            .iter()
            .map(|(_, request)| request.clone())
            .collect();
        let features = state.spec.features;
        let planned = match forecasting::plan(&spec, index, &snapshot, features, &requests) {
            Ok(planned) => planned,
            Err(_) => {
                // A request planning refuses, such as one with a cutoff per
                // series missing, is refused alone; the rest plan again.
                let mut kept = Vec::with_capacity(state.forecasts.len());
                let mut refused = Vec::new();
                for (request, asked) in state.forecasts.drain(..) {
                    let alone = std::slice::from_ref(&asked);
                    match forecasting::plan(&spec, index, &snapshot, features, alone) {
                        Ok(_) => kept.push((request, asked)),
                        Err(error) => refused.push((request, NodeError::Forecast(error))),
                    }
                }
                state.forecasts = kept;
                state.generation += refused.len() as u64;
                self.refused.extend(refused);
                if state.forecasts.is_empty() {
                    return Ok(());
                }
                let requests: Vec<SeriesRequest> = state
                    .forecasts
                    .iter()
                    .map(|(_, asked)| asked.clone())
                    .collect();
                forecasting::plan(&spec, index, &snapshot, features, &requests)?
            }
        };
        let entry_keys = state.spec.entry_keys(id, state.epoch);
        let keys: Vec<CacheKey> = planned
            .units
            .iter()
            .map(|unit| entry_keys.context(unit.key))
            .collect();
        // Contexts a step in flight encodes are read only once it completes.
        if keys.iter().any(|key| self.flight.encoding.contains(key)) {
            return Ok(());
        }
        let features = state.spec.features;
        let unit_bytes = |unit: &forecasting::PlannedUnit| {
            state.spec.sizes.series_bytes(
                &spec,
                features,
                unit.members.len(),
                unit.steps,
                slot_bytes,
            )
        };
        // A unit that may extend its group's entry holds the source's patches
        // and the new ones, which can pad past a fresh encoding's; it is
        // charged the larger, since placement may fall back to either.
        let extended_bytes = |unit: &forecasting::PlannedUnit| {
            let prev = state.recent.get(&unit.group).filter(|prev| {
                spec.extendable
                    && self.cache.contains(&prev.key)
                    && !self.flight.encoding.contains(&prev.key)
            })?;
            let (_, patches) = extension_size(&spec, unit, prev)?;
            let sizes = &state.spec.sizes;
            Some(extension_bytes(
                &spec, sizes, features, unit, patches, slot_bytes,
            ))
        };
        let cost: u64 = planned
            .units
            .iter()
            .zip(&keys)
            .filter(|(unit, key)| unit.whole || !self.flight.reuses(&self.cache, key))
            .map(|(unit, _)| {
                let fresh = unit_bytes(unit);
                match extended_bytes(unit) {
                    Some(bytes) if !unit.whole => bytes.max(fresh),
                    _ => fresh,
                }
            })
            .sum();
        if !budget.take(cost, scheduled.is_empty()) {
            return Ok(());
        }
        let state = self.tables.get_mut(&id).expect("planned above");
        let mut placing = Placing {
            cache: &mut self.cache,
            flight: &self.flight,
            ledger,
        };
        let first_unit = step.series_units.len();
        for (unit, key) in planned.units.iter().zip(keys) {
            let grid = forecasting::grid(&spec, unit, features, |rows| state.rows.rows(rows))?;
            let (mut placed, mut source, mut grown) = (Placed::Transient, None, 0);
            if !unit.whole {
                let bytes = state.spec.sizes.series_bytes(
                    &spec,
                    features,
                    unit.members.len(),
                    unit.steps,
                    slot_bytes,
                );
                // A kept unit whose entry is gone is nothing to extend.
                let recent = match state.recent.get(&unit.group).copied() {
                    Some(prev) if !placing.cache.contains(&prev.key) => {
                        state.recent.remove(&unit.group);
                        None
                    }
                    Some(prev) if placing.flight.encoding.contains(&prev.key) => None,
                    recent => recent,
                };
                let extended = match recent {
                    Some(prev) if spec.extendable && !placing.cache.contains(&key) => extend(
                        &mut placing,
                        &state.table.snapshot(),
                        &spec,
                        &state.spec.sizes,
                        features,
                        unit,
                        key,
                        prev,
                        slot_bytes,
                        &mut step.movements,
                    )?,
                    _ => None,
                };
                let mut patches = None;
                placed = match extended {
                    Some(extension) => {
                        source = Some(extension.source);
                        grown = extension.grown;
                        patches = Some(extension.patches);
                        extension.placed
                    }
                    None => place(&mut placing, key, bytes, unit.rows(), &mut step.movements)?,
                };
                if placed.action() == Action::Encode {
                    patches = Some(state.spec.sizes.series_patches(&spec, unit.steps));
                }
                if let Some(patches) = patches {
                    let recent = Recent {
                        cutoff: unit.cutoff,
                        steps: unit.steps,
                        patches,
                        key,
                    };
                    state.writing.insert(key, (unit.group, recent));
                }
            }
            step.series_units.push(SeriesUnit {
                table: id,
                key,
                members: unit.members.clone(),
                cutoff: unit.cutoff,
                grid,
                placed,
                source,
                grown,
            });
        }
        for ((request, _), forecasts) in state.forecasts.iter().zip(planned.forecasts) {
            for forecast in forecasts {
                let unit = &planned.units[forecast.unit];
                let future = forecasting::future(&spec, unit, &forecast, features, |rows| {
                    state.rows.rows(rows)
                })?;
                step.forecasts.push(Forecast {
                    request: *request,
                    table: id,
                    epoch: state.epoch,
                    unit: StepUnit(first_unit + forecast.unit),
                    target: forecast.target,
                    horizon: forecast.horizon,
                    future,
                });
            }
        }
        scheduled.push(id);
        Ok(())
    }

    /// Forgets every cached entry and fit, as when the worker starts again
    /// empty; call it with no step in flight. Each table starts a new epoch,
    /// so a step that completes late cannot touch the entries that follow, and
    /// is fit again on its next step.
    pub fn forget_cache(&mut self) {
        for (&id, state) in &mut self.tables {
            let _ = self
                .cache
                .drop_epochs_before(state.spec.model, id, Epoch(u64::MAX));
            state.epoch.0 += 1;
            state.fitted = None;
            state.recent.clear();
            state.writing.clear();
            state.scheduled = Scheduled::default();
            state.compiling = None;
        }
        self.compiled.clear();
        self.flight.doomed.clear();
        self.drops.clear();
        self.flight.moving.clear();
        self.flight.encoding.clear();
        self.flight.fitting.clear();
    }
}

/// The regions of a table's current state due a rebuild by `now`, and since
/// when each has been: those no step rebuilds yet whose built context the
/// cache still holds, since a query reading a region whose context is gone
/// builds it anew itself.
fn due_rebuilds(
    id: TableId,
    state: &mut TableState,
    cache: &Cache,
    flight: &Flight,
    rebuilt: &HashSet<(TableId, usize)>,
    now: Instant,
) -> Vec<(usize, Instant)> {
    let (Some(spec), Some(regions)) = (state.spec.regions, state.regions.as_mut()) else {
        return Vec::new();
    };
    if regions.epoch != state.epoch || state.fitted != Some(state.epoch) {
        return Vec::new();
    }
    let keys = state.spec.entry_keys(id, state.epoch);
    let due = regions.due(spec.rebuild, now);
    due.into_iter()
        .filter(|&(region, _)| !rebuilt.contains(&(id, region)))
        .filter(|&(region, _)| {
            regions.built[region]
                .as_ref()
                .is_some_and(|built| flight.holds(cache, &keys.context(built.key)))
        })
        .collect()
}
