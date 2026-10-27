// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A table's state between steps: what plans read, and how writes keep it current.

use super::*;

impl TableState {
    /// Whether the table holds more dead row ids than live rows, and enough
    /// to be worth renumbering.
    pub(super) fn sparse(&self) -> bool {
        let live = self.table.len();
        self.table.ids_used() - live > live.max(SPARSE_ROWS)
    }

    /// How many more rows can be inserted, counting the dead ids a sparse
    /// table frees when it is renumbered before its next insert.
    pub(super) fn ids_left(&self) -> u64 {
        let used = if self.sparse() {
            self.table.len()
        } else {
            self.table.ids_used()
        };
        self.row_ids.saturating_sub(used)
    }

    /// One row's features.
    pub(super) fn row(&self, row: RowId) -> Result<Vec<f32>, NodeError> {
        Ok(self.rows.features(row)?)
    }

    /// The live rows plans read: all of them, or for a model that reads no
    /// unlabeled context rows, the labeled ones.
    pub(super) fn served(&self) -> Snapshot {
        self.table.snapshot().without(self.hidden())
    }

    /// Whether each target is known. A series' rows are observations, not
    /// labels, so they are known even when missing.
    pub(super) fn labeled(&self, targets: &Targets) -> Vec<bool> {
        if self.spec.series.is_some() {
            vec![true; targets.len()]
        } else {
            targets.labeled()
        }
    }

    /// Drops pending requests, and from a pending given batch their contexts
    /// with them, so the others keep theirs.
    pub(super) fn drop_pending(&mut self, requests: &[RequestId]) {
        let width = self.spec.features;
        if let Some(given) = &mut self.given {
            let mut offset = 0;
            let mut kept = Vec::with_capacity(given.len());
            for (request, queries, _) in &self.pending {
                let count = queries.len() / width;
                if !requests.contains(request) {
                    kept.extend(given[offset..offset + count].iter().cloned());
                }
                offset += count;
            }
            *given = kept;
        }
        self.pending
            .retain(|(request, _, _)| !requests.contains(request));
    }

    /// Takes every request the table holds out of its queues, for a failure
    /// of the table's own to refuse them all.
    pub(super) fn take_requests(&mut self) -> Vec<RequestId> {
        let mut requests: Vec<RequestId> = self.pending.drain(..).map(|(id, _, _)| id).collect();
        requests.extend(self.later.drain(..).map(|(id, _, _)| id));
        requests.extend(self.givens.drain(..).map(|(id, _, _)| id));
        requests.extend(self.forecasts.drain(..).map(|(id, _)| id));
        self.given = None;
        self.deferred = None;
        self.generation += 1;
        requests
    }

    /// The pending requests a plan gives a context of unlabeled rows alone,
    /// which says nothing of any target.
    pub(super) fn blind(&self, planned: &Plan) -> Vec<RequestId> {
        if !self.spec.unlabeled_context || self.unlabeled.is_empty() {
            return Vec::new();
        }
        let width = self.spec.features;
        let mut offset = 0;
        let mut found = Vec::new();
        for (request, queries, _) in &self.pending {
            let count = queries.len() / width;
            let reads = &planned.queries[offset..offset + count];
            if reads.iter().any(|unit| {
                planned.units[unit.0]
                    .rows
                    .bitmap()
                    .is_subset(&self.unlabeled)
            }) {
                found.push(*request);
            }
            offset += count;
        }
        found
    }

    /// Whether the table's next fit, a first one or a refit its policy or
    /// cutoffs ask for, would read more rows than it may before the worker
    /// measures how many its fits take.
    pub(super) fn unmeasured_fit(&self) -> bool {
        let wanted = self.horizon_of(
            self.pending
                .first()
                .and_then(|(_, _, cutoffs)| cutoffs.as_deref()),
        );
        self.spec.sizes.fit_bound != usize::MAX
            && self.spec.summary == SummaryScope::Table
            && self.spec.series.is_none()
            && (self.fitted != Some(self.epoch) || self.refit_for(wanted))
            && self.table.len() > self.spec.sizes.fit_bound as u64
    }

    /// Rows plans leave out: unlabeled ones, unless the model reads them.
    pub(super) fn hidden(&self) -> &RoaringBitmap {
        static NONE: std::sync::LazyLock<RoaringBitmap> =
            std::sync::LazyLock::new(RoaringBitmap::new);
        if self.spec.unlabeled_context {
            &NONE
        } else {
            &self.unlabeled
        }
    }

    /// Whether plans read a row of this label.
    pub(super) fn serves(&self, labeled: bool) -> bool {
        labeled || self.spec.unlabeled_context
    }

    /// The labeled rows of `rows`' row-major model `features`, which drift reads.
    pub(super) fn labeled_values(&self, rows: &[RowId], features: &[f32]) -> Vec<f32> {
        features
            .chunks(self.spec.model_features().max(1))
            .zip(rows)
            .filter(|(_, row)| !self.unlabeled.contains(row.0))
            .flat_map(|(values, _)| values)
            .copied()
            .collect()
    }

    /// The live rows whose target is known, which scaling and drift read.
    pub(super) fn labeled_rows(&self) -> Context {
        Context::from_bitmap(self.table.snapshot().rows().bitmap() - &self.unlabeled)
    }

    /// The strategy the table plans with now: an adaptive one resolved from
    /// its live rows, the model's cap and whether its rows have times, any
    /// other as given.
    pub(super) fn resolved(&self) -> Strategy {
        self.spec.strategy.resolve(
            self.table.len(),
            self.spec.context_rows,
            self.times.has_timed(),
        )
    }

    /// As `resolved`, and makes the state fit it: when the resolution has
    /// changed since the last plan, as an adaptive table's does when it grows
    /// past the cap or shrinks back, the cores and regions drawn for the old
    /// one are dropped, so the next plan draws them anew, at the cost a
    /// `refit` pays for its region draw. Cached encodings stay, keyed by the
    /// rows they hold.
    pub(super) fn strategy(&mut self) -> Strategy {
        let strategy = self.resolved();
        if strategy != self.resolved {
            self.resolved = strategy;
            self.cores.1.clear();
            self.regions = None;
        }
        strategy
    }

    /// Sets the table's strategy, as given, and drops what was drawn for
    /// the old one: the live rows are ranked anew by the new core's seed,
    /// the cores and regions are dropped.
    pub(super) fn set_strategy(&mut self, strategy: Strategy) {
        self.spec.strategy = strategy;
        let served = self.served().rows();
        self.ranking = strategy
            .seed()
            .map(|seed| Ranking::new(seed, &served, &self.table));
        self.resolved = self.resolved();
        self.cores.1.clear();
        self.regions = None;
    }

    /// Sets what a plan for the state of `epoch` draws cores with: a
    /// regression table's target bins, from its live rows, and no diverse
    /// cores of an earlier state.
    pub(super) fn prepare_cores(&mut self, epoch: Epoch) {
        let Strategy::Retrieved {
            core: Some(core), ..
        } = self.strategy()
        else {
            return;
        };
        if self.cores.0 != epoch {
            self.cores = (epoch, Cores::default());
        }
        let binned = matches!(core.kind, CoreKind::Stratified | CoreKind::Balanced)
            && self.spec.task == Task::Values;
        if binned && self.edges.as_ref().is_none_or(|(at, _)| *at != epoch) {
            let mut values: Vec<f32> = self
                .served()
                .rows()
                .iter()
                .filter_map(|row| self.rows.target(row).map(f32::from_bits))
                .filter(|value| value.is_finite())
                .collect();
            values.sort_by(f32::total_cmp);
            let edges = (1..TARGET_BINS)
                .filter_map(|bin| values.get(bin * values.len() / TARGET_BINS).copied())
                .collect();
            self.edges = Some((epoch, edges));
        }
    }

    /// Whether serving requests that need a state fit on the rows before
    /// `wanted` refits: once the table is fit, if its policy asks for a refit
    /// or the state is fit on other rows.
    ///
    /// A table that summarizes each context never does: each unit is fit on
    /// its own rows and keyed by them, so units outlive drift and serve every
    /// cutoff whose window holds the same rows.
    pub(super) fn refit_for(&self, wanted: Option<i64>) -> bool {
        self.spec.summary == SummaryScope::Table
            && self.fitted == Some(self.epoch)
            && (self.refit_due() || self.stale_for(wanted))
    }

    /// Refuses query rows of the wrong width, or cutoffs other than one a query.
    pub(super) fn check_queries(
        &self,
        queries: &[f32],
        cutoffs: Option<&[i64]>,
    ) -> Result<(), NodeError> {
        let width = self.spec.features;
        if queries.is_empty() {
            return Err(NodeError::NoQueries);
        }
        if !queries.len().is_multiple_of(width) {
            return Err(NodeError::Width {
                expected: width,
                got: queries.len(),
            });
        }
        if let Some(cutoffs) = cutoffs
            && cutoffs.len() * width != queries.len()
        {
            return Err(NodeError::Count {
                what: Counted::Cutoffs,
                expected: queries.len() / width,
                got: cutoffs.len(),
            });
        }
        Ok(())
    }

    /// Counts a changed row, with these features, toward its region's rebuild.
    pub(super) fn note_change(&mut self, features: &[f32]) -> Result<(), NodeError> {
        if let Some(spec) = self.spec.regions
            && self.regions.is_some()
        {
            let vector = self.vector(features)?;
            let regions = self.regions.as_mut().expect("checked above");
            regions.change(&vector, spec.rebuild, Instant::now());
        }
        Ok(())
    }

    /// The rows' strata, by class or by target bin.
    pub(super) fn labels(&self) -> Labels<'_> {
        Labels {
            rows: &self.rows,
            task: self.spec.task,
            edges: self.edges.as_ref().map_or(&[], |(_, edges)| edges),
        }
    }

    /// The table's rows in its retrieval space, as its index reads them.
    pub(super) fn scaled(&self) -> Scaled<'_> {
        Scaled {
            rows: &self.rows,
            standardizer: self.standardizer.as_ref(),
        }
    }

    /// Whether the table's refit policy asks for a new fit.
    pub(super) fn refit_due(&self) -> bool {
        let Some(fitted) = &self.fitted_columns else {
            return false;
        };
        let policy = self.spec.refit;
        let changed = policy
            .changed
            .is_some_and(|fraction| self.changed as f64 >= fraction * fitted.rows.max(1) as f64);
        let shifted = policy.shift.is_some_and(|limit| {
            (0..self.spec.model_features()).any(|column| {
                let scale = if fitted.deviations[column] > 0.0 {
                    fitted.deviations[column]
                } else {
                    1.0
                };
                (self.columns.mean(column) - fitted.means[column]).abs() > limit * scale
            })
        });
        changed || shifted
    }

    /// The rows a request's state must be fit on so no query sees a row from
    /// after its cutoff: those before its earliest cutoff, named by the first
    /// row time at or after it, so cutoffs with the same rows before them
    /// share a state; or every row, for a request without cutoffs or one
    /// whose cutoffs come after every row.
    pub(super) fn horizon_of(&self, cutoffs: Option<&[i64]>) -> Option<i64> {
        let cutoff = cutoffs?.iter().min()?;
        self.times.first_outside(*cutoff, self.hidden())
    }

    /// Whether requests needing a state fit on the rows before `wanted` need
    /// a new fit. A state fit on any row after their cutoffs would leak it,
    /// so it always refits. One fit on fewer, earlier rows leaks nothing, so
    /// the table's `CutoffRefit` decides: `Each` refits, so each cutoff reads
    /// the state its rows give; `Changed` keeps it until rows after it reach
    /// the policy's share of the rows it was fit on; `Never` keeps it.
    pub(super) fn stale_for(&self, wanted: Option<i64>) -> bool {
        // A state fit on a row from at or after any pending cutoff would leak it.
        let earliest = self
            .pending
            .iter()
            .filter_map(|(_, _, cutoffs)| cutoffs.as_deref()?.iter().min().copied())
            .min();
        if let (Some(latest), Some(cutoff)) = (self.fitted_latest, earliest)
            && latest >= cutoff
        {
            return true;
        }
        if self.horizon == wanted {
            return false;
        }
        match (self.horizon, wanted, self.spec.refit.cutoffs) {
            // Fit on rows from at or after the cutoff.
            (Some(fitted), Some(wanted), _) if fitted > wanted => true,
            (_, _, CutoffRefit::Never) => false,
            (_, _, CutoffRefit::Each) => true,
            (fitted, wanted, CutoffRefit::Changed) => {
                // A state fit on every row holds those up to its latest time.
                let fitted = fitted.unwrap_or_else(|| {
                    self.fitted_latest
                        .map_or(i64::MIN, |latest| latest.saturating_add(1))
                });
                let share = self.spec.refit.changed.unwrap_or(0.0);
                let between = self.times.between(fitted, wanted.unwrap_or(i64::MAX));
                let since = (between.bitmap() - self.hidden()).len();
                let rows = self.fitted_columns.as_ref().map_or(0, |fitted| fitted.rows);
                since as f64 >= share * rows.max(1) as f64
            }
        }
    }

    /// Removes requests with a query that would read no rows: one whose cutoff
    /// precedes every row, or any once the table has been emptied.
    pub(super) fn take_empty(&mut self, table: TableId) -> Vec<(RequestId, NodeError)> {
        let mut refused = Vec::new();
        let width = self.spec.features;
        // Requests promoted in place of refused ones are checked in turn.
        while !self.pending.is_empty() {
            // A given context whose rows have all gone, or are all unlabeled
            // for a model that reads no such rows, reads nothing.
            let live = self.given.as_ref().map(|_| self.served().rows());
            let given = self.given.take();
            let rows = given.as_ref().map(|given| self.given_rows(given));
            let (mut kept, mut kept_given, mut offset) = (Vec::new(), Vec::new(), 0);
            for (request, queries, cutoffs) in std::mem::take(&mut self.pending) {
                let count = queries.len() / width;
                let contexts = given.as_ref().map(|given| &given[offset..offset + count]);
                let gone = rows
                    .as_ref()
                    .zip(live.as_ref())
                    .is_some_and(|(rows, live)| {
                        rows[offset..offset + count]
                            .iter()
                            .any(|context| (context & live).is_empty())
                    });
                offset += count;
                let error = if gone {
                    Some(NodeError::Given)
                } else {
                    self.empty(table, cutoffs.as_deref())
                };
                match error {
                    Some(error) => refused.push((request, error)),
                    None => {
                        kept_given.extend(contexts.into_iter().flatten().cloned());
                        kept.push((request, queries, cutoffs));
                    }
                }
            }
            self.pending = kept;
            self.given = given.map(|_| kept_given);
            if !self.pending.is_empty() {
                break;
            }
            self.promote();
        }
        self.generation += refused.len() as u64;
        refused
    }

    /// The rows of given contexts, those of their keys the table holds.
    pub(super) fn given_rows(&self, given: &[Given]) -> Vec<Context> {
        given
            .iter()
            .map(|keys| keys.iter().filter_map(|&key| self.table.row(key)).collect())
            .collect()
    }

    /// Moves the pending requests to the steps in flight, each given
    /// request with its contexts, and drops the forecasts, which a step
    /// answers whatever the worker does with its part.
    pub(super) fn schedule_pending(&mut self) {
        let width = self.spec.features;
        let mut given = self.given.take().map(Vec::into_iter);
        for (request, queries, cutoffs) in std::mem::take(&mut self.pending) {
            match &mut given {
                Some(contexts) => {
                    let count = queries.len() / width;
                    let contexts = contexts.by_ref().take(count).collect();
                    self.scheduled.given.push((request, queries, contexts));
                }
                None => self.scheduled.queries.push((request, queries, cutoffs)),
            }
        }
        self.forecasts.clear();
    }

    /// Queues again requests of a step whose part the worker deferred,
    /// ahead of those that came after them.
    pub(super) fn queue_again(&mut self, requests: Scheduled) {
        let Scheduled { mut queries, given } = requests;
        match &mut self.given {
            // A pending given batch: given requests join it ahead of its
            // own, and the others wait behind it, as requests that come
            // meanwhile do.
            Some(batch) => {
                self.later.splice(0..0, queries);
                let mut pending = Vec::with_capacity(given.len() + self.pending.len());
                let mut contexts = Vec::new();
                for (request, queries, given) in given {
                    pending.push((request, queries, None));
                    contexts.extend(given);
                }
                contexts.append(batch);
                *batch = contexts;
                pending.append(&mut self.pending);
                self.pending = pending;
            }
            None => {
                queries.append(&mut self.pending);
                self.pending = queries;
                for request in given.into_iter().rev() {
                    self.givens.push_front(request);
                }
            }
        }
        self.generation += 1;
        self.promote();
    }

    /// Once no request is pending, makes pending the queued given requests,
    /// as one batch, or the others that wait. The two take turns, so neither
    /// waits more than one step behind the other.
    pub(super) fn promote(&mut self) {
        if !self.pending.is_empty() {
            return;
        }
        self.given = None;
        if self.givens.is_empty() || (self.gave && !self.later.is_empty()) {
            self.pending = std::mem::take(&mut self.later);
            return;
        }
        let mut given = Vec::new();
        for (request, queries, contexts) in self.givens.drain(..) {
            self.pending.push((request, queries, None));
            given.extend(contexts);
        }
        self.given = Some(given);
    }

    /// Why a request with these cutoffs would read no rows, if it would.
    pub(super) fn empty(&self, table: TableId, cutoffs: Option<&[i64]>) -> Option<NodeError> {
        // A context needs a labeled row, even for a model that reads unlabeled ones.
        let hidden = &self.unlabeled;
        if self.table.len() == hidden.len() {
            return Some(NodeError::EmptyTable(table));
        }
        // The times of the labeled rows.
        let (untimed, timed) = (
            self.times.has_untimed_outside(hidden),
            self.times.first_outside(i64::MIN, hidden),
        );
        // Rows without a time are before every cutoff.
        let earliest = (!untimed).then_some(timed).flatten();
        let before = |first: i64| {
            cutoffs.is_some_and(|cutoffs| cutoffs.iter().any(|&cutoff| cutoff <= first))
        };
        if earliest.is_some_and(before) {
            return Some(NodeError::EmptyContext(table));
        }
        // A span of time holds timed rows alone, so it is empty with none of
        // them before the cutoff.
        let span = matches!(
            self.resolved(),
            Strategy::Retrieved {
                recency: Some(Recency::Span(_)),
                ..
            }
        );
        match timed {
            _ if !span => None,
            None => Some(NodeError::Untimed(table)),
            Some(timed) if before(timed) => Some(NodeError::EmptyContext(table)),
            Some(_) => None,
        }
    }

    /// Adds a row's fingerprint at `time`, or takes it away.
    pub(super) fn print(&mut self, time: Option<i64>, print: Fingerprint, add: bool) {
        let at = self.prints.entry(time).or_default();
        *at = if add { *at + print } else { *at - print };
        if *at == Fingerprint::default() {
            self.prints.remove(&time);
        }
    }

    /// A live row's fingerprint, from its version.
    pub(super) fn print_of(&self, row: RowId) -> Fingerprint {
        Fingerprint::of(row, self.table.version(row).unwrap_or(Version(0)))
    }

    /// Places every live row in its series again, as after a renumbering, from
    /// the series and time its key packs.
    pub(super) fn reindex_series(&mut self) {
        let Some(spec) = &self.spec.series else {
            return;
        };
        let mut index = SeriesIndex::new(spec.step).expect("the step was accepted at creation");
        let keyed: Vec<(RowKey, RowId)> = self.table.keys().collect();
        let rows: Vec<RowId> = keyed.iter().map(|&(_, row)| row).collect();
        let (_, targets) = self.gather(&rows).expect("live rows are stored");
        for (offset, &(key, row)) in keyed.iter().enumerate() {
            let (_, time) = SeriesKey::unpack(key);
            index
                .insert(row, key, time, targets.observed(offset))
                .expect("live rows were placed when inserted");
        }
        self.series = Some(index);
    }

    /// Checks that a batch of rows could be placed in their series, for a
    /// table of series: the whole batch against each series' grid, rows
    /// earlier in the batch included. With `replacing`, a time the table
    /// holds may be written over, as an update or upsert does.
    pub(super) fn check_series(
        &self,
        keys: &[u128],
        times: Option<&[Option<i64>]>,
        replacing: bool,
    ) -> Result<(), NodeError> {
        let Some(index) = &self.series else {
            return Ok(());
        };
        let times = times.ok_or(ForecastError::Untimed)?;
        let rows = keys
            .iter()
            .zip(times)
            .map(|(&key, &time)| time.map(|time| (RowKey(key), time)))
            .collect::<Option<Vec<_>>>()
            .ok_or(ForecastError::Untimed)?;
        index
            .check_batch(rows, |key| replacing && self.table.row(key).is_some())
            .map_err(|error| ForecastError::from(error).into())
    }

    /// The request a table has waited longest for, if any.
    pub(super) fn first_request(&self) -> Option<RequestId> {
        let pending = self.pending.first().map(|(request, _, _)| *request);
        let forecast = self.forecasts.first().map(|(request, _)| *request);
        pending.into_iter().chain(forecast).min()
    }

    /// Fingerprints and ranks every live row again, as after a renumbering.
    pub(super) fn reprint(&mut self) {
        let rows: Vec<RowId> = self.table.snapshot().rows().iter().collect();
        self.reindex_series();
        self.prints.clear();
        self.unlabeled.clear();
        let task = self.spec.task;
        for &row in &rows {
            let bits = self.rows.target(row).unwrap_or_default();
            let labeled = self.labeled(&Targets::from_bits(task, &[bits]))[0];
            if !labeled {
                self.unlabeled.insert(row.0);
            }
            if self.serves(labeled) {
                let (time, print) = (self.times.time(row), self.print_of(row));
                self.print(time, print, true);
            }
        }
        let served = self.served().rows();
        if let Some(ranking) = &mut self.ranking {
            *ranking = Ranking::new(ranking.seed(), &served, &self.table);
        }
        // Cores and regions' contexts name rows by id, which a renumbering changes.
        self.cores.1.clear();
        self.regions = None;
    }

    /// Rows' features, row-major, and their targets, in order.
    pub(super) fn gather(&self, rows: &[RowId]) -> Result<(Vec<f32>, Targets), NodeError> {
        let (features, bits) = self.rows.gather(rows)?;
        Ok((features, Targets::from_bits(self.spec.task, &bits)))
    }

    /// Rows in the order given, whose keys are `keys`, with their features,
    /// targets and times.
    pub(super) fn table_rows(
        &self,
        rows: &[RowId],
        keys: Vec<u128>,
    ) -> Result<TableRows, NodeError> {
        let (features, targets) = self.gather(rows)?;
        Ok(TableRows {
            keys,
            features,
            targets,
            times: rows.iter().map(|&row| self.times.time(row)).collect(),
        })
    }

    /// As `gather`, with only the features the model reads.
    pub(super) fn inputs(&self, rows: &[RowId]) -> Result<(Vec<f32>, Targets), NodeError> {
        let (features, targets) = self.gather(rows)?;
        Ok((self.spec.model_values(&features), targets))
    }

    /// Fits retrieval scaling on the live rows and places every row again with
    /// it, in row order: a table's first batch alone may say little.
    pub(super) fn rescale(&mut self) -> Result<(), NodeError> {
        let rows: Vec<RowId> = self.table.snapshot().rows().iter().collect();
        if rows.is_empty() {
            return Ok(());
        }
        // Unlabeled rows place nothing until they are labeled, unless none is.
        let labeled: Vec<RowId> = self.labeled_rows().iter().collect();
        let fit = if labeled.is_empty() { &rows } else { &labeled };
        // A scaling past `scale_rows` stays; the rows are only placed again.
        let frozen = self
            .spec
            .scale_rows
            .is_some_and(|limit| self.scaled_rows >= limit.get());
        let (standardizer, scaled_rows) = match &self.standardizer {
            Some(kept) if frozen => (kept.clone(), self.scaled_rows),
            _ => (
                {
                    let (features, targets) = self.gather(fit)?;
                    Standardizer::fit(&features, &targets, &self.spec)
                },
                labeled.len() as u64,
            ),
        };
        let mut index = PageIndex::new(self.dimensions, self.spec.page_size);
        let scaled = Scaled {
            rows: &self.rows,
            standardizer: Some(&standardizer),
        };
        let placed = rows
            .iter()
            .map(|&row| {
                Ok((
                    row,
                    self.spec.bucket(self.times.time(row)),
                    scaled.vector(row)?,
                ))
            })
            .collect::<Result<Vec<_>, RowsError>>()?;
        index.insert_many(&placed, &scaled)?;
        self.index = index;
        self.standardizer = Some(standardizer);
        self.scaled_rows = scaled_rows;
        Ok(())
    }

    /// Gives the live rows new ids, densely in key order, and places them
    /// again, so the table's ids and pages depend on its rows alone, not on
    /// the order of the writes that made them.
    pub(super) fn renumber(&mut self) -> Result<(), NodeError> {
        let mut keyed: Vec<(RowKey, RowId)> = self.table.keys().collect();
        keyed.sort_unstable();
        let rows: Vec<RowId> = keyed.iter().map(|&(_, row)| row).collect();
        let mut table = Table::new();
        let mut times = TimeIndex::new();
        for &(key, row) in &keyed {
            let renumbered = table.apply(Change::Insert(key))?;
            times.set(renumbered, self.times.time(row));
        }
        let (features, targets) = self.gather(&rows)?;
        let keys: Vec<u128> = keyed.iter().map(|&(key, _)| key.0).collect();
        self.rows.clear()?;
        self.rows.push(&features, &targets.bits(), &keys)?;
        self.table = table;
        self.times = times;
        self.reprint();
        if self.spec.series.is_some() {
            // A table of series places no rows in pages.
            return Ok(());
        }
        self.rescale()
    }

    /// Checks a batch's shape, keys and targets before anything changes.
    pub(super) fn check(
        &self,
        keys: &[u128],
        features: &[f32],
        targets: &Targets,
    ) -> Result<(), NodeError> {
        check_rows(keys.len(), features, self.spec.features)?;
        if targets.len() != keys.len() {
            return Err(NodeError::Count {
                what: Counted::Targets,
                expected: keys.len(),
                got: targets.len(),
            });
        }
        match (self.spec.task, targets) {
            (Task::Classes(classes), Targets::Classes(labels)) => {
                if let Some(&label) = labels
                    .iter()
                    .find(|&&label| label < UNLABELED || label >= 0 && label as u32 >= classes)
                {
                    return Err(NodeError::Label { label, classes });
                }
            }
            (Task::Values, Targets::Values(_)) => {}
            _ => return Err(NodeError::Task),
        }
        let mut seen = BTreeSet::new();
        if let Some(&key) = keys.iter().find(|&&key| !seen.insert(key)) {
            return Err(NodeError::DuplicateKey(RowKey(key)));
        }
        Ok(())
    }

    pub(super) fn vector(&self, row: &[f32]) -> Result<Vec<f32>, NodeError> {
        self.standardizer
            .as_ref()
            .map(|standardizer| standardizer.vector(row))
            .ok_or(NodeError::Unscaled)
    }
}
