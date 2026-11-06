// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The worker's load, in its own seconds: what each table's and model's
//! steps recently took, the wait a new request would have before its step
//! starts, and the refusal of one that would outwait its deadline.
//!
//! The worker runs steps one at a time, so how long a request waits is the
//! worker time of what is ahead of it, not how many requests are queued: a
//! handful of minute-long fits hold the worker as long as thousands of
//! cached decodes. Each step's time is split among its tables, as
//! `count_costs` charges it. A table's first step fits its state and
//! encodes the contexts it keeps, which later steps read, so it is expected
//! to take what its model's recent first steps did; a table's later steps
//! are expected to take what its own recent ones did, or before it has any,
//! its model's. A request context's table is new on every call with a new
//! context, so its model's first steps are what predicts it.
//!
//! A table's queued requests join one step, whose decodes take time by the
//! queries they answer: a step is expected to take at least its queries at
//! the table's recent decode seconds a query, so fifty callers queued on one
//! predictor weigh fifty times what one does.
//!
//! A step that runs past what it was expected to take, or whose time is not
//! known yet, as a model's first step on a worker just started, is expected
//! to run as long again as it has, but never past the moment every request
//! it answers is past its deadline, when the worker stops it.
//!
//! A request is refused at once, with how long until the queue would admit
//! it, when the work of other calls ahead of it and its own step would take
//! past the longest any caller waits, rather than answered with a timeout
//! once its caller has waited that long. The steps in flight are ahead of
//! it whatever their table, its own included: a step sent before the
//! request arrived cannot answer it. Its own call's work is never counted
//! against it, on its table or on the other tables a call split over
//! several, as a forecast over many aligned groups is, takes: a first fit
//! that takes longer than the deadline still runs, a call larger than one
//! step never refuses itself, and the retry that follows reads what it
//! kept. Only a server answering other callers refuses: an engine in a
//! caller's own process has no one else to wait for (see `shed`).

use super::*;

/// The weight of a step's time in the estimates it counts towards; the rest
/// is the estimate before it.
const WEIGHT: f64 = 0.5;

/// The longest `Retry-After` a refused request is given, in seconds.
const MAX_RETRY_SECONDS: u64 = 120;

/// What each table's steps, and each model's, recently took the worker.
#[derive(Debug, Default)]
pub(super) struct Costs {
    /// Steps each table has had.
    steps: HashMap<TableId, u64>,
    /// Each table's later steps, past its first.
    tables: HashMap<TableId, f64>,
    /// Each model's tables' first steps, and their later ones.
    first: HashMap<String, f64>,
    later: HashMap<String, f64>,
    /// Each table's decode seconds per query answered, and each model's.
    rates: HashMap<TableId, f64>,
    model_rates: HashMap<String, f64>,
}

/// Moves an estimate towards a new observation, or starts it there.
fn blend<K: Eq + std::hash::Hash>(estimates: &mut HashMap<K, f64>, key: K, seconds: f64) {
    estimates
        .entry(key)
        .and_modify(|old| *old += WEIGHT * (seconds - *old))
        .or_insert(seconds);
}

impl Costs {
    /// Counts a step's seconds for one of its tables, served by `model`.
    pub(super) fn record(&mut self, table: TableId, model: &str, seconds: f64) {
        if !(seconds.is_finite() && seconds >= 0.0) {
            return;
        }
        let steps = self.steps.entry(table).or_default();
        *steps += 1;
        if *steps == 1 {
            blend(&mut self.first, model.to_string(), seconds);
        } else {
            blend(&mut self.later, model.to_string(), seconds);
            blend(&mut self.tables, table, seconds);
        }
    }

    /// Counts the decode seconds a step took to answer `queries` of a table.
    pub(super) fn record_decodes(
        &mut self,
        table: TableId,
        model: &str,
        seconds: f64,
        queries: f64,
    ) {
        if !(seconds.is_finite() && seconds >= 0.0 && queries.is_finite() && queries >= 1.0) {
            return;
        }
        let rate = seconds / queries;
        blend(&mut self.rates, table, rate);
        blend(&mut self.model_rates, model.to_string(), rate);
    }

    /// The seconds the table's next step is expected to take: its model's
    /// first steps' for a table that has had none, else its own later
    /// steps', else its model's; nothing when none is known.
    pub(super) fn estimate(&self, table: TableId, model: Option<&str>) -> f64 {
        let of = |estimates: &HashMap<String, f64>| {
            model.and_then(|model| estimates.get(model)).copied()
        };
        let expected = if self.steps.get(&table).is_none_or(|&steps| steps == 0) {
            of(&self.first)
        } else {
            self.tables.get(&table).copied().or_else(|| of(&self.later))
        };
        expected.unwrap_or(0.0)
    }

    /// The seconds a step of the table answering `queries` queries is
    /// expected to take: its step's estimate, or its queries at the table's
    /// decode seconds a query, or else its model's, whichever is more.
    pub(super) fn estimate_for(&self, table: TableId, model: Option<&str>, queries: f64) -> f64 {
        let rate = self
            .rates
            .get(&table)
            .or_else(|| model.and_then(|model| self.model_rates.get(model)))
            .copied()
            .unwrap_or(0.0);
        self.estimate(table, model).max(rate * queries.max(0.0))
    }

    /// Forgets a table that is no longer served.
    pub(super) fn forget(&mut self, table: TableId) {
        self.steps.remove(&table);
        self.tables.remove(&table);
        self.rates.remove(&table);
    }
}

/// The seconds left of the step the worker is running: what it is expected
/// to take past the `elapsed` seconds it has run, or for one past that or
/// not known, as long again as it has run; never past `stops`, when every
/// request it answers is past its deadline, nor past `bound`.
pub(super) fn running_left(expected: f64, elapsed: f64, stops: f64, bound: f64) -> f64 {
    let expected = expected.clamp(0.0, bound);
    let elapsed = elapsed.max(0.0);
    let left = if expected > elapsed {
        expected - elapsed
    } else {
        elapsed
    };
    left.min(stops.max(0.0)).min(bound)
}

/// The seconds until the worker could start a new request's step: what is
/// left of the steps in flight, the first of them `elapsed` seconds into its
/// run and stopping `stops` seconds from now at the latest, then each
/// waiting table's expected step. Each step counts at most `bound` seconds,
/// the longest any caller waits, since a step whose callers have all gone
/// stops between its calls.
pub(super) fn wait_seconds(
    in_flight: &[f64],
    elapsed: f64,
    stops: f64,
    waiting: impl IntoIterator<Item = f64>,
    bound: f64,
) -> f64 {
    let capped = |seconds: f64| seconds.clamp(0.0, bound);
    let running = in_flight
        .iter()
        .enumerate()
        .map(|(index, &seconds)| {
            if index == 0 {
                running_left(seconds, elapsed, stops, bound)
            } else {
                capped(seconds)
            }
        })
        .sum::<f64>();
    running + waiting.into_iter().map(capped).sum::<f64>()
}

/// The seconds a request would take past its deadline, with `ahead` seconds
/// of other calls' work before its step and a step of `own` seconds; None
/// for one that would be answered in time. A step that alone takes past the
/// deadline (`alone`), as a large first fit does, is not counted: it runs
/// once nothing else is ahead, and the retry that follows reads what it kept.
pub(super) fn overrun(ahead: f64, own: f64, alone: f64, deadline: f64) -> Option<f64> {
    let own = if alone > deadline { 0.0 } else { own.max(0.0) };
    let total = ahead.max(0.0) + own;
    (total > deadline).then_some(total)
}

/// The `Retry-After` for a request refused with `wait` seconds of work
/// before its answer and a deadline of `deadline`: once the worker has run
/// the part of the queue past the deadline, a retry is answered within it.
/// At least a second and at most `MAX_RETRY_SECONDS`.
pub(super) fn retry_seconds(wait: f64, deadline: f64) -> u64 {
    let over = (wait - deadline).ceil();
    if over.is_finite() && over > 1.0 {
        (over as u64).min(MAX_RETRY_SECONDS)
    } else {
        1
    }
}

impl Shard {
    /// The model a table's steps run, durable or ephemeral.
    fn model_of(&self, id: TableId) -> Option<&str> {
        if let Some(ephemeral) = self.ephemeral.get(id) {
            return Some(ephemeral.definition.model.as_str());
        }
        let name = self.names.get(&id)?;
        self.tables
            .get(name)
            .map(|entry| entry.definition.model.as_str())
    }

    /// The seconds a step of the table is expected to take.
    pub(super) fn expected(&self, id: TableId) -> f64 {
        self.costs.estimate(id, self.model_of(id))
    }

    /// The seconds a step of the table answering `queries` queries is
    /// expected to take.
    pub(super) fn expected_for(&self, id: TableId, queries: f64) -> f64 {
        self.costs.estimate_for(id, self.model_of(id), queries)
    }

    /// The queries a request asks of a table: its rows of features, or the
    /// series it forecasts.
    pub(super) fn queries_of(&self, id: TableId, features: usize) -> f64 {
        let width = self.node.spec(id).map_or(1, |spec| spec.features.max(1));
        (features / width).max(1) as f64
    }

    /// Counts each table's part of a finished step towards its estimate and
    /// its model's, and tells the node; `decodes` is each table's decode
    /// seconds in the step and the queries they answered.
    pub(super) fn learn_costs(
        &mut self,
        cost: &HashMap<TableId, f64>,
        decodes: &HashMap<TableId, (f64, f64)>,
    ) {
        for (&table, &seconds) in cost {
            let Some(model) = self.model_of(table).map(str::to_string) else {
                continue;
            };
            self.costs.record(table, &model, seconds);
            if let Some(&(decoded, queries)) = decodes.get(&table) {
                self.costs.record_decodes(table, &model, decoded, queries);
            }
            let expected = self.expected(table);
            self.node.set_cost(table, expected);
        }
    }

    /// The tables of a step: those it fits, queries or forecasts.
    fn step_tables(step: &Step) -> HashSet<TableId> {
        step.fits()
            .iter()
            .map(|fit| fit.table)
            .chain(step.queries().iter().map(|batch| batch.table))
            .chain(step.forecasts().iter().map(|forecast| forecast.table))
            .chain(step.units().iter().map(|unit| unit.table))
            .collect()
    }

    /// The queries a step answers of each of its tables.
    fn step_queries(step: &Step) -> HashMap<TableId, f64> {
        let mut queries: HashMap<TableId, f64> = HashMap::new();
        for batch in step.queries() {
            *queries.entry(batch.table).or_default() += batch.units.len() as f64;
        }
        for forecast in step.forecasts() {
            *queries.entry(forecast.table).or_default() += 1.0;
        }
        queries
    }

    /// The tables holding work of the call `caller` traces, queued or in
    /// flight, as the parts of a forecast over several aligned groups are;
    /// none for a request without a trace, which is a call of its own.
    fn own_tables(&self, caller: &str) -> HashSet<TableId> {
        if caller.is_empty() {
            return HashSet::new();
        }
        self.pending
            .values()
            .filter(|pending| pending.trace == caller)
            .map(|pending| pending.table)
            .collect()
    }

    /// The requests the steps in flight answer.
    fn sent(&self) -> HashSet<RequestId> {
        self.in_flight
            .iter()
            .flat_map(|flight| {
                flight
                    .step
                    .queries()
                    .iter()
                    .map(|batch| batch.request)
                    .chain(
                        flight
                            .step
                            .forecasts()
                            .iter()
                            .map(|forecast| forecast.request),
                    )
            })
            .collect()
    }

    /// The queries of other calls waiting on `table` for its next step,
    /// which a new request of the call `caller` traces joins.
    fn queued_on(&self, table: TableId, caller: &str) -> f64 {
        let sent = self.sent();
        self.pending
            .iter()
            .filter(|(request, pending)| {
                pending.table == table
                    && !sent.contains(request)
                    && !pending.reply.is_closed()
                    && (caller.is_empty() || pending.trace != caller)
            })
            .map(|(_, pending)| pending.queries)
            .sum()
    }

    /// The seconds a new request would wait before the worker starts its
    /// step: the steps in flight, whatever their table, then every other
    /// table with requests waiting, each step as the queries it answers
    /// weigh; leaving out the tables in `own`, which hold the request's own
    /// call's work, as a call split over several tables or steps never waits
    /// for itself. The requests waiting on `table`, the request's own, join
    /// its step, so they are its step's, not ahead of it.
    pub(super) fn wait_ahead(&self, own: &HashSet<TableId>, table: Option<TableId>) -> f64 {
        let bound = self.request_timeout.as_secs_f64();
        let sent = self.sent();
        let in_flight: Vec<f64> = self
            .in_flight
            .iter()
            .map(|flight| {
                let queries = Self::step_queries(&flight.step);
                Self::step_tables(&flight.step)
                    .into_iter()
                    .filter(|table| !own.contains(table))
                    .map(|table| {
                        self.expected_for(table, queries.get(&table).copied().unwrap_or(0.0))
                    })
                    .sum()
            })
            .collect();
        let (elapsed, stops) = self.in_flight.front().map_or((0.0, bound), |flight| {
            let started = flight.sent.max(self.finished);
            let elapsed = Instant::now()
                .saturating_duration_since(started)
                .as_secs_f64();
            // A step answering no request, as a rebuild's, runs to its end.
            let stops = self
                .expires(&flight.step)
                .map_or(f64::INFINITY, |left| left.as_secs_f64());
            (elapsed, stops)
        });
        let mut waiting: HashMap<TableId, f64> = HashMap::new();
        for (request, pending) in &self.pending {
            if sent.contains(request)
                || own.contains(&pending.table)
                || Some(pending.table) == table
                || pending.reply.is_closed()
            {
                continue;
            }
            *waiting.entry(pending.table).or_default() += pending.queries;
        }
        wait_seconds(
            &in_flight,
            elapsed,
            stops,
            waiting
                .into_iter()
                .map(|(table, queries)| self.expected_for(table, queries)),
            bound,
        )
    }

    /// Whether a request of `queries` queries on `table`, of the call
    /// `caller` traces, would be answered past its deadline: the other
    /// calls' work ahead of it, its own step with the queries of other calls
    /// waiting on its table, and what that comes to; None if in time.
    fn late(&self, table: TableId, caller: &str, queries: f64) -> Option<(f64, f64, f64)> {
        let deadline = self.request_timeout.as_secs_f64();
        let ahead = self.wait_ahead(&self.own_tables(caller), Some(table));
        let alone = self.expected_for(table, queries);
        let own = self.expected_for(table, self.queued_on(table, caller) + queries);
        overrun(ahead, own, alone, deadline).map(|total| (ahead, own, total))
    }

    /// Refuses a request of `queries` queries on `table`, of the call
    /// `caller` traces, that would be answered past the longest any caller
    /// waits, unless the shard was told never to.
    pub(super) fn admit_wait(
        &mut self,
        table: TableId,
        caller: &str,
        queries: f64,
    ) -> Result<(), ServerError> {
        if !self.shed {
            return Ok(());
        }
        if self.late(table, caller, queries).is_none() {
            return Ok(());
        }
        // Callers that have gone free the worker, so look again without them.
        self.cancel_abandoned();
        let Some((ahead, own, total)) = self.late(table, caller, queries) else {
            return Ok(());
        };
        let deadline = self.request_timeout.as_secs_f64();
        self.metrics.requests_shed.add(1);
        let retry = retry_seconds(total, deadline);
        Err(ServerError::Backlogged(
            format!(
                "the worker has about {ahead:.0} s of other calls' work ahead of this \
                 request and about {own:.0} s in its step, past the {deadline:.0} s a \
                 request may wait; retry in {retry} s"
            ),
            retry,
        ))
    }

    /// Reports the wait a new request would have, for the pool's autoscaler.
    pub(super) fn report_wait(&self) {
        let wait = self.wait_ahead(&HashSet::new(), None);
        self.metrics.set_queue_wait(self.load_slot, wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tables_first_step_is_expected_as_its_models_and_later_ones_as_its_own() {
        let mut costs = Costs::default();
        // A tuned predictor's first step fits and encodes for 120 s; its
        // later steps read what it kept.
        assert_eq!(costs.estimate(TableId(1), Some("mitra")), 0.0);
        costs.record(TableId(1), "mitra", 120.0);
        assert_eq!(costs.estimate(TableId(1), Some("mitra")), 0.0);
        costs.record(TableId(1), "mitra", 2.0);
        costs.record(TableId(1), "mitra", 4.0);
        assert_eq!(costs.estimate(TableId(1), Some("mitra")), 3.0);

        // A new table of the model expects a first step; a second one, what
        // the model's later steps took.
        assert_eq!(costs.estimate(TableId(2), Some("mitra")), 120.0);
        costs.record(TableId(2), "mitra", 80.0);
        assert_eq!(costs.estimate(TableId(2), Some("mitra")), 3.0);
        assert_eq!(costs.estimate(TableId(3), Some("mitra")), 100.0);
        assert_eq!(costs.estimate(TableId(3), Some("tabicl")), 0.0);
        assert_eq!(costs.estimate(TableId(3), None), 0.0);

        costs.forget(TableId(1));
        assert_eq!(costs.estimate(TableId(1), Some("mitra")), 100.0);
        // A nonsense time changes nothing.
        costs.record(TableId(4), "mitra", f64::NAN);
        assert_eq!(costs.estimate(TableId(4), Some("mitra")), 100.0);
    }

    #[test]
    fn the_wait_counts_what_is_left_of_the_running_step_and_caps_each_step() {
        let never = f64::INFINITY;
        // A 75 s step 30 s in, a 10 s step behind it, and two tables waiting.
        assert_eq!(
            wait_seconds(&[75.0, 10.0], 30.0, never, [20.0, 5.0], 140.0),
            80.0
        );
        // A step counts at most the deadline, as one past it stops.
        assert_eq!(wait_seconds(&[], 0.0, never, [480.0, 480.0], 140.0), 280.0);
        assert_eq!(wait_seconds(&[], 0.0, never, [], 140.0), 0.0);
    }

    #[test]
    fn a_step_past_its_estimate_or_without_one_runs_as_long_again_until_it_stops() {
        let never = f64::INFINITY;
        // A 10 s step 30 s in is expected to run 30 s more, not none.
        assert_eq!(running_left(10.0, 30.0, never, 140.0), 30.0);
        // A model's first step on a worker just started, with no estimate.
        assert_eq!(running_left(0.0, 50.0, never, 140.0), 50.0);
        // Never past when every request it answers is past its deadline...
        assert_eq!(running_left(0.0, 100.0, 40.0, 140.0), 40.0);
        assert_eq!(running_left(75.0, 30.0, 20.0, 140.0), 20.0);
        // ...nor past the deadline, and nothing once its callers have gone.
        assert_eq!(running_left(0.0, 200.0, never, 140.0), 140.0);
        assert_eq!(running_left(0.0, 150.0, 0.0, 140.0), 0.0);
        // Within its estimate, what is left of it.
        assert_eq!(running_left(75.0, 30.0, never, 140.0), 45.0);
        assert_eq!(wait_seconds(&[10.0, 10.0], 30.0, never, [], 140.0), 40.0);
    }

    #[test]
    fn a_step_of_many_queries_takes_them_at_the_tables_decode_rate() {
        let mut costs = Costs::default();
        costs.record(TableId(1), "kumo", 30.0);
        costs.record(TableId(1), "kumo", 2.0);
        // 128 queries decoded in 1.28 s.
        costs.record_decodes(TableId(1), "kumo", 1.28, 128.0);
        // A step of one caller's queries takes what its steps have.
        assert_eq!(costs.estimate_for(TableId(1), Some("kumo"), 128.0), 2.0);
        // Fifty callers' queries queued on the table take fifty times the decodes.
        let fifty = costs.estimate_for(TableId(1), Some("kumo"), 50.0 * 128.0);
        assert!((fifty - 64.0).abs() < 1e-9, "{fifty}");
        // A new table of the model takes the model's rate until it has its own.
        let new = costs.estimate_for(TableId(2), Some("kumo"), 6400.0);
        assert!((new - 64.0).abs() < 1e-9, "{new}");
        assert_eq!(costs.estimate_for(TableId(2), Some("tabicl"), 6400.0), 0.0);
        // Nonsense counts change nothing.
        costs.record_decodes(TableId(1), "kumo", f64::NAN, 4.0);
        costs.record_decodes(TableId(1), "kumo", 1.0, 0.0);
        let same = costs.estimate_for(TableId(1), Some("kumo"), 6400.0);
        assert!((same - 64.0).abs() < 1e-9, "{same}");
        costs.forget(TableId(1));
        let forgotten = costs.estimate_for(TableId(1), Some("kumo"), 6400.0);
        assert!((forgotten - 64.0).abs() < 1e-9, "{forgotten}");
    }

    #[test]
    fn a_request_is_late_when_the_work_ahead_and_its_own_step_pass_the_deadline() {
        // 100 s of other calls' work ahead and a 50 s step: late by 10 s.
        assert_eq!(overrun(100.0, 50.0, 5.0, 140.0), Some(150.0));
        assert_eq!(overrun(100.0, 40.0, 5.0, 140.0), None);
        // An idle worker answers any step that takes no longer than the deadline.
        assert_eq!(overrun(0.0, 140.0, 140.0, 140.0), None);
        // Its table's queue alone past the deadline is late, idle worker or not.
        assert_eq!(overrun(0.0, 150.0, 1.0, 140.0), Some(150.0));
        // A step that alone takes past the deadline, as a large first fit,
        // runs once nothing else is ahead; with more ahead than the deadline
        // it is refused as before.
        assert_eq!(overrun(100.0, 300.0, 300.0, 140.0), None);
        assert_eq!(overrun(150.0, 300.0, 300.0, 140.0), Some(150.0));
    }

    #[test]
    fn a_refused_request_retries_once_the_queue_is_within_its_deadline() {
        assert_eq!(retry_seconds(200.0, 140.0), 60);
        assert_eq!(retry_seconds(140.2, 140.0), 1);
        assert_eq!(retry_seconds(10_000.0, 140.0), MAX_RETRY_SECONDS);
        assert_eq!(retry_seconds(f64::INFINITY, 140.0), 1);
    }
}
