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
//! A request whose estimated wait for the steps ahead of it is past the
//! longest any caller waits is refused at once, with how long until the
//! queue would admit it, rather than answered with a timeout once its
//! caller has waited that long. Its own table's step is never counted
//! against it: a first fit that takes longer than the deadline still runs,
//! and the retry that follows reads what it kept.

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

    /// Forgets a table that is no longer served.
    pub(super) fn forget(&mut self, table: TableId) {
        self.steps.remove(&table);
        self.tables.remove(&table);
    }
}

/// The seconds until the worker could start a new request's step: what is
/// left of the steps in flight, the first of them `elapsed` seconds into its
/// run, then each waiting table's expected step. Each step counts at most
/// `bound` seconds, the longest any caller waits, since a step whose callers
/// have all gone stops between its calls.
pub(super) fn wait_seconds(
    in_flight: &[f64],
    elapsed: f64,
    waiting: impl IntoIterator<Item = f64>,
    bound: f64,
) -> f64 {
    let capped = |seconds: f64| seconds.clamp(0.0, bound);
    let running = in_flight
        .iter()
        .enumerate()
        .map(|(index, &seconds)| {
            if index == 0 {
                (capped(seconds) - elapsed.max(0.0)).max(0.0)
            } else {
                capped(seconds)
            }
        })
        .sum::<f64>();
    running + waiting.into_iter().map(capped).sum::<f64>()
}

/// The `Retry-After` for a request refused with `wait` seconds ahead of it
/// and a deadline of `deadline`: once the worker has run the part of the
/// queue past the deadline, a retry's wait is within it. At least a second
/// and at most `MAX_RETRY_SECONDS`.
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

    /// Counts each table's part of a finished step towards its estimate and
    /// its model's, and tells the node.
    pub(super) fn learn_costs(&mut self, cost: &HashMap<TableId, f64>) {
        for (&table, &seconds) in cost {
            let Some(model) = self.model_of(table).map(str::to_string) else {
                continue;
            };
            self.costs.record(table, &model, seconds);
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

    /// The seconds a new request on `table` (or on a table with nothing
    /// queued, if None) would wait before the worker starts its step: the
    /// steps in flight, then every other table with requests waiting.
    pub(super) fn wait_ahead(&self, table: Option<TableId>) -> f64 {
        let mut sent: HashSet<RequestId> = HashSet::new();
        let in_flight: Vec<f64> = self
            .in_flight
            .iter()
            .map(|flight| {
                sent.extend(flight.step.queries().iter().map(|batch| batch.request));
                sent.extend(
                    flight
                        .step
                        .forecasts()
                        .iter()
                        .map(|forecast| forecast.request),
                );
                Self::step_tables(&flight.step)
                    .into_iter()
                    .map(|table| self.expected(table))
                    .sum()
            })
            .collect();
        let elapsed = self.in_flight.front().map_or(0.0, |flight| {
            let started = flight.sent.max(self.finished);
            Instant::now()
                .saturating_duration_since(started)
                .as_secs_f64()
        });
        let waiting: HashSet<TableId> = self
            .pending
            .iter()
            .filter(|(request, pending)| {
                !sent.contains(request)
                    && Some(pending.table) != table
                    && !pending.reply.is_closed()
            })
            .map(|(_, pending)| pending.table)
            .collect();
        wait_seconds(
            &in_flight,
            elapsed,
            waiting.into_iter().map(|table| self.expected(table)),
            self.request_timeout.as_secs_f64(),
        )
    }

    /// Refuses a request on `table` whose wait for the work ahead of it is
    /// past the longest any caller waits, unless the shard was told never to.
    pub(super) fn admit_wait(&mut self, table: TableId) -> Result<(), ServerError> {
        if !self.shed {
            return Ok(());
        }
        let deadline = self.request_timeout.as_secs_f64();
        let wait = self.wait_ahead(Some(table));
        if wait <= deadline {
            return Ok(());
        }
        // Callers that have gone free the worker, so look again without them.
        self.cancel_abandoned();
        let wait = self.wait_ahead(Some(table));
        if wait <= deadline {
            return Ok(());
        }
        self.metrics.requests_shed.add(1);
        let retry = retry_seconds(wait, deadline);
        Err(ServerError::Backlogged(
            format!(
                "the worker has about {wait:.0} s of work queued ahead of this request, past \
                 the {deadline:.0} s a request may wait; retry in {retry} s"
            ),
            retry,
        ))
    }

    /// Reports the wait a new request would have, for the pool's autoscaler.
    pub(super) fn report_wait(&self) {
        let wait = self.wait_ahead(None);
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
        // A 75 s step 30 s in, a 10 s step behind it, and two tables waiting.
        assert_eq!(wait_seconds(&[75.0, 10.0], 30.0, [20.0, 5.0], 140.0), 80.0);
        // A step running past its estimate leaves nothing of it.
        assert_eq!(wait_seconds(&[10.0], 30.0, [], 140.0), 0.0);
        // A step counts at most the deadline, as one past it stops.
        assert_eq!(wait_seconds(&[], 0.0, [480.0, 480.0], 140.0), 280.0);
        assert_eq!(wait_seconds(&[], 0.0, [], 140.0), 0.0);
    }

    #[test]
    fn a_refused_request_retries_once_the_queue_is_within_its_deadline() {
        assert_eq!(retry_seconds(200.0, 140.0), 60);
        assert_eq!(retry_seconds(140.2, 140.0), 1);
        assert_eq!(retry_seconds(10_000.0, 140.0), MAX_RETRY_SECONDS);
        assert_eq!(retry_seconds(f64::INFINITY, 140.0), 1);
    }
}
