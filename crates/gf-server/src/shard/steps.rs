// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Steps: planning them, sending them to the worker, and answering the
//! requests and failures each one's reply brings.

use super::*;

impl Shard {
    /// Plans pending predictions into a step and sends it to the worker.
    ///
    /// Returns whether a step was sent.
    pub(super) async fn dispatch(&mut self) -> bool {
        if self.down.is_some() {
            return false;
        }
        let planning = Instant::now();
        let scheduled = self.node.schedule();
        let planned = Instant::now();
        for table in self.node.take_past_cap() {
            if let Some(name) = self.names.get(&table) {
                tracing::warn!(
                    table = name.as_str(),
                    "a context is larger than the model was trained on; \
                     past_cap = \"clamp\" cuts contexts to that length"
                );
            }
        }
        for (request, error) in self.node.take_refused() {
            self.metrics.requests_failed.add(1);
            if let Some(pending) = self.take(request) {
                let _ = pending.reply.send(Err(error.into()));
            }
        }
        for table in self.node.take_advanced() {
            if let Some(metrics) = self.table_metrics(table) {
                metrics.epochs.add(1);
            }
        }
        // Tables that wait for their fit's rows to be measured plan again
        // once they are.
        let measured = self.measure_fits().await;
        let step = match scheduled {
            Ok(Some(step)) => step,
            Ok(None) => return measured,
            // A table whose planning fails has its own requests refused
            // above; what is left is the cache refusing a rebuild no request
            // asked for, which fails what waits rather than leave it waiting
            // on a rebuild that keeps failing.
            Err(error) => {
                self.fail_all(&ServerError::from(error));
                return false;
            }
        };
        self.metrics.plan_seconds.observe(planned - planning);
        for (name, calls, seconds) in self.node.plan_parts().iter() {
            let part = self.metrics.plan_part(name);
            part.calls.add(calls);
            part.seconds.add(seconds.as_secs_f64());
        }
        let mut tables = HashSet::new();
        for batch in step.queries() {
            tables.insert(batch.table);
            if let Some(pending) = self.pending.get(&batch.request) {
                self.metrics
                    .queue_seconds
                    .observe(planned - pending.arrived);
            }
        }
        self.metrics.step_tables.add(tables.len() as u64);
        self.report_cache();
        match self.worker.pipeline.send(&step_frame(&step)).await {
            Ok(reply) => {
                self.in_flight.push_back(InFlight {
                    step,
                    planned,
                    sent: Instant::now(),
                    reply,
                });
                true
            }
            Err(error) => {
                let dropped = self.node.fail(&step);
                self.fail_requests(&step, None, &ServerError::Internal(error));
                self.send_drops(&dropped).await;
                false
            }
        }
    }

    /// Waits for every step in flight and answers its requests; a worker
    /// that misses a step's deadline is started again, failing the rest.
    pub(super) async fn drain(&mut self) {
        while let Some(deadline) = self.deadline() {
            let step = self.in_flight.front_mut().expect("a step is in flight");
            match tokio::time::timeout_at(deadline, &mut step.reply).await {
                Ok(reply) => {
                    let reply = reply.unwrap_or_else(|_| Err("the worker stopped".to_string()));
                    self.finish(reply).await;
                }
                Err(_) if self.overdue() => self.restart(&self.hung()).await,
                Err(_) => {}
            }
        }
    }

    /// Completes the oldest step in flight with the worker's reply.
    pub(super) async fn finish(&mut self, reply: worker::Reply) {
        self.settle_controls().await;
        let InFlight {
            step,
            planned,
            sent,
            ..
        } = self.in_flight.pop_front().expect("a step is in flight");
        let now = Instant::now();
        let times = StepTimes {
            planned,
            sent,
            started: sent.max(self.finished),
            finished: now,
        };
        let traces: Vec<&str> = step
            .queries()
            .iter()
            .map(|batch| batch.request)
            .chain(step.forecasts().iter().map(|forecast| forecast.request))
            .filter_map(|request| self.pending.get(&request))
            .map(|pending| pending.trace.as_str())
            .collect();
        // A step starts once the one before it finished, since the worker
        // runs them in turn, or when it was sent, after an idle spell.
        let took = now - times.started;
        tracing::debug!(
            traces = ?traces,
            units = step.units().len(),
            fits = step.fits().len(),
            queries = step.queries().len(),
            milliseconds = took.as_secs_f64() * 1e3,
            "ran a step"
        );
        self.metrics.step_seconds.observe(took);
        self.finished = now;
        self.metrics.steps.add(1);
        for unit in step.units() {
            let (units, rows) = match unit.placed.action() {
                gf_core::node::Action::Encode => {
                    (&self.metrics.units_encoded, &self.metrics.rows_encoded)
                }
                gf_core::node::Action::Reuse => {
                    (&self.metrics.units_reused, &self.metrics.rows_reused)
                }
                gf_core::node::Action::Transient | gf_core::node::Action::Fuse => {
                    (&self.metrics.units_transient, &self.metrics.rows_encoded)
                }
                // Tabular units are never extended; counted as encoded if one were.
                gf_core::node::Action::Extend => {
                    (&self.metrics.units_encoded, &self.metrics.rows_encoded)
                }
            };
            units.add(1);
            rows.add(unit.rows.len() as u64);
        }
        for page in step.pages() {
            let Some(metrics) = self.table_metrics(page.table) else {
                continue;
            };
            let copied: usize = page
                .copies
                .iter()
                .map(|(_, positions)| positions.len())
                .sum();
            metrics.rows_copied.add(copied as u64);
            if page.embed.is_some() {
                metrics.rows_embedded.add((page.rows.len() - copied) as u64);
            }
        }
        let outcome = reply
            .map_err(ServerError::Internal)
            .and_then(|frame| outcome_from(&frame).map_err(|error| ServerError::Internal(error.0)));
        // A failed part's entries may be missing from the worker, so later steps encode them again.
        let dropped = match &outcome {
            Ok(outcome) => {
                let failed: HashSet<TableId> =
                    outcome.failed.iter().map(|(table, _)| *table).collect();
                self.node
                    .complete_deferring(&step, &failed, &outcome.deferred)
            }
            Err(_) => self.node.fail(&step),
        };
        match outcome {
            Ok(outcome) => {
                let shares = self.count_costs(&step, &outcome);
                let Outcome {
                    predictions,
                    failed,
                    deferred,
                    ..
                } = outcome;
                for (table, reason) in &failed {
                    self.fail_requests(&step, Some(*table), &ServerError::Internal(reason.clone()));
                }
                self.answer(&step, predictions, &shares, times, &deferred);
                let tables = step
                    .fits()
                    .iter()
                    .map(|fit| fit.table)
                    .chain(step.queries().iter().map(|batch| batch.table))
                    .chain(step.forecasts().iter().map(|forecast| forecast.table));
                for table in tables {
                    if !failed.iter().any(|(failing, _)| *failing == table) {
                        self.failures.remove(&table);
                    }
                }
                for (table, reason) in failed {
                    self.count_failure(table, &reason);
                }
            }
            Err(error) => {
                self.fail_requests(&step, None, &error);
                let reason = error.to_string();
                let fitted: Vec<TableId> = step.fits().iter().map(|fit| fit.table).collect();
                for table in fitted {
                    self.count_failure(table, &reason);
                }
            }
        }
        self.send_drops(&dropped).await;
        self.sweep_scratch().await;
        self.report_cache();
    }

    /// Counts a failed step against a table, and quarantines it once it has
    /// failed `FAILURES` steps running: its queued requests fail, and later
    /// ones are refused.
    fn count_failure(&mut self, table: TableId, reason: &str) {
        let failures = self.failures.entry(table).or_default();
        *failures += 1;
        if *failures < FAILURES || self.quarantined.contains_key(&table) {
            return;
        }
        let name = self
            .names
            .get(&table)
            .map_or_else(String::new, ToString::to_string);
        tracing::error!(
            table = name,
            "quarantining a table that failed {FAILURES} steps running: {reason}"
        );
        self.metrics.tables_quarantined.add(1);
        self.quarantined.insert(table, reason.to_string());
        let queued: Vec<RequestId> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.table == table)
            .map(|(&request, _)| request)
            .collect();
        for request in queued {
            self.node.cancel(request);
            if let Some(pending) = self.take(request) {
                self.metrics.requests_failed.add(1);
                let _ = pending.reply.send(Err(ServerError::Invalid(format!(
                    "table {name:?} takes no queries: it failed {FAILURES} steps running \
                     ({reason}); a write or refit lets it try again"
                ))));
            }
        }
    }

    /// Counts what a step cost the worker: its time, by stage and by table,
    /// and in each table's fits. Returns each request's share of the step.
    ///
    /// Each stage's seconds go to the step's tables by the rows each took in
    /// that stage: fits by the rows they read, encodes by the context rows
    /// newly encoded, decodes by query rows. So a decode against a cached
    /// context is charged by its query rows, not by the context's size.
    /// Other stages, such as page embeds, and the worker's time outside
    /// compiled calls go by every row each table took. A table's cost is
    /// then split among its requests in the step by their query rows and the
    /// context rows each newly encoded, those shared split evenly among the
    /// requests reading them. A table with no requests in the step, as in a
    /// region rebuild's, keeps its cost as rebuild seconds.
    fn count_costs(&self, step: &Step, outcome: &Outcome) -> HashMap<RequestId, Share> {
        self.metrics.worker_seconds.add(outcome.seconds);
        for (name, time) in &outcome.stages {
            let stage = self.metrics.stage(name);
            stage.calls.add(time.calls);
            stage.seconds.add(time.seconds);
            stage.flops.add(time.flops);
        }
        self.metrics
            .compile_cache_hits
            .add(outcome.compile_cache.hits);
        self.metrics
            .compile_cache_misses
            .add(outcome.compile_cache.misses);
        for (name, time) in &outcome.parts {
            let part = self.metrics.part(name);
            part.calls.add(time.calls);
            part.seconds.add(time.seconds);
        }
        for (kind, bytes) in &outcome.memory {
            self.metrics.memory_peak(kind, *bytes);
        }
        let fits: Vec<_> = step
            .fits()
            .iter()
            .filter(|fit| !outcome.failed.iter().any(|(table, _)| *table == fit.table))
            .collect();
        // Rows each table took in the fit, encode and decode stages, and in all.
        let mut fitted: HashMap<TableId, f64> = HashMap::new();
        let mut encoded: HashMap<TableId, f64> = HashMap::new();
        let mut queried: HashMap<TableId, f64> = HashMap::new();
        let mut all: HashMap<TableId, f64> = HashMap::new();
        for fit in &fits {
            *fitted.entry(fit.table).or_default() += fit.rows.len() as f64;
        }
        for unit in step.units() {
            if !matches!(unit.placed.action(), Action::Reuse) {
                *encoded.entry(unit.table).or_default() += unit.rows.len() as f64;
            }
        }
        for batch in step.queries() {
            *queried.entry(batch.table).or_default() += batch.units.len() as f64;
        }
        // A series unit's rows are its grid's; a forecast reads one unit.
        for unit in step.series_units() {
            if !matches!(unit.placed.action(), Action::Reuse) {
                *encoded.entry(unit.table).or_default() += unit.grid.rows as f64;
            }
        }
        for forecast in step.forecasts() {
            *queried.entry(forecast.table).or_default() += 1.0;
        }
        for page in step.pages().iter().filter(|page| page.embed.is_some()) {
            *all.entry(page.table).or_default() += page.rows.len() as f64;
        }
        for (table, rows) in fitted.iter().chain(&encoded).chain(&queried) {
            *all.entry(*table).or_default() += rows;
        }
        let mut cost: HashMap<TableId, f64> = HashMap::new();
        let mut rest = outcome.seconds;
        for (name, time) in &outcome.stages {
            let rows = match name.as_str() {
                "fit" => &fitted,
                // A unit fit on its own rows encodes with its fit, and its
                // queries embed with their decode; a series unit encodes or is
                // forecast whole, and its forecasts decode.
                "encode" | "fit_encode" | "series_encode" | "forecast" => &encoded,
                "decode" | "embed_decode" | "series_decode" => &queried,
                _ => &all,
            };
            if charge(&mut cost, time.seconds, rows) {
                rest -= time.seconds;
            }
        }
        charge(&mut cost, rest.max(0.0), &all);
        let fit_seconds = outcome.stages.get("fit").map_or(0.0, |time| time.seconds);
        let fit_rows: f64 = fitted.values().sum();
        for fit in fits {
            if let Some(metrics) = self.table_metrics(fit.table) {
                metrics.refits.add(1);
                metrics.refit_rows.add(fit.rows.len() as u64);
                metrics
                    .refit_seconds
                    .add(fit_seconds * fit.rows.len() as f64 / fit_rows);
            }
        }
        let read: Vec<HashSet<StepUnit>> = step
            .queries()
            .iter()
            .map(|batch| batch.units.iter().copied().collect())
            .collect();
        let mut readers: HashMap<StepUnit, usize> = HashMap::new();
        for unit in read.iter().flatten() {
            *readers.entry(*unit).or_default() += 1;
        }
        let context: Vec<f64> = read
            .iter()
            .map(|units| {
                units
                    .iter()
                    .map(|&index| {
                        let unit = step.unit(index);
                        let encoded = match unit.placed.action() {
                            Action::Reuse => 0,
                            Action::Encode | Action::Transient | Action::Fuse | Action::Extend => {
                                unit.rows.len()
                            }
                        };
                        encoded as f64 / readers[&index] as f64
                    })
                    .sum()
            })
            .collect();
        let mut weights: HashMap<TableId, f64> = HashMap::new();
        for (batch, context) in step.queries().iter().zip(&context) {
            *weights.entry(batch.table).or_default() += batch.units.len() as f64 + context;
        }
        // Each forecast request's table, forecasts, which weigh one each, and
        // the context steps its units newly encoded, shared among their readers.
        let mut reading: HashMap<StepUnit, usize> = HashMap::new();
        for forecast in step.forecasts() {
            *reading.entry(forecast.unit).or_default() += 1;
        }
        let mut forecasts: HashMap<RequestId, (TableId, f64, f64)> = HashMap::new();
        for forecast in step.forecasts() {
            let unit = &step.series_units()[forecast.unit.0];
            // An extension encodes its new steps and reuses the rest.
            let steps = match unit.placed.action() {
                Action::Reuse => 0,
                Action::Encode | Action::Transient | Action::Fuse => unit.grid.steps,
                Action::Extend => unit.grown,
            };
            let entry = forecasts
                .entry(forecast.request)
                .or_insert((forecast.table, 0.0, 0.0));
            entry.1 += 1.0;
            entry.2 += steps as f64 / reading[&forecast.unit] as f64;
            *weights.entry(forecast.table).or_default() += 1.0;
        }
        for (table, &seconds) in &cost {
            if let Some(metrics) = self.table_metrics(*table) {
                metrics.gpu_seconds.add(seconds);
                if !weights.contains_key(table) {
                    metrics.rebuild_seconds.add(seconds);
                }
            }
        }
        step.queries()
            .iter()
            .zip(context)
            .map(|(batch, context)| {
                let weight = batch.units.len() as f64 + context;
                let table = cost.get(&batch.table).copied().unwrap_or_default();
                let seconds = table * weight / weights[&batch.table];
                (batch.request, (context, seconds))
            })
            .chain(
                forecasts
                    .into_iter()
                    .map(|(request, (table, count, steps))| {
                        let seconds =
                            cost.get(&table).copied().unwrap_or_default() * count / weights[&table];
                        (request, (steps, seconds))
                    }),
            )
            .collect()
    }

    /// Answers a step's requests with the worker's predictions, each with
    /// its share of the step, and logs their timings.
    pub(super) fn answer(
        &mut self,
        step: &Step,
        predictions: Vec<(u64, Predictions)>,
        shares: &HashMap<RequestId, Share>,
        times: StepTimes,
        deferred: &[(TableId, String)],
    ) {
        for (request, mut predictions) in predictions {
            self.metrics.predictions.add(predictions.rows as u64);
            let Some(pending) = self.take(RequestId(request)) else {
                continue;
            };
            let (context_rows, gpu_seconds) =
                shares.get(&RequestId(request)).copied().unwrap_or_default();
            predictions.gpu_seconds = gpu_seconds;
            let forecasts: Vec<&forecasting::Forecast> = step
                .forecasts()
                .iter()
                .filter(|forecast| forecast.request.0 == request)
                .collect();
            predictions.epoch = step
                .queries()
                .iter()
                .find(|batch| batch.request.0 == request)
                .map(|batch| batch.epoch.0)
                .or_else(|| forecasts.first().map(|forecast| forecast.epoch.0))
                .unwrap_or(0);
            predictions.cutoffs = forecasts
                .iter()
                .map(|forecast| step.series_units()[forecast.unit.0].cutoff)
                .collect();
            self.metrics.log_timing(|| Timing {
                table: self
                    .name_of(pending.table)
                    .map_or_else(String::new, ToString::to_string),
                request,
                trace: pending.trace.clone(),
                arrived: pending.arrived.into_std(),
                planned: times.planned.into_std(),
                sent: times.sent.into_std(),
                started: times.started.into_std(),
                finished: times.finished.into_std(),
                queries: predictions.rows as u64,
                context_rows,
                gpu_seconds,
            });
            let _ = pending.reply.send(Ok(predictions));
        }
        // A request the reply leaves out would otherwise wait forever,
        // unless its table's part waits for a program and queued again.
        let unanswered = ServerError::Internal(
            "the worker returned no predictions for this request".to_string(),
        );
        for batch in step.queries() {
            if deferred.iter().any(|(table, _)| *table == batch.table) {
                continue;
            }
            if let Some(pending) = self.take(batch.request) {
                self.metrics.requests_failed.add(1);
                let _ = pending.reply.send(Err(unanswered.clone()));
            }
        }
        for forecast in step.forecasts() {
            if let Some(pending) = self.take(forecast.request) {
                self.metrics.requests_failed.add(1);
                let _ = pending.reply.send(Err(unanswered.clone()));
            }
        }
    }

    /// Fails a step's requests that are still waiting, for one table or all.
    fn fail_requests(&mut self, step: &Step, table: Option<TableId>, error: &ServerError) {
        let requests = step
            .queries()
            .iter()
            .map(|batch| (batch.table, batch.request))
            .chain(
                step.forecasts()
                    .iter()
                    .map(|forecast| (forecast.table, forecast.request)),
            );
        for (of, request) in requests {
            if table.is_some_and(|table| table != of) {
                continue;
            }
            if let Some(pending) = self.take(request) {
                self.metrics.requests_failed.add(1);
                let _ = pending.reply.send(Err(error.clone()));
            }
        }
    }

    /// Sends drops to the worker without waiting: the pipeline orders them
    /// with the tick's other controls, and a failure is only logged, since
    /// nothing reads those entries again.
    pub(super) async fn send_drops(&mut self, movements: &[Movement]) {
        // A worker starting again starts empty, so it has nothing to drop.
        if movements.is_empty() || self.down.is_some() {
            return;
        }
        let reply = self.worker.pipeline.control(WorkerRequest::Apply {
            movements: movements_json(movements),
        });
        tokio::spawn(async move {
            if let Ok(Err(error)) = reply.await {
                tracing::error!("applying drops after a step: {error}");
            }
        });
    }

    fn fail_all(&mut self, error: &ServerError) {
        self.queued_bytes = 0;
        let now = Instant::now();
        for (_, pending) in self.pending.drain() {
            self.ephemeral.unpin(pending.table, now);
            let _ = pending.reply.send(Err(error.clone()));
        }
    }
}

/// Adds `seconds` to tables' costs by the rows each took; returns whether
/// any took rows, and so whether the seconds went anywhere.
pub(super) fn charge(
    cost: &mut HashMap<TableId, f64>,
    seconds: f64,
    rows: &HashMap<TableId, f64>,
) -> bool {
    let total: f64 = rows.values().sum();
    if total <= 0.0 {
        return false;
    }
    for (table, rows) in rows {
        *cost.entry(*table).or_default() += seconds * rows / total;
    }
    true
}
