// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics: counters, gauges and time histograms, for the server and
//! for each table, and an optional log of each request's timings.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A count that only goes up.
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    /// Adds to the count.
    pub fn add(&self, value: u64) {
        self.0.fetch_add(value, Ordering::Relaxed);
    }

    /// The count.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Time summed, kept as nanoseconds so microsecond steps add up.
#[derive(Debug, Default)]
pub struct Seconds(AtomicU64);

impl Seconds {
    /// Adds a time in seconds; a negative one adds nothing.
    pub fn add(&self, seconds: f64) {
        self.0
            .fetch_add((seconds * 1e9).round() as u64, Ordering::Relaxed);
    }

    /// The time in seconds.
    #[must_use]
    pub fn get(&self) -> f64 {
        self.0.load(Ordering::Relaxed) as f64 / 1e9
    }
}

/// A sum of floating-point values that only goes up, such as FLOPs, which
/// would overflow a `Counter` within hours on an accelerator.
#[derive(Debug, Default)]
pub struct Total(AtomicU64);

impl Total {
    /// Adds a value; a negative one adds nothing.
    pub fn add(&self, value: f64) {
        if value > 0.0 {
            let _ = self
                .0
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                    Some((f64::from_bits(bits) + value).to_bits())
                });
        }
    }

    /// The sum.
    #[must_use]
    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

/// A value that goes up and down.
#[derive(Debug, Default)]
pub struct Gauge(AtomicU64);

impl Gauge {
    /// Sets the value.
    pub fn set(&self, value: u64) {
        self.0.store(value, Ordering::Relaxed);
    }

    /// Adds to the value.
    pub fn add(&self, value: u64) {
        self.0.fetch_add(value, Ordering::Relaxed);
    }

    /// Subtracts from the value.
    pub fn sub(&self, value: u64) {
        self.0.fetch_sub(value, Ordering::Relaxed);
    }

    /// Raises the value to `value`, if it is more.
    pub fn max(&self, value: u64) {
        self.0.fetch_max(value, Ordering::Relaxed);
    }

    /// The value.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Upper bounds of the histogram's buckets, in seconds, from a cached
/// decode's microseconds to a long fit's seconds.
const BUCKETS: [f64; 20] = [
    0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Durations, counted by bucket.
#[derive(Debug, Default)]
pub struct Histogram {
    /// Observations per bucket, the last for those above every bound.
    counts: [AtomicU64; BUCKETS.len() + 1],
    nanos: AtomicU64,
}

impl Histogram {
    /// Records one duration.
    pub fn observe(&self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        let bucket = BUCKETS
            .iter()
            .position(|&bound| seconds <= bound)
            .unwrap_or(BUCKETS.len());
        self.counts[bucket].fetch_add(1, Ordering::Relaxed);
        self.nanos.fetch_add(
            u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }
}

/// What the server reports for one table.
#[derive(Debug, Default)]
pub struct TableMetrics {
    pub rows_written: Counter,
    pub predictions: Counter,
    pub requests_failed: Counter,
    /// Page rows the worker embedded outside a fit, and page rows it copied
    /// from the page's earlier entry instead.
    pub rows_embedded: Counter,
    pub rows_copied: Counter,
    /// Time from a prediction request arriving to its answer.
    pub predict_seconds: Histogram,
    /// Worker time spent on the table, by the rows it took in each stage of
    /// each step; see the shard's `count_costs`.
    pub gpu_seconds: Seconds,
    /// The part of `gpu_seconds` in steps that answered none of the table's
    /// requests, as a region rebuild's will.
    pub rebuild_seconds: Seconds,
    /// Fits of the table's state that succeeded, one per epoch, the first
    /// included, the rows they read, and the worker's fit time they took.
    pub refits: Counter,
    pub refit_rows: Counter,
    pub refit_seconds: Seconds,
    /// Epochs the table started by refits, asked for or due, each leaving its
    /// encoded contexts behind.
    pub epochs: Counter,
    /// Bytes of the table's page index and of its rows in memory.
    pub index_bytes: Gauge,
    /// What its regions served and what rebuilding them cost, as the node
    /// counts them: queries read from a built context not due a rebuild,
    /// read from one due, waiting for one, and with a cutoff; contexts rebuilt and their
    /// rows; and the time rebuilt contexts spent due, in microseconds.
    pub region_hits: Gauge,
    pub region_stale: Gauge,
    pub region_misses: Gauge,
    pub region_fresh: Gauge,
    pub rebuilds: Gauge,
    pub rebuilt_rows: Gauge,
    pub rebuild_overdue_micros: Gauge,
}

/// A worker stage's compiled calls, the time they took, and their FLOPs.
#[derive(Debug, Default)]
pub struct Stage {
    pub calls: Counter,
    pub seconds: Seconds,
    pub flops: Total,
}

/// Everything the server reports.
#[derive(Debug, Default)]
pub struct Metrics {
    pub rows_written: Counter,
    pub predictions: Counter,
    pub requests_failed: Counter,
    pub steps: Counter,
    pub units_encoded: Counter,
    pub units_reused: Counter,
    pub units_transient: Counter,
    /// Context rows in units encoded, kept or not, and in units read from the cache.
    pub rows_encoded: Counter,
    pub rows_reused: Counter,
    /// Tables whose queries each step decodes, summed over steps.
    pub step_tables: Counter,
    /// Bytes cached contexts and pages hold, per tier: device, host and disk.
    pub cache_bytes: [Gauge; 3],
    /// Bytes of tables' rows in memory; the rest wait in sealed files.
    pub row_bytes: Gauge,
    pub tables: Gauge,
    /// Workers that stopped and are being started again.
    pub workers_down: Gauge,
    /// Tables whose fits kept failing, which take no queries until they change.
    pub tables_quarantined: Gauge,
    /// Tables out of service until the server restarts: their log refuses
    /// writes, or holds a record their rows could not take.
    pub tables_broken: Gauge,
    /// Request contexts' tables held in memory, across shards.
    pub ephemeral_tables: Gauge,
    pub step_seconds: Histogram,
    /// Time to plan a step, from the queue to the step sent to the worker.
    pub plan_seconds: Histogram,
    /// Time from a prediction request arriving to its step being sent.
    pub queue_seconds: Histogram,
    /// Time workers spent running steps, as they measure it.
    pub worker_seconds: Seconds,
    /// Programs workers compiled as they started, before taking steps, from
    /// the kinds of call earlier workers recorded, and the time it took.
    pub warm_compiles: Counter,
    pub warm_seconds: Seconds,
    /// Programs workers loaded from their persistent compile cache, and
    /// those they compiled because it did not hold them.
    pub compile_cache_hits: Counter,
    pub compile_cache_misses: Counter,
    /// How far the workers and the warm-up are through compiling programs
    /// as the server starts.
    pub compiling: crate::progress::Compiling,
    /// Where a prediction or write request's time goes in the service, beside
    /// the shard's queue and plan: decoding its Arrow body, finding its
    /// table's placement and route, waiting for the shard's answer, and
    /// encoding the reply.
    pub request_decode_seconds: Histogram,
    pub request_route_seconds: Histogram,
    pub request_shard_seconds: Histogram,
    pub request_encode_seconds: Histogram,
    /// Timing log lines dropped because the disk fell behind.
    pub timings_dropped: Counter,
    /// Tables moved between shards, and how long their calls waited for each move.
    pub moves: Counter,
    pub move_seconds: Histogram,
    /// The busiest shard's worker time over the mean at the last round, times 1000.
    pub placement_imbalance: Gauge,
    stages: Mutex<BTreeMap<String, Arc<Stage>>>,
    /// The parts of workers' steps outside their compiled calls, which split
    /// a step's time; see the worker's `WorkerStats.parts`.
    parts: Mutex<BTreeMap<String, Arc<Stage>>>,
    /// The parts of planning a step, which split `plan_seconds`; see the
    /// node's `PlanParts`.
    plan_parts: Mutex<BTreeMap<String, Arc<Stage>>>,
    /// The most each kind of memory held in any worker's step; see the
    /// worker's `generalforecast.profiling.memory`.
    memory_peaks: Mutex<BTreeMap<String, Arc<Gauge>>>,
    by_table: Mutex<BTreeMap<String, Arc<TableMetrics>>>,
    timings: Option<TimingLog>,
}

fn counter(text: &mut String, name: &str, help: &str, value: u64) {
    let _ = write!(
        text,
        "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"
    );
}

/// Reads one of a table's values.
type Value = fn(&TableMetrics) -> f64;

/// Writes a histogram's series, with `labels` (such as `table="x"`) on each.
fn histogram(text: &mut String, name: &str, labels: &str, histogram: &Histogram) {
    let (bucket, total) = if labels.is_empty() {
        (String::new(), String::new())
    } else {
        (format!("{labels},"), format!("{{{labels}}}"))
    };
    let mut cumulative = 0;
    for (bound, count) in BUCKETS.iter().zip(&histogram.counts) {
        cumulative += count.load(Ordering::Relaxed);
        let _ = writeln!(text, "{name}_bucket{{{bucket}le=\"{bound}\"}} {cumulative}");
    }
    cumulative += histogram.counts[BUCKETS.len()].load(Ordering::Relaxed);
    let seconds = histogram.nanos.load(Ordering::Relaxed) as f64 / 1e9;
    let _ = write!(
        text,
        "{name}_bucket{{{bucket}le=\"+Inf\"}} {cumulative}\n{name}_count{total} {cumulative}\n{name}_sum{total} {seconds}\n"
    );
}

impl Metrics {
    /// Metrics that also log each answered request's timings, appended to `path`.
    ///
    /// # Errors
    ///
    /// If the file cannot be opened.
    pub fn with_timing_log(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            timings: Some(TimingLog::open(path)?),
            ..Self::default()
        })
    }

    /// Logs a request's timings, if the server keeps a timing log; a line the
    /// log has no room for is dropped and counted rather than waited for.
    pub fn log_timing(&self, timing: impl FnOnce() -> Timing) {
        if let Some(log) = &self.timings
            && log.lines.try_send(timing()).is_err()
        {
            self.timings_dropped.add(1);
        }
    }

    /// A worker stage's metrics, made on first use.
    pub fn stage(&self, name: &str) -> Arc<Stage> {
        Self::named(&self.stages, name)
    }

    /// A step part's metrics, made on first use.
    pub fn part(&self, name: &str) -> Arc<Stage> {
        Self::named(&self.parts, name)
    }

    /// A plan part's metrics, made on first use.
    pub fn plan_part(&self, name: &str) -> Arc<Stage> {
        Self::named(&self.plan_parts, name)
    }

    /// Counts a step's memory of a kind, in bytes, towards the most any step held.
    pub fn memory_peak(&self, kind: &str, bytes: u64) {
        let peak = Arc::clone(
            self.memory_peaks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(kind.to_string())
                .or_default(),
        );
        peak.max(bytes);
    }

    fn named(map: &Mutex<BTreeMap<String, Arc<Stage>>>, name: &str) -> Arc<Stage> {
        let mut entries = map.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(entry) = entries.get(name) {
            return Arc::clone(entry);
        }
        Arc::clone(entries.entry(name.to_string()).or_default())
    }

    /// A table's metrics, made on first use.
    pub fn table(&self, name: &str) -> Arc<TableMetrics> {
        self.by_table
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(name.to_string())
            .or_default()
            .clone()
    }

    /// A table's metrics, if it has made any and was not dropped since.
    pub fn existing_table(&self, name: &str) -> Option<Arc<TableMetrics>> {
        self.by_table
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(name)
            .cloned()
    }

    /// Stops reporting a table, once it is dropped or released.
    pub fn forget(&self, name: &str) {
        self.by_table
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(name);
    }

    /// The metrics in Prometheus's text exposition format.
    #[must_use]
    pub fn render(&self) -> String {
        let mut text = String::new();
        counter(
            &mut text,
            "gf_rows_written_total",
            "Rows inserted, updated or deleted.",
            self.rows_written.get(),
        );
        counter(
            &mut text,
            "gf_predictions_total",
            "Query rows predicted.",
            self.predictions.get(),
        );
        counter(
            &mut text,
            "gf_requests_failed_total",
            "Requests that returned an error.",
            self.requests_failed.get(),
        );
        counter(
            &mut text,
            "gf_steps_total",
            "Steps run by workers.",
            self.steps.get(),
        );
        counter(
            &mut text,
            "gf_units_encoded_total",
            "Context units encoded and kept.",
            self.units_encoded.get(),
        );
        counter(
            &mut text,
            "gf_units_reused_total",
            "Context units read from the cache.",
            self.units_reused.get(),
        );
        counter(
            &mut text,
            "gf_units_transient_total",
            "Context units encoded without room to keep them.",
            self.units_transient.get(),
        );
        counter(
            &mut text,
            "gf_rows_encoded_total",
            "Context rows in units encoded, kept or not.",
            self.rows_encoded.get(),
        );
        counter(
            &mut text,
            "gf_rows_reused_total",
            "Context rows in units read from the cache.",
            self.rows_reused.get(),
        );
        counter(
            &mut text,
            "gf_step_tables_total",
            "Tables whose queries each step decodes, summed over steps.",
            self.step_tables.get(),
        );
        let _ = write!(
            text,
            "# HELP gf_cache_bytes Bytes cached contexts and pages hold.\n# TYPE gf_cache_bytes gauge\n"
        );
        for (tier, bytes) in ["device", "host", "disk"].iter().zip(&self.cache_bytes) {
            let _ = writeln!(text, "gf_cache_bytes{{tier=\"{tier}\"}} {}", bytes.get());
        }
        let _ = write!(
            text,
            "# HELP gf_row_bytes Bytes of tables' rows in memory.\n# TYPE gf_row_bytes gauge\ngf_row_bytes {}\n",
            self.row_bytes.get()
        );
        let _ = write!(
            text,
            "# HELP gf_tables Tables served.\n# TYPE gf_tables gauge\ngf_tables {}\n",
            self.tables.get()
        );
        let _ = write!(
            text,
            "# HELP gf_workers_down Workers being started again.\n# TYPE gf_workers_down gauge\ngf_workers_down {}\n",
            self.workers_down.get()
        );
        let _ = write!(
            text,
            "# HELP gf_tables_quarantined Tables whose fits kept failing, taking no queries until they change.\n# TYPE gf_tables_quarantined gauge\ngf_tables_quarantined {}\n",
            self.tables_quarantined.get()
        );
        let _ = write!(
            text,
            "# HELP gf_tables_broken Tables out of service until the server restarts.\n# TYPE gf_tables_broken gauge\ngf_tables_broken {}\n",
            self.tables_broken.get()
        );
        let _ = write!(
            text,
            "# HELP gf_ephemeral_tables Request contexts' tables held in memory.\n# TYPE gf_ephemeral_tables gauge\ngf_ephemeral_tables {}\n",
            self.ephemeral_tables.get()
        );
        let _ = write!(
            text,
            "# HELP gf_step_seconds Time a worker takes to run a step.\n# TYPE gf_step_seconds histogram\n"
        );
        histogram(&mut text, "gf_step_seconds", "", &self.step_seconds);
        let _ = write!(
            text,
            "# HELP gf_plan_seconds Time to plan a step.\n# TYPE gf_plan_seconds histogram\n"
        );
        histogram(&mut text, "gf_plan_seconds", "", &self.plan_seconds);
        let _ = write!(
            text,
            "# HELP gf_queue_seconds Time from a prediction request arriving to its step being sent.\n# TYPE gf_queue_seconds histogram\n"
        );
        histogram(&mut text, "gf_queue_seconds", "", &self.queue_seconds);
        let _ = write!(
            text,
            "# HELP gf_worker_seconds_total Time workers spent running steps, as they measure it.\n# TYPE gf_worker_seconds_total counter\ngf_worker_seconds_total {}\n",
            self.worker_seconds.get()
        );
        counter(
            &mut text,
            "gf_worker_warm_compiles_total",
            "Programs workers compiled as they started, before taking steps.",
            self.warm_compiles.get(),
        );
        let _ = write!(
            text,
            "# HELP gf_worker_warm_seconds_total Time workers spent compiling as they started.\n# TYPE gf_worker_warm_seconds_total counter\ngf_worker_warm_seconds_total {}\n",
            self.warm_seconds.get()
        );
        counter(
            &mut text,
            "gf_worker_compile_cache_hits_total",
            "Programs workers loaded from their persistent compile cache instead of compiling.",
            self.compile_cache_hits.get(),
        );
        counter(
            &mut text,
            "gf_worker_compile_cache_misses_total",
            "Programs workers compiled because their persistent compile cache did not hold them.",
            self.compile_cache_misses.get(),
        );
        let requests = [
            (
                "gf_request_decode_seconds",
                "Time decoding a request's Arrow body.",
                &self.request_decode_seconds,
            ),
            (
                "gf_request_route_seconds",
                "Time finding a request's table placement and route.",
                &self.request_route_seconds,
            ),
            (
                "gf_request_shard_seconds",
                "Time from a request reaching its shard to the shard's answer.",
                &self.request_shard_seconds,
            ),
            (
                "gf_request_encode_seconds",
                "Time encoding a request's reply.",
                &self.request_encode_seconds,
            ),
        ];
        for (name, help, values) in requests {
            let _ = write!(text, "# HELP {name} {help}\n# TYPE {name} histogram\n");
            histogram(&mut text, name, "", values);
        }
        counter(
            &mut text,
            "gf_timings_dropped_total",
            "Timing log lines dropped because the disk fell behind.",
            self.timings_dropped.get(),
        );
        counter(
            &mut text,
            "gf_moves_total",
            "Tables moved between shards.",
            self.moves.get(),
        );
        let _ = write!(
            text,
            "# HELP gf_move_seconds How long a table's calls waited for its move.\n# TYPE gf_move_seconds histogram\n"
        );
        histogram(&mut text, "gf_move_seconds", "", &self.move_seconds);
        let _ = write!(
            text,
            "# HELP gf_placement_imbalance The busiest shard's worker time over the mean at the last round.\n# TYPE gf_placement_imbalance gauge\ngf_placement_imbalance {}\n",
            self.placement_imbalance.get() as f64 / 1000.0
        );
        let stages = self
            .stages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let _ = write!(
            text,
            "# HELP gf_worker_stage_calls_total Compiled calls workers made, by stage.\n# TYPE gf_worker_stage_calls_total counter\n"
        );
        for (name, stage) in &stages {
            let _ = writeln!(
                text,
                "gf_worker_stage_calls_total{{stage=\"{name}\"}} {}",
                stage.calls.get()
            );
        }
        let _ = write!(
            text,
            "# HELP gf_worker_stage_seconds_total Time in workers' compiled calls, compiling included, by stage.\n# TYPE gf_worker_stage_seconds_total counter\n"
        );
        for (name, stage) in &stages {
            let _ = writeln!(
                text,
                "gf_worker_stage_seconds_total{{stage=\"{name}\"}} {}",
                stage.seconds.get()
            );
        }
        let _ = write!(
            text,
            "# HELP gf_worker_stage_flops_total Floating-point operations in workers' compiled calls, by stage, as XLA estimates each program; padding to a program's compiled shape counts.\n# TYPE gf_worker_stage_flops_total counter\n"
        );
        for (name, stage) in &stages {
            let _ = writeln!(
                text,
                "gf_worker_stage_flops_total{{stage=\"{name}\"}} {}",
                stage.flops.get()
            );
        }
        let parts = self
            .parts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let _ = write!(
            text,
            "# HELP gf_worker_part_calls_total Times each part of a step ran in workers, outside their compiled calls.\n# TYPE gf_worker_part_calls_total counter\n"
        );
        for (name, part) in &parts {
            let _ = writeln!(
                text,
                "gf_worker_part_calls_total{{part=\"{name}\"}} {}",
                part.calls.get()
            );
        }
        let _ = write!(
            text,
            "# HELP gf_worker_part_seconds_total Time in each part of workers' steps, which split the steps' seconds.\n# TYPE gf_worker_part_seconds_total counter\n"
        );
        for (name, part) in &parts {
            let _ = writeln!(
                text,
                "gf_worker_part_seconds_total{{part=\"{name}\"}} {}",
                part.seconds.get()
            );
        }
        let peaks = self
            .memory_peaks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let _ = write!(
            text,
            "# HELP gf_worker_memory_peak_bytes The most each kind of memory held in any worker's step.\n# TYPE gf_worker_memory_peak_bytes gauge\n"
        );
        for (kind, peak) in &peaks {
            let _ = writeln!(
                text,
                "gf_worker_memory_peak_bytes{{kind=\"{kind}\"}} {}",
                peak.get()
            );
        }
        let plan_parts = self
            .plan_parts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let _ = write!(
            text,
            "# HELP gf_plan_part_calls_total Times each part of planning a step ran.\n# TYPE gf_plan_part_calls_total counter\n"
        );
        for (name, part) in &plan_parts {
            let _ = writeln!(
                text,
                "gf_plan_part_calls_total{{part=\"{name}\"}} {}",
                part.calls.get()
            );
        }
        let _ = write!(
            text,
            "# HELP gf_plan_part_seconds_total Time in each part of planning steps, which split the plan seconds.\n# TYPE gf_plan_part_seconds_total counter\n"
        );
        for (name, part) in &plan_parts {
            let _ = writeln!(
                text,
                "gf_plan_part_seconds_total{{part=\"{name}\"}} {}",
                part.seconds.get()
            );
        }
        let tables = self
            .by_table
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let families: [(&str, &str, &str, Value); 19] = [
            (
                "gf_table_rows_written_total",
                "counter",
                "Rows inserted, updated or deleted, by table.",
                |table| table.rows_written.get() as f64,
            ),
            (
                "gf_table_rows_embedded_total",
                "counter",
                "Page rows embedded outside a fit, by table.",
                |table| table.rows_embedded.get() as f64,
            ),
            (
                "gf_table_rows_copied_total",
                "counter",
                "Page rows copied from the page's earlier entry instead of embedded, by table.",
                |table| table.rows_copied.get() as f64,
            ),
            (
                "gf_table_predictions_total",
                "counter",
                "Query rows predicted, by table.",
                |table| table.predictions.get() as f64,
            ),
            (
                "gf_table_requests_failed_total",
                "counter",
                "Requests that returned an error, by table.",
                |table| table.requests_failed.get() as f64,
            ),
            (
                "gf_table_gpu_seconds_total",
                "counter",
                "Worker time spent on a table, each stage's split by the rows it took.",
                |table| table.gpu_seconds.get(),
            ),
            (
                "gf_table_rebuild_seconds_total",
                "counter",
                "Worker time spent on a table in steps that answered none of its requests.",
                |table| table.rebuild_seconds.get(),
            ),
            (
                "gf_table_refits_total",
                "counter",
                "Fits of a table's state that succeeded, one per epoch, the first included.",
                |table| table.refits.get() as f64,
            ),
            (
                "gf_table_refit_rows_total",
                "counter",
                "Rows a table's successful fits read.",
                |table| table.refit_rows.get() as f64,
            ),
            (
                "gf_table_refit_seconds_total",
                "counter",
                "Worker time in a table's successful fits.",
                |table| table.refit_seconds.get(),
            ),
            (
                "gf_table_epochs_total",
                "counter",
                "Epochs a table started by refits.",
                |table| table.epochs.get() as f64,
            ),
            (
                "gf_table_index_bytes",
                "gauge",
                "Bytes of a table's page index and of its rows in memory.",
                |table| table.index_bytes.get() as f64,
            ),
            (
                "gf_table_region_hits_total",
                "counter",
                "Queries read from their region's built context, not due a rebuild.",
                |table| table.region_hits.get() as f64,
            ),
            (
                "gf_table_region_stale_total",
                "counter",
                "Queries read from their region's built context while it was due a rebuild.",
                |table| table.region_stale.get() as f64,
            ),
            (
                "gf_table_region_misses_total",
                "counter",
                "Queries that waited for their region's context to be encoded.",
                |table| table.region_misses.get() as f64,
            ),
            (
                "gf_table_region_fresh_total",
                "counter",
                "Queries with a cutoff, which read a context built for their window.",
                |table| table.region_fresh.get() as f64,
            ),
            (
                "gf_table_rebuilds_total",
                "counter",
                "Region contexts built again in rebuild steps.",
                |table| table.rebuilds.get() as f64,
            ),
            (
                "gf_table_rebuilt_rows_total",
                "counter",
                "Rows of region contexts built again.",
                |table| table.rebuilt_rows.get() as f64,
            ),
            (
                "gf_table_rebuild_overdue_seconds_total",
                "counter",
                "Time rebuilt region contexts spent due before their rebuild finished.",
                |table| table.rebuild_overdue_micros.get() as f64 / 1e6,
            ),
        ];
        for (name, kind, help, value) in families {
            let _ = write!(text, "# HELP {name} {help}\n# TYPE {name} {kind}\n");
            for (table, metrics) in &tables {
                let _ = writeln!(text, "{name}{{table=\"{table}\"}} {}", value(metrics));
            }
        }
        let _ = write!(
            text,
            "# HELP gf_table_predict_seconds Time from a prediction request to its answer, by table.\n# TYPE gf_table_predict_seconds histogram\n"
        );
        for (table, metrics) in &tables {
            let labels = format!("table=\"{table}\"");
            histogram(
                &mut text,
                "gf_table_predict_seconds",
                &labels,
                &metrics.predict_seconds,
            );
        }
        text
    }
}

/// One answered request's timings, as the timing log writes it.
#[derive(Clone, Debug)]
pub struct Timing {
    pub table: String,
    pub request: u64,
    pub trace: String,
    /// When the request reached its shard's queue.
    pub arrived: Instant,
    /// When the step holding it was planned, sent to the worker, started
    /// once the step before it finished, and answered.
    pub planned: Instant,
    pub sent: Instant,
    pub started: Instant,
    pub finished: Instant,
    pub queries: u64,
    /// Context rows the request newly encoded, with its share of those it
    /// shared with other requests.
    pub context_rows: f64,
    /// Its share of the step's worker time.
    pub gpu_seconds: f64,
}

/// Timing log lines that may wait for the disk; later ones are dropped.
const TIMINGS_WAITING: usize = 1 << 16;

/// A file of one JSON line per answered request, written on a thread of its
/// own so no step waits on the disk; lines are flushed each time the thread
/// catches up.
#[derive(Debug)]
struct TimingLog {
    lines: mpsc::SyncSender<Timing>,
}

impl TimingLog {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let (lines, received) = mpsc::sync_channel::<Timing>(TIMINGS_WAITING);
        // Instants mean nothing outside the process, so lines carry wall
        // clock microseconds, from one reading of both clocks.
        let (wall, start) = (SystemTime::now(), Instant::now());
        let base = wall.duration_since(UNIX_EPOCH).unwrap_or_default();
        let micros = move |at: Instant| {
            let since = base + at.saturating_duration_since(start);
            u64::try_from(since.as_micros()).unwrap_or(u64::MAX)
        };
        std::thread::Builder::new()
            .name("gf-timings".to_string())
            .spawn(move || {
                let mut out = std::io::BufWriter::new(file);
                while let Ok(first) = received.recv() {
                    let written = std::iter::once(first)
                        .chain(received.try_iter())
                        .try_for_each(|timing| {
                            let line = serde_json::json!({
                                "table": timing.table,
                                "request": timing.request,
                                "trace": timing.trace,
                                "arrived_us": micros(timing.arrived),
                                "planned_us": micros(timing.planned),
                                "sent_us": micros(timing.sent),
                                "started_us": micros(timing.started),
                                "finished_us": micros(timing.finished),
                                "queries": timing.queries,
                                "context_rows": timing.context_rows,
                                "gpu_seconds": timing.gpu_seconds,
                            });
                            writeln!(out, "{line}")
                        })
                        .and_then(|()| out.flush());
                    if let Err(error) = written {
                        tracing::error!("writing the timing log, which stops here: {error}");
                        return;
                    }
                }
            })?;
        Ok(Self { lines })
    }
}
