//! A shard driven with a fake worker: table checks, quarantine, hung workers
//! and the query queue.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::time::Duration;

use gf_core::cache::Eviction;
use gf_core::node::{Scheduling, Targets};
use gf_server::columns::{Column, ColumnKind};
use gf_server::metrics::Metrics;
use gf_server::name::TableName;
use gf_server::protocol::{
    Failure, Frame, ModelProfile, PredictionsHeader, StageTime, WorkerReply, WorkerRequest,
    f32_bytes,
};
use gf_server::shard::{Command, Fence, Incoming, Lease, Shard, ShardSettings, Write};
use gf_server::wal::{
    CreationId, EncryptionKey, Keys, Log, LogEntry, Record, Rows, TableDefinition, Terms, TimeKind,
    decode,
};
use gf_server::worker::WorkerConnection;
use serde_json::Value;
use tokio::net::UnixStream;
use tokio::sync::{Notify, mpsc, oneshot};
use tonic::Code;

/// How the fake worker answers steps.
const OK: u8 = 0;
const FAIL: u8 = 1;
const HANG: u8 = 2;
/// Fails the first table's part of every step, and answers the rest.
const FAIL_FIRST: u8 = 3;
/// Reports progress every 100 ms for 600 ms, as compiles would, then answers.
const COMPILE: u8 = 4;

/// Answers every frame the shard sends, in order; steps as `behaviour` says.
fn fake_worker(mut stream: UnixStream, behaviour: Arc<AtomicU8>, held: Arc<Notify>) {
    tokio::spawn(async move {
        while let Ok(frame) = Frame::read_from(&mut stream).await {
            let reply = if frame.kind() == "step" {
                match behaviour.load(Ordering::Relaxed) {
                    FAIL => WorkerReply::Error {
                        message: "out of memory".to_string(),
                    }
                    .frame(),
                    HANG => continue,
                    FAIL_FIRST => predictions(&frame.header, Some(0)),
                    COMPILE => {
                        for _ in 0..6 {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            if WorkerReply::Progress
                                .frame()
                                .write_to(&mut stream)
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                        predictions(&frame.header, None)
                    }
                    _ => predictions(&frame.header, None),
                }
            } else {
                match WorkerRequest::of(&frame).expect("the shard sends requests") {
                    WorkerRequest::Profile { model, .. } if model == "tiny-forecaster" => {
                        WorkerReply::Profile(ModelProfile {
                            bytes_per_row: 64,
                            regression: true,
                            outputs: 3,
                            series: true,
                            patch: 4,
                            output_patch: 4,
                            context_limit: 64,
                            quantiles: Some(vec![0.1, 0.5, 0.9]),
                            reusable: true,
                            max_rows: Some(8),
                            ..ModelProfile::default()
                        })
                        .frame()
                    }
                    WorkerRequest::Profile { .. } => WorkerReply::Profile(ModelProfile {
                        bytes_per_row: 64,
                        representation_bytes: 16,
                        regression: false,
                        class_slots: 10,
                        outputs: 10,
                        context_rows: None,
                        state_bytes: 32,
                        categorical_state_bytes: 64,
                        column_summary: true,
                        ..ModelProfile::default()
                    })
                    .frame(),
                    // The fake model holds checks of six classes until released.
                    WorkerRequest::Check { classes: 6, .. } => {
                        held.notified().await;
                        WorkerReply::Ok.frame()
                    }
                    // The fake model hangs over tables of eight classes.
                    WorkerRequest::Check { classes: 8, .. } => continue,
                    // The fake model refuses tables of seven classes.
                    WorkerRequest::Check { classes: 7, .. } => WorkerReply::Invalid {
                        message: "ValueError: seven classes".to_string(),
                    }
                    .frame(),
                    _ => WorkerReply::Ok.frame(),
                }
            };
            if reply.write_to(&mut stream).await.is_err() {
                break;
            }
        }
    });
}

/// Zero probabilities over three classes for every query of a step, except
/// those of the `failing` table, which the reply reports as failed.
fn predictions(header: &Value, failing: Option<u32>) -> Frame {
    let mut frame = Frame::header(Value::Null);
    let results: Vec<PredictionsHeader> = header["queries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|batch| batch["table"].as_u64() != failing.map(u64::from))
        .map(|batch| {
            let rows = batch["count"].as_u64().unwrap_or(0) as usize;
            PredictionsHeader {
                request: batch["request"].as_u64().unwrap_or(0),
                rows,
                columns: 3,
                buffer: frame.push(f32_bytes(&vec![0.0; rows * 3])),
                horizon: None,
            }
        })
        .collect();
    // A forecast request gets zeros for each of its series over its horizon,
    // three quantiles each.
    let mut forecasts: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
    for forecast in header["forecasts"].as_array().into_iter().flatten() {
        let entry = forecasts
            .entry(forecast["request"].as_u64().unwrap_or(0))
            .or_insert((0, forecast["horizon"].as_u64().unwrap_or(1) as usize));
        entry.0 += 1;
    }
    let mut results = results;
    for (request, (rows, horizon)) in forecasts {
        results.push(PredictionsHeader {
            request,
            rows,
            columns: 3,
            buffer: frame.push(f32_bytes(&vec![0.0; rows * horizon * 3])),
            horizon: Some(horizon),
        });
    }
    let failures = failing
        .map(|table| Failure {
            table,
            message: "ValueError: bad rows".to_string(),
        })
        .into_iter()
        .collect();
    // Every step takes the worker 4 ms: 1 ms decoding, and 2 ms fitting if it fits.
    let mut stages = BTreeMap::from([(
        "decode".to_string(),
        StageTime {
            calls: 1,
            seconds: 0.001,
        },
    )]);
    if header["fits"]
        .as_array()
        .is_some_and(|fits| !fits.is_empty())
    {
        let fit = StageTime {
            calls: 1,
            seconds: 0.002,
        };
        stages.insert("fit".to_string(), fit);
    }
    // The rest of the step, 1 ms, is the part `other`.
    frame.header = WorkerReply::Predictions {
        results,
        failures,
        seconds: 0.004,
        stages,
        parts: BTreeMap::from([(
            "other".to_string(),
            StageTime {
                calls: 1,
                seconds: 0.001,
            },
        )]),
    }
    .frame()
    .header;
    frame
}

struct Harness {
    commands: mpsc::Sender<Command>,
    metrics: Arc<Metrics>,
    behaviour: Arc<AtomicU8>,
    held: Arc<Notify>,
    directory: std::path::PathBuf,
}

fn name(text: &str) -> TableName {
    TableName::new(text).unwrap()
}

fn key() -> EncryptionKey {
    EncryptionKey::from_hex(&"ab".repeat(32)).unwrap()
}

async fn shard(settings: ShardSettings) -> Harness {
    static SHARDS: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "gf-shard-test-{}-{}",
        std::process::id(),
        SHARDS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    shard_in(settings, directory).await
}

/// A shard over a directory, which may hold the logs of an earlier one.
async fn shard_in(settings: ShardSettings, directory: std::path::PathBuf) -> Harness {
    let (ours, theirs) = UnixStream::pair().unwrap();
    let behaviour = Arc::new(AtomicU8::new(OK));
    let held = Arc::new(Notify::new());
    fake_worker(theirs, behaviour.clone(), held.clone());
    let connection = WorkerConnection::fake(
        ours,
        vec!["tiny".to_string(), "tiny-forecaster".to_string()],
    )
    .unwrap();
    let metrics = Arc::new(Metrics::default());
    let mut shard = Shard::new(
        connection,
        settings,
        key(),
        directory.clone(),
        metrics.clone(),
    )
    .unwrap();
    shard.allocate().await.unwrap();
    Harness {
        commands: shard.spawn().unwrap(),
        metrics,
        behaviour,
        held,
        directory,
    }
}

fn settings() -> ShardSettings {
    ShardSettings {
        device_bytes: 64 << 20,
        host_bytes: 64 << 20,
        row_memory_bytes: 64 << 20,
        disk_bytes: 0,
        slot_bytes: 1 << 20,
        bucket_minimum: 8,
        step_bytes: None,
        scheduling: Scheduling::Turns,
        eviction: Eviction::Gdsf,
        step_timeout: Duration::from_secs(60),
        queue_bytes: 1 << 20,
        snapshot_bytes: 1 << 30,
        timers: false,
    }
}

fn definition(name: &str, classes: u32) -> TableDefinition {
    TableDefinition {
        name: self::name(name),
        model: "tiny".to_string(),
        columns: vec![
            Column {
                name: "amount".to_string(),
                kind: ColumnKind::Numeric,
            },
            Column {
                name: "store".to_string(),
                kind: ColumnKind::Categorical,
            },
        ],
        classes,
        strategy: Default::default(),
        page_size: 8,
        time_bucket: None,
        refit_after: None,
        refit_shift: None,
        past_cap: Default::default(),
        regions: None,
        series: None,
        refit_cutoffs: "each".to_string(),
        retrieval_categoricals: "codes".to_string(),
        scale_rows: None,
        fit_rows: None,
        summary: Default::default(),
        created: CreationId::parse("0123456789abcdef"),
        times: TimeKind::Integer,
        route_by: None,
    }
}

impl Harness {
    async fn create(&self, classes: u32) -> Result<(), Code> {
        self.create_named("rows", classes).await
    }

    async fn create_named(&self, name: &str, classes: u32) -> Result<(), Code> {
        let (reply, answer) = oneshot::channel();
        let command = Command::Create {
            definition: definition(name, classes),
            terms: Terms::default(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer
            .await
            .unwrap()
            .map(|_| ())
            .map_err(|error| status(&error))
    }

    async fn insert(&self, keys: &[u128]) {
        self.insert_stores(keys, None).await;
    }

    /// Inserts rows whose store column holds `stores`, or the number 0.
    async fn insert_stores(&self, keys: &[u128], stores: Option<&[&str]>) {
        self.insert_into("rows", keys, stores).await;
    }

    async fn insert_into(&self, table: &str, keys: &[u128], stores: Option<&[&str]>) {
        self.write(table, keys, stores, None).await.unwrap();
    }

    /// Inserts rows under a client's write id; returns how many rows it added.
    async fn write(
        &self,
        table: &str,
        keys: &[u128],
        stores: Option<&[&str]>,
        id: Option<&str>,
    ) -> Result<usize, Code> {
        let (reply, answer) = oneshot::channel();
        let rows = Rows {
            keys: keys.to_vec(),
            features: keys.iter().flat_map(|&key| [key as f32, 0.0]).collect(),
            targets: Targets::Classes(keys.iter().map(|&key| (key % 3) as i32).collect()),
            times: None,
            id: id.map(str::to_string),
            words: Vec::new(),
            writes: Vec::new(),
            layout: None,
            table: None,
        };
        let strings = match stores {
            Some(stores) => vec![(
                1,
                stores.iter().map(|store| Some(store.to_string())).collect(),
            )],
            None => Vec::new(),
        };
        let command = Command::Write {
            table: name(table),
            write: Write::Insert(Incoming { rows, strings }),
            fence: Fence::default(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer
            .await
            .unwrap()
            .map(|written| written.rows)
            .map_err(|error| status(&error))
    }

    async fn delete(
        &self,
        keys: &[u128],
        missing_ok: bool,
        id: Option<&str>,
    ) -> Result<usize, Code> {
        let (reply, answer) = oneshot::channel();
        let command = Command::Write {
            table: name("rows"),
            write: Write::Delete {
                keys: keys.to_vec(),
                missing_ok,
                id: id.map(str::to_string),
            },
            fence: Fence::default(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer
            .await
            .unwrap()
            .map(|written| written.rows)
            .map_err(|error| status(&error))
    }

    async fn compact(&self) {
        let (reply, answer) = oneshot::channel();
        let command = Command::Compact {
            table: name("rows"),
            fence: Fence::default(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer.await.unwrap().unwrap();
    }

    /// Stops this shard and starts another over its directory, as a restart does.
    async fn restart(self) -> Harness {
        self.save_heads().await;
        drop(self.commands);
        let again = shard_in(settings(), self.directory.clone()).await;
        again.restore().await.unwrap();
        again
    }

    async fn save_heads(&self) {
        let (reply, answer) = oneshot::channel();
        self.commands
            .send(Command::SaveHeads { reply })
            .await
            .unwrap();
        answer.await.unwrap().unwrap();
    }

    /// Restores the table `rows` from its log.
    async fn restore(&self) -> Result<(), Code> {
        let replay = Log::open(&self.directory.join("rows.log"), &Keys::only(key())).unwrap();
        let (reply, answer) = oneshot::channel();
        let command = Command::Restore {
            replay: Box::new(replay),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer
            .await
            .unwrap()
            .map(|_| ())
            .map_err(|error| status(&error))
    }

    /// Applies log entries shipped from another copy.
    async fn replicate(&self, entries: Vec<(u64, Vec<u8>)>) -> Result<u64, Code> {
        let (reply, answer) = oneshot::channel();
        let command = Command::Replicate {
            table: name("rows"),
            entries,
            lineage: Box::default(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer.await.unwrap().map_err(|error| status(&error))
    }

    /// The table's log entries from `from` on.
    async fn read_log(&self, from: u64) -> Vec<(u64, Vec<u8>)> {
        let (reply, answer) = oneshot::channel();
        let command = Command::ReadLog {
            table: name("rows"),
            from,
            limit: 100,
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer.await.unwrap().unwrap()
    }

    async fn describe(&self) -> gf_server::shard::Description {
        let (reply, answer) = oneshot::channel();
        let command = Command::Describe {
            table: name("rows"),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer.await.unwrap().unwrap()
    }

    async fn predict(&self) -> Result<usize, Code> {
        self.predict_from("rows").await
    }

    async fn predict_from(&self, table: &str) -> Result<usize, Code> {
        self.send_predict(table).await.await
    }

    /// Sends a prediction, and returns its answer to await later.
    async fn send_predict(&self, table: &str) -> impl Future<Output = Result<usize, Code>> {
        let (reply, answer) = oneshot::channel();
        let command = Command::Predict {
            table: name(table),
            features: vec![0.5, 0.5],
            strings: Vec::new(),
            before: None,
            given: None,
            trace: String::new(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        async move {
            answer
                .await
                .unwrap()
                .map(|predictions| predictions.rows)
                .map_err(|error| status(&error))
        }
    }
}

/// The Flight status code a shard error maps to, as the service maps it.
fn status(error: &gf_server::shard::ServerError) -> Code {
    use gf_server::shard::ServerError;
    match error {
        ServerError::NotFound(_) => Code::NotFound,
        ServerError::Exists(_) => Code::AlreadyExists,
        ServerError::Invalid(_) => Code::InvalidArgument,
        ServerError::Gap(_) => Code::OutOfRange,
        ServerError::Fenced(_) => Code::FailedPrecondition,
        ServerError::Busy(_) => Code::ResourceExhausted,
        ServerError::Unavailable(_) => Code::Unavailable,
        ServerError::Internal(_) => Code::Internal,
    }
}

#[tokio::test]
async fn tables_are_checked_against_their_model_when_created() {
    let shard = shard(settings()).await;

    assert_eq!(shard.create(0).await, Err(Code::InvalidArgument));
    assert_eq!(shard.create(1).await, Err(Code::InvalidArgument));
    assert_eq!(shard.create(99).await, Err(Code::InvalidArgument));
    assert_eq!(shard.create(7).await, Err(Code::InvalidArgument));
    assert_eq!(shard.create(3).await, Ok(()));
}

#[tokio::test]
async fn a_table_whose_fit_keeps_failing_is_quarantined_until_it_changes() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;
    shard.behaviour.store(FAIL, Ordering::Relaxed);

    for _ in 0..3 {
        assert_eq!(shard.predict().await, Err(Code::Internal));
    }
    let refused = shard.predict().await;
    let quarantined = shard.metrics.tables_quarantined.get();
    shard.behaviour.store(OK, Ordering::Relaxed);
    shard.insert(&[5]).await;
    let after = shard.predict().await;

    assert_eq!(refused, Err(Code::InvalidArgument));
    assert_eq!(quarantined, 1);
    assert_eq!(after, Ok(1));
    assert_eq!(shard.metrics.tables_quarantined.get(), 0);
}

#[tokio::test]
async fn one_tables_failures_leave_the_tables_that_share_its_steps_alone() {
    let shard = shard(settings()).await;
    shard.create_named("bad", 3).await.unwrap();
    shard.create_named("good", 3).await.unwrap();
    shard.insert_into("bad", &[1, 2, 3, 4], None).await;
    shard.insert_into("good", &[1, 2, 3, 4], None).await;
    shard.behaviour.store(FAIL_FIRST, Ordering::Relaxed);

    // The shard takes the commands that queue while it handles one as a
    // batch, so queries sent while a held check blocks it share one step.
    for round in 0..3 {
        let (reply, created) = oneshot::channel();
        let command = Command::Create {
            definition: definition(&format!("held{round}"), 6),
            terms: Terms::default(),
            reply,
        };
        shard.commands.send(command).await.unwrap();
        let bad = shard.send_predict("bad").await;
        let good = shard.send_predict("good").await;
        shard.held.notify_one();
        assert!(created.await.unwrap().is_ok());
        assert_eq!(bad.await, Err(Code::Internal));
        assert_eq!(good.await, Ok(1));
    }
    assert_eq!(shard.metrics.steps.get(), 3);
    assert_eq!(shard.metrics.step_tables.get(), 6);

    assert_eq!(shard.predict_from("bad").await, Err(Code::InvalidArgument));
    assert_eq!(shard.predict_from("good").await, Ok(1));
    assert_eq!(shard.metrics.tables_quarantined.get(), 1);
}

#[tokio::test]
async fn a_shard_without_cache_memory_encodes_every_step_and_still_batches_tables() {
    let shard = shard(ShardSettings {
        device_bytes: 0,
        host_bytes: 0,
        ..settings()
    })
    .await;
    shard.create_named("first", 3).await.unwrap();
    shard.create_named("second", 3).await.unwrap();
    shard.insert_into("first", &[1, 2, 3, 4], None).await;
    shard.insert_into("second", &[1, 2, 3, 4], None).await;

    // The queries usually share each step; the shard runs on a thread of its
    // own, so under load it may take one before the other arrives.
    for _ in 0..2 {
        let answers = tokio::join!(shard.predict_from("first"), shard.predict_from("second"));
        assert_eq!(answers, (Ok(1), Ok(1)));
    }

    assert!((2..=4).contains(&shard.metrics.steps.get()));
    assert_eq!(shard.metrics.step_tables.get(), 4);
    assert_eq!(shard.metrics.units_reused.get(), 0);
    assert_eq!(shard.metrics.units_transient.get(), 4);
    assert_eq!(shard.metrics.rows_encoded.get(), 16);
    assert_eq!(shard.metrics.cache_bytes[0].get(), 0);
}

#[tokio::test]
async fn a_shard_without_timers_counts_no_plan_part() {
    let shard = shard(settings()).await;
    shard.create_named("rows", 3).await.unwrap();
    shard.insert_into("rows", &[1, 2, 3, 4], None).await;

    assert_eq!(shard.predict().await, Ok(1));

    assert!(shard.metrics.steps.get() > 0);
    let text = shard.metrics.render();
    assert!(text.contains("# TYPE gf_plan_part_calls_total counter\n"));
    assert!(!text.contains("gf_plan_part_calls_total{"));
}

#[tokio::test]
async fn a_shard_reports_the_rows_it_reuses_and_the_cache_they_take() {
    let shard = shard(settings()).await;
    shard.create_named("rows", 3).await.unwrap();
    shard.insert_into("rows", &[1, 2, 3, 4], None).await;

    for _ in 0..2 {
        assert_eq!(shard.predict().await, Ok(1));
    }

    assert_eq!(shard.metrics.rows_encoded.get(), 4);
    assert_eq!(shard.metrics.rows_reused.get(), 4);
    assert!(shard.metrics.cache_bytes[0].get() > 0);
}

#[tokio::test]
async fn a_shard_splits_the_workers_time_among_tables_and_counts_their_refits() {
    let shard = shard(ShardSettings {
        timers: true,
        ..settings()
    })
    .await;
    shard.create_named("first", 3).await.unwrap();
    shard.create_named("second", 3).await.unwrap();
    shard.insert_into("first", &[1, 2, 3, 4], None).await;
    shard.insert_into("second", &[1, 2], None).await;

    let answers = tokio::join!(shard.predict_from("first"), shard.predict_from("second"));
    assert_eq!(answers, (Ok(1), Ok(1)));
    let (reply, answer) = oneshot::channel();
    let refit = Command::Refit {
        table: name("first"),
        fence: Fence::default(),
        reply,
    };
    shard.commands.send(refit).await.unwrap();
    answer.await.unwrap().unwrap();
    assert_eq!(shard.predict_from("first").await, Ok(1));

    let metrics = &shard.metrics;
    let (first, second) = (metrics.table("first"), metrics.table("second"));
    let close = |left: f64, right: f64| (left - right).abs() < 1e-8;
    assert!(close(
        metrics.worker_seconds.get(),
        0.004 * metrics.steps.get() as f64
    ));
    assert!(close(
        first.gpu_seconds.get() + second.gpu_seconds.get(),
        metrics.worker_seconds.get()
    ));
    assert_eq!(metrics.stage("decode").calls.get(), metrics.steps.get());
    assert_eq!(metrics.part("other").calls.get(), metrics.steps.get());
    assert!(close(
        metrics.part("other").seconds.get(),
        0.001 * metrics.steps.get() as f64
    ));
    // Every step was planned, which checked its tables and placed its units.
    assert!(metrics.plan_part("checks").calls.get() >= metrics.steps.get());
    assert!(metrics.plan_part("units").calls.get() >= metrics.steps.get());
    assert_eq!((first.refits.get(), first.refit_rows.get()), (2, 8));
    assert_eq!((second.refits.get(), second.refit_rows.get()), (1, 2));
    assert!(close(
        first.refit_seconds.get() + second.refit_seconds.get(),
        metrics.stage("fit").seconds.get()
    ));
    // Every step answered both tables' requests, so none was a rebuild.
    assert_eq!(
        first.rebuild_seconds.get() + second.rebuild_seconds.get(),
        0.0
    );
    assert_eq!((first.epochs.get(), second.epochs.get()), (1, 0));
    assert!(first.index_bytes.get() > 0 && second.index_bytes.get() > 0);
}

#[tokio::test]
async fn a_hung_worker_is_started_again_and_its_requests_fail() {
    let mut settings = settings();
    settings.step_timeout = Duration::from_millis(200);
    let shard = shard(settings).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;
    shard.behaviour.store(HANG, Ordering::Relaxed);

    let failed = tokio::time::timeout(Duration::from_secs(10), shard.predict()).await;

    assert_eq!(
        failed.expect("answered before the timeout"),
        Err(Code::Internal)
    );
    assert_eq!(shard.metrics.workers_down.get(), 1);
}

#[tokio::test]
async fn a_worker_that_reports_progress_outlasts_the_step_timeout() {
    let mut settings = settings();
    settings.step_timeout = Duration::from_millis(200);
    let shard = shard(settings).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;
    shard.behaviour.store(COMPILE, Ordering::Relaxed);

    let answered = tokio::time::timeout(Duration::from_secs(10), shard.predict()).await;

    assert_eq!(answered.expect("answered before the timeout"), Ok(1));
    assert_eq!(shard.metrics.workers_down.get(), 0);
}

#[tokio::test]
async fn a_worker_that_hangs_outside_a_step_is_started_again() {
    let mut settings = settings();
    settings.step_timeout = Duration::from_millis(200);
    let shard = shard(settings).await;

    let failed = tokio::time::timeout(Duration::from_secs(10), shard.create(8)).await;

    assert_eq!(
        failed.expect("answered before the timeout"),
        Err(Code::Internal)
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(shard.metrics.workers_down.get(), 1);
}

#[tokio::test]
async fn queries_past_the_queue_budget_are_refused() {
    let mut settings = settings();
    settings.queue_bytes = 4;
    let shard = shard(settings).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;

    assert_eq!(shard.predict().await, Err(Code::ResourceExhausted));
}

#[tokio::test]
async fn strings_take_codes_in_order_of_arrival_and_the_log_keeps_them() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    shard
        .insert_stores(&[1, 2, 3], Some(&["north", "south", "north"]))
        .await;
    let first = shard.describe().await.vocabularies;
    shard.insert_stores(&[4, 5], Some(&["south", "east"])).await;
    let second = shard.describe().await.vocabularies;

    assert_eq!(
        first,
        [Vec::<String>::new(), vec!["north".into(), "south".into()]]
    );
    assert_eq!(second[1], ["north", "south", "east"]);
    // Each record carries only the words it added.
    let words: Vec<Vec<String>> = shard
        .read_log(1)
        .await
        .iter()
        .map(|(_, bytes)| match decode(bytes).unwrap() {
            LogEntry::Record(record) => record
                .words()
                .iter()
                .flat_map(|words| words.values.clone())
                .collect(),
            LogEntry::Definition(..) => unreachable!("read from 1"),
        })
        .collect();
    assert_eq!(words, [vec!["north", "south"], vec!["east"]]);

    // A compacted log and a restart keep the codes.
    shard.compact().await;
    let again = shard.restart().await;
    let restored = again.describe().await;

    assert_eq!(restored.rows, 5);
    assert_eq!(restored.vocabularies[1], ["north", "south", "east"]);
}

#[tokio::test]
async fn a_retried_write_applies_once_after_compaction_and_a_restart() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    assert_eq!(
        shard.write("rows", &[1, 2], None, Some("first")).await,
        Ok(2)
    );

    shard.compact().await;
    let again = shard.restart().await;

    // The retry finds the write the snapshot remembers, rather than failing
    // on keys the table holds.
    assert_eq!(
        again.write("rows", &[1, 2], None, Some("first")).await,
        Ok(2)
    );
    assert_eq!(again.describe().await.rows, 2);
    assert_eq!(
        again.write("rows", &[1, 2], None, Some("second")).await,
        Err(Code::InvalidArgument)
    );
}

#[tokio::test]
async fn a_retried_delete_with_missing_ok_answers_as_its_first_try_did() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3]).await;
    assert_eq!(shard.delete(&[1, 2], true, Some("gone")).await, Ok(2));

    // The retry finds the write before it skips the keys the table no
    // longer holds, so it reports the rows the first try deleted, and it
    // does not delete a key written again since.
    shard.insert(&[1]).await;
    assert_eq!(shard.delete(&[1, 2], true, Some("gone")).await, Ok(2));
    assert_eq!(shard.describe().await.rows, 2);
    assert_eq!(shard.delete(&[1, 2], true, Some("later")).await, Ok(1));
    assert_eq!(shard.delete(&[1, 2], true, None).await, Ok(0));
}

#[tokio::test]
async fn writes_go_on_while_the_worker_starts_again() {
    let mut settings = settings();
    settings.step_timeout = Duration::from_millis(200);
    let shard = shard(settings).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;
    shard.behaviour.store(HANG, Ordering::Relaxed);
    assert_eq!(shard.predict().await, Err(Code::Internal));

    // The fake worker never starts again, yet the log and rows take writes.
    let written = tokio::time::timeout(Duration::from_secs(5), shard.insert(&[5, 6])).await;
    assert!(written.is_ok(), "a write waited for the worker");
    assert_eq!(shard.describe().await.rows, 6);
    assert_eq!(shard.create_named("other", 3).await, Err(Code::Unavailable));
    assert_eq!(shard.metrics.workers_down.get(), 1);
}

#[tokio::test]
async fn an_abandoned_prediction_leaves_the_queue() {
    let mut settings = settings();
    // Room for one query of two features.
    settings.queue_bytes = 8;
    let shard = shard(settings).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;
    shard.behaviour.store(HANG, Ordering::Relaxed);

    // A caller gives up on a query the worker never answers.
    let abandoned = tokio::time::timeout(Duration::from_millis(100), shard.predict()).await;
    assert!(abandoned.is_err());

    // The next query takes its place rather than being refused.
    let next = tokio::time::timeout(Duration::from_millis(300), shard.predict()).await;
    assert!(next.is_err(), "refused: {next:?}");
}

#[tokio::test]
async fn a_table_whose_log_fails_to_replay_is_served_nowhere() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3]).await;
    shard.insert(&[4, 5]).await;
    shard.save_heads().await;
    let directory = shard.directory.clone();
    drop(shard.commands);
    // Damage the first record, which the head covers.
    let path = directory.join("rows.log");
    let mut bytes = std::fs::read(&path).unwrap();
    let first = 16 + 8 + u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    bytes[first + 40] ^= 1;
    std::fs::write(&path, &bytes).unwrap();

    let again = shard_in(settings(), directory.clone()).await;
    assert_eq!(again.restore().await, Err(Code::Internal));

    // Nothing serves the name, and its damaged log is set aside.
    assert_eq!(again.predict().await, Err(Code::NotFound));
    assert!(!path.exists());
    assert!(directory.join("rows.log.unreadable").exists());
    again.create(3).await.unwrap();
}

#[tokio::test]
async fn a_shipment_that_fails_midway_applies_what_it_staged() {
    let primary = shard(settings()).await;
    primary.create(3).await.unwrap();
    primary.insert(&[1, 2]).await;
    primary.insert(&[3]).await;
    let mut entries = primary.read_log(1).await;
    let replica = shard(settings()).await;
    replica.create(3).await.unwrap();

    // The second entry arrives damaged, after the first was staged.
    entries[1].1 = b"damaged".to_vec();
    assert!(replica.replicate(entries.clone()).await.is_err());
    let described = replica.describe().await;
    assert_eq!((described.rows, described.log_length), (2, 2));

    // The whole shipment, sent again, applies the rest once.
    let whole = primary.read_log(1).await;
    assert_eq!(replica.replicate(whole).await, Ok(3));
    assert_eq!(replica.describe().await.rows, 3);
}

#[tokio::test(start_paused = true)]
async fn a_lease_is_renewed_only_for_the_router_holding_it_until_it_lapses() {
    let lease = Lease::lapsed();
    assert!(lease.check().is_err());
    lease.renew("first", Duration::from_secs(60)).unwrap();

    tokio::time::advance(Duration::from_secs(59)).await;
    assert_eq!(
        lease.renew("second", Duration::from_secs(60)),
        Err("first".to_string())
    );
    assert!(lease.renew("first", Duration::from_secs(60)).is_ok());
    assert!(lease.check().is_ok());
    // Once it ends, writes are fenced and another router may take it.
    tokio::time::advance(Duration::from_secs(60)).await;
    assert!(lease.check().is_err());
    assert!(lease.renew("second", Duration::from_secs(60)).is_ok());
}

#[tokio::test]
async fn a_node_that_needs_leases_takes_writes_only_under_a_term() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    let lease = Lease::lapsed();
    lease.renew("router", Duration::from_secs(60)).unwrap();

    let write = |term: Option<u64>| {
        let (reply, answer) = oneshot::channel();
        let rows = Rows {
            keys: vec![1],
            features: vec![1.0, 0.0],
            targets: Targets::Classes(vec![0]),
            times: None,
            id: None,
            words: Vec::new(),
            writes: Vec::new(),
            layout: None,
            table: None,
        };
        let command = Command::Write {
            table: name("rows"),
            write: Write::Insert(rows.into()),
            fence: Fence {
                lease: Some(lease.clone()),
                term,
            },
            reply,
        };
        (command, answer)
    };
    // A router that has not learned the table's term may be behind a promotion.
    let (command, answer) = write(None);
    shard.commands.send(command).await.unwrap();
    assert_eq!(
        answer.await.unwrap().map_err(|error| status(&error)).err(),
        Some(Code::FailedPrecondition)
    );
    let (command, answer) = write(Some(0));
    shard.commands.send(command).await.unwrap();
    assert!(answer.await.unwrap().is_ok());
}

#[tokio::test]
async fn a_truncate_empties_the_table_and_later_writes_in_its_batch_follow_it() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3]).await;

    // Sent together, so they may commit in one batch: the insert of key 1 must
    // see the table the truncate leaves, not the rows it removes.
    let (reply, truncated) = oneshot::channel();
    let truncate = Command::Write {
        table: name("rows"),
        write: Write::Truncate,
        fence: Fence::default(),
        reply,
    };
    let (reply, inserted) = oneshot::channel();
    let insert = Command::Write {
        table: name("rows"),
        write: Write::Insert(
            Rows {
                keys: vec![1],
                features: vec![1.0, 0.0],
                targets: Targets::Classes(vec![1]),
                times: None,
                id: None,
                words: Vec::new(),
                writes: Vec::new(),
                layout: None,
                table: None,
            }
            .into(),
        ),
        fence: Fence::default(),
        reply,
    };
    shard.commands.send(truncate).await.unwrap();
    shard.commands.send(insert).await.unwrap();
    assert_eq!(truncated.await.unwrap().map(|written| written.rows), Ok(3));
    assert_eq!(inserted.await.unwrap().map(|written| written.rows), Ok(1));

    let again = shard.restart().await;
    assert_eq!(again.describe().await.rows, 1);
}

#[tokio::test]
async fn a_compaction_that_cannot_write_its_log_leaves_the_table_as_it_was() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;
    let log = shard.directory.join("rows.log");
    let before = std::fs::read(&log).unwrap();
    // The rewrite goes to a file beside the log; a directory in its place
    // fails it before anything is written.
    let blocking = shard.directory.join("rows.compacting");
    std::fs::create_dir(&blocking).unwrap();

    let (reply, answer) = oneshot::channel();
    let command = Command::Compact {
        table: name("rows"),
        fence: Fence::default(),
        reply,
    };
    shard.commands.send(command).await.unwrap();
    assert!(answer.await.unwrap().is_err());

    // The log, its rows and its row order are as they were, so a restart
    // reads what the shard serves, and the next compaction goes through.
    assert_eq!(std::fs::read(&log).unwrap(), before);
    assert_eq!(shard.describe().await.rows, 4);
    shard.insert(&[5]).await;
    assert_eq!(shard.predict().await, Ok(1));
    std::fs::remove_dir(&blocking).unwrap();
    shard.compact().await;
    let again = shard.restart().await;
    assert_eq!(again.describe().await.rows, 5);
    assert_eq!(again.predict().await, Ok(1));
}

#[tokio::test]
async fn a_large_table_compacts_into_records_that_each_fit_an_entry() {
    let mut settings = settings();
    // A row at a time, so the snapshot is one row and the rest follow it.
    settings.snapshot_bytes = 1;
    let shard = shard(settings).await;
    shard.create(3).await.unwrap();
    shard.insert(&[5, 1, 4, 2, 3]).await;
    shard.insert(&[6]).await;
    let before = shard.predict().await;

    shard.compact().await;

    // After the definition: the snapshot, then the other rows as inserts, a
    // row each, in page order.
    let records: Vec<Record> = shard
        .read_log(1)
        .await
        .iter()
        .filter_map(|(_, bytes)| match decode(bytes).unwrap() {
            LogEntry::Record(record) => Some(record),
            LogEntry::Definition(..) => None,
        })
        .collect();
    let keys: Vec<Vec<u128>> = records
        .iter()
        .map(|record| match record {
            Record::Snapshot(rows) | Record::Insert(rows) => rows.keys.clone(),
            other => panic!("unexpected record {other:?}"),
        })
        .collect();
    assert!(matches!(records[0], Record::Snapshot(_)));
    assert!(keys.iter().all(|keys| keys.len() == 1));
    let mut every: Vec<u128> = keys.concat();
    every.sort_unstable();
    assert_eq!(every, [1, 2, 3, 4, 5, 6]);

    // A restart and a replica read the same rows from them.
    let again = shard.restart().await;
    assert_eq!(again.describe().await.rows, 6);
    assert_eq!(again.predict().await, before);
    let replica = self::shard(self::settings()).await;
    replica.create(3).await.unwrap();
    assert_eq!(replica.replicate(again.read_log(0).await).await, Ok(9));
    assert_eq!(replica.describe().await.rows, 6);
    assert_eq!(replica.predict().await, before);
}

#[tokio::test]
async fn a_compaction_keeps_the_tables_rows_and_caches() {
    let shard = shard(settings()).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;
    assert_eq!(shard.predict().await, Ok(1));
    let encoded = shard.metrics.units_encoded.get();

    shard.compact().await;

    assert_eq!(shard.predict().await, Ok(1));
    assert_eq!(shard.metrics.units_encoded.get(), encoded);
    assert_eq!(shard.metrics.units_reused.get(), 1);
}

#[tokio::test]
async fn writes_during_a_compaction_wait_for_it_and_then_apply() {
    let mut settings = settings();
    // A row a record, so the rewrite takes several records.
    settings.snapshot_bytes = 1;
    let shard = shard(settings).await;
    shard.create(3).await.unwrap();
    shard.insert(&[1, 2, 3, 4]).await;

    // Queued together: the insert waits for the rewrite, then follows it.
    tokio::join!(shard.compact(), shard.insert(&[5]));

    assert_eq!(shard.describe().await.rows, 5);
    let keys: Vec<u128> = shard
        .read_log(1)
        .await
        .iter()
        .filter_map(|(_, bytes)| match decode(bytes).unwrap() {
            LogEntry::Record(Record::Snapshot(rows) | Record::Insert(rows)) => {
                Some(rows.keys.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .concat();
    assert_eq!(keys.last(), Some(&5));
    let again = shard.restart().await;
    assert_eq!(again.describe().await.rows, 5);
}

/// A table of two series of one covariate, forecast in pairs.
fn series_definition(name: &str) -> TableDefinition {
    use gf_server::wal::{GroupSpec, SeriesDefinition};

    TableDefinition {
        model: "tiny-forecaster".to_string(),
        columns: vec![Column {
            name: "price".to_string(),
            kind: ColumnKind::Numeric,
        }],
        classes: 0,
        series: Some(Box::new(SeriesDefinition {
            series: "item".to_string(),
            time: "t".to_string(),
            target: "units".to_string(),
            step: 10,
            known: Vec::new(),
            group: GroupSpec::Random { size: 2, seed: 1 },
            horizon: Some(3),
        })),
        ..definition(name, 0)
    }
}

impl Harness {
    async fn create_series(&self, name: &str) -> Result<Option<Vec<f64>>, Code> {
        let (reply, answer) = oneshot::channel();
        let command = Command::Create {
            definition: series_definition(name),
            terms: Terms::default(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer.await.unwrap().map_err(|error| status(&error))
    }

    /// Inserts observations of series at times, each row's key packing both.
    async fn observe(&self, table: &str, placed: &[(u64, i64)]) -> Result<usize, Code> {
        let (reply, answer) = oneshot::channel();
        let rows = Rows {
            keys: placed
                .iter()
                .map(|&(series, time)| (u128::from(series) << 64) | time as u64 as u128)
                .collect(),
            features: placed
                .iter()
                .map(|&(_, time)| time as f32 / 100.0)
                .collect(),
            targets: Targets::Values(placed.iter().map(|&(series, _)| series as f32).collect()),
            times: Some(placed.iter().map(|&(_, time)| Some(time)).collect()),
            id: None,
            words: Vec::new(),
            writes: Vec::new(),
            layout: None,
            table: None,
        };
        let command = Command::Write {
            table: name(table),
            write: Write::Insert(Incoming {
                rows,
                strings: Vec::new(),
            }),
            fence: Fence::default(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer
            .await
            .unwrap()
            .map(|written| written.rows)
            .map_err(|error| status(&error))
    }

    async fn forecast(
        &self,
        table: &str,
        series: &[u64],
        before: Option<Vec<i64>>,
    ) -> Result<gf_server::protocol::Predictions, Code> {
        self.forecast_over(table, series, before, 3).await
    }

    async fn forecast_over(
        &self,
        table: &str,
        series: &[u64],
        before: Option<Vec<i64>>,
        horizon: usize,
    ) -> Result<gf_server::protocol::Predictions, Code> {
        let (reply, answer) = oneshot::channel();
        let command = Command::Forecast {
            table: name(table),
            series: series.to_vec(),
            before,
            horizon: std::num::NonZeroUsize::new(horizon).unwrap(),
            groups: None,
            trace: String::new(),
            reply,
        };
        self.commands.send(command).await.unwrap();
        answer.await.unwrap().map_err(|error| status(&error))
    }
}

#[tokio::test]
async fn series_tables_forecast_from_their_observations_and_after_a_restart() {
    let shard = shard(settings()).await;
    // A forecaster needs a series definition, and a tabular model refuses one.
    assert_eq!(
        shard.create_series("rows").await,
        Ok(Some(vec![0.1, 0.5, 0.9]))
    );
    let (reply, answer) = oneshot::channel();
    let tabular = Command::Create {
        definition: TableDefinition {
            model: "tiny-forecaster".to_string(),
            classes: 0,
            ..definition("plain", 0)
        },
        terms: Terms::default(),
        reply,
    };
    shard.commands.send(tabular).await.unwrap();
    assert_eq!(
        answer.await.unwrap().map_err(|error| status(&error)),
        Err(Code::InvalidArgument)
    );
    let placed: Vec<(u64, i64)> = (0..6)
        .flat_map(|step| [(1, 100 + 10 * step), (2, 100 + 10 * step)])
        .collect();
    assert_eq!(shard.observe("rows", &placed).await, Ok(12));
    // A row off its series' grid is refused before the log takes it.
    assert_eq!(
        shard.observe("rows", &[(1, 165)]).await,
        Err(Code::InvalidArgument)
    );

    let forecast = shard.forecast("rows", &[1, 2], None).await.unwrap();

    assert_eq!(
        (forecast.rows, forecast.horizon, forecast.columns),
        (2, Some(3), 3)
    );
    // Each series is forecast from the step after its latest observation.
    assert_eq!(forecast.cutoffs, vec![160, 160]);
    assert_eq!(forecast.values.len(), 2 * 3 * 3);

    // The log replays the observations and their grids.
    let shard = shard.restart().await;
    let again = shard.forecast("rows", &[2], Some(vec![140])).await.unwrap();
    assert_eq!((again.rows, again.cutoffs), (1, vec![140]));
    assert_eq!(
        shard.observe("rows", &[(1, 165)]).await,
        Err(Code::InvalidArgument)
    );
}

#[tokio::test]
async fn a_series_table_compacted_and_restarted_forecasts_as_before() {
    let shard = shard(settings()).await;
    shard.create_series("rows").await.unwrap();
    let placed: Vec<(u64, i64)> = (0..6)
        .flat_map(|step| [(1, 100 + 10 * step), (2, 100 + 10 * step)])
        .collect();
    shard.observe("rows", &placed).await.unwrap();
    let before = shard.forecast("rows", &[1, 2], None).await.unwrap();

    shard.compact().await;
    let shard = shard.restart().await;

    let after = shard.forecast("rows", &[1, 2], None).await.unwrap();
    assert_eq!(
        (after.cutoffs.clone(), after.values),
        (vec![160, 160], before.values)
    );
    // A series the table does not hold is forecast cold from its group, given
    // a cutoff; a cutoff nothing in the group is observed before is refused.
    let cold = shard.forecast("rows", &[3], Some(vec![160])).await.unwrap();
    assert_eq!((cold.rows, cold.cutoffs), (1, vec![160]));
    assert_eq!(
        shard.forecast("rows", &[3], None).await,
        Err(Code::InvalidArgument)
    );
    assert_eq!(
        shard.forecast("rows", &[1], Some(vec![100])).await,
        Err(Code::InvalidArgument)
    );
    // The model's output length caps the horizon at 64 passes.
    assert!(shard.forecast_over("rows", &[1], None, 256).await.is_ok());
    assert_eq!(
        shard.forecast_over("rows", &[1], None, 257).await,
        Err(Code::InvalidArgument)
    );
}
