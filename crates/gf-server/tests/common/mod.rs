// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! What the tests of the server's APIs share: shards over a fake worker,
//! each logging under a directory of its own.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use gf_core::cache::Eviction;
use gf_core::node::Scheduling;
use gf_server::metrics::Metrics;
use gf_server::protocol::{
    CompileCache, Frame, ModelProfile, PredictionsHeader, WorkerReply, WorkerRequest, f32_bytes,
};
use gf_server::shard::{Command, EphemeralLimits, Shard, ShardSettings};
use gf_server::wal::EncryptionKey;
use gf_server::worker::WorkerConnection;
use serde_json::Value;
use tokio::net::UnixStream;
use tokio::sync::mpsc;

/// Answers profiles as a ten-class classifier, and every query with 0.25.
pub fn fake_worker(mut stream: UnixStream) {
    tokio::spawn(async move {
        while let Ok(frame) = Frame::read_from(&mut stream).await {
            let reply = if frame.kind() == "step" {
                let mut reply = Frame::header(Value::Null);
                let results: Vec<PredictionsHeader> = frame.header["queries"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|batch| {
                        let rows = batch["count"].as_u64().unwrap_or(0) as usize;
                        PredictionsHeader {
                            request: batch["request"].as_u64().unwrap_or(0),
                            rows,
                            columns: 10,
                            buffer: reply.push(f32_bytes(&vec![0.25; rows * 10])),
                            horizon: None,
                        }
                    })
                    .collect();
                reply.header = WorkerReply::Predictions {
                    results,
                    failures: Vec::new(),
                    deferred: Vec::new(),
                    seconds: 0.001,
                    stages: BTreeMap::new(),
                    parts: BTreeMap::new(),
                    memory: BTreeMap::new(),
                    compile_cache: CompileCache::default(),
                }
                .frame()
                .header;
                reply
            } else {
                match WorkerRequest::of(&frame).expect("the shard sends requests") {
                    WorkerRequest::Profile { .. } => WorkerReply::Profile(ModelProfile {
                        bytes_per_row: 64,
                        representation_bytes: 16,
                        regression: false,
                        class_slots: 10,
                        outputs: 10,
                        state_bytes: 32,
                        ..ModelProfile::default()
                    })
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

/// The directories `scratch` made, which the process removes as it exits.
static SCRATCH: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());

/// Removes every directory `scratch` made; run as the test process exits.
extern "C" fn remove_scratch() {
    let made = SCRATCH
        .lock()
        .map(|mut made| std::mem::take(&mut *made))
        .unwrap_or_default();
    for directory in made {
        let _ = std::fs::remove_dir_all(directory);
    }
}

/// A new directory under the system's temporary directory, named for
/// `name`, this process and a count, and removed with everything in it as
/// the test process exits, however its tests ended. Shards and files the
/// tests leave running keep it until then.
pub fn scratch(name: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    // SAFETY: atexit stores a function pointer, run once as the process exits.
    REGISTERED.call_once(|| unsafe {
        libc::atexit(remove_scratch);
    });
    let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!("gf-{name}-{}-{index}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    SCRATCH
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(directory.clone());
    directory
}

/// A directory of its own for each shard a test starts.
pub fn fresh_directory() -> std::path::PathBuf {
    scratch("http-test")
}

pub async fn shard() -> mpsc::Sender<Command> {
    shard_in(fresh_directory()).await
}

/// A shard over a fake worker, logging its tables under `directory`.
pub async fn shard_in(directory: std::path::PathBuf) -> mpsc::Sender<Command> {
    let (ours, theirs) = UnixStream::pair().unwrap();
    fake_worker(theirs);
    let connection = WorkerConnection::fake(ours, vec!["tiny".to_string()]);
    shard_over(connection, directory).await
}

/// A small shard's settings, its steps waiting a minute for their worker.
pub fn settings() -> ShardSettings {
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
        ephemeral: EphemeralLimits {
            tables: 4,
            bytes: 1 << 20,
            idle: Duration::from_secs(300),
        },
    }
}

/// A shard over `connection`, logging its tables under `directory`.
pub async fn shard_over(
    connection: WorkerConnection,
    directory: std::path::PathBuf,
) -> mpsc::Sender<Command> {
    let settings = settings();
    std::fs::create_dir_all(&directory).unwrap();
    let key = EncryptionKey::from_hex(&"ab".repeat(32)).unwrap();
    let mut shard = Shard::new(
        connection,
        settings,
        key,
        directory,
        Arc::new(Metrics::default()),
    )
    .unwrap();
    shard.allocate().await.unwrap();
    shard.spawn().unwrap()
}
