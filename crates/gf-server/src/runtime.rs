// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A server's shards, from its configuration and its hardware: one worker
//! per GPU, or on CPU one per run of `cpu_threads` cores, each behind its
//! own shard, which a request's context reaches by `routes::Routes`. The
//! server starts its shards so, and an engine embedding them does too.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use gf_core::node::Task;
use tokio::sync::{mpsc, oneshot, watch};

use crate::config::{Config, WorkerConfig};
use crate::engine;
use crate::error::ServerError;
use crate::metrics::Metrics;
use crate::name::TableName;
use crate::routes::Routes;
use crate::service::{Placement, SeriesPlacement};
use crate::setup;
use crate::shard::{Command, Shard, ShardSettings};
use crate::wal::{EncryptionKey, Keys};
use crate::worker::{
    CPU_WORKER_CORES, Launch, WorkerConnection, allowed_cpus, cpu_groups, visible_gpus,
};

/// Where a worker runs: its config, and the GPU or the cores it has to itself.
#[derive(Clone, Debug)]
pub struct Seat {
    pub worker: WorkerConfig,
    pub gpu: Option<String>,
    pub cpus: Option<Vec<usize>>,
}

/// A seat for each worker the hardware holds: one per visible GPU for a
/// `per_gpu` worker, else one per run of its `cpu_threads` allowed cores, or
/// of `CPU_WORKER_CORES` if it gives none, else one where the allowed cores
/// are unknown.
#[must_use]
pub fn seats(workers: &[WorkerConfig]) -> Vec<Seat> {
    let gpus = visible_gpus();
    let cpus = allowed_cpus();
    workers
        .iter()
        .flat_map(|worker| {
            let seat = |gpu: Option<String>, cpus: Option<Vec<usize>>| Seat {
                worker: worker.clone(),
                gpu,
                cpus,
            };
            if worker.per_gpu && !gpus.is_empty() {
                gpus.iter()
                    .map(|gpu| seat(Some(gpu.clone()), None))
                    .collect()
            } else if let Some(cpus) = &cpus {
                let threads = worker.cpu_threads.unwrap_or(CPU_WORKER_CORES);
                cpu_groups(cpus, threads)
                    .into_iter()
                    .map(|group| seat(None, Some(group)))
                    .collect()
            } else {
                vec![seat(None, None)]
            }
        })
        .collect::<Vec<_>>()
}

/// Starts a worker on each seat, all at once, and the shard that drives it,
/// with the configuration's memory, scheduling and ephemeral-table limits,
/// tables logged under `tables` and worker files under `cache`, and the
/// routes between them, which remember as many contexts as they hold.
///
/// # Errors
///
/// If a worker does not start or report ready, or a shard cannot allocate
/// its tiers.
pub async fn start(
    config: &Config,
    seats: &[Seat],
    key: &EncryptionKey,
    tables: &Path,
    cache: &Path,
    metrics: &Arc<Metrics>,
) -> Result<(Vec<mpsc::Sender<Command>>, Arc<Routes>), Box<dyn Error + Send + Sync>> {
    let routes = Arc::new(Routes::new(
        seats.len().max(1),
        config.ephemeral().tables.saturating_mul(seats.len()),
    ));
    // Each worker loads its models and compiles its recorded programs before
    // its shard takes steps, so the workers start side by side.
    let started = seats.iter().enumerate().map(|(index, seat)| {
        let routes = routes.clone();
        async move {
            let disk = (config.disk_bytes > 0).then(|| cache.join(format!("worker-{index}.slots")));
            let connection = WorkerConnection::spawn(Launch {
                index,
                workers: seats.len(),
                config: seat.worker.clone(),
                bucket_minimum: config.bucket_minimum,
                slot_bytes: config.slot_bytes,
                gpu: seat.gpu.clone(),
                cpus: seat.cpus.clone(),
                disk,
                secrets: config.key_envs(),
                shared_fallback: cache.to_path_buf(),
                timers: config.timers(),
                compiling: metrics.compiling.clone(),
            })
            .await?;
            let memory = ShardSettings {
                device_bytes: config
                    .device_bytes
                    .or_else(|| {
                        config
                            .device_fraction
                            .map(|fraction| (connection.device_memory as f64 * fraction) as u64)
                    })
                    .unwrap_or(connection.device_bytes),
                host_bytes: config.host_bytes.unwrap_or(connection.host_bytes),
                disk_bytes: config.disk_bytes,
                row_memory_bytes: config
                    .row_memory_bytes
                    .unwrap_or(connection.row_memory_bytes),
                slot_bytes: config.slot_bytes,
                bucket_minimum: config.bucket_minimum,
                step_bytes: config.step_bytes,
                scheduling: config.scheduling,
                eviction: config.eviction,
                step_timeout: Duration::from_secs_f64(config.step_timeout_seconds),
                queue_bytes: config.predict_queue_bytes,
                snapshot_bytes: 1 << 30,
                timers: config.timers(),
                ephemeral: config.ephemeral(),
            };
            tracing::info!(
                worker = index,
                gpu = seat.gpu.as_deref(),
                cpus = ?seat.cpus,
                models = ?connection.models,
                device_bytes = memory.device_bytes,
                host_bytes = memory.host_bytes,
                "worker ready"
            );
            let mut shard = Shard::new(
                connection,
                memory,
                key.clone(),
                tables.to_path_buf(),
                metrics.clone(),
            )?
            .with_routes(routes, index);
            shard.allocate().await?;
            Ok::<_, Box<dyn Error + Send + Sync>>(shard)
        }
    });
    let mut shards = Vec::with_capacity(seats.len());
    for shard in futures::future::try_join_all(started).await? {
        shards.push(shard.spawn()?);
    }
    Ok((shards, routes))
}

/// Serves every table logged under `tables` from its log, until
/// `interrupted`, and answers where the tables Flight serves went.
///
/// A table one of `bound`, the front's snapshot predictors, binds its rows
/// into is restored on the shard its fingerprint names, where binding
/// finds it to bind again, and Flight does not serve it. Every other table
/// is Flight's, as it was before.
///
/// # Errors
///
/// If a log is sealed under a key `keys` lacks, a shard stopped, or a log
/// sealed under a previous key cannot be rewritten under the current one.
pub async fn restore(
    shards: &[mpsc::Sender<Command>],
    tables: &Path,
    keys: &Keys,
    bound: Option<&HashMap<TableName, [u8; 32]>>,
    interrupted: &watch::Receiver<bool>,
) -> Result<HashMap<TableName, Placement>, Box<dyn Error + Send + Sync>> {
    let mut placement = HashMap::new();
    for (index, path) in setup::table_logs(tables)?.iter().enumerate() {
        if *interrupted.borrow() {
            break;
        }
        // A log under a key the server lacks refuses the start, rather than
        // leave its table out until a start that has the key.
        let Some(replay) =
            setup::open_log(path, keys).map_err(|error| format!("{}: {error}", path.display()))?
        else {
            set_aside_binding(path, bound);
            continue;
        };
        let definition = replay.definition().clone();
        let name = definition.name.clone();
        let rotate = replay.needs_rotation();
        let fingerprint = bound.and_then(|bound| bound.get(&name));
        let shard = fingerprint.map_or(index % shards.len(), |fingerprint| {
            engine::bound_shard(fingerprint, shards.len())
        });
        let restored = ask(&shards[shard], |reply| Command::Restore {
            replay: Box::new(replay),
            reply,
        })
        .await?;
        // A table that cannot be served, such as one whose model left the
        // configuration, keeps its log and leaves the others to start.
        let levels = match restored {
            Ok(levels) => levels,
            Err(error) => {
                tracing::error!(table = name.as_str(), "not serving a table: {error}");
                set_aside_binding(path, bound);
                continue;
            }
        };
        if rotate {
            // Rewrites the log under the current key before it takes writes.
            let table = name.clone();
            ask(&shards[shard], |reply| Command::Rotate { table, reply }).await??;
            tracing::info!(
                table = name.as_str(),
                "rewrote the log under the current key"
            );
        }
        tracing::info!(table = name.as_str(), shard, "restored");
        if fingerprint.is_none() {
            placement.insert(
                name,
                Placement {
                    shard,
                    task: Task::of_classes(definition.classes),
                    columns: Arc::new(definition.columns.clone()),
                    created: definition.created.clone(),
                    levels: levels.map(Arc::new),
                    series: SeriesPlacement::of(&definition),
                },
            );
        }
    }
    Ok(placement)
}

/// Sets aside the log at `path` if it holds one of `bound`, the front's
/// bindings, which did not restore: its rows are bound again from the
/// predictor's snapshot, which the log in place would refuse. Any other
/// table's log stays where it is, for an operator to look at.
fn set_aside_binding(path: &Path, bound: Option<&HashMap<TableName, [u8; 32]>>) {
    let binding = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| TableName::new(stem).ok())
        .is_some_and(|name| bound.is_some_and(|bound| bound.contains_key(&name)));
    if !binding || !path.exists() {
        return;
    }
    match crate::wal::set_aside(path, "unrestorable") {
        Ok(aside) => tracing::warn!(
            log = %path.display(),
            aside = %aside.display(),
            "set aside a predictor's binding that did not restore; it is bound again"
        ),
        Err(error) => tracing::error!(
            log = %path.display(),
            "setting aside a predictor's binding that did not restore: {error}"
        ),
    }
}

/// Sends `shard` a command and waits for its answer.
async fn ask<T>(
    shard: &mpsc::Sender<Command>,
    command: impl FnOnce(oneshot::Sender<Result<T, ServerError>>) -> Command,
) -> Result<Result<T, ServerError>, Box<dyn Error + Send + Sync>> {
    let (reply, answer) = oneshot::channel();
    shard
        .send(command(reply))
        .await
        .map_err(|_| "a shard stopped while the tables restored")?;
    Ok(answer
        .await
        .map_err(|_| "a shard stopped while the tables restored")?)
}
