// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `gf-server run --config generalforecast.toml`: serves tables over Arrow Flight.
//! `init` writes a starter configuration, and `check` looks at everything a
//! configuration needs before starting.

use std::collections::HashMap;
use std::error::Error;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use clap::Parser;
use gf_core::node::Task;
use gf_server::auth::Tokens;
use gf_server::config::{Config, WorkerConfig};
use gf_server::metrics::Metrics;
use gf_server::service::{Placement, SeriesPlacement, Service};
use gf_server::setup;
use gf_server::shard::{Command, Shard, ShardSettings};
use gf_server::worker::{Launch, Private, WorkerConnection, shared_parent, visible_gpus};
use tokio::sync::{oneshot, watch};
use tonic::transport::{Certificate, Identity, ServerTlsConfig};

#[derive(Parser)]
#[command(about = "Serve tabular foundation models over Arrow Flight")]
struct Arguments {
    #[command(subcommand)]
    command: Subcommand,
}

#[derive(clap::Subcommand)]
enum Subcommand {
    /// Serves the tables a configuration names.
    Run {
        /// The configuration file.
        #[arg(long)]
        config: PathBuf,
    },
    /// Writes a starter configuration for one server on this machine.
    Init {
        /// Where to write it.
        #[arg(long, default_value = "generalforecast.toml")]
        path: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8815")]
        listen: String,
        /// Where table logs go.
        #[arg(long, default_value = "gf-data")]
        data: PathBuf,
        /// The Python that runs the workers, with generalforecast installed.
        #[arg(long, default_value = "python")]
        python: String,
        /// A model to serve, as NAME=SPEC; the built-in TabICL models if none.
        #[arg(long = "model", value_name = "NAME=SPEC")]
        models: Vec<String>,
        /// The client token, which `GF_TOKEN` may give instead, so it stays
        /// out of the process list; a random one if neither is set.
        #[arg(long)]
        token: Option<String>,
        /// A file holding the log encryption key; `GF_LOG_KEY` otherwise.
        #[arg(long)]
        key_file: Option<PathBuf>,
        /// Overwrite the file if it exists.
        #[arg(long)]
        force: bool,
    },
    /// Checks a configuration's keys, files, workers and table logs without starting.
    Check {
        /// The configuration file.
        #[arg(long)]
        config: PathBuf,
    },
}

/// What the HTTP endpoint reads: the metrics, and whether the server is draining.
#[derive(Clone)]
struct Status {
    metrics: Arc<Metrics>,
    draining: Arc<AtomicBool>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    match Arguments::parse().command {
        Subcommand::Run { config } => run(&config).await,
        Subcommand::Init {
            path,
            listen,
            data,
            python,
            models,
            token,
            key_file,
            force,
        } => init(
            &path,
            setup::Starter {
                listen,
                data,
                python,
                models: Vec::new(),
                token: String::new(),
                key_file,
            },
            &models,
            token,
            force,
        ),
        Subcommand::Check { config: path } => {
            let config = Config::load(&path)?;
            let report = setup::check(&config);
            for line in &report.lines {
                println!("{line}");
            }
            for warning in config.warnings(&path) {
                println!("warn   {warning}");
            }
            if report.failed {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

/// Writes a starter configuration, and says how to go on.
fn init(
    path: &std::path::Path,
    mut starter: setup::Starter,
    models: &[String],
    token: Option<String>,
    force: bool,
) -> Result<(), Box<dyn Error>> {
    if path.exists() && !force {
        return Err(format!("{} exists; pass --force to overwrite it", path.display()).into());
    }
    let models = models
        .iter()
        .map(|entry| {
            entry
                .split_once('=')
                .map(|(name, spec)| (name.to_string(), spec.to_string()))
                .ok_or_else(|| format!("--model takes NAME=SPEC, not {entry:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let given = token.or_else(|| std::env::var("GF_TOKEN").ok());
    let random = given.is_none();
    let token = match given {
        Some(token) => token,
        None => setup::random_hex(16)?,
    };
    starter.models = models;
    starter.token.clone_from(&token);
    setup::write_private(path, setup::starter_config(&starter).as_bytes())?;
    if random && std::io::stdout().is_terminal() {
        println!("wrote {}; the client token is {token}", path.display());
    } else {
        println!("wrote {}, which holds the client token", path.display());
    }
    let mut steps = Vec::new();
    if starter.key_file.is_none() {
        steps.push("export GF_LOG_KEY=$(openssl rand -hex 32)".to_string());
    }
    steps.push(format!("gf-server check --config {}", path.display()));
    steps.push(format!("gf-server run --config {}", path.display()));
    for (index, step) in steps.iter().enumerate() {
        println!("{}{step}", if index == 0 { "next: " } else { "      " });
    }
    Ok(())
}

async fn run(config: &std::path::Path) -> Result<(), Box<dyn Error>> {
    gf_server::trace::init_logging();
    gf_server::trace::abort_on_panic();
    let path = config;
    let config = Config::load(path)?;
    for warning in config.warnings(path) {
        tracing::warn!("{warning}");
    }
    let keys = setup::keys(&config)?;
    let key = keys.current.clone();
    let tables = config.data.join("tables");
    std::fs::create_dir_all(&tables)?;
    let cache = config.data.join("cache");
    std::fs::create_dir_all(&cache)?;
    // Servers killed outright leave their worker directories behind.
    Private::remove_stale(&shared_parent());
    Private::remove_stale(&std::env::temp_dir());
    Private::remove_stale(&cache);
    let metrics = Arc::new(match &config.timing_log {
        Some(path) => Metrics::with_timing_log(path)
            .map_err(|error| format!("the timing log {}: {error}", path.display()))?,
        None => Metrics::default(),
    });

    let gpus = visible_gpus();
    let launches: Vec<(&WorkerConfig, Option<&str>)> = config
        .workers
        .iter()
        .flat_map(|worker| {
            if worker.per_gpu && !gpus.is_empty() {
                gpus.iter()
                    .map(|gpu| (worker, Some(gpu.as_str())))
                    .collect()
            } else {
                vec![(worker, None)]
            }
        })
        .collect();
    let mut shards = Vec::new();
    for (index, (worker, gpu)) in launches.iter().enumerate() {
        let disk = (config.disk_bytes > 0).then(|| cache.join(format!("worker-{index}.slots")));
        let connection = WorkerConnection::spawn(Launch {
            index,
            workers: launches.len(),
            config: (*worker).clone(),
            bucket_minimum: config.bucket_minimum,
            slot_bytes: config.slot_bytes,
            gpu: gpu.map(str::to_string),
            disk,
            secrets: config.key_envs(),
            shared_fallback: cache.clone(),
            timers: config.timers(),
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
        };
        tracing::info!(
            worker = index,
            gpu,
            models = ?connection.models,
            device_bytes = memory.device_bytes,
            host_bytes = memory.host_bytes,
            "worker ready"
        );
        let mut shard = Shard::new(
            connection,
            memory,
            key.clone(),
            tables.clone(),
            metrics.clone(),
        )?;
        shard.allocate().await?;
        shards.push(shard.spawn()?);
    }

    let mut placement = HashMap::new();
    for (index, path) in setup::table_logs(&tables)?.iter().enumerate() {
        // A log under a key the server lacks refuses the start, rather than
        // leave its table out until a start that has the key.
        let Some(replay) =
            setup::open_log(path, &keys).map_err(|error| format!("{}: {error}", path.display()))?
        else {
            continue;
        };
        let definition = replay.definition().clone();
        let shard = index % shards.len();
        let name = definition.name.clone();
        let rotate = replay.needs_rotation();
        let (reply, answer) = oneshot::channel();
        shards[shard]
            .send(Command::Restore {
                replay: Box::new(replay),
                reply,
            })
            .await?;
        // A table that cannot be served, such as one whose model left the
        // configuration, keeps its log and leaves the others to start.
        let levels = match answer.await? {
            Ok(levels) => levels,
            Err(error) => {
                tracing::error!(table = name.as_str(), "not serving a table: {error}");
                continue;
            }
        };
        let placed = Placement {
            shard,
            task: Task::of_classes(definition.classes),
            columns: Arc::new(definition.columns.clone()),
            created: definition.created.clone(),
            levels: levels.map(Arc::new),
            series: SeriesPlacement::of(&definition),
        };
        if rotate {
            // Rewrites the log under the current key before it takes writes.
            let (reply, answer) = oneshot::channel();
            let table = name.clone();
            shards[shard].send(Command::Rotate { table, reply }).await?;
            answer.await??;
            tracing::info!(
                table = name.as_str(),
                "rewrote the log under the current key"
            );
        }
        tracing::info!(table = name.as_str(), shard, "restored");
        placement.insert(name, placed);
    }

    // One signal starts the drain for both servers; calls still open after
    // the drain deadline are cut off, since every write is already on disk.
    let (stop, stopped) = watch::channel(false);
    let draining = Arc::new(AtomicBool::new(false));
    let drain = Duration::from_secs_f64(config.drain_seconds);
    tokio::spawn({
        let draining = draining.clone();
        async move {
            shutdown().await;
            draining.store(true, Ordering::SeqCst);
            let _ = stop.send(true);
            tokio::time::sleep(drain).await;
            tracing::warn!(
                "calls still open after {} seconds; exiting",
                drain.as_secs()
            );
            std::process::exit(0);
        }
    });

    if let Some(address) = &config.metrics_listen {
        let listener = tokio::net::TcpListener::bind(address).await?;
        let status = Status {
            metrics: metrics.clone(),
            draining: draining.clone(),
        };
        let router = Router::new()
            .route(
                "/metrics",
                get(|State(status): State<Status>| async move { status.metrics.render() }),
            )
            .route("/healthz", get(|| async { "ok" }))
            .route(
                "/readyz",
                get(|State(status): State<Status>| async move {
                    if status.draining.load(Ordering::SeqCst) {
                        (StatusCode::SERVICE_UNAVAILABLE, "draining")
                    } else if status.metrics.workers_down.get() > 0 {
                        (StatusCode::SERVICE_UNAVAILABLE, "restarting a worker")
                    } else {
                        (StatusCode::OK, "ready")
                    }
                }),
            )
            .with_state(status);
        let mut stopped = stopped.clone();
        tokio::spawn(async move {
            let shutdown = async move {
                let _ = stopped.wait_for(|stop| *stop).await;
            };
            if let Err(error) = axum::serve(listener, router)
                .with_graceful_shutdown(shutdown)
                .await
            {
                tracing::error!("metrics endpoint: {error}");
            }
        });
    }

    let address: std::net::SocketAddr = config.listen.parse()?;
    let tokens = Tokens::new(&config.tokens);
    let replication_tokens = Tokens::new(&config.replication_tokens);
    let mut server = tonic::transport::Server::builder()
        .timeout(Duration::from_secs_f64(config.request_timeout_seconds))
        .concurrency_limit_per_connection(config.connection_concurrency)
        .http2_keepalive_interval(Some(Duration::from_secs(20)))
        .http2_keepalive_timeout(Some(Duration::from_secs(10)))
        .tcp_keepalive(Some(Duration::from_secs(60)));
    if let Some(tls) = &config.tls {
        let identity =
            Identity::from_pem(std::fs::read(&tls.certificate)?, std::fs::read(&tls.key)?);
        let mut tls_config = ServerTlsConfig::new().identity(identity);
        if let Some(client_ca) = &tls.client_ca {
            tls_config =
                tls_config.client_ca_root(Certificate::from_pem(std::fs::read(client_ca)?));
        }
        server = server.tls_config(tls_config)?;
    } else if !address.ip().is_loopback() {
        tracing::warn!("serving without TLS beyond this machine, as allow_plaintext says");
    }
    tracing::info!(%address, tls = config.tls.is_some(), "serving");
    let mut service = Service::new(shards.clone(), placement, metrics)
        .with_request_budget(config.request_bytes_in_flight)
        .with_max_tables(config.max_tables)
        .with_profiling(
            config
                .profiling
                .as_ref()
                .map(|profiling| profiling.directory.clone()),
        )
        .with_placement(&config.placement, tables.clone(), keys.clone());
    if config.require_lease {
        service = service.requiring_leases();
    }
    if config.placement.assign == gf_server::config::Assign::Load
        && config.placement.rebalance_seconds > 0.0
    {
        let every = Duration::from_secs_f64(config.placement.rebalance_seconds);
        tokio::spawn(service.clone().rebalance(every));
    }
    let mut stopped = stopped.clone();
    server
        .add_service(service.into_server(tokens, replication_tokens))
        .serve_with_shutdown(address, async move {
            let _ = stopped.wait_for(|stop| *stop).await;
        })
        .await?;
    // Every write is synced already; saving the heads lets the next start
    // check the log's whole length.
    for shard in &shards {
        let (reply, answer) = oneshot::channel();
        if shard.send(Command::SaveHeads { reply }).await.is_ok() {
            let _ = answer.await;
        }
    }
    tracing::info!("drained");
    Ok(())
}

/// Resolves on Ctrl-C or SIGTERM.
async fn shutdown() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("signal handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
    tracing::info!("draining");
}
