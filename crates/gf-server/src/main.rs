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
use gf_server::config::{Config, HttpConfig, KeysConfig};
use gf_server::http::{Front, Introspect, KeyStore, Principal, ServedModel, Static};
use gf_server::metrics::Metrics;
use gf_server::revisions::{Limits as RevisionLimits, Revisions};
use gf_server::runtime;
use gf_server::service::{Placement, RequestBudget, SeriesPlacement, Service};
use gf_server::setup;
use gf_server::shard::Command;
use gf_server::upstream::{Cidr, Upstream};
use gf_server::worker::{Private, shared_parent};
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
    // One signal starts the drain for every listener; calls still open after
    // the drain deadline are cut off, since every write is already on disk.
    let (stop, stopped) = watch::channel(false);
    let draining = Arc::new(AtomicBool::new(false));
    // The Models API listens before the workers load, so liveness answers
    // and readiness says when inference starts.
    let front = match &config.http {
        Some(http) => Some(listen_http(http, &config, draining.clone(), stopped.clone()).await?),
        None => None,
    };
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

    let seats = runtime::seats(&config.workers);
    let (shards, routes) = runtime::start(&config, &seats, &key, &tables, &cache, &metrics)
        .await
        .map_err(|error| -> Box<dyn Error> { error })?;

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
    if let Some((front, url, _)) = &front {
        front.start(shards.clone(), routes.clone()).await?;
        event(&serde_json::json!({"event": "ready", "url": url}));
    }
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
    if let Some((_, url, served)) = front {
        let _ = served.await;
        event(&serde_json::json!({"event": "stopped", "url": url}));
    }
    tracing::info!("drained");
    Ok(())
}

/// Writes one NDJSON event on stdout, as `gf serve` reports its progress.
fn event(value: &serde_json::Value) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{value}");
    let _ = stdout.flush();
}

/// Binds the Models API, says it is serving, and answers on its own task
/// until the drain; the front starts inference once the shards are up.
async fn listen_http(
    http: &HttpConfig,
    config: &Config,
    draining: Arc<AtomicBool>,
    mut stopped: watch::Receiver<bool>,
) -> Result<(Arc<Front>, String, tokio::task::JoinHandle<()>), Box<dyn Error>> {
    let text = std::fs::read_to_string(&http.models)
        .map_err(|error| format!("{}: {error}", http.models.display()))?;
    let models: Vec<ServedModel> = serde_json::from_str(&text)
        .map_err(|error| format!("{}: {error}", http.models.display()))?;
    let listener = tokio::net::TcpListener::bind(&http.listen)
        .await
        .map_err(|error| format!("cannot listen on {}: {error}", http.listen))?;
    let address = listener.local_addr()?;
    let keys: Box<dyn KeyStore> = match &http.keys {
        KeysConfig::Static {
            env,
            project,
            permissions,
        } => {
            let key = std::env::var(env).ok().filter(|key| !key.is_empty());
            if key.is_none() && !address.ip().is_loopback() {
                eprintln!(
                    "gf serve: warning: listening on {}:{} without an API key; anyone who can \
                     reach this address can run inference. Set {env} to require a bearer key.",
                    address.ip(),
                    address.port()
                );
            }
            Box::new(Static::scoped(
                key.as_deref(),
                Principal {
                    project: project.clone(),
                    permissions: permissions.clone(),
                },
            ))
        }
        KeysConfig::Introspect { url, token_env } => {
            let token = std::env::var(token_env)
                .map_err(|_| format!("set {token_env} to the control plane's service token"))?;
            Box::new(Introspect::new(url, token)?)
        }
    };
    let mut front = Front::new(
        models,
        keys,
        http.surface,
        address.ip().is_loopback(),
        RequestBudget::new(config.request_bytes_in_flight),
        Duration::from_secs_f64(http.request_timeout_seconds),
        draining,
    );
    if let Some(url) = &http.upstream {
        let trusted = http
            .trusted_proxies
            .iter()
            .map(|range| range.parse::<Cidr>())
            .collect::<Result<Vec<_>, _>>()?;
        let timeout = Duration::try_from_secs_f64(http.upstream_timeout_seconds)
            .map_err(|_| "upstream_timeout_seconds must be a positive number of seconds")?;
        front = front.with_upstream(Upstream::new(
            url,
            timeout,
            http.upstream_body_bytes,
            trusted,
        )?);
    }
    if let Some(revisions) = &http.revisions {
        let token = std::env::var(&revisions.token_env).map_err(|_| {
            format!(
                "set {} to the control plane's service token",
                revisions.token_env
            )
        })?;
        let seconds = |value: f64, field: &str| {
            Duration::try_from_secs_f64(value)
                .ok()
                .filter(|duration| !duration.is_zero())
                .ok_or_else(|| format!("revisions.{field} must be a positive number of seconds"))
        };
        front = front.with_revisions(Revisions::new(
            &revisions.url,
            token,
            RevisionLimits {
                bytes: usize::try_from(revisions.cache_bytes)
                    .map_err(|_| "revisions.cache_bytes is too large")?,
                idle: seconds(revisions.idle_seconds, "idle_seconds")?,
                resolve: seconds(revisions.resolve_seconds, "resolve_seconds")?,
                timeout: seconds(revisions.timeout_seconds, "timeout_seconds")?,
            },
        )?);
    }
    let front = Arc::new(front.with_hosts(http.hosts.clone()));
    let url = format!("http://{address}");
    event(&serde_json::json!({
        "event": "serving",
        "url": url,
        "models": front.names(),
        "auth": front.auth(),
    }));
    let router = gf_server::http::router(front.clone());
    let served = tokio::spawn(async move {
        let shutdown = async move {
            let _ = stopped.wait_for(|stop| *stop).await;
        };
        let router = router.into_make_service_with_connect_info::<std::net::SocketAddr>();
        if let Err(error) = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown)
            .await
        {
            tracing::error!("the Models API: {error}");
        }
    });
    Ok((front, url, served))
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
