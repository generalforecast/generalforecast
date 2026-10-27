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
use gf_server::progress::{Compiling, Progress};
use gf_server::revisions::{Limits as RevisionLimits, Revisions};
use gf_server::runtime;
use gf_server::service::{Placement, RequestBudget, SeriesPlacement, Service};
use gf_server::setup;
use gf_server::shard::Command;
use gf_server::upstream::{Cidr, Upstream};
use gf_server::worker::{Private, shared_parent};
use tokio::sync::{mpsc, oneshot, watch};
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
    steps.push(format!("gf admin node check --config {}", path.display()));
    steps.push(format!("gf admin node serve --config {}", path.display()));
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
    let drain = Duration::from_secs_f64(config.drain_seconds);
    let began = std::time::Instant::now();
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
    let mut interrupted = stopped.clone();
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

    let names = front
        .as_ref()
        .map_or_else(String::new, |(front, _, _)| served_names(front));
    let mut live = Live::default();
    let seats = runtime::seats(&config.workers);
    // Dropping a start that is under way kills the workers it started.
    let started = tokio::select! {
        started = runtime::start(&config, &seats, &key, &tables, &cache, &metrics) => started,
        () = live.follow(&metrics.compiling) => unreachable!("the progress line never ends"),
        _ = interrupted.wait_for(|stop| *stop) => {
            live.clear();
            stopped_status(front.as_ref().map(|(_, url, _)| url.as_str()), &names, began.elapsed());
            return Ok(());
        }
    };
    let (shards, routes) = started.map_err(|error| -> Box<dyn Error> { error })?;

    let mut placement = HashMap::new();
    for (index, path) in setup::table_logs(&tables)?.iter().enumerate() {
        if *interrupted.borrow() {
            break;
        }
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

    // Ctrl-C while the tables restored: the workers are up, so stop them.
    if *interrupted.borrow() {
        stop_shards(&shards).await;
        stopped_status(
            front.as_ref().map(|(_, url, _)| url.as_str()),
            &names,
            began.elapsed(),
        );
        return Ok(());
    }

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
        // A cap on streams, not a queue for them: a stream waiting its turn would
        // hold its rows in the connection's flow-control window, which the
        // rows of the calls it waits on may then not get through.
        .max_concurrent_streams(u32::try_from(config.connection_concurrency).unwrap_or(u32::MAX))
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
        let started = std::time::Instant::now();
        let buckets = tokio::select! {
            buckets = front.warm(&metrics.compiling) => buckets,
            () = live.follow(&metrics.compiling) => unreachable!("the progress line never ends"),
            _ = interrupted.wait_for(|stop| *stop) => {
                live.clear();
                stop_shards(&shards).await;
                stopped_status(Some(url), &names, began.elapsed());
                return Ok(());
            }
        };
        live.clear();
        let programs = metrics.warm_compiles.get() + buckets as u64;
        let cached = programs > 0 && metrics.compile_cache_misses.get() == 0;
        compiled(programs, started.elapsed().as_secs_f64(), cached);
        ready(&names, url);
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
    stop_shards(&shards).await;
    let url = match front {
        Some((_, url, served)) => {
            let _ = served.await;
            Some(url)
        }
        None => None,
    };
    stopped_status(url.as_deref(), &names, began.elapsed());
    tracing::info!("drained");
    Ok(())
}

/// Saves every shard's log heads, then stops its worker. Every write is
/// synced already; the heads let the next start check each log's whole
/// length. A worker still running after its deadline is killed, so none
/// outlives the server.
async fn stop_shards(shards: &[mpsc::Sender<Command>]) {
    for shard in shards {
        let (reply, answer) = oneshot::channel();
        if shard.send(Command::SaveHeads { reply }).await.is_ok() {
            let _ = answer.await;
        }
    }
    let stopping = shards.iter().map(|shard| async move {
        let (reply, answer) = oneshot::channel();
        if shard.send(Command::Shutdown { reply }).await.is_ok() {
            let _ = answer.await;
        }
    });
    futures::future::join_all(stopping).await;
}

/// The models the front serves, joined for the `Serving` and `Stopped` lines.
fn served_names(front: &Front) -> String {
    front
        .models()
        .iter()
        .map(|served| served.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether stdout gets status lines for a person rather than NDJSON events:
/// a terminal does, unless `GF_EVENTS=json` asks for the events anyway, as
/// `gf serve --json` does.
fn for_person() -> bool {
    std::io::stdout().is_terminal()
        && std::env::var_os("GF_EVENTS").is_none_or(|value| value != "json")
}

/// Writes one NDJSON event on stdout, as `gf serve` reports its progress.
/// Says the Models API is up: a status line on a terminal, else a JSON event.
fn ready(names: &str, url: &str) {
    if !for_person() {
        event(&serde_json::json!({"event": "ready", "url": url}));
        return;
    }
    status(
        "Serving",
        &format!("{names} at {url}"),
        Some("(Ctrl-C to stop)"),
    );
}

/// The `Compiled N programs in Xs` line, once the workers' recorded programs
/// and the warm shape buckets are compiled; `cached` when every one loaded
/// from the compile cache, which is why it took no time.
fn compiled(programs: u64, seconds: f64, cached: bool) {
    if !for_person() {
        event(&serde_json::json!({
            "event": "compiled", "programs": programs, "seconds": seconds, "cached": cached,
        }));
        return;
    }
    status(
        "Compiled",
        &compiled_message(programs, seconds),
        cached.then_some("(cached)"),
    );
}

fn compiled_message(programs: u64, seconds: f64) -> String {
    format!("{} in {seconds:.1}s", programs_noun(programs))
}

fn programs_noun(programs: u64) -> String {
    let noun = if programs == 1 { "program" } else { "programs" };
    format!("{} {noun}", thousands(programs))
}

/// Says the server has stopped, once its workers have exited: the `Stopped
/// <models> after <uptime>` line on a terminal, else a `stopped` event for
/// the Models API at `url`, if one was served.
fn stopped_status(url: Option<&str>, names: &str, uptime: Duration) {
    if !for_person() {
        if let Some(url) = url {
            event(&serde_json::json!({"event": "stopped", "url": url}));
        }
        return;
    }
    status("Stopped", &stopped_message(names, uptime), None);
}

fn stopped_message(names: &str, uptime: Duration) -> String {
    if names.is_empty() {
        format!("after {}", elapsed(uptime))
    } else {
        format!("{names} after {}", elapsed(uptime))
    }
}

/// A duration as a person reads it: `14s`, `2m 14s`, `1h 2m`.
fn elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs();
    match (seconds / 3600, seconds / 60 % 60, seconds % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m {s}s"),
        (h, m, _) => format!("{h}h {m}m"),
    }
}

/// The `Compiling 3/12 programs` line a terminal rewrites in place while
/// the workers and the warm-up compile; piped, a few `compiling` events.
#[derive(Default)]
struct Live {
    /// Columns the terminal line last drawn took, to blank it.
    drawn: usize,
    /// The total and the quarter of it the last `compiling` event reported.
    reported: Option<(u64, u64)>,
}

impl Live {
    /// Shows each change in `compiling`; never returns, so it goes in a
    /// `select!` beside what it follows.
    async fn follow(&mut self, compiling: &Compiling) {
        let mut changes = compiling.changes();
        loop {
            let progress = *changes.borrow_and_update();
            if progress.total > 0 {
                self.show(progress);
            }
            if changes.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    }

    fn show(&mut self, progress: Progress) {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        if !for_person() {
            if !self.event_due(progress) {
                return;
            }
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({
                    "event": "compiling", "done": progress.done, "total": progress.total,
                })
            );
            let _ = stdout.flush();
            return;
        }
        let line = status_line("Compiling", &compiling_message(progress), None, colour());
        let shown = line.chars().count();
        let pad = self.drawn.saturating_sub(shown);
        let _ = write!(stdout, "\r{line}{:pad$}", "");
        let _ = stdout.flush();
        self.drawn = shown.max(self.drawn);
    }

    /// Whether a piped `compiling` event is due: one as each quarter of the
    /// programs is done, counted afresh when the total grows (the warm-up's
    /// shape buckets join once the workers have compiled their programs).
    fn event_due(&mut self, progress: Progress) -> bool {
        let quarter = (progress.done * 4) / progress.total.max(1);
        if self
            .reported
            .is_some_and(|(total, last)| total == progress.total && quarter <= last)
        {
            return false;
        }
        self.reported = Some((progress.total, quarter));
        true
    }

    /// Blanks the line, so the next status line takes its place.
    fn clear(&mut self) {
        use std::io::Write;
        if self.drawn == 0 {
            return;
        }
        let mut stdout = std::io::stdout().lock();
        let _ = write!(stdout, "\r{:width$}\r", "", width = self.drawn);
        let _ = stdout.flush();
        self.drawn = 0;
    }
}

fn compiling_message(progress: Progress) -> String {
    format!(
        "{}/{}",
        thousands(progress.done),
        programs_noun(progress.total)
    )
}

/// The `Loading <model>` line for each model the front serves.
fn loading(models: &[ServedModel]) {
    if !for_person() {
        return;
    }
    for served in models {
        status("Loading", &served.resolved, None);
    }
}

/// A status line: the past-tense verb right-aligned in twelve columns, a
/// space, the message, and a quiet trailer if any.
fn status(verb: &str, message: &str, trailer: Option<&str>) {
    println!("{}", status_line(verb, message, trailer, colour()));
}

fn colour() -> bool {
    std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty())
}

fn status_line(verb: &str, message: &str, trailer: Option<&str>, colour: bool) -> String {
    let verb = format!("{verb:>12}");
    let trailer = match (trailer, colour) {
        (Some(trailer), true) => format!(" \x1b[2m{trailer}\x1b[0m"),
        (Some(trailer), false) => format!(" {trailer}"),
        (None, _) => String::new(),
    };
    if colour {
        format!("\x1b[1;32m{verb}\x1b[0m {message}{trailer}")
    } else {
        format!("{verb} {message}{trailer}")
    }
}

fn thousands(count: u64) -> String {
    let digits = count.to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// A JSON line for whatever launched this server; a person at a terminal
/// gets the `ready` status line instead.
fn event(value: &serde_json::Value) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    if for_person() {
        return;
    }
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
    loading(&models);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_lines_right_align_the_verb_in_twelve_columns() {
        assert_eq!(
            status_line(
                "Compiling",
                &compiling_message(Progress { done: 3, total: 12 }),
                None,
                false
            ),
            "   Compiling 3/12 programs"
        );
        assert_eq!(
            status_line("Compiled", &compiled_message(12, 6.8), None, false),
            "    Compiled 12 programs in 6.8s"
        );
        assert_eq!(
            status_line(
                "Compiled",
                &compiled_message(1, 0.02),
                Some("(cached)"),
                false
            ),
            "    Compiled 1 program in 0.0s (cached)"
        );
        assert_eq!(
            status_line(
                "Stopped",
                &stopped_message("wine", Duration::from_secs(134)),
                None,
                false
            ),
            "     Stopped wine after 2m 14s"
        );
        assert_eq!(
            status_line(
                "Stopped",
                &stopped_message("", Duration::from_secs(9)),
                None,
                false
            ),
            "     Stopped after 9s"
        );
    }

    #[test]
    fn status_lines_colour_the_verb_and_dim_the_trailer() {
        assert_eq!(
            status_line("Compiled", "12 programs in 0.0s", Some("(cached)"), true),
            "\x1b[1;32m    Compiled\x1b[0m 12 programs in 0.0s \x1b[2m(cached)\x1b[0m"
        );
    }

    #[test]
    fn compiling_counts_programs_with_thousands_separators() {
        assert_eq!(
            compiling_message(Progress {
                done: 1_000,
                total: 2_500
            }),
            "1,000/2,500 programs"
        );
        assert_eq!(
            compiling_message(Progress { done: 0, total: 1 }),
            "0/1 program"
        );
    }

    #[test]
    fn piped_compiling_events_come_once_a_quarter_and_again_when_the_total_grows() {
        let mut live = Live::default();
        let at = |done, total| Progress { done, total };
        let due: Vec<bool> = [at(1, 12), at(2, 12), at(3, 12), at(6, 12), at(12, 12)]
            .into_iter()
            .map(|progress| live.event_due(progress))
            .collect();
        assert_eq!(due, [true, false, true, true, true]);
        assert!(live.event_due(at(12, 18)));
        assert!(!live.event_due(at(13, 18)));
        assert!(live.event_due(at(14, 18)));
        assert!(live.event_due(at(18, 18)));
        assert!(!live.event_due(at(18, 18)));
    }

    #[test]
    fn elapsed_reads_as_seconds_minutes_or_hours() {
        assert_eq!(elapsed(Duration::from_secs(0)), "0s");
        assert_eq!(elapsed(Duration::from_secs(59)), "59s");
        assert_eq!(elapsed(Duration::from_secs(60)), "1m 0s");
        assert_eq!(elapsed(Duration::from_secs(3_720)), "1h 2m");
    }
}
