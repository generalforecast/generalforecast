// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `gf-router run --config router.toml`: forwards clients to the nodes that
//! serve their tables. `init` writes a starter configuration, and `check`
//! looks at the configuration and reaches each node.
//!
//! SIGHUP reloads the node list, the replica count, and the client tokens and
//! quotas; tables move to match.

use std::collections::HashMap;
use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use arrow_flight::flight_service_client::FlightServiceClient;
use clap::Parser;
use gf_router::config::{Config, NodeConfig};
use gf_router::quota::{Limit, Quotas};
use gf_router::service::{Cluster, Node, Router, Timeouts};
use gf_server::action::LeaseAnswer;
use gf_server::auth::{Scope, digest};
use serde::Deserialize;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity, ServerTlsConfig};

/// Seconds a client call may run past its node calls, for the router's own work.
const CHECK_SECONDS: f64 = 5.0;

#[derive(Parser)]
#[command(about = "Route clients to the generalforecast nodes that serve their tables")]
struct Arguments {
    #[command(subcommand)]
    command: Subcommand,
}

#[derive(clap::Subcommand)]
enum Subcommand {
    /// Routes clients to the nodes a configuration names.
    Run {
        /// The configuration file.
        #[arg(long)]
        config: PathBuf,
    },
    /// Writes a starter configuration for a router over a few nodes.
    Init {
        /// Where to write it.
        #[arg(long, default_value = "router.toml")]
        path: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8815")]
        listen: String,
        /// A node's address, such as http://127.0.0.1:8816; repeat for each.
        #[arg(long = "node", value_name = "ADDRESS", required = true)]
        nodes: Vec<String>,
        /// The client token; a random one if unset.
        #[arg(long)]
        token: Option<String>,
        /// The replication token the nodes list; a random one if unset.
        #[arg(long)]
        replication_token: Option<String>,
        /// Overwrite the file if it exists.
        #[arg(long)]
        force: bool,
    },
    /// Checks a configuration and reaches each node, without serving.
    Check {
        /// The configuration file.
        #[arg(long)]
        config: PathBuf,
    },
}

/// A lazily connecting client for a node, with TLS and a client certificate if configured.
fn node(config: &NodeConfig, timeouts: Timeouts) -> Result<Node, Box<dyn Error>> {
    let mut endpoint = Endpoint::from_shared(config.address.clone())?;
    if config.address.starts_with("https://") {
        // A node without a certificate authority of its own is verified
        // against the system's, as one with a public certificate is.
        let mut tls = match &config.ca {
            Some(ca) => {
                ClientTlsConfig::new().ca_certificate(Certificate::from_pem(std::fs::read(ca)?))
            }
            None => ClientTlsConfig::new().with_native_roots(),
        };
        if let Some(domain) = &config.domain {
            tls = tls.domain_name(domain.clone());
        }
        if let (Some(certificate), Some(key)) = (&config.certificate, &config.key) {
            tls = tls.identity(Identity::from_pem(
                std::fs::read(certificate)?,
                std::fs::read(key)?,
            ));
        }
        endpoint = endpoint.tls_config(tls)?;
    }
    Ok(Node::new(
        config.name.clone(),
        FlightServiceClient::new(endpoint.connect_lazy()),
        format!("Bearer {}", config.token).parse()?,
        timeouts,
    ))
}

/// The configured nodes and replica count.
fn cluster(config: &Config) -> Result<Cluster, Box<dyn Error>> {
    let timeouts = Timeouts {
        call: Duration::from_secs_f64(config.call_timeout_seconds),
        write: Duration::from_secs_f64(config.write_timeout_seconds),
    };
    let nodes = config
        .nodes
        .iter()
        .map(|settings| node(settings, timeouts))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Cluster::new(nodes, config.replicas))
}

/// The client tokens by digest, the tables each reaches, and their quotas.
fn clients(config: &Config) -> (HashMap<[u8; 32], Scope>, Quotas) {
    let tokens = config
        .clients
        .iter()
        .map(|client| {
            let scope = client.tables.clone().map_or(Scope::All, Scope::Tables);
            (digest(&client.token), scope)
        })
        .collect();
    let limits = config
        .clients
        .iter()
        .filter_map(|client| {
            client.per_second.map(|per_second| {
                let burst = client
                    .burst
                    .unwrap_or_else(|| per_second.ceil().max(1.0) as u32);
                (digest(&client.token), Limit { per_second, burst })
            })
        })
        .collect();
    (tokens, Quotas::new(limits))
}

/// Reloads the cluster and clients from the configuration file on each
/// SIGHUP; settings that take effect only on restart keep their running
/// values, and the log names any that changed.
async fn reload(router: Router, path: PathBuf, running: Config) {
    use tokio::signal::unix::{SignalKind, signal};
    let mut hangups = signal(SignalKind::hangup()).expect("signal handler");
    while hangups.recv().await.is_some() {
        let loaded = Config::load(&path)
            .map_err(Into::<Box<dyn Error>>::into)
            .and_then(|config| Ok((cluster(&config)?, clients(&config), config)));
        match loaded {
            Ok((cluster, (tokens, quotas), config)) => {
                router.reconfigure(cluster);
                router.set_clients(tokens, quotas);
                let ignored = running.restart_differences(&config);
                if ignored.is_empty() {
                    tracing::info!("reloaded the configuration");
                } else {
                    tracing::error!(
                        "reloaded the nodes, replicas and clients; {} change only on \
                         restart, so they keep their running values",
                        ignored.join(", ")
                    );
                }
            }
            Err(error) => tracing::error!("keeping the old configuration: {error}"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    match Arguments::parse().command {
        Subcommand::Run { config } => run(config).await,
        Subcommand::Init {
            path,
            listen,
            nodes,
            token,
            replication_token,
            force,
        } => init(&path, &listen, &nodes, token, replication_token, force),
        Subcommand::Check { config } => check(&config).await,
    }
}

/// Writes a starter configuration, and says how to go on.
fn init(
    path: &std::path::Path,
    listen: &str,
    nodes: &[String],
    token: Option<String>,
    replication_token: Option<String>,
    force: bool,
) -> Result<(), Box<dyn Error>> {
    if path.exists() && !force {
        return Err(format!("{} exists; pass --force to overwrite it", path.display()).into());
    }
    let token = match token {
        Some(token) => token,
        None => gf_server::setup::random_hex(16)?,
    };
    let replication = match replication_token {
        Some(token) => token,
        None => gf_server::setup::random_hex(16)?,
    };
    let config = gf_router::setup::starter_config(listen, nodes, &token, &replication);
    std::fs::write(path, config)?;
    println!("wrote {}; the client token is {token}", path.display());
    println!("each node lists the replication token {replication} in replication_tokens");
    println!("next: gf-router check --config {}", path.display());
    println!("      gf-router run --config {}", path.display());
    Ok(())
}

/// Loads a configuration, looks for its TLS files, and reaches each node.
async fn check(path: &std::path::Path) -> Result<(), Box<dyn Error>> {
    let config = Config::load(path)?;
    let mut failed = false;
    let mut report = |ok: bool, line: String| {
        failed |= !ok;
        println!("{} {line}", if ok { "ok    " } else { "failed" });
    };
    for (what, file) in config
        .tls
        .iter()
        .flat_map(|tls| {
            [
                ("tls certificate", Some(&tls.certificate)),
                ("tls key", Some(&tls.key)),
                ("tls client ca", tls.client_ca.as_ref()),
            ]
        })
        .filter_map(|(what, file)| file.map(|file| (what, file)))
    {
        match std::fs::metadata(file) {
            Ok(_) => report(true, format!("{what}: {}", file.display())),
            Err(error) => report(false, format!("{what}: {}: {error}", file.display())),
        }
    }
    if config.tls.is_none() {
        report(
            true,
            "tls: off (fine on this machine; beyond it, add [tls])".to_string(),
        );
    }
    for plaintext in config.plaintext_nodes() {
        report(
            false,
            format!(
                "node {plaintext}: its replication token would cross the network unencrypted; use https://"
            ),
        );
    }
    let timeouts = Timeouts {
        call: Duration::from_secs(5),
        write: Duration::from_secs(5),
    };
    for settings in &config.nodes {
        let reached = match node(settings, timeouts) {
            // Reading log lengths needs the replication token the router uses.
            Ok(node) => match node.action("log_lengths", serde_json::json!({})).await {
                Ok(_) => lease_report(&node, &config).await,
                Err(status) => Err(status.message().to_string()),
            },
            Err(error) => Err(error.to_string()),
        };
        match reached {
            Ok(detail) => report(
                true,
                format!(
                    "node {}: {} answers to the replication token{detail}",
                    settings.name, settings.address
                ),
            ),
            Err(error) => report(
                false,
                format!("node {}: {}: {error}", settings.name, settings.address),
            ),
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}

/// What a node says of its write lease, for a router that fails tables
/// over: whether it requires one, and whether another router holds it. Asks
/// without taking the lease.
async fn lease_report(node: &Node, config: &Config) -> Result<String, String> {
    let (Some(_), Some(name)) = (config.failover_seconds, &config.name) else {
        return Ok(String::new());
    };
    let probe = serde_json::json!({"seconds": 0.0, "holder": name, "probe": true});
    let answer = node
        .action("lease", probe)
        .await
        .map_err(|status| status.message().to_string())?;
    let answer = answer
        .first()
        .and_then(|answer| LeaseAnswer::deserialize(answer).ok())
        .unwrap_or_default();
    if !answer.enforced {
        return Err("it takes writes without a lease; set require_lease = true".to_string());
    }
    match answer.holder {
        Some(holder) => Err(format!(
            "router {holder} holds its lease, and only one router may set failover_seconds"
        )),
        None => Ok(", and requires leases".to_string()),
    }
}

async fn run(path: PathBuf) -> Result<(), Box<dyn Error>> {
    gf_server::trace::init_logging();
    gf_server::trace::abort_on_panic();
    let config = Config::load(&path)?;

    let plaintext = config.plaintext_nodes();
    if !plaintext.is_empty() {
        tracing::warn!(
            nodes = ?plaintext,
            "replication tokens cross the network unencrypted to these nodes; use https://"
        );
    }
    let mut router = Router::new(cluster(&config)?);
    if let (Some(seconds), Some(lease), Some(name)) =
        (config.failover_seconds, config.lease(), config.name.clone())
    {
        router = router.with_failover(Duration::from_secs_f64(seconds), lease, name);
    }
    let (tokens, quotas) = clients(&config);
    router.set_clients(tokens, quotas);
    // Writes name their table's term, so the router learns every table's
    // copies before it serves any.
    router.prepare().await;
    let interval = Duration::from_secs_f64(config.check_seconds);
    tokio::spawn(router.clone().maintain(interval));
    if let Some(address) = &config.metrics_listen {
        let listener = tokio::net::TcpListener::bind(address).await?;
        let app = axum::Router::new()
            .route(
                "/metrics",
                axum::routing::get(
                    |axum::extract::State(router): axum::extract::State<Router>| async move {
                        router.metrics()
                    },
                ),
            )
            .route("/healthz", axum::routing::get(|| async { "ok" }))
            .with_state(router.clone());
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                tracing::error!("serving metrics: {error}");
            }
        });
    }
    tokio::spawn(reload(router.clone(), path, config.clone()));

    let address: SocketAddr = config.listen.parse()?;
    // A call lasts at most a node call, or a write and its shipment to replicas.
    let calls = config
        .call_timeout_seconds
        .max(2.0 * config.write_timeout_seconds);
    let mut server = tonic::transport::Server::builder()
        .timeout(Duration::from_secs_f64(calls + CHECK_SECONDS))
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
        tracing::warn!("routing without TLS on a non-loopback address");
    }
    tracing::info!(%address, nodes = config.nodes.len(), replicas = config.replicas, "routing");
    server
        .add_service(router.into_server())
        .serve_with_shutdown(address, shutdown())
        .await?;
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
}
