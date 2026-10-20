//! The router's configuration file (TOML).

use std::path::{Path, PathBuf};

use gf_server::config::toml_error;
use serde::Deserialize;

/// Everything a router needs to start.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The address to serve Arrow Flight on.
    pub listen: String,
    /// The address to serve `/metrics` and `/healthz` on, if any. They take
    /// no token and the metrics name nodes, so bind it to this machine or a
    /// private network.
    #[serde(default)]
    pub metrics_listen: Option<String>,
    /// Clients that may call, each with an optional quota.
    pub clients: Vec<ClientConfig>,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
    /// The nodes tables are placed on.
    pub nodes: Vec<NodeConfig>,
    /// Copies of each table, on its top-ranked nodes.
    #[serde(default = "one")]
    pub replicas: usize,
    /// How often to check nodes and repair replicas.
    #[serde(default = "one_second")]
    pub check_seconds: f64,
    /// The router's name, unique among the cluster's routers. A router that
    /// fails tables over needs one: nodes hold its leases under it.
    #[serde(default)]
    pub name: Option<String>,
    /// Promote a replica once a table's primary has been out of reach this
    /// long, or has left the configuration; unset, such a table takes no
    /// writes until its primary returns. Nodes must set `require_lease`, and
    /// only one router in a cluster may set this.
    #[serde(default)]
    pub failover_seconds: Option<f64>,
    /// How long each node's write lease lasts, renewed every check by the
    /// router that fails tables over, which alone takes leases; half of
    /// `failover_seconds` by default.
    #[serde(default)]
    pub lease_seconds: Option<f64>,
    /// How long a call forwarded to a node may take, including a prediction
    /// that waits for a fit.
    #[serde(default = "call_timeout_seconds")]
    pub call_timeout_seconds: f64,
    /// How long a write of rows or keys may take on its primary, and again
    /// to reach its replicas. Writes wait for no fit, so this is short: a
    /// primary that stops answering holds a client's write this long at most.
    #[serde(default = "write_timeout_seconds")]
    pub write_timeout_seconds: f64,
    /// Calls one client connection may have open at once.
    #[serde(default = "connection_concurrency")]
    pub connection_concurrency: usize,
}

fn connection_concurrency() -> usize {
    64
}

fn call_timeout_seconds() -> f64 {
    600.0
}

fn write_timeout_seconds() -> f64 {
    30.0
}

fn one() -> usize {
    1
}

fn one_second() -> f64 {
    1.0
}

/// A client's token, the tables it may reach, and its quota.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    pub token: String,
    /// Tables the token may reach, as names or prefixes ending in `*`; every
    /// table if unset.
    #[serde(default)]
    pub tables: Option<Vec<String>>,
    /// Sustained calls per second, if limited.
    #[serde(default)]
    pub per_second: Option<f64>,
    /// Calls allowed at once; defaults to one second's worth.
    #[serde(default)]
    pub burst: Option<u32>,
}

/// The router's TLS identity, and a client certificate authority for mutual TLS.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    pub certificate: PathBuf,
    pub key: PathBuf,
    #[serde(default)]
    pub client_ca: Option<PathBuf>,
}

/// A node: its address, the token the router uses, and how to verify and present TLS.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeConfig {
    pub name: String,
    /// `http://host:port`, or `https://host:port` for TLS.
    pub address: String,
    /// One of the node's replication tokens, since the router copies table logs.
    pub token: String,
    /// A certificate authority to verify the node with.
    #[serde(default)]
    pub ca: Option<PathBuf>,
    /// The name to verify the node's certificate against, if not the address's host,
    /// so one certificate can serve nodes at any address.
    #[serde(default)]
    pub domain: Option<String>,
    /// A client certificate and key to present, for nodes that require one.
    #[serde(default)]
    pub certificate: Option<PathBuf>,
    #[serde(default)]
    pub key: Option<PathBuf>,
}

impl Config {
    /// The settings that take effect only on restart, by name, that differ
    /// between this configuration and `loaded`.
    #[must_use]
    pub fn restart_differences(&self, loaded: &Self) -> Vec<&'static str> {
        let differs = [
            ("listen", self.listen != loaded.listen),
            (
                "metrics_listen",
                self.metrics_listen != loaded.metrics_listen,
            ),
            (
                "tls",
                format!("{:?}", self.tls) != format!("{:?}", loaded.tls),
            ),
            ("name", self.name != loaded.name),
            (
                "failover_seconds",
                self.failover_seconds != loaded.failover_seconds,
            ),
            ("lease_seconds", self.lease_seconds != loaded.lease_seconds),
            ("check_seconds", self.check_seconds != loaded.check_seconds),
            (
                "connection_concurrency",
                self.connection_concurrency != loaded.connection_concurrency,
            ),
        ];
        differs
            .into_iter()
            .filter_map(|(name, differs)| differs.then_some(name))
            .collect()
    }

    /// Reads a configuration file.
    ///
    /// # Errors
    ///
    /// If the file cannot be read or is not a valid configuration.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let config: Self =
            toml::from_str(&text).map_err(|error| toml_error(path, &text, &error))?;
        if config.nodes.is_empty() {
            return Err("the configuration has no nodes".to_string());
        }
        if config.clients.is_empty() {
            return Err("the configuration has no clients".to_string());
        }
        if config.replicas == 0 {
            return Err("replicas must be at least 1".to_string());
        }
        if !config.check_seconds.is_finite() || config.check_seconds <= 0.0 {
            return Err("check_seconds must be positive".to_string());
        }
        // Leases last half of failover_seconds and are renewed every check, at least twice.
        if config
            .failover_seconds
            .is_some_and(|seconds| !seconds.is_finite() || seconds < 4.0 * config.check_seconds)
        {
            return Err("failover_seconds must be at least four times check_seconds".to_string());
        }
        if config.failover_seconds.is_some() && config.name.as_deref().is_none_or(str::is_empty) {
            return Err(
                "a router that fails tables over needs a name, unique among the routers, \
                 which nodes hold its leases under"
                    .to_string(),
            );
        }
        if let Some(seconds) = config.lease_seconds {
            if config.failover_seconds.is_none() {
                return Err(
                    "lease_seconds needs failover_seconds: only the router that fails tables \
                     over takes leases"
                        .to_string(),
                );
            }
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err("lease_seconds must be positive".to_string());
            }
            if config
                .failover_seconds
                .is_some_and(|failover| seconds > failover / 2.0)
            {
                return Err("lease_seconds must be at most half of failover_seconds".to_string());
            }
        }
        for (name, seconds) in [
            ("call_timeout_seconds", config.call_timeout_seconds),
            ("write_timeout_seconds", config.write_timeout_seconds),
        ] {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(format!("{name} must be positive"));
            }
        }
        for client in &config.clients {
            if client
                .per_second
                .is_some_and(|rate| !rate.is_finite() || rate <= 0.0)
            {
                return Err("per_second must be positive".to_string());
            }
            if client.burst == Some(0) {
                return Err("burst must be at least 1".to_string());
            }
        }
        if config.connection_concurrency == 0 {
            return Err("connection_concurrency must be at least 1".to_string());
        }
        let mut names = std::collections::HashSet::new();
        if let Some(node) = config.nodes.iter().find(|node| !names.insert(&node.name)) {
            return Err(format!("two nodes are named {:?}", node.name));
        }
        Ok(config)
    }

    /// How long the leases this router takes last, if it fails tables over.
    #[must_use]
    pub fn lease(&self) -> Option<std::time::Duration> {
        let failover = self.failover_seconds?;
        Some(std::time::Duration::from_secs_f64(
            self.lease_seconds.unwrap_or(failover / 2.0),
        ))
    }

    /// Nodes reached over plaintext beyond this machine, whose replication
    /// tokens would cross the network unencrypted.
    #[must_use]
    pub fn plaintext_nodes(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|node| {
                node.address.strip_prefix("http://").is_some_and(|rest| {
                    let host = rest.split('/').next().unwrap_or_default();
                    let host = host.rsplit_once(':').map_or(host, |(host, _)| host);
                    let host = host.trim_matches(['[', ']']);
                    host != "localhost"
                        && host
                            .parse::<std::net::IpAddr>()
                            .ok()
                            .is_none_or(|ip| !ip.is_loopback())
                })
            })
            .map(|node| node.name.as_str())
            .collect()
    }
}
