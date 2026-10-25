// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The server's configuration file (TOML).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use gf_core::cache::Eviction;
use gf_core::catalog::Surface;
use gf_core::node::Scheduling;
use serde::{Deserialize, Deserializer};

use crate::auth::TokenConfig;
use crate::shard::EphemeralLimits;

/// Everything a server needs to start.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The address to serve Arrow Flight on, such as `0.0.0.0:8815`.
    pub listen: String,
    /// The address to serve `/metrics`, `/healthz` and `/readyz` on, if any.
    /// They take no token and the metrics name tables, so bind it to this
    /// machine or a private network.
    #[serde(default)]
    pub metrics_listen: Option<String>,
    /// A file to append one JSON line to per answered prediction request:
    /// its table, when it arrived, when its step was planned, sent, started
    /// and finished, its rows and its share of the worker's time, for exact
    /// latency percentiles. None by default.
    #[serde(default)]
    pub timing_log: Option<PathBuf>,
    /// Present, it also turns on timers for the server's life.
    #[serde(default)]
    pub profiling: Option<ProfilingConfig>,
    /// A certificate and key, in PEM, to serve Flight over TLS.
    #[serde(default)]
    pub tls: Option<TlsConfig>,
    /// Serve beyond this machine without TLS, sending tokens and rows in the
    /// clear, on a network you trust.
    #[serde(default)]
    pub allow_plaintext: bool,
    /// Where table logs live.
    pub data: PathBuf,
    /// Bearer tokens that may call the server, each reaching every table or only some.
    pub tokens: Vec<TokenConfig>,
    /// Tokens for routers, which may also read and apply the logs of the
    /// tables they reach, and those logs hold raw rows. A router forwards
    /// every client call under its token, so give these to routers alone, and
    /// scope each router's token to the tables it serves, as `tokens` are
    /// scoped, where routers serve different tables.
    #[serde(default)]
    pub replication_tokens: Vec<TokenConfig>,
    /// Take client writes only while a router keeps this node's lease, as nodes
    /// behind a router with `failover_seconds` must.
    #[serde(default)]
    pub require_lease: bool,
    /// The environment variable holding the log encryption key, as 64 hex
    /// digits; or give `encryption_key_file`.
    #[serde(default)]
    pub encryption_key_env: Option<String>,
    /// A file holding the log encryption key, as 64 hex digits, readable by
    /// the server's user alone; or give `encryption_key_env`.
    #[serde(default)]
    pub encryption_key_file: Option<PathBuf>,
    /// Environment variables holding keys logs were sealed under before the
    /// current one. Their logs open, and are rewritten under the current key
    /// at startup; once every log is rewritten, the old keys can go.
    #[serde(default)]
    pub previous_key_envs: Vec<String>,
    /// Device memory for encoded contexts, per worker; `device_fraction` of the
    /// worker's device if unset. With 0 no context is kept, so every step
    /// encodes the contexts it reads again.
    #[serde(default)]
    pub device_bytes: Option<u64>,
    /// The share of each worker's device memory for encoded contexts, when
    /// `device_bytes` is unset; the worker's memory budget gives half if this
    /// is unset too. The rest is left for activations.
    #[serde(default)]
    pub device_fraction: Option<f64>,
    /// Host memory for contexts evicted from the device, per worker; the
    /// worker's memory budget gives a quarter of the machine's memory, split
    /// between workers, if unset. With 0, contexts evicted from the device
    /// are dropped.
    #[serde(default)]
    pub host_bytes: Option<u64>,
    /// Memory for tables' rows, per worker; the worker's memory budget gives a
    /// quarter of the machine's memory, split between workers, if unset. Rows
    /// past it wait in sealed files under `data` until read, so a table may
    /// hold more rows than memory.
    #[serde(default)]
    pub row_memory_bytes: Option<u64>,
    #[serde(default = "slot_bytes")]
    pub slot_bytes: u64,
    /// Disk for contexts evicted from host memory, per worker, in files under
    /// `data`; none by default.
    #[serde(default)]
    pub disk_bytes: u64,
    #[serde(default = "bucket_minimum")]
    pub bucket_minimum: usize,
    /// Context bytes one step may newly encode, per worker; the device memory
    /// for contexts if unset. Tables past it wait for a later step, so a large
    /// table that must encode holds up the rest for less time.
    #[serde(default)]
    pub step_bytes: Option<u64>,
    /// The order each step takes waiting tables in: `turns`, where each step
    /// starts after the table the last one started with, or `arrival`, where
    /// the table whose oldest request has waited longest goes first.
    #[serde(default, deserialize_with = "scheduling")]
    pub scheduling: Scheduling,
    /// Which cached context a full tier gives up first: `gdsf` keeps what is
    /// costly to recompute per byte and often read, `lru` keeps what was read
    /// most recently.
    #[serde(default, deserialize_with = "eviction")]
    pub eviction: Eviction,
    /// How long a worker may go without progress on a step before it counts
    /// as hung and is started again; each compile it finishes is progress.
    #[serde(default = "step_timeout_seconds")]
    pub step_timeout_seconds: f64,
    /// How long one client call may take, including a prediction that waits
    /// for a fit.
    #[serde(default = "request_timeout_seconds")]
    pub request_timeout_seconds: f64,
    /// Calls one connection may have open at once.
    #[serde(default = "connection_concurrency")]
    pub connection_concurrency: usize,
    /// Bytes of client writes and queries the server holds in memory at once,
    /// across every connection; calls past it are refused until others finish.
    #[serde(default = "request_bytes_in_flight")]
    pub request_bytes_in_flight: u64,
    /// Bytes of queries one worker may have waiting for a step; requests past
    /// it are refused until the queue drains.
    #[serde(default = "predict_queue_bytes")]
    pub predict_queue_bytes: u64,
    /// How long to wait for open calls to finish on shutdown before exiting anyway.
    #[serde(default = "drain_seconds")]
    pub drain_seconds: f64,
    /// The most ephemeral tables, one per request context, each worker keeps
    /// in memory; past it the least recently used unpinned ones go.
    #[serde(default = "ephemeral_tables")]
    pub ephemeral_tables: usize,
    /// Bytes of request contexts' rows each worker keeps in ephemeral
    /// tables; past it the least recently used unpinned ones go.
    #[serde(default = "ephemeral_bytes")]
    pub ephemeral_bytes: u64,
    /// How long an ephemeral table may go unused before it goes.
    #[serde(default = "ephemeral_idle_seconds")]
    pub ephemeral_idle_seconds: f64,
    /// The most tables the server holds; creations past it are refused. Each
    /// table keeps its log open, so the server's open-file limit must allow
    /// as many files.
    #[serde(default = "max_tables")]
    pub max_tables: usize,
    /// The Models API over HTTP, if the server answers it.
    #[serde(default)]
    pub http: Option<HttpConfig>,
    /// One worker per accelerator.
    pub workers: Vec<WorkerConfig>,
    /// How tables are placed on and moved between workers.
    #[serde(default)]
    pub placement: PlacementConfig,
}

/// The Models API over HTTP: where it listens, the models it serves and
/// whose keys it takes.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpConfig {
    /// The address to serve the Models API on, such as `127.0.0.1:8000`.
    pub listen: String,
    /// A JSON file listing the served models, as `gf serve` and the hosted
    /// runtime write it from the catalog.
    pub models: PathBuf,
    /// Whose bearer keys requests must present.
    pub keys: KeysConfig,
    /// Where the server runs, which decides the models it may serve:
    /// `self-hosted` by an operator, or `remote` on General Forecast's hosts.
    #[serde(deserialize_with = "surface")]
    pub surface: Surface,
    /// How long one inference request may take, queued and run.
    #[serde(default = "inference_timeout_seconds")]
    pub request_timeout_seconds: f64,
    /// The origin every route the front does not own streams to, such as
    /// the control plane on `http://127.0.0.1:8001`. Unset, those routes
    /// are not found.
    #[serde(default)]
    pub upstream: Option<String>,
    /// How long the upstream may take to connect, to answer, and between
    /// the chunks of its answer.
    #[serde(default = "upstream_timeout_seconds")]
    pub upstream_timeout_seconds: f64,
    /// The largest request body streamed upstream.
    #[serde(default = "upstream_body_bytes")]
    pub upstream_body_bytes: u64,
    /// Peers, as address ranges, whose `X-Forwarded-*` headers are kept.
    #[serde(default)]
    pub trusted_proxies: Vec<String>,
    /// The exact `Host` names requests must carry, as a service behind
    /// one endpoint answers only for it; requests from browsers, which send
    /// `Origin`, are then refused. Empty takes any host.
    #[serde(default)]
    pub hosts: Vec<String>,
    /// Where deployed revisions are published, if this front predicts from
    /// them.
    #[serde(default)]
    pub revisions: Option<RevisionsConfig>,
}

/// The control plane a front loads deployed revisions from.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionsConfig {
    /// The control plane's origin, e.g. `http://127.0.0.1:8001`.
    pub url: String,
    /// The environment variable holding the control plane's service token.
    pub token_env: String,
    /// Bytes of published predictors held at once.
    #[serde(default = "revision_cache_bytes")]
    pub cache_bytes: u64,
    /// How long a predictor nobody predicted from is held.
    #[serde(default = "revision_idle_seconds")]
    pub idle_seconds: f64,
    /// How long an environment's revision is trusted before it is asked
    /// again, which bounds how late a promote or rollback is seen.
    #[serde(default = "revision_resolve_seconds")]
    pub resolve_seconds: f64,
    /// How long one control-plane call may take.
    #[serde(default = "revision_timeout_seconds")]
    pub timeout_seconds: f64,
}

fn revision_cache_bytes() -> u64 {
    256 * 1024 * 1024
}

fn revision_idle_seconds() -> f64 {
    600.0
}

fn revision_resolve_seconds() -> f64 {
    5.0
}

fn revision_timeout_seconds() -> f64 {
    30.0
}

/// Whose bearer keys the Models API takes.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum KeysConfig {
    /// One key, from the environment variable `env` if it is set; without
    /// it, every request is served, and a server listening beyond this
    /// machine warns that anyone who reaches it can run inference. A key
    /// reaches only `project` and holds only `permissions`, when given.
    Static {
        env: String,
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        permissions: Option<Vec<String>>,
    },
    /// The control plane's keys, checked at `url` under the service token
    /// in the environment variable `token_env`.
    Introspect { url: String, token_env: String },
}

fn surface<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Surface, D::Error> {
    match String::deserialize(deserializer)?.as_str() {
        "self-hosted" => Ok(Surface::SelfHosted),
        "remote" => Ok(Surface::Remote),
        other => Err(serde::de::Error::custom(format!(
            "surface is self-hosted or remote, not {other:?}"
        ))),
    }
}

fn inference_timeout_seconds() -> f64 {
    60.0
}

fn upstream_timeout_seconds() -> f64 {
    300.0
}

/// The control plane's own request body cap, so the front refuses what it would.
fn upstream_body_bytes() -> u64 {
    32 * 1024 * 1024
}

/// How a server places its tables on its workers.
///
/// A move holds the table's calls while the source shard finishes its
/// in-flight steps and the destination shard replays the table's log, and
/// the destination serves nothing else meanwhile. Placement is not written
/// down, so a restart puts tables back on shards by index and undoes every
/// move.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementConfig {
    #[serde(default)]
    pub assign: Assign,
    /// Seconds between rounds that move tables by load, with `assign = "load"`.
    /// Absent or 0, no round runs on its own, and tables move only through
    /// the `rebalance` and `move_table` actions.
    #[serde(default)]
    pub rebalance_seconds: f64,
    /// The share of all tables' rows one round may move.
    #[serde(default = "churn_budget")]
    pub churn_budget: f64,
    /// No moves while the busiest worker's load over the mean is below this.
    #[serde(default = "imbalance_floor")]
    pub imbalance_floor: f64,
}

/// How tables find their worker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Assign {
    /// A new table goes to the worker with the fewest tables, and stays.
    #[default]
    Count,
    /// A new table goes to the worker with the least worker time in the last
    /// round, or the fewest tables until a round has run, and a round moves
    /// tables between workers as their measured load changes.
    Load,
}

fn churn_budget() -> f64 {
    gf_assign::Policy::default().churn_budget
}

fn imbalance_floor() -> f64 {
    gf_assign::Policy::default().imbalance_floor
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            assign: Assign::default(),
            rebalance_seconds: 0.0,
            churn_budget: churn_budget(),
            imbalance_floor: imbalance_floor(),
        }
    }
}

/// For generalforecast's developers.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilingConfig {
    /// The root `profile_workers` writes under.
    pub directory: PathBuf,
}

/// The server's TLS identity.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    pub certificate: PathBuf,
    pub key: PathBuf,
    /// A certificate authority whose client certificates are required (mutual TLS).
    #[serde(default)]
    pub client_ca: Option<PathBuf>,
}

/// How to start a worker, and the models it serves by name.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    /// The worker's command line. It must start the worker itself, or `exec`
    /// it from any wrapper: the server talks only to the process it started.
    pub command: Vec<String>,
    /// Start one copy per visible NVIDIA GPU, each seeing only its own.
    #[serde(default)]
    pub per_gpu: bool,
    /// Without GPUs, start one copy per this many of the cores the server
    /// may use, each pinned to its own, so each copy's XLA threads run on
    /// its cores alone; `worker::CPU_WORKER_CORES` if unset.
    #[serde(default)]
    pub cpu_threads: Option<usize>,
    /// Model name to spec: a built-in model's repository id.
    pub models: BTreeMap<String, String>,
}

fn slot_bytes() -> u64 {
    1 << 20
}

fn scheduling<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Scheduling, D::Error> {
    named(
        deserializer,
        "scheduling",
        &Scheduling::ALL,
        Scheduling::name,
    )
}

fn eviction<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Eviction, D::Error> {
    named(deserializer, "eviction", &Eviction::ALL, Eviction::name)
}

/// One of a setting's values, read by name.
fn named<'de, D: Deserializer<'de>, T: Copy>(
    deserializer: D,
    setting: &str,
    values: &[T],
    name: fn(T) -> &'static str,
) -> Result<T, D::Error> {
    let given = String::deserialize(deserializer)?;
    values
        .iter()
        .copied()
        .find(|&value| name(value) == given)
        .ok_or_else(|| {
            let names: Vec<&str> = values.iter().map(|&value| name(value)).collect();
            serde::de::Error::custom(format!(
                "{setting} is one of {}, not {given:?}",
                names.join(", ")
            ))
        })
}

fn bucket_minimum() -> usize {
    8
}

fn step_timeout_seconds() -> f64 {
    300.0
}

fn request_timeout_seconds() -> f64 {
    600.0
}

fn connection_concurrency() -> usize {
    64
}

fn ephemeral_tables() -> usize {
    1024
}

fn ephemeral_bytes() -> u64 {
    1 << 30
}

fn ephemeral_idle_seconds() -> f64 {
    300.0
}

fn max_tables() -> usize {
    100_000
}

/// Tokens shorter than this draw a warning: they are guessable.
const SHORT_TOKEN: usize = 16;

fn request_bytes_in_flight() -> u64 {
    2 << 30
}

fn predict_queue_bytes() -> u64 {
    1 << 30
}

fn drain_seconds() -> f64 {
    30.0
}

impl Config {
    /// The ephemeral tables each shard keeps.
    #[must_use]
    pub fn ephemeral(&self) -> EphemeralLimits {
        EphemeralLimits {
            tables: self.ephemeral_tables,
            bytes: self.ephemeral_bytes,
            idle: Duration::from_secs_f64(self.ephemeral_idle_seconds),
        }
    }

    #[must_use]
    pub fn timers(&self) -> bool {
        self.profiling.is_some()
    }

    /// The environment variables that hold log keys, current and previous,
    /// which workers must never see.
    #[must_use]
    pub fn key_envs(&self) -> Vec<String> {
        self.encryption_key_env
            .iter()
            .chain(&self.previous_key_envs)
            .cloned()
            .collect()
    }

    /// What a server should warn of before starting: short tokens, and a
    /// configuration others may read although it holds tokens.
    #[must_use]
    pub fn warnings(&self, path: &Path) -> Vec<String> {
        use std::os::unix::fs::PermissionsExt;
        let mut warnings = Vec::new();
        let tokens = self.tokens.iter().chain(&self.replication_tokens);
        if tokens
            .clone()
            .any(|token| token.token().len() < SHORT_TOKEN)
        {
            warnings.push(format!(
                "a token is shorter than {SHORT_TOKEN} characters, which makes it guessable"
            ));
        }
        let shared = |path: &Path| {
            std::fs::metadata(path).is_ok_and(|meta| meta.permissions().mode() & 0o077 != 0)
        };
        if shared(path) && tokens.count() > 0 {
            warnings.push(format!(
                "{} holds tokens but others may read it; chmod 600 it",
                path.display()
            ));
        }
        if let Some(key) = self
            .encryption_key_file
            .as_deref()
            .filter(|key| shared(key))
        {
            warnings.push(format!(
                "others may read the log encryption key in {}; chmod 600 it",
                key.display()
            ));
        }
        warnings
    }

    /// Reads a configuration file.
    ///
    /// # Errors
    ///
    /// If the file cannot be read or is not a valid configuration.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let config = Self::parse(path, &text)?;
        config.listening()?;
        Ok(config)
    }

    /// A configuration's text, checked for what its shards run on; `load`
    /// checks what serving Flight beyond it takes as well, and an
    /// in-process engine needs no more than this.
    ///
    /// # Errors
    ///
    /// If the text is not a valid configuration.
    pub fn parse(path: &Path, text: &str) -> Result<Self, String> {
        let config: Self = toml::from_str(text).map_err(|error| toml_error(path, text, &error))?;
        if config.workers.is_empty() {
            return Err("the configuration has no workers".to_string());
        }
        if config
            .device_fraction
            .is_some_and(|fraction| !(fraction > 0.0 && fraction <= 1.0))
        {
            return Err("device_fraction must be in (0, 1]".to_string());
        }
        let positive = [
            ("step_timeout_seconds", config.step_timeout_seconds),
            ("request_timeout_seconds", config.request_timeout_seconds),
            ("drain_seconds", config.drain_seconds),
            ("ephemeral_idle_seconds", config.ephemeral_idle_seconds),
            (
                "http.request_timeout_seconds",
                config
                    .http
                    .as_ref()
                    .map_or(1.0, |http| http.request_timeout_seconds),
            ),
        ];
        if let Some((name, _)) = positive
            .iter()
            .find(|(_, value)| !(value.is_finite() && *value > 0.0))
        {
            return Err(format!("{name} must be positive"));
        }
        let placement = &config.placement;
        if !(placement.rebalance_seconds.is_finite() && placement.rebalance_seconds >= 0.0) {
            return Err("placement.rebalance_seconds must be 0 or positive".to_string());
        }
        if !(placement.churn_budget > 0.0 && placement.churn_budget <= 1.0) {
            return Err("placement.churn_budget must be in (0, 1]".to_string());
        }
        if !(placement.imbalance_floor.is_finite() && placement.imbalance_floor >= 1.0) {
            return Err("placement.imbalance_floor must be at least 1".to_string());
        }
        let nonzero = [
            ("slot_bytes", config.slot_bytes),
            ("bucket_minimum", config.bucket_minimum as u64),
            (
                "connection_concurrency",
                config.connection_concurrency as u64,
            ),
            ("request_bytes_in_flight", config.request_bytes_in_flight),
            ("predict_queue_bytes", config.predict_queue_bytes),
            ("step_bytes", config.step_bytes.unwrap_or(1)),
            ("max_tables", config.max_tables as u64),
            ("ephemeral_tables", config.ephemeral_tables as u64),
            ("ephemeral_bytes", config.ephemeral_bytes),
        ];
        if let Some((name, _)) = nonzero.iter().find(|(_, value)| *value == 0) {
            return Err(format!("{name} must be positive"));
        }
        Ok(config)
    }

    /// Whether Flight may listen as configured: with tokens, a log key, and
    /// TLS beyond this machine.
    fn listening(&self) -> Result<(), String> {
        if self.tokens.is_empty() && self.replication_tokens.is_empty() {
            return Err("the configuration has no tokens".to_string());
        }
        let tokens = self.tokens.iter().chain(&self.replication_tokens);
        if tokens.clone().any(|token| token.token().is_empty()) {
            return Err("a token is empty".to_string());
        }
        if self.encryption_key_env.is_some() == self.encryption_key_file.is_some() {
            return Err(
                "give the log encryption key as encryption_key_env or encryption_key_file"
                    .to_string(),
            );
        }
        if self.tls.is_none() && !self.allow_plaintext && !loopback(&self.listen) {
            return Err(format!(
                "listening on {} without TLS sends tokens and rows in the clear; add [tls], \
                 or set allow_plaintext = true on a network you trust",
                self.listen
            ));
        }
        Ok(())
    }
}

/// Whether an address to listen on reaches this machine alone.
fn loopback(listen: &str) -> bool {
    match listen.parse::<SocketAddr>() {
        Ok(address) => address.ip().is_loopback(),
        Err(_) => listen
            .rsplit_once(':')
            .is_some_and(|(host, _)| host == "localhost"),
    }
}

/// A TOML error with the file and line, without the line's text, which the
/// error itself quotes and may be a token.
#[must_use]
pub fn toml_error(path: &Path, text: &str, error: &toml::de::Error) -> String {
    match error.span() {
        Some(span) => {
            let line = text[..span.start.min(text.len())].matches('\n').count() + 1;
            format!("{}:{line}: {}", path.display(), error.message())
        }
        None => format!("{}: {}", path.display(), error.message()),
    }
}
