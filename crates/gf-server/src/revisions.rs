//! Deployed revisions, loaded on demand from the control plane that
//! publishes them, and the metrics their predictions report back to it.
//!
//! An environment resolves to its revision for a few seconds at a time, so
//! a promote or rollback is seen within that window. A revision's package
//! is held in a byte-budgeted LRU that also drops what sat idle; its rows
//! bind through `gf_core::package` into the same ephemeral tables, under the
//! same budget, as a request's own context.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::http::{ApiError, ServedModel};

/// What a deployment's predict route names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selector {
    Environment(String),
    Revision(u64),
}

/// The revision an environment serves, and the version of its context.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resolved {
    pub revision: u64,
    #[serde(default)]
    pub context_version: Option<Value>,
}

/// A revision's locked package, as the control plane publishes it.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Published {
    /// A catalog package gf-server binds and predicts itself.
    Native {
        /// The package as `gf serve` would serve it.
        served: Box<ServedModel>,
        /// The context version a bound context reports, as the control
        /// plane captured it.
        #[serde(default)]
        context: Option<Value>,
        /// The bound context's rows; None for a request-context package.
        #[serde(default)]
        rows: Option<Vec<Value>>,
    },
    /// A Python package, which only the control plane's isolated runtime
    /// may execute.
    Python {},
}

/// One prediction, as the control plane's metrics and events count it.
#[derive(Clone, Debug, Serialize)]
pub struct Record {
    pub project: String,
    pub deployment: String,
    pub environment: Option<String>,
    pub revision: u64,
    pub ok: bool,
    pub latency_seconds: f64,
}

/// How [`Revisions`] reaches the control plane and what it holds.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Bytes of published packages held at once.
    pub bytes: usize,
    /// How long a package nobody predicted from is held.
    pub idle: Duration,
    /// How long an environment's revision is trusted before it is asked again.
    pub resolve: Duration,
    /// How long one control-plane call may take.
    pub timeout: Duration,
}

/// A held package: project, deployment, revision and context version.
type Key = (String, String, u64, String);
/// An environment: project, deployment and name.
type Environment = (String, String, String);

struct Entry {
    published: std::sync::Arc<Published>,
    bytes: usize,
    used: Instant,
    /// When a dataset context, whose rows move under the same version, must
    /// be captured again.
    stale: Option<Instant>,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<Key, Entry>,
    bytes: usize,
}

impl Cache {
    fn get(&mut self, key: &Key, now: Instant) -> Option<std::sync::Arc<Published>> {
        let entry = self.entries.get_mut(key)?;
        if entry.stale.is_some_and(|stale| stale <= now) {
            let bytes = entry.bytes;
            self.entries.remove(key);
            self.bytes -= bytes;
            return None;
        }
        entry.used = now;
        Some(entry.published.clone())
    }

    /// Drops what sat idle, then the least recently used, until `incoming`
    /// fits; a package larger than the whole budget is served unheld.
    fn insert(&mut self, key: Key, entry: Entry, limits: &Limits, now: Instant) {
        let idle: Vec<Key> = self
            .entries
            .iter()
            .filter(|(_, held)| held.used + limits.idle <= now)
            .map(|(key, _)| key.clone())
            .collect();
        for key in idle {
            self.remove(&key);
        }
        if entry.bytes > limits.bytes {
            return;
        }
        self.remove(&key);
        while self.bytes + entry.bytes > limits.bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, held)| held.used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            self.remove(&oldest);
        }
        self.bytes += entry.bytes;
        self.entries.insert(key, entry);
    }

    fn remove(&mut self, key: &Key) {
        if let Some(entry) = self.entries.remove(key) {
            self.bytes -= entry.bytes;
        }
    }
}

/// The deployed revisions a front predicts from, by the control plane at
/// `origin`.
pub struct Revisions {
    client: reqwest::Client,
    origin: String,
    token: String,
    limits: Limits,
    resolved: Mutex<HashMap<Environment, (Instant, Resolved)>>,
    cache: Mutex<Cache>,
    loads: std::sync::atomic::AtomicU64,
    records: mpsc::Sender<Record>,
}

/// The largest control-plane answer read: a snapshot's rows as JSON.
const MAX_ANSWER: usize = 256 * 1024 * 1024;
/// Records sent in one `predictions:record` call.
const BATCH: usize = 256;

impl Revisions {
    /// Reaches the control plane at `origin` with its service `token`, and
    /// starts reporting predictions to it on the current runtime.
    ///
    /// # Errors
    ///
    /// Why `origin` is not a bare HTTP(S) origin.
    pub fn new(origin: &str, token: String, limits: Limits) -> Result<Self, String> {
        let origin = crate::upstream::origin(origin)?;
        let client = reqwest::Client::builder()
            .timeout(limits.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| format!("revisions client: {error}"))?;
        let (records, pending) = mpsc::channel(4 * BATCH);
        tokio::spawn(report(
            client.clone(),
            format!("{origin}/internal/v1alpha1/predictions:record"),
            token.clone(),
            pending,
        ));
        Ok(Self {
            client,
            origin,
            token,
            limits,
            resolved: Mutex::new(HashMap::new()),
            cache: Mutex::new(Cache::default()),
            loads: std::sync::atomic::AtomicU64::new(0),
            records,
        })
    }

    /// How many packages were fetched from the control plane so far.
    #[must_use]
    pub fn loads(&self) -> u64 {
        self.loads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The packages held, and their bytes.
    ///
    /// # Panics
    ///
    /// If the cache's lock was poisoned.
    #[must_use]
    pub fn held(&self) -> (usize, usize) {
        let cache = self.cache.lock().expect("revision cache");
        (cache.entries.len(), cache.bytes)
    }

    /// The revision `environment` serves, from the last few seconds' answer.
    ///
    /// # Errors
    ///
    /// The control plane's own refusal, or `RuntimeUnavailable`.
    ///
    /// # Panics
    ///
    /// If the resolution cache's lock was poisoned.
    pub async fn resolve(
        &self,
        project: &str,
        deployment: &str,
        environment: &str,
    ) -> Result<Resolved, ApiError> {
        let key = (
            project.to_string(),
            deployment.to_string(),
            environment.to_string(),
        );
        let now = Instant::now();
        if let Some((at, resolved)) = self.resolved.lock().expect("resolved").get(&key)
            && now.duration_since(*at) < self.limits.resolve
        {
            return Ok(resolved.clone());
        }
        let resolved: Resolved = self
            .get(&format!(
                "/internal/v1alpha1/projects/{project}/deployments/{deployment}/environments/{environment}:revision"
            ))
            .await?;
        let mut held = self.resolved.lock().expect("resolved");
        held.retain(|_, (at, _)| now.duration_since(*at) < self.limits.resolve);
        held.insert(key, (now, resolved.clone()));
        Ok(resolved)
    }

    /// A revision's published package, fetched on a miss. `version` is the
    /// context version its environment resolved to, if a route named one.
    ///
    /// # Errors
    ///
    /// The control plane's own refusal, or `RuntimeUnavailable`.
    ///
    /// # Panics
    ///
    /// If the cache's lock was poisoned.
    pub async fn load(
        &self,
        project: &str,
        deployment: &str,
        revision: u64,
        version: Option<&Value>,
    ) -> Result<std::sync::Arc<Published>, ApiError> {
        let key = (
            project.to_string(),
            deployment.to_string(),
            revision,
            version.map(Value::to_string).unwrap_or_default(),
        );
        if let Some(published) = self
            .cache
            .lock()
            .expect("revision cache")
            .get(&key, Instant::now())
        {
            return Ok(published);
        }
        let (published, bytes): (Published, usize) = self
            .fetch(&format!(
                "/internal/v1alpha1/projects/{project}/deployments/{deployment}/revisions/{revision}:package"
            ))
            .await?;
        self.loads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let now = Instant::now();
        let dataset = matches!(
            &published,
            Published::Native { context: Some(context), .. } if context["source"] == "dataset"
        );
        let published = std::sync::Arc::new(published);
        self.cache.lock().expect("revision cache").insert(
            key,
            Entry {
                published: published.clone(),
                bytes,
                used: now,
                stale: dataset.then(|| now + self.limits.resolve),
            },
            &self.limits,
            now,
        );
        Ok(published)
    }

    /// A Python revision's prediction, from the isolated runtime the control
    /// plane executes it in.
    ///
    /// # Errors
    ///
    /// The control plane's own refusal, or `RuntimeUnavailable`.
    pub async fn execute(
        &self,
        project: &str,
        deployment: &str,
        revision: u64,
        environment: Option<&str>,
        payload: &Value,
    ) -> Result<Value, ApiError> {
        let answer = self
            .client
            .post(format!(
                "{}/internal/v1alpha1/projects/{project}/deployments/{deployment}/revisions/{revision}:execute",
                self.origin
            ))
            .bearer_auth(&self.token)
            .json(&json!({"environment": environment, "payload": payload}))
            .send()
            .await;
        let (answer, _): (Executed, usize) = self.read(answer).await?;
        Ok(answer.result)
    }

    /// Counts one prediction for the control plane, off the request's path;
    /// a full queue drops it rather than slow predictions down.
    pub fn record(&self, record: Record) {
        if self.records.try_send(record).is_err() {
            tracing::warn!("prediction records are backed up; dropping one");
        }
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, ApiError> {
        Ok(self.fetch(path).await?.0)
    }

    async fn fetch<T: DeserializeOwned>(&self, path: &str) -> Result<(T, usize), ApiError> {
        let answer = self
            .client
            .get(format!("{}{path}", self.origin))
            .bearer_auth(&self.token)
            .send()
            .await;
        self.read(answer).await
    }

    /// A control-plane answer, its refusals kept and everything else
    /// `RuntimeUnavailable`.
    async fn read<T: DeserializeOwned>(
        &self,
        answer: reqwest::Result<reqwest::Response>,
    ) -> Result<(T, usize), ApiError> {
        let unavailable =
            || ApiError::new("RuntimeUnavailable", "control plane is unavailable", 503);
        let mut answer = answer.map_err(|error| {
            tracing::warn!(%error, "control plane call failed");
            unavailable()
        })?;
        let status = answer.status().as_u16();
        let mut bytes = Vec::new();
        while let Some(chunk) = answer.chunk().await.map_err(|_| unavailable())? {
            if bytes.len() + chunk.len() > MAX_ANSWER {
                return Err(unavailable());
            }
            bytes.extend_from_slice(&chunk);
        }
        if status != 200 {
            let refused: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            let error = &refused["error"];
            return match (error["code"].as_str(), error["message"].as_str()) {
                (Some(code), Some(message)) if (400..600).contains(&status) && status != 401 => {
                    Err(ApiError::new(code, message, status))
                }
                _ => {
                    tracing::warn!(status, "control plane refused gf-server");
                    Err(unavailable())
                }
            };
        }
        let size = bytes.len();
        let parsed = serde_json::from_slice(&bytes).map_err(|error| {
            tracing::warn!(%error, "control plane answered unreadably");
            unavailable()
        })?;
        Ok((parsed, size))
    }
}

#[derive(Deserialize)]
struct Executed {
    result: Value,
}

/// Sends records in batches until the front is dropped.
async fn report(
    client: reqwest::Client,
    url: String,
    token: String,
    mut pending: mpsc::Receiver<Record>,
) {
    let mut batch = Vec::with_capacity(BATCH);
    while pending.recv_many(&mut batch, BATCH).await > 0 {
        let sent = client
            .post(&url)
            .bearer_auth(&token)
            .json(&json!({"records": batch}))
            .send()
            .await;
        match sent {
            Ok(answer) if answer.status().is_success() => {}
            Ok(answer) => tracing::warn!(status = %answer.status(), "prediction records refused"),
            Err(error) => tracing::warn!(%error, "prediction records not sent"),
        }
        batch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(bytes: usize, used: Instant) -> Entry {
        Entry {
            published: std::sync::Arc::new(Published::Python {}),
            bytes,
            used,
            stale: None,
        }
    }

    fn key(revision: u64) -> Key {
        ("p".into(), "d".into(), revision, String::new())
    }

    fn limits() -> Limits {
        Limits {
            bytes: 100,
            idle: Duration::from_secs(60),
            resolve: Duration::from_secs(5),
            timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn the_least_recently_used_package_goes_first_to_fit_the_budget() {
        let mut cache = Cache::default();
        let now = Instant::now();
        cache.insert(key(1), entry(40, now), &limits(), now);
        cache.insert(
            key(2),
            entry(40, now),
            &limits(),
            now + Duration::from_millis(1),
        );
        assert!(cache.get(&key(1), now + Duration::from_millis(2)).is_some());
        cache.insert(
            key(3),
            entry(40, now),
            &limits(),
            now + Duration::from_millis(3),
        );
        assert!(cache.get(&key(2), now).is_none());
        assert!(cache.get(&key(1), now).is_some());
        assert_eq!(cache.bytes, 80);
    }

    #[test]
    fn idle_packages_are_dropped_and_an_oversized_one_is_never_held() {
        let mut cache = Cache::default();
        let now = Instant::now();
        cache.insert(key(1), entry(10, now), &limits(), now);
        let later = now + Duration::from_secs(61);
        cache.insert(key(2), entry(10, later), &limits(), later);
        assert!(cache.get(&key(1), later).is_none());
        cache.insert(key(3), entry(101, later), &limits(), later);
        assert!(cache.get(&key(3), later).is_none());
        assert_eq!((cache.entries.len(), cache.bytes), (1, 10));
    }

    #[test]
    fn a_stale_dataset_context_is_fetched_again() {
        let mut cache = Cache::default();
        let now = Instant::now();
        let mut stale = entry(10, now);
        stale.stale = Some(now + Duration::from_secs(5));
        cache.insert(key(1), stale, &limits(), now);
        assert!(cache.get(&key(1), now + Duration::from_secs(1)).is_some());
        assert!(cache.get(&key(1), now + Duration::from_secs(5)).is_none());
        assert_eq!(cache.bytes, 0);
    }
}
