// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deployed revisions, loaded on demand from the control plane that
//! publishes them, and the metrics their predictions report back to it.
//!
//! An environment resolves to its revision for a few seconds at a time, so
//! a promote or rollback is seen within that window. A revision's predictor
//! is held in a byte-budgeted LRU that also drops what sat idle; its rows
//! bind through `gf_core::predictor` into the same ephemeral tables, under the
//! same budget, as a request's own context.
//!
//! # The control plane's internal routes
//!
//! gf owns this contract; the control plane implements it. Every call
//! carries `Authorization: Bearer <service token>`, and a refusal answers a
//! 4xx with `{"error": {"code", "message", "param"?}}`, which gf-server
//! passes on to its caller as it came; anything else is `ModelNotReady`.
//!
//! - `GET /internal/v1/projects/{project}/predictors/{predictor}/environments/{environment}:revision`
//!   answers `{"revision": int, "context_version": any}`: the revision the
//!   environment serves now, and the version of its context.
//! - `GET /internal/v1/projects/{project}/predictors/{predictor}/revisions/{revision}:lock`
//!   answers the revision's `Published` package: `served`, the predictor as
//!   gf serve would serve it; `context`, its context as captured,
//!   `{"source", "name", "version", "digest", "as_of"}` (a live source's
//!   `source` may also be written `"live/<name>"`); and `rows`, the
//!   context's rows, which a request-context predictor has none of.
//! - The same route with `?as_of=<RFC 3339 UTC>`, such as
//!   `?as_of=2026-10-27T08:00:00Z`, answers the package of a live-context
//!   revision with its context as of that time: the rows and `version` of
//!   the context version that stood then, and that version's `as_of`. A
//!   time before the history the control plane keeps is refused with 422
//!   `ValidationError` (param `as_of`), or 409 `Conflict` where that
//!   version is gone; gf-server answers its caller the same. gf-server asks
//!   it only for a revision whose context is live: a prediction with
//!   `as_of` of any other context is refused with 422 `ValidationError`
//!   (param `as_of`) before the control plane is asked.
//! - `POST /internal/v1/predictions:record` takes `{"records": [...]}`, the
//!   predictions gf-server answered, for the control plane's metrics.
//!
//! `as_of` is read and written as `crate::as_of` has it, in every direction.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::as_of::{self, AsOf};
use crate::http::{ApiError, ServedModel};

/// What a predictor's predict route names.
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

/// A revision's locked predictor, as the control plane publishes it.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Published {
    /// A catalog predictor gf-server binds and predicts itself.
    Native {
        /// The predictor as `gf serve` would serve it.
        served: Box<ServedModel>,
        /// The context version a bound context reports, as the control
        /// plane captured it.
        #[serde(default)]
        context: Option<Value>,
        /// The bound context's rows; None for a request-context predictor.
        #[serde(default)]
        rows: Option<Vec<Value>>,
    },
}

/// What a revision's bound context was captured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    /// Rows locked with the revision, which never move.
    Snapshot,
    /// A source whose rows move under the same context version, so a held
    /// revision is captured again once its environment's resolution lapses.
    Live,
    /// A kind this server does not know, held as a snapshot is.
    Other,
}

/// Where a revision's bound context came from: its kind, and the source's
/// name where it has one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextSource {
    pub kind: SourceKind,
    pub name: Option<String>,
}

impl ContextSource {
    /// The source a published context names under `source`: the explicit
    /// pair `{"kind": ..., "name": ...}`, or the text the control plane
    /// writes, a kind alone or `<kind>/<name>`. None where it names none.
    #[must_use]
    pub fn of(context: &Value) -> Option<Self> {
        let source = context.get("source")?;
        let (kind, name) = match source {
            Value::Object(pair) => (
                pair.get("kind")?.as_str()?,
                pair.get("name").and_then(Value::as_str),
            ),
            Value::String(text) => match text.split_once('/') {
                Some((kind, name)) => (kind, Some(name)),
                None => (text.as_str(), None),
            },
            _ => return None,
        };
        let kind = match kind {
            "snapshot" => SourceKind::Snapshot,
            "live" => SourceKind::Live,
            _ => SourceKind::Other,
        };
        Some(Self {
            kind,
            name: name.filter(|name| !name.is_empty()).map(str::to_string),
        })
    }
}

impl Published {
    /// Whether the revision's context comes from a live source.
    #[must_use]
    pub fn live(&self) -> bool {
        match self {
            Self::Native {
                context: Some(context),
                ..
            } => ContextSource::of(context).is_some_and(|source| source.kind == SourceKind::Live),
            Self::Native { context: None, .. } => false,
        }
    }
}

/// One prediction, as the control plane's metrics and events count it.
#[derive(Clone, Debug, Serialize)]
pub struct Record {
    pub project: String,
    pub predictor: String,
    pub environment: Option<String>,
    pub revision: u64,
    pub ok: bool,
    pub latency_seconds: f64,
}

/// How [`Revisions`] reaches the control plane and what it holds.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Bytes of published predictors held at once.
    pub bytes: usize,
    /// How long a predictor nobody predicted from is held.
    pub idle: Duration,
    /// How long an environment's revision is trusted before it is asked again.
    pub resolve: Duration,
    /// How long one control-plane call may take.
    pub timeout: Duration,
}

/// A held predictor: project, predictor, revision and context version.
type Key = (String, String, u64, String);
/// An environment: project, predictor and name.
type Environment = (String, String, String);

struct Entry {
    published: std::sync::Arc<Published>,
    bytes: usize,
    used: Instant,
    /// When a live context, whose rows move under the same version, must
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
    /// fits; a predictor larger than the whole budget is served unheld.
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
            format!("{origin}/internal/v1/predictions:record"),
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

    /// How many predictors were fetched from the control plane so far.
    #[must_use]
    pub fn loads(&self) -> u64 {
        self.loads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The predictors held, and their bytes.
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
    /// The control plane's own refusal, or `ModelNotReady`.
    ///
    /// # Panics
    ///
    /// If the resolution cache's lock was poisoned.
    pub async fn resolve(
        &self,
        project: &str,
        predictor: &str,
        environment: &str,
    ) -> Result<Resolved, ApiError> {
        let key = (
            project.to_string(),
            predictor.to_string(),
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
                "/internal/v1/projects/{project}/predictors/{predictor}/environments/{environment}:revision"
            ))
            .await?;
        let mut held = self.resolved.lock().expect("resolved");
        held.retain(|_, (at, _)| now.duration_since(*at) < self.limits.resolve);
        held.insert(key, (now, resolved.clone()));
        Ok(resolved)
    }

    /// A revision's published predictor, fetched on a miss. `version` is the
    /// context version its environment resolved to, if a route named one.
    ///
    /// # Errors
    ///
    /// The control plane's own refusal, or `ModelNotReady`.
    ///
    /// # Panics
    ///
    /// If the cache's lock was poisoned.
    pub async fn load(
        &self,
        project: &str,
        predictor: &str,
        revision: u64,
        version: Option<&Value>,
    ) -> Result<std::sync::Arc<Published>, ApiError> {
        let version = version.map(Value::to_string).unwrap_or_default();
        self.package(project, predictor, revision, version, None)
            .await
    }

    /// A live-context revision's package with its context as of `as_of`,
    /// from the control plane's history; see the module's contract. A past
    /// version never changes, so it is held until it sits idle or is evicted.
    ///
    /// # Errors
    ///
    /// The control plane's own refusal, such as a time before the history
    /// it keeps, or `ModelNotReady`.
    ///
    /// # Panics
    ///
    /// If the cache's lock was poisoned.
    pub async fn load_as_of(
        &self,
        project: &str,
        predictor: &str,
        revision: u64,
        as_of: AsOf,
    ) -> Result<std::sync::Arc<Published>, ApiError> {
        let at = as_of.rfc3339();
        self.package(
            project,
            predictor,
            revision,
            format!("{}={at}", as_of::FIELD),
            Some(at),
        )
        .await
    }

    /// The package held under `version`, or fetched: as of `at` if given.
    async fn package(
        &self,
        project: &str,
        predictor: &str,
        revision: u64,
        version: String,
        at: Option<String>,
    ) -> Result<std::sync::Arc<Published>, ApiError> {
        let key = (
            project.to_string(),
            predictor.to_string(),
            revision,
            version,
        );
        if let Some(published) = self
            .cache
            .lock()
            .expect("revision cache")
            .get(&key, Instant::now())
        {
            return Ok(published);
        }
        let mut path = format!(
            "/internal/v1/projects/{project}/predictors/{predictor}/revisions/{revision}:lock"
        );
        if let Some(at) = &at {
            path.push_str(&format!("?{}={at}", as_of::FIELD));
        }
        let (published, bytes): (Published, usize) = self.fetch(&path).await?;
        self.loads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let now = Instant::now();
        // A live source's rows move under one version, unless they are the
        // ones that stood at a past time.
        let live = at.is_none() && published.live();
        let published = std::sync::Arc::new(published);
        self.cache.lock().expect("revision cache").insert(
            key,
            Entry {
                published: published.clone(),
                bytes,
                used: now,
                stale: live.then(|| now + self.limits.resolve),
            },
            &self.limits,
            now,
        );
        Ok(published)
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
    /// `ModelNotReady`.
    async fn read<T: DeserializeOwned>(
        &self,
        answer: reqwest::Result<reqwest::Response>,
    ) -> Result<(T, usize), ApiError> {
        let unavailable = || ApiError::new("ModelNotReady", "control plane is unavailable", 503);
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
                    let refused = ApiError::new(code, message, status);
                    Err(match error["param"].as_str() {
                        Some(param) => refused.param(param),
                        None => refused,
                    })
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
            published: std::sync::Arc::new(
                serde_json::from_value(json!({
                    "kind": "native",
                    "served": {
                        "name": "churn",
                        "reference": "general-forecast/tabicl-v2-classifier",
                        "worker": "churn",
                        "resolved": "general-forecast/tabicl-v2-classifier",
                        "outputs": [],
                        "description": {},
                    },
                }))
                .expect("a native published revision"),
            ),
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

    #[test]
    fn a_context_source_reads_as_an_explicit_pair_or_as_the_planes_text() {
        let source = |value: Value| ContextSource::of(&json!({"source": value}));
        let live = |name: Option<&str>| {
            Some(ContextSource {
                kind: SourceKind::Live,
                name: name.map(str::to_string),
            })
        };
        assert_eq!(source(json!("live/orders")), live(Some("orders")));
        assert_eq!(
            source(json!({"kind": "live", "name": "orders"})),
            live(Some("orders"))
        );
        assert_eq!(source(json!("live")), live(None));
        assert_eq!(source(json!({"kind": "live"})), live(None));
        assert_eq!(source(json!("live/")), live(None));
        assert_eq!(
            source(json!("snapshot")).map(|source| source.kind),
            Some(SourceKind::Snapshot)
        );
        // A name that starts like a kind is not that kind.
        for other in [
            json!("liveliness/x"),
            json!("dataset"),
            json!({"kind": "lived"}),
        ] {
            assert_eq!(
                source(other).map(|source| source.kind),
                Some(SourceKind::Other)
            );
        }
        assert_eq!(ContextSource::of(&json!({"rows": 4})), None);
        assert_eq!(source(json!(7)), None);
    }
}
