// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The Models API over HTTP: the routes, envelopes, headers and errors the
//! hosted API and `gf serve` answer, in front of the server's shards.
//!
//! A request is admitted against the server's request byte budget, checked
//! by the one validator, `gf_core::request`, and answered from its
//! context's ephemeral table on the shard its fingerprint names, so
//! concurrent requests share steps and repeated contexts reuse their fits.
//! Bearer keys are checked by a `KeyStore`: one static key, or the control
//! plane's introspection behind a short cache.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::Response;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use gf_core::catalog::{self, Surface};

use gf_core::forecasting::KNOWN;
use gf_core::node::Targets;
use gf_core::request::{
    self, ErrorDetail, Forecast, Inference, Label, Limits, Options, RequestError,
};
use serde::Deserialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::auth::digest;
use crate::engine::{self, Bound, Engine};
use crate::name::TableName;
use crate::predictors::{self, ContextMode, ForecastPredictor, ServedPredictor};
use crate::progress::Compiling;
use crate::revisions::{Published, Record, Revisions};
use crate::routes::Routes;
use crate::service::{Charge, RequestBudget, quantile_values};
use crate::setup::random_hex;
use crate::shard::{Asked, Command, Inferred, ServerError};
use crate::upstream::Upstream;
use arrow_array::RecordBatch;
use gf_core::predictor;

/// The API version every envelope carries.
pub const VERSION: &str = "gf/v1";
/// The most bytes a request body may hold.
pub const MAX_BYTES: usize = 1_048_576;
/// The deepest and largest JSON a body may be, as the hosted API bounds it.
const MAX_DEPTH: usize = 16;
const MAX_NODES: usize = 65_536;
const MAX_INTEGER: i64 = (1 << 53) - 1;
/// The first path segment of the control plane's service-to-service routes.
const INTERNAL: &str = "internal";
/// The kinds of resource an envelope may name.
const RESOURCE_KINDS: [&str; 9] = [
    "workspace",
    "project",
    "context",
    "source",
    "model",
    "predictor",
    "operator",
    "api-key",
    "secret",
];
/// The most query and context rows a predictor's `/predictions` takes.
const MAX_QUERY_ROWS: usize = 10_000;
const MAX_CONTEXT_ROWS: usize = 100_000;
/// The quantiles a regressor's `/predictions` reports when it asks for none.
const DEFAULT_QUANTILES: [f64; 3] = [0.1, 0.5, 0.9];
/// The (context rows, queries) buckets `Front::warm` compiles at startup:
/// the shapes a first request most often takes.
const WARM_SHAPES: &[(usize, usize)] = &[(100, 1), (1_000, 1), (10_000, 1)];
/// How many input columns the warm contexts have.
const WARM_INPUTS: usize = 2;
/// How long one warm bucket may take to compile and run.
const WARM_TIMEOUT: Duration = Duration::from_secs(600);

/// A synthetic ad hoc body of `context_rows` rows over `WARM_INPUTS`
/// deterministic pseudo-uniform inputs and a 0/1 (or real) target; `seed`
/// varies the rows, so two bodies are two contexts.
fn warm_payload(
    model: &str,
    context_rows: usize,
    queries: usize,
    regression: bool,
    seed: u64,
) -> Value {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        unit
    };
    let row = |uniform: &mut dyn FnMut() -> f64, target: bool| {
        let a = uniform();
        let b = uniform();
        let mut row = json!({"a": a, "b": b});
        if target {
            row["y"] = if regression {
                json!(a + b)
            } else {
                json!(i32::from(a + b > 1.0))
            };
        }
        row
    };
    let context: Vec<Value> = (0..context_rows).map(|_| row(&mut uniform, true)).collect();
    let queries: Vec<Value> = (0..queries).map(|_| row(&mut uniform, false)).collect();
    json!({
        "model": model,
        "target": "y",
        "inputs": ["a", "b"],
        "context": context,
        "queries": queries,
    })
}
/// How long introspection answers are trusted: valid keys, and refused ones.
const POSITIVE_TTL: Duration = Duration::from_secs(30);
const NEGATIVE_TTL: Duration = Duration::from_secs(5);
/// The most introspection answers kept; past it the cache starts over.
const CACHED_KEYS: usize = 65_536;

/// A safe public error: its code, message and HTTP status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub status: u16,
    /// The optional fields of the error object (`param`, `hint`, `limit`,
    /// `actual`), boxed so the common error stays small.
    pub detail: Option<Box<ErrorDetail>>,
}

impl ApiError {
    #[must_use]
    pub fn new(code: &str, message: impl Into<String>, status: u16) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            status,
            detail: None,
        }
    }

    fn detail_mut(&mut self) -> &mut ErrorDetail {
        self.detail.get_or_insert_with(Box::default)
    }

    /// Names the field at fault.
    #[must_use]
    pub fn param(mut self, param: impl Into<String>) -> Self {
        self.detail_mut().param = Some(param.into());
        self
    }

    /// Says what to do instead.
    #[must_use]
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.detail_mut().hint = Some(hint.into());
        self
    }

    /// Malformed input, as the hosted API's `require` refuses it.
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::new("ValidationError", message, 422)
    }

    fn not_served() -> Self {
        Self::new("NotFound", "model is not served here", 404)
    }

    fn unauthorized(message: &str) -> Self {
        Self::new("Unauthorized", message, 401)
    }

    fn body(&self) -> Value {
        let mut error = json!({"code": self.code, "message": self.message});
        if let Some(detail) = &self.detail {
            let fields = [
                ("param", detail.param.as_ref().map(|value| json!(value))),
                ("hint", detail.hint.as_ref().map(|value| json!(value))),
                ("limit", detail.limit.map(|value| json!(value))),
                ("actual", detail.actual.map(|value| json!(value))),
            ];
            for (key, value) in fields {
                if let Some(value) = value {
                    error[key] = value;
                }
            }
        }
        json!({"error": error})
    }
}

impl From<RequestError> for ApiError {
    fn from(error: RequestError) -> Self {
        Self {
            code: error.code.to_string(),
            message: error.message,
            status: error.status,
            detail: error.detail,
        }
    }
}

impl From<ServerError> for ApiError {
    fn from(error: ServerError) -> Self {
        match error {
            ServerError::Busy(message) => Self::new("CapacityExceeded", message, 503),
            ServerError::Invalid(message) => Self::new("InvalidContext", message, 422),
            ServerError::Unavailable(_) => Self::new(
                "ModelUnavailable",
                "the model is restarting; retry shortly",
                503,
            ),
            error => {
                tracing::error!("inference failed: {error}");
                Self::new("InferenceFailed", "native inference failed", 503)
            }
        }
    }
}

fn require(condition: bool, message: &str) -> Result<(), ApiError> {
    if condition {
        Ok(())
    } else {
        Err(ApiError::validation(message))
    }
}

/// One model the front serves, as `gf serve` or the hosted runtime writes
/// it from the catalog when the server starts.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServedModel {
    /// The model resource's name, as routes name it.
    pub name: String,
    /// The reference served, in its canonical spelling.
    pub reference: String,
    /// The worker's name for the model.
    pub worker: String,
    /// The pinned identity `/predictions` reports as its model.
    pub resolved: String,
    /// The outputs `/predictions` may select.
    pub outputs: Vec<String>,
    /// The `GET models/{name}` result, as the catalog describes the model.
    pub description: Value,
    /// The file declaring the model's license, which a forecast's artifact
    /// reports, as the catalog's forecasts do.
    #[serde(default)]
    pub license_source: Option<String>,
    /// The locked predictor served, if the model is one.
    #[serde(default)]
    pub predictor: Option<ServedPredictor>,
    /// The one project whose routes reach this predictor, or None for
    /// every project, as on a server that serves one owner alone.
    #[serde(default)]
    pub project: Option<String>,
}

/// Who a bearer key belongs to, and what it may do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    /// The one project the key reaches, or None for every project.
    pub project: Option<String>,
    /// What the key may do, or None for everything.
    pub permissions: Option<Vec<String>>,
}

impl Principal {
    /// A principal that reaches everything, as a server without keys has.
    #[must_use]
    pub fn operator() -> Self {
        Self {
            project: None,
            permissions: None,
        }
    }

    fn authorize(&self, project: &str, permission: &str) -> Result<(), ApiError> {
        let scoped = self.project.as_deref().is_none_or(|own| own == project);
        let allowed = self
            .permissions
            .as_ref()
            .is_none_or(|permissions| permissions.iter().any(|held| held == permission));
        if scoped && allowed {
            Ok(())
        } else {
            Err(ApiError::new(
                "Forbidden",
                "permission denied for this scope",
                403,
            ))
        }
    }
}

/// Checks the bearer key a request presents.
pub trait KeyStore: Send + Sync {
    /// The principal behind the `Authorization` header's value, if any.
    fn verify<'a>(
        &'a self,
        authorization: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Principal, ApiError>>;

    /// What clients must present: `bearer`, or `none` without a key.
    fn auth(&self) -> &'static str;
}

/// The key a `Bearer` header presents.
fn presented(authorization: Option<&str>) -> Result<&str, ApiError> {
    let (scheme, key) = authorization
        .unwrap_or_default()
        .split_once(' ')
        .unwrap_or_default();
    if scheme != "Bearer" || key.is_empty() || key.contains(' ') {
        return Err(ApiError::unauthorized("application bearer required"));
    }
    Ok(key)
}

/// One key for every project, as `gf serve` takes it from `GF_API_KEY`; or
/// none, where only this machine, or a network the operator trusts, reaches
/// the server.
pub struct Static {
    /// The key's digest: comparing digests takes the same time for every guess.
    key: Option<[u8; 32]>,
    /// Who the key is.
    principal: Principal,
}

impl Static {
    #[must_use]
    pub fn new(key: Option<&str>) -> Self {
        Self::scoped(key, Principal::operator())
    }

    /// One key that reaches only what `principal` may; without a key,
    /// every request is `principal`.
    #[must_use]
    pub fn scoped(key: Option<&str>, principal: Principal) -> Self {
        Self {
            key: key.map(digest),
            principal,
        }
    }
}

impl KeyStore for Static {
    fn verify<'a>(
        &'a self,
        authorization: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Principal, ApiError>> {
        Box::pin(async move {
            let Some(key) = &self.key else {
                return Ok(self.principal.clone());
            };
            let given = digest(presented(authorization)?);
            let differs = given
                .iter()
                .zip(key)
                .fold(0u8, |acc, (left, right)| acc | (left ^ right));
            if differs == 0 {
                Ok(self.principal.clone())
            } else {
                Err(ApiError::unauthorized("invalid API key"))
            }
        })
    }

    fn auth(&self) -> &'static str {
        if self.key.is_some() { "bearer" } else { "none" }
    }
}

/// The control plane's keys, checked through its introspection endpoint
/// under a service token, with answers cached briefly: valid keys for 30
/// seconds or until they expire, refused ones for 5. Concurrent requests
/// presenting one key share one introspection call.
pub struct Introspect {
    client: reqwest::Client,
    url: Arc<str>,
    token: Arc<str>,
    cache: Mutex<HashMap<[u8; 32], Cached>>,
    asking: Mutex<HashMap<[u8; 32], Asking>>,
}

/// An introspection answer, and until when it is trusted.
type Cached = (Option<Principal>, Instant);

/// An introspection call under way, which every request presenting its key awaits.
type Asking = Shared<BoxFuture<'static, Result<(Option<Principal>, Duration), ApiError>>>;

/// What the introspection endpoint answers for a valid key.
#[derive(Deserialize)]
struct Introspected {
    /// Whether the key is live; a refused key answers `active: false` alone.
    active: bool,
    #[serde(default)]
    project: String,
    #[serde(default)]
    permissions: Vec<String>,
    /// When the key expires, in seconds since 1970, if it does.
    #[serde(default)]
    expires: Option<f64>,
}

impl Introspect {
    /// Checks keys against the control plane at `base`, such as
    /// `https://api.generalforecast.com`, under `token`.
    ///
    /// # Errors
    ///
    /// If the HTTP client cannot be built.
    pub fn new(base: &str, token: String) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            client,
            url: format!("{}/internal/v1/keys:introspect", base.trim_end_matches('/')).into(),
            token: token.into(),
            cache: Mutex::new(HashMap::new()),
            asking: Mutex::new(HashMap::new()),
        })
    }

    fn cached(&self, verifier: &[u8; 32]) -> Option<Option<Principal>> {
        let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .get(verifier)
            .filter(|(_, until)| *until > Instant::now())
            .map(|(principal, _)| principal.clone())
    }

    /// Caches an answer. A full cache sheds what has expired, then the
    /// refused keys, which a caller presenting random keys fills it with,
    /// and only then the valid keys it still holds.
    fn remember(&self, verifier: [u8; 32], principal: Option<Principal>, ttl: Duration) {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if cache.len() >= CACHED_KEYS {
            let now = Instant::now();
            cache.retain(|_, (_, until)| *until > now);
        }
        if cache.len() >= CACHED_KEYS {
            cache.retain(|_, (principal, _)| principal.is_some());
        }
        if cache.len() >= CACHED_KEYS {
            cache.clear();
        }
        cache.insert(verifier, (principal, Instant::now() + ttl));
    }

    /// The introspection call for `verifier`: the one under way, or a new one.
    fn asking(&self, verifier: [u8; 32]) -> Asking {
        let mut asking = self.asking.lock().unwrap_or_else(PoisonError::into_inner);
        asking
            .entry(verifier)
            .or_insert_with(|| {
                ask(
                    self.client.clone(),
                    self.url.clone(),
                    self.token.clone(),
                    verifier,
                )
                .boxed()
                .shared()
            })
            .clone()
    }

    /// Forgets a finished call, unless a later one for the key took its place.
    fn asked(&self, verifier: &[u8; 32], call: &Asking) {
        let mut asking = self.asking.lock().unwrap_or_else(PoisonError::into_inner);
        if asking
            .get(verifier)
            .is_some_and(|current| current.ptr_eq(call))
        {
            asking.remove(verifier);
        }
    }
}

/// Asks the control plane who `verifier`'s key is, and for how long to trust the answer.
async fn ask(
    client: reqwest::Client,
    url: Arc<str>,
    token: Arc<str>,
    verifier: [u8; 32],
) -> Result<(Option<Principal>, Duration), ApiError> {
    let unavailable = || ApiError::new("CredentialUnavailable", "cannot check API keys now", 503);
    let response = client
        .post(&*url)
        .bearer_auth(&token)
        .json(&json!({"verifier": request::hex(&verifier)}))
        .send()
        .await
        .map_err(|_| unavailable())?;
    // Unknown keys answer 200 `{"active": false}`; any other status says
    // the control plane or this server's service token is at fault, not
    // the key, so it is neither cached nor answered as a bad key.
    if response.status() != StatusCode::OK {
        return Err(unavailable());
    }
    let found: Introspected = response.json().await.map_err(|_| unavailable())?;
    if !found.active {
        return Ok((None, NEGATIVE_TTL));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let left = match found.expires {
        Some(expires) if expires <= now.as_secs_f64() => return Ok((None, NEGATIVE_TTL)),
        // An expiry too far off for a duration is as good as none.
        Some(expires) => {
            Duration::try_from_secs_f64(expires - now.as_secs_f64()).unwrap_or(POSITIVE_TTL)
        }
        None => POSITIVE_TTL,
    };
    let principal = Principal {
        project: Some(found.project),
        permissions: Some(found.permissions),
    };
    Ok((Some(principal), left.min(POSITIVE_TTL)))
}

impl KeyStore for Introspect {
    fn verify<'a>(
        &'a self,
        authorization: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Principal, ApiError>> {
        Box::pin(async move {
            let verifier = digest(presented(authorization)?);
            let principal = match self.cached(&verifier) {
                Some(principal) => principal,
                None => {
                    let call = self.asking(verifier);
                    let answered = call.clone().await;
                    self.asked(&verifier, &call);
                    let (principal, ttl) = answered?;
                    self.remember(verifier, principal.clone(), ttl);
                    principal
                }
            };
            principal.ok_or_else(|| ApiError::unauthorized("invalid API key"))
        })
    }

    fn auth(&self) -> &'static str {
        "bearer"
    }
}

/// A request's correlation and resource, as its envelope names them.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Scope {
    project: String,
    request_id: String,
    kind: String,
    name: String,
}

/// The Models API actions.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Action {
    /// The catalog models served, with their limits.
    List,
    Get,
    /// The predictors served, declared in a lock.
    Predictors,
    Predictor,
    /// A declared predictor's prediction, `POST predictors/{name}/predict`.
    PredictorPredict,
    /// An ad hoc prediction from a manifest fragment, `POST predict`.
    AdHoc,
}

impl Action {
    fn permission(&self) -> &'static str {
        match self {
            Self::List | Self::Get | Self::Predictors | Self::Predictor => "models:read",
            Self::PredictorPredict | Self::AdHoc => "predictors:predict",
        }
    }

    /// The resource kind the action's envelope names.
    fn kind(&self) -> &'static str {
        match self {
            Self::Predictors | Self::Predictor | Self::PredictorPredict | Self::AdHoc => {
                "predictor"
            }
            Self::List | Self::Get => "model",
        }
    }
}

/// What the HTTP front serves, and with what bounds.
pub struct Front {
    models: Vec<ServedModel>,
    keys: Box<dyn KeyStore>,
    surface: Surface,
    /// Whether the front listens on this machine alone, where it refuses
    /// requests a browser could forge from another origin.
    loopback: bool,
    /// The exact hosts requests must name, if any; see `with_hosts`.
    hosts: Vec<String>,
    budget: RequestBudget,
    timeout: Duration,
    engine: OnceLock<Engine>,
    /// Snapshot predictors' tables, by predictor name, once they are bound.
    bound: OnceLock<HashMap<String, Bound>>,
    /// Snapshot forecast predictors' series, by predictor name, once aligned.
    series: OnceLock<HashMap<String, SeriesBound>>,
    draining: Arc<AtomicBool>,
    /// Where the routes the front does not own stream to, if anywhere.
    upstream: Option<Upstream>,
    /// The deployed revisions predicted from, if any are.
    revisions: Option<Revisions>,
    /// The shape buckets compiled at startup, which `GET models` reports as
    /// `warm_shapes`.
    warm: OnceLock<Vec<Value>>,
}

/// A forecast predictor's context: its series aligned, and their forecast
/// request parsed, once.
struct SeriesBound {
    context: predictors::ForecastContext,
    forecast: Forecast,
    rows: usize,
}

impl Front {
    #[must_use]
    pub fn new(
        models: Vec<ServedModel>,
        keys: Box<dyn KeyStore>,
        surface: Surface,
        loopback: bool,
        budget: RequestBudget,
        timeout: Duration,
        draining: Arc<AtomicBool>,
    ) -> Self {
        Self {
            models,
            keys,
            surface,
            loopback,
            hosts: Vec::new(),
            budget,
            timeout,
            engine: OnceLock::new(),
            bound: OnceLock::new(),
            series: OnceLock::new(),
            draining,
            upstream: None,
            revisions: None,
            warm: OnceLock::new(),
        }
    }

    /// Compiles the common shape buckets for every tabular catalog model
    /// served, by scoring a synthetic context in each, so the first request
    /// does not pay for the compile; the buckets go to `GET models` as
    /// `warm_shapes`. Each bucket is scored once per shard at the same time,
    /// pinned to it, so every worker compiles it. A bucket that fails is
    /// logged and skipped, so a
    /// slow compile never keeps the server from starting. Returns how many
    /// buckets compiled.
    pub async fn warm(&self, compiling: &Compiling) -> usize {
        let mut shapes = Vec::new();
        let Ok(engine) = self.engine() else {
            return 0;
        };
        let engine = engine.with_timeout(WARM_TIMEOUT);
        let mut models = Vec::new();
        for served in self.catalog_models() {
            let Ok(model) = catalog::resolve(Some(&served.reference), self.surface) else {
                continue;
            };
            if model.entry.task == catalog::Task::Forecast {
                continue;
            }
            models.push((served, model));
        }
        compiling.expect((models.len() * WARM_SHAPES.len()) as u64);
        for (served, model) in models {
            let regression = model.entry.task == catalog::Task::Regression;
            for &(context_rows, queries) in WARM_SHAPES {
                let replicas = (0..engine.shards()).map(|shard| {
                    let payload = warm_payload(
                        &model.canonical_id(),
                        context_rows,
                        queries,
                        regression,
                        shard as u64,
                    );
                    let engine = engine.pinned(shard);
                    async move {
                        adhoc_predict_with(self, &engine, &payload, self.budget.charge()).await
                    }
                });
                let mut failed = None;
                for outcome in futures::future::join_all(replicas).await {
                    if let Err(error) = outcome {
                        failed = Some(error.message);
                    }
                }
                compiling.advance(1);
                if let Some(message) = failed {
                    tracing::warn!(
                        model = served.name.as_str(),
                        context_rows,
                        "not warming a shape bucket: {message}"
                    );
                    continue;
                }
                shapes.push(json!({
                    "model": served.resolved,
                    "context_rows": context_rows,
                    "inputs": WARM_INPUTS,
                    "queries": queries,
                }));
            }
        }
        let count = shapes.len();
        let _ = self.warm.set(shapes);
        count
    }

    /// Everything served here, catalog models and predictors alike.
    #[must_use]
    pub fn models(&self) -> &[ServedModel] {
        &self.models
    }

    /// The catalog models served here, in the order they were declared.
    #[must_use]
    pub fn catalog_models(&self) -> Vec<&ServedModel> {
        self.models
            .iter()
            .filter(|served| served.predictor.is_none())
            .collect()
    }

    /// Streams every route the front does not own to `upstream`.
    #[must_use]
    pub fn with_upstream(mut self, upstream: Upstream) -> Self {
        self.upstream = Some(upstream);
        self
    }

    /// Predicts predictors' revisions, loaded from `revisions` on demand.
    #[must_use]
    pub fn with_revisions(mut self, revisions: Revisions) -> Self {
        self.revisions = Some(revisions);
        self
    }

    /// The deployed revisions predicted from, if any are.
    #[must_use]
    pub fn revisions(&self) -> Option<&Revisions> {
        self.revisions.as_ref()
    }

    /// Answers only requests whose `Host` is one of `hosts` exactly, and
    /// none from a browser, as a predictor behind one endpoint does.
    #[must_use]
    pub fn with_hosts(mut self, hosts: Vec<String>) -> Self {
        self.hosts = hosts;
        self
    }

    /// Starts answering inference once the shards' workers are up, and is
    /// ready once each snapshot predictor's rows are bound into their table.
    ///
    /// # Errors
    ///
    /// Which snapshot predictor could not be bound, and why.
    pub async fn start(
        &self,
        shards: Vec<mpsc::Sender<Command>>,
        routes: Arc<Routes>,
    ) -> Result<(), String> {
        let _ = self.engine.set(Engine::new(
            shards,
            routes,
            self.budget.clone(),
            self.timeout,
        ));
        let engine = self.engine().map_err(|error| error.message)?;
        let mut bound = HashMap::new();
        let mut series = HashMap::new();
        for served in &self.models {
            if let Some(predictor) = &served.predictor
                && predictor.context == ContextMode::Snapshot
            {
                let refused =
                    |error: ApiError| format!("predictor {:?}: {}", served.name, error.message);
                if let Some(forecast) = &predictor.forecast {
                    let rows = snapshot_rows(predictor).map_err(refused)?;
                    let aligned = self
                        .bind_series(served, predictor, forecast, &rows, forecast.frequency)
                        .map_err(refused)?;
                    series.insert(served.name.clone(), aligned);
                } else {
                    let snapshot = engine
                        .bind_snapshot(&served.worker, predictor)
                        .await
                        .map_err(refused)?;
                    bound.insert(served.name.clone(), snapshot);
                }
            }
        }
        let _ = self.bound.set(bound);
        let _ = self.series.set(series);
        Ok(())
    }

    /// The durable tables `start` binds snapshot predictors' rows into, by
    /// name, with the fingerprints that place them, so a restart restores
    /// each where `start` looks for it.
    #[must_use]
    pub fn bound_tables(&self) -> HashMap<TableName, [u8; 32]> {
        self.models
            .iter()
            .filter_map(|served| served.predictor.as_ref())
            .filter(|predictor| {
                predictor.context == ContextMode::Snapshot && predictor.forecast.is_none()
            })
            .filter_map(|predictor| engine::bound_table(&predictor.lock_digest))
            .collect()
    }

    /// Whether the front takes inference: its shards are up, its snapshot
    /// predictors bound, and it is not draining.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.bound.get().is_some() && !self.draining.load(Ordering::SeqCst)
    }

    /// What clients must present, as the `serving` event reports it.
    #[must_use]
    pub fn auth(&self) -> &'static str {
        self.keys.auth()
    }

    /// The served models' names.
    #[must_use]
    /// What this server serves, predictors by name and catalog models by
    /// canonical id.
    pub fn names(&self) -> Vec<String> {
        self.models
            .iter()
            .map(|model| match model.predictor {
                Some(_) => model.name.clone(),
                None => model.reference.clone(),
            })
            .collect()
    }
}

/// The router answering every path, so unknown routes get the API's errors.
pub fn router(front: Arc<Front>) -> Router {
    Router::new().fallback(handle).with_state(front)
}

/// The reply while a served model is still loading or compiling.
fn not_ready() -> ApiError {
    ApiError::new(
        "ModelNotReady",
        "models are still loading; retry once /healthz answers 200",
        503,
    )
}

fn respond(status: u16, value: &Value) -> Response {
    let mut response = Response::new(Body::from(value.to_string()));
    *response.status_mut() =
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

fn envelope(scope: &Scope) -> Map<String, Value> {
    let mut envelope = Map::new();
    envelope.insert("api_version".into(), json!(VERSION));
    envelope.insert("project".into(), json!(scope.project));
    envelope.insert("request_id".into(), json!(scope.request_id));
    envelope.insert(
        "resource".into(),
        json!({"kind": scope.kind, "name": scope.name}),
    );
    envelope
}

fn failure(scope: Option<&Scope>, error: &ApiError) -> Response {
    let body = match scope {
        Some(scope) => {
            let mut body = envelope(scope);
            body.insert("error".into(), error.body()["error"].take());
            Value::Object(body)
        }
        None => error.body(),
    };
    respond(error.status, &body)
}

async fn handle(State(front): State<Arc<Front>>, request: Request) -> Response {
    if internal(request.uri().path()) {
        return failure(None, &ApiError::new("NotFound", "route not found", 404));
    }
    if let Some(upstream) = &front.upstream
        && !owns(request.method(), request.uri().path())
    {
        let peer = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|info| info.0);
        return match upstream.forward(request, peer).await {
            Ok(response) => response,
            Err(error) => failure(None, &error),
        };
    }
    let mut scope = None;
    match answer(&front, request, &mut scope).await {
        Ok(response) => response,
        Err(error) => failure(scope.as_ref(), &error),
    }
}

/// Answers one request; `scope` is set once its envelope is known, so a
/// later failure carries it.
///
/// Everything a request's headers and path decide, its key among them, is
/// checked before its body is read: a body counts against the request
/// budget only once its sender is known, and must arrive within the
/// front's request timeout, so neither an anonymous caller nor a stalled
/// one can hold the budget.
async fn answer(
    front: &Front,
    request: Request,
    scope: &mut Option<Scope>,
) -> Result<Response, ApiError> {
    let (parts, body) = request.into_parts();
    let headers = &parts.headers;
    check_headers(headers)?;
    let target = parts
        .uri
        .path_and_query()
        .map_or("/", |target| target.as_str());
    require(
        headers.len() <= 64 && target.len() <= 8192,
        "request headers too large",
    )?;
    if parts.method != Method::GET && parts.method != Method::POST {
        return Err(ApiError::new(
            "MethodNotAllowed",
            "only GET and POST are supported",
            405,
        ));
    }
    if headers.contains_key(header::TRANSFER_ENCODING) {
        return Err(ApiError::new(
            "ValidationError",
            "invalid HTTP message",
            400,
        ));
    }
    let length = match headers.get(header::CONTENT_LENGTH) {
        Some(length) => length
            .to_str()
            .ok()
            .filter(|length| length.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|length| length.parse::<usize>().ok())
            .ok_or_else(|| ApiError::new("ValidationError", "invalid HTTP message", 400))?,
        None => 0,
    };
    if length > MAX_BYTES {
        return Err(ApiError::new(
            "CapacityExceeded",
            format!("request body exceeds {MAX_BYTES} bytes"),
            413,
        ));
    }
    require(
        parts.method != Method::GET || length == 0,
        "GET bodies are forbidden",
    )?;
    if target == "/livez" {
        return Ok(respond(200, &json!({"status": "ok"})));
    }
    // Health is readiness: 200 only once every served model is loaded and
    // its recorded programs compiled, so a probe never admits traffic the
    // inference routes would refuse.
    if target == "/healthz" || target == "/readyz" {
        let ready = front.ready()
            && match &front.upstream {
                Some(upstream) => upstream.ready().await,
                None => true,
            };
        return Ok(if ready {
            respond(200, &json!({"status": "ok"}))
        } else {
            respond(503, &not_ready().body())
        });
    }
    if !front.hosts.is_empty() {
        check_endpoint(headers, &front.hosts)?;
    } else if front.loopback {
        check_origin(headers)?;
    }
    require(
        parts.uri.scheme().is_none()
            && parts.uri.authority().is_none()
            && !target.contains('#')
            && !parts.uri.path().contains('%')
            && !parts.uri.path().contains('\\'),
        "invalid request target",
    )?;
    require(
        parts.uri.query().is_none(),
        "query parameters are not supported",
    )?;
    let (project, action, resource) = resolve(&parts.method, parts.uri.path())?;
    let header = |name: &str| headers.get(name).and_then(|value| value.to_str().ok());
    let request_id = match header("x-gf-request-id") {
        Some(id) => id.to_string(),
        None => format!(
            "r-{}",
            random_hex(16).map_err(|_| ApiError::new(
                "InternalError",
                "request failed safely",
                500
            ))?
        ),
    };
    require(
        request_id.len() <= 128 && request_id.bytes().all(|byte| (33..=126).contains(&byte)),
        "invalid request id",
    )?;
    if let Some(given) = header("x-gf-project") {
        require(given == project, "project header/path mismatch")?;
    }
    if parts.method == Method::GET {
        *scope = Some(Scope {
            project: project.clone(),
            request_id: request_id.clone(),
            kind: action.kind().into(),
            name: resource.clone(),
        });
    }
    let principal = front.keys.verify(header("authorization")).await?;
    principal.authorize(&project, action.permission())?;
    // What the body holds counts against the server's request budget until answered.
    let mut charge = front.budget.charge();
    charge.grow(length as u64).map_err(|_| {
        ApiError::new(
            "CapacityExceeded",
            "the server is at capacity; retry shortly",
            503,
        )
    })?;
    let body = tokio::time::timeout(front.timeout, axum::body::to_bytes(body, MAX_BYTES))
        .await
        .map_err(|_| {
            ApiError::new(
                "RequestTimeout",
                "the request body did not arrive in time",
                408,
            )
        })?
        .map_err(|_| ApiError::new("ValidationError", "invalid HTTP message", 400))?;
    require(body.len() == length, "truncated request body")?;
    let payload = if parts.method == Method::GET {
        Value::Object(Map::new())
    } else {
        let content_type = header("content-type").unwrap_or_default();
        require(
            content_type
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .eq_ignore_ascii_case("application/json"),
            "application/json required",
        )?;
        let (parsed, payload) = parse_envelope(&body)?;
        *scope = Some(parsed.clone());
        require(parsed.project == project, "project path/envelope mismatch")?;
        if header("x-gf-request-id").is_some() {
            require(parsed.request_id == request_id, "request id mismatch")?;
        }
        require(
            parsed.kind == action.kind() && names_match(&parsed.name, &resource),
            "resource route/envelope mismatch",
        )?;
        payload
    };
    if !front.ready() {
        return Err(not_ready());
    }
    let result = dispatch(front, &project, action, &resource, payload, charge).await?;
    let mut body = envelope(scope.as_ref().expect("set above"));
    body.insert("result".into(), result);
    Ok(respond(200, &Value::Object(body)))
}

/// Refuses repeated headers and header values that are too long or split lines.
fn check_headers(headers: &HeaderMap) -> Result<(), ApiError> {
    for name in headers.keys() {
        require(
            headers.get_all(name).iter().count() == 1,
            "duplicate HTTP header",
        )?;
    }
    for value in headers.values() {
        require(value.len() <= 8192, "invalid HTTP header")?;
    }
    Ok(())
}

/// On loopback, refuses DNS-rebound hosts and cross-origin browser calls.
fn check_origin(headers: &HeaderMap) -> Result<(), ApiError> {
    let host = headers
        .get(header::HOST)
        .and_then(|host| host.to_str().ok())
        .unwrap_or_default();
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split_once(']').map(|(name, _)| name),
        None => Some(host.rsplit_once(':').map_or(host, |(name, port)| {
            if port.bytes().all(|byte| byte.is_ascii_digit()) {
                name
            } else {
                host
            }
        })),
    };
    let local = name.is_some_and(|name| {
        ["127.0.0.1", "localhost", "::1"].contains(&name)
            || name.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
    });
    require(
        local && !host.contains('@') && !host.contains('/'),
        "untrusted Host",
    )?;
    if let Some(origin) = headers.get(header::ORIGIN) {
        require(
            origin.to_str().ok() == Some(&format!("http://{host}")),
            "untrusted Origin",
        )?;
    }
    Ok(())
}

/// Behind one endpoint, refuses other hosts and every browser call.
fn check_endpoint(headers: &HeaderMap, hosts: &[String]) -> Result<(), ApiError> {
    let host = headers
        .get(header::HOST)
        .and_then(|host| host.to_str().ok());
    if !host.is_some_and(|host| hosts.iter().any(|allowed| allowed == host)) {
        return Err(ApiError::new(
            "Forbidden",
            "unrecognized endpoint Host",
            403,
        ));
    }
    if headers.contains_key(header::ORIGIN) {
        return Err(ApiError::new(
            "Forbidden",
            "browser origins are not supported",
            403,
        ));
    }
    Ok(())
}

/// An envelope's resource name: a plain name, or for a model its canonical
/// id `general-forecast/<name>`, kept as sent.
fn resource_name(kind: &str, value: &str) -> Result<String, ApiError> {
    match value.strip_prefix(&format!("{}/", catalog::ORG)) {
        Some(bare) if kind == "model" => {
            name(bare)?;
            Ok(value.to_string())
        }
        _ => name(value),
    }
}

/// Whether an envelope's resource name names the route's resource; a model's
/// canonical id names its bare route segment.
fn names_match(envelope: &str, resource: &str) -> bool {
    envelope == resource
        || envelope
            .strip_prefix(&format!("{}/", catalog::ORG))
            .is_some_and(|bare| bare == resource)
}

/// A resource name: a lowercase letter, then up to 62 letters, digits or `-`.
fn name(value: &str) -> Result<String, ApiError> {
    let mut bytes = value.bytes();
    let valid = value.len() <= 63
        && bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
    require(valid, "invalid name")?;
    Ok(value.to_string())
}

/// Whether `method` on `path` is the front's to answer: health, or one of
/// the routes `resolve` serves, however its segments are cased or split, so
/// no inference route, nor a misspelling of one, is ever proxied. Every
/// other route, such as a hosted predictor's revisions or environments,
/// streams upstream.
#[must_use]
pub fn owns(method: &Method, path: &str) -> bool {
    if matches!(path, "/livez" | "/healthz" | "/readyz") {
        return true;
    }
    let parts: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
    let [api, projects, _, tail @ ..] = parts.as_slice() else {
        return false;
    };
    if !api.eq_ignore_ascii_case("v1") || !projects.eq_ignore_ascii_case("projects") {
        return false;
    }
    // An escaped segment could spell any route once decoded; it is refused
    // here rather than proxied.
    if tail.iter().any(|part| part.contains('%')) {
        return true;
    }
    let lower: Vec<String> = tail.iter().map(|part| part.to_ascii_lowercase()).collect();
    let lower: Vec<&str> = lower.iter().map(String::as_str).collect();
    route(method, &lower).is_some()
}

/// The action a route's segments after its project name, if it is one the
/// front serves.
fn route(method: &Method, tail: &[&str]) -> Option<Action> {
    match (method, tail) {
        (&Method::GET, ["models"]) => Some(Action::List),
        (&Method::GET, ["models", _]) => Some(Action::Get),
        (&Method::GET, ["predictors"]) => Some(Action::Predictors),
        (&Method::GET, ["predictors", _]) => Some(Action::Predictor),
        (&Method::POST, ["predictors", _, "predict"]) => Some(Action::PredictorPredict),
        (&Method::POST, ["predict"]) => Some(Action::AdHoc),
        _ => None,
    }
}

/// Whether `path` is one of the control plane's service-to-service routes,
/// under `/internal`, which only this server's own calls may reach; a
/// client's request for one is refused rather than proxied.
#[must_use]
pub fn internal(path: &str) -> bool {
    path.split('/')
        .find(|part| !part.is_empty())
        .is_some_and(|first| percent_decoded(first).eq_ignore_ascii_case(INTERNAL))
}

/// `segment` with its `%XX` escapes decoded; an escape that is not one is
/// kept as it is.
fn percent_decoded(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let escaped = (bytes[index] == b'%')
            .then(|| bytes.get(index + 1..index + 3))
            .flatten()
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match escaped {
            Some(byte) => {
                decoded.push(byte);
                index += 3;
            }
            None => {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

/// Maps the Models API paths to a project, action and resource; every other
/// path is NotFound.
fn resolve(method: &Method, path: &str) -> Result<(String, Action, String), ApiError> {
    let not_found = || ApiError::new("NotFound", "route not found", 404);
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 5 || parts[..3] != ["", "v1", "projects"] {
        return Err(not_found());
    }
    if parts[3..].iter().any(|part| part.is_empty()) {
        return Err(not_found());
    }
    let project = name(parts[3])?;
    let tail = &parts[4..];
    let action = route(method, tail).ok_or_else(not_found)?;
    let resource = match action {
        Action::List => "models".to_string(),
        Action::Predictors => "predictors".to_string(),
        Action::AdHoc => "ad-hoc".to_string(),
        Action::Get | Action::Predictor | Action::PredictorPredict => name(tail[1])?,
    };
    Ok((project, action, resource))
}

/// JSON parsed without accepting ambiguous objects, which repeat a key.
struct Unique(Value);

impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueVisitor).map(Unique)
    }
}

struct UniqueVisitor;

impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(Unique(item)) = seq.next_element()? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let Unique(value) = map.next_value()?;
            if object.insert(key, value).is_some() {
                return Err(de::Error::custom(DUPLICATE));
            }
        }
        Ok(Value::Object(object))
    }
}

const DUPLICATE: &str = "duplicate JSON field";

/// Checks JSON is portable and bounded: nested at most 16 deep, at most
/// 65,536 values, integers exact in a double.
fn bounded(value: &Value) -> Result<(), ApiError> {
    fn visit(value: &Value, depth: usize, count: &mut usize) -> Result<(), ApiError> {
        *count += 1;
        require(
            depth <= MAX_DEPTH && *count <= MAX_NODES,
            "JSON structure too large",
        )?;
        match value {
            Value::Object(object) => object
                .values()
                .try_for_each(|child| visit(child, depth + 1, count)),
            Value::Array(items) => items
                .iter()
                .try_for_each(|child| visit(child, depth + 1, count)),
            Value::Number(number) => require(
                number.as_f64().is_some()
                    && (number.is_f64()
                        || number
                            .as_i64()
                            .is_some_and(|integer| integer.abs() <= MAX_INTEGER)),
                "integer exceeds portable JSON range",
            ),
            _ => Ok(()),
        }
    }
    let mut count = 0;
    visit(value, 0, &mut count)
}

/// Checks an object has exactly the `required` keys and perhaps the `optional` ones.
fn fields<'a>(
    value: &'a Value,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a Map<String, Value>, ApiError> {
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::validation("expected an object"))?;
    let mut sorted = required.to_vec();
    sorted.sort_unstable();
    let listed: Vec<String> = sorted.iter().map(|key| format!("'{key}'")).collect();
    require(
        required.iter().all(|key| object.contains_key(*key)),
        &format!("required fields: [{}]", listed.join(", ")),
    )?;
    require(
        object
            .keys()
            .all(|key| required.contains(&key.as_str()) || optional.contains(&key.as_str())),
        "unknown fields",
    )?;
    Ok(object)
}

/// A request envelope: its scope and resource, and its payload.
fn parse_envelope(body: &[u8]) -> Result<(Scope, Value), ApiError> {
    require(
        !body.is_empty() && body.len() <= MAX_BYTES,
        "bounded JSON body required",
    )?;
    let value = match serde_json::from_slice::<Unique>(body) {
        Ok(Unique(value)) => value,
        Err(error) if error.to_string().starts_with(DUPLICATE) => {
            return Err(ApiError::validation(DUPLICATE));
        }
        Err(_) => return Err(ApiError::validation("invalid JSON body")),
    };
    require(value.is_object(), "expected JSON object")?;
    bounded(&value)?;
    let object = fields(
        &value,
        &[
            "api_version",
            "project",
            "request_id",
            "resource",
            "payload",
        ],
        &[],
    )?;
    require(object["api_version"] == VERSION, "unsupported api_version")?;
    let resource = fields(&object["resource"], &["kind", "name"], &[])?;
    let project = name(object["project"].as_str().unwrap_or_default())?;
    let request_id = object["request_id"]
        .as_str()
        .filter(|id| !id.is_empty() && id.chars().count() <= 256)
        .ok_or_else(|| ApiError::validation("invalid request_id"))?;
    let kind = resource["kind"]
        .as_str()
        .filter(|kind| RESOURCE_KINDS.contains(kind))
        .ok_or_else(|| ApiError::validation("invalid resource kind"))?;
    require(object["payload"].is_object(), "expected JSON object")?;
    let resource_name = resource_name(kind, resource["name"].as_str().unwrap_or_default())?;
    let scope = Scope {
        project,
        request_id: request_id.to_string(),
        kind: kind.to_string(),
        name: resource_name,
    };
    Ok((scope, object["payload"].clone()))
}

impl Front {
    /// The served model a request names, by its resource name or any
    /// reference the catalog resolves to it.
    fn served(&self, model: &str) -> Result<&ServedModel, ApiError> {
        let canonical = catalog::resolve(Some(model), self.surface)
            .ok()
            .map(|resolved| resolved.entry.name.to_string());
        self.models
            .iter()
            .filter(|served| served.predictor.is_none())
            .find(|served| {
                served.name == model || canonical.as_deref() == Some(served.name.as_str())
            })
            .ok_or_else(ApiError::not_served)
    }

    /// The predictor `name` serves to `project`, with its lock.
    fn predictor<'a>(
        &'a self,
        project: &'a str,
        name: &str,
    ) -> Result<(&'a ServedModel, &'a ServedPredictor), ApiError> {
        self.predictors(project)
            .find(|(served, _)| served.name == name)
            .ok_or_else(|| {
                ApiError::new(
                    "NotFound",
                    format!("predictor {name} is not served here"),
                    404,
                )
                .param("name")
            })
    }

    /// The predictors `project` reaches here, with their locks.
    fn predictors<'a>(
        &'a self,
        project: &'a str,
    ) -> impl Iterator<Item = (&'a ServedModel, &'a ServedPredictor)> + 'a {
        self.models.iter().filter_map(move |served| {
            let reaches = served.project.as_deref().is_none_or(|own| own == project);
            served
                .predictor
                .as_ref()
                .filter(|_| reaches)
                .map(|predictor| (served, predictor))
        })
    }

    /// A catalog model's `GET models` entry: the catalog's description, the
    /// limits a request must keep to, and the shapes compiled at startup.
    fn model_description(&self, served: &ServedModel) -> Value {
        let policy = &catalog::catalog().policy;
        let mut description = served.description.clone();
        description["model"] = json!(served.resolved);
        description["limits"] = json!({
            "context_rows": policy.context_rows,
            "queries": policy.query_rows,
            "inputs": policy.features,
        });
        description["warm_shapes"] = json!(self.warm.get().cloned().unwrap_or_default());
        description
    }

    /// A predictor's `GET predictors/{name}` entry: its resolved manifest,
    /// lock digests, context kind and size, and what it answers.
    fn predictor_description(&self, served: &ServedModel, predictor: &ServedPredictor) -> Value {
        let rows = match predictor.context {
            ContextMode::Snapshot => self
                .bound
                .get()
                .and_then(|bound| bound.get(&served.name))
                .map(|bound| bound.rows)
                .or_else(|| {
                    self.series
                        .get()
                        .and_then(|series| series.get(&served.name))
                        .map(|series| series.rows)
                }),
            ContextMode::Request => None,
        };
        json!({
            "name": served.name,
            "model": served.resolved,
            "task": predictor.task,
            "target": predictor.target,
            "inputs": predictor.inputs,
            "id": predictor.id,
            "context": {
                "kind": predictor.context.kind(),
                "rows": rows,
                "digest": predictor.context_digest,
            },
            "lock_digest": predictor.lock_digest,
            "outputs": served.outputs,
            "quantiles": predictor.quantiles,
        })
    }

    /// The catalog model served as exactly this reference, variant and all.
    fn serving(&self, model: &catalog::Model) -> Result<&ServedModel, ApiError> {
        let reference = model.to_string();
        self.models
            .iter()
            .find(|served| {
                served.predictor.is_none()
                    && (served.reference == reference
                        || (served.name == model.entry.name
                            && model.digest.is_none()
                            && !model.variant())
                        || catalog::resolve(Some(&served.reference), self.surface)
                            .is_ok_and(|resolved| resolved.to_string() == reference))
            })
            .ok_or_else(ApiError::not_served)
    }

    /// Admits a payload's model only if it is exactly the reference served,
    /// variant and all.
    fn engine(&self) -> Result<&Engine, ApiError> {
        self.engine.get().ok_or_else(not_ready)
    }

    /// Aligns a forecast predictor's context and parses its group's forecast
    /// request, as `ForecastContext.build` does in process.
    fn bind_series(
        &self,
        served: &ServedModel,
        predictor: &ServedPredictor,
        forecast: &ForecastPredictor,
        rows: &RecordBatch,
        frequency: Option<f64>,
    ) -> Result<SeriesBound, ApiError> {
        let failed = || ApiError::new("InternalError", "request failed safely", 500);
        let target = predictor.target.as_deref().ok_or_else(|| {
            ApiError::new(
                "InvalidPredictor",
                "a forecast predictor names its target",
                422,
            )
        })?;
        let context = forecast.context(
            &served.reference,
            target,
            &predictor.quantiles,
            rows,
            frequency,
        )?;
        let bytes = serde_json::to_vec(&context.request).map_err(|_| failed())?;
        let options = Options {
            surface: self.surface,
            limits: Limits::Predictor,
            levels: None,
        };
        let Inference::Forecast(parsed) = request::parse(&bytes, &options)? else {
            return Err(failed());
        };
        Ok(SeriesBound {
            context,
            forecast: parsed,
            rows: rows.num_rows(),
        })
    }
}

/// A class's key in a probabilities mapping: `true`/`false`, or the label as text.
fn class_key(label: &Label) -> String {
    match label {
        Label::Bool(value) => value.to_string(),
        Label::Int(value) => value.to_string(),
        Label::Text(value) => value.clone(),
    }
}

fn label_json(label: &Label) -> Value {
    match label {
        Label::Bool(value) => json!(value),
        Label::Int(value) => json!(value),
        Label::Text(value) => json!(value),
    }
}

/// A quantile level as Python prints a float, which is how outputs key it.
fn level_key(level: f64) -> String {
    let text = format!("{level}");
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

/// Each query's class probabilities, keyed by class.
fn probabilities(classes: &[Label], inferred: &Inferred) -> Vec<Map<String, Value>> {
    let predictions = &inferred.predictions;
    (0..predictions.rows)
        .map(|row| {
            classes
                .iter()
                .enumerate()
                .map(|(class, label)| {
                    let value = predictions.values[row * predictions.columns + class];
                    (class_key(label), json!(f64::from(value)))
                })
                .collect()
        })
        .collect()
}

/// Each query's values at `levels`, keyed by level, from the table's own levels.
fn quantiles(inferred: &Inferred, levels: &[f64]) -> Result<Vec<Map<String, Value>>, ApiError> {
    let known = inferred.levels.as_deref().ok_or_else(|| {
        ApiError::new(
            "ModelOutputError",
            "native model returned invalid outputs",
            503,
        )
    })?;
    let columns =
        quantile_values(&inferred.predictions.values, known, levels).ok_or_else(|| {
            ApiError::new(
                "ModelOutputError",
                "native model returned invalid outputs",
                503,
            )
        })?;
    Ok((0..inferred.predictions.rows)
        .map(|row| {
            levels
                .iter()
                .zip(&columns)
                .map(|(&level, column)| (level_key(level), json!(f64::from(column[row]))))
                .collect()
        })
        .collect())
}

async fn dispatch(
    front: &Front,
    project: &str,
    action: Action,
    resource: &str,
    payload: Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    match action {
        Action::List => Ok(json!({
            "models": front
                .models
                .iter()
                .filter(|model| model.predictor.is_none())
                .map(|model| front.model_description(model))
                .collect::<Vec<_>>()
        })),
        Action::Get => Ok(front.model_description(front.served(resource)?)),
        Action::Predictors => Ok(json!({
            "predictors": front
                .predictors(project)
                .map(|(served, predictor)| front.predictor_description(served, predictor))
                .collect::<Vec<_>>()
        })),
        Action::Predictor => {
            let (served, predictor) = front.predictor(project, resource)?;
            Ok(front.predictor_description(served, predictor))
        }
        Action::PredictorPredict => match front.predictor(project, resource) {
            Ok((served, predictor)) => {
                predictor_predict(front, served, predictor, &payload, charge).await
            }
            Err(error) => match front.revisions.as_ref() {
                Some(revisions) => {
                    revision_predict(front, revisions, project, resource, &payload, charge).await
                }
                None => Err(error),
            },
        },
        Action::AdHoc => adhoc_predict(front, &payload, charge).await,
    }
}

fn forecasting_unserved() -> ApiError {
    ApiError::new(
        "ValidationError",
        "this server answers tabular models; forecasters are served by the Python engine",
        422,
    )
}

/// The `/predictions` body for a catalog model: flat rows, bound as an
/// inference request whose context is the rows with a target, then
/// answered as the shared `PredictionResult`.
/// Refuses the fields of `payload` named in `fields`, each with `reason`
/// after its name, so a caller learns which field to drop.
fn refuse_fields(payload: &Value, fields: &[&str], reason: &str) -> Result<(), ApiError> {
    if let Some(object) = payload.as_object() {
        for field in fields {
            if object.contains_key(*field) {
                return Err(ApiError::validation(format!("{field} {reason}")).param(*field));
            }
        }
    }
    Ok(())
}

fn context_declared() -> ApiError {
    ApiError::validation("context is declared by the predictor")
        .param("context")
        .hint("drop context: this predictor's context is pinned in its lock")
}

/// A declared predictor's body: `queries`, and `context` only when the lock
/// says each request brings it; what the manifest declares is refused.
async fn predictor_predict(
    front: &Front,
    served: &ServedModel,
    predictor: &ServedPredictor,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    refuse_fields(
        payload,
        &[
            "target",
            "inputs",
            "model",
            "task",
            "features",
            "options",
            "quantiles",
            "outputs",
            "id",
        ],
        "is declared by the predictor",
    )?;
    let body = fields(payload, &["queries"], &["context", "as_of"])?;
    let context = match predictor.context {
        ContextMode::Request => body.get("context").cloned().ok_or_else(|| {
            ApiError::validation(
                "context is required: this predictor takes its context with each request",
            )
            .param("context")
        })?,
        ContextMode::Snapshot => {
            if body.contains_key("context") {
                return Err(context_declared());
            }
            json!([])
        }
    };
    let inputs: Vec<&str> = match &predictor.forecast {
        Some(forecast) => {
            let queried = forecast.query_columns();
            predictor
                .schema
                .columns
                .iter()
                .map(|column| column.name.as_str())
                .filter(|name| queried.contains(name))
                .collect()
        }
        None => predictor.inputs.iter().map(String::as_str).collect(),
    };
    let call = json!({
        "target": predictor.target,
        "inputs": inputs,
        "context": context,
        "queries": body["queries"],
    });
    served_predict(front, served, predictor, &call, charge).await
}

/// The ad hoc route's body: a manifest fragment (`model`, `target`, and
/// optionally `inputs`, `id`, `quantiles`, `outputs`, `selection`) with the
/// `context` and `queries` rows, for a catalog model served here.
async fn adhoc_predict(front: &Front, payload: &Value, charge: Charge) -> Result<Value, ApiError> {
    adhoc_predict_with(front, front.engine()?, payload, charge).await
}

async fn adhoc_predict_with(
    front: &Front,
    engine: &Engine,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    refuse_fields(
        payload,
        &["task"],
        "is not a field: the model determines the task",
    )?;
    refuse_fields(
        payload,
        &["options"],
        "is not a field: id, quantiles and outputs are top-level fields",
    )?;
    refuse_fields(payload, &["features"], "is spelled inputs")?;
    let body = fields(
        payload,
        &["model", "target", "context", "queries"],
        &[
            "inputs",
            "id",
            "time",
            "series",
            "frequency",
            "horizon",
            "quantiles",
            "outputs",
            "selection",
        ],
    )?;
    let model = catalog::canonical(body["model"].as_str(), front.surface)?;
    if model.entry.task == catalog::Task::Forecast {
        return Err(forecasting_unserved());
    }
    let served = front.serving(&model)?;
    let task = match model.entry.task {
        catalog::Task::Classification => "classification",
        _ => "regression",
    };
    for field in ["time", "series", "frequency", "horizon"] {
        if body.contains_key(field) {
            return Err(ApiError::validation(format!(
                "{field} is a forecast field; {} is a {task} model",
                model.canonical_id()
            ))
            .param(field));
        }
    }
    if let Some(selection) = body.get("selection")
        && !selection.is_object()
    {
        return Err(ApiError::validation("selection must be an object").param("selection"));
    }
    let target = &body["target"];
    let target_name = target
        .as_str()
        .filter(|target| !target.is_empty())
        .ok_or_else(|| ApiError::validation("target must name a column").param("target"))?;
    let (Some(rows), Some(queries)) = (body["context"].as_array(), body["queries"].as_array())
    else {
        return Err(ApiError::validation(
            "context and queries must be lists of rows",
        ));
    };
    require(
        rows.iter().chain(queries).all(Value::is_object),
        "rows must be objects",
    )?;
    let id = match body.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) => Some(id.as_str()),
        Some(_) => return Err(ApiError::validation("id must name a column").param("id")),
    };
    let inputs: Vec<Value> = match body.get("inputs") {
        None | Some(Value::Null) => rows
            .first()
            .and_then(Value::as_object)
            .map(|row| {
                row.keys()
                    .filter(|column| column.as_str() != target_name && Some(column.as_str()) != id)
                    .map(|column| json!(column))
                    .collect()
            })
            .unwrap_or_default(),
        Some(Value::Array(inputs)) if inputs.iter().all(Value::is_string) => inputs.clone(),
        Some(_) => {
            return Err(
                ApiError::validation("inputs must be a list of column names").param("inputs"),
            );
        }
    };
    if inputs.is_empty() {
        return Err(ApiError::validation(
            "no input columns: the context rows need a column besides the target",
        )
        .param("inputs"));
    }
    let features = &inputs;
    let outputs: Vec<String> = match body.get("outputs") {
        Some(Value::Array(outputs)) if !outputs.is_empty() => outputs
            .iter()
            .map(|output| output.as_str().map(str::to_string))
            .collect::<Option<_>>()
            .ok_or_else(|| ApiError::validation("model does not support requested outputs"))?,
        Some(Value::Array(_)) | None => {
            if task == "classification" {
                vec!["label".into(), "probabilities".into()]
            } else {
                vec!["value".into()]
            }
        }
        Some(_) => {
            return Err(ApiError::validation(
                "model does not support requested outputs",
            ));
        }
    };
    let unique: std::collections::HashSet<&String> = outputs.iter().collect();
    require(
        unique.len() == outputs.len()
            && outputs.iter().all(|output| served.outputs.contains(output)),
        "model does not support requested outputs",
    )?;
    let levels = body.get("quantiles");
    require(
        levels.is_none() || outputs.iter().any(|output| output == "quantiles"),
        "quantiles require the quantiles output",
    )?;
    let select = |row: &Value| -> Value {
        Value::Object(
            features
                .iter()
                .filter_map(Value::as_str)
                .map(|feature| {
                    (
                        feature.to_string(),
                        row.get(feature).cloned().unwrap_or(Value::Null),
                    )
                })
                .collect(),
        )
    };
    let context: Vec<Value> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.get(target_name).is_some_and(|label| !label.is_null()))
        .map(|(index, row)| json!({"key": index.to_string(), "features": select(row), "label": row[target_name]}))
        .collect();
    let keyed: Vec<Value> = queries
        .iter()
        .enumerate()
        .map(|(index, row)| json!({"key": index.to_string(), "features": select(row)}))
        .collect();
    let mut request_options = Map::new();
    if task == "regression" {
        request_options.insert(
            "quantiles".into(),
            levels.cloned().unwrap_or_else(|| json!(DEFAULT_QUANTILES)),
        );
    }
    let request = json!({
        "model": body["model"],
        "task": task,
        "target": target,
        "features": features,
        "context": context,
        "queries": keyed,
        "options": request_options,
    });
    let bytes = serde_json::to_vec(&request)
        .map_err(|_| ApiError::new("InternalError", "request failed safely", 500))?;
    let Inference::Tabular(parsed) = request::parse(&bytes, &Options::request(front.surface))?
    else {
        return Err(forecasting_unserved());
    };
    let inferred = engine.tabular(&served.worker, &parsed, charge).await?;
    let predictions = prediction_rows(
        &parsed.classes,
        &parsed.levels,
        &inferred,
        &outputs,
        queries,
        id,
    )?;
    Ok(prediction_result(
        &served.resolved,
        task,
        "request",
        None,
        parsed.usable_rows(),
        predictions,
    ))
}

/// The `/predictions` rows: the id column, if every query has it, then
/// each output asked for, from the classes or quantile levels predicted.
fn prediction_rows(
    classes: &[Label],
    levels: &[f64],
    inferred: &Inferred,
    outputs: &[String],
    queries: &[Value],
    id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let mut values: HashMap<&str, Vec<Value>> = HashMap::new();
    if classes.is_empty() {
        let median = quantiles(inferred, &[0.5])?;
        values.insert(
            "value",
            median.into_iter().map(|row| row["0.5"].clone()).collect(),
        );
        values.insert(
            "quantiles",
            quantiles(inferred, levels)?
                .into_iter()
                .map(Value::Object)
                .collect(),
        );
    } else {
        let scored = probabilities(classes, inferred);
        let labels = scored
            .iter()
            .map(|probabilities| {
                let mut best = 0;
                for (class, label) in classes.iter().enumerate() {
                    let at = |label: &Label| {
                        probabilities[&class_key(label)]
                            .as_f64()
                            .unwrap_or(f64::NAN)
                    };
                    if at(label) > at(&classes[best]) {
                        best = class;
                    }
                }
                label_json(&classes[best])
            })
            .collect();
        values.insert("label", labels);
        values.insert(
            "probabilities",
            scored.into_iter().map(Value::Object).collect(),
        );
    }
    Ok(projected(
        values,
        outputs,
        queries,
        id,
        inferred.predictions.rows,
    ))
}

/// Prediction rows: each query's declared id, when every query has one,
/// then the lock's outputs, in their order.
fn projected(
    mut values: HashMap<&str, Vec<Value>>,
    outputs: &[String],
    queries: &[Value],
    id: Option<&str>,
    rows: usize,
) -> Vec<Value> {
    let mut columns: Vec<(String, Vec<Value>)> = Vec::new();
    if let Some(id) = id
        && queries.iter().all(|row| row.get(id).is_some())
    {
        columns.push((
            id.to_string(),
            queries.iter().map(|row| row[id].clone()).collect(),
        ));
    }
    for output in outputs {
        columns.push((
            output.clone(),
            values.remove(output.as_str()).unwrap_or_default(),
        ));
    }
    (0..rows)
        .map(|row| {
            Value::Object(
                columns
                    .iter()
                    .map(|(name, column)| (name.clone(), column[row].clone()))
                    .collect(),
            )
        })
        .collect()
}

/// The shared `PredictionResult` wire value, with the context rows used.
fn prediction_result(
    model: &str,
    task: &str,
    source: &str,
    version: Option<&str>,
    rows: usize,
    predictions: Vec<Value>,
) -> Value {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    json!({
        "model": model,
        "task": task,
        "context": {"kind": source, "rows": rows, "digest": version, "as_of": now},
        "predictions": predictions,
    })
}

/// A locked predictor's `/predictions`: the body checked against the lock,
/// then the queries encoded by the binder and predicted from the
/// snapshot's durable table, or from the ephemeral table of the request's
/// own bound context.
async fn served_predict(
    front: &Front,
    served: &ServedModel,
    predictor: &ServedPredictor,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let body = fields(payload, &["target", "inputs", "context", "queries"], &[])?;
    let features: Vec<String> = body["inputs"]
        .as_array()
        .ok_or_else(|| ApiError::validation("inputs must be a list"))?
        .iter()
        .map(|input| input.as_str().map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| ApiError::validation("inputs differ from lock"))?;
    let queries = body["queries"]
        .as_array()
        .filter(|queries| (1..=MAX_QUERY_ROWS).contains(&queries.len()))
        .ok_or_else(|| ApiError::validation("queries must contain 1..10000 rows"))?;
    let context = body["context"]
        .as_array()
        .ok_or_else(|| ApiError::validation("rows must be lists"))?;
    require(
        queries.iter().chain(context).all(Value::is_object),
        "rows must be objects",
    )?;
    if let Some(forecast) = &predictor.forecast {
        return series_predict(
            front, served, predictor, forecast, body, &features, queries, context, charge,
        )
        .await;
    }
    let schema_free = predictor.schema_free();
    let target = if schema_free {
        let target = body["target"]
            .as_str()
            .ok_or_else(|| ApiError::validation("request target is required"))?;
        require(
            predictor
                .target
                .as_deref()
                .is_none_or(|locked| locked == target),
            "target differs from lock",
        )?;
        target.to_string()
    } else {
        require(
            body["target"].as_str().is_some()
                && body["target"].as_str() == predictor.target.as_deref(),
            "target differs from lock",
        )?;
        require(features == predictor.inputs, "inputs differ from lock")?;
        predictor.target.clone().unwrap_or_default()
    };
    let (inferred, classes, rows, source, version) = match predictor.context {
        ContextMode::Snapshot => {
            if !context.is_empty() {
                return Err(context_declared());
            }
            let bound = front
                .bound
                .get()
                .and_then(|bound| bound.get(&served.name))
                .ok_or_else(not_ready)?;
            let encoded = bound
                .encoder
                .encode(&predictors::records(&predictor.schema.columns, queries)?)?;
            let inferred = front.engine()?.predicted(bound, encoded).await?;
            (
                inferred,
                bound.classes.clone(),
                bound.rows,
                "snapshot",
                predictor.context_digest.as_deref(),
            )
        }
        ContextMode::Request => {
            require(
                !context.is_empty() && context.len() <= MAX_CONTEXT_ROWS,
                "context must contain 1..100000 rows",
            )?;
            let schema = if schema_free {
                predictors::inferred(&features, &target, context)
            } else {
                predictor.schema.clone()
            };
            let bound = predictor::bind(
                &schema,
                predictor.task()?,
                &target,
                &predictors::records(&schema.columns, context)?,
            )?;
            let encoded = bound
                .encoder
                .encode(&predictors::records(&schema.columns, queries)?)?;
            let fingerprint = bound_fingerprint(served, &bound);
            let context = engine::rows_context(
                &served.worker,
                &fingerprint,
                bound.encoder.width(),
                bound.classes.len(),
                bound.features,
                bound.targets,
                bound.rows,
            )?;
            let inferred = front
                .engine()?
                .ask(context, Asked::Rows(encoded), &fingerprint)
                .await?;
            (inferred, bound.classes, bound.rows, "request", None)
        }
    };
    drop(charge);
    let predictions = prediction_rows(
        &classes,
        &predictor.quantiles,
        &inferred,
        &served.outputs,
        queries,
        predictor.id.as_deref(),
    )?;
    Ok(prediction_result(
        &served.resolved,
        &predictor.task,
        source,
        version,
        rows,
        predictions,
    ))
}

/// A snapshot predictor's context rows, from the file `gf serve` wrote.
pub(crate) fn snapshot_rows(predictor: &ServedPredictor) -> Result<RecordBatch, ApiError> {
    let path = predictor.snapshot.as_deref().ok_or_else(|| {
        ApiError::new(
            "ContextRequired",
            "snapshot predictors are served from their predictor directory",
            422,
        )
    })?;
    predictors::read_snapshot(path).map_err(|error| ApiError::new("MissingFile", error, 503))
}

/// A forecast predictor's `/predictions`: the queries placed on the bound
/// group's series and horizon, the known inputs' future filled in, and the
/// group forecast on the shard path catalog forecasts take, from the table
/// its context and future values name.
#[allow(clippy::too_many_arguments)]
async fn series_predict(
    front: &Front,
    served: &ServedModel,
    predictor: &ServedPredictor,
    forecast: &ForecastPredictor,
    body: &Map<String, Value>,
    features: &[String],
    queries: &[Value],
    context: &[Value],
    charge: Charge,
) -> Result<Value, ApiError> {
    let failed = || ApiError::new("InternalError", "request failed safely", 500);
    let target = body["target"]
        .as_str()
        .ok_or_else(|| ApiError::validation("target differs from lock"))?;
    require(
        predictor
            .target
            .as_deref()
            .is_none_or(|locked| locked == target),
        "target differs from lock",
    )?;
    let aligned;
    let (bound, schema, source, version) = match predictor.context {
        ContextMode::Snapshot => {
            if !context.is_empty() {
                return Err(context_declared());
            }
            let bound = front
                .series
                .get()
                .and_then(|series| series.get(&served.name))
                .ok_or_else(not_ready)?;
            (
                bound,
                predictor.schema.clone(),
                "snapshot",
                predictor.context_digest.as_deref(),
            )
        }
        ContextMode::Request => {
            require(
                !context.is_empty() && context.len() <= MAX_CONTEXT_ROWS,
                "context must contain 1..100000 rows",
            )?;
            let schema = if predictor.schema_free() {
                predictors::inferred(&forecast.context_columns(), target, context)
            } else {
                predictor.schema.clone()
            };
            let rows = predictors::records(&schema.columns, context)?;
            aligned = front.bind_series(served, predictor, forecast, &rows, None)?;
            (&aligned, schema, "request", None)
        }
    };
    let expected: Vec<&str> = schema
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .filter(|name| forecast.query_columns().contains(name))
        .collect();
    require(features == expected.as_slice(), "features differ from lock")?;
    let rows = forecast.query_rows(&schema, predictor.id.as_deref(), queries)?;
    let queried = forecast.queries(&bound.context, &rows)?;
    let query = bound.forecast.queries.first().ok_or_else(failed)?;
    let horizon = forecast.horizon;
    let mut group = query.group.clone();
    let mut covariates: Vec<&String> = forecast
        .past_inputs
        .iter()
        .chain(&forecast.future_inputs)
        .collect();
    covariates.sort();
    covariates.dedup();
    let mut known = Sha256::new();
    known.update(b"predictor-forecast\0");
    known.update(query.fingerprint);
    for (name, values) in forecast.future_inputs.iter().zip(&queried.future) {
        let row = bound.context.identities.len()
            + covariates
                .iter()
                .position(|covariate| *covariate == name)
                .ok_or_else(failed)?;
        if group.roles.get(row) != Some(&KNOWN) || group.horizon != horizon {
            return Err(failed());
        }
        for (step, &value) in values.iter().enumerate() {
            group.future[row * horizon + step] = value as f32;
            group.future_observed[row * horizon + step] = true;
            known.update((value as f32).to_le_bytes());
        }
    }
    let fingerprint: [u8; 32] = if forecast.future_inputs.is_empty() {
        query.fingerprint
    } else {
        known.finalize().into()
    };
    let inferred = front
        .engine()?
        .forecast(
            &served.worker,
            &group,
            &fingerprint,
            &queried.members,
            horizon,
        )
        .await?;
    drop(charge);
    let invalid = || {
        ApiError::new(
            "ModelOutputError",
            "native forecaster returned invalid forecasts",
            503,
        )
    };
    let levels = inferred.levels.as_deref().ok_or_else(invalid)?;
    let column = |level: f64| levels.iter().position(|known| *known == level);
    let median = column(0.5).ok_or_else(invalid)?;
    let columns: Vec<usize> = predictor
        .quantiles
        .iter()
        .map(|&level| column(level))
        .collect::<Option<_>>()
        .ok_or_else(|| {
            ApiError::new(
                "UnsupportedQuantiles",
                "the model only returns its own published quantile levels",
                422,
            )
        })?;
    let predicted = &inferred.predictions;
    if (predicted.rows, predicted.horizon, predicted.columns)
        != (queried.members.len(), Some(horizon), levels.len())
        || !predicted.values.iter().all(|value| value.is_finite())
    {
        return Err(invalid());
    }
    let at = |member: usize, position: usize, column: usize| {
        let row = queried
            .members
            .iter()
            .position(|&queried| queried == member)
            .unwrap_or_default();
        f64::from(predicted.values[(row * horizon + position) * predicted.columns + column])
    };
    let mut values: HashMap<&str, Vec<Value>> = HashMap::new();
    values.insert(
        "value",
        queried
            .positions
            .iter()
            .map(|&(member, position)| json!(at(member, position, median)))
            .collect(),
    );
    values.insert(
        "quantiles",
        queried
            .positions
            .iter()
            .map(|&(member, position)| {
                Value::Object(
                    predictor
                        .quantiles
                        .iter()
                        .zip(&columns)
                        .map(|(&level, &column)| {
                            (level_key(level), json!(at(member, position, column)))
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    let predictions = projected(
        values,
        &served.outputs,
        queries,
        predictor.id.as_deref(),
        queried.positions.len(),
    );
    Ok(prediction_result(
        &served.resolved,
        &predictor.task,
        source,
        version,
        bound.rows,
        predictions,
    ))
}

/// A bound request context's fingerprint: its model and its rows as the
/// binder encoded them, which is all its table holds, so contexts that
/// encode alike share one table and one fit.
fn bound_fingerprint(served: &ServedModel, bound: &predictor::Context) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"predictor\0");
    hasher.update(served.worker.as_bytes());
    hasher.update([0]);
    hasher.update((bound.encoder.width() as u64).to_le_bytes());
    hasher.update((bound.classes.len() as u64).to_le_bytes());
    let features: Vec<u8> = bound
        .features
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    hasher.update(&features);
    let targets: Vec<u8> = match &bound.targets {
        Targets::Classes(codes) => codes.iter().flat_map(|code| code.to_le_bytes()).collect(),
        Targets::Values(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    };
    hasher.update(&targets);
    hasher.finalize().into()
}

/// A hosted predictor: the revision its `production` environment serves,
/// loaded from the control plane on demand and predicted as a native predictor.
async fn revision_predict(
    front: &Front,
    revisions: &Revisions,
    project: &str,
    predictor: &str,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let environment = "production";
    let resolved = revisions.resolve(project, predictor, environment).await?;
    let revision = resolved.revision;
    let version = resolved.context_version.unwrap_or(Value::Null);
    let started = std::time::Instant::now();
    let scored = async {
        let published = revisions
            .load(project, predictor, revision, Some(&version))
            .await?;
        let Published::Native {
            served,
            context,
            rows,
        } = &*published;
        native_predict(
            front,
            served,
            context.as_ref(),
            rows.as_deref(),
            payload,
            charge,
        )
        .await
    }
    .await;
    revisions.record(Record {
        project: project.to_string(),
        predictor: predictor.to_string(),
        environment: Some(environment.to_string()),
        revision,
        ok: scored.is_ok(),
        latency_seconds: started.elapsed().as_secs_f64(),
    });
    let mut result = scored?;
    // The hosted result keeps the control plane's context shape: `source`
    // and `version`, which its Python path and console read.
    if let Some(context) = result["context"].as_object_mut() {
        if let Some(kind) = context.remove("kind") {
            context.insert("source".into(), kind);
        }
        if let Some(digest) = context.remove("digest") {
            context.insert("version".into(), digest);
        }
    }
    result["predictor"] = json!(predictor);
    result["revision"] = json!(revision);
    result["environment"] = json!(environment);
    Ok(result)
}

async fn native_predict(
    front: &Front,
    served: &ServedModel,
    captured: Option<&Value>,
    rows: Option<&[Value]>,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    refuse_fields(
        payload,
        &["target", "inputs", "model", "task", "features", "options"],
        "is declared by the predictor",
    )?;
    let body = fields(payload, &["queries"], &["context", "as_of"])?;
    let failed = || ApiError::new("InternalError", "request failed safely", 500);
    let predictor = served.predictor.as_ref().ok_or_else(failed)?;
    let base = front
        .models
        .iter()
        .find(|model| model.predictor.is_none() && model.resolved == served.resolved)
        .ok_or_else(|| {
            ApiError::new(
                "ModelUnavailable",
                "the revision's model is not served here",
                503,
            )
        })?;
    require(
        body["queries"]
            .as_array()
            .is_some_and(|queries| (1..=MAX_QUERY_ROWS).contains(&queries.len())),
        "queries must contain 1..10000 rows",
    )?;
    let context = match predictor.context {
        ContextMode::Request => body.get("context").cloned().ok_or_else(|| {
            ApiError::validation(
                "context is required: this predictor takes its context with each request",
            )
            .param("context")
        })?,
        ContextMode::Snapshot => {
            if body.contains_key("context") {
                return Err(context_declared());
            }
            Value::Array(rows.ok_or_else(failed)?.to_vec())
        }
    };
    let mut ephemeral = served.clone();
    ephemeral.worker.clone_from(&base.worker);
    let mut bound = predictor.clone();
    bound.context = ContextMode::Request;
    bound.snapshot = None;
    ephemeral.predictor = Some(bound.clone());
    let call = json!({
        "target": predictor.target,
        "inputs": predictor.inputs,
        "context": context,
        "queries": body["queries"],
    });
    let mut result = served_predict(front, &ephemeral, &bound, &call, charge).await?;
    if predictor.context == ContextMode::Snapshot
        && let Some(captured) = captured
    {
        let rows = result["context"]["rows"].clone();
        result["context"] = captured.clone();
        result["context"]["rows"] = rows;
    }
    Ok(result)
}
