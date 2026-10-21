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
use futures::future::BoxFuture;
use gf_core::catalog::{self, Admission, Surface};

use gf_core::forecasting::KNOWN;
use gf_core::node::Targets;
use gf_core::request::{self, Forecast, Inference, Label, Limits, Num, Options, RequestError};
use serde::Deserialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::engine::{self, Bound, Engine};
use crate::packages::{self, ContextMode, ForecastPackage, ServedPackage};
use crate::revisions::{Published, Record, Revisions, Selector};
use crate::routes::Routes;
use crate::service::{Charge, RequestBudget, quantile_values};
use crate::shard::{Asked, Command, Inferred, ServerError};
use crate::upstream::Upstream;
use arrow_array::RecordBatch;
use gf_core::package;

/// The API version every envelope carries.
pub const VERSION: &str = "gf/v1alpha1";
/// The most bytes a request body may hold.
pub const MAX_BYTES: usize = 1_048_576;
/// The deepest and largest JSON a body may be, as the hosted API bounds it.
const MAX_DEPTH: usize = 16;
const MAX_NODES: usize = 65_536;
const MAX_INTEGER: i64 = (1 << 53) - 1;
/// The kinds of resource an envelope may name.
const RESOURCE_KINDS: [&str; 11] = [
    "workspace",
    "project",
    "dataset",
    "source",
    "model",
    "deployment",
    "job",
    "package",
    "operator",
    "api-key",
    "secret",
];
/// The most query and context rows a package's `/predictions` takes.
const MAX_QUERY_ROWS: usize = 10_000;
const MAX_CONTEXT_ROWS: usize = 100_000;
/// The quantiles a regressor's `/predictions` reports when it asks for none.
const DEFAULT_QUANTILES: [f64; 3] = [0.1, 0.5, 0.9];
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
}

impl ApiError {
    #[must_use]
    pub fn new(code: &str, message: impl Into<String>, status: u16) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            status,
        }
    }

    /// Malformed input, as the hosted API's `require` refuses it.
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::new("ValidationError", message, 422)
    }

    fn not_served() -> Self {
        Self::new("ModelNotFound", "model is not served here", 404)
    }

    fn unauthorized(message: &str) -> Self {
        Self::new("Unauthorized", message, 401)
    }

    fn body(&self) -> Value {
        json!({"error": {"code": self.code, "message": self.message}})
    }
}

impl From<RequestError> for ApiError {
    fn from(error: RequestError) -> Self {
        Self::new(error.code, error.message, error.status)
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
    /// The locked package served, if the model is one.
    #[serde(default)]
    pub package: Option<ServedPackage>,
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

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
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
/// seconds or until they expire, refused ones for 5.
pub struct Introspect {
    client: reqwest::Client,
    url: String,
    token: String,
    cache: Mutex<HashMap<[u8; 32], Cached>>,
}

/// An introspection answer, and until when it is trusted.
type Cached = (Option<Principal>, Instant);

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
            url: format!(
                "{}/internal/v1alpha1/keys:introspect",
                base.trim_end_matches('/')
            ),
            token,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn cached(&self, verifier: &[u8; 32]) -> Option<Option<Principal>> {
        let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .get(verifier)
            .filter(|(_, until)| *until > Instant::now())
            .map(|(principal, _)| principal.clone())
    }

    fn remember(&self, verifier: [u8; 32], principal: Option<Principal>, ttl: Duration) {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if cache.len() >= CACHED_KEYS {
            cache.clear();
        }
        cache.insert(verifier, (principal, Instant::now() + ttl));
    }

    async fn ask(&self, verifier: &[u8; 32]) -> Result<(Option<Principal>, Duration), ApiError> {
        let unavailable =
            || ApiError::new("CredentialUnavailable", "cannot check API keys now", 503);
        let hex: String = verifier.iter().map(|byte| format!("{byte:02x}")).collect();
        let response = self
            .client
            .post(&self.url)
            .bearer_auth(&self.token)
            .json(&json!({"verifier": hex}))
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
            Some(expires) => Duration::from_secs_f64(expires - now.as_secs_f64()),
            None => POSITIVE_TTL,
        };
        let principal = Principal {
            project: Some(found.project),
            permissions: Some(found.permissions),
        };
        Ok((Some(principal), left.min(POSITIVE_TTL)))
    }
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
                    let (principal, ttl) = self.ask(&verifier).await?;
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
    List,
    Get,
    Infer,
    Predict,
    /// A deployment's prediction, from the revision the selector names.
    Score(Selector),
}

impl Action {
    fn permission(&self) -> &'static str {
        match self {
            Self::List | Self::Get => "models:read",
            Self::Infer | Self::Predict => "models:infer",
            Self::Score(_) => "deployments:score",
        }
    }

    /// The resource kind the action's envelope names.
    fn kind(&self) -> &'static str {
        match self {
            Self::Score(_) => "deployment",
            _ => "model",
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
    /// Snapshot packages' tables, by package name, once they are bound.
    bound: OnceLock<HashMap<String, Bound>>,
    /// Snapshot forecast packages' series, by package name, once aligned.
    series: OnceLock<HashMap<String, SeriesBound>>,
    draining: Arc<AtomicBool>,
    /// Where the routes the front does not own stream to, if anywhere.
    upstream: Option<Upstream>,
    /// The deployed revisions predicted from, if any are.
    revisions: Option<Revisions>,
}

/// A forecast package's context: its series aligned, and their forecast
/// request parsed, once.
struct SeriesBound {
    context: packages::ForecastContext,
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
        }
    }

    /// Streams every route the front does not own to `upstream`.
    #[must_use]
    pub fn with_upstream(mut self, upstream: Upstream) -> Self {
        self.upstream = Some(upstream);
        self
    }

    /// Predicts deployments' revisions, loaded from `revisions` on demand.
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
    /// none from a browser, as a deployment behind one endpoint does.
    #[must_use]
    pub fn with_hosts(mut self, hosts: Vec<String>) -> Self {
        self.hosts = hosts;
        self
    }

    /// Starts answering inference once the shards' workers are up, and is
    /// ready once each snapshot package's rows are bound into their table.
    ///
    /// # Errors
    ///
    /// Which snapshot package could not be bound, and why.
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
            if let Some(package) = &served.package
                && package.context == ContextMode::Snapshot
            {
                let refused =
                    |error: ApiError| format!("package {:?}: {}", served.name, error.message);
                if let Some(forecast) = &package.forecast {
                    let rows = snapshot_rows(package).map_err(refused)?;
                    let aligned = self
                        .bind_series(served, package, forecast, &rows, forecast.frequency)
                        .map_err(refused)?;
                    series.insert(served.name.clone(), aligned);
                } else {
                    let snapshot = engine
                        .bind_snapshot(&served.worker, package)
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

    /// Whether the front takes inference: its shards are up, its snapshot
    /// packages bound, and it is not draining.
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
    pub fn names(&self) -> Vec<String> {
        self.models.iter().map(|model| model.name.clone()).collect()
    }
}

/// The router answering every path, so unknown routes get the API's errors.
pub fn router(front: Arc<Front>) -> Router {
    Router::new().fallback(handle).with_state(front)
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
            body.insert(
                "error".into(),
                json!({"code": error.code, "message": error.message}),
            );
            Value::Object(body)
        }
        None => error.body(),
    };
    respond(error.status, &body)
}

async fn handle(State(front): State<Arc<Front>>, request: Request) -> Response {
    if let Some(upstream) = &front.upstream
        && !owns(request.uri().path())
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
    // What the body holds counts against the server's request budget until answered.
    let mut charge = front.budget.charge();
    charge.grow(length as u64).map_err(|_| {
        ApiError::new(
            "CapacityExceeded",
            "the server is at capacity; retry shortly",
            503,
        )
    })?;
    let body = axum::body::to_bytes(body, MAX_BYTES)
        .await
        .map_err(|_| ApiError::new("ValidationError", "invalid HTTP message", 400))?;
    require(body.len() == length, "truncated request body")?;
    require(
        parts.method != Method::GET || body.is_empty(),
        "GET bodies are forbidden",
    )?;
    if target == "/livez" || target == "/healthz" {
        return Ok(respond(200, &json!({"live": true})));
    }
    if target == "/readyz" {
        let ready = front.ready()
            && match &front.upstream {
                Some(upstream) => upstream.ready().await,
                None => true,
            };
        return Ok(respond(
            if ready { 200 } else { 503 },
            &json!({"ready": ready}),
        ));
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
        None => format!("r-{}", random_hex()),
    };
    require(
        request_id.len() <= 128 && request_id.bytes().all(|byte| (33..=126).contains(&byte)),
        "invalid request id",
    )?;
    if let Some(given) = header("x-gf-project") {
        require(given == project, "project header/path mismatch")?;
    }
    let payload = if parts.method == Method::GET {
        *scope = Some(Scope {
            project: project.clone(),
            request_id,
            kind: "model".into(),
            name: resource.clone(),
        });
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
            parsed.kind == action.kind() && parsed.name == resource,
            "resource route/envelope mismatch",
        )?;
        payload
    };
    let principal = front.keys.verify(header("authorization")).await?;
    principal.authorize(&project, action.permission())?;
    if !front.ready() {
        return Err(ApiError::new("NotReady", "models are not ready", 503));
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

/// A resource name: a lowercase letter, then up to 62 letters, digits or `-`.
fn name(value: &str) -> Result<String, ApiError> {
    let mut bytes = value.bytes();
    let valid = value.len() <= 63
        && bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
    require(valid, "invalid name")?;
    Ok(value.to_string())
}

/// Maps the Models API paths to a project, action and resource; every other
/// path is NotFound.
/// Whether `path` is the front's to answer, whatever its method or
/// spelling, so no inference route, nor a malformed one, is ever proxied.
#[must_use]
pub fn owns(path: &str) -> bool {
    if matches!(path, "/livez" | "/healthz" | "/readyz") {
        return true;
    }
    let parts: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
    let [api, projects, _, tail @ ..] = parts.as_slice() else {
        return false;
    };
    if !api.eq_ignore_ascii_case("v1alpha1") || !projects.eq_ignore_ascii_case("projects") {
        return false;
    }
    let lower: Vec<String> = tail.iter().map(|part| part.to_ascii_lowercase()).collect();
    match lower
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => false,
        [first, ..] if first.contains('%') => true,
        ["models", ..] | ["predictions", ..] => true,
        [
            "deployments",
            _,
            "revisions" | "environments",
            _,
            "predict",
            ..,
        ] => true,
        _ => false,
    }
}

fn resolve(method: &Method, path: &str) -> Result<(String, Action, String), ApiError> {
    let not_found = || ApiError::new("NotFound", "route not found", 404);
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 5 || parts[..3] != ["", "v1alpha1", "projects"] {
        return Err(not_found());
    }
    if parts[3..].iter().any(|part| part.is_empty()) {
        return Err(not_found());
    }
    let project = name(parts[3])?;
    let tail = &parts[4..];
    match (method, tail) {
        (&Method::GET, ["models"]) => Ok((project, Action::List, "models".into())),
        (&Method::GET, ["models", model]) => Ok((project, Action::Get, name(model)?)),
        (&Method::POST, ["models", model, "infer"]) => Ok((project, Action::Infer, name(model)?)),
        (&Method::POST, ["predictions"]) => Ok((project, Action::Predict, "predictions".into())),
        (
            &Method::POST,
            [
                "deployments",
                deployment,
                "environments",
                environment,
                "predict",
            ],
        ) => Ok((
            project,
            Action::Score(Selector::Environment(name(environment)?)),
            name(deployment)?,
        )),
        (&Method::POST, ["deployments", deployment, "revisions", number, "predict"]) => {
            let revision = Some(number)
                .filter(|number| {
                    number.len() <= 18
                        && !number.starts_with('0')
                        && number.bytes().all(|byte| byte.is_ascii_digit())
                })
                .and_then(|number| number.parse::<u64>().ok())
                .ok_or_else(not_found)?;
            Ok((
                project,
                Action::Score(Selector::Revision(revision)),
                name(deployment)?,
            ))
        }
        _ => Err(not_found()),
    }
}

fn random_hex() -> String {
    let mut bytes = [0u8; 16];
    let _ = getrandom::fill(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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
    let resource_name = name(resource["name"].as_str().unwrap_or_default())?;
    require(object["payload"].is_object(), "expected JSON object")?;
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
            .find(|served| {
                served.name == model || canonical.as_deref() == Some(served.name.as_str())
            })
            .ok_or_else(ApiError::not_served)
    }

    /// The catalog model served as exactly this reference, variant and all.
    fn serving(&self, model: &catalog::Model) -> Result<&ServedModel, ApiError> {
        let reference = model.to_string();
        self.models
            .iter()
            .find(|served| served.package.is_none() && served.reference == reference)
            .ok_or_else(ApiError::not_served)
    }

    /// Admits a payload's model only if it is exactly the reference served,
    /// variant and all.
    fn admitted(&self, served: &ServedModel, model: &Value) -> Result<(), ApiError> {
        let reference = model
            .as_str()
            .and_then(|model| catalog::resolve(Some(model), self.surface).ok())
            .map(|resolved| resolved.to_string());
        if reference.as_deref() == Some(served.reference.as_str()) {
            Ok(())
        } else {
            Err(ApiError::not_served())
        }
    }

    /// The engine, once the shards' workers are up.
    fn engine(&self) -> Result<&Engine, ApiError> {
        self.engine
            .get()
            .ok_or_else(|| ApiError::new("NotReady", "models are not ready", 503))
    }

    /// Aligns a forecast package's context and parses its group's forecast
    /// request, as `ForecastContext.build` does in process.
    fn bind_series(
        &self,
        served: &ServedModel,
        package: &ServedPackage,
        forecast: &ForecastPackage,
        rows: &RecordBatch,
        frequency: Option<f64>,
    ) -> Result<SeriesBound, ApiError> {
        let failed = || ApiError::new("InternalError", "request failed safely", 500);
        let target = package.target.as_deref().ok_or_else(|| {
            ApiError::new("InvalidPackage", "a forecast package names its target", 422)
        })?;
        let context = forecast.context(
            &served.reference,
            target,
            &package.quantiles,
            rows,
            frequency,
        )?;
        let bytes = serde_json::to_vec(&context.request).map_err(|_| failed())?;
        let options = Options {
            surface: self.surface,
            limits: Limits::Package,
            levels: None,
            admission: Admission::Catalog,
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
            "models": front.models.iter().map(|model| model.description.clone()).collect::<Vec<_>>()
        })),
        Action::Get => Ok(front.served(resource)?.description.clone()),
        Action::Infer => {
            if front
                .models
                .iter()
                .any(|model| model.name == resource && model.package.is_some())
            {
                return Err(ApiError::new(
                    "UnsupportedCapability",
                    "packages serve POST predictions with the package name as model",
                    422,
                ));
            }
            // The catalog's answer comes first: a request it refuses gets its
            // code, and only a valid one this server does not serve is not found.
            let bytes = serde_json::to_vec(&payload)
                .map_err(|_| ApiError::new("InternalError", "request failed safely", 500))?;
            let inference = request::parse(&bytes, &Options::request(front.surface))?;
            let named =
                catalog::resolve(payload.get("model").and_then(Value::as_str), front.surface)?;
            let route = catalog::resolve(Some(resource), front.surface).ok();
            require(
                route.is_some_and(|route| route.entry.name == named.entry.name),
                "model route mismatch",
            )?;
            let served = front.serving(&named)?;
            infer(front, served, inference, charge).await
        }
        Action::Predict => {
            let model = payload
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let packaged = front
                .models
                .iter()
                .find(|served| served.name == model && served.package.is_some());
            let served = match packaged {
                Some(served) => served,
                None => front.serving(&catalog::resolve(Some(model), front.surface)?)?,
            };
            match &served.package {
                Some(package) => package_predict(front, served, package, &payload, charge).await,
                None => predict(front, served, &payload, charge).await,
            }
        }
        Action::Score(selector) => {
            revision_predict(front, project, resource, &selector, &payload, charge).await
        }
    }
}

/// A deployment's prediction, from the revision its environment serves or
/// the route names, in the control plane's `PredictionResult`; each one is
/// reported back for the deployment's metrics and events.
async fn revision_predict(
    front: &Front,
    project: &str,
    deployment: &str,
    selector: &Selector,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let revisions = front.revisions.as_ref().ok_or_else(|| {
        ApiError::new(
            "RuntimeUnavailable",
            "deployed revisions are not served here",
            503,
        )
    })?;
    let (environment, revision, version) = match selector {
        Selector::Environment(environment) => {
            let resolved = revisions.resolve(project, deployment, environment).await?;
            (
                Some(environment.as_str()),
                resolved.revision,
                Some(resolved.context_version.unwrap_or(Value::Null)),
            )
        }
        Selector::Revision(revision) => (None, *revision, None),
    };
    let started = std::time::Instant::now();
    let scored = async {
        let published = revisions
            .load(project, deployment, revision, version.as_ref())
            .await?;
        match &*published {
            Published::Python {} => {
                drop(charge);
                revisions
                    .execute(project, deployment, revision, environment, payload)
                    .await
            }
            Published::Native {
                served,
                context,
                rows,
            } => {
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
        }
    }
    .await;
    revisions.record(Record {
        project: project.to_string(),
        deployment: deployment.to_string(),
        environment: environment.map(str::to_string),
        revision,
        ok: scored.is_ok(),
        latency_seconds: started.elapsed().as_secs_f64(),
    });
    let mut result = scored?;
    result["deployment"] = json!(deployment);
    result["revision"] = json!(revision);
    result["environment"] = json!(environment);
    Ok(result)
}

/// A catalog revision's prediction: the payload checked as the control
/// plane checks it against the lock, then predicted on the package path with
/// the revision's captured rows, or the request's own, as an ephemeral
/// context.
async fn native_predict(
    front: &Front,
    served: &ServedModel,
    captured: Option<&Value>,
    rows: Option<&[Value]>,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let body = fields(
        payload,
        &["queries"],
        &["context", "target", "features", "options"],
    )?;
    let failed = || ApiError::new("InternalError", "request failed safely", 500);
    let package = served.package.as_ref().ok_or_else(failed)?;
    let base = front
        .models
        .iter()
        .find(|model| model.package.is_none() && model.resolved == served.resolved)
        .ok_or_else(|| {
            ApiError::new(
                "ModelUnavailable",
                "the revision's model is not served here",
                503,
            )
        })?;
    if !package.schema_free() {
        require(
            body.get("target")
                .is_none_or(|target| target.as_str() == package.target.as_deref()),
            "target differs from lock",
        )?;
        require(
            body.get("features")
                .is_none_or(|features| *features == json!(package.features)),
            "features differ from lock",
        )?;
    }
    require(
        body.get("options").is_none_or(|options| {
            options.is_null() || options.as_object().is_some_and(Map::is_empty)
        }),
        "prediction options must be compiled into the package",
    )?;
    require(
        body["queries"]
            .as_array()
            .is_some_and(|queries| (1..=MAX_QUERY_ROWS).contains(&queries.len())),
        "queries must contain 1..10000 rows",
    )?;
    let context = match package.context {
        ContextMode::Request => body
            .get("context")
            .cloned()
            .ok_or_else(|| ApiError::validation("request context is required"))?,
        ContextMode::Snapshot => {
            require(
                !body.contains_key("context"),
                "bound context cannot be overridden",
            )?;
            Value::Array(rows.ok_or_else(failed)?.to_vec())
        }
    };
    let mut ephemeral = served.clone();
    ephemeral.worker.clone_from(&base.worker);
    let mut bound = package.clone();
    bound.context = ContextMode::Request;
    bound.snapshot = None;
    ephemeral.package = Some(bound.clone());
    let call = json!({
        "model": ephemeral.name,
        "task": package.task,
        "target": body.get("target").cloned().unwrap_or_else(|| json!(package.target)),
        "features": body.get("features").cloned().unwrap_or_else(|| json!(package.features)),
        "context": context,
        "queries": body["queries"],
        "options": {},
    });
    let mut result = package_predict(front, &ephemeral, &bound, &call, charge).await?;
    if package.context == ContextMode::Snapshot
        && let Some(captured) = captured
    {
        let rows = result["context"]["rows"].clone();
        result["context"] = captured.clone();
        result["context"]["rows"] = rows;
    }
    Ok(result)
}

/// The hosted `models/{name}/infer` result for the served reference.
async fn infer(
    front: &Front,
    served: &ServedModel,
    inference: Inference,
    charge: Charge,
) -> Result<Value, ApiError> {
    let request = match inference {
        Inference::Tabular(request) => request,
        Inference::Forecast(request) => return forecasts(front, served, &request, charge).await,
    };
    let inferred = front
        .engine()?
        .tabular(&served.worker, &request, charge)
        .await?;
    let outputs: Vec<Value> = if request.classes.is_empty() {
        quantiles(&inferred, &request.levels)?
            .into_iter()
            .zip(&request.query_keys)
            .map(|(quantiles, key)| json!({"key": key, "quantiles": quantiles}))
            .collect()
    } else {
        probabilities(&request.classes, &inferred)
            .into_iter()
            .zip(&request.query_keys)
            .map(|(probabilities, key)| json!({"key": key, "probabilities": probabilities}))
            .collect()
    };
    Ok(json!({
        "task": request.task.as_str(),
        "outputs": outputs,
        "artifact": verified(served),
        "context_provenance": {
            "source": "request",
            "supplied_rows": request.supplied_rows,
            "usable_labeled_rows": request.usable_rows(),
            "features": request.features,
            "target": request.target,
            "persisted": false,
            "classes": request.classes.iter().map(label_json).collect::<Vec<_>>(),
        },
    }))
}

/// The served model's artifact, verified by the worker that loaded it.
/// A forecast's artifact: the verified artifact and its license's source,
/// as the catalog's forecasts report them.
fn forecast_artifact(served: &ServedModel) -> Value {
    let mut artifact = verified(served);
    if let Some(fields) = artifact.as_object_mut() {
        fields.insert("license_source".into(), json!(served.license_source));
    }
    artifact
}

fn verified(served: &ServedModel) -> Value {
    let mut artifact = served.description["artifact"].clone();
    if let Some(artifact) = artifact.as_object_mut() {
        artifact.insert("verified".into(), json!(true));
    }
    artifact
}

fn num_json(value: Num) -> Value {
    match value {
        Num::Int(value) => json!(value),
        Num::Float(value) => json!(value),
    }
}

/// The hosted `models/{name}/infer` result of a forecast: each query's
/// series forecast from its aligned group's table, a table per group
/// fingerprint, the groups asked of the engine at once.
async fn forecasts(
    front: &Front,
    served: &ServedModel,
    request: &Forecast,
    charge: Charge,
) -> Result<Value, ApiError> {
    let groups = front
        .engine()?
        .forecast_request(&served.worker, request)
        .await?;
    drop(charge);
    let invalid = || {
        ApiError::new(
            "ModelOutputError",
            "native forecaster returned invalid forecasts",
            503,
        )
    };
    let mut outputs = vec![Value::Null; request.queries.len()];
    for (indices, inferred) in &groups {
        let levels = inferred.levels.as_deref().ok_or_else(invalid)?;
        let columns: Vec<usize> = request
            .levels
            .iter()
            .map(|level| levels.iter().position(|known| known == level))
            .collect::<Option<_>>()
            .ok_or_else(|| {
                ApiError::new(
                    "UnsupportedQuantiles",
                    "the model only returns its own published quantile levels",
                    422,
                )
            })?;
        let forecast = &inferred.predictions;
        let steps = request.horizon;
        if (forecast.rows, forecast.horizon, forecast.columns)
            != (indices.len(), Some(steps), levels.len())
            || !forecast.values.iter().all(|value| value.is_finite())
        {
            return Err(invalid());
        }
        for (row, &index) in indices.iter().enumerate() {
            let query = &request.queries[index];
            let quantiles: Map<String, Value> = request
                .levels
                .iter()
                .zip(&columns)
                .map(|(&level, &column)| {
                    let path: Vec<f64> = (0..steps)
                        .map(|step| {
                            f64::from(
                                forecast.values[(row * steps + step) * forecast.columns + column],
                            )
                        })
                        .collect();
                    (level_key(level), json!(path))
                })
                .collect();
            outputs[index] = json!({
                "key": query.key,
                "series": query.series,
                "origin": num_json(query.origin),
                "times": query.times.iter().copied().map(num_json).collect::<Vec<_>>(),
                "quantiles": quantiles,
            });
        }
    }
    Ok(json!({
        "task": "forecast",
        "outputs": outputs,
        "artifact": forecast_artifact(served),
        "context_provenance": {
            "source": "request",
            "supplied_rows": request.supplied_rows,
            "usable_observations": request.usable_observations,
            "persisted": false,
            "covariate_policy": "shared-within-query-group",
            "time_axis": request.clock.as_str(),
        },
    }))
}

fn forecasting_unserved() -> ApiError {
    ApiError::new(
        "UnsupportedCapability",
        "this server answers tabular models; forecasters are served by the Python engine",
        422,
    )
}

/// The `/predictions` body for a catalog model: flat rows, bound as an
/// inference request whose context is the rows with a target, then
/// answered as the shared `PredictionResult`.
async fn predict(
    front: &Front,
    served: &ServedModel,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let body = fields(
        payload,
        &[
            "model", "task", "target", "features", "context", "queries", "options",
        ],
        &[],
    )?;
    front.admitted(served, &body["model"])?;
    let options = fields(&body["options"], &[], &["id", "quantiles", "outputs"])?;
    let task = body["task"].as_str().unwrap_or_default();
    let target = &body["target"];
    let (Some(rows), Some(queries)) = (body["context"].as_array(), body["queries"].as_array())
    else {
        return Err(ApiError::validation("rows must be lists"));
    };
    require(
        rows.iter().chain(queries).all(Value::is_object),
        "rows must be objects",
    )?;
    let features = body["features"]
        .as_array()
        .ok_or_else(|| ApiError::validation("features must be a list"))?;
    let outputs: Vec<String> = match options.get("outputs") {
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
    let levels = options.get("quantiles");
    require(
        levels.is_none() || outputs.iter().any(|output| output == "quantiles"),
        "quantiles require the quantiles output",
    )?;
    let target_name = target.as_str().unwrap_or_default();
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
    let inferred = front
        .engine()?
        .tabular(&served.worker, &parsed, charge)
        .await?;
    let predictions = prediction_rows(
        &parsed.classes,
        &parsed.levels,
        &inferred,
        &outputs,
        queries,
        options.get("id").and_then(Value::as_str),
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
        "context": {"source": source, "version": version, "rows": rows, "as_of": now},
        "deployment": null,
        "revision": null,
        "environment": null,
        "predictions": predictions,
    })
}

/// A locked package's `/predictions`: the body checked against the lock,
/// then the queries encoded by the binder and predicted from the
/// snapshot's durable table, or from the ephemeral table of the request's
/// own bound context.
async fn package_predict(
    front: &Front,
    served: &ServedModel,
    package: &ServedPackage,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let body = fields(
        payload,
        &[
            "model", "task", "target", "features", "context", "queries", "options",
        ],
        &[],
    )?;
    if body["model"].as_str() != Some(served.name.as_str()) {
        return Err(ApiError::not_served());
    }
    if body["task"].as_str() != Some(package.task.as_str()) {
        return Err(ApiError::new(
            "UnsupportedTask",
            "package does not serve this task",
            422,
        ));
    }
    require(
        body["options"] == json!({}),
        "package options are compiled into its lock",
    )?;
    let features: Vec<String> = body["features"]
        .as_array()
        .ok_or_else(|| ApiError::validation("features must be a list"))?
        .iter()
        .map(|feature| feature.as_str().map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| ApiError::validation("features differ from lock"))?;
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
    if let Some(forecast) = &package.forecast {
        return series_predict(
            front, served, package, forecast, body, &features, queries, context, charge,
        )
        .await;
    }
    let schema_free = package.schema_free();
    let target = if schema_free {
        let target = body["target"]
            .as_str()
            .ok_or_else(|| ApiError::validation("request target is required"))?;
        require(
            package
                .target
                .as_deref()
                .is_none_or(|locked| locked == target),
            "target differs from lock",
        )?;
        target.to_string()
    } else {
        require(
            body["target"].as_str().is_some()
                && body["target"].as_str() == package.target.as_deref(),
            "target differs from lock",
        )?;
        require(features == package.features, "features differ from lock")?;
        package.target.clone().unwrap_or_default()
    };
    let (inferred, classes, rows, source, version) = match package.context {
        ContextMode::Snapshot => {
            require(context.is_empty(), "bound context cannot be overridden")?;
            let bound = front
                .bound
                .get()
                .and_then(|bound| bound.get(&served.name))
                .ok_or_else(|| ApiError::new("NotReady", "model is loading", 503))?;
            let encoded = bound
                .encoder
                .encode(&packages::records(&package.schema.columns, queries)?)?;
            let inferred = front.engine()?.predicted(bound, encoded).await?;
            (
                inferred,
                bound.classes.clone(),
                bound.rows,
                "snapshot",
                package.data_digest.as_deref(),
            )
        }
        ContextMode::Request => {
            require(
                !context.is_empty() && context.len() <= MAX_CONTEXT_ROWS,
                "context must contain 1..100000 rows",
            )?;
            let schema = if schema_free {
                packages::inferred(&features, &target, context)
            } else {
                package.schema.clone()
            };
            let bound = package::bind(
                &schema,
                package.task()?,
                &target,
                &packages::records(&schema.columns, context)?,
            )?;
            let encoded = bound
                .encoder
                .encode(&packages::records(&schema.columns, queries)?)?;
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
        &package.quantiles,
        &inferred,
        &served.outputs,
        queries,
        package.id.as_deref(),
    )?;
    Ok(prediction_result(
        &served.resolved,
        &package.task,
        source,
        version,
        rows,
        predictions,
    ))
}

/// A snapshot package's context rows, from the file `gf serve` wrote.
pub(crate) fn snapshot_rows(package: &ServedPackage) -> Result<RecordBatch, ApiError> {
    let path = package.snapshot.as_deref().ok_or_else(|| {
        ApiError::new(
            "ContextRequired",
            "snapshot packages are served from their package directory",
            422,
        )
    })?;
    packages::read_snapshot(path).map_err(|error| ApiError::new("MissingFile", error, 503))
}

/// A forecast package's `/predictions`: the queries placed on the bound
/// group's series and horizon, the known inputs' future filled in, and the
/// group forecast on the shard path catalog forecasts take, from the table
/// its context and future values name.
#[allow(clippy::too_many_arguments)]
async fn series_predict(
    front: &Front,
    served: &ServedModel,
    package: &ServedPackage,
    forecast: &ForecastPackage,
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
        package
            .target
            .as_deref()
            .is_none_or(|locked| locked == target),
        "target differs from lock",
    )?;
    let aligned;
    let (bound, schema, source, version) = match package.context {
        ContextMode::Snapshot => {
            require(context.is_empty(), "bound context cannot be overridden")?;
            let bound = front
                .series
                .get()
                .and_then(|series| series.get(&served.name))
                .ok_or_else(|| ApiError::new("NotReady", "model is loading", 503))?;
            (
                bound,
                package.schema.clone(),
                "snapshot",
                package.data_digest.as_deref(),
            )
        }
        ContextMode::Request => {
            require(
                !context.is_empty() && context.len() <= MAX_CONTEXT_ROWS,
                "context must contain 1..100000 rows",
            )?;
            let schema = if package.schema_free() {
                packages::inferred(&forecast.context_columns(), target, context)
            } else {
                package.schema.clone()
            };
            let rows = packages::records(&schema.columns, context)?;
            aligned = front.bind_series(served, package, forecast, &rows, None)?;
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
    let rows = forecast.query_rows(&schema, package.id.as_deref(), queries)?;
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
    known.update(b"package-forecast\0");
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
    let columns: Vec<usize> = package
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
                    package
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
        package.id.as_deref(),
        queried.positions.len(),
    );
    Ok(prediction_result(
        &served.resolved,
        &package.task,
        source,
        version,
        bound.rows,
        predictions,
    ))
}

/// A bound request context's fingerprint: its model and its rows as the
/// binder encoded them, which is all its table holds, so contexts that
/// encode alike share one table and one fit.
fn bound_fingerprint(served: &ServedModel, bound: &package::Context) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"package\0");
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
