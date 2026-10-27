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

use gf_core::node::Targets;
use gf_core::request::{self, ErrorDetail, Inference, Label, Limits, Options, RequestError};
use serde::Deserialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::as_of::{self, AsOf};
use crate::auth::digest;
use crate::budget::{Charge, RequestBudget};
use crate::engine::{self, Bound, Engine};
use crate::error::ServerError;
use crate::name::TableName;
use crate::predictors::{self, ContextMode, ForecastPredictor, ServedPredictor};
use crate::progress::Compiling;
use crate::quantiles::quantile_values;
use crate::revisions::{ContextSource, Published, Record, Revisions, SourceKind};
use crate::routes::Routes;
use crate::setup::random_hex;
use crate::shard::{Asked, Command, Inferred};
use crate::upstream::Upstream;
use arrow_array::RecordBatch;
use gf_core::predictor;

mod adhoc;
mod incoming;
mod keys;
mod predictions;

use adhoc::adhoc_predict_with;
pub use incoming::{ambiguous, internal, owns};
use incoming::{
    check_endpoint, check_headers, check_origin, environment, names_match, parse_envelope, resolve,
};
pub use keys::{Introspect, KeyStore, Principal, Static};
use predictions::dispatch;
pub(crate) use predictions::snapshot_rows;

/// The API version every envelope carries.
pub const VERSION: &str = "gf/v1";
/// The most bytes a request body may hold.
pub const MAX_BYTES: usize = 1_048_576;
/// The deepest and largest JSON a body may be, as the hosted API bounds it.
const MAX_DEPTH: usize = 16;
const MAX_NODES: usize = 65_536;
const MAX_INTEGER: i64 = (1 << 53) - 1;
/// The query parameter that names a hosted predictor's environment.
const ENVIRONMENT_PARAMETER: &str = "env";
/// The environment a hosted predictor's prediction scores when it names none.
const DEFAULT_ENVIRONMENT: &str = "production";
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

    /// Carries the finer reason behind the code, as a service relayed gave it.
    #[must_use]
    pub fn reason(mut self, reason: impl Into<String>) -> Self {
        self.detail_mut().reason = Some(reason.into());
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
                ("reason", detail.reason.as_ref().map(|value| json!(value))),
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
        let (status, code, message) = error.http();
        if matches!(error, ServerError::Internal(_)) {
            tracing::error!("inference failed: {error}");
        }
        Self::new(code, message, status)
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
    /// The shape buckets compiled at startup, by the model's resolved id,
    /// which `GET models` reports as each model's `warm_shapes`.
    warm: OnceLock<Vec<(String, Value)>>,
}

/// A forecast predictor's context: its series aligned once, and checked as
/// a forecast request of every series.
struct SeriesBound {
    context: predictors::ForecastContext,
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
                shapes.push((
                    served.resolved.clone(),
                    json!({
                        "context_rows": context_rows,
                        "inputs": WARM_INPUTS,
                        "queries": queries,
                    }),
                ));
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

    /// What this server serves, predictors by name and catalog models by
    /// canonical id.
    #[must_use]
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

/// The catalog model a spelling names, as the catalog's references resolve
/// it: a reference with its pin and variant, a canonical id, a short name,
/// or an old alias of one.
fn catalog_entry(spelling: &str) -> Option<&'static catalog::Entry> {
    let catalog = catalog::catalog();
    catalog::parse(spelling)
        .ok()
        .map(|model| model.entry)
        .or_else(|| catalog.entry(spelling))
        .or_else(|| catalog.named(spelling))
        .or_else(|| catalog.named(&format!("{}/{spelling}", catalog::ORG)))
        .or_else(|| {
            catalog
                .aliases
                .get(spelling)
                .and_then(|canonical| catalog.named(canonical))
        })
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
    if ambiguous(request.uri().path()) {
        return failure(None, &ApiError::validation("invalid request target"));
    }
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
        // The budget is charged by the length a body states before it is read,
        // so a body that states none, as HTTP/2 allows, is refused.
        None if parts.method == Method::POST => {
            return Err(ApiError::new(
                "LengthRequired",
                "a request body states its Content-Length",
                411,
            ));
        }
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
    let (project, action, resource) = resolve(&parts.method, parts.uri.path())?;
    let environment = environment(parts.uri.query(), &action)?;
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
    let result = dispatch(
        front,
        &project,
        action,
        &resource,
        environment.as_deref(),
        payload,
        charge,
    )
    .await?;
    let mut body = envelope(scope.as_ref().expect("set above"));
    body.insert("result".into(), result);
    Ok(respond(200, &Value::Object(body)))
}

impl Front {
    /// The served model a request names: by its resource name, or by any
    /// spelling the catalog knows its model by, its short name, canonical
    /// id or an old alias, whichever of them the server's own entry names.
    fn served(&self, model: &str) -> Result<&ServedModel, ApiError> {
        let wanted = catalog_entry(model).map(|entry| entry.id.as_str());
        self.models
            .iter()
            .filter(|served| served.predictor.is_none())
            .find(|served| {
                served.name == model
                    || wanted.is_some_and(|wanted| {
                        [served.reference.as_str(), served.name.as_str()]
                            .into_iter()
                            .filter_map(catalog_entry)
                            .any(|entry| entry.id == wanted)
                    })
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
        let warmed: Vec<&Value> = self
            .warm
            .get()
            .into_iter()
            .flatten()
            .filter(|(model, _)| *model == served.resolved)
            .map(|(_, shape)| shape)
            .collect();
        description["warm_shapes"] = json!(warmed);
        description
    }

    /// A predictor's `GET predictors/{name}` entry, as the API's `Predictor`
    /// resource has it: its model, task, target and inputs, a forecaster's
    /// time and series, its context as the lock pins it, and its lock's
    /// digest. A predictor served from its lock has no environments.
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
        let mut context = json!({"kind": predictor.context.kind()});
        if let Some(rows) = rows {
            context["rows"] = json!(rows);
        }
        if let Some(digest) = &predictor.context_digest {
            context["digest"] = json!(digest);
        }
        let mut described = json!({
            "name": served.name,
            "environments": [],
            "model": served.resolved,
            "task": predictor.task,
            "inputs": predictor.described_inputs(),
            "context": context,
            "lock_digest": predictor.lock_digest,
        });
        if let Some(target) = &predictor.target {
            described["target"] = json!(target);
        }
        if let Some(id) = &predictor.id {
            described["id"] = json!(id);
        }
        if let Some(forecast) = &predictor.forecast {
            described["time"] = json!(forecast.time);
            described["series"] = json!(forecast.series);
        }
        described
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

    /// The engine inference runs through, once the shards are up.
    fn engine(&self) -> Result<&Engine, ApiError> {
        self.engine.get().ok_or_else(not_ready)
    }

    /// Aligns a forecast predictor's series and checks the forecast request
    /// of every one of them, as `ForecastContext.build` does in process.
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
        let zeros: Map<String, Value> = forecast
            .future_inputs
            .iter()
            .map(|name| (name.clone(), json!(vec![0.0; forecast.horizon])))
            .collect();
        let every = (0..context.identities.len())
            .map(|series| (series, zeros.clone()))
            .collect();
        let bytes = serde_json::to_vec(&context.request(&every)).map_err(|_| failed())?;
        let Inference::Forecast(_) = request::parse(&bytes, &self.predictor_options())? else {
            return Err(failed());
        };
        Ok(SeriesBound {
            context,
            rows: rows.num_rows(),
        })
    }

    /// How a predictor's own forecast requests parse: under its lock's
    /// bounds, not a Models API request's.
    fn predictor_options(&self) -> Options<'_> {
        Options {
            surface: self.surface,
            limits: Limits::Predictor,
            levels: None,
        }
    }
}
