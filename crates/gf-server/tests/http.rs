// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The Models API front: routes, envelopes, headers, errors and keys, with
//! inference through a shard driven by a fake worker.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gf_core::cache::Eviction;
use gf_core::catalog::Surface;
use gf_core::node::Scheduling;
use gf_server::http::{Front, KeyStore, Principal, ServedModel, Static, router};
use gf_server::metrics::Metrics;
use gf_server::protocol::{
    CompileCache, Frame, ModelProfile, PredictionsHeader, WorkerReply, WorkerRequest, f32_bytes,
};
use gf_server::routes::Routes;
use gf_server::service::RequestBudget;
use gf_server::shard::{EphemeralLimits, Shard, ShardSettings};
use gf_server::wal::EncryptionKey;
use gf_server::worker::WorkerConnection;
use serde_json::{Value, json};
use tokio::net::UnixStream;
use tower::ServiceExt;

const KEY: &str = "gf-serve-test-key";

/// Answers profiles as a ten-class classifier, and every query with 0.25.
fn fake_worker(mut stream: UnixStream) {
    tokio::spawn(async move {
        while let Ok(frame) = Frame::read_from(&mut stream).await {
            let reply = if frame.kind() == "step" {
                let mut reply = Frame::header(Value::Null);
                let results: Vec<PredictionsHeader> = frame.header["queries"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|batch| {
                        let rows = batch["count"].as_u64().unwrap_or(0) as usize;
                        PredictionsHeader {
                            request: batch["request"].as_u64().unwrap_or(0),
                            rows,
                            columns: 10,
                            buffer: reply.push(f32_bytes(&vec![0.25; rows * 10])),
                            horizon: None,
                        }
                    })
                    .collect();
                reply.header = WorkerReply::Predictions {
                    results,
                    failures: Vec::new(),
                    deferred: Vec::new(),
                    seconds: 0.001,
                    stages: BTreeMap::new(),
                    parts: BTreeMap::new(),
                    memory: BTreeMap::new(),
                    compile_cache: CompileCache::default(),
                }
                .frame()
                .header;
                reply
            } else {
                match WorkerRequest::of(&frame).expect("the shard sends requests") {
                    WorkerRequest::Profile { .. } => WorkerReply::Profile(ModelProfile {
                        bytes_per_row: 64,
                        representation_bytes: 16,
                        regression: false,
                        class_slots: 10,
                        outputs: 10,
                        state_bytes: 32,
                        ..ModelProfile::default()
                    })
                    .frame(),
                    _ => WorkerReply::Ok.frame(),
                }
            };
            if reply.write_to(&mut stream).await.is_err() {
                break;
            }
        }
    });
}

async fn shard() -> tokio::sync::mpsc::Sender<gf_server::shard::Command> {
    let (ours, theirs) = UnixStream::pair().unwrap();
    fake_worker(theirs);
    let connection = WorkerConnection::fake(ours, vec!["tiny".to_string()]).unwrap();
    let settings = ShardSettings {
        device_bytes: 64 << 20,
        host_bytes: 64 << 20,
        row_memory_bytes: 64 << 20,
        disk_bytes: 0,
        slot_bytes: 1 << 20,
        bucket_minimum: 8,
        step_bytes: None,
        scheduling: Scheduling::Turns,
        eviction: Eviction::Gdsf,
        step_timeout: Duration::from_secs(60),
        queue_bytes: 1 << 20,
        snapshot_bytes: 1 << 30,
        timers: false,
        ephemeral: EphemeralLimits {
            tables: 4,
            bytes: 1 << 20,
            idle: Duration::from_secs(300),
        },
    };
    let directory = std::env::temp_dir().join(format!("gf-http-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let key = EncryptionKey::from_hex(&"ab".repeat(32)).unwrap();
    let mut shard = Shard::new(
        connection,
        settings,
        key,
        directory,
        Arc::new(Metrics::default()),
    )
    .unwrap();
    shard.allocate().await.unwrap();
    shard.spawn().unwrap()
}

fn served() -> ServedModel {
    ServedModel {
        name: "tabicl-v2-classifier".to_string(),
        reference: "tabicl-v2-classifier".to_string(),
        worker: "tiny".to_string(),
        resolved: "general-forecast/tabicl-v2-classifier@5fb6a4c".to_string(),
        outputs: vec!["label".to_string(), "probabilities".to_string()],
        description: json!({"name": "tabicl-v2-classifier", "artifact": {"license": "BSD-3-Clause"}}),
        license_source: None,
        predictor: None,
    }
}

async fn front(keys: Box<dyn KeyStore>, ready: bool) -> Arc<Front> {
    let front = Arc::new(Front::new(
        vec![served()],
        keys,
        Surface::SelfHosted,
        true,
        RequestBudget::new(1 << 30),
        Duration::from_secs(30),
        Arc::new(AtomicBool::new(false)),
    ));
    if ready {
        front
            .start(vec![shard().await], Arc::new(Routes::new(1, 4)))
            .await
            .unwrap();
    }
    front
}

fn envelope(resource: &str, payload: &Value) -> Value {
    json!({
        "api_version": "gf/v1",
        "project": "local",
        "request_id": "r-test",
        "resource": {"kind": if resource == "ad-hoc" { "predictor" } else { "model" }, "name": resource},
        "payload": payload,
    })
}

fn adhoc_payload() -> Value {
    json!({
        "model": "general-forecast/tabicl-v2-classifier",
        "target": "y",
        "inputs": ["x"],
        "context": [{"x": 0.0, "y": false}, {"x": 3.0, "y": true}],
        "queries": [{"x": 1.5}],
    })
}

#[allow(dead_code)]
fn infer_payload() -> Value {
    json!({
        "model": "tabicl-classifier",
        "task": "classification",
        "target": "y",
        "features": ["x"],
        "context": [
            {"key": "c1", "features": {"x": 0.0}, "label": false},
            {"key": "c2", "features": {"x": 3.0}, "label": true},
        ],
        "queries": [{"key": "q", "features": {"x": 1.5}}],
        "options": {},
    })
}

async fn call(
    front: &Arc<Front>,
    method: &str,
    path: &str,
    body: Option<&Value>,
    headers: &[(&str, &str)],
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("host", "127.0.0.1:8000");
    if body.is_some() {
        request = request.header("content-type", "application/json");
    }
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let bytes = body.map(|body| body.to_string()).unwrap_or_default();
    let request = request
        .header("content-length", bytes.len())
        .body(Body::from(bytes))
        .unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(30),
        router(front.clone()).oneshot(request),
    )
    .await
    .expect("answered in time")
    .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, headers, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn health_readiness_and_the_hosted_headers() {
    let waiting = front(Box::new(Static::new(None)), false).await;
    let (status, headers, body) = call(&waiting, "GET", "/livez", None, &[]).await;
    assert_eq!((status, body), (StatusCode::OK, json!({"status": "ok"})));
    assert_eq!(headers["cache-control"], "no-store");
    assert_eq!(headers["x-content-type-options"], "nosniff");
    assert_eq!(headers["content-type"], "application/json; charset=utf-8");
    // Health and readiness both say the models are not ready until the
    // shards are up and their programs compiled.
    for probe in ["/healthz", "/readyz"] {
        let (status, _, body) = call(&waiting, "GET", probe, None, &[]).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{probe}");
        assert_eq!(body["error"]["code"], "ModelNotReady", "{probe}");
        assert!(body["error"]["message"].is_string(), "{probe}");
    }
    let (status, _, body) = call(&waiting, "GET", "/v1/projects/local/models", None, &[]).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "ModelNotReady");
    assert_eq!(body["project"], "local");
    let ready = front(Box::new(Static::new(None)), true).await;
    for probe in ["/healthz", "/readyz"] {
        let (status, _, body) = call(&ready, "GET", probe, None, &[]).await;
        assert_eq!(
            (status, body),
            (StatusCode::OK, json!({"status": "ok"})),
            "{probe}"
        );
    }
    let (status, _, body) = call(&ready, "GET", "/livez", None, &[]).await;
    assert_eq!((status, body), (StatusCode::OK, json!({"status": "ok"})));
}

#[tokio::test]
async fn models_are_listed_and_described_in_envelopes() {
    let front = front(Box::new(Static::new(None)), true).await;
    let (status, _, body) = call(&front, "GET", "/v1/projects/team-a/models", None, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["api_version"], "gf/v1");
    assert_eq!(body["project"], "team-a");
    assert_eq!(body["resource"], json!({"kind": "model", "name": "models"}));
    let described = &body["result"]["models"][0];
    assert_eq!(described["name"], "tabicl-v2-classifier");
    assert_eq!(described["model"], served().resolved);
    assert_eq!(described["artifact"], served().description["artifact"]);
    let policy = &gf_core::catalog::catalog().policy;
    assert_eq!(
        described["limits"],
        json!({"context_rows": policy.context_rows, "queries": policy.query_rows, "inputs": policy.features})
    );
    assert_eq!(described["warm_shapes"], json!([]));
    assert_eq!(body["result"]["models"].as_array().unwrap().len(), 1);
    assert!(body["request_id"].as_str().unwrap().starts_with("r-"));
    let path = "/v1/projects/local/models/tabicl-v2-classifier";
    let (status, _, body) = call(&front, "GET", path, None, &[("x-gf-request-id", "r-1")]).await;
    assert_eq!(
        (status, &body["request_id"]),
        (StatusCode::OK, &json!("r-1"))
    );
    assert_eq!(body["result"]["name"], "tabicl-v2-classifier");
    assert!(body["result"]["limits"].is_object());
    let (status, _, body) = call(&front, "GET", "/v1/projects/local/models/other", None, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        body["error"],
        json!({"code": "NotFound", "message": "model is not served here"})
    );
}

#[tokio::test]
async fn unknown_routes_methods_and_names_get_the_apis_errors() {
    let front = front(Box::new(Static::new(None)), true).await;
    let cases = [
        (
            "GET",
            "/v1/projects/local/nothing",
            404,
            "NotFound",
            "route not found",
        ),
        ("GET", "/elsewhere", 404, "NotFound", "route not found"),
        (
            "DELETE",
            "/v1/projects/local/models",
            405,
            "MethodNotAllowed",
            "only GET and POST are supported",
        ),
        (
            "GET",
            "/v1/projects/Local/models",
            422,
            "ValidationError",
            "invalid name",
        ),
        (
            "GET",
            "/v1/projects/local/models?x=1",
            422,
            "ValidationError",
            "query parameters are not supported",
        ),
    ];
    for (method, path, status, code, message) in cases {
        let (got, _, body) = call(&front, method, path, None, &[]).await;
        assert_eq!(got.as_u16(), status, "{path}");
        assert_eq!(
            body,
            json!({"error": {"code": code, "message": message}}),
            "{path}"
        );
    }
    let (status, _, body) = call(
        &front,
        "GET",
        "/v1/projects/local/models",
        None,
        &[("x-gf-project", "other")],
    )
    .await;
    assert_eq!(status.as_u16(), 422);
    assert_eq!(body["error"]["message"], "project header/path mismatch");
    let (status, _, body) = call(
        &front,
        "GET",
        "/v1/projects/local/models",
        None,
        &[("origin", "http://evil.example")],
    )
    .await;
    assert_eq!(
        (status.as_u16(), &body["error"]["message"]),
        (422, &json!("untrusted Origin"))
    );
}

#[tokio::test]
async fn envelopes_must_match_their_routes() {
    let front = front(Box::new(Static::new(None)), true).await;
    let path = "/v1/projects/local/predict";
    let mut wrong_project = envelope("ad-hoc", &adhoc_payload());
    wrong_project["project"] = json!("other");
    let mut wrong_version = envelope("ad-hoc", &adhoc_payload());
    wrong_version["api_version"] = json!("gf/v0");
    let mut extra = envelope("ad-hoc", &adhoc_payload());
    extra["more"] = json!(1);
    let cases = [
        (wrong_project, "project path/envelope mismatch", false),
        (wrong_version, "unsupported api_version", false),
        (extra, "unknown fields", false),
        (
            envelope("other", &adhoc_payload()),
            "resource route/envelope mismatch",
            true,
        ),
    ];
    for (body, message, enveloped) in cases {
        let (status, _, answer) = call(&front, "POST", path, Some(&body), &[]).await;
        assert_eq!(status.as_u16(), 422, "{message}");
        assert_eq!(answer["error"]["message"], message);
        assert_eq!(
            answer.get("request_id").is_some(),
            enveloped || message != "unsupported api_version" && message != "unknown fields",
            "{message}"
        );
    }
}

#[tokio::test]
async fn ambiguous_and_oversized_bodies_are_refused() {
    let front = front(Box::new(Static::new(None)), true).await;
    let path = "/v1/projects/local/predict";
    let duplicated = r#"{"api_version":"gf/v1","api_version":"gf/v1"}"#;
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("host", "127.0.0.1:8000")
        .header("content-type", "application/json")
        .header("content-length", duplicated.len())
        .body(Body::from(duplicated))
        .unwrap();
    let response = router(front.clone()).oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 422);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["error"]["message"], "duplicate JSON field");
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("host", "127.0.0.1:8000")
        .header("content-type", "application/json")
        .header("content-length", 2_000_000)
        .body(Body::empty())
        .unwrap();
    let response = router(front).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn a_static_key_is_required_once_configured() {
    let front = front(Box::new(Static::new(Some(KEY))), true).await;
    let path = "/v1/projects/local/models";
    let (status, _, body) = call(&front, "GET", path, None, &[]).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["message"], "application bearer required");
    let (status, _, body) = call(
        &front,
        "GET",
        path,
        None,
        &[("authorization", "Bearer wrong")],
    )
    .await;
    assert_eq!(
        (status, &body["error"]["message"]),
        (StatusCode::UNAUTHORIZED, &json!("invalid API key"))
    );
    let bearer = format!("Bearer {KEY}");
    let (status, _, _) = call(&front, "GET", path, None, &[("authorization", &bearer)]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_scoped_static_key_reaches_only_its_project_and_permissions() {
    let principal = Principal {
        project: Some("local".to_string()),
        permissions: Some(vec!["predictors:predict".to_string()]),
    };
    let front = front(Box::new(Static::scoped(Some(KEY), principal)), true).await;
    let bearer = format!("Bearer {KEY}");
    let authorized = [("authorization", bearer.as_str())];
    let infer = envelope("ad-hoc", &adhoc_payload());
    let path = "/v1/projects/local/predict";
    let (status, _, body) = call(&front, "POST", path, Some(&infer), &[]).await;
    assert_eq!(
        (status, &body["error"]["code"]),
        (StatusCode::UNAUTHORIZED, &json!("Unauthorized"))
    );
    let (status, _, _) = call(&front, "POST", path, Some(&infer), &authorized).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, body) = call(
        &front,
        "GET",
        "/v1/projects/local/models",
        None,
        &authorized,
    )
    .await;
    assert_eq!(
        (status, &body["error"]["message"]),
        (
            StatusCode::FORBIDDEN,
            &json!("permission denied for this scope")
        )
    );
    let mut elsewhere = infer.clone();
    elsewhere["project"] = json!("other");
    let (status, _, body) = call(
        &front,
        "POST",
        "/v1/projects/other/predict",
        Some(&elsewhere),
        &authorized,
    )
    .await;
    assert_eq!(
        (status, &body["error"]["code"]),
        (StatusCode::FORBIDDEN, &json!("Forbidden"))
    );
}

#[tokio::test]
async fn an_endpoint_front_answers_only_its_host_and_no_browser() {
    let endpoint = Arc::new(
        Arc::try_unwrap(front(Box::new(Static::new(None)), true).await)
            .ok()
            .expect("one owner")
            .with_hosts(vec!["127.0.0.1:8000".to_string()]),
    );
    let path = "/v1/projects/local/models";
    let (status, _, _) = call(&endpoint, "GET", path, None, &[]).await;
    assert_eq!(status, StatusCode::OK);
    for (headers, message) in [
        (
            vec![("origin", "http://127.0.0.1:8000")],
            "browser origins are not supported",
        ),
        (
            vec![("origin", "null")],
            "browser origins are not supported",
        ),
    ] {
        let (status, _, body) = call(&endpoint, "GET", path, None, &headers).await;
        assert_eq!(
            (status, &body["error"]["message"]),
            (StatusCode::FORBIDDEN, &json!(message))
        );
    }
    let other = Arc::new(
        Arc::try_unwrap(front(Box::new(Static::new(None)), true).await)
            .ok()
            .expect("one owner")
            .with_hosts(vec!["unit--gf-byoc.modal.run".to_string()]),
    );
    let (status, _, body) = call(&other, "GET", path, None, &[]).await;
    assert_eq!(
        (status, &body["error"]["message"]),
        (StatusCode::FORBIDDEN, &json!("unrecognized endpoint Host"))
    );
}

#[tokio::test]
async fn the_ad_hoc_route_binds_flat_rows_and_answers_a_prediction_result() {
    let front = front(Box::new(Static::new(None)), true).await;
    let payload = json!({
        "model": "general-forecast/tabicl-v2-classifier",
        "target": "y",
        "inputs": ["x"],
        "id": "id",
        "context": [{"x": 0.0, "y": false}, {"x": 3.0, "y": true}, {"x": 1.0, "y": null}],
        "queries": [{"id": "a", "x": 1.5}],
    });
    let path = "/v1/projects/local/predict";
    let (status, _, answer) = call(
        &front,
        "POST",
        path,
        Some(&envelope("ad-hoc", &payload)),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let result = &answer["result"];
    assert_eq!(result["model"], served().resolved);
    assert_eq!(result["task"], "classification");
    assert_eq!(result["context"]["rows"], 2);
    assert_eq!(result["context"]["kind"], "request");
    assert!(result["context"]["as_of"].is_number());
    assert!(result.get("deployment").is_none());
    assert_eq!(
        result["predictions"],
        json!([{"id": "a", "label": false, "probabilities": {"false": 0.25, "true": 0.25}}])
    );
    // Inputs default to every column of the first context row but the
    // target and the id.
    let mut implicit = payload.clone();
    implicit.as_object_mut().unwrap().remove("inputs");
    implicit["context"][0]["id"] = json!("c");
    let (status, _, answer) = call(
        &front,
        "POST",
        path,
        Some(&envelope("ad-hoc", &implicit)),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{answer}");
}

#[tokio::test]
async fn the_ad_hoc_route_refuses_bare_model_ids_and_retired_fields_by_name() {
    let front = front(Box::new(Static::new(None)), true).await;
    let path = "/v1/projects/local/predict";
    let good = json!({
        "model": "general-forecast/tabicl-v2-classifier",
        "target": "y",
        "inputs": ["x"],
        "context": [{"x": 0.0, "y": false}, {"x": 3.0, "y": true}],
        "queries": [{"x": 1.5}],
    });
    let mut bare = good.clone();
    bare["model"] = json!("tabicl-v2-classifier");
    let (status, _, body) = call(&front, "POST", path, Some(&envelope("ad-hoc", &bare)), &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "ValidationError");
    assert_eq!(body["error"]["param"], "model");
    assert_eq!(
        body["error"]["hint"],
        "use general-forecast/tabicl-v2-classifier"
    );
    for (field, value) in [
        ("task", json!("classification")),
        ("options", json!({})),
        ("features", json!(["x"])),
        ("horizon", json!(3)),
    ] {
        let mut payload = good.clone();
        payload[field] = value;
        let (status, _, body) = call(
            &front,
            "POST",
            path,
            Some(&envelope("ad-hoc", &payload)),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{field}: {body}");
        assert_eq!(body["error"]["param"], field, "{body}");
    }
    // A text column is named, with the rule.
    let mut text = good.clone();
    text["context"][0]["x"] = json!("low");
    let (status, _, body) = call(&front, "POST", path, Some(&envelope("ad-hoc", &text)), &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(
        body["error"]["message"],
        "column x has text values; this route takes numeric inputs"
    );
    assert_eq!(body["error"]["param"], "inputs.x");
    // Too many queries carry the limit, what was sent, and the model.
    let policy = &gf_core::catalog::catalog().policy;
    let mut many = good.clone();
    many["queries"] = json!(vec![json!({"x": 1.5}); policy.query_rows + 1]);
    let (status, _, body) = call(&front, "POST", path, Some(&envelope("ad-hoc", &many)), &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "CapacityExceeded");
    assert_eq!(body["error"]["limit"], policy.query_rows);
    assert_eq!(body["error"]["actual"], policy.query_rows + 1);
    assert!(body["error"]["hint"].is_string());
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("tabicl-v2-classifier"),
        "{body}"
    );
    // The old route is gone.
    let (status, _, body) = call(
        &front,
        "POST",
        "/v1/projects/local/predictions",
        Some(&envelope("predictions", &good)),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

#[tokio::test]
async fn introspected_keys_are_scoped_to_their_project_and_cached() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Json;
    use axum::routing::post;
    use gf_server::http::Introspect;

    let asked = Arc::new(AtomicUsize::new(0));
    let counter = asked.clone();
    let good = request_digest(KEY);
    let plane = axum::Router::new().route(
        "/internal/v1/keys:introspect",
        post(move |headers: axum::http::HeaderMap, Json(body): Json<Value>| {
            let counter = counter.clone();
            let good = good.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                assert_eq!(headers["authorization"], "Bearer service-token");
                if body["verifier"] == good {
                    Json(json!({"active": true, "project": "team-a", "permissions": ["models:read"], "expires": null}))
                } else {
                    Json(json!({"active": false}))
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, plane).await.unwrap() });
    let keys = Introspect::new(&format!("http://{address}"), "service-token".to_string()).unwrap();
    let front = front(Box::new(keys), true).await;
    let bearer = format!("Bearer {KEY}");
    let auth = [("authorization", bearer.as_str())];
    let (status, _, _) = call(&front, "GET", "/v1/projects/team-a/models", None, &auth).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = call(&front, "GET", "/v1/projects/team-a/models", None, &auth).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        asked.load(Ordering::SeqCst),
        1,
        "the second call is answered from the cache"
    );
    let (status, _, body) = call(&front, "GET", "/v1/projects/team-b/models", None, &auth).await;
    assert_eq!(
        (status, &body["error"]["code"]),
        (StatusCode::FORBIDDEN, &json!("Forbidden"))
    );
    let path = "/v1/projects/team-a/predict";
    let mut body = envelope("ad-hoc", &adhoc_payload());
    body["project"] = json!("team-a");
    let (status, _, _) = call(&front, "POST", path, Some(&body), &auth).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "reading models does not grant prediction"
    );
    let wrong = [("authorization", "Bearer other-key")];
    for _ in 0..2 {
        let (status, _, body) =
            call(&front, "GET", "/v1/projects/team-a/models", None, &wrong).await;
        assert_eq!(
            (status, &body["error"]["message"]),
            (StatusCode::UNAUTHORIZED, &json!("invalid API key"))
        );
    }
    assert_eq!(asked.load(Ordering::SeqCst), 2, "refusals are cached too");
}

fn request_digest(key: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(key.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[tokio::test]
async fn a_refused_service_token_is_unavailable_and_never_cached() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::routing::post;
    use gf_server::http::Introspect;

    let asked = Arc::new(AtomicUsize::new(0));
    let counter = asked.clone();
    let plane = axum::Router::new().route(
        "/internal/v1/keys:introspect",
        post(move || {
            let counter = counter.clone();
            async move {
                if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                    StatusCode::UNAUTHORIZED
                } else {
                    StatusCode::FORBIDDEN
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, plane).await.unwrap() });
    let keys = Introspect::new(&format!("http://{address}"), "rotated-token".to_string()).unwrap();
    let front = front(Box::new(keys), true).await;
    let bearer = format!("Bearer {KEY}");
    let auth = [("authorization", bearer.as_str())];
    for _ in 0..2 {
        let (status, _, body) =
            call(&front, "GET", "/v1/projects/team-a/models", None, &auth).await;
        assert_eq!(
            (status, &body["error"]["code"]),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                &json!("CredentialUnavailable")
            )
        );
    }
    assert_eq!(
        asked.load(Ordering::SeqCst),
        2,
        "a refused service token says nothing about the key, so nothing is cached"
    );
}

/// A locked predictor of `x` predicting `y`, its queries named by `id`.
fn predictor_model(name: &str, context: &str, snapshot: Option<&std::path::Path>) -> ServedModel {
    let predictor = json!({
        "schema": {
            "columns": [
                {"name": "id", "type": "string", "nullable": false, "role": "id", "semantic": null},
                {"name": "x", "type": "float64", "nullable": false, "role": "input", "semantic": null},
                {"name": "y", "type": "boolean", "nullable": false, "role": "target", "semantic": null},
            ],
            "rows": 4,
            "classes": [false, true],
            "frequency": null,
        },
        "task": "classification",
        "target": "y",
        "inputs": ["x"],
        "id": "id",
        "context": context,
        "quantiles": [],
        "context_digest": "sha256:data",
        "lock_digest": format!("sha256:{name}"),
        "snapshot": snapshot,
    });
    ServedModel {
        name: name.to_string(),
        reference: "tabicl-v2-classifier".to_string(),
        worker: "tiny".to_string(),
        resolved: "general-forecast/tabicl-v2-classifier@5fb6a4c".to_string(),
        outputs: vec!["label".to_string(), "probabilities".to_string()],
        description: json!({"name": name, "kind": "predictor"}),
        license_source: None,
        predictor: Some(serde_json::from_value(predictor).unwrap()),
    }
}

/// Four labeled rows as the Arrow IPC stream `gf serve` writes a snapshot as.
fn snapshot_file() -> std::path::PathBuf {
    use arrow_array::{ArrayRef, BooleanArray, Float64Array, RecordBatch, StringArray};
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("id", Arc::new(StringArray::from(vec!["a", "b", "c", "d"]))),
        ("x", Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0]))),
        (
            "y",
            Arc::new(BooleanArray::from(vec![false, false, true, true])),
        ),
    ];
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let path = std::env::temp_dir().join(format!("gf-http-snapshot-{}.arrow", std::process::id()));
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = arrow_ipc::writer::StreamWriter::try_new(file, &batch.schema()).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    path
}

async fn predictor_front(model: ServedModel) -> Arc<Front> {
    let front = Arc::new(Front::new(
        vec![model],
        Box::new(Static::new(None)),
        Surface::SelfHosted,
        true,
        RequestBudget::new(1 << 30),
        Duration::from_secs(30),
        Arc::new(AtomicBool::new(false)),
    ));
    front
        .start(vec![shard().await], Arc::new(Routes::new(1, 4)))
        .await
        .unwrap();
    front
}

/// A declared predictor's body; an empty `context` is left out, as a
/// snapshot predictor's callers leave it.
fn predictor_payload(_model: &str, context: Value, queries: Value) -> Value {
    let mut payload = json!({"queries": queries});
    if context.as_array().is_some_and(|rows| !rows.is_empty()) {
        payload["context"] = context;
    }
    payload
}

fn predictor_envelope(name: &str, payload: &Value) -> Value {
    json!({
        "api_version": "gf/v1",
        "project": "local",
        "request_id": "r-test",
        "resource": {"kind": "predictor", "name": name},
        "payload": payload,
    })
}

#[tokio::test]
async fn snapshot_packages_bind_once_into_a_durable_table() {
    let snapshot = snapshot_file();
    let front = predictor_front(predictor_model("churn", "snapshot", Some(&snapshot))).await;
    assert!(front.ready());
    let path = "/v1/projects/local/predictors/churn/predict";
    let payload = predictor_payload("churn", json!([]), json!([{"id": "q", "x": 1.5}]));
    for _ in 0..2 {
        let (status, _, body) = call(
            &front,
            "POST",
            path,
            Some(&predictor_envelope("churn", &payload)),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["resource"],
            json!({"kind": "predictor", "name": "churn"})
        );
        assert_eq!(
            body["result"]["predictions"],
            json!([{"id": "q", "label": false, "probabilities": {"false": 0.25, "true": 0.25}}])
        );
        let context = &body["result"]["context"];
        assert_eq!(
            (&context["kind"], &context["rows"], &context["digest"]),
            (&json!("snapshot"), &json!(4), &json!("sha256:data"))
        );
    }
    // The predictor is listed and described, with its context's size.
    let (status, _, body) = call(&front, "GET", "/v1/projects/local/predictors", None, &[]).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["predictors"][0]["name"], "churn");
    let (status, _, body) = call(
        &front,
        "GET",
        "/v1/projects/local/predictors/churn",
        None,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["resource"],
        json!({"kind": "predictor", "name": "churn"})
    );
    let described = &body["result"];
    assert_eq!(described["model"], served().resolved);
    assert_eq!(described["target"], "y");
    assert_eq!(described["inputs"], json!(["x"]));
    assert_eq!(
        described["context"],
        json!({"kind": "snapshot", "rows": 4, "digest": "sha256:data"})
    );
    assert_eq!(described["lock_digest"], "sha256:churn");
    let (status, _, body) = call(
        &front,
        "GET",
        "/v1/projects/local/predictors/other",
        None,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "NotFound");
    // Catalog models and predictors are listed apart.
    let (_, _, body) = call(&front, "GET", "/v1/projects/local/models", None, &[]).await;
    assert_eq!(body["result"]["models"], json!([]));
    let mut declared = predictor_payload("churn", json!([]), json!([{"x": 1.5}]));
    declared["target"] = json!("y");
    for (payload, message, param) in [
        (
            predictor_payload("churn", json!([{"x": 1.0, "y": true}]), json!([{"x": 1.5}])),
            "context is declared by the predictor",
            Some("context"),
        ),
        (
            declared,
            "target is declared by the predictor",
            Some("target"),
        ),
        (
            predictor_payload("churn", json!([]), json!([{"x": "high"}])),
            "column \"x\" holds values that are not float64",
            None,
        ),
        (
            predictor_payload("churn", json!([]), json!([])),
            "queries must contain 1..10000 rows",
            None,
        ),
    ] {
        let (status, _, body) = call(
            &front,
            "POST",
            path,
            Some(&predictor_envelope("churn", &payload)),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["error"]["message"], message);
        assert_eq!(
            body["error"]["param"],
            param.map_or(Value::Null, |param| json!(param))
        );
    }
    std::fs::remove_file(snapshot).unwrap();
}

#[tokio::test]
async fn request_packages_bind_each_context_into_an_ephemeral_table() {
    let front = predictor_front(predictor_model("scorer", "request", None)).await;
    let path = "/v1/projects/local/predictors/scorer/predict";
    let context = json!([
        {"x": 0.0, "y": false},
        {"x": 1.0, "y": false},
        {"x": 3.0, "y": true},
        {"x": 4.0, "y": null},
    ]);
    let payload = predictor_payload(
        "scorer",
        context,
        json!([{"id": "q", "x": 2.5}, {"x": 0.5}]),
    );
    let (status, _, body) = call(
        &front,
        "POST",
        path,
        Some(&predictor_envelope("scorer", &payload)),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let result = &body["result"];
    assert_eq!(
        (&result["context"]["kind"], &result["context"]["rows"]),
        (&json!("request"), &json!(3))
    );
    // Not every query has an id, so none is repeated.
    assert_eq!(
        result["predictions"][1],
        json!({"label": false, "probabilities": {"false": 0.25, "true": 0.25}})
    );
    let payload = predictor_payload("scorer", json!([]), json!([{"x": 1.5}]));
    let (status, _, body) = call(
        &front,
        "POST",
        path,
        Some(&predictor_envelope("scorer", &payload)),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body["error"]["message"],
        "context is required: this predictor takes its context with each request"
    );
    assert_eq!(body["error"]["param"], "context");
    let (_, _, body) = call(
        &front,
        "GET",
        "/v1/projects/local/predictors/scorer",
        None,
        &[],
    )
    .await;
    assert_eq!(
        body["result"]["context"],
        json!({"kind": "request", "rows": null, "digest": "sha256:data"})
    );
    let mut payload = predictor_payload(
        "scorer",
        json!([{"x": 0.0, "y": false}]),
        json!([{"x": 1.5}]),
    );
    payload["inputs"] = json!(["z"]);
    let (status, _, body) = call(
        &front,
        "POST",
        path,
        Some(&predictor_envelope("scorer", &payload)),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body["error"]["message"],
        "inputs is declared by the predictor"
    );
    assert_eq!(body["error"]["param"], "inputs");
}

/// A control plane that publishes `scorer`'s revisions: 1 a snapshot
/// predictor, 2 a Python predictor, 3 a request predictor. `prod` serves whichever
/// revision `serving` holds.
struct Plane {
    serving: std::sync::atomic::AtomicU64,
    predictors: std::sync::atomic::AtomicUsize,
    records: std::sync::Mutex<Vec<Value>>,
}

fn published(revision: u64) -> Value {
    let context = if revision == 3 { "request" } else { "snapshot" };
    let served = json!({
        "name": "scorer",
        "reference": "tabicl-v2-classifier",
        "worker": "tabicl-v2-classifier",
        "resolved": "general-forecast/tabicl-v2-classifier@5fb6a4c",
        "outputs": ["label", "probabilities"],
        "license_source": null,
        "description": {"name": "scorer", "kind": "predictor"},
        "predictor": {
            "schema": {
                "columns": [
                    {"name": "id", "type": "string", "nullable": false, "role": "id", "semantic": null},
                    {"name": "x", "type": "float64", "nullable": false, "role": "input", "semantic": null},
                    {"name": "y", "type": "boolean", "nullable": false, "role": "target", "semantic": null},
                ],
                "rows": 4,
                "classes": [false, true],
                "frequency": null,
            },
            "task": "classification",
            "target": "y",
            "inputs": ["x"],
            "id": "id",
            "context": context,
            "quantiles": [],
            "context_digest": "sha256:data",
            "lock_digest": format!("sha256:scorer-{revision}"),
            "snapshot": null,
            "forecast": null,
        },
    });
    if revision == 3 {
        return json!({"kind": "native", "served": served, "context": null, "rows": null});
    }
    json!({
        "kind": "native",
        "served": served,
        "context": {"source": "snapshot", "version": "sha256:data", "rows": 4, "as_of": null},
        "rows": [
            {"id": "a", "x": 0.0, "y": false},
            {"id": "b", "x": 1.0, "y": false},
            {"id": "c", "x": 2.0, "y": true},
            {"id": "d", "x": 3.0, "y": true},
        ],
    })
}

const SCORER: &str = "gf-scorer-key";
const INFERRER: &str = "gf-inferrer-key";

async fn plane(state: Arc<Plane>) -> String {
    use std::sync::atomic::Ordering;

    let handler = move |request: axum::extract::Request| {
        let state = state.clone();
        async move {
            assert_eq!(request.headers()["authorization"], "Bearer service-token");
            let path = request.uri().path().to_string();
            let body = axum::body::to_bytes(request.into_body(), 1 << 20)
                .await
                .unwrap();
            let base = "/internal/v1/projects/team-a/predictors/scorer";
            let answer = |status: u16, value: Value| {
                axum::response::Response::builder()
                    .status(status)
                    .header("content-type", "application/json")
                    .body(Body::from(value.to_string()))
                    .unwrap()
            };
            if path == "/internal/v1/keys:introspect" {
                let body: Value = serde_json::from_slice(&body).unwrap();
                let permissions = if body["verifier"] == request_digest(SCORER) {
                    json!(["predictors:predict"])
                } else if body["verifier"] == request_digest(INFERRER) {
                    json!(["models:read"])
                } else {
                    return answer(200, json!({"active": false}));
                };
                return answer(
                    200,
                    json!({"active": true, "project": "team-a", "permissions": permissions, "expires": null}),
                );
            }
            if path == "/internal/v1/predictions:record" {
                let body: Value = serde_json::from_slice(&body).unwrap();
                state
                    .records
                    .lock()
                    .unwrap()
                    .extend(body["records"].as_array().unwrap().iter().cloned());
                return answer(200, json!({}));
            }
            if path == format!("{base}/environments/production:revision") {
                let revision = state.serving.load(Ordering::SeqCst);
                return answer(200, json!({"revision": revision, "context_version": null}));
            }
            if path.starts_with(&format!("{base}/environments/")) {
                return answer(
                    404,
                    json!({"error": {"code": "NotFound", "message": "environment not found"}}),
                );
            }
            if let Some(revision) = path
                .strip_prefix(&format!("{base}/revisions/"))
                .and_then(|tail| tail.strip_suffix(":lock"))
            {
                state.predictors.fetch_add(1, Ordering::SeqCst);
                return answer(200, published(revision.parse().unwrap()));
            }
            answer(
                404,
                json!({"error": {"code": "NotFound", "message": "route not found"}}),
            )
        }
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, axum::Router::new().fallback(handler))
            .await
            .unwrap();
    });
    format!("http://{address}")
}

async fn revision_front(origin: &str, bytes: usize) -> Arc<Front> {
    use gf_server::http::Introspect;
    use gf_server::revisions::{Limits, Revisions};

    let keys = Introspect::new(origin, "service-token".to_string()).unwrap();
    let revisions = Revisions::new(
        origin,
        "service-token".to_string(),
        Limits {
            bytes,
            idle: Duration::from_secs(600),
            resolve: Duration::from_millis(200),
            timeout: Duration::from_secs(5),
        },
    )
    .unwrap();
    let front = Arc::new(
        Front::new(
            vec![served()],
            Box::new(keys),
            Surface::SelfHosted,
            true,
            RequestBudget::new(1 << 30),
            Duration::from_secs(30),
            Arc::new(AtomicBool::new(false)),
        )
        .with_revisions(revisions),
    );
    front
        .start(vec![shard().await], Arc::new(Routes::new(1, 4)))
        .await
        .unwrap();
    front
}

fn served_envelope(payload: &Value) -> Value {
    json!({
        "api_version": "gf/v1",
        "project": "team-a",
        "request_id": "r-test",
        "resource": {"kind": "predictor", "name": "scorer"},
        "payload": payload,
    })
}

async fn score(
    front: &Arc<Front>,
    _route: &str,
    key: &str,
    payload: &Value,
) -> (StatusCode, Value) {
    let bearer = format!("Bearer {key}");
    let (status, _, body) = call(
        front,
        "POST",
        "/v1/projects/team-a/predictors/scorer/predict",
        Some(&served_envelope(payload)),
        &[("authorization", bearer.as_str())],
    )
    .await;
    (status, body)
}

#[tokio::test]
async fn revisions_load_on_demand_evict_and_reload_when_an_environment_moves() {
    use std::sync::atomic::Ordering;

    let state = Arc::new(Plane {
        serving: 1.into(),
        predictors: 0.into(),
        records: std::sync::Mutex::new(Vec::new()),
    });
    let origin = plane(state.clone()).await;
    // Room for either predictor, never both.
    let sizes = [1, 3].map(|revision| published(revision).to_string().len());
    let front = revision_front(&origin, sizes[0].max(sizes[1]) + 1).await;
    let held = || front.revisions().unwrap().held();

    let queries = json!({"queries": [{"id": "q", "x": 2.5}]});
    for _ in 0..2 {
        let (status, body) = score(&front, "environments/prod", SCORER, &queries).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let result = &body["result"];
        assert_eq!(
            (
                &result["predictor"],
                &result["revision"],
                &result["environment"]
            ),
            (&json!("scorer"), &json!(1), &json!("production"))
        );
        assert_eq!(
            result["context"],
            json!({"source": "snapshot", "version": "sha256:data", "rows": 4, "as_of": null})
        );
        assert_eq!(
            result["model"],
            "general-forecast/tabicl-v2-classifier@5fb6a4c"
        );
        assert_eq!(result["predictions"][0]["id"], "q");
    }
    assert_eq!(state.predictors.load(Ordering::SeqCst), 1, "loaded once");
    assert_eq!(held().0, 1);

    // A promote is seen once the environment's answer expires; revision 1
    // makes room for revision 3.
    state.serving.store(3, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(250)).await;
    let request = json!({
        "queries": [{"x": 2.5}],
        "context": [{"x": 0.0, "y": false}, {"x": 3.0, "y": true}],
    });
    let (status, body) = score(&front, "environments/prod", SCORER, &request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        (
            &body["result"]["revision"],
            &body["result"]["context"]["source"]
        ),
        (&json!(3), &json!("request"))
    );
    assert_eq!(state.predictors.load(Ordering::SeqCst), 2);
    assert_eq!(held().0, 1, "revision 1 was evicted for revision 3");

    // The rollback reloads the evicted revision.
    state.serving.store(1, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(250)).await;
    let (status, body) = score(&front, "environments/prod", SCORER, &queries).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["revision"], 1);
    assert_eq!(state.predictors.load(Ordering::SeqCst), 3);

    // The route names the predictor; the production environment picks the revision.
    let (status, body) = score(&front, "environments/prod", SCORER, &queries).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        (&body["result"]["revision"], &body["result"]["environment"]),
        (&json!(1), &json!("production"))
    );

    for _ in 0..100 {
        if state.records.lock().unwrap().len() >= 5 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let records = state.records.lock().unwrap().clone();
    assert_eq!(records.len(), 5, "{records:?}");
    assert_eq!(
        (
            &records[2]["environment"],
            &records[2]["revision"],
            &records[2]["ok"]
        ),
        (&json!("production"), &json!(3), &json!(true))
    );
    assert_eq!(records[4]["environment"], json!("production"));
}

#[tokio::test]
async fn revision_predictions_are_checked_against_the_lock_and_the_key() {
    let state = Arc::new(Plane {
        serving: 1.into(),
        predictors: 0.into(),
        records: std::sync::Mutex::new(Vec::new()),
    });
    let origin = plane(state.clone()).await;
    let front = revision_front(&origin, 1 << 20).await;
    let queries = json!({"queries": [{"x": 2.5}]});

    let (status, body) = score(&front, "environments/prod", INFERRER, &queries).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
    assert_eq!(body["error"]["code"], "Forbidden");
    let (status, _) = score(&front, "environments/prod", "gf-unknown", &queries).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let bearer = format!("Bearer {SCORER}");
    let mut elsewhere = served_envelope(&queries);
    elsewhere["project"] = json!("team-b");
    let (status, _, _) = call(
        &front,
        "POST",
        "/v1/projects/team-b/predictors/scorer/predict",
        Some(&elsewhere),
        &[("authorization", bearer.as_str())],
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(
        state.predictors.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "nothing is loaded for a refused key"
    );

    let mut model = served_envelope(&queries);
    model["resource"] = json!({"kind": "model", "name": "scorer"});
    let (status, _, body) = call(
        &front,
        "POST",
        "/v1/projects/team-a/predictors/scorer/predict",
        Some(&model),
        &[("authorization", bearer.as_str())],
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["message"], "resource route/envelope mismatch");

    for (payload, message) in [
        (
            json!({"queries": [{"x": 1.0}], "features": ["z"]}),
            "features is declared by the predictor",
        ),
        (
            json!({"queries": [{"x": 1.0}], "inputs": ["z"]}),
            "inputs is declared by the predictor",
        ),
        (
            json!({"queries": [{"x": 1.0}], "target": "z"}),
            "target is declared by the predictor",
        ),
        (
            json!({"queries": [{"x": 1.0}], "options": {"n": 1}}),
            "options is declared by the predictor",
        ),
        (json!({"queries": []}), "queries must contain 1..10000 rows"),
        (
            json!({"queries": [{"x": 1.0}], "context": []}),
            "context is declared by the predictor",
        ),
        (
            json!({"queries": [{"x": 1.0}], "extra": 1}),
            "unknown fields",
        ),
    ] {
        let (status, body) = score(&front, "environments/prod", SCORER, &payload).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["error"]["message"], message);
    }
}

#[tokio::test]
async fn a_front_without_revisions_refuses_deployment_predictions() {
    let front = front(Box::new(Static::new(None)), true).await;
    let (status, _, body) = call(
        &front,
        "POST",
        "/v1/projects/team-a/predictors/scorer/predict",
        Some(&served_envelope(&json!({"queries": [{"x": 1.0}]}))),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "NotFound");
}
