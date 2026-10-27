// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The front door: every route the front does not own streams to the
//! upstream, and none of its own ever does.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::Response;
use futures::StreamExt;
use gf_core::catalog::Surface;
use gf_server::budget::RequestBudget;
use gf_server::http::{Front, Static, dotted, internal, owns, router};
use gf_server::upstream::Upstream;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Notify;

#[derive(Default)]
struct Origin {
    hits: AtomicUsize,
    release: Notify,
    down: AtomicBool,
}

/// Echoes each request, with a redirect, a streamed answer, a slow answer
/// and an upload counter on their own paths.
async fn origin(State(origin): State<Arc<Origin>>, request: Request) -> Response {
    origin.hits.fetch_add(1, Ordering::SeqCst);
    let (parts, body) = request.into_parts();
    let path = parts.uri.path().to_owned();
    match path.as_str() {
        "/readyz" if origin.down.load(Ordering::SeqCst) => Response::builder()
            .status(503)
            .body(Body::from("{}"))
            .unwrap(),
        "/auth/redirect" => Response::builder()
            .status(302)
            .header(header::LOCATION, "/auth/callback?code=abc&state=xyz")
            .header(header::SET_COOKIE, "gf_state=xyz; HttpOnly; Path=/")
            .header(header::SET_COOKIE, "gf_flow=1; Secure; Path=/auth")
            .body(Body::empty())
            .unwrap(),
        "/stream" => {
            let first = futures::stream::once(async { Ok::<_, std::io::Error>("first;") });
            let rest = futures::stream::once(async move {
                origin.release.notified().await;
                Ok("rest")
            });
            Response::new(Body::from_stream(first.chain(rest)))
        }
        "/slow" => {
            tokio::time::sleep(Duration::from_secs(5)).await;
            Response::new(Body::from("late"))
        }
        _ => {
            let mut seen = 0usize;
            let mut stream = body.into_data_stream();
            while let Some(chunk) = stream.next().await {
                seen += chunk.unwrap().len();
            }
            let headers: serde_json::Map<String, Value> = parts
                .headers
                .iter()
                .map(|(name, value)| (name.to_string(), json!(value.to_str().unwrap())))
                .collect();
            let echo = json!({
                "method": parts.method.as_str(),
                "target": parts.uri.path_and_query().map(|t| t.as_str()),
                "headers": headers,
                "body": seen,
            });
            Response::builder()
                .status(if path == "/missing" { 404 } else { 200 })
                .header("x-upstream", "1")
                .header(header::CONNECTION, "x-hop")
                .header("x-hop", "1")
                .body(Body::from(echo.to_string()))
                .unwrap()
        }
    }
}

async fn serve(app: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    address
}

/// A front with a static key, before its models are ready, in front of `upstream`.
async fn front_door(upstream: &str, timeout: Duration, max_body: u64) -> String {
    let front = Front::new(
        Vec::new(),
        Box::new(Static::new(Some("k"))),
        Surface::SelfHosted,
        true,
        RequestBudget::new(1 << 30),
        Duration::from_secs(30),
        Arc::new(AtomicBool::new(false)),
    )
    .with_upstream(Upstream::new(upstream, timeout, max_body, Vec::new()).unwrap());
    format!("http://{}", serve(router(Arc::new(front))).await)
}

async fn origin_and_front() -> (Arc<Origin>, String) {
    let state = Arc::new(Origin::default());
    let address = serve(Router::new().fallback(origin).with_state(state.clone())).await;
    let front = front_door(&format!("http://{address}"), Duration::from_secs(1), 1024).await;
    (state, front)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap()
}

#[test]
fn the_fronts_routes_are_owned_however_they_are_spelled() {
    for (method, path) in [
        (Method::GET, "/livez"),
        (Method::GET, "/readyz"),
        (Method::GET, "/v1/projects/p/models"),
        (Method::GET, "/v1/projects/p/models/"),
        (Method::GET, "//v1//projects/p//models"),
        (Method::GET, "/V1/Projects/p/Models/m"),
        (Method::POST, "/v1/projects/p/predict"),
        (Method::POST, "/v1/projects/p/predictors/x/predict"),
        (Method::POST, "/v1/projects/p/Predictors/x/PREDICT"),
        (Method::GET, "/v1/projects/p/%6dodels"),
        (Method::POST, "/v1/proj%65cts/p/predict"),
    ] {
        assert!(owns(&method, path), "{method} {path}");
    }
    for (method, path) in [
        (Method::GET, "/"),
        (Method::GET, "/console"),
        (Method::GET, "/auth/callback"),
        (Method::GET, "/v1/projects"),
        (Method::GET, "/v1/projects/p"),
        (Method::POST, "/v1/projects/p/keys"),
        (Method::GET, "/v1/projects/p/contexts/c"),
        (Method::GET, "/v1/projects/p/environments/e"),
        (Method::GET, "/v1/projects/p/predictors"),
        (Method::GET, "/v1/projects/p/predictors/x"),
        (Method::POST, "/v1/projects/p/predictors"),
        (Method::DELETE, "/v1/projects/p/predictors/x"),
        (Method::GET, "/v1/projects/p/predictors/x/revisions"),
        (
            Method::POST,
            "/v1/projects/p/predictors/x/environments/prod/promote",
        ),
        (Method::GET, "/v1/projects/p/predictors/x/metrics"),
        (Method::GET, "/v1/projects/p/predictors/x/predict"),
        (Method::POST, "/v1/projects/p/models"),
        (Method::DELETE, "/v1/projects/p/models/m"),
    ] {
        assert!(!owns(&method, path), "{method} {path}");
    }
}

#[test]
fn the_control_planes_internal_routes_are_recognized_however_they_are_spelled() {
    for path in [
        "/internal/v1/keys:introspect",
        "//internal/v1/predictions:record",
        "/INTERNAL/v1/projects/p/predictors/x/revisions/1:lock",
        "/%69nternal/v1/keys:introspect",
    ] {
        assert!(internal(path), "{path}");
    }
    for path in ["/", "/v1/internal", "/internals", "/console/internal"] {
        assert!(!internal(path), "{path}");
    }
}

#[tokio::test]
async fn hosted_predictor_routes_go_upstream_and_internal_ones_nowhere() {
    let (origin, front) = origin_and_front().await;
    // Every step of a predictor's life the clients take through the front,
    // as `gf push` and the predictor commands call them.
    let base = "/v1/projects/p/predictors";
    for (method, path) in [
        ("POST", base.to_string()),
        ("GET", base.to_string()),
        ("GET", format!("{base}/churn")),
        ("POST", format!("{base}/churn/revisions")),
        ("GET", format!("{base}/churn/revisions")),
        ("GET", format!("{base}/churn/revisions/3")),
        ("GET", format!("{base}/churn/environments")),
        ("GET", format!("{base}/churn/environments/production")),
        (
            "POST",
            format!("{base}/churn/environments/production/promote"),
        ),
        (
            "POST",
            format!("{base}/churn/environments/production/rollback"),
        ),
        (
            "GET",
            format!("{base}/churn/environments/production/metrics"),
        ),
        ("GET", format!("{base}/churn/environments/production/logs")),
        ("POST", format!("{base}/churn/delete")),
        ("DELETE", format!("{base}/churn")),
    ] {
        let response = client()
            .request(method.parse().unwrap(), format!("{front}{path}"))
            .header("content-type", "application/json")
            .body("{}")
            .send()
            .await
            .unwrap();
        assert!(
            response.headers().contains_key("x-upstream"),
            "{method} {path}"
        );
        let echo: Value = response.json().await.unwrap();
        assert_eq!(
            (&echo["method"], &echo["target"]),
            (&json!(method), &json!(path)),
            "forwarded as sent"
        );
    }
    let forwarded = origin.hits.load(Ordering::SeqCst);
    for path in [
        "/internal/v1/keys:introspect",
        "/%69nternal/v1/predictions:record",
    ] {
        let response = client()
            .post(format!("{front}{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "NotFound");
    }
    assert_eq!(origin.hits.load(Ordering::SeqCst), forwarded);
}

#[tokio::test]
async fn unowned_routes_pass_through_unchanged() {
    let (origin, front) = origin_and_front().await;
    let response = client()
        .post(format!("{front}/v1/projects/p/keys?page=2&a=%2F"))
        .header("cookie", "gf_session=s1")
        .header("x-custom", "kept")
        .header("connection", "x-private")
        .header("x-private", "dropped")
        .header("x-forwarded-for", "6.6.6.6")
        .header("x-forwarded-proto", "https")
        .body("hello")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-upstream"], "1");
    assert!(!response.headers().contains_key("x-hop"));
    let echo: Value = response.json().await.unwrap();
    assert_eq!(echo["method"], "POST");
    assert_eq!(echo["target"], "/v1/projects/p/keys?page=2&a=%2F");
    assert_eq!(echo["body"], 5);
    let headers = &echo["headers"];
    assert_eq!(headers["cookie"], "gf_session=s1");
    assert_eq!(headers["x-custom"], "kept");
    assert!(headers.get("x-private").is_none());
    assert_eq!(headers["x-forwarded-for"], "127.0.0.1");
    assert!(headers.get("x-forwarded-proto").is_none());
    assert_eq!(headers["x-forwarded-host"], headers["host"]);
    assert_eq!(origin.hits.load(Ordering::SeqCst), 1);

    let missing = client()
        .get(format!("{front}/missing"))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);
    assert_eq!(missing.headers()["x-upstream"], "1");
}

#[tokio::test]
async fn redirects_and_cookies_come_back_as_sent() {
    let (_, front) = origin_and_front().await;
    let response = client()
        .get(format!("{front}/auth/redirect"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FOUND);
    assert_eq!(
        response.headers()[header::LOCATION],
        "/auth/callback?code=abc&state=xyz"
    );
    let cookies: Vec<&HeaderValue> = response
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .collect();
    assert_eq!(
        cookies,
        [
            "gf_state=xyz; HttpOnly; Path=/",
            "gf_flow=1; Secure; Path=/auth"
        ]
    );
}

#[tokio::test]
async fn answers_stream_back_before_they_finish() {
    let (origin, front) = origin_and_front().await;
    let response = client()
        .get(format!("{front}/stream"))
        .send()
        .await
        .unwrap();
    let mut chunks = response.bytes_stream();
    let first = tokio::time::timeout(Duration::from_secs(5), chunks.next())
        .await
        .expect("the first chunk arrives while the upstream still writes")
        .unwrap()
        .unwrap();
    assert_eq!(&first[..], b"first;");
    origin.release.notify_one();
    let rest = chunks.next().await.unwrap().unwrap();
    assert_eq!(&rest[..], b"rest");
}

#[tokio::test]
async fn request_bodies_stream_up_within_their_bound() {
    let (_, front) = origin_and_front().await;
    let chunks = futures::stream::iter(
        (0..4).map(|_| Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b'x'; 200]))),
    );
    let response = client()
        .post(format!("{front}/upload"))
        .body(reqwest::Body::wrap_stream(chunks))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let echo: Value = response.json().await.unwrap();
    assert_eq!(echo["body"], 800);

    let response = client()
        .post(format!("{front}/upload"))
        .body(vec![b'x'; 2048])
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 413);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "CapacityExceeded");

    let chunks = futures::stream::iter(
        (0..8).map(|_| Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b'x'; 200]))),
    );
    let response = client()
        .post(format!("{front}/upload"))
        .body(reqwest::Body::wrap_stream(chunks))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 413);
}

#[tokio::test]
async fn upstream_failures_answer_502_and_504_in_the_error_envelope() {
    let (_, front) = origin_and_front().await;
    let response = client().get(format!("{front}/slow")).send().await.unwrap();
    assert_eq!(response.status(), 504);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "UpstreamTimeout");

    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = closed.local_addr().unwrap();
    drop(closed);
    let front = front_door(&format!("http://{address}"), Duration::from_secs(1), 1024).await;
    let response = client()
        .get(format!("{front}/console"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 502);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/json; charset=utf-8"
    );
    let body: Value = response.json().await.unwrap();
    assert_eq!(
        body,
        json!({"error": {"code": "UpstreamUnavailable", "message": "the upstream is unavailable"}})
    );
}

#[tokio::test]
async fn inference_routes_are_never_proxied() {
    let (origin, front) = origin_and_front().await;
    for (method, path) in [
        ("GET", "/v1/projects/local/models"),
        ("POST", "/v1/projects/local/predict"),
        ("POST", "/v1/projects/local/predictors/x/predict"),
        (
            "POST",
            "/v1/projects/local/predictors/x/predict?env=staging",
        ),
        ("GET", "/v1/projects/local/models/m"),
        ("GET", "/v1/projects/local/models/"),
        ("GET", "/readyz"),
    ] {
        let response = client()
            .request(method.parse().unwrap(), format!("{front}{path}"))
            .send()
            .await
            .unwrap();
        assert!(
            !response.headers().contains_key("x-upstream"),
            "{method} {path}"
        );
    }
    assert_eq!(origin.hits.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn the_upstream_is_ready_only_while_its_readyz_answers_200() {
    let state = Arc::new(Origin::default());
    let address = serve(Router::new().fallback(origin).with_state(state.clone())).await;
    let upstream = Upstream::new(
        &format!("http://{address}"),
        Duration::from_secs(1),
        1024,
        Vec::new(),
    )
    .unwrap();
    assert!(upstream.ready().await);
    state.down.store(true, Ordering::SeqCst);
    assert!(!upstream.ready().await);

    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gone = closed.local_addr().unwrap();
    drop(closed);
    let upstream = Upstream::new(
        &format!("http://{gone}"),
        Duration::from_secs(1),
        1024,
        Vec::new(),
    )
    .unwrap();
    assert!(!upstream.ready().await);
}

#[test]
fn dot_segments_are_recognized_however_they_are_escaped() {
    for path in [
        "/./internal/v1/keys:introspect",
        "/%2e/internal/v1/keys:introspect",
        "/v1/../internal/v1/keys:introspect",
        "/v1/projects/p/x/%2E%2e/predict",
    ] {
        assert!(dotted(path), "{path}");
    }
    for path in [
        "/v1/projects/p/predict",
        "/v1/projects/p/.hidden",
        "/v1/projects/p/a..b",
    ] {
        assert!(!dotted(path), "{path}");
    }
}

#[tokio::test]
async fn no_spelling_of_a_route_reaches_the_upstream_as_another() {
    let (origin, front) = origin_and_front().await;
    for path in [
        "/./internal/v1/keys:introspect",
        "/%2e/internal/v1/keys:introspect",
        "/v1/../internal/v1/keys:introspect",
        "/v1/projects/p/x/../predict",
        "/v1/proj%65cts/p/predict",
    ] {
        // Sent as written, without the client resolving its dot segments.
        let mut stream = tokio::net::TcpStream::connect(front.trim_start_matches("http://"))
            .await
            .unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).await.unwrap();
        assert!(!answer.contains("x-upstream"), "{path}: {answer}");
        assert!(answer.starts_with("HTTP/1.1 4"), "{path}: {answer}");
    }
    assert_eq!(origin.hits.load(Ordering::SeqCst), 0);
}
