// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The Flight service, called as clients call it: scopes, names, quantile
//! levels and the request budget are checked where a call arrives.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_flight::error::FlightError;
use arrow_flight::{Action, FlightClient, FlightData, FlightDescriptor};
use futures::{StreamExt, TryStreamExt, stream};
use gf_server::auth::{QUANTILES, TokenConfig, Tokens};
use gf_server::metrics::Metrics;
use gf_server::service::Service;
use serde_json::json;
use tonic::Code;

const TOKEN: &str = "service-test-token";
const SCOPED: &str = "service-test-scoped";

/// A Flight service over one shard, with `budget` bytes of requests, and a
/// client for `token`.
async fn client(budget: u64, token: &str) -> FlightClient {
    let service = Service::new(
        vec![common::shard().await],
        HashMap::new(),
        Arc::new(Metrics::default()),
    )
    .with_request_budget(budget);
    let tokens = Tokens::new(&[
        TokenConfig::Any(TOKEN.to_string()),
        TokenConfig::Scoped {
            token: SCOPED.to_string(),
            tables: vec!["tenant-*".to_string()],
        },
    ]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(service.into_server(tokens, Tokens::default()))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
    );
    let channel = tonic::transport::Channel::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightClient::new(channel);
    client
        .add_header("authorization", &format!("Bearer {token}"))
        .unwrap();
    client
}

fn code(error: FlightError) -> Code {
    match error {
        FlightError::Tonic(status) => status.code(),
        other => panic!("not a status: {other}"),
    }
}

async fn act(client: &mut FlightClient, kind: &str, body: serde_json::Value) -> Result<(), Code> {
    let action = Action {
        r#type: kind.to_string(),
        body: body.to_string().into(),
    };
    let answers = client.do_action(action).await.map_err(code)?;
    answers.try_collect::<Vec<_>>().await.map_err(code)?;
    Ok(())
}

fn table(name: &str) -> serde_json::Value {
    json!({
        "name": name,
        "model": "tiny",
        "columns": [{"name": "x", "kind": "numeric"}],
        "classes": 2,
    })
}

#[tokio::test]
async fn a_scoped_token_may_not_rebalance_the_whole_server() {
    let mut scoped = client(1 << 30, SCOPED).await;
    assert_eq!(
        act(&mut scoped, "rebalance", json!({})).await,
        Err(Code::PermissionDenied)
    );
}

#[tokio::test]
async fn table_names_may_not_differ_only_by_case_nor_take_the_bound_prefix() {
    let mut client = client(1 << 30, TOKEN).await;
    act(&mut client, "create_table", table("Payments"))
        .await
        .unwrap();
    assert_eq!(
        act(&mut client, "create_table", table("payments")).await,
        Err(Code::AlreadyExists)
    );
    assert_eq!(
        act(&mut client, "create_table", table("pkg-0123")).await,
        Err(Code::InvalidArgument)
    );
}

#[tokio::test]
async fn a_delete_action_holds_its_keys_against_the_request_budget() {
    let mut client = client(1024, TOKEN).await;
    act(&mut client, "create_table", table("rows"))
        .await
        .unwrap();
    let keys: Vec<u64> = (0..100).collect();
    let body = json!({"table": "rows", "keys": keys, "missing_ok": true});
    assert_eq!(
        act(&mut client, "delete", body).await,
        Err(Code::ResourceExhausted)
    );
    let body = json!({"table": "rows", "keys": [1, 2], "missing_ok": true});
    act(&mut client, "delete", body).await.unwrap();
}

#[tokio::test]
async fn quantile_levels_are_bounded_as_the_models_api_bounds_them() {
    let mut client = client(1 << 30, TOKEN).await;
    // Past the Models API's most levels, and small enough for a header.
    let many: Vec<String> = (1..=120)
        .map(|k| format!("{:.4}", k as f64 / 121.0))
        .collect();
    for levels in [many.join(","), "0.5,1.5".to_string(), "0.5,".to_string()] {
        client.add_header(QUANTILES, &levels).unwrap();
        let descriptor = FlightDescriptor::new_path(vec!["predict".into(), "rows".into()]);
        let messages = stream::iter([Ok(FlightData::new().with_descriptor(descriptor))]);
        let refused = match client.do_exchange(messages).await {
            Ok(mut answers) => answers.next().await.and_then(Result::err),
            Err(error) => Some(error),
        };
        assert_eq!(
            refused.map(code),
            Some(Code::InvalidArgument),
            "{}",
            &levels[..levels.len().min(40)]
        );
    }
}
