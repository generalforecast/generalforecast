// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Tables split into partitions, through routers over nodes that answer
//! the actions the router makes of them and keep tables as names alone.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::{Stream, TryStreamExt, stream};
use gf_router::partitions;
use gf_router::placement::Placement;
use gf_router::service::{Cluster, Node, Router, Timeouts};
use gf_server::action::{CopyLength, LogLengths};
use gf_server::auth::Scope;
use gf_server::wal::{CreationId, Term, Terms};
use serde_json::{Value, json};
use tonic::metadata::MetadataValue;
use tonic::transport::server::TcpIncoming;
use tonic::transport::{Channel, Server};
use tonic::{Code, Request, Response, Status, Streaming};

type Boxed<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// A delete a node took: its table and keys.
type Deleted = (String, Vec<i128>);

/// A node's table: its definition as created, its log and its terms.
#[derive(Clone)]
struct Held {
    definition: Value,
    length: u64,
    terms: Terms,
    created: CreationId,
}

/// A node that keeps tables by name and logs each write as one entry.
#[derive(Clone, Default)]
struct Fake {
    tables: Arc<Mutex<BTreeMap<String, Held>>>,
    /// Each delete the node took.
    deleted: Arc<Mutex<Vec<Deleted>>>,
    made: Arc<Mutex<u64>>,
    /// Set while the node is out of reach: it answers nothing.
    down: Arc<AtomicBool>,
}

impl Fake {
    /// Makes a table as another router would, unseen by those under test.
    fn make(&self, name: &str, definition: Value) {
        let mut made = self.made.lock().unwrap();
        *made += 1;
        self.tables.lock().unwrap().insert(
            name.to_string(),
            Held {
                definition,
                length: 1,
                terms: Terms::default(),
                created: CreationId::parse(&format!("creation-{made}")).unwrap(),
            },
        );
    }

    fn answer(&self, kind: &str, body: &Value) -> Result<Value, Status> {
        let table = body
            .get("table")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let mut tables = self.tables.lock().unwrap();
        let missing = || Status::not_found(format!("table {table:?} does not exist"));
        match kind {
            "health" => Ok(json!("ok")),
            "list_tables" => Ok(json!(tables.keys().collect::<Vec<_>>())),
            "log_lengths" => {
                let lengths: LogLengths = tables
                    .iter()
                    .map(|(name, held)| {
                        let copy = CopyLength {
                            length: held.length,
                            terms: held.terms.clone(),
                            created: Some(held.created.clone()),
                        };
                        (name.clone(), copy)
                    })
                    .collect();
                Ok(serde_json::to_value(lengths).unwrap())
            }
            "create_table" => {
                let name = body["name"].as_str().unwrap().to_string();
                if tables.contains_key(&name) {
                    return Err(gf_server::service::exists(format!("table {name:?} exists")));
                }
                drop(tables);
                self.make(&name, body.clone());
                let created = self.tables.lock().unwrap()[&name].created.clone();
                Ok(json!({"log_length": 1, "created": created.as_str()}))
            }
            "describe" => {
                let held = tables.get(table).ok_or_else(missing)?;
                let mut described = held.definition.clone();
                described["rows"] = json!(0);
                Ok(described)
            }
            "promote" => {
                let held = tables.get_mut(table).ok_or_else(missing)?;
                held.terms.begin(Term {
                    term: body["term"].as_u64().unwrap(),
                    primary: body["primary"].as_str().unwrap().to_string(),
                    start: held.length,
                });
                Ok(json!({"log_length": held.length}))
            }
            "delete" => {
                let held = tables.get_mut(table).ok_or_else(missing)?;
                held.length += 1;
                let keys: Vec<i128> = serde_json::from_value(body["keys"].clone()).unwrap();
                let (rows, length) = (keys.len(), held.length);
                self.deleted.lock().unwrap().push((table.to_string(), keys));
                Ok(json!({"rows": rows, "log_length": length}))
            }
            "drop_table" => tables.remove(table).map(|_| json!({})).ok_or_else(missing),
            other => Err(Status::unimplemented(other.to_string())),
        }
    }
}

#[tonic::async_trait]
impl FlightService for Fake {
    type HandshakeStream = Boxed<HandshakeResponse>;
    type ListFlightsStream = Boxed<FlightInfo>;
    type DoGetStream = Boxed<FlightData>;
    type DoPutStream = Boxed<PutResult>;
    type DoActionStream = Boxed<arrow_flight::Result>;
    type ListActionsStream = Boxed<ActionType>;
    type DoExchangeStream = Boxed<FlightData>;

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        if self.down.load(Ordering::Relaxed) {
            return Err(Status::unavailable("the node is down"));
        }
        let action = request.into_inner();
        let body: Value = serde_json::from_slice(&action.body).unwrap_or(Value::Null);
        let answer = self.answer(&action.r#type, &body)?;
        let result = arrow_flight::Result {
            body: answer.to_string().into(),
        };
        Ok(Response::new(Box::pin(stream::iter([Ok(result)]))))
    }

    async fn handshake(
        &self,
        _: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("handshake"))
    }

    async fn list_flights(
        &self,
        _: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights"))
    }

    async fn get_flight_info(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info"))
    }

    async fn poll_flight_info(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }

    async fn get_schema(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }

    async fn do_get(&self, _: Request<Ticket>) -> Result<Response<Self::DoGetStream>, Status> {
        Err(Status::unimplemented("do_get"))
    }

    async fn do_put(
        &self,
        _: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put"))
    }

    async fn do_exchange(
        &self,
        _: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange"))
    }

    async fn list_actions(
        &self,
        _: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions"))
    }
}

/// A fake node serving on a port of its own, and its address.
async fn serve(fake: &Fake) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let service = FlightServiceServer::new(fake.clone());
    tokio::spawn(
        Server::builder()
            .add_service(service)
            .serve_with_incoming(TcpIncoming::from(listener)),
    );
    address
}

/// A router over one node, which has learned what the node holds.
async fn router(address: &str) -> Router {
    router_over(&[address]).await
}

/// A router over nodes `node-0` on, one copy of each table, which has
/// learned what the nodes hold.
async fn router_over(addresses: &[&str]) -> Router {
    let timeouts = Timeouts {
        call: Duration::from_secs(5),
        write: Duration::from_secs(5),
    };
    let nodes = addresses
        .iter()
        .enumerate()
        .map(|(index, address)| {
            let channel = Channel::from_shared(address.to_string())
                .unwrap()
                .connect_lazy();
            Node::new(
                format!("node-{index}"),
                FlightServiceClient::new(channel),
                MetadataValue::from_static("Bearer replicator"),
                timeouts,
            )
        })
        .collect();
    let router = Router::new(Cluster::new(nodes, 1));
    router.prepare().await;
    router
}

/// Calls an action on a router as a client with `scope`.
async fn act(
    router: &Router,
    scope: &Scope,
    kind: &str,
    body: Value,
) -> Result<Vec<Value>, Status> {
    let mut request = Request::new(Action {
        r#type: kind.to_string(),
        body: body.to_string().into(),
    });
    request.extensions_mut().insert(scope.clone());
    let answers = router.do_action(request).await?.into_inner();
    let answers: Vec<arrow_flight::Result> = answers.try_collect().await?;
    Ok(answers
        .iter()
        .map(|answer| serde_json::from_slice(&answer.body).unwrap())
        .collect())
}

fn definition(name: &str, partitions: Option<usize>) -> Value {
    let mut definition = json!({
        "name": name,
        "model": "tiny",
        "columns": [{"name": "amount", "kind": "numeric"}],
        "classes": 0,
    });
    if let Some(partitions) = partitions {
        definition["partitions"] = json!(partitions);
    }
    definition
}

async fn create(router: &Router, name: &str, partitions: Option<usize>) -> Result<(), Status> {
    act(
        router,
        &Scope::All,
        "create_table",
        definition(name, partitions),
    )
    .await
    .map(drop)
}

#[tokio::test]
async fn a_name_serves_one_table_however_it_is_split() {
    let fake = Fake::default();
    let router = router(&serve(&fake).await).await;

    create(&router, "split", Some(4)).await.unwrap();

    // A second table under the name would hide the first one's rows.
    for partitions in [None, Some(1), Some(8)] {
        let refused = create(&router, "split", partitions).await.unwrap_err();
        assert_eq!(refused.code(), Code::AlreadyExists, "{partitions:?}");
    }
    create(&router, "whole", None).await.unwrap();
    let refused = create(&router, "whole", Some(2)).await.unwrap_err();
    assert_eq!(refused.code(), Code::AlreadyExists);
    // Tables made through another router since this one's check count too.
    fake.make(
        &partitions::name("elsewhere", 0, 2),
        definition("elsewhere", None),
    );
    fake.make("plain", definition("plain", None));
    let refused = create(&router, "elsewhere", None).await.unwrap_err();
    assert_eq!(refused.code(), Code::AlreadyExists);
    let refused = create(&router, "plain", Some(3)).await.unwrap_err();
    assert_eq!(refused.code(), Code::AlreadyExists);

    // Once dropped, every partition goes, and the name is free again.
    act(
        &router,
        &Scope::All,
        "drop_table",
        json!({"table": "split"}),
    )
    .await
    .unwrap();
    assert!(
        fake.tables
            .lock()
            .unwrap()
            .keys()
            .all(|name| !name.starts_with("split"))
    );
    create(&router, "split", None).await.unwrap();
}

#[tokio::test]
async fn a_partition_is_reached_only_through_its_table_and_its_scope() {
    let fake = Fake::default();
    let router = router(&serve(&fake).await).await;
    create(&router, "team", Some(2)).await.unwrap();
    let partition = partitions::name("team", 0, 2);
    let describe = json!({"table": partition});

    // A scope that matches the partition's name but not its table's reaches neither.
    let scoped = Scope::Tables(vec!["team-*".to_string()]);
    let refused = act(&router, &scoped, "describe", describe.clone())
        .await
        .unwrap_err();
    assert_eq!(refused.code(), Code::NotFound);
    let refused = act(&router, &scoped, "truncate", describe.clone())
        .await
        .unwrap_err();
    assert_eq!(refused.code(), Code::NotFound);

    // In scope, a partition may be read but not written past its table.
    let answer = act(&router, &Scope::All, "describe", describe.clone())
        .await
        .unwrap();
    assert_eq!(answer[0]["name"], json!(partition));
    for (kind, body) in [
        ("truncate", describe.clone()),
        ("refit", describe.clone()),
        ("delete", json!({"table": partition, "keys": [1]})),
        ("drop_table", describe.clone()),
    ] {
        let refused = act(&router, &Scope::All, kind, body).await.unwrap_err();
        assert_eq!(refused.code(), Code::InvalidArgument, "{kind}");
        assert!(refused.message().contains("partition of table"), "{kind}");
    }
    assert!(fake.deleted.lock().unwrap().is_empty());
    assert_eq!(fake.tables.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn a_write_id_too_long_for_its_parts_is_refused_before_any_part_is_written() {
    let fake = Fake::default();
    let router = router(&serve(&fake).await).await;
    create(&router, "ids", Some(12)).await.unwrap();
    let keys: Vec<i128> = (0..64).collect();

    // `/9` fits 62 characters into a write id, and `/11` does not.
    let long = "w".repeat(62);
    let body = json!({"table": "ids", "keys": keys, "write_id": long});
    let refused = act(&router, &Scope::All, "delete", body).await.unwrap_err();
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(
        refused.message().contains("shorter id"),
        "{}",
        refused.message()
    );
    assert!(fake.deleted.lock().unwrap().is_empty());

    let body = json!({"table": "ids", "keys": keys, "write_id": "w".repeat(61)});
    act(&router, &Scope::All, "delete", body).await.unwrap();
    assert_eq!(fake.deleted.lock().unwrap().len(), 12);
}

#[tokio::test]
async fn a_router_learns_another_routers_drop_and_new_table_at_its_check() {
    let fake = Fake::default();
    let address = serve(&fake).await;
    let (first, second) = (router(&address).await, router(&address).await);
    let mut routed = definition("orders", Some(2));
    routed["route_by"] = json!({"column": "amount"});
    act(&first, &Scope::All, "create_table", routed)
        .await
        .unwrap();
    second.prepare().await;
    let delete = json!({"table": "orders", "keys": [7]});

    // A routed table's delete goes to every partition, keys saying nothing of the route.
    act(&second, &Scope::All, "delete", delete.clone())
        .await
        .unwrap();
    assert_eq!(fake.deleted.lock().unwrap().len(), 2);

    // Made again through the first router, unrouted and of another count.
    act(
        &first,
        &Scope::All,
        "drop_table",
        json!({"table": "orders"}),
    )
    .await
    .unwrap();
    create(&first, "orders", Some(3)).await.unwrap();
    second.prepare().await;

    fake.deleted.lock().unwrap().clear();
    act(&second, &Scope::All, "delete", delete).await.unwrap();
    let deleted = fake.deleted.lock().unwrap().clone();
    let owner = partitions::name("orders", partitions::of(7, 3), 3);
    assert_eq!(deleted, [(owner, vec![7])]);
    let described = act(&second, &Scope::All, "describe", json!({"table": "orders"}))
        .await
        .unwrap();
    assert_eq!(described[0]["partitions"], json!(3));
}

#[tokio::test]
async fn two_creates_of_one_name_through_one_router_make_one_table() {
    let fake = Fake::default();
    let router = router(&serve(&fake).await).await;

    for (index, (first, second)) in [(None, Some(3)), (Some(2), Some(3)), (None, None)]
        .into_iter()
        .enumerate()
    {
        let name = format!("race-{index}");
        let (one, other) = tokio::join!(
            create(&router, &name, first),
            create(&router, &name, second)
        );
        let codes = [one, other].map(|made| made.err().map(|status| status.code()));
        assert!(
            codes.contains(&None) && codes.contains(&Some(Code::AlreadyExists)),
            "{name}: {codes:?}"
        );
        let held = fake.tables.lock().unwrap().clone();
        let serving: BTreeMap<_, _> = held
            .keys()
            .filter(|table| {
                *table == &name
                    || partitions::parse(table).is_some_and(|(owner, _, _)| owner == name)
            })
            .map(|table| {
                (
                    table.clone(),
                    partitions::parse(table).map(|(_, _, count)| count),
                )
            })
            .collect();
        let counts: std::collections::BTreeSet<_> = serving.values().collect();
        assert_eq!(counts.len(), 1, "{name}: {serving:?}");
    }
}

#[tokio::test]
async fn a_name_a_node_out_of_reach_holds_stays_taken() {
    let (first, second) = (Fake::default(), Fake::default());
    let (one, other) = (serve(&first).await, serve(&second).await);
    // Names the up node ranks first, so a create of them could go ahead there.
    let placement = Placement::new(vec!["node-0".to_string(), "node-1".to_string()]);
    let mut names = (0..).map(|index| format!("held-{index}"));
    let mut ranked_up = || names.find(|name| placement.primary(name) == 0).unwrap();
    let (whole, split) = (ranked_up(), ranked_up());
    second.make(&whole, definition(&whole, None));
    second.make(&partitions::name(&split, 0, 2), definition(&split, None));
    let router = router_over(&[&one, &other]).await;

    second.down.store(true, Ordering::Relaxed);
    router.prepare().await;

    for (name, partitions) in [(&whole, None), (&split, None), (&whole, Some(2))] {
        let refused = create(&router, name, partitions).await.unwrap_err();
        assert_eq!(refused.code(), Code::AlreadyExists, "{name} {partitions:?}");
    }
    assert!(first.tables.lock().unwrap().is_empty());
}

#[tokio::test]
async fn dropping_a_partitioned_table_already_gone_is_not_found() {
    let fake = Fake::default();
    let router = router(&serve(&fake).await).await;
    create(&router, "gone", Some(2)).await.unwrap();
    // Dropped through another router, before this one's next check.
    fake.tables.lock().unwrap().clear();

    let refused = act(&router, &Scope::All, "drop_table", json!({"table": "gone"}))
        .await
        .unwrap_err();

    assert_eq!(refused.code(), Code::NotFound);
}

#[tokio::test]
async fn writes_follow_the_route_the_table_has_now() {
    let fake = Fake::default();
    let address = serve(&fake).await;
    let (first, second) = (router(&address).await, router(&address).await);
    let mut routed = definition("orders", Some(2));
    routed["route_by"] = json!({"column": "amount"});
    act(&first, &Scope::All, "create_table", routed)
        .await
        .unwrap();
    second.prepare().await;
    let delete = json!({"table": "orders", "keys": [7]});
    act(&second, &Scope::All, "delete", delete.clone())
        .await
        .unwrap();
    assert_eq!(fake.deleted.lock().unwrap().len(), 2);
    // The second router reads the route at the partitions' copies as it
    // last saw them, writing nothing.
    second.prepare().await;
    act(
        &second,
        &Scope::All,
        "delete",
        json!({"table": "orders", "keys": []}),
    )
    .await
    .unwrap();

    // Made again unrouted with the same count, while the second router has
    // not checked since: its view of the partitions is the old table's.
    act(
        &first,
        &Scope::All,
        "drop_table",
        json!({"table": "orders"}),
    )
    .await
    .unwrap();
    create(&first, "orders", Some(2)).await.unwrap();
    fake.deleted.lock().unwrap().clear();
    act(&second, &Scope::All, "delete", delete).await.unwrap();

    let owner = partitions::name("orders", partitions::of(7, 2), 2);
    assert_eq!(*fake.deleted.lock().unwrap(), [(owner, vec![7])]);
}
