// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Flight actions parsed once, and answers read across versions.

use std::path::Path;

use arrow_flight::Action;
use gf_server::action::{Answer, CopyLength, FlightAction, profile_directory};
use gf_server::auth::Scope;
use serde_json::json;
use tonic::Code;

fn action(kind: &str, body: serde_json::Value) -> Action {
    Action {
        r#type: kind.to_string(),
        body: body.to_string().into(),
    }
}

#[test]
fn actions_are_parsed_once_with_what_they_need() {
    let delete = action("delete", json!({"table": "rows", "keys": [1, -2]}));
    let parsed = FlightAction::parse(&delete).unwrap();
    assert_eq!(parsed.table().map(|table| table.as_str()), Some("rows"));
    assert!(parsed.needs_lease() && parsed.changes_rows() && !parsed.needs_replication());
    let lease = action("lease", json!({"seconds": 5.0, "holder": "router"}));
    assert!(FlightAction::parse(&lease).unwrap().needs_replication());

    let refused = |kind: &str, body: serde_json::Value| {
        FlightAction::parse(&action(kind, body)).unwrap_err().code()
    };
    assert_eq!(refused("rename", json!({})), Code::Unimplemented);
    assert_eq!(
        refused("describe", json!({"table": "../rows"})),
        Code::InvalidArgument
    );
    let long = "x".repeat(65);
    let body = json!({"table": "rows", "keys": [1], "write_id": long});
    assert_eq!(refused("delete", body), Code::InvalidArgument);
    assert_eq!(
        refused("lease", json!({"seconds": 5.0, "holder": ""})),
        Code::InvalidArgument
    );
}

#[test]
fn answers_from_older_nodes_still_read() {
    // Older nodes answered an unknown creation with an empty string.
    let answer = Answer::of(&json!({"rows": 2, "log_length": 7, "created": "", "new": 1}));
    assert_eq!((answer.log_length, answer.created), (Some(7), None));
    assert_eq!(Answer::of(&json!("not an answer")), Answer::default());

    let copy =
        json!({"length": 3, "terms": [{"term": 0, "primary": "", "start": 0}], "created": ""});
    let copy: CopyLength = serde_json::from_value(copy).unwrap();
    assert_eq!((copy.length, copy.created), (3, None));
    assert!(serde_json::from_value::<CopyLength>(json!({"length": 3, "terms": []})).is_err());
}

#[test]
fn a_worker_profile_names_its_directory_and_lasts_ten_seconds_unless_told() {
    let parsed = FlightAction::parse(&action("profile_workers", json!({"directory": "/tmp/p"})));
    let FlightAction::ProfileWorkers {
        seconds,
        directory,
        trace,
    } = parsed.unwrap()
    else {
        panic!("not a profile")
    };
    assert_eq!(
        (seconds, directory.as_str(), trace),
        (10.0, "/tmp/p", false)
    );
    let traced = FlightAction::parse(&action(
        "profile_workers",
        json!({"directory": "/tmp/p", "seconds": 2, "trace": true}),
    ));
    assert!(matches!(
        traced.unwrap(),
        FlightAction::ProfileWorkers { trace: true, .. }
    ));
    assert!(
        !FlightAction::parse(&action("profile_workers", json!({"directory": "/tmp/p"})))
            .unwrap()
            .needs_lease()
    );

    let refused = |body: serde_json::Value| {
        FlightAction::parse(&action("profile_workers", body))
            .unwrap_err()
            .code()
    };
    assert_eq!(refused(json!({})), Code::InvalidArgument);
    assert_eq!(
        refused(json!({"directory": "/tmp/p", "seconds": 0})),
        Code::InvalidArgument
    );
}

#[test]
fn a_worker_profile_needs_every_table_and_stays_under_the_profiling_directory() {
    let profile =
        FlightAction::parse(&action("profile_workers", json!({"directory": "p"}))).unwrap();
    assert!(profile.check_scope(&Scope::All).is_ok());
    let scoped = Scope::Tables(vec!["*".to_string()]);
    assert_eq!(
        profile.check_scope(&scoped).unwrap_err().code(),
        Code::PermissionDenied
    );

    let root = Path::new("/var/profiles");
    assert_eq!(
        profile_directory(root, "run-1/a").unwrap(),
        root.join("run-1/a")
    );
    assert_eq!(
        profile_directory(root, "./run-1").unwrap(),
        root.join("run-1")
    );
    for escaping in ["/tmp/p", "../p", "run/../../p"] {
        assert_eq!(
            profile_directory(root, escaping).unwrap_err().code(),
            Code::InvalidArgument,
            "{escaping}"
        );
    }
}

#[test]
fn actions_on_every_table_need_an_unscoped_token() {
    let scoped = Scope::Tables(vec!["tenant-a*".to_string()]);
    for (kind, body) in [
        ("rebalance", json!({})),
        ("profile_workers", json!({"directory": "run"})),
    ] {
        let parsed = FlightAction::parse(&action(kind, body)).unwrap();
        assert!(parsed.needs_every_table(), "{kind}");
        assert_eq!(
            parsed.check_scope(&scoped).unwrap_err().code(),
            Code::PermissionDenied,
            "{kind}"
        );
        assert!(parsed.check_scope(&Scope::All).is_ok(), "{kind}");
    }
    // A table's own actions stay within the scope that reaches the table.
    let moved = action("move_table", json!({"table": "tenant-a-rows", "shard": 1}));
    let parsed = FlightAction::parse(&moved).unwrap();
    assert!(!parsed.needs_every_table());
    assert!(parsed.check_scope(&scoped).is_ok());
}
