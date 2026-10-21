// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The router's configuration: what it refuses, and the lease it derives.

use std::time::Duration;

use gf_router::config::Config;

/// A configuration with `top` among its settings and `client` in its client's.
fn load(top: &str, client: &str, address: &str) -> Result<Config, String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "gf-router-config-{}-{index}.toml",
        std::process::id()
    ));
    let text = format!(
        "listen = \"127.0.0.1:8815\"\n{top}\n[[clients]]\ntoken = \"secret\"\n{client}\n\
         [[nodes]]\nname = \"node-0\"\naddress = \"{address}\"\ntoken = \"replicator\"\n"
    );
    std::fs::write(&path, text).unwrap();
    let loaded = Config::load(&path);
    let _ = std::fs::remove_file(&path);
    loaded
}

const PLAIN: &str = "http://10.0.0.5:8816";

#[test]
fn quotas_leases_and_timeouts_must_be_positive() {
    for setting in ["per_second = 0.0", "per_second = inf", "burst = 0"] {
        assert!(load("", setting, PLAIN).is_err(), "{setting} was accepted");
    }
    assert!(load("call_timeout_seconds = -1", "", PLAIN).is_err());
    let failover = "name = \"router-0\"\nfailover_seconds = 10";
    assert!(load(&format!("{failover}\nlease_seconds = 0"), "", PLAIN).is_err());
    assert!(load(&format!("{failover}\nlease_seconds = 6"), "", PLAIN).is_err());
    assert!(load(&format!("{failover}\nlease_seconds = 5"), "", PLAIN).is_ok());
}

#[test]
fn only_a_named_router_that_fails_over_takes_leases() {
    // Nodes hold a lease under its holder's name, so two routers never share one.
    assert!(load("failover_seconds = 10", "", PLAIN).is_err());
    // A router that does not fail over has no use for leases.
    assert!(load("lease_seconds = 5", "", PLAIN).is_err());
    assert!(load("name = \"router-0\"\nfailover_seconds = 10", "", PLAIN).is_ok());
}

#[test]
fn leases_default_to_half_the_failover_and_plaintext_nodes_are_named() {
    let config = load("name = \"router-0\"\nfailover_seconds = 10", "", PLAIN).unwrap();
    assert_eq!(config.lease(), Some(Duration::from_secs(5)));
    assert_eq!(load("", "", PLAIN).unwrap().lease(), None);
    assert_eq!(config.plaintext_nodes(), ["node-0"]);

    let local = load("", "", "http://127.0.0.1:8816").unwrap();
    assert!(local.plaintext_nodes().is_empty());
    let secure = load("", "", "https://10.0.0.5:8816").unwrap();
    assert!(secure.plaintext_nodes().is_empty());
    let named = load("", "", "http://gf-node:8816").unwrap();
    assert_eq!(named.plaintext_nodes(), ["node-0"]);
    let loopback = load("", "", "http://[::1]:8816").unwrap();
    assert!(loopback.plaintext_nodes().is_empty());
}
