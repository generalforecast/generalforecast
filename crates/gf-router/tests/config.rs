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

#[test]
fn a_burst_needs_a_rate() {
    // A client without per_second is not limited, so a burst alone would be ignored.
    assert!(load("", "burst = 5", PLAIN).is_err());
    assert!(load("", "per_second = 2.0\nburst = 5", PLAIN).is_ok());
}

#[test]
fn two_nodes_may_not_be_one_server() {
    let second = |address: &str| {
        format!("[[nodes]]\nname = \"node-1\"\naddress = \"{address}\"\ntoken = \"replicator\"")
    };
    for same in [
        PLAIN,
        "HTTP://10.0.0.5:8816/",
        "http://10.0.0.5:8816/flight",
    ] {
        let refused = load(&second(same), "", PLAIN).unwrap_err();
        assert!(refused.contains("one server"), "{same}: {refused}");
    }
    assert!(load(&second("http://gf-node:80"), "", "http://GF-NODE").is_err());
    assert!(load(&second("http://10.0.0.5:8817"), "", PLAIN).is_ok());
    assert!(load(&second("https://10.0.0.5:8816"), "", PLAIN).is_ok());
}

#[test]
fn a_reload_keeps_the_settings_that_take_effect_on_restart() {
    let running = load("call_timeout_seconds = 60\nreplicas = 1", "", PLAIN).unwrap();
    let loaded = load(
        "call_timeout_seconds = 1200\nwrite_timeout_seconds = 5\nreplicas = 2\n\
         check_seconds = 0.5",
        "per_second = 3.0",
        "http://10.0.0.6:8816",
    )
    .unwrap();

    let (reloaded, ignored) = running.reloaded(loaded.clone());

    // The router's own call timeout is sized from the node timeouts at start,
    // so they change with it, on restart.
    assert_eq!(
        ignored,
        [
            "call_timeout_seconds",
            "check_seconds",
            "write_timeout_seconds"
        ]
    );
    assert_eq!(reloaded.call_timeout_seconds, 60.0);
    assert_eq!(
        reloaded.write_timeout_seconds,
        running.write_timeout_seconds
    );
    assert_eq!(reloaded.check_seconds, running.check_seconds);
    assert_eq!(reloaded.replicas, 2);
    assert_eq!(reloaded.nodes[0].address, "http://10.0.0.6:8816");
    assert_eq!(reloaded.clients[0].per_second, Some(3.0));
    // Every setting is either reloaded or named as kept, and nothing else differs.
    let value = |config: &Config| serde_json::to_value(config).unwrap();
    let (reloaded, running, loaded) = (value(&reloaded), value(&running), value(&loaded));
    for (name, setting) in reloaded.as_object().unwrap() {
        let source = if ignored.contains(name) {
            &running
        } else {
            &loaded
        };
        assert_eq!(Some(setting), source.get(name), "{name}");
    }
}
