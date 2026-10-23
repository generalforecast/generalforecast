// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `gf-router init`: a starter configuration for a router over a few nodes.

/// A router over `nodes` servers on this machine, as `init` writes it.
#[must_use]
pub fn starter_config(listen: &str, nodes: &[String], token: &str, replication: &str) -> String {
    let nodes: Vec<String> = nodes
        .iter()
        .enumerate()
        .map(|(index, address)| {
            format!(
                "[[nodes]]\nname = \"node-{index}\"\naddress = {address:?}\ntoken = {replication:?}\n"
            )
        })
        .collect();
    let replicas = nodes.len().min(3);
    format!(
        r#"# A generalforecast router. Each node is a gf-server whose configuration lists the
# replication token in `replication_tokens` and sets `require_lease = true`.
# Check it with `gf serve --check --config` and start it with `gf serve --config`.

listen = {listen:?}
name = "router-0"        # unique among routers; nodes hold leases under it
replicas = {replicas}             # copies of each table
check_seconds = 1.0      # how often to check nodes and repair copies
failover_seconds = 10    # promote a replica once a primary is out of reach this long

# Beyond this machine, serve over TLS and give nodes https:// addresses.
# [tls]
# certificate = "/etc/generalforecast/router.pem"
# key = "/etc/generalforecast/router.key"

[[clients]]
token = {token:?}

{nodes}"#,
        nodes = nodes.join("\n"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn the_starter_configuration_loads() {
        let directory =
            std::env::temp_dir().join(format!("gf-router-setup-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("router.toml");
        let nodes = [
            "http://127.0.0.1:8816".to_string(),
            "http://127.0.0.1:8817".to_string(),
        ];
        std::fs::write(&path, starter_config("127.0.0.1:8815", &nodes, "t", "r")).unwrap();

        let config = Config::load(&path).unwrap();

        assert_eq!(config.nodes.len(), 2);
        assert_eq!(config.replicas, 2);
        assert_eq!(config.clients[0].token, "t");
    }
}
