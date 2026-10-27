// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Failing tables over: setting aside copies that diverged, and promoting a
//! replica once a primary has been out of reach for long enough.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde_json::json;
use tokio::time::Instant;

use super::{Cluster, Router};
use crate::replication::{TableCopy, current, diverged, promotion};

/// Leases and promotions, for the router that fails tables over.
#[derive(Clone)]
pub(super) struct Failover {
    /// How long a primary must be out of reach before a replica takes over.
    pub(super) after: Duration,
    /// How long each lease lasts; less than `after`.
    pub(super) lease: Duration,
    /// The name nodes hold this router's leases under.
    pub(super) holder: String,
    /// When each node last took a lease from this router, by name.
    pub(super) leased: Arc<Mutex<HashMap<String, Instant>>>,
    /// When the cluster last changed, standing in for nodes never leased.
    pub(super) since: Arc<Mutex<Instant>>,
    /// Set once a node refuses a lease that another router holds: two routers
    /// failing tables over could each promote a copy, so this one stops.
    pub(super) contested: Arc<AtomicBool>,
}

impl Router {
    /// Sets aside copies whose writes diverged, and promotes a replica for
    /// each table whose primary has been out of reach for long enough that its
    /// lease has lapsed.
    pub(super) async fn fail_over(
        &self,
        cluster: &Cluster,
        tables: &[String],
        failover: &Failover,
    ) {
        for table in tables {
            let copies = self.copies(cluster, table);
            let creation = self.creation(cluster, table, &copies);
            let diverged = diverged(&copies, creation.as_ref());
            for (index, (copy, diverged)) in copies.iter().zip(diverged).enumerate() {
                let Some(copy) = copy else { continue };
                let node = &cluster.nodes[index];
                if diverged && node.is_up() {
                    tracing::warn!(
                        table,
                        node = node.name,
                        length = copy.length,
                        "setting aside a copy whose writes diverged from the primary's"
                    );
                    let release = json!({"table": table, "keep": true});
                    match node.control("release", release).await {
                        Ok(_) => self.forget(table, node),
                        Err(status) => tracing::warn!(table, "setting aside a copy: {status}"),
                    }
                }
            }
        }
        for (table, lost, index) in self.lapsed(cluster, tables, failover) {
            let copies = self.kept(cluster, &table);
            let (_, up) = self.view(cluster, &table);
            let seen = self.acknowledged(cluster, &table);
            let primary = index.unwrap_or(usize::MAX);
            let desired = self.desired(cluster, &table);
            let Some(chosen) = promotion(&desired, &copies, &up, primary, seen) else {
                tracing::warn!(
                    table,
                    primary = lost,
                    "no replica holds every acknowledged write, so none is promoted"
                );
                continue;
            };
            let term = current(&copies).map_or(0, |copy| copy.term().term) + 1;
            let node = &cluster.nodes[chosen];
            match self.promote(cluster, &table, chosen, term, chosen).await {
                Ok(_) => {
                    self.promotions.add(1);
                    tracing::info!(table, from = lost, to = node.name, term, "failed over");
                }
                Err(status) => tracing::warn!(table, "promoting {}: {status}", node.name),
            }
        }
    }

    /// The tables whose primary has been out of reach for `failover.after`:
    /// each with that primary's name and index, if it is still configured.
    pub(super) fn lapsed(
        &self,
        cluster: &Cluster,
        tables: &[String],
        failover: &Failover,
    ) -> Vec<(String, String, Option<usize>)> {
        let leases = failover
            .leased
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let since = *failover
            .since
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        tables
            .iter()
            .filter_map(|table| {
                let copies = self.kept(cluster, table);
                let name = match current(&copies).map(TableCopy::term) {
                    Some(term) if term.term > 0 => term.primary.clone(),
                    _ => cluster.nodes[cluster.replicas(table)[0]].name.clone(),
                };
                let index = cluster.index(&name);
                let reachable = index.is_some_and(|index| cluster.nodes[index].is_up());
                let last = leases.get(&name).copied().unwrap_or(since);
                (!reachable && last.elapsed() >= failover.after)
                    .then(|| (table.clone(), name, index))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::Node;
    use super::*;
    use arrow_flight::flight_service_client::FlightServiceClient;
    use std::time::Duration;
    use tonic::metadata::MetadataValue;
    use tonic::transport::Channel;

    use super::super::Timeouts;

    #[tokio::test(start_paused = true)]
    async fn a_table_fails_over_once_its_primary_is_out_of_reach_long_enough() {
        let channel = Channel::from_static("http://127.0.0.1:1").connect_lazy();
        let timeouts = Timeouts {
            call: Duration::from_secs(1),
            write: Duration::from_secs(1),
        };
        let node = |name: &str| {
            let client = FlightServiceClient::new(channel.clone());
            Node::new(
                name.to_string(),
                client,
                MetadataValue::from_static("x"),
                timeouts,
            )
        };
        let after = Duration::from_secs(10);
        let router = Router::new(Cluster::new(vec![node("a")], 1)).with_failover(
            after,
            after / 2,
            "router".to_string(),
        );
        let failover = router.failover.clone().unwrap();
        let tables = ["rows".to_string()];

        // Up, the primary keeps its tables however long passes.
        tokio::time::advance(after * 2).await;
        assert!(
            router
                .lapsed(&router.cluster(), &tables, &failover)
                .is_empty()
        );
        // Never leased, it is out of reach from when the cluster last changed,
        // and keeps its tables until that has lasted `after`.
        router.reconfigure(Cluster::new(vec![node("a")], 1));
        let cluster = router.cluster();
        cluster.nodes[0].down();
        tokio::time::advance(after - Duration::from_secs(1)).await;
        assert!(router.lapsed(&cluster, &tables, &failover).is_empty());
        tokio::time::advance(Duration::from_secs(1)).await;
        let lapsed = router.lapsed(&cluster, &tables, &failover);
        assert_eq!(lapsed, [("rows".to_string(), "a".to_string(), Some(0))]);
    }
}
