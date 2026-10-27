// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Write leases: the router that fails tables over renews every node's, and
//! stands down while another router holds them.

use std::collections::BTreeSet;
use std::sync::PoisonError;
use std::sync::atomic::Ordering;
use std::time::Duration;

use gf_server::action::{LeaseAnswer, unix_ms};
use serde::Deserialize;
use serde_json::json;
use tokio::time::Instant;

use super::failover::Failover;
use super::{CHECK_TIMEOUT, Router};

impl Router {
    /// Renews the write lease of every node that is up, every interval, each
    /// with a deadline shorter than the lease, and dated, so a node that reads
    /// it late grants only what is left of it.
    ///
    /// A node that refuses because another router holds its lease makes this
    /// router stand down: it stops renewing, so its own leases lapse, and
    /// neither fails over nor moves tables. Every `failover.after`, with a
    /// little jitter so two routers do not probe together, it asks every node
    /// who holds its lease, changing nothing, and once no other router holds
    /// any, it takes them all again. So after the router failing tables over
    /// restarts or is lost, one of the routers takes over, and only one.
    pub(super) async fn renew(self, interval: Duration, failover: Failover) {
        let deadline = CHECK_TIMEOUT.min(failover.lease / 2);
        let mut ticks = tokio::time::interval(interval);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut unenforced: BTreeSet<String> = BTreeSet::new();
        // When this router stood down, if it has.
        let mut stood_down: Option<Instant> = None;
        loop {
            ticks.tick().await;
            let probe = stood_down.is_some();
            // Spread over the failover time, so two routers that stood down
            // together probe apart, and the first to probe takes the leases.
            if let Some(since) = stood_down
                && since.elapsed() < failover.after + jitter(failover.after)
            {
                continue;
            }
            let cluster = self.cluster();
            let body = json!({
                "seconds": failover.lease.as_secs_f64(),
                "holder": failover.holder,
                "sent_unix_ms": unix_ms(),
                "probe": probe,
            });
            let answers =
                futures::future::join_all(cluster.nodes.iter().filter(|node| node.is_up()).map(
                    |node| {
                        let body = body.clone();
                        async move {
                            let lease = node.action("lease", body);
                            let answer = tokio::time::timeout(deadline, lease).await;
                            (node, answer)
                        }
                    },
                ))
                .await;
            // Leases last from when nodes took them, so this time is never early.
            let now = Instant::now();
            let mut refused = None;
            {
                let mut leases = failover
                    .leased
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                for (node, answer) in answers {
                    let Ok(Ok(answer)) = answer else { continue };
                    let answer = answer
                        .first()
                        .and_then(|answer| LeaseAnswer::deserialize(answer).ok())
                        .unwrap_or_default();
                    if !answer.granted {
                        let holder = answer.holder.unwrap_or_else(|| "another router".into());
                        refused = Some((node.name.clone(), holder));
                        continue;
                    }
                    if !probe {
                        leases.insert(node.name.clone(), now);
                    }
                    if !answer.enforced && unenforced.insert(node.name.clone()) {
                        tracing::error!(
                            node = node.name,
                            "the node takes writes without a lease, so a primary there that \
                             is cut off could still take writes after failing over; set \
                             require_lease = true on every node behind this router"
                        );
                    }
                }
            }
            match refused {
                Some((node, holder)) => {
                    if stood_down.is_none() {
                        tracing::error!(
                            node,
                            holder,
                            "another router holds this node's write lease, so this router \
                             stands down: it neither fails over nor moves tables, and checks \
                             again every {} seconds; only one router may set failover_seconds",
                            failover.after.as_secs_f64()
                        );
                    }
                    failover.contested.store(true, Ordering::Relaxed);
                    stood_down = Some(Instant::now());
                }
                None if probe => {
                    tracing::info!(
                        "no other router holds a write lease, so this router takes over"
                    );
                    stood_down = None;
                    failover.contested.store(false, Ordering::Relaxed);
                    // Takes the leases at the next tick.
                    ticks.reset_immediately();
                }
                None => {}
            }
        }
    }
}

/// Up to `interval` more, which differs between routers and calls.
fn jitter(interval: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.subsec_nanos());
    interval.mul_f64(f64::from(nanos) / 1e9)
}
