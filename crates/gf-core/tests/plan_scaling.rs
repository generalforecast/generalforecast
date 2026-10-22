// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Planning a step costs the same however many rows a table holds.
//!
//! A table shaped like one of the Radar workload's: a classifier over 256
//! features, a balanced core and neighbors, regions of 16384 rows rebuilt at
//! a tenth of churn, and steps of one-row predicts each followed by a one-row
//! insert.

use std::num::{NonZeroU64, NonZeroUsize};
use std::time::{Duration, Instant};

use gf_core::cache::{ModelId, TierConfig};
use gf_core::categories::CategoryEncoding;
use gf_core::node::{
    DEFAULT_FIT_ROWS, EntrySizes, Node, PastCap, PlanParts, RefitPolicy, SummaryScope, TableId,
    TableSpec, Targets, Task,
};
use gf_core::planner::{Core, CoreKind, Strategy};
use gf_core::regions::{Rebuild, RebuildWait, RegionSpec};

const FEATURES: usize = 256;
const PER_STEP: usize = 8;

/// A deterministic stream of floats in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    fn row(&mut self) -> Vec<f32> {
        (0..FEATURES).map(|_| self.next()).collect()
    }
}

fn radar_spec() -> TableSpec {
    TableSpec {
        model: ModelId(0),
        features: FEATURES,
        retrieval: 0,
        categorical: Vec::new(),
        categories: CategoryEncoding::Codes,
        unlabeled_context: false,
        task: Task::Classes(2),
        page_size: NonZeroUsize::new(128).unwrap(),
        strategy: Strategy::Retrieved {
            core: Some(Core {
                rows: 512,
                seed: 7,
                kind: CoreKind::Balanced,
            }),
            neighbors: 512,
            among: None,
            recency: None,
        },
        sizes: EntrySizes {
            bytes_per_row: 2048,
            bytes_per_representation: 1024,
            bucket_minimum: NonZeroUsize::new(8).unwrap(),
            state_bytes: 0,
            state_rows: usize::MAX,
            fused_rows: usize::MAX,
            fit_bound: usize::MAX,
        },
        time_bucket: None,
        refit: RefitPolicy::default(),
        fit_rows: NonZeroUsize::new(DEFAULT_FIT_ROWS).unwrap(),
        context_rows: None,
        past_cap: PastCap::Warn,
        regions: Some(RegionSpec {
            rows: 16384,
            rebuild: Rebuild::Churn(0.1),
            wait: RebuildWait::default(),
        }),
        scale_rows: None,
        summary: SummaryScope::Table,
        series: None,
    }
}

/// A node with a Radar-shaped table of `rows` rows, a fiftieth disputed.
fn radar_table(rows: u64) -> (Node, TableId, Lcg) {
    let tier = |slots| TierConfig {
        slot_bytes: NonZeroU64::new(1 << 20).unwrap(),
        slots,
    };
    let mut node = Node::new(tier(4096), tier(8192)).unwrap();
    node.set_timers(true);
    let table = node.create_table(radar_spec()).unwrap();
    let mut random = Lcg(rows);
    let keys: Vec<u128> = (0..u128::from(rows)).collect();
    let features: Vec<f32> = (0..rows).flat_map(|_| random.row()).collect();
    let labels = (0..rows).map(|row| i32::from(row % 50 == 0)).collect();
    node.insert(table, &keys, &features, Targets::Classes(labels))
        .unwrap();
    (node, table, random)
}

/// Time the planner spent in each part over `steps` steps of `PER_STEP`
/// one-row predicts, each followed by a one-row insert, after the same number
/// of steps to warm up; a step's plan time is every `schedule` it runs,
/// including the rebuild steps it runs with no request waiting.
fn plan_per_step(rows: u64, steps: usize) -> (PlanParts, Duration, Duration) {
    let (mut node, table, mut random) = radar_table(rows);
    let mut next_key = u128::from(rows);
    let mut parts = PlanParts::default();
    let mut planned = Duration::ZERO;
    let mut written = Duration::ZERO;
    for step in 0..steps * 2 {
        let measured = step >= steps;
        for _ in 0..PER_STEP {
            node.submit(table, &random.row()).unwrap();
        }
        loop {
            let since = Instant::now();
            let scheduled = node.schedule().unwrap();
            let took = since.elapsed();
            let Some(step) = scheduled else { break };
            if measured {
                planned += took;
                parts.extend(node.plan_parts());
            }
            node.complete(&step);
        }
        node.take_drops();
        node.take_refused();
        node.take_advanced();
        node.take_past_cap();
        let keys: Vec<u128> = (0..PER_STEP as u128).map(|at| next_key + at).collect();
        next_key += PER_STEP as u128;
        let features: Vec<f32> = (0..PER_STEP).flat_map(|_| random.row()).collect();
        let since = Instant::now();
        node.insert(table, &keys, &features, Targets::Classes(vec![0; PER_STEP]))
            .unwrap();
        if measured {
            written += since.elapsed();
        }
    }
    (parts, planned / steps as u32, written / steps as u32)
}

/// Prints the planner's time by part at each of the sweep's table sizes.
/// Run with `cargo test -p gf-core --release --test plan_scaling -- --ignored --nocapture`.
#[test]
#[ignore = "prints a sweep"]
fn radar_plan_time_by_part_and_size() {
    let steps = 100;
    for rows in [3_000, 12_000, 50_000] {
        let (parts, planned, written) = plan_per_step(rows, steps);
        println!(
            "rows {rows}: plan {:.3} ms a step, write {:.3} ms a step",
            planned.as_secs_f64() * 1e3,
            written.as_secs_f64() * 1e3
        );
        for (part, calls, seconds) in parts.iter() {
            println!(
                "  {part:<9} {:>8.3} ms a step  {:>6.2} calls a step",
                seconds.as_secs_f64() * 1e3 / steps as f64,
                calls as f64 / steps as f64
            );
        }
    }
}
