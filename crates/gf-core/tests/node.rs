// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A node: many tables, one queue, one cache.

use std::collections::HashSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use gf_core::cache::{Kind, ModelId, TierConfig};
use gf_core::categories::CategoryEncoding;
use gf_core::node::{
    Action, Compaction, CutoffRefit, DEFAULT_FIT_ROWS, EntrySizes, Node, NodeError, Page, PastCap,
    RefitPolicy, RequestId, RowLayout, Scheduling, Step, SummaryScope, TableId, TableRows,
    TableSpec, Targets, Task, UNLABELED,
};
use gf_core::pages::PageId;
use gf_core::planner::{CoreKind, Recency, Strategy, StrategyError};
use gf_core::regions::{Rebuild, RebuildWait, RegionSpec};
use gf_core::rows::{RowMemory, Sealer};
use gf_core::table::{RowId, RowKey, TableError};

const FEATURES: usize = 2;

fn node(device_slots: u32) -> Node {
    node_with_host(device_slots, 64)
}

fn node_with_host(device_slots: u32, host_slots: u32) -> Node {
    let tier = |slots| TierConfig {
        slot_bytes: NonZeroU64::new(80).unwrap(),
        slots,
    };
    Node::new(tier(device_slots), tier(host_slots)).unwrap()
}

fn spec(strategy: Strategy) -> TableSpec {
    TableSpec {
        model: ModelId(0),
        features: FEATURES,
        retrieval: 0,
        categorical: Vec::new(),
        categories: CategoryEncoding::Codes,
        unlabeled_context: false,
        task: Task::Classes(3),
        page_size: NonZeroUsize::new(8).unwrap(),
        strategy,
        sizes: EntrySizes {
            bytes_per_row: 10,
            bytes_per_representation: 10,
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
        regions: None,
        scale_rows: None,
        summary: SummaryScope::Table,
        series: None,
    }
}

/// Rows 0..count: even keys around (0, 0), odd keys around (100, 100).
fn rows(count: u64) -> (Vec<u128>, Vec<f32>, Targets) {
    let keys: Vec<u128> = (0..u128::from(count)).collect();
    let features = keys
        .iter()
        .flat_map(|&key| {
            let offset = (key / 2) as f32 * 0.01;
            if key.is_multiple_of(2) {
                [offset, 0.0]
            } else {
                [100.0 + offset, 100.0]
            }
        })
        .collect();
    let labels = keys.iter().map(|&key| (key % 3) as i32).collect();
    (keys, features, Targets::Classes(labels))
}

fn table(node: &mut Node, strategy: Strategy, count: u64) -> TableId {
    let table = node.create_table(spec(strategy)).unwrap();
    let (keys, features, targets) = rows(count);
    node.insert(table, &keys, &features, targets).unwrap();
    table
}

fn run(node: &mut Node) -> Step {
    let step = node.schedule().unwrap().expect("requests are pending");
    node.complete(&step);
    step
}

fn actions(step: &Step) -> Vec<Action> {
    step.units()
        .iter()
        .map(|unit| unit.placed.action())
        .collect()
}

/// The pages a step embeds: neither reused nor filled by a fit.
fn embedded(step: &Step) -> Vec<&Page> {
    step.pages()
        .iter()
        .filter(|page| page.placed.action() != Action::Reuse && page.embed.is_some())
        .collect()
}

const A: [f32; 2] = [0.0, 0.0];
const B: [f32; 2] = [100.0, 100.0];

#[test]
fn the_first_step_fits_the_table_and_later_steps_reuse_its_contexts() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 32);

    node.submit(table, &A).unwrap();
    let first = run(&mut node);
    node.submit(table, &A).unwrap();
    let second = run(&mut node);

    assert_eq!(first.fits().len(), 1);
    assert_eq!(first.fits()[0].rows.len(), 32);
    assert_eq!(first.fits()[0].features.len(), 32 * FEATURES);
    assert!(embedded(&first).is_empty());
    assert_eq!(actions(&first), [Action::Encode]);
    assert!(second.fits().is_empty() && second.pages().is_empty());
    assert_eq!(actions(&second), [Action::Reuse]);
}

#[test]
fn a_failed_step_frees_its_entries_and_fit_for_the_next() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 32);

    node.submit(table, &A).unwrap();
    let failed = node.schedule().unwrap().unwrap();
    let dropped = node.fail(&failed);
    node.submit(table, &A).unwrap();
    let retried = run(&mut node);

    let encoded: HashSet<_> = failed
        .units()
        .iter()
        .map(|unit| (unit.key, unit.placed.action()))
        .chain(
            failed
                .pages()
                .iter()
                .map(|page| (page.key, page.placed.action())),
        )
        .filter(|&(_, action)| action == Action::Encode)
        .map(|(key, _)| key)
        .collect();
    assert_eq!(
        dropped.iter().map(|drop| drop.key).collect::<HashSet<_>>(),
        encoded
    );
    assert_eq!(retried.fits().len(), 1);
    assert_eq!(actions(&retried), [Action::Encode]);
}

#[test]
fn a_query_before_every_row_is_refused_alone() {
    let mut node = node(64);
    let table = node.create_table(spec(Strategy::Exact)).unwrap();
    let (keys, features, targets) = rows(16);
    let times: Vec<Option<i64>> = (10..26).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();

    let early = node.submit_before(table, &A, Some(&[10])).unwrap();
    let late = node.submit_before(table, &A, Some(&[11])).unwrap();
    let step = run(&mut node);

    assert_eq!(
        node.take_refused(),
        [(early, NodeError::EmptyContext(table))]
    );
    assert_eq!(step.queries().len(), 1);
    assert_eq!(step.queries()[0].request, late);
}

#[test]
fn a_fit_fills_its_tables_representation_pages() {
    let mut node = node(64);
    let table = table(&mut node, NEAREST, 32);
    node.submit(table, &A).unwrap();

    let step = run(&mut node);

    let fit = &step.fits()[0];
    let filled: Vec<RowId> = fit
        .pages
        .iter()
        .flat_map(|&page| step.page(page).rows.clone())
        .collect();
    assert_eq!(filled, fit.rows);
    assert!(step.pages().iter().all(|page| page.embed.is_none()));
    assert!(
        step.pages()
            .iter()
            .all(|page| page.placed.action() == Action::Encode)
    );
    assert!(step.pages().iter().all(|page| page.rows.len() <= 8));
}

#[test]
fn contexts_take_whole_slots_of_rows() {
    let mut node = node(64);
    let mut wide = spec(Strategy::Exact);
    // 30 bytes a row in 80-byte slots: two rows a slot, so 8 rows take 4 slots.
    wide.sizes.bytes_per_row = 30;
    let table = node.create_table(wide).unwrap();
    let (keys, features, targets) = rows(8);
    node.insert(table, &keys, &features, targets).unwrap();
    node.submit(table, &A).unwrap();

    let step = run(&mut node);

    assert_eq!(step.units()[0].placed.slots().len(), 4);
}

#[test]
fn a_step_stops_at_its_byte_budget_and_the_rest_waits() {
    let mut node = node(64);
    // Each table's context is 16 rows of 10 bytes: two 80-byte slots.
    node.set_step_bytes(160);
    let first = table(&mut node, Strategy::Exact, 16);
    let second = table(&mut node, Strategy::Exact, 16);
    node.submit(first, &A).unwrap();
    node.submit(second, &A).unwrap();

    let one = run(&mut node);
    let two = run(&mut node);

    assert_eq!(one.queries().len(), 1);
    assert_eq!(two.queries().len(), 1);
    assert_ne!(one.queries()[0].table, two.queries()[0].table);
    assert!(node.schedule().unwrap().is_none());
}

#[test]
fn cached_contexts_cost_nothing_against_the_budget() {
    let mut node = node(64);
    node.set_step_bytes(160);
    let first = table(&mut node, Strategy::Exact, 16);
    let second = table(&mut node, Strategy::Exact, 16);
    node.submit(first, &A).unwrap();
    run(&mut node);
    node.submit(second, &A).unwrap();
    run(&mut node);

    node.submit(first, &A).unwrap();
    node.submit(second, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(step.queries().len(), 2);
    assert_eq!(actions(&step), [Action::Reuse, Action::Reuse]);
}

#[test]
fn without_a_device_tier_every_table_joins_one_step() {
    let mut node = node(0);
    let first = table(&mut node, Strategy::Exact, 16);
    let second = table(&mut node, Strategy::Exact, 16);
    node.submit(first, &A).unwrap();
    node.submit(second, &A).unwrap();

    let step = run(&mut node);

    assert_eq!(step.queries().len(), 2);
    assert_eq!(actions(&step), [Action::Transient, Action::Transient]);
}

#[test]
fn arrival_scheduling_serves_the_longest_waiting_table_first() {
    for (scheduling, expected) in [
        (Scheduling::Turns, [0, 1, 2]),
        (Scheduling::Arrival, [2, 0, 1]),
    ] {
        let mut node = node(64);
        // Each table's context is two 80-byte slots, so a step takes one table.
        node.set_step_bytes(160);
        node.set_scheduling(scheduling);
        let tables: Vec<TableId> = (0..3)
            .map(|_| table(&mut node, Strategy::Exact, 16))
            .collect();
        for index in [2, 0, 1] {
            node.submit(tables[index], &A).unwrap();
        }

        let served: Vec<TableId> = (0..3).map(|_| run(&mut node).queries()[0].table).collect();

        assert_eq!(
            served,
            expected.map(|index| tables[index]),
            "{scheduling:?}"
        );
    }
}

#[test]
fn a_request_larger_than_the_budget_still_runs_alone() {
    let mut node = node(64);
    node.set_step_bytes(80);
    let table = table(&mut node, Strategy::Exact, 32);
    node.submit(table, &A).unwrap();

    let step = run(&mut node);

    assert_eq!(step.queries().len(), 1);
    assert_eq!(actions(&step), [Action::Encode]);
}

#[test]
fn tables_take_turns_when_the_budget_binds() {
    let mut node = node(2);
    node.set_step_bytes(160);
    let tables: Vec<TableId> = (0..3).map(|_| table(&mut node, NEAREST, 32)).collect();
    let mut served = Vec::new();
    for _ in 0..6 {
        for &table in &tables {
            node.submit(table, &[A, B].concat()).unwrap();
        }
        let step = run(&mut node);
        served.push(step.queries()[0].table);
    }

    // With room for one table a step, each table is served every third step.
    assert_eq!(served[..3].iter().collect::<HashSet<_>>().len(), 3);
    assert_eq!(served[3..], served[..3]);
}

#[test]
fn a_plan_left_waiting_is_made_again_once_its_table_changes() {
    let mut node = node(64);
    // Room for the small table's context, one slot, but not the large one's four.
    node.set_step_bytes(160);
    let small = table(&mut node, Strategy::Exact, 8);
    let large = table(&mut node, Strategy::Exact, 32);
    node.submit(small, &A).unwrap();
    node.submit(large, &A).unwrap();
    let first = run(&mut node);
    assert!(first.queries().iter().all(|batch| batch.table == small));

    // Rows of the table that waited change before its turn.
    node.delete(large, &[0, 2]).unwrap();
    let later = run(&mut node);

    assert_eq!(later.queries()[0].table, large);
    let read = &later.units()[0].rows;
    assert_eq!(read.len(), 30);
    assert!(!read.contains(&RowId(0)) && !read.contains(&RowId(2)));
}

#[test]
fn a_table_whose_rows_mostly_left_is_renumbered_before_its_next_insert() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 5_000);
    let gone: Vec<u128> = (10..5_000).collect();
    node.delete(table, &gone).unwrap();

    let (_, features, targets) = rows(1);
    let added = node.insert(table, &[100_000], &features, targets).unwrap();

    // The ten rows left took ids 0 to 9, so the new row takes the next.
    assert_eq!(added, [RowId(10)]);
    assert_eq!(node.len(table).unwrap(), 11);
    node.submit(table, &A).unwrap();
    assert_eq!(run(&mut node).units()[0].rows.len(), 11);
}

#[test]
fn nothing_is_scheduled_without_requests() {
    let mut node = node(64);
    table(&mut node, Strategy::Exact, 8);

    assert!(node.schedule().unwrap().is_none());
}

#[test]
fn one_step_serves_every_table_with_requests() {
    let mut node = node(64);
    let first = table(&mut node, Strategy::Exact, 16);
    let second = table(&mut node, Strategy::Exact, 16);

    let one = node.submit(first, &A).unwrap();
    let two = node.submit(second, &A).unwrap();
    let step = run(&mut node);

    let tables: HashSet<TableId> = step.units().iter().map(|unit| unit.table).collect();
    assert_eq!(tables, HashSet::from([first, second]));
    assert_eq!(step.fits().len(), 2);
    // Identical rows in two tables are still two cache entries.
    assert_ne!(step.units()[0].key, step.units()[1].key);
    let requests: Vec<_> = step
        .queries()
        .iter()
        .map(|batch| (batch.request, batch.table))
        .collect();
    assert_eq!(requests, [(one, first), (two, second)]);
}

#[test]
fn queued_requests_for_a_table_share_its_plan() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::neighbors(16), 32);

    node.submit(table, &[A, B].concat()).unwrap();
    node.submit(table, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(step.units().len(), 2);
    assert_eq!(step.queries().len(), 2);
    assert_eq!(step.queries()[0].units.len(), 2);
    assert_eq!(step.queries()[0].features.len(), 2 * FEATURES);
    // The second request's query reads the same unit as the first request's first query.
    assert_eq!(step.queries()[1].units[0], step.queries()[0].units[0]);
}

#[test]
fn a_changed_row_re_embeds_only_its_page() {
    let mut node = node(64);
    let table = table(&mut node, NEAREST, 32);
    node.submit(table, &[A, B].concat()).unwrap();
    run(&mut node);

    let updated = node
        .update(table, &[0], &[0.0, 0.0], Targets::Classes(vec![1]))
        .unwrap();
    node.submit(table, &[A, B].concat()).unwrap();
    let step = run(&mut node);

    assert!(step.fits().is_empty());
    assert_eq!(actions(&step), [Action::Encode, Action::Reuse]);
    let embedded = embedded(&step);
    assert_eq!(embedded.len(), 1);
    assert!(embedded[0].rows.contains(&updated[0]));
    // The changed row alone embeds; the page's earlier entry gives the rest.
    let (features, targets) = embedded[0].embed.as_ref().unwrap();
    assert_eq!(features.len(), FEATURES);
    assert_eq!(targets.len(), 1);
    let copied: usize = embedded[0]
        .copies
        .iter()
        .map(|(_, positions)| positions.len())
        .sum();
    assert_eq!(copied, embedded[0].rows.len() - 1);
    let reused = step
        .pages()
        .iter()
        .filter(|page| page.placed.action() == Action::Reuse);
    assert!(reused.count() >= 1);
    let encoded: HashSet<RowId> = step.units()[0].rows.iter().copied().collect();
    let covered: HashSet<RowId> = step
        .pages()
        .iter()
        .flat_map(|page| page.rows.clone())
        .collect();
    assert!(encoded.is_subset(&covered));
}

#[test]
fn representation_pages_compete_for_the_host_tier() {
    let mut node = node_with_host(64, 1);
    let first = table(&mut node, Strategy::Exact, 8);
    let second = table(&mut node, Strategy::Exact, 8);
    node.submit(first, &A).unwrap();
    let kept = run(&mut node);
    node.submit(second, &A).unwrap();

    let step = run(&mut node);

    assert_eq!(kept.pages().len(), 1);
    assert!(
        step.movements()
            .iter()
            .any(|movement| movement.key == kept.pages()[0].key && movement.to.is_none())
    );
}

#[test]
fn pages_without_room_are_embedded_when_next_needed() {
    let mut node = node_with_host(64, 2);
    let table = table(&mut node, Strategy::Exact, 32);
    node.submit(table, &A).unwrap();
    let fitted = run(&mut node);
    let rows_of = |action: Action| -> Vec<Vec<RowId>> {
        fitted
            .pages()
            .iter()
            .filter(|page| page.placed.action() == action)
            .map(|page| page.rows.clone())
            .collect()
    };
    let (kept, without_room) = (rows_of(Action::Encode), rows_of(Action::Transient));

    // Same features, new label: the row stays in its page, which changes.
    let changed = kept[0][0].0 as usize;
    let (_, features, _) = rows(32);
    let same = &features[changed * FEATURES..(changed + 1) * FEATURES];
    node.update(table, &[changed as u128], same, Targets::Classes(vec![2]))
        .unwrap();
    node.submit(table, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(kept.len(), 2);
    assert!(!without_room.is_empty());
    let sorted = |mut pages: Vec<Vec<RowId>>| {
        pages.sort();
        pages
    };
    let embedded: Vec<Vec<RowId>> = embedded(&step)
        .iter()
        .map(|page| page.rows.clone())
        .collect();
    let reused: Vec<Vec<RowId>> = step
        .pages()
        .iter()
        .filter(|page| page.placed.action() == Action::Reuse)
        .map(|page| page.rows.clone())
        .collect();
    let mut expected = without_room.clone();
    expected.push(kept[0].clone());
    assert_eq!(sorted(embedded), sorted(expected));
    assert_eq!(sorted(reused), sorted(kept[1..].to_vec()));
}

#[test]
fn retrieval_scaling_is_fit_again_as_rows_double_and_at_refits() {
    let strategy = Strategy::neighbors(8);
    // The second column spreads far less, which only scaling on every row sees.
    let (keys, mut features, targets) = rows(32);
    for (index, value) in features.iter_mut().enumerate() {
        *value = if index % 2 == 0 {
            *value
        } else {
            (index / 2 % 7) as f32 * 0.01
        };
    }
    let mut node = node(64);
    let (grown, whole) = (
        node.create_table(spec(strategy)).unwrap(),
        node.create_table(spec(strategy)).unwrap(),
    );
    node.insert(
        grown,
        &keys[..1],
        &features[..FEATURES],
        targets_of(&targets, 0..1),
    )
    .unwrap();
    node.insert(
        grown,
        &keys[1..],
        &features[FEATURES..],
        targets_of(&targets, 1..32),
    )
    .unwrap();
    let all_targets = targets.clone();
    node.insert(whole, &keys, &features, targets).unwrap();

    let read = |node: &mut Node| {
        for table in [grown, whole] {
            node.submit(table, &[50.0, 0.03]).unwrap();
        }
        let step = run(node);
        [grown, whole].map(|table| -> Vec<Vec<RowId>> {
            step.units()
                .iter()
                .filter(|unit| unit.table == table)
                .map(|unit| unit.rows.clone())
                .collect()
        })
    };
    // The first batch, one row, set the scaling until the rows doubled.
    let [doubled, all] = read(&mut node);
    assert_eq!(doubled, all);
    node.refit(grown).unwrap();
    let [refit, all] = read(&mut node);
    assert_eq!(refit, all);

    // A scaling frozen at its first row stays, renumbered or not.
    let frozen = node
        .create_table(TableSpec {
            scale_rows: NonZeroU64::new(1),
            ..spec(strategy)
        })
        .unwrap();
    for range in [0..1, 1..32] {
        node.insert(
            frozen,
            &keys[range.clone()],
            &features[range.start * FEATURES..range.end * FEATURES],
            targets_of(&all_targets, range),
        )
        .unwrap();
    }
    node.renumber(frozen).unwrap();
    node.submit(frozen, &[50.0, 0.03]).unwrap();
    let step = run(&mut node);
    assert_ne!(step.units()[0].rows, all[0]);
}

#[test]
fn copies_with_the_same_rows_plan_alike_whatever_wrote_them() {
    for strategy in [
        Strategy::subsample(8, 3, CoreKind::Random),
        Strategy::neighbors(16),
    ] {
        let mut node = node(64);
        // One table takes a history of deletes, updates and inserts.
        let written = table(&mut node, strategy, 64);
        let doomed: Vec<u128> = (0..64).filter(|key| key % 3 == 0).collect();
        node.delete(written, &doomed).unwrap();
        let (_, features, _) = rows(80);
        node.update(
            written,
            &[1, 2],
            &features[20..24],
            Targets::Classes(vec![0, 0]),
        )
        .unwrap();
        node.insert(
            written,
            &(64..80).collect::<Vec<_>>(),
            &features[128..160],
            Targets::Classes(vec![1; 16]),
        )
        .unwrap();
        // The other is built from a snapshot of its rows, as a replica is.
        let rows = node.rows_with_times(written).unwrap();
        let copied = node.create_table(spec(strategy)).unwrap();
        node.replace_rows(
            copied,
            &rows.keys,
            &rows.features,
            rows.targets,
            Some(&rows.times),
        )
        .unwrap();
        node.renumber(written).unwrap();

        for table in [written, copied] {
            node.submit(table, &[A, B].concat()).unwrap();
        }
        let step = run(&mut node);
        let plan = |table: TableId| -> Vec<_> {
            step.units()
                .iter()
                .filter(|unit| unit.table == table)
                .map(|unit| (unit.key.rows, unit.rows.clone()))
                .collect()
        };
        assert_eq!(plan(written), plan(copied), "{strategy:?}");
    }
}

#[test]
fn an_entry_a_running_step_moves_is_read_only_once_it_completes() {
    let strategy = Strategy::neighbors(16);
    let mut node = node(2);
    let (first, second) = (
        table(&mut node, strategy, 32),
        table(&mut node, strategy, 32),
    );
    node.submit(first, &A).unwrap();
    run(&mut node);
    node.submit(second, &A).unwrap();
    run(&mut node);

    // The first table's context moves back to the device in a step in flight.
    node.submit(first, &A).unwrap();
    let moving = node.schedule().unwrap().unwrap();
    node.submit(first, &A).unwrap();
    let next = node.schedule().unwrap().unwrap();

    assert!(
        moving
            .movements()
            .iter()
            .any(|movement| { movement.to.is_some() && movement.key == moving.units()[0].key })
    );
    // Its slots may not hold it until the move is made, so the next step
    // computes it for itself.
    assert_eq!(actions(&next), [Action::Transient]);
    node.complete(&moving);
    node.complete(&next);
    node.submit(first, &A).unwrap();
    assert_eq!(actions(&run(&mut node)), [Action::Reuse]);
}

#[test]
fn a_step_that_completes_after_the_cache_is_forgotten_touches_no_new_entry() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 32);
    node.submit(table, &A).unwrap();
    let late = node.schedule().unwrap().unwrap();

    node.forget_cache();
    node.submit(table, &A).unwrap();
    let fresh = node.schedule().unwrap().unwrap();

    // The same rows encode under a new epoch, so the late step's release
    // cannot unpin what the fresh step reads.
    assert_ne!(late.units()[0].key, fresh.units()[0].key);
    let dropped = node.complete(&late);
    assert!(dropped.iter().all(|drop| drop.key != fresh.units()[0].key));
}

#[test]
fn a_table_the_budget_leaves_waiting_keeps_its_cache_until_its_refit() {
    let mut node = node(64);
    // Each table's context is 16 rows of 10 bytes: two 80-byte slots.
    node.set_step_bytes(160);
    let first = table(&mut node, Strategy::Exact, 16);
    let mut watched = spec(Strategy::Exact);
    watched.refit = RefitPolicy::new(Some(0.5), None).unwrap();
    let second = node.create_table(watched).unwrap();
    let (keys, features, targets) = rows(24);
    node.insert(
        second,
        &keys[..16],
        &features[..32],
        targets_of(&targets, 0..16),
    )
    .unwrap();
    node.submit(second, &A).unwrap();
    run(&mut node);
    // Enough new rows that the second table's policy asks for a refit.
    node.insert(
        second,
        &keys[16..],
        &features[32..],
        targets_of(&targets, 16..24),
    )
    .unwrap();

    node.submit(first, &A).unwrap();
    node.submit(second, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(step.queries().len(), 1);
    assert_eq!(step.queries()[0].table, first);
    assert!(
        step.movements()
            .iter()
            .all(|movement| movement.key.table != second)
    );
}

fn targets_of(targets: &Targets, range: std::ops::Range<usize>) -> Targets {
    match targets {
        Targets::Classes(labels) => Targets::Classes(labels[range].to_vec()),
        Targets::Values(values) => Targets::Values(values[range].to_vec()),
    }
}

#[test]
fn a_new_epoch_refits_only_its_table() {
    let mut node = node(64);
    let first = table(&mut node, Strategy::Exact, 16);
    let second = table(&mut node, Strategy::Exact, 16);
    node.submit(first, &A).unwrap();
    node.submit(second, &A).unwrap();
    run(&mut node);

    node.advance_epoch(first).unwrap();
    node.submit(first, &A).unwrap();
    node.submit(second, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(
        step.fits().iter().map(|fit| fit.table).collect::<Vec<_>>(),
        [first]
    );
    let by_table: HashSet<_> = step
        .units()
        .iter()
        .map(|unit| (unit.table, unit.placed.action()))
        .collect();
    assert_eq!(
        by_table,
        HashSet::from([(first, Action::Encode), (second, Action::Reuse)])
    );
}

#[test]
fn tables_compete_for_one_cache() {
    let mut node = node(2);
    let first = table(&mut node, Strategy::neighbors(16), 32);
    let second = table(&mut node, Strategy::neighbors(16), 32);
    node.submit(first, &A).unwrap();
    let kept = run(&mut node);

    node.submit(second, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(actions(&step), [Action::Encode]);
    assert_eq!(step.movements().len(), 1);
    assert_eq!(step.movements()[0].key, kept.units()[0].key);
}

#[test]
fn a_failed_step_forgets_the_entries_it_moved() {
    let strategy = Strategy::neighbors(16);
    let mut node = node(2);
    let (first, second) = (
        table(&mut node, strategy, 32),
        table(&mut node, strategy, 32),
    );
    node.submit(first, &A).unwrap();
    run(&mut node);
    node.submit(second, &A).unwrap();
    run(&mut node);

    // Bringing the first table's context back moves it to the device.
    node.submit(first, &A).unwrap();
    let failed = node.schedule().unwrap().unwrap();
    let dropped = node.fail(&failed);
    node.submit(first, &A).unwrap();
    let retried = run(&mut node);

    assert!(
        failed
            .movements()
            .iter()
            .any(|movement| movement.to.is_some())
    );
    assert!(dropped.iter().any(|drop| drop.key == failed.units()[0].key));
    assert_eq!(actions(&retried), [Action::Encode]);
}

#[test]
fn removing_a_table_frees_its_cache_entries() {
    let mut node = node(64);
    let removed = table(&mut node, Strategy::Exact, 4);
    let kept = table(&mut node, Strategy::Exact, 4);
    node.submit(removed, &A).unwrap();
    node.submit(kept, &A).unwrap();
    assert_eq!(run(&mut node).units().len(), 2);

    let drops = node.remove_table(removed).unwrap();

    assert_eq!(drops.len(), 2);
    assert!(drops.iter().all(|movement| movement.key.table == removed));
    assert!(
        drops
            .iter()
            .any(|movement| movement.key.kind == Kind::Context)
    );
    assert!(drops.iter().any(|movement| movement.key.kind == Kind::Rows));
    assert!(drops.iter().all(|movement| movement.to.is_none()));
    assert!(matches!(
        node.submit(removed, &A),
        Err(NodeError::UnknownTable(_))
    ));
    node.submit(kept, &A).unwrap();
    assert_eq!(actions(&run(&mut node)), [Action::Reuse]);
}

#[test]
fn bad_batches_change_nothing() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 4);

    let short = node.insert(
        table,
        &[10, 11],
        &[1.0, 2.0, 3.0],
        Targets::Classes(vec![0, 1]),
    );
    let duplicate = node.insert(table, &[10, 10], &[1.0; 4], Targets::Classes(vec![0, 1]));
    let existing = node.insert(table, &[3], &[1.0; 2], Targets::Classes(vec![0]));
    let label = node.insert(table, &[10], &[1.0; 2], Targets::Classes(vec![3]));
    let task = node.insert(table, &[10], &[1.0; 2], Targets::Values(vec![0.5]));
    let missing = node.delete(table, &[1, 99]);

    assert!(matches!(short, Err(NodeError::Count { .. })));
    assert!(matches!(duplicate, Err(NodeError::DuplicateKey(_))));
    assert!(matches!(
        existing,
        Err(NodeError::Table(TableError::DuplicateKey(_)))
    ));
    assert!(existing.unwrap_err().to_string().contains("already exists"));
    assert!(matches!(label, Err(NodeError::Label { .. })));
    assert!(matches!(task, Err(NodeError::Task)));
    assert!(matches!(missing, Err(NodeError::UnknownKey(_))));
    assert_eq!(node.len(table).unwrap(), 4);
    assert!(matches!(
        node.submit(TableId(9), &A),
        Err(NodeError::UnknownTable(_))
    ));
    let empty = node.create_table(spec(Strategy::Exact)).unwrap();
    assert!(matches!(
        node.submit(empty, &A),
        Err(NodeError::EmptyTable(_))
    ));
}

const NEAREST: Strategy = Strategy::neighbors(16);

#[test]
fn an_update_keeps_every_context_without_the_row() {
    let mut node = node(64);
    let table = table(&mut node, NEAREST, 32);
    node.submit(table, &[A, B].concat()).unwrap();
    run(&mut node);

    node.update(table, &[0], &[0.0, 0.0], Targets::Classes(vec![1]))
        .unwrap();
    node.submit(table, &[A, B].concat()).unwrap();
    let step = run(&mut node);

    assert_eq!(actions(&step), [Action::Encode, Action::Reuse]);
}

#[test]
fn deleted_rows_leave_every_context() {
    let mut node = node(64);
    let table = table(&mut node, NEAREST, 32);

    let deleted = node.delete(table, &[4]).unwrap();
    node.submit(table, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(node.len(table).unwrap(), 31);
    assert!(
        step.units()
            .iter()
            .all(|unit| !unit.rows.contains(&deleted[0]))
    );
    assert!(step.fits()[0].rows.iter().all(|row| *row != deleted[0]));
}

#[test]
fn a_running_step_keeps_its_epoch_until_it_completes() {
    let mut node = node(64);
    let table = table(&mut node, NEAREST, 32);
    node.submit(table, &A).unwrap();
    let running = node.schedule().unwrap().unwrap();

    let dropped_now = node.advance_epoch(table).unwrap();
    let dropped_later = node.complete(&running);

    assert!(dropped_now.is_empty());
    let read: HashSet<_> = running
        .units()
        .iter()
        .map(|unit| unit.key)
        .chain(running.pages().iter().map(|page| page.key))
        .collect();
    let dropped: HashSet<_> = dropped_later.iter().map(|movement| movement.key).collect();
    assert_eq!(dropped, read);
}

/// Keys of entries the node dropped, from purges since the last call.
fn purged(node: &mut Node) -> HashSet<gf_core::cache::CacheKey> {
    node.take_drops().into_iter().map(|drop| drop.key).collect()
}

#[test]
fn changing_a_row_purges_every_cached_entry_holding_it() {
    let mut node = node(64);
    let table = table(&mut node, NEAREST, 32);
    node.submit(table, &[A, B].concat()).unwrap();
    let step = run(&mut node);
    let near_a = step.units()[0].key;
    let near_b = step.units()[1].key;

    // Row 0 is near A.
    node.delete(table, &[0]).unwrap();
    let dropped = purged(&mut node);

    assert!(dropped.contains(&near_a));
    assert!(!dropped.contains(&near_b));
    // The page holding row 0 stays, as a source for its next version, and
    // goes once that version is kept.
    let pages: Vec<_> = step
        .pages()
        .iter()
        .filter(|page| page.rows.contains(&RowId(0)))
        .map(|page| page.key)
        .collect();
    assert!(!pages.is_empty() && pages.iter().all(|key| !dropped.contains(key)));
    assert!(purged(&mut node).is_empty());
    node.submit(table, &A).unwrap();
    let step = node.schedule().unwrap().unwrap();
    let sources: Vec<_> = step
        .pages()
        .iter()
        .flat_map(|page| page.copies.iter().map(|(source, _)| *source))
        .collect();
    assert_eq!(sources, pages);
    let dropped: HashSet<_> = node
        .complete(&step)
        .into_iter()
        .map(|drop| drop.key)
        .collect();
    assert!(pages.iter().all(|key| dropped.contains(key)));
}

#[test]
fn updates_purge_like_deletes() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 16);
    node.submit(table, &A).unwrap();
    let step = run(&mut node);

    node.update(table, &[3], &[0.5, 0.5], Targets::Classes(vec![1]))
        .unwrap();

    assert!(purged(&mut node).contains(&step.units()[0].key));
}

#[test]
fn entries_a_running_step_reads_are_purged_when_it_completes() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 16);
    node.submit(table, &A).unwrap();
    let running = node.schedule().unwrap().unwrap();

    node.delete(table, &[3]).unwrap();
    let now = purged(&mut node);
    let later: HashSet<_> = node
        .complete(&running)
        .into_iter()
        .map(|drop| drop.key)
        .collect();

    assert!(!now.contains(&running.units()[0].key));
    assert!(later.contains(&running.units()[0].key));
}

#[test]
fn replacing_rows_changes_only_what_differs() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 4);
    node.submit(table, &A).unwrap();
    run(&mut node);
    let (keys, features, _) = rows(4);

    // Keep 0 and 1, change 2's label, drop 3, add 9.
    let mut wanted = features[..3 * FEATURES].to_vec();
    wanted.extend_from_slice(&[7.0, 7.0]);
    node.replace_rows(
        table,
        &[0, 1, 2, 9],
        &wanted,
        Targets::Classes(vec![0, 1, 0, 2]),
        None,
    )
    .unwrap();

    let (now_keys, now_features, now_targets) = node.rows(table).unwrap();
    assert_eq!(now_keys, [0, 1, 2, 9]);
    assert_eq!(now_features, wanted);
    assert_eq!(now_targets, Targets::Classes(vec![0, 1, 0, 2]));
    assert_eq!(keys.len(), 4);
    assert!(!purged(&mut node).is_empty());
}

#[test]
fn upserts_replace_known_keys_and_add_the_rest() {
    let mut node = node(64);
    let table = node.create_table(spec(Strategy::Exact)).unwrap();
    let (keys, features, targets) = rows(4);
    node.insert(table, &keys, &features, targets).unwrap();

    node.upsert_at(
        table,
        &[9, 1],
        &[7.0, 7.0, 5.0, 5.0],
        Targets::Classes(vec![2, 2]),
        None,
    )
    .unwrap();

    let (now_keys, now_features, now_targets) = node.rows(table).unwrap();
    assert_eq!(now_keys, [0, 1, 2, 3, 9]);
    assert_eq!(now_features[2..4], [5.0, 5.0]);
    assert_eq!(now_features[8..], [7.0, 7.0]);
    assert_eq!(now_targets, Targets::Classes(vec![0, 2, 2, 0, 2]));
    assert_eq!(node.present(table, &[3, 4, 9]).unwrap(), [3, 9]);
}

#[test]
fn queries_with_a_cutoff_read_only_earlier_rows() {
    let mut node = node(64);
    let table = node.create_table(spec(Strategy::Exact)).unwrap();
    let (keys, features, targets) = rows(16);
    let times: Vec<Option<i64>> = (0..16).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    // A row without a time is visible to every query, and one at the
    // earliest time only to queries after it.
    node.insert(table, &[99], &[0.0, 0.0], Targets::Classes(vec![0]))
        .unwrap();
    let earliest = Some(&[Some(i64::MIN)][..]);
    node.insert_at(
        table,
        &[98],
        &[0.0, 0.0],
        Targets::Classes(vec![0]),
        earliest,
    )
    .unwrap();

    node.submit_before(table, &[A, A, A].concat(), Some(&[i64::MIN, 8, 100]))
        .unwrap();
    let step = run(&mut node);

    let read =
        |query: usize| -> Vec<RowId> { step.unit(step.queries()[0].units[query]).rows.clone() };
    assert_eq!(read(0), [RowId(16)]);
    assert_eq!(read(1).len(), 10);
    assert!(read(1).iter().all(|row| row.0 < 8 || row.0 >= 16));
    assert_eq!(read(2).len(), 18);
    let stored = node.rows_with_times(table).unwrap().times;
    assert_eq!(stored[..16], times[..]);
    // In key order: 98 has the earliest time and 99 none.
    assert_eq!(stored[16..], [Some(i64::MIN), None]);
}

#[test]
fn a_window_keeps_its_cache_through_changes_after_its_cutoff_alone() {
    let mut node = node(64);
    let table = node.create_table(spec(Strategy::Exact)).unwrap();
    let (keys, features, targets) = rows(16);
    let times: Vec<Option<i64>> = (0..16).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    let before = |node: &mut Node| {
        node.submit_before(table, &A, Some(&[8])).unwrap();
        actions(&run(node))
    };
    assert_eq!(before(&mut node), [Action::Encode]);

    // Rows at or after the cutoff change, and the window before it does not.
    node.update(table, &[12], &[1.0, 1.0], Targets::Classes(vec![2]))
        .unwrap();
    node.delete(table, &[13]).unwrap();
    node.insert_at(
        table,
        &[40],
        &[2.0, 2.0],
        Targets::Classes(vec![0]),
        Some(&[Some(20)]),
    )
    .unwrap();
    assert_eq!(before(&mut node), [Action::Reuse]);
    // A row before it changes, and so does the window.
    node.update(table, &[3], &[1.0, 1.0], Targets::Classes(vec![2]))
        .unwrap();
    assert_eq!(before(&mut node), [Action::Encode]);
    // A row moved to before it joins the window.
    node.update_at(
        table,
        &[14],
        &features[28..30],
        Targets::Classes(vec![2]),
        Some(&[Some(2)]),
    )
    .unwrap();
    assert_eq!(before(&mut node), [Action::Encode]);
    assert_eq!(before(&mut node), [Action::Reuse]);
}

#[test]
fn time_buckets_keep_pages_within_one_window() {
    let mut node = node(64);
    let mut bucketed = spec(NEAREST);
    bucketed.time_bucket = NonZeroU64::new(10);
    let table = node.create_table(bucketed).unwrap();
    let (keys, features, targets) = rows(40);
    // Neighboring keys are similar but fall in different ten-unit windows.
    let times: Vec<Option<i64>> = (0..40).map(|key| Some((key % 4) * 10)).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    node.submit(table, &[A, B].concat()).unwrap();

    let step = run(&mut node);

    for page in step.pages() {
        let windows: HashSet<i64> = page
            .rows
            .iter()
            .map(|row| times[row.0 as usize].unwrap() / 10)
            .collect();
        assert_eq!(windows.len(), 1);
    }
}

fn drifting(refit: RefitPolicy) -> (Node, TableId) {
    let mut node = node(64);
    let mut watched = spec(Strategy::Exact);
    watched.refit = refit;
    let table = node.create_table(watched).unwrap();
    let (keys, features, targets) = rows(16);
    node.insert(table, &keys, &features, targets).unwrap();
    node.submit(table, &A).unwrap();
    run(&mut node);
    (node, table)
}

#[test]
fn enough_changed_rows_refit_the_table() {
    let (mut node, table) = drifting(RefitPolicy::new(Some(0.5), None).unwrap());
    let (_, features, _) = rows(24);

    node.insert(
        table,
        &(16..20).collect::<Vec<_>>(),
        &features[32..40],
        Targets::Classes(vec![0; 4]),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    let below = run(&mut node);
    node.insert(
        table,
        &(20..24).collect::<Vec<_>>(),
        &features[40..48],
        Targets::Classes(vec![0; 4]),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    let reached = run(&mut node);

    assert!(below.fits().is_empty());
    assert_eq!(reached.fits().len(), 1);
    assert_eq!(reached.fits()[0].rows.len(), 24);
}

#[test]
fn a_shifted_column_refits_the_table() {
    let (mut node, table) = drifting(RefitPolicy::new(None, Some(1.0)).unwrap());

    // Rows far out along the first feature move its mean by many deviations.
    let far: Vec<f32> = (0..4).flat_map(|_| [1_000.0, 0.0]).collect();
    node.insert(table, &[30, 31, 32, 33], &far, Targets::Classes(vec![0; 4]))
        .unwrap();
    node.submit(table, &A).unwrap();
    let step = run(&mut node);

    assert_eq!(step.fits().len(), 1);
}

#[test]
fn an_entry_two_running_steps_read_outlives_the_first_completing() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 16);
    node.submit(table, &A).unwrap();
    run(&mut node);
    node.submit(table, &A).unwrap();
    let first = node.schedule().unwrap().unwrap();
    node.submit(table, &A).unwrap();
    let second = node.schedule().unwrap().unwrap();
    let context = first.units()[0].key;
    assert_eq!(second.units()[0].key, context);
    assert_eq!(second.units()[0].placed.action(), Action::Reuse);

    node.complete(&first);
    node.delete(table, &[3]).unwrap();
    let now = purged(&mut node);
    let later: HashSet<_> = node
        .complete(&second)
        .into_iter()
        .map(|drop| drop.key)
        .collect();

    assert!(!now.contains(&context));
    assert!(later.contains(&context));
}

#[test]
fn an_ended_epoch_waits_for_every_step_reading_it() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 16);
    node.submit(table, &A).unwrap();
    let warm = run(&mut node);
    node.submit(table, &A).unwrap();
    let first = node.schedule().unwrap().unwrap();
    node.submit(table, &A).unwrap();
    let second = node.schedule().unwrap().unwrap();
    let keys = |drops: Vec<gf_core::cache::Movement>| -> HashSet<_> {
        drops.into_iter().map(|drop| drop.key).collect()
    };

    let advanced = keys(node.advance_epoch(table).unwrap());
    let after_first = keys(node.complete(&first));
    let after_second = keys(node.complete(&second));

    // No running step reads the pages; both read the context.
    let pages: HashSet<_> = warm.pages().iter().map(|page| page.key).collect();
    assert_eq!(advanced, pages);
    assert!(after_first.is_empty());
    assert_eq!(after_second, HashSet::from([warm.units()[0].key]));
}

#[test]
fn requests_wait_for_the_step_encoding_what_they_read() {
    let mut node = node(64);
    let other = table(&mut node, Strategy::Exact, 16);
    let table = table(&mut node, Strategy::Exact, 16);
    node.submit(table, &A).unwrap();
    let first = node.schedule().unwrap().unwrap();
    node.submit(table, &A).unwrap();

    // The context and the fit are in flight, so the table waits.
    assert!(node.schedule().unwrap().is_none());
    node.submit(other, &A).unwrap();
    let apart = node.schedule().unwrap().unwrap();
    assert!(apart.queries().iter().all(|batch| batch.table == other));
    node.complete(&apart);

    // Had the worker failed to encode it, the waiting request encodes it again.
    node.fail(&first);
    let again = node.schedule().unwrap().unwrap();
    assert_eq!(again.units()[0].key, first.units()[0].key);
    assert_eq!(actions(&again), [Action::Encode]);
    assert_eq!(again.fits().len(), 1);
    node.complete(&again);

    node.submit(table, &A).unwrap();
    assert_eq!(actions(&run(&mut node)), [Action::Reuse]);
}

#[test]
fn removing_a_table_refuses_its_requests_and_drops_what_running_steps_read() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 16);
    node.submit(table, &A).unwrap();
    let running = node.schedule().unwrap().unwrap();
    let queued = node.submit(table, &A).unwrap();

    node.remove_table(table).unwrap();
    let refused = node.take_refused();
    let dropped: HashSet<_> = node
        .complete(&running)
        .into_iter()
        .map(|drop| drop.key)
        .collect();

    assert_eq!(refused, [(queued, NodeError::UnknownTable(table))]);
    assert!(dropped.contains(&running.units()[0].key));
    assert!(
        running
            .pages()
            .iter()
            .all(|page| dropped.contains(&page.key))
    );
}

#[test]
fn tables_need_features_and_strategies_need_counts_a_table_can_hold() {
    let mut node = node(64);
    let mut flat = spec(Strategy::Exact);
    flat.features = 0;
    let empty = spec(Strategy::neighbors(0));
    let huge = spec(Strategy::neighbors(usize::MAX));

    assert_eq!(
        node.create_table(huge),
        Err(NodeError::Strategy(StrategyError::TooLarge))
    );

    assert_eq!(node.create_table(flat), Err(NodeError::NoFeatures));
    for (changed, shift) in [(Some(0.0), None), (None, Some(f64::NAN))] {
        assert_eq!(
            RefitPolicy::new(changed, shift),
            Err(NodeError::RefitPolicy)
        );
    }
    assert!(matches!(
        node.create_table(empty),
        Err(NodeError::Strategy(_))
    ));
    assert!(matches!(
        node.len(TableId(0)),
        Err(NodeError::UnknownTable(_))
    ));
}

/// A deterministic stream of numbers.
struct Random(u64);

impl Random {
    fn below(&mut self, limit: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 33) % limit
    }
}

#[test]
fn every_entry_holding_a_changed_row_is_dropped_at_once() {
    let mut random = Random(11);
    let mut node = node(64);
    let table = table(&mut node, NEAREST, 48);
    let mut next_key = 48;
    // Each cached entry's rows, from the steps that encoded them.
    let mut cached: std::collections::HashMap<gf_core::cache::CacheKey, Vec<RowId>> =
        std::collections::HashMap::new();
    for _ in 0..200 {
        match random.below(4) {
            0 => {
                let key = next_key;
                next_key += 1;
                let offset = random.below(100) as f32;
                node.insert(table, &[key], &[offset, offset], Targets::Classes(vec![0]))
                    .unwrap();
            }
            1 | 2 => {
                let live = node.rows(table).unwrap().0;
                let key = live[random.below(live.len() as u64) as usize];
                let rows = if random.below(2) == 0 {
                    node.delete(table, &[key]).unwrap()
                } else {
                    let offset = random.below(100) as f32;
                    node.update(table, &[key], &[offset, offset], Targets::Classes(vec![1]))
                        .unwrap()
                };
                let dropped: HashSet<_> =
                    node.take_drops().into_iter().map(|drop| drop.key).collect();
                // Contexts go at once; a page stays as a source for its
                // next version.
                for (key, held) in &cached {
                    if key.kind == Kind::Context && held.iter().any(|row| rows.contains(row)) {
                        assert!(dropped.contains(key), "{key:?} holds a changed row");
                    }
                }
                cached.retain(|key, _| !dropped.contains(key));
            }
            _ => {
                if node.len(table).unwrap() == 0 {
                    continue;
                }
                let query = [random.below(100) as f32, random.below(100) as f32];
                node.submit(table, &query).unwrap();
                let step = node.schedule().unwrap().unwrap();
                for drop in node.complete(&step) {
                    cached.remove(&drop.key);
                }
                for unit in step
                    .units()
                    .iter()
                    .filter(|unit| unit.placed.action() == Action::Encode)
                {
                    cached.insert(unit.key, unit.rows.clone());
                }
                for page in step
                    .pages()
                    .iter()
                    .filter(|page| page.placed.action() == Action::Encode)
                {
                    cached.insert(page.key, page.rows.clone());
                }
            }
        }
    }
}

#[test]
fn many_distinct_cutoffs_plan_in_one_pass_over_the_rows() {
    let mut node = node(64);
    let strategy = Strategy::subsample(16, 3, CoreKind::Random);
    let table = node.create_table(spec(strategy)).unwrap();
    let count = 20_000u64;
    let (keys, features, targets) = rows(count);
    let times: Vec<Option<i64>> = (0..count as i64).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    // One query per distinct cutoff, so every window is built and sampled.
    let cutoffs: Vec<i64> = (1..=500).map(|index| index * 40).collect();
    let queries: Vec<f32> = cutoffs.iter().flat_map(|_| A).collect();
    node.submit_before(table, &queries, Some(&cutoffs)).unwrap();

    let started = std::time::Instant::now();
    let step = run(&mut node);

    // Each window samples the rows before its cutoff; windows holding the
    // same lowest-ranked rows share one unit, so later cutoffs add few.
    assert!(step.units().len() > 1 && step.units().len() <= 500);
    assert_eq!(step.queries()[0].units.len(), 500);
    assert!(step.units().iter().all(|unit| unit.rows.len() == 16));
    let first = step.unit(step.queries()[0].units[0]);
    assert!(first.rows.iter().all(|row| row.0 < 40));
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "planning took {:?}",
        started.elapsed()
    );
}

#[test]
fn errors_report_one_kind_for_servers_and_bindings_alike() {
    use gf_core::node::ErrorKind;
    use gf_core::pages::IndexError;
    use gf_core::table::RowKey;

    assert_eq!(NodeError::UnknownKey(RowKey(7)).kind(), ErrorKind::NotFound);
    // Queries that cannot be placed are bad input, not a fault of the node.
    assert_eq!(
        NodeError::Index(IndexError::NotFinite).kind(),
        ErrorKind::Invalid
    );
    assert_eq!(
        NodeError::Index(IndexError::Unknown(RowId(3))).kind(),
        ErrorKind::Internal
    );
    assert_eq!(NodeError::NoQueries.kind(), ErrorKind::Invalid);
}

#[test]
fn a_table_past_its_fit_rows_fits_on_a_sample_and_embeds_only_pages_it_reads() {
    let mut node = node(64);
    let table = node
        .create_table(TableSpec {
            fit_rows: NonZeroUsize::new(4).unwrap(),
            ..spec(NEAREST)
        })
        .unwrap();
    let (keys, features, targets) = rows(32);
    node.insert(table, &keys, &features, targets).unwrap();
    node.submit(table, &A).unwrap();

    let step = run(&mut node);

    let fit = &step.fits()[0];
    assert_eq!(fit.rows.len(), 4);
    assert!(fit.pages.is_empty());
    // Only the pages near the query, each embedded from its rows.
    let read: usize = step.pages().iter().map(|page| page.rows.len()).sum();
    assert!(read < 32);
    assert!(step.pages().iter().all(|page| page.embed.is_some()));
}

#[test]
fn a_fit_past_the_workers_call_budget_fits_its_state_alone_on_the_rows_it_takes() {
    let fit = |state_rows, fused_rows| {
        let mut node = node(64);
        let sizes = EntrySizes {
            state_rows,
            fused_rows,
            ..spec(NEAREST).sizes
        };
        let table = node
            .create_table(TableSpec {
                sizes,
                ..spec(NEAREST)
            })
            .unwrap();
        let (keys, features, targets) = rows(32);
        node.insert(table, &keys, &features, targets).unwrap();
        node.submit(table, &A).unwrap();
        let step = run(&mut node);
        let read: usize = step.pages().iter().map(|page| page.rows.len()).sum();
        let fit = &step.fits()[0];
        assert!(step.pages().iter().all(|page| page.embed.is_some()));
        assert_eq!(
            node.spec(table).unwrap().effective_fit_rows(),
            state_rows.min(DEFAULT_FIT_ROWS)
        );
        (fit.rows.len(), fit.pages.len(), read)
    };

    // Past the rows a fit embeds with its state, the state is fit alone on
    // every row, and only the pages near the query are embedded.
    let (rows, pages, read) = fit(usize::MAX, 16);
    assert_eq!((rows, pages), (32, 0));
    assert!(read < 32);
    // Past the rows the state takes, it is fit on a sample of that many.
    let (rows, pages, _) = fit(8, 16);
    assert_eq!((rows, pages), (8, 0));
}

#[test]
fn a_fit_past_its_bound_waits_for_its_limits_to_be_measured() {
    let mut node = node(64);
    let sizes = EntrySizes {
        fit_bound: 16,
        ..spec(NEAREST).sizes
    };
    let table = node
        .create_table(TableSpec {
            sizes,
            ..spec(NEAREST)
        })
        .unwrap();
    let (keys, features, targets) = rows(32);
    node.insert(table, &keys, &features, targets).unwrap();
    node.submit(table, &A).unwrap();

    // It waits, and is named once.
    assert!(node.schedule().unwrap().is_none());
    assert_eq!(node.take_unmeasured(), [table]);
    assert!(node.schedule().unwrap().is_none());
    assert!(node.take_unmeasured().is_empty());

    node.set_fit_limits(table, 8, 0).unwrap();
    let step = run(&mut node);
    assert_eq!(step.fits()[0].rows.len(), 8);
    assert!(step.fits()[0].pages.is_empty());
}

#[test]
fn a_fitted_table_that_outgrows_its_bound_waits_before_its_refit() {
    let mut node = node(64);
    let sizes = EntrySizes {
        fit_bound: 16,
        ..spec(NEAREST).sizes
    };
    let table = node
        .create_table(TableSpec {
            sizes,
            refit: RefitPolicy::new(Some(0.5), None).unwrap(),
            ..spec(NEAREST)
        })
        .unwrap();
    let (keys, features, targets) = rows(8);
    node.insert(table, &keys, &features, targets).unwrap();
    node.submit(table, &A).unwrap();
    assert_eq!(run(&mut node).fits()[0].rows.len(), 8);

    // Enough rows arrive for the policy to refit, past the bound.
    let (keys, features, _) = rows(48);
    node.insert(
        table,
        &keys[8..],
        &features[8 * FEATURES..],
        Targets::Classes(vec![0; 40]),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    assert!(node.schedule().unwrap().is_none());
    assert_eq!(node.take_unmeasured(), [table]);

    node.set_fit_limits(table, 8, 0).unwrap();
    assert_eq!(run(&mut node).fits()[0].rows.len(), 8);
}

#[test]
fn a_table_that_summarizes_each_context_sends_units_their_rows_and_keeps_their_state() {
    let mut node = node(64);
    let sizes = EntrySizes {
        state_bytes: 100,
        ..spec(NEAREST).sizes
    };
    let table = node
        .create_table(TableSpec {
            summary: SummaryScope::Context,
            sizes,
            // Its last column places rows for retrieval, and no unit reads it.
            retrieval: 1,
            ..spec(NEAREST)
        })
        .unwrap();
    let (keys, features, targets) = rows(32);
    node.insert(table, &keys, &features, targets).unwrap();

    node.submit(table, &A).unwrap();
    let first = run(&mut node);
    node.submit(table, &A).unwrap();
    let second = run(&mut node);

    assert!(first.fits().is_empty() && first.pages().is_empty());
    let unit = &first.units()[0];
    let (features, targets) = unit.fit.as_ref().expect("an encoded unit has its rows");
    assert_eq!(features.len(), unit.rows.len() * (FEATURES - 1));
    assert_eq!(targets.len(), unit.rows.len());
    // Two slots of 8 rows each, then two for its 100 bytes of state.
    assert_eq!(unit.placed.slots().len(), 4);
    assert_eq!(actions(&second), [Action::Reuse]);
    assert!(second.units()[0].fit.is_none());
}

#[test]
fn units_fit_on_their_own_rows_outlive_drift_and_serve_each_cutoff_again() {
    let mut node = node(64);
    let table = node
        .create_table(TableSpec {
            summary: SummaryScope::Context,
            refit: RefitPolicy::new(Some(0.01), None).unwrap(),
            ..spec(NEAREST)
        })
        .unwrap();
    let (keys, features, targets) = rows(32);
    let times: Vec<Option<i64>> = (0..32).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    let mut at = |cutoff: i64| {
        node.submit_before(table, &A, Some(&[cutoff])).unwrap();
        run(&mut node)
    };

    let early = at(20);
    let late = at(30);
    let again = at(20);

    assert!(early.fits().is_empty() && early.pages().is_empty());
    assert_eq!(actions(&late), [Action::Encode]);
    // The early cutoff's unit is kept: no cutoff or drift refits the table.
    assert_eq!(actions(&again), [Action::Reuse]);
    assert_eq!(again.units()[0].key, early.units()[0].key);
}

/// A compaction's rows and layout from `start` to `end`, with the layouts of
/// the pages that begin in that range, as a log splits it into records.
fn part(compaction: &Compaction, start: usize, end: usize) -> (TableRows, RowLayout) {
    let rows = &compaction.rows;
    let width = rows.features.len() / rows.keys.len();
    let Targets::Classes(labels) = &rows.targets else {
        unreachable!("the test tables are classifiers")
    };
    let begun: HashSet<PageId> = compaction.layout.pages[..start].iter().copied().collect();
    let pages = &compaction.layout.pages[start..end];
    let new_pages = compaction
        .layout
        .new_pages
        .iter()
        .filter(|page| pages.contains(&page.id) && !begun.contains(&page.id))
        .cloned()
        .collect();
    let part = TableRows {
        keys: rows.keys[start..end].to_vec(),
        features: rows.features[start * width..end * width].to_vec(),
        targets: Targets::Classes(labels[start..end].to_vec()),
        times: rows.times[start..end].to_vec(),
    };
    let layout = RowLayout {
        ids: compaction.layout.ids[start..end].to_vec(),
        pages: pages.to_vec(),
        new_pages,
    };
    (part, layout)
}

#[test]
fn a_table_restored_from_its_compaction_is_the_same_table() {
    let mut original = node(64);
    let table = original.create_table(spec(NEAREST)).unwrap();
    let (keys, features, targets) = rows(48);
    original.insert(table, &keys, &features, targets).unwrap();
    original.delete(table, &[3, 10, 11]).unwrap();
    let (keys, features, targets) = rows(3);
    original.update(table, &keys, &features, targets).unwrap();
    let compaction = original.compaction(table).unwrap();

    let mut copy = node(64);
    let restored = copy.create_table(spec(NEAREST)).unwrap();
    let middle = compaction.rows.keys.len() / 2;
    let (rows, layout) = part(&compaction, 0, middle);
    copy.restore(restored, &rows, &layout, Some(&compaction.table))
        .unwrap();
    // Half restored, the table waits for the rest before it plans requests.
    copy.submit(restored, &A).unwrap();
    assert!(copy.schedule().unwrap().is_none());
    let (rows, layout) = part(&compaction, middle, compaction.rows.keys.len());
    copy.restore(restored, &rows, &layout, None).unwrap();

    assert_eq!(copy.compaction(restored).unwrap(), compaction);
    // Rows added later land on the same pages in both.
    let keys: Vec<u128> = (100..140).collect();
    let features: Vec<f32> = keys.iter().flat_map(|&key| [key as f32, 50.0]).collect();
    let labels = Targets::Classes(keys.iter().map(|&key| (key % 3) as i32).collect());
    original
        .insert(table, &keys, &features, labels.clone())
        .unwrap();
    copy.insert(restored, &keys, &features, labels).unwrap();
    assert_eq!(
        copy.compaction(restored).unwrap(),
        original.compaction(table).unwrap()
    );
}

#[test]
fn a_restore_that_continues_nothing_is_refused() {
    let mut original = node(64);
    let table = original.create_table(spec(NEAREST)).unwrap();
    let (keys, features, targets) = rows(8);
    original.insert(table, &keys, &features, targets).unwrap();
    let compaction = original.compaction(table).unwrap();
    let (rows, layout) = part(&compaction, 0, 8);

    let mut copy = node(64);
    let restored = copy.create_table(spec(NEAREST)).unwrap();

    assert!(matches!(
        copy.restore(restored, &rows, &layout, None),
        Err(NodeError::Restore(_))
    ));
}

/// A sealer that keeps blocks as they are, binding their place and version.
#[derive(Debug)]
struct Plain;

impl Sealer for Plain {
    fn seal(&self, block: u32, version: u32, plain: &[u8]) -> Vec<u8> {
        [plain, &block.to_le_bytes(), &version.to_le_bytes()].concat()
    }

    fn open(&self, block: u32, version: u32, sealed: &[u8]) -> Option<Vec<u8>> {
        let (plain, tail) = sealed.split_at(sealed.len().checked_sub(8)?);
        (tail == [block.to_le_bytes(), version.to_le_bytes()].concat()).then(|| plain.to_vec())
    }
}

#[test]
fn a_table_whose_rows_spill_to_disk_holds_what_one_in_memory_does() {
    let directory = std::env::temp_dir().join(format!("gf-node-rows-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut in_memory = node(64);
    let mut spilling = node(64).with_row_memory(RowMemory::bounded(0));
    let kept = in_memory.create_table(spec(NEAREST)).unwrap();
    let spilled = spilling
        .create_table_on_disk(spec(NEAREST), &directory.join("t.rows"), Arc::new(Plain))
        .unwrap();
    for (node, table) in [(&mut in_memory, kept), (&mut spilling, spilled)] {
        let (keys, features, targets) = rows(200);
        node.insert(table, &keys, &features, targets).unwrap();
        node.delete(table, &[5, 6, 7]).unwrap();
        let (keys, features, _) = rows(20);
        node.update(
            table,
            &keys[8..],
            &features[16..],
            Targets::Classes(vec![1; 12]),
        )
        .unwrap();
    }

    assert_eq!(spilling.row_bytes(), 0);
    assert_eq!(
        spilling.compaction(spilled).unwrap(),
        in_memory.compaction(kept).unwrap()
    );
    spilling.submit(spilled, &A).unwrap();
    in_memory.submit(kept, &A).unwrap();
    let (read, expected) = (run(&mut spilling), run(&mut in_memory));
    assert_eq!(read.fits()[0].features, expected.fits()[0].features);
}

fn ids(rows: &[u32]) -> gf_core::context::Context {
    rows.iter().map(|&row| RowId(row)).collect()
}

#[test]
fn given_contexts_are_planned_as_given_and_shared_ones_encode_once() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 32);
    let given = vec![ids(&[0, 2, 4]), ids(&[1, 3]), ids(&[0, 2, 4])];

    node.submit_given(table, &[0.0, 0.0, 100.0, 100.0, 1.0, 1.0], given)
        .unwrap();
    let step = run(&mut node);

    let units: Vec<Vec<RowId>> = step.units().iter().map(|unit| unit.rows.clone()).collect();
    assert_eq!(
        units,
        [vec![RowId(0), RowId(2), RowId(4)], vec![RowId(1), RowId(3)]]
    );
    let reads: Vec<usize> = step.queries()[0].units.iter().map(|unit| unit.0).collect();
    assert_eq!(reads, [0, 1, 0]);
}

#[test]
fn given_contexts_must_hold_live_rows_and_wait_their_turn() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    let query = [0.0, 0.0];

    for given in [ids(&[99]), ids(&[])] {
        assert_eq!(
            node.submit_given(table, &query, vec![given]),
            Err(NodeError::Given)
        );
    }
    // A given request behind a pending one is planned alone after it, and
    // requests after it wait their turn.
    let first = node.submit(table, &query).unwrap();
    let given = node.submit_given(table, &query, vec![ids(&[0])]).unwrap();
    let last = node.submit(table, &query).unwrap();
    let steps: Vec<Step> = (0..3).map(|_| run(&mut node)).collect();

    let served: Vec<Vec<RequestId>> = steps
        .iter()
        .map(|step| step.queries().iter().map(|batch| batch.request).collect())
        .collect();
    assert_eq!(served, [vec![first], vec![given], vec![last]]);
    assert_eq!(steps[1].units()[0].rows, [RowId(0)]);
    assert!(node.schedule().unwrap().is_none());
}

/// The rows a table plans for one query on its own.
fn planned(node: &mut Node, table: TableId, query: &[f32]) -> HashSet<RowId> {
    node.submit(table, query).unwrap();
    let step = run(node);
    step.units()
        .iter()
        .flat_map(|unit| unit.rows.iter().copied())
        .collect()
}

#[test]
fn candidates_hold_what_a_table_plans_with_their_ranks() {
    let query = [0.5, 0.5];
    for strategy in [
        Strategy::Exact,
        Strategy::subsample(5, 3, CoreKind::Random),
        Strategy::neighbors(3),
        Strategy::neighbors(8),
    ] {
        let mut node = node(64);
        let table = table(&mut node, strategy, 32);
        let expected = planned(&mut node, table, &query);

        let offered = node.candidates(table, &query, None, None).unwrap();

        let rows: HashSet<RowId> = offered.iter().map(|candidate| candidate.row).collect();
        assert_eq!(rows, expected, "{strategy:?}");
        let shared = matches!(
            strategy,
            Strategy::Exact | Strategy::Retrieved { neighbors: 0, .. }
        );
        assert!(
            offered
                .iter()
                .all(|candidate| candidate.query == (!shared).then_some(0)),
            "{strategy:?}"
        );
        assert!(
            offered
                .iter()
                .all(|candidate| candidate.score.is_finite() && (!shared || candidate.score == 0.0)),
            "{strategy:?}"
        );
    }
}

#[test]
fn a_partitions_share_of_a_sample_is_its_own_sample() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::subsample(16, 3, CoreKind::Random), 32);

    let offered = node.candidates(table, &[0.0, 0.0], Some(4), None).unwrap();
    let rows = node
        .rows_of(
            table,
            &offered
                .iter()
                .map(|candidate| candidate.row)
                .collect::<Vec<_>>(),
        )
        .unwrap();

    assert_eq!(offered.len(), 4);
    let (keys, features, _) = self::rows(32);
    for (index, candidate) in offered.iter().enumerate() {
        let key = u128::from(candidate.row.0);
        assert_eq!(rows.keys[index], key);
        let at = 2 * candidate.row.0 as usize;
        assert_eq!(
            &rows.features[2 * index..2 * index + 2],
            &features[at..at + 2]
        );
        assert_eq!(keys[candidate.row.0 as usize], key);
    }
}

/// A table of 16 rows at times 10 ..= 25, row `k` at time `10 + k`.
fn timed(node: &mut Node) -> TableId {
    let table = node.create_table(spec(Strategy::Exact)).unwrap();
    let (keys, features, targets) = rows(16);
    let times: Vec<Option<i64>> = (10..26).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    table
}

/// The rows a step's fit read, by id.
fn fit_rows(step: &Step) -> Vec<u32> {
    let mut rows: Vec<u32> = step.fits()[0].rows.iter().map(|row| row.0).collect();
    rows.sort_unstable();
    rows
}

#[test]
fn a_query_with_a_cutoff_reads_a_state_fit_only_on_earlier_rows() {
    let mut node = node(64);
    let table = timed(&mut node);

    node.submit_before(table, &A, Some(&[14])).unwrap();
    let early = run(&mut node);
    node.submit_before(table, &A, Some(&[14])).unwrap();
    let again = run(&mut node);
    node.submit_before(table, &A, Some(&[20])).unwrap();
    let later = run(&mut node);
    // A cutoff after every row reads every row, as a query without one does.
    node.submit_before(table, &A, Some(&[99])).unwrap();
    let after = run(&mut node);
    node.submit(table, &A).unwrap();
    let whole = run(&mut node);

    assert_eq!(fit_rows(&early), [0, 1, 2, 3]);
    assert!(again.fits().is_empty());
    assert_eq!(fit_rows(&later), (0..10).collect::<Vec<_>>());
    assert_eq!(fit_rows(&after), (0..16).collect::<Vec<_>>());
    assert!(whole.fits().is_empty());
}

#[test]
fn requests_needing_states_fit_on_other_rows_run_in_turn() {
    let mut node = node(64);
    let table = timed(&mut node);

    let first = node.submit_before(table, &A, Some(&[14])).unwrap();
    let second = node.submit_before(table, &A, Some(&[20])).unwrap();
    let third = node.submit_before(table, &A, Some(&[14])).unwrap();
    let one = run(&mut node);
    let two = run(&mut node);

    let served =
        |step: &Step| -> Vec<_> { step.queries().iter().map(|batch| batch.request).collect() };
    assert_eq!(served(&one), [first, third]);
    assert_eq!(served(&two), [second]);
    assert_eq!(fit_rows(&two), (0..10).collect::<Vec<_>>());
}

#[test]
fn candidates_with_a_cutoff_come_from_before_it() {
    let mut node = node(64);
    let table = timed(&mut node);

    let offered = node.candidates(table, &A, None, Some(14)).unwrap();

    let mut rows: Vec<u32> = offered.iter().map(|candidate| candidate.row.0).collect();
    rows.sort_unstable();
    assert_eq!(rows, [0, 1, 2, 3]);
}

#[test]
fn a_cutoff_refits_once_rows_its_state_read_have_gone_or_moved() {
    let mut node = node(64);
    let table = timed(&mut node);
    node.submit(table, &A).unwrap();
    run(&mut node);
    // The rows from 14 on leave, so a cutoff at 14 has none after it, yet
    // the state was fit on them.
    node.delete(table, &(4..16).collect::<Vec<_>>()).unwrap();
    node.submit_before(table, &A, Some(&[14])).unwrap();
    assert_eq!(fit_rows(&run(&mut node)), [0, 1, 2, 3]);

    let mut node = self::node(64);
    let table = timed(&mut node);
    node.submit_before(table, &A, Some(&[20])).unwrap();
    run(&mut node);
    // Rows at 14 to 19 move past every cutoff; the cutoff at 14 now names
    // the same rows after it as the one at 20 did.
    let (_, features, _) = rows(16);
    node.update_at(
        table,
        &(4..10).collect::<Vec<_>>(),
        &features[8..20],
        Targets::Classes(vec![0; 6]),
        Some(&[Some(30); 6]),
    )
    .unwrap();
    node.submit_before(table, &A, Some(&[14])).unwrap();
    assert_eq!(fit_rows(&run(&mut node)), [0, 1, 2, 3]);
}

#[test]
fn cores_stratify_by_class_or_by_target_bin() {
    let mut node = node(64);
    let (keys, features, _) = rows(64);
    // Classes 0, 1 and 2 in 32, 16 and 16 rows.
    let labels = keys
        .iter()
        .map(|&key| [0, 0, 1, 2][key as usize % 4])
        .collect();
    let classes = node
        .create_table(spec(Strategy::subsample(16, 1, CoreKind::Balanced)))
        .unwrap();
    node.insert(classes, &keys, &features, Targets::Classes(labels))
        .unwrap();
    let values = node
        .create_table(TableSpec {
            task: Task::Values,
            ..spec(Strategy::subsample(16, 1, CoreKind::Stratified))
        })
        .unwrap();
    let targets = keys.iter().map(|&key| key as f32).collect();
    node.insert(values, &keys, &features, Targets::Values(targets))
        .unwrap();
    for table in [classes, values] {
        node.submit(table, &A).unwrap();
    }

    let step = run(&mut node);

    let counts = |table: TableId, stratum: &dyn Fn(u32) -> usize, strata: usize| {
        let unit = step
            .units()
            .iter()
            .find(|unit| unit.table == table)
            .unwrap();
        let mut counts = vec![0; strata];
        for row in &unit.rows {
            counts[stratum(row.0)] += 1;
        }
        counts
    };
    // Rows are inserted in key order, so a row's id is its key.
    let class = |row: u32| [0, 0, 1, 2][row as usize % 4];
    // Five rows a class, and the one left over to a class by its seeded hash.
    assert_eq!(counts(classes, &class, 3), [5, 5, 6]);
    // Eight bins of eight values each, two rows from every bin.
    assert_eq!(counts(values, &|row| row as usize / 8, 8), [2; 8]);
}

#[test]
fn a_regression_core_offered_to_a_router_is_the_one_planned() {
    let mut node = node(64);
    let table = node
        .create_table(TableSpec {
            task: Task::Values,
            ..spec(Strategy::subsample(16, 1, CoreKind::Stratified))
        })
        .unwrap();
    let (keys, features, _) = rows(64);
    let targets = keys.iter().map(|&key| key as f32).collect();
    node.insert(table, &keys, &features, Targets::Values(targets))
        .unwrap();

    let offered: HashSet<RowId> = node
        .candidates(table, &A, None, None)
        .unwrap()
        .iter()
        .map(|candidate| candidate.row)
        .collect();

    assert_eq!(offered, planned(&mut node, table, &A));
    // Two rows of each of the eight bins, not the first sixteen by hash.
    let bins: HashSet<u32> = offered.iter().map(|row| row.0 / 8).collect();
    assert_eq!(bins.len(), 8);
}

#[test]
fn contexts_past_the_models_length_are_cut_or_reported_once() {
    let mut node = node(64);
    let capped = |past_cap| TableSpec {
        context_rows: NonZeroU64::new(8),
        past_cap,
        ..spec(Strategy::Exact)
    };
    let clamped = node.create_table(capped(PastCap::Clamp)).unwrap();
    let warned = node.create_table(capped(PastCap::Warn)).unwrap();
    let (keys, features, targets) = rows(32);
    for table in [clamped, warned] {
        node.insert(table, &keys, &features, targets.clone())
            .unwrap();
    }
    let read = |node: &mut Node| {
        for table in [clamped, warned] {
            node.submit(table, &A).unwrap();
        }
        let step = run(node);
        [clamped, warned].map(|table| {
            step.units()
                .iter()
                .find(|unit| unit.table == table)
                .map(|unit| unit.rows.len())
        })
    };

    assert_eq!(read(&mut node), [Some(8), Some(32)]);
    assert_eq!(node.take_past_cap(), [warned]);
    read(&mut node);
    assert!(node.take_past_cap().is_empty());
    assert_eq!(
        node.spec(clamped).unwrap().strategy,
        Strategy::Exact.capped(8)
    );
    assert_eq!(
        Strategy::Retrieved {
            core: Some(gf_core::planner::Core {
                rows: 6,
                seed: 1,
                kind: CoreKind::Diverse,
            }),
            neighbors: 4,
            among: None,
            recency: None,
        }
        .capped(8),
        Strategy::Retrieved {
            core: Some(gf_core::planner::Core {
                rows: 6,
                seed: 1,
                kind: CoreKind::Diverse,
            }),
            neighbors: 2,
            among: None,
            recency: None,
        }
    );
}

#[test]
fn contexts_are_the_rows_a_step_plans_for_the_same_queries() {
    let mut node = node(64);
    let strategy = Strategy::Retrieved {
        core: Some(gf_core::planner::Core {
            rows: 4,
            seed: 2,
            kind: CoreKind::Diverse,
        }),
        neighbors: 3,
        among: Some(5),
        recency: Some(Recency::Last(12)),
    };
    let table = node.create_table(spec(strategy)).unwrap();
    let (keys, features, targets) = rows(32);
    let times: Vec<Option<i64>> = (0..32).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    let queries = [A, B, A].concat();
    let cutoffs = [8, 32, 32];

    let contexts = node.contexts(table, &queries, Some(&cutoffs)).unwrap();
    node.submit_before(table, &queries, Some(&cutoffs)).unwrap();
    let step = run(&mut node);

    // Rows are inserted in key order, so a row's id is its key.
    let planned: Vec<Vec<u128>> = step.queries()[0]
        .units
        .iter()
        .map(|&unit| {
            step.unit(unit)
                .rows
                .iter()
                .map(|row| u128::from(row.0))
                .collect()
        })
        .collect();
    assert_eq!(contexts, planned);
    assert_eq!(contexts[1].len(), 7);
    assert!(contexts[0].iter().all(|&key| key < 8));
}

#[test]
fn contexts_change_nothing_a_later_prediction_reads() {
    let strategy = Strategy::subsample(8, 0, CoreKind::Diverse);
    let (keys, features, targets) = rows(128);
    let Targets::Classes(labels) = targets else {
        unreachable!("rows are labelled")
    };
    let times: Vec<Option<i64>> = (0..128).map(Some).collect();
    let mut planned = Vec::new();
    for probe in [false, true] {
        let mut node = node(64);
        let table = node.create_table(spec(strategy)).unwrap();
        for range in [0..64, 64..128] {
            node.insert_at(
                table,
                &keys[range.clone()],
                &features[range.start * FEATURES..range.end * FEATURES],
                Targets::Classes(labels[range.clone()].to_vec()),
                Some(&times[range.clone()]),
            )
            .unwrap();
            if range.start == 0 {
                node.submit(table, &A).unwrap();
                run(&mut node);
            }
        }
        // A cutoff before rows the state was fit on would refit it.
        if probe {
            node.contexts(table, &A, Some(&[20])).unwrap();
        }
        node.submit(table, &A).unwrap();
        planned.push(run(&mut node).units()[0].rows.clone());
    }
    assert_eq!(planned[0], planned[1]);
}

fn regioned(rebuild: Rebuild) -> TableSpec {
    TableSpec {
        regions: Some(RegionSpec {
            rows: 32,
            rebuild,
            wait: RebuildWait::default(),
        }),
        ..spec(Strategy::neighbors(8))
    }
}

#[test]
fn a_regions_queries_share_its_context_and_read_it_stale_until_a_rebuild_step() {
    let mut node = node(64);
    let table = node.create_table(regioned(Rebuild::Always)).unwrap();
    let (keys, features, targets) = rows(64);
    node.insert(table, &keys, &features, targets).unwrap();
    let near_a = [A, [A[0] + 0.02, A[1]]].concat();

    node.submit(table, &near_a).unwrap();
    let first = run(&mut node);
    // Rows at the middle of A's rows, from keys past the table's, so the
    // region's nearest rows change.
    let (more, _, _) = rows(80);
    let added: Vec<f32> = (64..80).flat_map(|_| [0.155, 0.0]).collect();
    node.insert(table, &more[64..], &added, Targets::Classes(vec![0; 16]))
        .unwrap();
    node.submit(table, &A).unwrap();
    let stale = run(&mut node);
    let rebuild = run_idle(&mut node);
    node.submit(table, &A).unwrap();
    let rebuilt = run(&mut node);

    // Two queries of one region read one context.
    assert_eq!(first.units().len(), 1);
    assert_eq!(stale.units()[0].key, first.units()[0].key);
    assert_eq!(actions(&stale), [Action::Reuse]);
    // The rebuild encodes alone, and later queries read what it built.
    assert!(rebuild.queries().is_empty());
    assert_eq!(actions(&rebuild), [Action::Encode]);
    assert_eq!(rebuilt.units()[0].key, rebuild.units()[0].key);
    assert_eq!(actions(&rebuilt), [Action::Reuse]);
    let stats = node.region_stats(table).unwrap().unwrap();
    assert_eq!(stats.regions, 2);
    assert_eq!((stats.misses, stats.stale, stats.hits), (2, 1, 1));
    assert_eq!((stats.rebuilds, stats.rebuilt_rows), (1, 8));
}

/// Runs the step the node schedules with no request waiting.
fn run_idle(node: &mut Node) -> Step {
    let step = node.schedule().unwrap().expect("a rebuild is due");
    node.complete(&step);
    step
}

#[test]
fn a_region_built_never_again_is_read_until_its_entry_goes() {
    let mut node = node(64);
    let table = node.create_table(regioned(Rebuild::Never)).unwrap();
    let (keys, features, targets) = rows(64);
    node.insert(
        table,
        &keys[..48],
        &features[..96],
        targets_of(&targets, 0..48),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    let first = run(&mut node);
    node.insert(
        table,
        &keys[48..],
        &features[96..],
        targets_of(&targets, 48..64),
    )
    .unwrap();

    assert!(node.schedule().unwrap().is_none());
    node.submit(table, &A).unwrap();
    assert_eq!(run(&mut node).units()[0].key, first.units()[0].key);
}

#[test]
fn a_changed_region_built_each_period_falls_due_when_its_period_ends() {
    let period = std::time::Duration::from_secs(2);
    let mut node = node(64);
    let (table, more, middle) = regioned_table(&mut node, Rebuild::Every(period));
    planned_rows(&mut node, table, &A);
    node.insert(table, &more, &middle, Targets::Classes(vec![0; 64]))
        .unwrap();

    assert!(node.schedule().unwrap().is_none());
    std::thread::sleep(period);
    assert!(!run_idle(&mut node).units().is_empty());
}

#[test]
fn rebuilds_waiting_on_steps_alone_come_at_the_same_steps_however_long_a_run_takes() {
    // Each run adds a row to A's region before each request, so its context
    // falls due, and returns what every step read.
    let steps = |time: Option<std::time::Duration>, pause: std::time::Duration| {
        let mut node = node(64);
        let spec = TableSpec {
            regions: Some(RegionSpec {
                rows: 32,
                rebuild: Rebuild::Always,
                wait: RebuildWait { steps: 2, time },
            }),
            ..spec(Strategy::neighbors(8))
        };
        let table = node.create_table(spec).unwrap();
        let (keys, features, targets) = rows(64);
        node.insert(table, &keys, &features, targets).unwrap();
        node.submit(table, &A).unwrap();
        run(&mut node);
        let mut read = Vec::new();
        for round in 0..4 {
            node.insert(
                table,
                &[64 + round],
                &[0.155, 0.0],
                Targets::Classes(vec![0]),
            )
            .unwrap();
            std::thread::sleep(pause);
            node.submit(table, &A).unwrap();
            loop {
                let step = run(&mut node);
                let units: Vec<Vec<RowId>> =
                    step.units().iter().map(|unit| unit.rows.clone()).collect();
                read.push((step.queries().len(), units));
                if !step.queries().is_empty() {
                    break;
                }
            }
        }
        read
    };
    let pause = std::time::Duration::from_millis(50);
    let quick = steps(None, std::time::Duration::ZERO);

    assert_eq!(steps(None, pause), quick);
    // A rebuild took a step of its own once two request steps had read stale.
    assert!(quick.iter().any(|(queries, _)| *queries == 0));
    // Counting wall time, the slower run rebuilds sooner.
    let timed = Some(std::time::Duration::from_millis(20));
    assert_ne!(steps(timed, pause), steps(timed, std::time::Duration::ZERO));
}

#[test]
fn tables_of_the_same_rows_fit_and_plan_alike() {
    let mut node = node(64);
    let tables = [0, 1].map(|_| node.create_table(regioned(Rebuild::Always)).unwrap());
    let (keys, features, targets) = rows(64);
    for table in tables {
        node.insert(table, &keys, &features, targets.clone())
            .unwrap();
        node.submit(table, &[A, B].concat()).unwrap();
    }

    let step = run(&mut node);

    let state = |table: TableId| {
        let fit = step.fits().iter().find(|fit| fit.table == table).unwrap();
        let units: Vec<Vec<RowId>> = step
            .units()
            .iter()
            .filter(|unit| unit.table == table)
            .map(|unit| unit.rows.clone())
            .collect();
        (fit.rows.clone(), fit.features.clone(), units)
    };
    assert_eq!(state(tables[0]), state(tables[1]));
}

/// A table of regions over the 64 rows of two clusters, and rows at A's
/// middle to insert, from keys past the table's.
fn regioned_table(node: &mut Node, rebuild: Rebuild) -> (TableId, Vec<u128>, Vec<f32>) {
    let table = node.create_table(regioned(rebuild)).unwrap();
    let (keys, features, targets) = rows(64);
    node.insert(table, &keys, &features, targets).unwrap();
    let (more, _, _) = rows(128);
    let middle: Vec<f32> = (0..64).flat_map(|_| [0.155, 0.0]).collect();
    (table, more[64..].to_vec(), middle)
}

/// The rows a step plans for one query, without other requests.
fn planned_rows(node: &mut Node, table: TableId, query: &[f32]) -> Vec<RowId> {
    node.submit(table, query).unwrap();
    run(node).units()[0].rows.clone()
}

#[test]
fn a_rebuild_planned_in_regions_drawn_since_is_dropped() {
    let (_, features, _) = rows(64);
    // Far from both clusters, where the regions drawn again put new ones.
    let spread: Vec<f32> = (0..124)
        .flat_map(|row| [row as f32 * 3.0, 50.0 + row as f32])
        .collect();
    let queries = [features, spread.clone()].concat();
    let mut contexts = Vec::new();
    for rebuilding in [false, true] {
        let mut node = node(64);
        let (table, more, _) = regioned_table(&mut node, Rebuild::Always);
        planned_rows(&mut node, table, &[A, B].concat());
        let middles: Vec<f32> = [[0.155, 0.0], [100.155, 100.0]].repeat(2).concat();
        node.insert(table, &more[..4], &middles, Targets::Classes(vec![0; 4]))
            .unwrap();
        let rebuild = rebuilding.then(|| node.schedule().unwrap().expect("A and B are due"));
        // Rows past twice those the scaling was fit on refit it, so the next
        // plan draws the regions again, while the rebuild still runs.
        let (far, _, _) = rows(256);
        node.insert(table, &far[132..], &spread, Targets::Classes(vec![1; 124]))
            .unwrap();
        node.submit(table, &A).unwrap();
        if let Some(rebuild) = rebuild {
            assert!(node.schedule().unwrap().is_none());
            node.complete(&rebuild);
        }
        run(&mut node);
        contexts.push(node.contexts(table, &queries, None).unwrap());
    }

    // Every region reads what it would have without the rebuild.
    assert_eq!(contexts[0], contexts[1]);
}

#[test]
fn rows_changed_while_a_rebuild_runs_count_after_it() {
    let mut node = node(64);
    let (table, more, middle) = regioned_table(&mut node, Rebuild::Always);
    planned_rows(&mut node, table, &A);
    node.insert(
        table,
        &more[..4],
        &middle[..8],
        Targets::Classes(vec![0; 4]),
    )
    .unwrap();
    let rebuild = node.schedule().unwrap().expect("A's rebuild is due");
    node.insert(
        table,
        &more[4..8],
        &middle[8..16],
        Targets::Classes(vec![0; 4]),
    )
    .unwrap();
    node.complete(&rebuild);

    // The rows inserted meanwhile make the new context due again.
    let again = node.schedule().unwrap().expect("A is due again");
    assert!(again.queries().is_empty());
    assert_eq!(actions(&again), [Action::Encode]);
}

#[test]
fn every_query_counts_once_as_a_hit_stale_read_miss_or_fresh_context() {
    let mut node = node(64);
    let (table, more, middle) = regioned_table(&mut node, Rebuild::Always);
    let times: Vec<Option<i64>> = (64..68).map(Some).collect();
    node.submit(table, &A).unwrap();
    let encoding = node.schedule().unwrap().unwrap();
    // A second request waits for the context in flight, and a write plans it
    // again before it runs.
    node.submit(table, &A).unwrap();
    assert!(node.schedule().unwrap().is_none());
    node.insert_at(
        table,
        &more[..4],
        &middle[..8],
        Targets::Classes(vec![0; 4]),
        Some(&times),
    )
    .unwrap();
    node.complete(&encoding);
    run(&mut node);
    node.submit_before(table, &[A, B].concat(), Some(&[30, 30]))
        .unwrap();
    run(&mut node);

    let stats = node.region_stats(table).unwrap().unwrap();
    assert_eq!(stats.hits + stats.stale + stats.misses + stats.fresh, 4);
    assert_eq!((stats.misses, stats.fresh), (1, 2));
}

#[test]
fn rebuilds_take_regions_while_the_step_budget_lasts() {
    let mut node = node(64);
    let table = node
        .create_table(TableSpec {
            regions: Some(RegionSpec {
                rows: 8,
                rebuild: Rebuild::Always,
                wait: RebuildWait::default(),
            }),
            ..spec(Strategy::neighbors(8))
        })
        .unwrap();
    let (keys, features, targets) = rows(64);
    node.insert(table, &keys, &features, targets.clone())
        .unwrap();
    // Every region builds a context, then a copy of each row changes it.
    let regions = node
        .region_stats(table)
        .unwrap()
        .map_or(0, |stats| stats.regions);
    node.submit(table, &features).unwrap();
    run(&mut node);
    let (more, _, _) = rows(128);
    node.insert(table, &more[64..], &features, targets).unwrap();
    node.set_step_bytes(1);

    let step = node.schedule().unwrap().expect("rebuilds are due");

    assert_eq!(step.units().len(), 1);
    assert!(regions == 0 || regions > 1);
    assert!(node.region_stats(table).unwrap().unwrap().regions > 1);
}

#[test]
fn many_regions_are_two_levels_drawn_alike_for_the_same_rows() {
    let mut node = node(64);
    let spec = TableSpec {
        regions: Some(RegionSpec {
            rows: 1,
            rebuild: Rebuild::Never,
            wait: RebuildWait::default(),
        }),
        ..spec(Strategy::neighbors(4))
    };
    let tables = [0, 1].map(|_| node.create_table(spec.clone()).unwrap());
    let (keys, features, targets) = rows(2_048);
    let queries: Vec<f32> = features[..64].to_vec();
    let mut planned = Vec::new();
    for table in tables {
        node.insert(table, &keys, &features, targets.clone())
            .unwrap();
        node.submit(table, &queries).unwrap();
        let step = run(&mut node);
        let rows: Vec<Vec<RowId>> = step.units().iter().map(|unit| unit.rows.clone()).collect();
        planned.push(rows);
    }

    let stats = node.region_stats(tables[0]).unwrap().unwrap();
    assert_eq!(stats.levels, 2);
    assert!(stats.regions > 1_024, "{stats:?}");
    assert_eq!(planned[0], planned[1]);
}

#[test]
fn a_recent_context_holds_the_windows_latest_rows_and_neighbors() {
    let mut node = node(64);
    let strategy = Strategy::Retrieved {
        core: Some(gf_core::planner::Core {
            rows: 4,
            seed: 0,
            kind: CoreKind::Random,
        }),
        neighbors: 3,
        among: Some(8),
        recency: Some(Recency::Last(16)),
    };
    let table = node.create_table(spec(strategy)).unwrap();
    let (keys, features, targets) = rows(32);
    let times: Vec<Option<i64>> = (0..32).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();

    let rows = planned(&mut node, table, &A);

    // Only the 16 latest rows, by time; the neighbors near A are even rows,
    // the latest of the 8 nearest.
    assert_eq!(rows.len(), 7);
    assert!(rows.iter().all(|row| row.0 >= 16), "{rows:?}");
    let core: HashSet<RowId> = node
        .candidates(table, &A, None, None)
        .unwrap()
        .iter()
        .filter(|candidate| candidate.query.is_none())
        .map(|candidate| candidate.row)
        .collect();
    let mut nearest: Vec<u32> = rows.difference(&core).map(|row| row.0).collect();
    nearest.sort_unstable();
    let mut latest: Vec<u32> = (16..32)
        .step_by(2)
        .filter(|&row| !core.contains(&RowId(row)))
        .collect();
    latest.reverse();
    latest.truncate(3);
    latest.reverse();
    assert_eq!(nearest, latest);
}

#[test]
fn a_span_with_no_timed_row_is_an_empty_context() {
    let spanned = Strategy::Retrieved {
        core: None,
        neighbors: 4,
        among: None,
        recency: Some(Recency::Span(10)),
    };
    let mut node = node(64);
    let untimed = node.create_table(spec(spanned)).unwrap();
    let (keys, features, targets) = rows(16);
    node.insert(untimed, &keys, &features, targets.clone())
        .unwrap();
    let timed = node.create_table(spec(spanned)).unwrap();
    let times: Vec<Option<i64>> = (0..16).map(|row| (row >= 8).then_some(row)).collect();
    node.insert_at(timed, &keys, &features, targets, Some(&times))
        .unwrap();

    let without = node.submit(untimed, &A).unwrap();
    // Before 8, only the untimed rows are left, which a span leaves out.
    let early = node.submit_before(timed, &A, Some(&[8])).unwrap();
    let late = node.submit_before(timed, &A, Some(&[12])).unwrap();
    let step = run(&mut node);

    assert_eq!(
        node.take_refused(),
        [
            (without, NodeError::Untimed(untimed)),
            (early, NodeError::EmptyContext(timed))
        ]
    );
    assert_eq!(step.queries()[0].request, late);
    // Contexts refuse as predictions do.
    assert_eq!(
        node.contexts(untimed, &A, None),
        Err(NodeError::Untimed(untimed))
    );
    assert_eq!(
        node.contexts(timed, &A, Some(&[8])),
        Err(NodeError::EmptyContext(timed))
    );
}

#[test]
fn a_regions_recent_window_ages_out_and_rebuilds_on_churn() {
    let mut node = node(64);
    let table = node
        .create_table(TableSpec {
            regions: Some(RegionSpec {
                rows: 32,
                rebuild: Rebuild::Churn(0.5),
                wait: RebuildWait::default(),
            }),
            ..spec(Strategy::Retrieved {
                core: None,
                neighbors: 4,
                among: None,
                recency: Some(Recency::Last(32)),
            })
        })
        .unwrap();
    let (keys, features, targets) = rows(64);
    let times: Vec<Option<i64>> = (0..64).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    planned_rows(&mut node, table, &[A, B].concat());
    // Later rows near B alone push A's rows out of the latest 32.
    let (more, _, _) = rows(128);
    let near_b: Vec<f32> = (0..32)
        .flat_map(|row| [100.0 + row as f32 * 0.01, 100.0])
        .collect();
    let later: Vec<Option<i64>> = (64..96).map(Some).collect();
    node.insert_at(
        table,
        &more[64..96],
        &near_b,
        Targets::Classes(vec![1; 32]),
        Some(&later),
    )
    .unwrap();

    let rebuild = node.schedule().unwrap().expect("A's window has aged out");

    // A rebuilds too, not B alone, and every context holds only the
    // latest 32 rows.
    assert!(rebuild.queries().is_empty());
    assert_eq!(rebuild.units().len(), 2);
    assert!(
        rebuild
            .units()
            .iter()
            .all(|unit| unit.rows.iter().all(|row| row.0 >= 64))
    );
}

#[test]
fn rows_at_one_time_age_out_by_row() {
    let mut node = node(64);
    let strategy = Strategy::Retrieved {
        core: None,
        neighbors: 8,
        among: None,
        recency: Some(Recency::Last(3)),
    };
    let table = node.create_table(spec(strategy)).unwrap();
    let (keys, features, targets) = rows(8);
    // Rows 0 to 3 at time 0, rows 4 to 7 at time 1.
    let times: Vec<Option<i64>> = (0..8).map(|row| Some(row / 4)).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();

    let read = planned(&mut node, table, &A);

    // The latest three: rows of the later time, the higher first.
    assert_eq!(read, HashSet::from([RowId(5), RowId(6), RowId(7)]));
}

#[test]
fn cutoffs_refit_each_time_on_churn_or_never_but_never_read_a_later_row() {
    let changed = RefitPolicy::new(Some(0.5), None).unwrap();
    for (cutoffs, refits) in [
        (CutoffRefit::Each, [true, true, true]),
        (CutoffRefit::Changed, [false, true, true]),
        (CutoffRefit::Never, [false, false, true]),
    ] {
        let mut node = node(64);
        let table = node
            .create_table(TableSpec {
                refit: changed.with_cutoffs(cutoffs).unwrap(),
                ..spec(Strategy::Exact)
            })
            .unwrap();
        let (keys, features, targets) = rows(64);
        let times: Vec<Option<i64>> = (0..64).map(Some).collect();
        node.insert_at(table, &keys, &features, targets, Some(&times))
            .unwrap();
        node.submit_before(table, &A, Some(&[32])).unwrap();
        run(&mut node);

        // A quarter more rows, then three quarters, then a cutoff before
        // rows the state was fit on.
        let fitted = [40, 56, 16].map(|cutoff| {
            node.submit_before(table, &A, Some(&[cutoff])).unwrap();
            !run(&mut node).fits().is_empty()
        });

        assert_eq!(fitted, refits, "{cutoffs:?}");
    }

    // A state fit on every row counts only rows after its latest as changed.
    let mut node = node(64);
    let table = node
        .create_table(TableSpec {
            refit: changed.with_cutoffs(CutoffRefit::Changed).unwrap(),
            ..spec(Strategy::Exact)
        })
        .unwrap();
    let (keys, features, Targets::Classes(labels)) = rows(72) else {
        unreachable!()
    };
    let times: Vec<Option<i64>> = (0..72).map(Some).collect();
    let split = |end: usize, start: usize| {
        (
            &keys[start..end],
            &features[start * 2..end * 2],
            Targets::Classes(labels[start..end].to_vec()),
            &times[start..end],
        )
    };
    let (first, features_first, targets_first, times_first) = split(64, 0);
    node.insert_at(
        table,
        first,
        features_first,
        targets_first,
        Some(times_first),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    run(&mut node);
    let (more, features_more, targets_more, times_more) = split(72, 64);
    node.insert_at(table, more, features_more, targets_more, Some(times_more))
        .unwrap();
    node.submit_before(table, &A, Some(&[70])).unwrap();
    assert!(run(&mut node).fits().is_empty());

    assert_eq!(
        RefitPolicy::default().with_cutoffs(CutoffRefit::Changed),
        Err(NodeError::RefitPolicy)
    );
}

/// A regioned table of 64 rows whose region around A was built once, then
/// given 4 rows at its middle.
fn built_then_changed(node: &mut Node, spec: TableSpec) -> TableId {
    let table = node.create_table(spec).unwrap();
    let (keys, features, targets) = rows(64);
    node.insert(table, &keys, &features, targets).unwrap();
    node.submit(table, &A).unwrap();
    run(node);
    let (more, _, _) = rows(68);
    let added: Vec<f32> = (64..68).flat_map(|_| [0.155, 0.0]).collect();
    node.insert(table, &more[64..], &added, Targets::Classes(vec![0; 4]))
        .unwrap();
    table
}

/// The keys `contexts` reports for A, those the next prediction reads, and
/// whether its step refit.
fn reported_and_served(node: &mut Node, table: TableId) -> (Vec<u128>, Vec<u128>, bool) {
    let reported = node.contexts(table, &A, None).unwrap().remove(0);
    node.submit(table, &A).unwrap();
    let step = run(node);
    let unit = step.queries()[0].units[0];
    let served = step
        .unit(unit)
        .rows
        .iter()
        .map(|row| u128::from(row.0))
        .collect();
    (reported, served, !step.fits().is_empty())
}

#[test]
fn contexts_are_the_built_rows_a_stale_region_serves() {
    let mut node = node(64);
    let table = built_then_changed(&mut node, regioned(Rebuild::Churn(0.5)));

    let (reported, served, _) = reported_and_served(&mut node, table);

    assert_eq!(reported, served);
    assert!(
        !served.contains(&64),
        "the built rows, not the changed ones"
    );
}

#[test]
fn contexts_are_planned_afresh_once_a_regions_entry_is_gone() {
    let mut node = node(64);
    let table = built_then_changed(&mut node, regioned(Rebuild::Never));
    node.forget_cache();

    let (reported, served, _) = reported_and_served(&mut node, table);

    assert_eq!(reported, served);
    assert!(served.contains(&64), "a context planned with the new rows");
}

#[test]
fn a_region_whose_pinned_context_lost_a_row_is_built_again() {
    let mut node = node(64);
    let table = node.create_table(regioned(Rebuild::Never)).unwrap();
    let (keys, features, targets) = rows(64);
    node.insert(table, &keys, &features, targets).unwrap();
    node.submit(table, &A).unwrap();
    run(&mut node);
    node.submit(table, &A).unwrap();
    let running = node.schedule().unwrap().unwrap();
    assert_eq!(actions(&running), [Action::Reuse]);
    let deleted = running.units()[0].rows[0];

    node.delete(table, &[keys[deleted.0 as usize]]).unwrap();
    node.submit(table, &A).unwrap();
    let next = run(&mut node);
    node.complete(&running);

    assert_ne!(next.units()[0].key, running.units()[0].key);
    assert!(!next.units()[0].rows.contains(&deleted));
}

#[test]
fn contexts_foresee_the_refit_a_prediction_would_make() {
    let mut node = node(64);
    let spec = TableSpec {
        refit: RefitPolicy::new(Some(0.05), None).unwrap(),
        ..regioned(Rebuild::Never)
    };
    let table = built_then_changed(&mut node, spec);

    let (reported, served, refit) = reported_and_served(&mut node, table);

    assert!(refit);
    assert_eq!(reported, served);
}

#[test]
fn contexts_while_a_rebuild_runs_are_what_a_prediction_then_reads() {
    let mut node = node(64);
    let table = built_then_changed(&mut node, regioned(Rebuild::Always));
    let rebuild = node.schedule().unwrap().expect("a rebuild is due");

    let (reported, served, _) = reported_and_served(&mut node, table);
    node.complete(&rebuild);

    assert_eq!(actions(&rebuild), [Action::Encode]);
    assert_eq!(reported, served);
}

mod series {
    //! Tables of series: observations keyed by series and time, forecast in groups.

    use std::num::{NonZeroU64, NonZeroUsize};

    use gf_core::forecasting::{
        ForecastError, GroupStrategy, KNOWN, Neighbors, PAST, SeriesSpec, TARGET,
    };
    use gf_core::node::{Action, Node, NodeError, TableId, Targets, Task};
    use gf_core::planner::Strategy;
    use gf_core::series::{SeriesError, SeriesKey};

    use super::{node, run, spec};

    const A: SeriesKey = SeriesKey(1);
    const B: SeriesKey = SeriesKey(2);

    fn series_spec(known: Vec<usize>, group: GroupStrategy) -> SeriesSpec {
        SeriesSpec {
            step: NonZeroU64::new(10).unwrap(),
            patch: NonZeroUsize::new(4).unwrap(),
            context: NonZeroUsize::new(16).unwrap(),
            max_horizon: NonZeroUsize::new(32).unwrap(),
            max_rows: None,
            known,
            group,
            neighbors: Neighbors::Targets,
            reusable: true,
            extendable: true,
            patch_bytes: 0,
        }
    }

    /// A series table with A at 100..=150 and B at 120..=160, each row's
    /// target its time and its two features the time plus a half and a tenth.
    fn table(node: &mut Node, known: Vec<usize>, group: GroupStrategy) -> TableId {
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(series_spec(known, group));
        let table = node.create_table(spec).unwrap();
        let placed: Vec<(SeriesKey, i64)> = (100..=150)
            .step_by(10)
            .map(|time| (A, time))
            .chain((120..=160).step_by(10).map(|time| (B, time)))
            .collect();
        insert(node, table, &placed);
        table
    }

    fn insert(node: &mut Node, table: TableId, placed: &[(SeriesKey, i64)]) {
        let keys: Vec<u128> = placed
            .iter()
            .map(|&(series, time)| series.row_key(time).0)
            .collect();
        let features: Vec<f32> = placed
            .iter()
            .flat_map(|&(_, time)| [time as f32 + 0.5, time as f32 + 0.1])
            .collect();
        let targets = Targets::Values(placed.iter().map(|&(_, time)| time as f32).collect());
        let times: Vec<Option<i64>> = placed.iter().map(|&(_, time)| Some(time)).collect();
        node.insert_at(table, &keys, &features, targets, Some(&times))
            .unwrap();
    }

    fn horizon(steps: usize) -> NonZeroUsize {
        NonZeroUsize::new(steps).unwrap()
    }

    #[test]
    fn forecast_horizon_uses_the_model_pass_limit() {
        let mut node = node(64);
        let table = table(&mut node, vec![], GroupStrategy::Alone);

        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(32), None)
            .unwrap();
        assert!(matches!(
            node.submit_forecast(table, &[A.0], Some(&[160]), horizon(33), None),
            Err(NodeError::Forecast(ForecastError::HorizonTooLong {
                horizon: 33,
                limit: 32
            }))
        ));
        assert_eq!(
            ForecastError::HorizonTooLong {
                horizon: 33,
                limit: 32
            }
            .to_string(),
            "a horizon of 33 steps is past the 32 the model forecasts in 64 passes"
        );
        assert!(node.schedule().unwrap().is_some());

        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(17), None)
            .unwrap();
    }

    #[test]
    fn invalid_series_queries_and_empty_candidates_preserve_forecasts() {
        let mut node = node(64);
        let table = table(&mut node, vec![], GroupStrategy::Alone);
        let empty = node.create_table(spec(Strategy::Exact)).unwrap();
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(2), None)
            .unwrap();

        assert!(matches!(
            node.candidates(empty, &[1.0, 2.0], None, None),
            Err(NodeError::Unscaled)
        ));
        assert_eq!(
            node.contexts(table, &[1.0, 2.0], None),
            Err(NodeError::Forecast(ForecastError::OnlyForecasts))
        );
        assert_eq!(node.schedule().unwrap().unwrap().forecasts().len(), 1);
    }

    #[test]
    fn a_series_table_restored_from_its_compaction_forecasts_as_before() {
        let mut original = node(64);
        let table = table(&mut original, vec![], GroupStrategy::Alone);
        let compaction = original.compaction(table).unwrap();
        assert_eq!(compaction.rows.keys.len(), 11);

        let mut copy = node(64);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(series_spec(vec![], GroupStrategy::Alone));
        let restored = copy.create_table(spec).unwrap();
        copy.restore(
            restored,
            &compaction.rows,
            &compaction.layout,
            Some(&compaction.table),
        )
        .unwrap();

        assert_eq!(copy.compaction(restored).unwrap(), compaction);
        let mut grids = Vec::new();
        for (node, table) in [(&mut original, table), (&mut copy, restored)] {
            node.submit_forecast(table, &[A.0, B.0], Some(&[160, 160]), horizon(5), None)
                .unwrap();
            let step = run(node);
            let grid = &step.series_units()[0].grid;
            grids.push((
                grid.rows,
                grid.steps,
                grid.values.clone(),
                grid.observed.clone(),
            ));
        }
        assert_eq!(grids[0].0, 3);
        assert_eq!(grids[0], grids[1]);
    }

    #[test]
    fn a_cutoff_advanced_by_whole_patches_extends_the_kept_unit() {
        let mut node = node(64);
        // Caches padded to the context's patches leave no room to grow, so
        // this table pads to no minimum.
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(series_spec(vec![], GroupStrategy::Alone));
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec).unwrap();
        let placed: Vec<(SeriesKey, i64)> = (100..=150).step_by(10).map(|time| (A, time)).collect();
        insert(&mut node, table, &placed);
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        let first = run(&mut node);
        let kept = &first.series_units()[0];
        assert_eq!((kept.placed.action(), kept.grid.steps), (Action::Encode, 8));
        let (key, slots) = (kept.key, kept.placed.slots().to_vec());

        // Four more steps, one patch, and the cutoff four steps on: the grid
        // grows by the patch and the kept encoding is extended.
        insert(&mut node, table, &[(A, 160), (A, 170), (A, 180), (A, 190)]);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let second = run(&mut node);
        let grown = &second.series_units()[0];
        assert_eq!(grown.placed.action(), Action::Extend);
        assert_eq!((grown.grid.steps, grown.grown), (12, 4));
        assert_eq!(grown.source, Some(key));
        assert_eq!(grown.placed.source(), slots.as_slice());
        assert!(!grown.placed.slots().is_empty() && grown.placed.slots() != slots.as_slice());

        // An advance of five steps is not whole patches: encoded afresh.
        insert(
            &mut node,
            table,
            &[(A, 200), (A, 210), (A, 220), (A, 230), (A, 240)],
        );
        node.submit_forecast(table, &[A.0], Some(&[250]), horizon(4), None)
            .unwrap();
        let third = run(&mut node);
        assert_eq!(third.series_units()[0].placed.action(), Action::Encode);
        assert_eq!(third.series_units()[0].source, None);
    }

    /// A table whose caches pad to no minimum within a context of 16
    /// patches, on a node of `device_slots`, with A at 100..=150 encoded at
    /// cutoff 160 (8 steps, 2 patches) and 160..=190 inserted since.
    fn kept(device_slots: u32, group: GroupStrategy) -> (Node, TableId) {
        let mut node = node(device_slots);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(vec![], group)
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec).unwrap();
        let placed: Vec<(SeriesKey, i64)> = (100..=150).step_by(10).map(|time| (A, time)).collect();
        insert(&mut node, table, &placed);
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        let first = run(&mut node);
        assert_eq!(first.series_units()[0].placed.action(), Action::Encode);
        let later: Vec<(SeriesKey, i64)> = (160..=190).step_by(10).map(|time| (A, time)).collect();
        insert(&mut node, table, &later);
        (node, table)
    }

    #[test]
    fn a_change_inside_the_prefix_encodes_afresh() {
        // A late value, a correction and a deletion before the earlier cutoff
        // each change the rows the kept entry encoded.
        let (mut late, table) = kept(64, GroupStrategy::Alone);
        insert(&mut late, table, &[(A, 90)]);
        late.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let step = run(&mut late);
        assert_eq!(step.series_units()[0].grid.steps, 12);
        assert_eq!(step.series_units()[0].placed.action(), Action::Encode);

        let (mut corrected, table) = kept(64, GroupStrategy::Alone);
        corrected
            .update_at(
                table,
                &[A.row_key(120).0],
                &[1.0, 2.0],
                Targets::Values(vec![999.0]),
                Some(&[Some(120)]),
            )
            .unwrap();
        corrected
            .submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        assert_eq!(
            run(&mut corrected).series_units()[0].placed.action(),
            Action::Encode
        );

        let (mut deleted, table) = kept(64, GroupStrategy::Alone);
        deleted.delete(table, &[A.row_key(120).0]).unwrap();
        deleted
            .submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        assert_eq!(
            run(&mut deleted).series_units()[0].placed.action(),
            Action::Encode
        );
    }

    #[test]
    fn a_group_whose_members_changed_encodes_afresh() {
        let mut node = node(64);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(
                vec![],
                GroupStrategy::Random {
                    size: NonZeroUsize::new(2).unwrap(),
                    seed: 5,
                },
            )
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec).unwrap();
        for series in [A, B] {
            let rows: Vec<(SeriesKey, i64)> =
                (100..=150).step_by(10).map(|time| (series, time)).collect();
            insert(&mut node, table, &rows);
        }
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        let first = run(&mut node);
        assert_eq!(first.series_units()[0].placed.action(), Action::Encode);
        for series in [A, B] {
            let rows: Vec<(SeriesKey, i64)> =
                (160..=190).step_by(10).map(|time| (series, time)).collect();
            insert(&mut node, table, &rows);
        }
        // A third series may change A's drawn partner: then the unit is
        // another group's and encodes; the same group extends.
        let c = SeriesKey(3);
        let rows: Vec<(SeriesKey, i64)> = (100..=190).step_by(10).map(|time| (c, time)).collect();
        insert(&mut node, table, &rows);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let step = run(&mut node);
        let unit = &step.series_units()[0];
        let expected = if unit.members == first.series_units()[0].members {
            Action::Extend
        } else {
            Action::Encode
        };
        assert_eq!(unit.placed.action(), expected);
    }

    #[test]
    fn an_extension_keeps_its_source_on_the_device_while_making_room() {
        // Rows bucket to 4 of 10 bytes, two patches a slot: the kept entry of
        // 2 patches takes 1 slot and its extension of 3 takes 2, so on two
        // device slots the extension would have to evict its own source.
        let (mut node, table) = kept(2, GroupStrategy::Alone);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let step = run(&mut node);
        let unit = &step.series_units()[0];
        assert_ne!(unit.placed.action(), Action::Extend);
        assert!(
            step.movements()
                .iter()
                .all(|movement| Some(movement.key) != unit.source)
        );
    }

    #[test]
    fn a_failed_extension_is_forgotten_and_the_source_extended_again() {
        let (mut node, table) = kept(64, GroupStrategy::Alone);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let step = node.schedule().unwrap().unwrap();
        assert_eq!(step.series_units()[0].placed.action(), Action::Extend);
        let source = step.series_units()[0].source;
        node.fail(&step);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let again = run(&mut node);
        assert_eq!(again.series_units()[0].placed.action(), Action::Extend);
        assert_eq!(again.series_units()[0].source, source);
    }

    #[test]
    fn a_unit_extends_only_an_entry_a_completed_step_wrote() {
        // Two cutoffs of one group in one step: the later cannot extend the
        // earlier, which the step itself is encoding.
        let mut node = node(64);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(vec![], GroupStrategy::Alone)
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec).unwrap();
        let rows: Vec<(SeriesKey, i64)> = (100..=190).step_by(10).map(|time| (A, time)).collect();
        insert(&mut node, table, &rows);
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let step = run(&mut node);
        let actions: Vec<Action> = step
            .series_units()
            .iter()
            .map(|unit| unit.placed.action())
            .collect();
        assert_eq!(actions, [Action::Encode, Action::Encode]);

        // Nor an entry a step in flight is still encoding.
        let (mut node, table) = kept(64, GroupStrategy::Alone);
        let rows: Vec<(SeriesKey, i64)> = (200..=230).step_by(10).map(|time| (A, time)).collect();
        insert(&mut node, table, &rows);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let first = node.schedule().unwrap().unwrap();
        assert_eq!(first.series_units()[0].placed.action(), Action::Extend);
        node.submit_forecast(table, &[A.0], Some(&[240]), horizon(4), None)
            .unwrap();
        let second = node.schedule().unwrap().unwrap();
        // The entry at 160 is written, two patches back: that one is grown.
        assert_eq!(second.series_units()[0].placed.action(), Action::Extend);
        assert_eq!(
            second.series_units()[0].source,
            first.series_units()[0].source
        );
        assert_ne!(
            second.series_units()[0].source,
            Some(first.series_units()[0].key)
        );
        node.complete(&first);
        node.complete(&second);
    }

    #[test]
    fn a_kept_unit_whose_entry_is_gone_or_doomed_or_of_an_old_epoch_is_not_extended() {
        // Evicted: three device slots hold A's kept unit until B's and C's
        // encodes push it out, so A's advance encodes.
        let mut node = super::node_with_host(3, 0);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(vec![], GroupStrategy::Alone)
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec.clone()).unwrap();
        let c = SeriesKey(3);
        for series in [A, B, c] {
            let rows: Vec<(SeriesKey, i64)> =
                (100..=190).step_by(10).map(|time| (series, time)).collect();
            insert(&mut node, table, &rows);
        }
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        run(&mut node);
        for series in [B, c] {
            node.submit_forecast(table, &[series.0], Some(&[200]), horizon(4), None)
                .unwrap();
            run(&mut node);
        }
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        assert_eq!(
            run(&mut node).series_units()[0].placed.action(),
            Action::Encode
        );

        // Doomed: a row of the kept entry changes while the step encoding it
        // runs, so it is dropped as the step completes, and never a source.
        let (mut node, table) = kept_in_flight();
        node.update_at(
            table,
            &[A.row_key(120).0],
            &[1.0, 2.0],
            Targets::Values(vec![7.0]),
            Some(&[Some(120)]),
        )
        .unwrap();
        assert_eq!(advance(&mut node, table), Action::Encode);

        // An epoch that ends while the step encoding the kept entry runs.
        let (mut node, table) = kept_in_flight();
        node.advance_epoch(table).unwrap();
        let step = finish_and_advance(&mut node, table);
        assert_eq!(step.series_units()[0].placed.action(), Action::Encode);
        assert_eq!(step.series_units()[0].key.epoch.0, 1);

        // The cache forgotten while the step runs.
        let (mut node, table) = kept_in_flight();
        node.forget_cache();
        assert_eq!(advance(&mut node, table), Action::Encode);

        // An epoch that ends while a later step reuses the kept entry: the
        // next advance encodes in the new epoch rather than growing the old.
        let (mut node, table) = kept(64, GroupStrategy::Alone);
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        let reading = node.schedule().unwrap().unwrap();
        assert_eq!(reading.series_units()[0].placed.action(), Action::Reuse);
        node.advance_epoch(table).unwrap();
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let step = node.schedule().unwrap().unwrap();
        assert_eq!(step.series_units()[0].placed.action(), Action::Encode);
        node.complete(&reading);
        node.complete(&step);
    }

    /// A node with A at 100..=150 whose encode at cutoff 160 is in flight,
    /// and 160..=190 inserted: `advance` completes it and forecasts at 200.
    fn kept_in_flight() -> (Node, TableId) {
        let mut node = node(64);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(vec![], GroupStrategy::Alone)
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec).unwrap();
        let rows: Vec<(SeriesKey, i64)> = (100..=150).step_by(10).map(|time| (A, time)).collect();
        insert(&mut node, table, &rows);
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        let first = node.schedule().unwrap().unwrap();
        assert_eq!(first.series_units()[0].placed.action(), Action::Encode);
        IN_FLIGHT.with(|held| *held.borrow_mut() = Some(first));
        (node, table)
    }

    thread_local! {
        static IN_FLIGHT: std::cell::RefCell<Option<gf_core::node::Step>> = const { std::cell::RefCell::new(None) };
    }

    fn finish_and_advance(node: &mut Node, table: TableId) -> gf_core::node::Step {
        let first = IN_FLIGHT.with(|held| held.borrow_mut().take()).unwrap();
        node.complete(&first);
        let rows: Vec<(SeriesKey, i64)> = (160..=190).step_by(10).map(|time| (A, time)).collect();
        insert(node, table, &rows);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        run(node)
    }

    fn advance(node: &mut Node, table: TableId) -> Action {
        finish_and_advance(node, table).series_units()[0]
            .placed
            .action()
    }

    #[test]
    fn a_restored_table_encodes_once_then_extends() {
        let (original, table) = kept(64, GroupStrategy::Alone);
        let compaction = original.compaction(table).unwrap();
        let mut copy = node(64);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(vec![], GroupStrategy::Alone)
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let restored = copy.create_table(spec).unwrap();
        copy.restore(
            restored,
            &compaction.rows,
            &compaction.layout,
            Some(&compaction.table),
        )
        .unwrap();
        copy.submit_forecast(restored, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        assert_eq!(
            run(&mut copy).series_units()[0].placed.action(),
            Action::Encode
        );
        let rows: Vec<(SeriesKey, i64)> = (200..=230).step_by(10).map(|time| (A, time)).collect();
        insert(&mut copy, restored, &rows);
        copy.submit_forecast(restored, &[A.0], Some(&[240]), horizon(4), None)
            .unwrap();
        assert_eq!(
            run(&mut copy).series_units()[0].placed.action(),
            Action::Extend
        );
    }

    #[test]
    fn kept_units_past_the_prune_mark_are_pruned_not_forgotten() {
        // 4200 groups, each kept at cutoff 160 with its entry cached: every
        // advance extends, where clearing the map past 4096 would extend none.
        let mut node = super::node_with_host(200_000, 0);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(vec![], GroupStrategy::Alone)
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec).unwrap();
        let series: Vec<SeriesKey> = (1..=4200).map(SeriesKey).collect();
        let rows: Vec<(SeriesKey, i64)> = series
            .iter()
            .flat_map(|&s| (100..=190).step_by(10).map(move |t| (s, t)))
            .collect();
        insert(&mut node, table, &rows);
        let keys: Vec<u64> = series.iter().map(|s| s.0).collect();
        for chunk in keys.chunks(512) {
            node.submit_forecast(
                table,
                chunk,
                Some(&vec![160; chunk.len()]),
                horizon(4),
                None,
            )
            .unwrap();
            while let Some(step) = node.schedule().unwrap() {
                node.complete(&step);
            }
        }
        let (mut extended, mut total) = (0, 0);
        for chunk in keys.chunks(512) {
            node.submit_forecast(
                table,
                chunk,
                Some(&vec![200; chunk.len()]),
                horizon(4),
                None,
            )
            .unwrap();
            while let Some(step) = node.schedule().unwrap() {
                for unit in step.series_units() {
                    total += 1;
                    extended += usize::from(unit.placed.action() == Action::Extend);
                }
                node.complete(&step);
            }
        }
        assert_eq!((extended, total), (4200, 4200));
    }

    #[test]
    fn a_source_doomed_while_its_extension_runs_is_dropped() {
        let (mut node, table) = kept(64, GroupStrategy::Alone);
        node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
            .unwrap();
        let step = node.schedule().unwrap().unwrap();
        assert_eq!(step.series_units()[0].placed.action(), Action::Extend);
        let source = step.series_units()[0].source.unwrap();
        node.update_at(
            table,
            &[A.row_key(120).0],
            &[1.0, 2.0],
            Targets::Values(vec![7.0]),
            Some(&[Some(120)]),
        )
        .unwrap();
        let drops = node.complete(&step);
        assert!(
            drops
                .iter()
                .any(|movement| movement.key == source && movement.to.is_none())
        );
    }

    #[test]
    fn an_earlier_cutoff_leaves_the_later_entry_as_the_one_extended() {
        for same_step in [false, true] {
            let (mut node, table) = kept(64, GroupStrategy::Alone);
            node.submit_forecast(table, &[A.0], Some(&[200]), horizon(4), None)
                .unwrap();
            let mut units = Vec::new();
            if !same_step {
                units.extend(run(&mut node).series_units().to_vec());
            }
            // A historical forecast encodes an earlier cutoff, written after
            // the later one.
            node.submit_forecast(table, &[A.0], Some(&[120]), horizon(4), None)
                .unwrap();
            units.extend(run(&mut node).series_units().to_vec());
            let later = units.iter().find(|unit| unit.cutoff == 200).unwrap();
            let earlier = units.iter().find(|unit| unit.cutoff == 120).unwrap();
            assert_eq!(later.placed.action(), Action::Extend, "{same_step}");
            assert_eq!(earlier.placed.action(), Action::Encode, "{same_step}");

            let rows: Vec<(SeriesKey, i64)> =
                (200..=230).step_by(10).map(|time| (A, time)).collect();
            insert(&mut node, table, &rows);
            node.submit_forecast(table, &[A.0], Some(&[240]), horizon(4), None)
                .unwrap();
            let step = run(&mut node);
            let unit = &step.series_units()[0];
            assert_eq!(unit.placed.action(), Action::Extend, "{same_step}");
            assert_eq!(unit.source, Some(later.key), "{same_step}");
        }
    }

    #[test]
    fn tables_extended_in_one_step_stay_within_its_budget() {
        // 100..=180 at cutoff 190: 3 patches padded to 4, two to a slot, then
        // a patch on. Whatever an extension holds, a step of several tables
        // writes no more slots than its budget of 5.
        let mut node = node(64);
        let tables: Vec<TableId> = (0..2)
            .map(|_| {
                let mut spec = spec(Strategy::Exact);
                spec.task = Task::Values;
                spec.series = Some(SeriesSpec {
                    context: NonZeroUsize::new(64).unwrap(),
                    ..series_spec(vec![], GroupStrategy::Alone)
                });
                spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
                let table = node.create_table(spec).unwrap();
                let rows: Vec<(SeriesKey, i64)> =
                    (100..=180).step_by(10).map(|time| (A, time)).collect();
                insert(&mut node, table, &rows);
                node.submit_forecast(table, &[A.0], Some(&[190]), horizon(4), None)
                    .unwrap();
                table
            })
            .collect();
        run(&mut node);
        node.set_step_bytes(400);
        for &table in &tables {
            let rows: Vec<(SeriesKey, i64)> =
                (190..=220).step_by(10).map(|time| (A, time)).collect();
            insert(&mut node, table, &rows);
            node.submit_forecast(table, &[A.0], Some(&[230]), horizon(4), None)
                .unwrap();
        }
        let mut served = 0;
        while served < tables.len() {
            let step = run(&mut node);
            let units = step.series_units();
            assert!(
                units
                    .iter()
                    .all(|unit| unit.placed.action() == Action::Extend)
            );
            let slots: usize = units.iter().map(|unit| unit.placed.slots().len()).sum();
            assert!(units.len() == 1 || slots * 80 <= 400, "{slots} slots");
            served += units.len();
        }
    }

    #[test]
    fn chained_extensions_are_sized_for_the_patches_the_worker_holds() {
        // 100..=180 at cutoff 190: 9 steps on a grid of 12, 3 patches padded
        // to 4, which the worker stores; two patches fill a slot. Each round
        // adds a patch, and the slots given hold the stored patches padded to
        // their bucket.
        let mut node = node(64);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(SeriesSpec {
            context: NonZeroUsize::new(64).unwrap(),
            ..series_spec(vec![], GroupStrategy::Alone)
        });
        spec.sizes.bucket_minimum = NonZeroUsize::new(1).unwrap();
        let table = node.create_table(spec).unwrap();
        let rows: Vec<(SeriesKey, i64)> = (100..=180).step_by(10).map(|time| (A, time)).collect();
        insert(&mut node, table, &rows);
        node.submit_forecast(table, &[A.0], Some(&[190]), horizon(4), None)
            .unwrap();
        let first = run(&mut node);
        assert_eq!(first.series_units()[0].placed.slots().len(), 2);
        let mut stored: usize = 4;
        for round in 1..=4 {
            let cutoff = 190 + 40 * round;
            let rows: Vec<(SeriesKey, i64)> = (cutoff - 40..cutoff)
                .step_by(10)
                .map(|time| (A, time))
                .collect();
            insert(&mut node, table, &rows);
            node.submit_forecast(table, &[A.0], Some(&[cutoff]), horizon(4), None)
                .unwrap();
            let step = run(&mut node);
            let unit = &step.series_units()[0];
            assert_eq!(unit.placed.action(), Action::Extend, "round {round}");
            stored += 1;
            assert_eq!(
                unit.placed.slots().len(),
                stored.next_power_of_two().div_ceil(2),
                "round {round}"
            );
        }
    }

    #[test]
    fn given_groups_are_forecast_as_named_and_shared() {
        use gf_core::node::ErrorKind;

        let mut node = node(64);
        let given = GroupStrategy::Given {
            size: NonZeroUsize::new(2).unwrap(),
        };
        let given = table(&mut node, vec![], given);
        let d = SeriesKey(4);
        let rows: Vec<(SeriesKey, i64)> = (120..=160).step_by(10).map(|time| (d, time)).collect();
        insert(&mut node, given, &rows);

        // A names B and B names both: one group, so one unit serves both.
        let groups = [vec![B.0], vec![A.0, B.0]];
        node.submit_forecast(
            given,
            &[A.0, B.0],
            Some(&[160, 160]),
            horizon(5),
            Some(&groups),
        )
        .unwrap();
        let step = run(&mut node);
        assert_eq!(step.series_units().len(), 1);
        assert_eq!(step.series_units()[0].members, [A, B]);
        assert_eq!(step.forecasts().len(), 2);

        // A series the given never saw is forecast through the group given it.
        let c = SeriesKey(3);
        node.submit_forecast(given, &[c.0], Some(&[160]), horizon(5), Some(&[vec![A.0]]))
            .unwrap();
        let step = run(&mut node);
        assert_eq!(step.series_units()[0].members, [A, c]);

        // Refused: a forecast without its group, a group past the given's
        // size, a member the given lacks (not found, as a row key given to
        // `predict_given` is), and groups given to a given drawing its own.
        node.submit_forecast(given, &[A.0], Some(&[160]), horizon(5), None)
            .unwrap();
        assert!(node.schedule().unwrap().is_none());
        assert!(matches!(
            node.take_refused()[0].1,
            NodeError::Forecast(ForecastError::NoGroup(A))
        ));
        node.submit_forecast(
            given,
            &[A.0],
            Some(&[160]),
            horizon(5),
            Some(&[vec![B.0, d.0]]),
        )
        .unwrap();
        assert!(node.schedule().unwrap().is_none());
        assert!(matches!(
            node.take_refused()[0].1,
            NodeError::Forecast(ForecastError::GroupPastSize { given: 3, size: 2 })
        ));
        node.submit_forecast(
            given,
            &[A.0],
            Some(&[160]),
            horizon(5),
            Some(&[vec![SeriesKey(9).0]]),
        )
        .unwrap();
        assert!(node.schedule().unwrap().is_none());
        let refused = node.take_refused().pop().unwrap().1;
        assert!(matches!(
            refused,
            NodeError::Forecast(ForecastError::UnknownMember(SeriesKey(9)))
        ));
        assert_eq!(refused.kind(), ErrorKind::NotFound);
        let drawn = table(
            &mut node,
            vec![],
            GroupStrategy::Random {
                size: NonZeroUsize::new(2).unwrap(),
                seed: 5,
            },
        );
        node.submit_forecast(drawn, &[A.0], Some(&[160]), horizon(5), Some(&[vec![B.0]]))
            .unwrap();
        assert!(node.schedule().unwrap().is_none());
        assert!(matches!(
            node.take_refused()[0].1,
            NodeError::Forecast(ForecastError::UngivenGroups)
        ));
        assert_eq!(
            node.submit_forecast(drawn, &[A.0, B.0], None, horizon(5), Some(&[vec![B.0]]))
                .unwrap_err()
                .to_string(),
            "expected 2 groups, got 1"
        );
    }

    #[test]
    fn a_forecast_plans_a_unit_on_a_grid_and_a_later_one_reuses_it() {
        let mut node = node(64);
        let table = table(&mut node, vec![], GroupStrategy::Alone);
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(5), None)
            .unwrap();

        let step = run(&mut node);

        assert!(step.units().is_empty() && step.fits().is_empty());
        assert_eq!(step.series_units().len(), 1);
        let unit = &step.series_units()[0];
        assert_eq!(unit.placed.action(), Action::Encode);
        assert_eq!(unit.members, [A]);
        assert_eq!(unit.cutoff, 160);
        // Six steps before 160, padded to eight; a target row and two columns.
        assert_eq!((unit.grid.rows, unit.grid.steps), (3, 8));
        assert_eq!(unit.grid.roles, [TARGET, PAST, PAST]);
        assert_eq!(
            unit.grid.values[2..8],
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        );
        assert_eq!(unit.grid.values[8 + 2], 100.5);
        assert_eq!(unit.grid.values[16 + 2], 100.1);
        assert!(!unit.grid.observed[0] && unit.grid.observed[2]);
        assert_eq!(step.forecasts().len(), 1);
        let forecast = &step.forecasts()[0];
        assert_eq!(
            (forecast.unit.0, forecast.target, forecast.horizon),
            (0, 0, 5)
        );
        // Without known columns, the future carries no rows.
        assert_eq!((forecast.future.rows, forecast.future.steps), (0, 5));

        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(9), None)
            .unwrap();
        let again = run(&mut node);
        assert_eq!(again.series_units()[0].placed.action(), Action::Reuse);
        assert_eq!(again.forecasts()[0].horizon, 9);
    }

    #[test]
    fn known_columns_make_units_transient_and_fill_the_future() {
        let mut node = node(64);
        let table = table(&mut node, vec![1], GroupStrategy::Alone);
        node.submit_forecast(table, &[B.0], Some(&[140]), horizon(3), None)
            .unwrap();

        let step = run(&mut node);

        let unit = &step.series_units()[0];
        assert_eq!(unit.placed.action(), Action::Transient);
        assert_eq!(unit.grid.roles, [TARGET, PAST, KNOWN]);
        let future = &step.forecasts()[0].future;
        assert_eq!((future.rows, future.steps), (3, 3));
        assert_eq!(future.observed[6..9], [true, true, true]);
        assert_eq!(future.values[6..9], [140.1, 150.1, 160.1]);
        assert!(future.observed[..6].iter().all(|&seen| !seen));
    }

    #[test]
    fn groups_share_a_unit_across_requests() {
        let mut node = node(64);
        let group = GroupStrategy::Random {
            size: NonZeroUsize::new(2).unwrap(),
            seed: 3,
        };
        let table = table(&mut node, vec![], group);
        node.submit_forecast(table, &[A.0], Some(&[160]), horizon(4), None)
            .unwrap();
        node.submit_forecast(table, &[B.0], None, horizon(4), None)
            .unwrap();

        let step = run(&mut node);

        // Two cutoffs, 160 and the latest plus a step, 170: two units of both.
        assert_eq!(step.series_units().len(), 2);
        assert!(
            step.series_units()
                .iter()
                .all(|unit| unit.members == [A, B])
        );
        assert_eq!(step.forecasts().len(), 2);
        assert_eq!(step.forecasts()[1].target, 1);
        assert_eq!(step.series_units()[1].cutoff, 170);
    }

    #[test]
    fn series_rows_need_their_time_and_grid_and_tables_of_series_take_no_queries() {
        let mut node = node(64);
        let table = table(&mut node, vec![], GroupStrategy::Alone);
        let key = A.row_key(165).0;

        let untimed = node.insert(table, &[key], &[0.0, 0.0], Targets::Values(vec![1.0]));
        assert_eq!(
            untimed.unwrap_err(),
            NodeError::Forecast(ForecastError::Untimed)
        );
        let off_grid = node.insert_at(
            table,
            &[key],
            &[0.0, 0.0],
            Targets::Values(vec![1.0]),
            Some(&[Some(165)]),
        );
        assert!(matches!(
            off_grid.unwrap_err(),
            NodeError::Forecast(ForecastError::Series(_))
        ));
        assert!(matches!(
            node.submit(table, &[0.0, 0.0]).unwrap_err(),
            NodeError::Forecast(ForecastError::OnlyForecasts)
        ));
        let plain = super::table(&mut node, Strategy::Exact, 4);
        assert!(matches!(
            node.submit_forecast(plain, &[1], None, horizon(2), None)
                .unwrap_err(),
            NodeError::Forecast(ForecastError::NotSeries)
        ));
    }

    #[test]
    fn deletes_and_renumbering_keep_the_series_index() {
        let mut node = node(64);
        let table = table(&mut node, vec![], GroupStrategy::Alone);
        node.delete(table, &[A.row_key(150).0]).unwrap();
        // Replacing the rows renumbers them densely in key order.
        let placed: Vec<(SeriesKey, i64)> = (100..=140).step_by(10).map(|time| (A, time)).collect();
        let keys: Vec<u128> = placed.iter().map(|&(s, t)| s.row_key(t).0).collect();
        let features: Vec<f32> = placed
            .iter()
            .flat_map(|&(_, time)| [time as f32 + 0.5, time as f32 + 0.1])
            .collect();
        let targets = Targets::Values(placed.iter().map(|&(_, time)| time as f32).collect());
        let times: Vec<Option<i64>> = placed.iter().map(|&(_, time)| Some(time)).collect();
        node.replace_rows(table, &keys, &features, targets, Some(&times))
            .unwrap();
        node.submit_forecast(table, &[A.0], Some(&[150]), horizon(2), None)
            .unwrap();

        let step = run(&mut node);

        let unit = &step.series_units()[0];
        assert_eq!((unit.grid.rows, unit.grid.steps), (3, 8));
        assert_eq!(
            unit.grid.observed[..8],
            [false, false, false, true, true, true, true, true]
        );
        assert_eq!(unit.grid.values[3..8], [100.0, 110.0, 120.0, 130.0, 140.0]);
    }

    #[test]
    fn a_batch_leaving_a_series_grid_is_refused_whole_before_anything_changes() {
        let mut node = node(64);
        let table = table(&mut node, vec![], GroupStrategy::Alone);
        let c = SeriesKey(3);
        let rows = |placed: &[(SeriesKey, i64)]| {
            let keys: Vec<u128> = placed.iter().map(|&(s, t)| s.row_key(t).0).collect();
            let features: Vec<f32> = placed.iter().flat_map(|_| [0.5, 0.1]).collect();
            let targets = Targets::Values(placed.iter().map(|&(_, t)| t as f32).collect());
            let times: Vec<Option<i64>> = placed.iter().map(|&(_, t)| Some(t)).collect();
            (keys, features, targets, times)
        };
        let before = node.len(table).unwrap();

        // A new series' grid is set by its first row; its third leaves it.
        let (keys, features, targets, times) = rows(&[(c, 0), (c, 10), (c, 15)]);
        let refused = node.insert_at(table, &keys, &features, targets, Some(&times));
        assert!(matches!(
            refused,
            Err(NodeError::Forecast(ForecastError::Series(
                SeriesError::Irregular { .. }
            )))
        ));
        let (keys, features, targets, times) = rows(&[(c, 0), (c, 5)]);
        assert!(
            node.upsert_at(table, &keys, &features, targets, Some(&times))
                .is_err()
        );
        // A batch may not place one time twice, but may write over the table's.
        let (keys, features, targets, times) = rows(&[(A, 150), (A, 150)]);
        assert!(
            node.upsert_at(table, &keys, &features, targets, Some(&times))
                .is_err()
        );
        let (keys, features, targets, times) = rows(&[(A, 150), (A, 160)]);
        node.upsert_at(table, &keys, &features, targets, Some(&times))
            .unwrap();

        assert_eq!(node.len(table).unwrap(), before + 1);
        node.submit_forecast(table, &[A.0], Some(&[170]), horizon(2), None)
            .unwrap();
        let step = run(&mut node);
        let grid = &step.series_units()[0].grid;
        let target_row = &grid.observed[..grid.steps];
        assert_eq!(target_row.iter().filter(|&&seen| seen).count(), 7);
    }

    #[test]
    fn a_request_that_cannot_be_planned_is_refused_alone() {
        let mut node = node(64);
        let table = table(&mut node, vec![], GroupStrategy::Alone);
        // A series with no observed value has no time to forecast from.
        let c = SeriesKey(3);
        node.insert_at(
            table,
            &[c.row_key(100).0],
            &[0.5, 0.1],
            Targets::Values(vec![f32::NAN]),
            Some(&[Some(100)]),
        )
        .unwrap();
        let bad = node
            .submit_forecast(table, &[c.0], None, horizon(2), None)
            .unwrap();
        let good = node
            .submit_forecast(table, &[A.0], Some(&[160]), horizon(2), None)
            .unwrap();

        let step = run(&mut node);

        assert_eq!(step.forecasts()[0].request, good);
        let refused = node.take_refused();
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0].0, bad);
    }

    #[test]
    fn series_tables_refuse_unknown_known_columns_and_small_caps_at_creation() {
        let mut node = node(64);
        let mut spec = spec(Strategy::Exact);
        spec.task = Task::Values;
        spec.series = Some(series_spec(vec![2], GroupStrategy::Alone));
        assert!(matches!(
            node.create_table(spec.clone()),
            Err(NodeError::Forecast(ForecastError::KnownColumn {
                column: 2,
                features: 2
            }))
        ));
        let mut capped = series_spec(vec![], GroupStrategy::Alone);
        capped.max_rows = NonZeroUsize::new(2);
        spec.series = Some(capped);
        assert!(matches!(
            node.create_table(spec),
            Err(NodeError::Forecast(ForecastError::TooManyRows {
                rows: 3,
                cap: 2
            }))
        ));
    }

    #[test]
    fn units_forecast_whole_are_each_forecasts_own() {
        let mut node = node(64);
        let table = table(&mut node, vec![1], GroupStrategy::Alone);
        node.submit_forecast(table, &[B.0], Some(&[140]), horizon(3), None)
            .unwrap();
        node.submit_forecast(table, &[B.0], Some(&[140]), horizon(5), None)
            .unwrap();

        let step = run(&mut node);

        assert_eq!(step.series_units().len(), 2);
        assert!(
            step.series_units()
                .iter()
                .all(|unit| unit.placed.action() == Action::Transient)
        );
        assert_eq!(step.forecasts()[1].future.steps, 5);
    }

    #[test]
    fn a_forecast_without_a_cutoff_starts_after_the_series_last_observed_value() {
        let mut node = node(64);
        let table = table(&mut node, vec![1], GroupStrategy::Alone);
        // B's covariates stored ahead, with no target yet.
        let ahead = [170i64, 180];
        let keys: Vec<u128> = ahead.iter().map(|&time| B.row_key(time).0).collect();
        let features: Vec<f32> = ahead
            .iter()
            .flat_map(|&t| [t as f32 + 0.5, t as f32 + 0.1])
            .collect();
        let times: Vec<Option<i64>> = ahead.iter().map(|&time| Some(time)).collect();
        node.insert_at(
            table,
            &keys,
            &features,
            Targets::Values(vec![f32::NAN; 2]),
            Some(&times),
        )
        .unwrap();
        node.submit_forecast(table, &[B.0], None, horizon(2), None)
            .unwrap();

        let step = run(&mut node);

        assert_eq!(step.series_units()[0].cutoff, 170);
        let future = &step.forecasts()[0].future;
        assert_eq!(future.values[4..6], [170.1, 180.1]);
        // A series without an observed value has nothing to forecast from.
        let c = SeriesKey(3);
        node.insert_at(
            table,
            &[c.row_key(100).0],
            &[0.5, 0.1],
            Targets::Values(vec![f32::NAN]),
            Some(&[Some(100)]),
        )
        .unwrap();
        node.submit_forecast(table, &[c.0], None, horizon(2), None)
            .unwrap();
        assert!(node.schedule().unwrap().is_none());
        assert!(matches!(
            node.take_refused()[0].1,
            NodeError::Forecast(ForecastError::NoRows(_))
        ));
    }
}

#[test]
fn rows_held_unlabeled_first_take_the_contexts_of_rows_inserted_once_labeled() {
    let regioned = TableSpec {
        regions: Some(RegionSpec {
            rows: 16,
            rebuild: Rebuild::Always,
            wait: RebuildWait::default(),
        }),
        ..spec(Strategy::neighbors(8))
    };
    // A state fit on a sample of the rows.
    let sampled = TableSpec {
        fit_rows: NonZeroUsize::new(16).unwrap(),
        ..spec(Strategy::neighbors(8))
    };
    let specs = [
        spec(Strategy::subsample(8, 3, CoreKind::Random)),
        spec(Strategy::subsample(8, 3, CoreKind::Stratified)),
        spec(Strategy::subsample(2, 3, CoreKind::Diverse)),
        regioned,
        sampled,
    ];
    for spec in specs {
        let mut node = node(64);
        let (keys, features, Targets::Classes(labels)) = rows(64) else {
            unreachable!()
        };
        let part = |range: std::ops::Range<usize>| {
            (
                keys[range.clone()].to_vec(),
                features[2 * range.start..2 * range.end].to_vec(),
                Targets::Classes(labels[range].to_vec()),
            )
        };
        // One table takes each row once labeled, in key order; the other holds
        // keys 48 .. 64 unlabeled before keys 32 .. 48 arrive, so its rows
        // take other ids.
        let (first, late, early) = (part(0..32), part(32..48), part(48..64));
        let labeled = node.create_table(spec.clone()).unwrap();
        let held = node.create_table(spec).unwrap();
        for table in [labeled, held] {
            node.insert(table, &first.0, &first.1, first.2.clone())
                .unwrap();
        }
        node.insert(
            held,
            &early.0,
            &early.1,
            Targets::Classes(vec![UNLABELED; 16]),
        )
        .unwrap();
        for (keys, features, targets) in [&late, &early] {
            node.insert(labeled, keys, features, targets.clone())
                .unwrap();
        }
        node.insert(held, &late.0, &late.1, late.2.clone()).unwrap();
        node.label(held, &early.0, early.2.clone()).unwrap();

        let queries = [A, B].concat();
        assert_eq!(
            node.contexts(labeled, &queries, None).unwrap(),
            node.contexts(held, &queries, None).unwrap()
        );
        // The rows each state is fit on, and each context's rows in the order
        // the model reads them, by key.
        for table in [labeled, held] {
            node.submit(table, &queries).unwrap();
        }
        let step = run(&mut node);
        let read = |table: TableId| -> (Vec<u128>, Vec<Vec<u128>>) {
            let keys = |rows: &[RowId]| node.rows_of(table, rows).unwrap().keys;
            let fit = step.fits().iter().find(|fit| fit.table == table).unwrap();
            let mut fit = keys(&fit.rows);
            // A sampled fit reads its rows by key; a full one reads them as
            // its pages hold them.
            if fit.len() < 64 {
                assert!(fit.is_sorted());
            }
            fit.sort_unstable();
            let units = step.units().iter().filter(|unit| unit.table == table);
            (fit, units.map(|unit| keys(&unit.rows)).collect())
        };
        assert_eq!(read(labeled), read(held));
    }
}

#[test]
fn rows_at_equal_distances_tie_by_key_whatever_order_they_came_in() {
    // Eight rows at one point, so neighbors and a diverse core's later picks
    // all tie.
    let keys: Vec<u128> = (0..8).collect();
    let features: Vec<f32> = keys.iter().flat_map(|_| A).collect();
    for strategy in [
        Strategy::neighbors(3),
        Strategy::subsample(3, 3, CoreKind::Diverse),
    ] {
        let mut node = node(64);
        let read = |node: &mut Node, keys: &[u128]| {
            let table = node.create_table(spec(strategy)).unwrap();
            node.insert(table, keys, &features, Targets::Classes(vec![0; 8]))
                .unwrap();
            node.contexts(table, &A, None).unwrap()
        };
        let reversed: Vec<u128> = keys.iter().rev().copied().collect();
        assert_eq!(read(&mut node, &keys), read(&mut node, &reversed));
    }
}

/// A table of 32 labeled rows and 8 more, keys 32 .. 40, whose labels are
/// not known yet.
fn partly_labeled(node: &mut Node, spec: TableSpec) -> TableId {
    let table = node.create_table(spec).unwrap();
    let (keys, features, Targets::Classes(mut labels)) = rows(40) else {
        unreachable!()
    };
    labels[32..].fill(UNLABELED);
    node.insert(table, &keys, &features, Targets::Classes(labels))
        .unwrap();
    table
}

#[test]
fn unlabeled_rows_wait_out_of_fits_and_contexts_until_labeled() {
    let cases = [
        (false, SummaryScope::Table),
        (true, SummaryScope::Table),
        (false, SummaryScope::Context),
    ];
    for (unlabeled_context, summary) in cases {
        let mut node = node(64);
        let spec = TableSpec {
            unlabeled_context,
            summary,
            ..spec(Strategy::Exact)
        };
        let table = partly_labeled(&mut node, spec);
        let waiting: Vec<u128> = (32..40).collect();

        node.submit(table, &A).unwrap();
        let step = run(&mut node);
        // A table summarizing each context fits the unit on its own rows.
        let unit = &step.units()[0];
        let fit = match summary {
            SummaryScope::Table => step.fits()[0].rows.clone(),
            SummaryScope::Context => unit.rows.clone(),
        };
        let fit: HashSet<RowId> = fit.into_iter().collect();
        let read: HashSet<RowId> = unit.rows.iter().copied().collect();
        node.label(table, &waiting, Targets::Classes(vec![1; 8]))
            .unwrap();
        let labeled = node.contexts(table, &A, None).unwrap().remove(0);

        let all: HashSet<RowId> = (0..40).map(RowId).collect();
        let first: HashSet<RowId> = (0..32).map(RowId).collect();
        let held = if unlabeled_context { &all } else { &first };
        assert_eq!(
            (&fit, &read),
            (held, held),
            "{unlabeled_context} {summary:?}"
        );
        assert_eq!(labeled, (0..40).collect::<Vec<u128>>());
    }
    // A class below -1 is no label.
    let mut node = node(64);
    let table = node.create_table(spec(Strategy::Exact)).unwrap();
    assert!(matches!(
        node.insert(table, &[0], &A, Targets::Classes(vec![-2])),
        Err(NodeError::Label { label: -2, .. })
    ));
}

#[test]
fn a_row_counts_toward_its_regions_rebuild_once_it_is_labeled() {
    let mut node = node(64);
    let table = node.create_table(regioned(Rebuild::Churn(0.25))).unwrap();
    let (keys, features, targets) = rows(64);
    node.insert(table, &keys, &features, targets).unwrap();
    let queries = [A, B].concat();
    assert_eq!(node.region_churn(table, &queries), Ok(vec![None, None]));
    node.submit(table, &A).unwrap();
    let step = node.schedule().unwrap().expect("requests are pending");
    // A context still being encoded is not built yet.
    assert_eq!(node.region_churn(table, &queries), Ok(vec![None, None]));
    node.complete(&step);
    let context = node.contexts(table, &A, None).unwrap()[0].len() as f64;
    // A quarter of the region's 32 rows, at its middle, without labels.
    let (more, _, _) = rows(72);
    let added: Vec<f32> = (64..72).flat_map(|_| [0.155, 0.0]).collect();
    node.insert(
        table,
        &more[64..],
        &added,
        Targets::Classes(vec![UNLABELED; 8]),
    )
    .unwrap();

    assert!(node.schedule().unwrap().is_none());
    assert_eq!(
        node.region_churn(table, &queries),
        Ok(vec![Some(0.0), None])
    );
    node.label(table, &more[64..], Targets::Classes(vec![0; 8]))
        .unwrap();
    // B's region has no built context; A's has 8 changed rows on its context's.
    assert_eq!(
        node.region_churn(table, &queries),
        Ok(vec![Some(8.0 / context), None])
    );
    let rebuild = run_idle(&mut node);

    assert_eq!(actions(&rebuild), [Action::Encode]);
    assert!(rebuild.units()[0].rows.iter().any(|row| row.0 >= 64));
    assert_eq!(node.region_churn(table, &A), Ok(vec![Some(0.0)]));
    // Deleting a row of the context drops it from the cache, so none is held.
    let row = rebuild.units()[0].rows.first().unwrap();
    node.delete(table, &[u128::from(row.0)]).unwrap();
    assert_eq!(node.region_churn(table, &A), Ok(vec![None]));
}

#[test]
fn rows_a_model_reads_only_once_labeled_count_toward_refits_from_their_label() {
    let (keys, features, Targets::Classes(labels)) = rows(28) else {
        unreachable!()
    };
    let batch = |at: usize| at..at + 2;
    let labeled = |at: usize| Targets::Classes(labels[batch(at)].to_vec());
    // A whole fit, and one on a sample.
    for fit_rows in [DEFAULT_FIT_ROWS, 8] {
        let mut node = node(64);
        let spec = TableSpec {
            refit: RefitPolicy::new(Some(0.25), None).unwrap(),
            fit_rows: NonZeroUsize::new(fit_rows).unwrap(),
            ..spec(Strategy::Exact)
        };
        // `late` gets each batch once labeled; `held` holds it unlabeled a
        // batch before, through every fit, and another row it deletes unlabeled.
        let late = node.create_table(spec.clone()).unwrap();
        let held = node.create_table(spec).unwrap();
        for table in [late, held] {
            node.insert(
                table,
                &keys[..16],
                &features[..32],
                Targets::Classes(labels[..16].to_vec()),
            )
            .unwrap();
        }
        let mut epochs = Vec::new();
        for at in (16..28).step_by(2) {
            let unlabeled = Targets::Classes(vec![UNLABELED; 2]);
            node.insert(
                held,
                &keys[batch(at)],
                &features[2 * at..2 * at + 4],
                unlabeled,
            )
            .unwrap();
            node.insert(held, &[99], &A, Targets::Classes(vec![UNLABELED]))
                .unwrap();
            node.delete(held, &[99]).unwrap();
            if at > 16 {
                let before = at - 2;
                node.insert(
                    late,
                    &keys[batch(before)],
                    &features[2 * before..2 * at],
                    labeled(before),
                )
                .unwrap();
                node.label(held, &keys[batch(before)], labeled(before))
                    .unwrap();
            }
            node.submit(late, &A).unwrap();
            node.submit(held, &A).unwrap();
            let step = run(&mut node);
            let fit = |table| {
                let fit = step.fits().iter().find(|fit| fit.table == table)?;
                Some((fit.epoch, fit.rows.len()))
            };
            epochs.push((fit(late), fit(held)));
        }

        assert!(epochs.iter().all(|(late, held)| late == held), "{epochs:?}");
        assert_eq!(epochs.iter().filter(|(late, _)| late.is_some()).count(), 3);
    }
}

#[test]
fn a_model_reading_unlabeled_rows_counts_their_insert_and_label_once_each() {
    let mut node = node(64);
    let spec = TableSpec {
        unlabeled_context: true,
        refit: RefitPolicy::new(Some(0.5), None).unwrap(),
        ..spec(Strategy::Exact)
    };
    let table = node.create_table(spec).unwrap();
    let (keys, features, Targets::Classes(mut labels)) = rows(20) else {
        unreachable!()
    };
    labels[8..].fill(UNLABELED);
    node.insert(
        table,
        &keys[..16],
        &features[..32],
        Targets::Classes(labels[..16].to_vec()),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    run(&mut node);

    // The fit read all 16 rows, so 4 labels fall short of half of them and
    // 4 more rows inserted unlabeled reach it.
    node.label(table, &keys[8..12], Targets::Classes(vec![0; 4]))
        .unwrap();
    node.submit(table, &A).unwrap();
    let labeled = run(&mut node);
    node.insert(
        table,
        &keys[16..],
        &features[32..],
        Targets::Classes(labels[16..].to_vec()),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    let inserted = run(&mut node);

    assert!(labeled.fits().is_empty());
    assert_eq!(inserted.fits().len(), 1);
    assert_eq!(inserted.fits()[0].rows.len(), 20);
}

#[test]
fn keyed_contexts_read_their_rows_cut_to_the_trained_length_under_clamp() {
    let mut node = node(64);
    let table = node
        .create_table(TableSpec {
            past_cap: PastCap::Clamp,
            context_rows: NonZeroU64::new(4),
            ..spec(Strategy::Exact)
        })
        .unwrap();
    let (keys, features, targets) = rows(8);
    node.insert(table, &keys, &features, targets).unwrap();

    node.submit_keys(table, &[A, B].concat(), &[vec![1, 3], (0..8).collect()])
        .unwrap();
    let step = run(&mut node);

    let read = |query: usize| step.unit(step.queries()[0].units[query]).rows.clone();
    assert_eq!(read(0), [RowId(1), RowId(3)]);
    assert_eq!(read(1).len(), 4);
    assert_eq!(
        node.submit_keys(table, &A, &[vec![99]]),
        Err(NodeError::UnknownKey(RowKey(99)))
    );
}

/// The requests a step answers.
fn answered(step: &Step) -> Vec<RequestId> {
    step.queries().iter().map(|batch| batch.request).collect()
}

#[test]
fn given_and_other_requests_take_turns_and_givens_share_a_step() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    let first = node.submit_keys(table, &A, &[vec![0]]).unwrap();
    let plain = node.submit(table, &A).unwrap();
    let mut steps = Vec::new();
    // A given request arrives before each step: the plain one still waits
    // only for the step in front of it.
    for _ in 0..3 {
        node.submit_keys(table, &A, &[vec![1]]).unwrap();
        steps.push(answered(&run(&mut node)));
    }

    // Given requests not yet planned join one batch, and the two kinds
    // alternate.
    assert_eq!(steps[0].len(), 2);
    assert_eq!(steps[0][0], first);
    assert_eq!(steps[1], [plain]);
    assert_eq!(steps[2].len(), 2);
}

#[test]
fn a_given_request_whose_rows_went_is_refused_alone() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    let plain = node.submit(table, &A).unwrap();
    let gone = node.submit_keys(table, &A, &[vec![6, 7]]).unwrap();
    let thinned = node.submit_keys(table, &A, &[vec![4, 5]]).unwrap();
    node.delete(table, &[5, 6, 7]).unwrap();

    let steps = [run(&mut node), run(&mut node)];

    assert_eq!(answered(&steps[0]), [plain]);
    assert_eq!(answered(&steps[1]), [thinned]);
    assert_eq!(
        steps[1].unit(steps[1].queries()[0].units[0]).rows,
        [RowId(4)]
    );
    assert_eq!(
        node.take_refused()
            .into_iter()
            .map(|(request, _)| request)
            .collect::<Vec<_>>(),
        [gone]
    );
}

#[test]
fn removing_a_table_refuses_every_request_it_queued() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    let queued = [
        node.submit(table, &A).unwrap(),
        node.submit_keys(table, &A, &[vec![0]]).unwrap(),
        node.submit(table, &A).unwrap(),
    ];

    node.remove_table(table).unwrap();

    let refused: HashSet<RequestId> = node
        .take_refused()
        .into_iter()
        .map(|(request, _)| request)
        .collect();
    assert_eq!(refused, HashSet::from(queued));
}

#[test]
fn a_cancelled_given_request_takes_its_contexts_from_the_batch() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    let requests: Vec<RequestId> = (0..3)
        .map(|row| node.submit_keys(table, &A, &[vec![row]]).unwrap())
        .collect();

    node.cancel(requests[0]);
    let step = run(&mut node);

    let read: Vec<Vec<RowId>> = step
        .queries()
        .iter()
        .map(|batch| step.unit(batch.units[0]).rows.clone())
        .collect();
    assert_eq!(answered(&step), requests[1..]);
    assert_eq!(read, [[RowId(1)], [RowId(2)]]);
}

#[test]
fn given_rows_without_labels_are_left_out_and_a_context_of_them_refused() {
    let mut node = node(64);
    let table = partly_labeled(&mut node, spec(Strategy::Exact));
    // Keys 32 .. 40 have no labels yet.
    let mixed = node.submit_keys(table, &A, &[vec![1, 33]]).unwrap();
    let unlabeled = node.submit_keys(table, &A, &[vec![34, 35]]).unwrap();

    let step = run(&mut node);

    assert_eq!(answered(&step), [mixed]);
    assert_eq!(step.unit(step.queries()[0].units[0]).rows, [RowId(1)]);
    assert_eq!(
        node.take_refused()
            .into_iter()
            .map(|(request, _)| request)
            .collect::<Vec<_>>(),
        [unlabeled]
    );
}

#[test]
fn a_context_needs_a_labeled_row_even_for_a_model_reading_unlabeled_ones() {
    let mut node = node(64);
    let spec = TableSpec {
        unlabeled_context: true,
        ..spec(Strategy::neighbors(4))
    };
    let table = node.create_table(spec.clone()).unwrap();
    // Rows near B, the odd keys, have no labels yet.
    let (keys, features, Targets::Classes(mut labels)) = rows(16) else {
        unreachable!()
    };
    for label in labels.iter_mut().skip(1).step_by(2) {
        *label = UNLABELED;
    }
    node.insert(table, &keys, &features, Targets::Classes(labels))
        .unwrap();
    let unlabeled = node.create_table(spec).unwrap();
    node.insert(
        unlabeled,
        &keys,
        &features,
        Targets::Classes(vec![UNLABELED; 16]),
    )
    .unwrap();

    let near_b = node.submit(table, &B).unwrap();
    let near_a = node.submit(table, &A).unwrap();
    let none = node.submit(unlabeled, &A).unwrap();
    let step = run(&mut node);

    let answered: Vec<RequestId> = step.queries().iter().map(|batch| batch.request).collect();
    assert_eq!(answered, [near_a]);
    let refused: Vec<_> = node.take_refused();
    assert!(refused.contains(&(near_b, NodeError::EmptyContext(table))));
    assert!(refused.contains(&(none, NodeError::EmptyTable(unlabeled))));
    assert_eq!(
        node.contexts(table, &B, None),
        Err(NodeError::EmptyContext(table))
    );
}

/// Each page a step computes: the rows it copies and the rows it embeds.
fn copied_and_embedded(step: &Step) -> Vec<(usize, usize)> {
    embedded(step)
        .iter()
        .map(|page| {
            let copied = page
                .copies
                .iter()
                .map(|(_, positions)| positions.len())
                .sum();
            (copied, page.rows.len() - copied)
        })
        .collect()
}

#[test]
fn a_page_written_once_copies_its_other_rows_from_its_earlier_entry() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    node.submit(table, &A).unwrap();
    let first = run(&mut node);
    let earlier = first.pages()[0].key;

    // One row changes: its page embeds it and copies the seven others from
    // the entry it had, at the same positions.
    node.update(table, &[3], &[0.5, 0.5], Targets::Classes(vec![2]))
        .unwrap();
    node.submit(table, &A).unwrap();
    let step = node.schedule().unwrap().unwrap();
    assert_eq!(copied_and_embedded(&step), [(7, 1)]);
    let page = embedded(&step)[0];
    let (source, positions) = &page.copies[0];
    assert_eq!(*source, earlier);
    let expected: Vec<(usize, usize)> = (0..8).filter(|&at| at != 3).map(|at| (at, at)).collect();
    assert_eq!(*positions, expected);
    let (features, targets) = page.embed.as_ref().unwrap();
    assert_eq!(
        (features.as_slice(), targets.len()),
        ([0.5, 0.5].as_slice(), 1)
    );
    // The earlier entry goes once the new one is kept.
    let dropped: HashSet<_> = node
        .complete(&step)
        .into_iter()
        .map(|drop| drop.key)
        .collect();
    assert!(dropped.contains(&earlier));

    // A row inserted splits the page; both halves copy from the entry.
    node.insert(table, &[8], &[0.25, 0.25], Targets::Classes(vec![0]))
        .unwrap();
    node.submit(table, &A).unwrap();
    let step = run(&mut node);
    let (copied, embedded): (Vec<usize>, Vec<usize>) =
        copied_and_embedded(&step).into_iter().unzip();
    assert_eq!(copied.iter().sum::<usize>(), 8);
    assert_eq!(embedded.iter().sum::<usize>(), 1);

    // A new epoch embeds rows differently: nothing copies across it.
    node.refit(table).unwrap();
    node.submit(table, &A).unwrap();
    let step = run(&mut node);
    assert!(step.pages().iter().all(|page| page.copies.is_empty()));
}

#[test]
fn a_page_with_every_row_written_drops_at_the_write() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    node.submit(table, &A).unwrap();
    let first = run(&mut node);
    let earlier = first.pages()[0].key;

    // Three writes touch every row: the entry is a source after the first
    // two and has nothing left to copy after the third, so it goes then.
    let (keys, features, targets) = rows(8);
    for (first, last) in [(0, 3), (3, 6), (6, 8)] {
        let labels = match &targets {
            Targets::Classes(classes) => Targets::Classes(classes[first..last].to_vec()),
            other => other.clone(),
        };
        node.update(
            table,
            &keys[first..last],
            &features[first * 2..last * 2],
            labels,
        )
        .unwrap();
    }
    let dropped: HashSet<_> = node.take_drops().into_iter().map(|drop| drop.key).collect();
    assert!(dropped.contains(&earlier));
    node.submit(table, &A).unwrap();
    let step = node.schedule().unwrap().unwrap();
    assert_eq!(copied_and_embedded(&step), [(0, 8)]);
}

#[test]
fn a_source_two_running_steps_copy_from_outlives_the_first_completing() {
    // Sixteen rows at times 0..16 fill two pages, and the host holds the two
    // and no more: a page's next version is never kept, so a second step
    // need not wait for the first to embed it. Windows before 8 and before
    // 12 read both pages through different contexts, from one state fit on
    // the rows before 8.
    let mut node = node_with_host(64, 2);
    let refit = RefitPolicy::default()
        .with_cutoffs(CutoffRefit::Never)
        .unwrap();
    let table = node
        .create_table(TableSpec {
            refit,
            ..spec(Strategy::Exact)
        })
        .unwrap();
    let (keys, features, targets) = rows(16);
    let times: Vec<Option<i64>> = (0..16).map(Some).collect();
    node.insert_at(table, &keys, &features, targets, Some(&times))
        .unwrap();
    node.submit_before(table, &A, Some(&[8])).unwrap();
    let first = run(&mut node);
    let earlier = first
        .pages()
        .iter()
        .find(|page| page.rows.contains(&RowId(0)))
        .expect("a page holds row 0")
        .key;

    // Row 0 changes, so both windows encode again, and both steps copy its
    // page's seven other rows from its entry.
    node.update(table, &[0], &[0.5, 0.0], Targets::Classes(vec![2]))
        .unwrap();
    node.submit_before(table, &A, Some(&[8])).unwrap();
    let one = node.schedule().unwrap().unwrap();
    node.submit_before(table, &A, Some(&[12])).unwrap();
    let two = node.schedule().unwrap().unwrap();
    assert!(one.fits().is_empty() && two.fits().is_empty());
    assert_ne!(one.units()[0].key, two.units()[0].key);
    for step in [&one, &two] {
        let pages = embedded(step);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].placed.action(), Action::Transient);
        let (source, positions) = &pages[0].copies[0];
        assert_eq!((*source, positions.len()), (earlier, 7));
    }

    // The entry's epoch ends while both read it: each step pinned it, so it
    // goes with the last one, not the first.
    let keys = |drops: Vec<gf_core::cache::Movement>| -> HashSet<_> {
        drops.into_iter().map(|drop| drop.key).collect()
    };
    assert!(!keys(node.advance_epoch(table).unwrap()).contains(&earlier));
    assert!(!keys(node.complete(&one)).contains(&earlier));
    assert!(keys(node.complete(&two)).contains(&earlier));
}

#[test]
fn a_source_two_pages_of_one_step_copy_from_is_dropped_once() {
    let mut node = node(64);
    let table = table(&mut node, Strategy::Exact, 8);
    node.submit(table, &A).unwrap();
    let earlier = run(&mut node).pages()[0].key;

    // A ninth row splits the page: each half copies its rows from the one
    // entry, between them all eight.
    node.insert(table, &[8], &[0.25, 0.25], Targets::Classes(vec![0]))
        .unwrap();
    node.submit(table, &A).unwrap();
    let step = node.schedule().unwrap().unwrap();
    let pages = embedded(&step);
    assert_eq!(pages.len(), 2);
    let mut from = Vec::new();
    for page in &pages {
        assert_eq!(page.copies.len(), 1);
        assert_eq!(page.copies[0].0, earlier);
        from.extend(page.copies[0].1.iter().map(|&(at, _)| at));
    }
    from.sort_unstable();
    assert_eq!(from, (0..8).collect::<Vec<_>>());
    // Both halves pinned it, so admitting them moved it nowhere, and it is
    // dropped once, when the step completes.
    assert!(
        step.movements()
            .iter()
            .all(|movement| movement.key != earlier)
    );
    assert!(purged(&mut node).is_empty());
    let drops: Vec<_> = node
        .complete(&step)
        .into_iter()
        .filter(|drop| drop.key == earlier)
        .collect();
    assert_eq!(drops.len(), 1);
    assert!(drops[0].to.is_none());
}

#[test]
fn admitting_a_page_under_host_pressure_never_evicts_a_pinned_source() {
    // Three host slots: the table's two pages, and room to keep one next
    // version of them. No device, so every read encodes the context and
    // reads the pages.
    let mut node = node_with_host(0, 3);
    let table = table(&mut node, Strategy::Exact, 16);
    node.submit(table, &A).unwrap();
    let first = run(&mut node);
    let sources: HashSet<_> = first.pages().iter().map(|page| page.key).collect();
    assert_eq!(sources.len(), 2);

    // A row of each page changes, so both next versions copy seven rows. The
    // second finds no room: its source and the first's are pinned, so neither
    // is evicted for it.
    node.update(
        table,
        &[0, 1],
        &[0.5, 0.0, 100.5, 100.0],
        Targets::Classes(vec![2, 2]),
    )
    .unwrap();
    node.submit(table, &A).unwrap();
    let step = node.schedule().unwrap().unwrap();
    assert_eq!(copied_and_embedded(&step), [(7, 1), (7, 1)]);
    let placed = |action| {
        embedded(&step)
            .into_iter()
            .filter(|page| page.placed.action() == action)
            .map(|page| page.copies[0].0)
            .collect::<Vec<_>>()
    };
    let (kept, transient) = (placed(Action::Encode), placed(Action::Transient));
    assert_eq!((kept.len(), transient.len()), (1, 1));
    assert!(sources.contains(&kept[0]) && sources.contains(&transient[0]));
    assert!(
        step.movements()
            .iter()
            .all(|movement| !sources.contains(&movement.key))
    );

    // The kept page's source has served; the transient page's stays, and the
    // page's next version copies from it again.
    let dropped: HashSet<_> = node
        .complete(&step)
        .into_iter()
        .map(|drop| drop.key)
        .collect();
    assert!(dropped.contains(&kept[0]));
    assert!(!dropped.contains(&transient[0]));
    node.submit(table, &A).unwrap();
    let again = node.schedule().unwrap().unwrap();
    assert_eq!(copied_and_embedded(&again), [(7, 1)]);
    assert_eq!(embedded(&again)[0].copies[0].0, transient[0]);
}

#[test]
fn only_new_single_use_contexts_waiting_is_told_apart_from_other_work() {
    let mut node = node(64);
    let durable = table(&mut node, Strategy::Exact, 32);
    let fresh = table(&mut node, Strategy::Exact, 32);
    node.set_single_use(fresh).unwrap();
    assert!(!node.only_new_contexts_wait());

    node.submit(fresh, &A).unwrap();
    assert!(node.only_new_contexts_wait());
    node.submit(durable, &A).unwrap();
    assert!(!node.only_new_contexts_wait());

    run(&mut node);
    assert!(!node.only_new_contexts_wait());
    node.submit(fresh, &B).unwrap();
    assert!(!node.only_new_contexts_wait());
}
