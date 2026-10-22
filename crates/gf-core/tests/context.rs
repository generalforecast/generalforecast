// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Contexts as sets of rows, and their fingerprints.

use gf_core::context::{Context, Fingerprint};
use gf_core::table::{Change, RowId, RowKey, Table, Version};

fn context(rows: &[u32]) -> Context {
    rows.iter().map(|&row| RowId(row)).collect()
}

fn rows(context: &Context) -> Vec<u32> {
    context.iter().map(|row| row.0).collect()
}

fn table(rows: u128) -> Table {
    let mut table = Table::new();
    for key in 0..rows {
        table.apply(Change::Insert(RowKey(key))).unwrap();
    }
    table
}

#[test]
fn contexts_support_set_algebra() {
    let a = context(&[1, 2, 3, 8]);
    let b = context(&[3, 4, 8, 9]);

    assert_eq!(rows(&(&a | &b)), [1, 2, 3, 4, 8, 9]);
    assert_eq!(rows(&(&a & &b)), [3, 8]);
    assert_eq!(rows(&(&a - &b)), [1, 2]);
    assert_eq!((&a & &b).len(), 2);
    assert!(a.contains(RowId(8)) && !a.contains(RowId(4)));
}

#[test]
fn fingerprints_ignore_the_order_rows_were_added() {
    let snapshot = table(6).snapshot();

    let forward = snapshot.fingerprint(&context(&[0, 2, 5]));
    let backward = snapshot.fingerprint(&context(&[5, 2, 0]));

    assert_eq!(forward, backward);
    assert_ne!(forward, snapshot.fingerprint(&context(&[0, 2, 4])));
    assert_ne!(forward, Fingerprint::EMPTY);
}

#[test]
fn fingerprints_change_when_a_row_changes() {
    let mut table = table(4);
    let rows = context(&[0, 1, 3]);
    let before = table.snapshot().fingerprint(&rows);

    table.apply(Change::Update(RowKey(1))).unwrap();

    assert_ne!(table.snapshot().fingerprint(&rows), before);
}

#[test]
fn fingerprints_update_incrementally() {
    let mut table = table(4);
    let rows = context(&[0, 1, 3]);
    let before = table.snapshot().fingerprint(&rows);

    table.apply(Change::Update(RowKey(1))).unwrap();

    // Swapping one row's old version for its new one gives the full recompute.
    let updated =
        before - Fingerprint::of(RowId(1), Version(0)) + Fingerprint::of(RowId(1), Version(1));
    assert_eq!(updated, table.snapshot().fingerprint(&rows));
}

#[test]
fn fingerprints_of_disjoint_contexts_add() {
    let snapshot = table(8).snapshot();
    let left = context(&[0, 1, 2]);
    let right = context(&[5, 7]);

    let whole = snapshot.fingerprint(&(&left | &right));

    assert_eq!(
        whole,
        snapshot.fingerprint(&left) + snapshot.fingerprint(&right)
    );
}

#[test]
fn fingerprints_count_only_rows_live_in_the_snapshot() {
    let mut table = table(3);
    table.apply(Change::Delete(RowKey(2))).unwrap();
    let snapshot = table.snapshot();

    let with_deleted = snapshot.fingerprint(&context(&[0, 2]));

    assert_eq!(with_deleted, snapshot.fingerprint(&context(&[0])));
    assert_eq!(rows(&snapshot.rows()), [0, 1]);
}

#[test]
fn incremental_fingerprints_match_fresh_sums_under_random_changes() {
    let mut state = 9u64;
    let mut random = move |limit: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (state >> 33) % limit
    };
    let mut table = table(8);
    let mut next = 8;
    for _ in 0..300 {
        let live: Vec<u128> = (0..next)
            .filter(|&key| table.row(RowKey(key)).is_some())
            .collect();
        let change = match random(3) {
            0 => {
                next += 1;
                Change::Insert(RowKey(next - 1))
            }
            1 if !live.is_empty() => {
                Change::Update(RowKey(live[random(live.len() as u64) as usize]))
            }
            _ if !live.is_empty() => {
                Change::Delete(RowKey(live[random(live.len() as u64) as usize]))
            }
            _ => continue,
        };
        table.apply(change).unwrap();
        let snapshot = table.snapshot();
        let fresh: Fingerprint = snapshot
            .rows()
            .iter()
            .map(|row| Fingerprint::of(row, snapshot.version(row).unwrap()))
            .sum();
        assert_eq!(snapshot.fingerprint(&snapshot.rows()), fresh);
    }
}

#[test]
fn salted_fingerprints_differ_by_salt_and_keep_their_rows_apart() {
    let one = Fingerprint::of(RowId(0), Version(0));
    let two = one + Fingerprint::of(RowId(1), Version(0));

    assert_ne!(one.salted(1), one);
    assert_ne!(one.salted(1), one.salted(2));
    assert_eq!(one.salted(1), one.salted(1));
    assert_ne!(one.salted(1), two.salted(1));
}
