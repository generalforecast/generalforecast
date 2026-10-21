// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deciding where each table's copies should be shipped, released and read,
//! and which may take over its writes.

use gf_router::replication::{
    Repair, TableCopy, creation, diverged, promotion, read_order, repairs,
};
use gf_server::wal::{CreationId, Term, Terms};

/// A copy of `length` entries that has seen terms begun at `starts`, in
/// order, each promoted to `node-{term}`.
fn copy(length: u64, starts: &[u64]) -> Option<TableCopy> {
    promoted(length, starts, |term| format!("node-{term}"))
}

/// A copy of `length` entries whose terms begun at `starts` named `primary`.
fn promoted(length: u64, starts: &[u64], primary: impl Fn(u64) -> String) -> Option<TableCopy> {
    let mut terms = Terms::default();
    for (term, &start) in (1..).zip(starts) {
        terms.begin(Term {
            term,
            primary: primary(term),
            start,
        });
    }
    Some(TableCopy {
        length,
        terms,
        created: CreationId::parse("same"),
    })
}

/// Which copies diverged, with the table's creation as its copies vote.
fn diverged_of(copies: &[Option<TableCopy>]) -> Vec<bool> {
    diverged(copies, creation(copies, &[]).as_ref())
}

#[test]
fn a_copy_of_an_earlier_creation_of_the_name_diverged() {
    // The stale copy's node had the later clock when it created its table.
    let mut stale = copy(9, &[]).unwrap();
    stale.created = CreationId::parse("0000019b00000000");
    let mut current = copy(5, &[]).unwrap();
    current.created = CreationId::parse("0000019a00000000");

    // The creation the replicas hold is the table, however long the stale
    // copy's log and whatever the clocks said.
    let copies = [Some(current.clone()), Some(stale.clone()), None];
    let chosen = creation(&copies, &[0, 2]);
    assert_eq!(chosen, current.created);
    assert_eq!(diverged(&copies, chosen.as_ref()), [false, true, false]);
    // Among copies off the replicas alike, the latest creation is the table.
    let copies = [None, Some(stale.clone()), Some(current)];
    let chosen = creation(&copies, &[0]);
    assert_eq!(chosen, stale.created);
    assert_eq!(diverged(&copies, chosen.as_ref()), [false, false, true]);
}

#[test]
fn behind_replicas_are_shipped_from_the_longest_log() {
    // Node 2 now ranks first and has no copy; node 1 held the table before.
    let lengths = [Some(3), Some(5), None];

    let planned = repairs(&[2, 0], &lengths, &[true; 3]);

    assert_eq!(
        planned,
        [
            Repair::Ship {
                source: 1,
                target: 2,
                from: 0
            },
            Repair::Ship {
                source: 1,
                target: 0,
                from: 3
            },
        ]
    );
}

#[test]
fn extra_copies_are_released_once_every_replica_is_current() {
    let current = repairs(&[0, 1], &[Some(5), Some(5), Some(5)], &[true; 3]);
    let replica_down = repairs(&[0, 1], &[Some(5), None, Some(5)], &[true, false, true]);
    let nothing_to_do = repairs(&[0, 1], &[Some(5), Some(5), None], &[true; 3]);

    assert_eq!(current, [Repair::Release { node: 2 }]);
    assert_eq!(replica_down, []);
    assert_eq!(nothing_to_do, []);
}

#[test]
fn reads_go_to_current_copies_in_rank_order() {
    let all_up = [true; 3];

    assert_eq!(
        read_order(&[1, 0], &[Some(5), Some(5), None], &all_up),
        [1, 0]
    );
    assert_eq!(
        read_order(&[1, 0], &[Some(5), Some(5), None], &[true, false, true]),
        [0]
    );
    // Moving: the new replicas are behind, so the old holder serves.
    assert_eq!(
        read_order(&[2, 0], &[Some(3), Some(5), Some(1)], &all_up),
        [1]
    );
    // Nothing known yet: the replicas that are up, primary first.
    assert_eq!(
        read_order(&[1, 0], &[None, None, None], &[true, true, false]),
        [1, 0]
    );
}

#[test]
fn copies_past_the_start_of_a_later_term_diverged() {
    // Term 1 began at entry 10 and term 2 at entry 14.
    let copies = [
        copy(10, &[]),
        copy(12, &[]),
        copy(14, &[10]),
        copy(16, &[10]),
        copy(20, &[10, 14]),
        None,
    ];

    assert_eq!(
        diverged_of(&copies),
        [false, true, false, true, false, false]
    );
}

#[test]
fn of_two_promotions_at_one_term_the_shorter_history_diverged() {
    let other = promoted(12, &[10], |_| "node-2".to_string());

    let copies = [copy(14, &[10]), other];

    assert_eq!(diverged_of(&copies), [false, true]);
}

#[test]
fn only_a_replica_with_every_acknowledged_write_is_promoted() {
    let copies = [copy(12, &[]), copy(10, &[]), copy(12, &[])];
    let up = [false, true, true];

    // Node 1 ranks first after the lost primary but lacks two writes.
    assert_eq!(promotion(&[0, 1, 2], &copies, &up, 0, 12), Some(2));
    assert_eq!(promotion(&[0, 1], &copies, &up, 0, 12), None);
}

#[test]
fn no_replica_is_promoted_while_another_may_hold_writes_unseen() {
    let copies = [copy(12, &[]), copy(10, &[]), copy(10, &[])];

    // With node 2 down as well, it may hold acknowledged writes node 1 lacks.
    assert_eq!(
        promotion(&[0, 1, 2], &copies, &[false, true, false], 0, 10),
        None
    );
    assert_eq!(
        promotion(&[0, 1, 2], &copies, &[false, true, true], 0, 10),
        Some(1)
    );
}

#[test]
fn the_longest_replica_is_promoted_whatever_the_router_saw() {
    // A router that restarted has seen nothing, and node 2 alone took the last write.
    let copies = [copy(14, &[]), copy(10, &[]), copy(12, &[])];

    assert_eq!(
        promotion(&[0, 1, 2], &copies, &[false, true, true], 0, 0),
        Some(2)
    );
}
