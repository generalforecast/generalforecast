//! Rows, versions, the change log and snapshots.

use gf_core::context::Fingerprint;
use gf_core::table::{Change, LogPosition, RowId, RowKey, Table, TableError, Version};

fn table_with(keys: &[u128]) -> Table {
    let mut table = Table::new();
    for &key in keys {
        table.apply(Change::Insert(RowKey(key))).unwrap();
    }
    table
}

#[test]
fn inserts_get_dense_row_ids_in_log_order() {
    let mut table = Table::new();

    let first = table.apply(Change::Insert(RowKey(40))).unwrap();
    let second = table.apply(Change::Insert(RowKey(7))).unwrap();

    assert_eq!((first, second), (RowId(0), RowId(1)));
    assert_eq!(table.row(RowKey(7)), Some(RowId(1)));
    assert_eq!(table.position(), LogPosition(2));
    assert_eq!(table.len(), 2);
}

#[test]
fn changes_to_unknown_or_duplicate_keys_fail_without_advancing_the_log() {
    let mut table = table_with(&[1]);

    let duplicate = table.apply(Change::Insert(RowKey(1)));
    let update = table.apply(Change::Update(RowKey(2)));
    let delete = table.apply(Change::Delete(RowKey(2)));

    assert_eq!(duplicate, Err(TableError::DuplicateKey(RowKey(1))));
    assert_eq!(update, Err(TableError::UnknownKey(RowKey(2))));
    assert_eq!(delete, Err(TableError::UnknownKey(RowKey(2))));
    assert_eq!(table.position(), LogPosition(1));
}

#[test]
fn updates_bump_the_row_version() {
    let mut table = table_with(&[1, 2]);

    table.apply(Change::Update(RowKey(2))).unwrap();
    table.apply(Change::Update(RowKey(2))).unwrap();

    let snapshot = table.snapshot();
    assert_eq!(snapshot.version(RowId(0)), Some(Version(0)));
    assert_eq!(snapshot.version(RowId(1)), Some(Version(2)));
}

#[test]
fn row_ids_are_never_reused() {
    let mut table = table_with(&[1, 2]);

    let deleted = table.apply(Change::Delete(RowKey(1))).unwrap();
    let again = table.apply(Change::Insert(RowKey(1))).unwrap();

    // A cache entry keyed by the old row id can never match the new row.
    assert_eq!(deleted, RowId(0));
    assert_eq!(again, RowId(2));
    assert_eq!(table.snapshot().version(deleted), None);
    assert_eq!(table.len(), 2);
}

#[test]
fn snapshots_do_not_see_later_changes() {
    let mut table = table_with(&[1, 2, 3]);
    let before = table.snapshot();

    table.apply(Change::Update(RowKey(1))).unwrap();
    table.apply(Change::Delete(RowKey(2))).unwrap();
    table.apply(Change::Insert(RowKey(4))).unwrap();
    let after = table.snapshot();

    assert_eq!(before.position(), LogPosition(3));
    assert_eq!(
        before.rows().iter().collect::<Vec<_>>(),
        [RowId(0), RowId(1), RowId(2)]
    );
    assert_eq!(before.version(RowId(0)), Some(Version(0)));
    assert_eq!(after.position(), LogPosition(6));
    assert_eq!(
        after.rows().iter().collect::<Vec<_>>(),
        [RowId(0), RowId(2), RowId(3)]
    );
    assert_eq!(after.version(RowId(0)), Some(Version(1)));
}

#[test]
fn the_whole_tables_fingerprint_follows_its_changes() {
    let mut table = table_with(&[1, 2, 3]);
    table.apply(Change::Update(RowKey(2))).unwrap();
    table.apply(Change::Delete(RowKey(1))).unwrap();
    table.apply(Change::Insert(RowKey(4))).unwrap();

    let snapshot = table.snapshot();
    let expected: Fingerprint = snapshot
        .rows()
        .iter()
        .map(|row| Fingerprint::of(row, snapshot.version(row).unwrap()))
        .sum();

    assert_eq!(snapshot.fingerprint(&snapshot.rows()), expected);
}

#[test]
fn keys_print_as_integers_or_as_digests() {
    assert_eq!(RowKey(7).to_string(), "7");
    assert_eq!(
        RowKey(u128::from(u64::MAX)).to_string(),
        u64::MAX.to_string()
    );
    assert_eq!(RowKey((-3_i128) as u128).to_string(), "-3");
    assert_eq!(
        RowKey(1 << 100).to_string(),
        "0x00000010000000000000000000000000"
    );
}

#[test]
fn keys_find_their_rows_through_many_inserts_and_deletes() {
    let mut table = Table::new();
    let mut expected = std::collections::HashMap::new();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..20_000 {
        // Few distinct keys, so deletes and inserts of the same keys interleave.
        let key = RowKey(u128::from(next() % 3_000) << 64 | 5);
        match expected.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                if next() % 2 == 0 {
                    table.apply(Change::Delete(key)).unwrap();
                    entry.remove();
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(table.apply(Change::Insert(key)).unwrap());
            }
        }
    }

    for probe in 0..3_000_u128 {
        let key = RowKey(probe << 64 | 5);
        assert_eq!(table.row(key), expected.get(&key).copied());
    }
    let mut keys: Vec<(RowKey, RowId)> = table.keys().collect();
    keys.sort();
    let mut wanted: Vec<(RowKey, RowId)> = expected.into_iter().collect();
    wanted.sort();
    assert_eq!(keys, wanted);
}
