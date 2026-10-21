// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rows by id in blocks, in memory and spilled to disk.

use std::path::PathBuf;
use std::sync::Arc;

use gf_core::rows::{BLOCK_BYTES, Row, RowMemory, RowStore, RowsError, Sealer};
use gf_core::table::RowId;

/// A sealer that binds a block's place and version without encrypting, so
/// tests can see that swapped or stale blocks are refused.
#[derive(Debug)]
struct Bound;

impl Sealer for Bound {
    fn seal(&self, block: u32, version: u32, plain: &[u8]) -> Vec<u8> {
        let mut sealed = plain.to_vec();
        sealed.extend_from_slice(&block.to_le_bytes());
        sealed.extend_from_slice(&version.to_le_bytes());
        sealed
    }

    fn open(&self, block: u32, version: u32, sealed: &[u8]) -> Option<Vec<u8>> {
        let (plain, tail) = sealed.split_at(sealed.len().checked_sub(8)?);
        let expected: Vec<u8> = [block.to_le_bytes(), version.to_le_bytes()].concat();
        (tail == expected).then(|| plain.to_vec())
    }
}

fn path(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("gf-rows-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    directory.join("table.rows")
}

/// Rows 0..count of two features, their targets and keys derived from their ids.
fn rows(count: u32) -> (Vec<f32>, Vec<u32>, Vec<u128>) {
    let features = (0..count)
        .flat_map(|row| [row as f32, -(row as f32)])
        .collect();
    let targets = (0..count).map(|row| row % 3).collect();
    let keys = (0..count).map(|row| u128::from(row) << 64 | 7).collect();
    (features, targets, keys)
}

fn ids(range: std::ops::Range<u32>) -> Vec<RowId> {
    range.map(RowId).collect()
}

#[test]
fn rows_come_back_as_they_went_in() {
    let memory = RowMemory::unbounded();
    let mut store = RowStore::new(&memory, 2);
    let (features, targets, keys) = rows(500);
    store.push(&features, &targets, &keys).unwrap();

    let (read, read_targets) = store.gather(&[RowId(3), RowId(499), RowId(0)]).unwrap();

    assert_eq!(read, [3.0, -3.0, 499.0, -499.0, 0.0, -0.0]);
    assert_eq!(read_targets, [0, 1, 0]);
    assert_eq!(
        store.rows(&[RowId(4)]).unwrap(),
        [Row {
            features: vec![4.0, -4.0],
            target: 1,
            key: 4 << 64 | 7
        }]
    );
    assert_eq!(
        store.gather(&[RowId(500)]),
        Err(RowsError::Unknown(RowId(500)))
    );
}

#[test]
fn a_store_past_its_budget_spills_to_disk_and_reads_back() {
    let memory = RowMemory::bounded(4 * BLOCK_BYTES as u64);
    let file = path("spill");
    let mut store = RowStore::on_disk(&memory, 2, &file, Arc::new(Bound)).unwrap();
    let (features, targets, keys) = rows(5_000);
    store.push(&features, &targets, &keys).unwrap();
    assert!(memory.bytes() <= 4 * BLOCK_BYTES as u64);
    assert!(std::fs::metadata(&file).unwrap().len() > 0);

    let every = ids(0..5_000);
    let (read, read_targets) = store.gather(&every).unwrap();

    assert_eq!(read, features);
    assert_eq!(read_targets, targets);
    assert!(memory.bytes() <= 4 * BLOCK_BYTES as u64);
    let keys_back: Vec<u128> = store
        .rows(&every)
        .unwrap()
        .iter()
        .map(|row| row.key)
        .collect();
    assert_eq!(keys_back, keys);
}

#[test]
fn changed_rows_are_written_again_and_old_copies_counted_as_garbage() {
    let memory = RowMemory::bounded(BLOCK_BYTES as u64);
    let mut store = RowStore::on_disk(&memory, 2, &path("garbage"), Arc::new(Bound)).unwrap();
    let (features, targets, keys) = rows(2_000);
    store.push(&features, &targets, &keys).unwrap();
    store.gather(&ids(1_000..2_000)).unwrap();

    store.set(&[RowId(1)], &[10.0, 20.0], &[2], None).unwrap();
    store.gather(&ids(1_000..2_000)).unwrap();

    assert!(store.garbage() > 0);
    let row = &store.rows(&[RowId(1)]).unwrap()[0];
    assert_eq!(
        (row.features.as_slice(), row.target, row.key),
        ([10.0, 20.0].as_slice(), 2, keys[1])
    );
}

#[test]
fn a_snapshot_reads_rows_as_they_were_while_the_store_changes() {
    let memory = RowMemory::bounded(BLOCK_BYTES as u64);
    let mut store = RowStore::on_disk(&memory, 2, &path("snapshot"), Arc::new(Bound)).unwrap();
    let (features, targets, keys) = rows(3_000);
    store.push(&features, &targets, &keys).unwrap();
    let snapshot = store.snapshot();

    store
        .set(
            &[RowId(0), RowId(2_999)],
            &[1.0, 1.0, 2.0, 2.0],
            &[9, 9],
            None,
        )
        .unwrap();
    let (more_features, more_targets, more_keys) = rows(10);
    store
        .push(&more_features, &more_targets, &more_keys)
        .unwrap();

    let before = snapshot.rows(&[RowId(0), RowId(2_999)]).unwrap();
    assert_eq!(before[0].features, [0.0, -0.0]);
    assert_eq!(before[1].features, [2_999.0, -2_999.0]);
    assert_eq!(snapshot.len(), 3_000);
    assert_eq!(
        snapshot.rows(&[RowId(3_000)]),
        Err(RowsError::Unknown(RowId(3_000)))
    );
    assert_eq!(store.features(RowId(0)).unwrap(), [1.0, 1.0]);
}

#[test]
fn a_block_changed_on_disk_is_refused() {
    let memory = RowMemory::bounded(0);
    let file = path("damaged");
    let mut store = RowStore::on_disk(&memory, 2, &file, Arc::new(Bound)).unwrap();
    let (features, targets, keys) = rows(1_000);
    store.push(&features, &targets, &keys).unwrap();
    // Swap the first block's place tag for the second's, as a swapped block would carry.
    let mut bytes = std::fs::read(&file).unwrap();
    let rows_per_block = BLOCK_BYTES / (2 * 4 + 4 + 16);
    let tail = rows_per_block * (2 * 4 + 4 + 16);
    bytes[tail..tail + 4].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&file, bytes).unwrap();

    assert_eq!(
        store.gather(&[RowId(0)]),
        Err(RowsError::Damaged { block: 0 })
    );
}

#[test]
fn a_dropped_store_removes_its_file_and_frees_its_memory() {
    let memory = RowMemory::unbounded();
    let file = path("dropped");
    let mut store = RowStore::on_disk(&memory, 2, &file, Arc::new(Bound)).unwrap();
    let (features, targets, keys) = rows(100);
    store.push(&features, &targets, &keys).unwrap();
    assert!(memory.bytes() > 0);

    drop(store);

    assert_eq!(memory.bytes(), 0);
    assert!(!file.exists());
}

#[test]
fn reserved_rows_take_values_in_any_order() {
    let memory = RowMemory::unbounded();
    let mut store = RowStore::new(&memory, 1);
    store.reserve(10).unwrap();

    store
        .set(&[RowId(7), RowId(2)], &[7.0, 2.0], &[1, 0], Some(&[70, 20]))
        .unwrap();

    let read = store.rows(&[RowId(2), RowId(5), RowId(7)]).unwrap();
    let keys: Vec<u128> = read.iter().map(|row| row.key).collect();
    assert_eq!(keys, [20, 0, 70]);
    assert_eq!(read[2].features, [7.0]);
}
