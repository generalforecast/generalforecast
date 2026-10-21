// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The write-ahead log: encrypted, durable, and replayed in order.

use std::fs::OpenOptions;
use std::io::Write;

use std::path::Path;
use std::time::Instant;

use gf_core::node::{FittedColumns, RowLayout, Scaling, TableLayout, Targets};
use gf_core::pages::{Bucket, CellLayout, IndexLayout, PageId, PageLayout};
use gf_core::table::RowId;
use gf_server::columns::{Column, ColumnKind, Words};
use gf_server::name::TableName;
use gf_server::protocol::{RecencySpec, StrategySpec};
use gf_server::setup;
use gf_server::wal::{
    self, CreationId, EncryptionKey, HEAD_INTERVAL, Keys, Log, LogEntry, LogError, PastCapSpec,
    PastWrite, Record, Rows, TableDefinition, Term, Terms, TimeKind, decode,
};

/// Opens a log and replays it whole: its definition, its records, and the log.
fn open(path: &Path, keys: &Keys) -> Result<(TableDefinition, Vec<Record>, Log), LogError> {
    let mut replay = Log::open(path, keys)?;
    let definition = replay.definition().clone();
    let records = replay.by_ref().collect::<Result<Vec<_>, _>>()?;
    Ok((definition, records, replay.finish()?))
}

/// A log of `records` whose head covers them, as a running server's does.
fn written(path: &Path, records: &[Record]) -> Log {
    let mut log = Log::create(path, &key(7), &definition(), Terms::default()).unwrap();
    for record in records {
        log.append(record).unwrap();
    }
    log.save_head().unwrap();
    log
}

fn definition() -> TableDefinition {
    TableDefinition {
        name: TableName::new("payments").unwrap(),
        model: "general-forecast/tabicl-v2-classifier".to_string(),
        columns: vec![
            Column {
                name: "amount".to_string(),
                kind: ColumnKind::Numeric,
            },
            Column {
                name: "store".to_string(),
                kind: ColumnKind::Categorical,
            },
        ],
        classes: 3,
        strategy: StrategySpec::Retrieved {
            core: None,
            neighbors: 512,
            among: Some(2048),
            recency: Some(RecencySpec::Last(4096)),
        },
        page_size: 64,
        time_bucket: Some(3_600),
        refit_after: Some(0.25),
        refit_shift: None,
        fit_rows: None,
        past_cap: PastCapSpec::Clamp,
        regions: None,
        series: None,
        refit_cutoffs: "each".to_string(),
        retrieval_categoricals: "codes".to_string(),
        scale_rows: None,
        summary: Default::default(),
        created: CreationId::parse("0123456789abcdef"),
        times: TimeKind::Integer,
        route_by: None,
    }
}

fn key(byte: u8) -> EncryptionKey {
    EncryptionKey::from_hex(&format!("{byte:02x}").repeat(32)).unwrap()
}

fn keys(byte: u8) -> Keys {
    Keys::only(key(byte))
}

#[test]
fn an_incompatible_storage_header_is_rejected_without_rewriting() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    drop(written(&path, &[]));
    let original = std::fs::read(&path).unwrap();
    assert_eq!(&original[..6], b"GF_LOG");
    assert_eq!(original[6], 4);
    for (offset, value) in [(0, b'X'), (6, 3)] {
        let mut incompatible = original.clone();
        incompatible[offset] = value;
        std::fs::write(&path, &incompatible).unwrap();
        assert!(matches!(open(&path, &keys(7)), Err(LogError::Format(_))));
        assert_eq!(std::fs::read(&path).unwrap(), incompatible);
    }
}

fn records() -> Vec<Record> {
    vec![
        Record::Insert(Rows {
            keys: vec![1, 2],
            features: vec![0.5, f32::NAN, 1.0, 2.0],
            targets: Targets::Classes(vec![0, 2]),
            times: Some(vec![Some(-5), Some(1_700_000_000)]),
            id: None,
            words: vec![Words {
                column: 1,
                values: vec!["north".to_string(), "south".to_string()],
            }],
            writes: Vec::new(),
            layout: None,
            table: None,
        }),
        Record::Update(Rows {
            keys: vec![2],
            features: vec![3.0, 4.0],
            targets: Targets::Classes(vec![1]),
            times: None,
            id: None,
            words: Vec::new(),
            writes: Vec::new(),
            layout: None,
            table: None,
        }),
        Record::Upsert(Rows {
            keys: vec![2, 3],
            features: vec![5.0, 6.0, 7.0, 8.0],
            targets: Targets::Classes(vec![0, 1]),
            // A row without a time stays apart from one at the earliest time.
            times: Some(vec![None, Some(i64::MIN)]),
            id: None,
            words: Vec::new(),
            writes: Vec::new(),
            layout: None,
            table: None,
        }),
        Record::Delete {
            keys: vec![1],
            id: Some("write-4".to_string()),
        },
        Record::Refit,
    ]
}

fn same(left: &[Record], right: &[Record]) -> bool {
    // NaN never equals itself, so compare the bits.
    format!("{left:?}") == format!("{right:?}")
}

#[test]
fn records_replay_in_order_after_reopening() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    for record in records() {
        log.append(&record).unwrap();
    }
    drop(log);

    let (definition_again, replayed, _) = open(&path, &keys(7)).unwrap();

    assert_eq!(definition_again, definition());
    assert!(same(&replayed, &records()));
}

#[test]
fn the_log_is_unreadable_without_the_key() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    log.append(&records()[0]).unwrap();
    drop(log);

    let bytes = std::fs::read(&path).unwrap();
    assert!(!bytes.windows(8).any(|window| window == b"payments"));
    assert!(matches!(open(&path, &keys(8)), Err(LogError::Key { .. })));
}

#[test]
fn another_tables_log_does_not_open_as_this_ones() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    log.append(&records()[0]).unwrap();
    drop(log);

    let other = directory.join("refunds.log");
    std::fs::copy(&path, &other).unwrap();
    std::fs::copy(
        directory.join("payments.head"),
        directory.join("refunds.head"),
    )
    .unwrap();

    assert!(matches!(open(&other, &keys(7)), Err(LogError::Head(_))));
}

#[test]
fn a_torn_length_at_the_end_is_dropped_but_not_one_in_the_middle() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    log.append(&records()[0]).unwrap();
    drop(log);
    let whole = std::fs::read(&path).unwrap();

    // A final append torn after its length, with its complement unwritten.
    let mut torn = whole.clone();
    torn.extend_from_slice(&64u32.to_le_bytes());
    torn.extend_from_slice(&[0; 20]);
    std::fs::write(&path, &torn).unwrap();
    let (_, replayed, _) = open(&path, &keys(7)).unwrap();
    assert_eq!(replayed.len(), 1);
    assert_eq!(std::fs::read(&path).unwrap(), whole);

    // So is a tail of zeros, as a file extended before its data landed leaves.
    let mut zeros = whole.clone();
    zeros.extend_from_slice(&[0; 64]);
    std::fs::write(&path, &zeros).unwrap();
    let (_, replayed, _) = open(&path, &keys(7)).unwrap();
    assert_eq!(replayed.len(), 1);
    assert_eq!(std::fs::read(&path).unwrap(), whole);

    // The same break with a record after it is damage, not a tear.
    let mut log = open(&path, &keys(7)).unwrap().2;
    log.append(&records()[1]).unwrap();
    log.append(&records()[2]).unwrap();
    log.save_head().unwrap();
    drop(log);
    let mut damaged = std::fs::read(&path).unwrap();
    damaged[whole.len() + 4] ^= 1;
    std::fs::write(&path, &damaged).unwrap();
    assert!(matches!(
        open(&path, &keys(7)),
        Err(LogError::Length { .. })
    ));
}

#[test]
fn a_torn_final_record_is_dropped_and_appends_continue() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    log.append(&records()[0]).unwrap();
    drop(log);
    // A crash in the middle of a write leaves part of a record.
    let mut file = OpenOptions::new().append(true).open(&path).unwrap();
    let length = 40u32;
    file.write_all(&length.to_le_bytes()).unwrap();
    file.write_all(&(!length).to_le_bytes()).unwrap();
    file.write_all(&[1, 2, 3]).unwrap();
    drop(file);

    let (_, replayed, mut log) = open(&path, &keys(7)).unwrap();
    log.append(&records()[2]).unwrap();
    drop(log);
    let (_, again, _) = open(&path, &keys(7)).unwrap();

    assert!(same(&replayed, &records()[..1]));
    assert!(same(&again, &[records()[0].clone(), records()[2].clone()]));
}

#[test]
fn a_changed_byte_is_refused() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    drop(written(&path, &records()));
    let mut bytes = std::fs::read(&path).unwrap();
    // The last byte is in the final record's authentication tag.
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&path, bytes).unwrap();

    assert!(matches!(
        open(&path, &keys(7)),
        Err(LogError::Decrypt { .. })
    ));
}

#[test]
fn a_changed_length_is_refused_rather_than_taken_for_a_torn_write() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    drop(written(&path, &records()));
    let mut bytes = std::fs::read(&path).unwrap();
    // The first record follows the file header, then the definition and its prefix.
    let first = 16 + 8 + u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    bytes[first + 1] ^= 0x40;
    std::fs::write(&path, &bytes).unwrap();

    assert!(matches!(
        open(&path, &keys(7)),
        Err(LogError::Length { .. })
    ));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[test]
fn logs_read_back_from_any_sequence_for_replicas() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    for record in records() {
        log.append(&record).unwrap();
    }

    let entries = log.read(0, usize::MAX).unwrap();
    let tail: Vec<Record> = log
        .read(2, usize::MAX)
        .unwrap()
        .iter()
        .map(|(_, bytes)| record(bytes))
        .collect();

    // The definition, then each record.
    let length = records().len() as u64 + 1;
    assert_eq!(log.length(), length);
    let sequences: Vec<u64> = entries.iter().map(|(sequence, _)| *sequence).collect();
    assert_eq!(sequences, (0..length).collect::<Vec<_>>());
    assert_eq!(
        decode(&entries[0].1).unwrap(),
        LogEntry::Definition(definition(), Terms::default())
    );
    assert!(same(&tail, &records()[1..]));
    assert!(log.read(length, usize::MAX).unwrap().is_empty());
    // A read past the end, as from a replica ahead of this copy, is refused.
    for from in [length + 1, u64::MAX] {
        assert!(matches!(
            log.read(from, usize::MAX),
            Err(LogError::Beyond { .. })
        ));
    }

    drop(log);
    let (_, _, mut log) = open(&path, &keys(7)).unwrap();
    log.append(&Record::Refit).unwrap();
    assert_eq!(log.length(), length + 1);
    assert_eq!(log.read(length, usize::MAX).unwrap().len(), 1);
}

#[test]
fn compaction_keeps_sequence_numbers() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    for record in records() {
        log.append(&record).unwrap();
    }
    let snapshot = Record::Snapshot(Rows {
        keys: vec![2],
        features: vec![3.0, 4.0],
        targets: Targets::Classes(vec![1]),
        times: Some(vec![Some(9)]),
        id: None,
        words: Vec::new(),
        writes: vec![PastWrite {
            id: "retried".to_string(),
            rows: 1,
            log_length: 4,
        }],
        layout: None,
        table: None,
    });
    let before = std::fs::metadata(&path).unwrap().len();

    // Rows past one record follow the snapshot as inserts, from the same compaction.
    log.compact(&definition(), 5, &[snapshot.clone(), records()[0].clone()])
        .unwrap();
    log.append(&Record::Refit).unwrap();
    drop(log);
    let (definition_again, replayed, log) = open(&path, &keys(7)).unwrap();

    assert_eq!(definition_again, definition());
    assert!(same(
        &replayed,
        &[snapshot.clone(), records()[0].clone(), Record::Refit]
    ));
    assert_eq!(log.length(), 8);
    let sequences: Vec<u64> = log
        .read(0, usize::MAX)
        .unwrap()
        .iter()
        .map(|(sequence, _)| *sequence)
        .collect();
    assert_eq!(sequences, [0, 5, 6, 7]);
    // A replica that far behind gets the definition, for its terms, then the snapshot.
    let behind = log.read(2, usize::MAX).unwrap();
    let sequences: Vec<u64> = behind.iter().map(|(sequence, _)| *sequence).collect();
    assert_eq!(sequences, [0, 5, 6, 7]);
    assert!(same(&[record(&behind[1].1)], &[snapshot]));
    assert!(std::fs::metadata(&path).unwrap().len() < before);
}

#[test]
fn terms_survive_reopening_and_compaction() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    log.append(&records()[0]).unwrap();
    let promotion = Record::Term {
        term: 1,
        primary: "node-1".to_string(),
    };
    log.append(&promotion).unwrap();
    let terms = Terms::try_from(vec![
        Term::default(),
        Term {
            term: 1,
            primary: "node-1".to_string(),
            start: 2,
        },
    ])
    .unwrap();
    assert_eq!(log.terms(), &terms);
    drop(log);

    let (_, replayed, mut log) = open(&path, &keys(7)).unwrap();
    assert!(same(&replayed[1..], &[promotion]));
    assert_eq!(log.terms(), &terms);
    log.compact(&definition(), 3, &records()[..1]).unwrap();
    drop(log);

    let (_, _, log) = open(&path, &keys(7)).unwrap();
    assert_eq!(log.terms(), &terms);
    assert_eq!(
        decode(&log.read(1, usize::MAX).unwrap()[0].1).unwrap(),
        LogEntry::Definition(definition(), terms)
    );
}

fn record(bytes: &[u8]) -> Record {
    match decode(bytes).unwrap() {
        LogEntry::Record(record) => record,
        LogEntry::Definition(..) => panic!("only the first entry is the definition"),
    }
}

#[test]
fn keys_must_be_32_bytes_of_hex() {
    assert!(EncryptionKey::from_hex("abcd").is_err());
    assert!(EncryptionKey::from_hex(&"zz".repeat(32)).is_err());
    assert_ne!(key(7).id(), key(8).id());
    assert_eq!(key(7).id().len(), 16);
}

#[test]
fn a_log_under_a_previous_key_opens_and_is_rewritten_by_compaction() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    log.append(&records()[0]).unwrap();
    drop(log);
    let rotated = Keys {
        current: key(8),
        previous: vec![key(7)],
    };

    let (_, replayed, mut log) = open(&path, &rotated).unwrap();
    assert!(log.needs_rotation());
    assert!(matches!(log.append(&Record::Refit), Err(LogError::Rotate)));
    assert!(same(&replayed, &records()[..1]));
    log.compact(&definition(), 2, &records()[..1]).unwrap();
    log.append(&Record::Refit).unwrap();
    drop(log);

    let (_, again, log) = open(&path, &keys(8)).unwrap();
    assert!(!log.needs_rotation());
    assert!(same(&again, &[records()[0].clone(), Record::Refit]));
    assert!(matches!(open(&path, &keys(7)), Err(LogError::Key { .. })));
}

#[test]
fn records_moved_within_a_log_or_from_another_creation_are_refused() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let log = written(&path, &records());
    let entries: Vec<(u64, u64)> = {
        let bytes = std::fs::read(&path).unwrap();
        let mut offsets = Vec::new();
        let mut at = 16;
        while at + 8 <= bytes.len() {
            let length = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as u64;
            offsets.push((at as u64, 8 + length));
            at += 8 + length as usize;
        }
        offsets
    };
    drop(log);
    let whole = std::fs::read(&path).unwrap();

    // Swapping two complete records breaks the chain of tags.
    let (first, second) = (entries[1], entries[2]);
    let mut swapped = whole[..first.0 as usize].to_vec();
    swapped.extend_from_slice(&whole[second.0 as usize..(second.0 + second.1) as usize]);
    swapped.extend_from_slice(&whole[first.0 as usize..(first.0 + first.1) as usize]);
    swapped.extend_from_slice(&whole[(second.0 + second.1) as usize..]);
    std::fs::write(&path, &swapped).unwrap();
    assert!(matches!(
        open(&path, &keys(7)),
        Err(LogError::Decrypt { .. })
    ));

    // A record from a log of another creation of the same name does not open here.
    let other = directory.join("other").join("payments.log");
    std::fs::create_dir_all(other.parent().unwrap()).unwrap();
    let mut recreated = definition();
    recreated.created = CreationId::parse("fedcba9876543210");
    let mut log = Log::create(&other, &key(7), &recreated, Terms::default()).unwrap();
    log.append(&records()[0]).unwrap();
    drop(log);
    let foreign = std::fs::read(&other).unwrap();
    let mut spliced = whole[..entries[1].0 as usize].to_vec();
    spliced.extend_from_slice(&foreign[entries[1].0 as usize..]);
    std::fs::write(&path, &spliced).unwrap();
    assert!(matches!(
        open(&path, &keys(7)),
        Err(LogError::Decrypt { .. })
    ));
}

#[test]
fn staged_records_become_durable_together_and_read_in_pages() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    for record in &records()[..3] {
        log.stage(record).unwrap();
    }
    assert!(log.has_staged());
    log.sync().unwrap();
    assert!(!log.has_staged());
    drop(log);

    let (_, replayed, log) = open(&path, &keys(7)).unwrap();
    assert!(same(&replayed, &records()[..3]));
    let page: Vec<u64> = log
        .read(1, 2)
        .unwrap()
        .iter()
        .map(|(sequence, _)| *sequence)
        .collect();
    assert_eq!(page, [1, 2]);
    assert_eq!(log.read(3, 2).unwrap().len(), 1);
}

#[test]
fn a_log_that_lost_whole_records_from_its_end_is_refused() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let head = directory.join("payments.head");
    let mut log = written(&path, &records()[..1]);
    let (shorter, lagging) = (std::fs::read(&path).unwrap(), std::fs::read(&head).unwrap());
    log.append(&records()[1]).unwrap();
    log.append(&records()[2]).unwrap();
    log.save_head().unwrap();
    drop(log);
    let whole = std::fs::read(&path).unwrap();

    // Cut back to a record boundary, the log would replay as if whole.
    std::fs::write(&path, &shorter).unwrap();
    assert!(matches!(open(&path, &keys(7)), Err(LogError::Head(_))));
    // A head that lags its log, as after a crash, still opens.
    std::fs::write(&path, &whole).unwrap();
    std::fs::write(&head, &lagging).unwrap();
    let (_, replayed, _) = open(&path, &keys(7)).unwrap();
    assert!(same(&replayed, &records()[..3]));
    // A log without its head is refused, since every log is created with one.
    std::fs::remove_file(&head).unwrap();
    assert!(matches!(open(&path, &keys(7)), Err(LogError::Head(_))));
}

#[test]
fn a_crash_on_either_side_of_compactions_rename_leaves_a_whole_log() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let head = directory.join("payments.head");
    let mut log = written(&path, &records());
    let (old, old_head) = (std::fs::read(&path).unwrap(), std::fs::read(&head).unwrap());
    let snapshot = Record::Snapshot(Rows {
        keys: vec![2],
        features: vec![3.0, 4.0],
        targets: Targets::Classes(vec![1]),
        times: Some(vec![Some(9)]),
        id: None,
        words: Vec::new(),
        writes: Vec::new(),
        layout: None,
        table: None,
    });
    log.compact(&definition(), 5, std::slice::from_ref(&snapshot))
        .unwrap();
    drop(log);
    let new = std::fs::read(&path).unwrap();

    // Before the rename: the old log with its head, beside a part-written new one.
    std::fs::write(&path, &old).unwrap();
    std::fs::write(&head, &old_head).unwrap();
    std::fs::write(
        directory.join("payments.compacting.7"),
        &new[..new.len() / 2],
    )
    .unwrap();
    std::fs::write(directory.join("payments.compacting"), &new[..new.len() / 2]).unwrap();
    let (_, replayed, mut log) = open(&path, &keys(7)).unwrap();
    assert!(same(&replayed, &records()));
    assert!(!directory.join("payments.compacting.7").exists());
    assert!(!directory.join("payments.compacting").exists());
    // The next compaction writes over what the crash left.
    log.compact(&definition(), 5, std::slice::from_ref(&snapshot))
        .unwrap();
    drop(log);
    assert!(same(
        &open(&path, &keys(7)).unwrap().1,
        std::slice::from_ref(&snapshot)
    ));

    // After the rename but before the head caught up: the new log, the old head.
    std::fs::write(&path, &new).unwrap();
    std::fs::write(&head, &old_head).unwrap();
    let (_, replayed, log) = open(&path, &keys(7)).unwrap();
    assert!(same(&replayed, &[snapshot]));
    assert_eq!(log.length(), 6);
}

#[test]
fn a_late_compaction_of_a_dropped_table_leaves_its_successor_alone() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let old = written(&path, &records()[..2]);
    let mut old_compactor = old.begin_compaction(&definition(), old.length()).unwrap();
    old_compactor.append(&records()[0]).unwrap();
    let old_staged = directory.join(format!("payments.compacting.{}", old_compactor.number()));

    drop(old);
    wal::remove(&path).unwrap();
    let mut new_definition = definition();
    new_definition.created = Some(CreationId::new(1, [2, 3]));
    let mut new = Log::create(&path, &key(7), &new_definition, Terms::default()).unwrap();
    new.append(&records()[2]).unwrap();
    new.append(&records()[3]).unwrap();
    assert!(!old_staged.exists());

    let mut new_compactor = new.begin_compaction(&new_definition, new.length()).unwrap();
    let snapshot = Record::Snapshot(Rows {
        keys: vec![2],
        features: vec![3.0, 4.0],
        targets: Targets::Classes(vec![1]),
        times: Some(vec![Some(9)]),
        id: None,
        words: Vec::new(),
        writes: Vec::new(),
        layout: None,
        table: None,
    });
    new_compactor.append(&snapshot).unwrap();
    let new_staged = directory.join(format!("payments.compacting.{}", new_compactor.number()));

    old_compactor.append(&records()[1]).unwrap();
    old_compactor.finish().unwrap().discard();
    new.install_compaction(new_compactor.finish().unwrap())
        .unwrap();
    drop(new);

    let (replayed_definition, replayed, log) = open(&path, &keys(7)).unwrap();
    assert_eq!(replayed_definition, new_definition);
    assert!(same(&replayed, std::slice::from_ref(&snapshot)));
    assert_eq!(log.length(), 4);
    drop(log);
    assert!(!new_staged.exists());
    assert!(std::fs::read_dir(&directory).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("payments.compacting")
    }));
}

#[test]
fn a_closed_log_heads_every_acknowledged_record() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    for record in &records()[..3] {
        log.append(record).unwrap();
    }
    log.close().unwrap();

    assert!(same(&open(&path, &keys(7)).unwrap().1, &records()[..3]));
    let mut damaged = std::fs::read(&path).unwrap();
    let last = damaged.len() - 1;
    damaged[last] ^= 1;
    std::fs::write(&path, damaged).unwrap();
    assert!(matches!(
        open(&path, &keys(7)),
        Err(LogError::Decrypt { .. })
    ));
}

#[test]
fn a_damaged_record_past_the_head_is_a_torn_append_and_is_cut_off() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = written(&path, &records()[..1]);
    let synced = std::fs::read(&path).unwrap();
    // Written but never synced, so never acknowledged.
    log.stage(&records()[1]).unwrap();
    drop(log);
    // A crash tore the last append: its length landed, its bytes as zeros.
    let mut torn = std::fs::read(&path).unwrap();
    let end = torn.len();
    torn[synced.len() + 8..end].fill(0);
    std::fs::write(&path, &torn).unwrap();

    // A check reads it without changing the file.
    let mut replay = Log::open_read_only(&path, &keys(7)).unwrap();
    assert_eq!(replay.by_ref().count(), 1);
    replay.finish().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), torn);
    // Opening cuts it off, since it was never acknowledged.
    let (_, replayed, mut log) = open(&path, &keys(7)).unwrap();
    assert!(same(&replayed, &records()[..1]));
    assert_eq!(std::fs::read(&path).unwrap(), synced);

    // The same damage to a synced record, which its head covers, is refused.
    log.append(&records()[1]).unwrap();
    log.save_head().unwrap();
    drop(log);
    std::fs::write(&path, &torn).unwrap();
    assert!(matches!(
        open(&path, &keys(7)),
        Err(LogError::Decrypt { .. })
    ));
}

#[test]
fn startup_refuses_a_missing_key_keeps_short_logs_and_sets_damaged_ones_aside() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    drop(written(&path, &records()));
    let whole = std::fs::read(&path).unwrap();

    assert!(matches!(
        setup::open_log(&path, &keys(8)),
        Err(LogError::Key { .. })
    ));
    assert!(setup::open_log(&path, &keys(7)).unwrap().is_some());
    // A log its head says lost records stays in place, for the next start.
    std::fs::write(&path, &whole[..whole.len() - 1]).unwrap();
    let mut replay = setup::open_log(&path, &keys(7)).unwrap().unwrap();
    assert!(replay.by_ref().any(|record| record.is_err()));
    assert!(path.exists());
    // A damaged log moves aside with its head, so a replica can take its place.
    std::fs::write(&path, b"not a log at all").unwrap();
    assert!(setup::open_log(&path, &keys(7)).unwrap().is_none());
    assert!(!path.exists());
    assert!(directory.join("payments.log.unreadable").exists());
    assert!(directory.join("payments.head.unreadable").exists());
}

fn tempdir() -> std::path::PathBuf {
    // Tests run in parallel, so a counter keeps their directories apart.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("gf-wal-{}-{index}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn rotation_keeps_every_entry_and_its_sequence() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    drop(written(&path, &records()));
    let rotated = Keys {
        current: key(8),
        previous: vec![key(7)],
    };
    let (_, before, mut log) = open(&path, &rotated).unwrap();
    let entries = log.read(0, usize::MAX).unwrap();

    log.rotate().unwrap();

    // The same entries at the same sequence numbers, as copies elsewhere hold
    // them, where a compaction would have added one.
    assert!(!log.needs_rotation());
    assert_eq!(log.read(0, usize::MAX).unwrap(), entries);
    drop(log);
    let (_, after, log) = open(&path, &keys(8)).unwrap();
    assert!(same(&after, &before));
    assert_eq!(log.length(), records().len() as u64 + 1);
    log.read(1, usize::MAX).unwrap();
}

#[test]
fn definitions_are_checked_once_into_their_settings() {
    let config = definition().config().unwrap();
    assert_eq!(config.page_size.get(), 64);
    assert_eq!(config.time_bucket.map(|width| width.get()), Some(3_600));
    // A bucket of no width was taken as none; it is refused, as clients refuse it.
    let mut unbucketed = definition();
    unbucketed.time_bucket = Some(0);
    assert!(unbucketed.config().is_err());
    let mut unbounded = definition();
    unbounded.refit_after = Some(-1.0);
    assert!(unbounded.config().is_err());
}

#[test]
fn terms_are_never_empty_and_an_empty_creation_is_none() {
    assert!(Terms::try_from(Vec::new()).is_err());
    assert!(serde_json::from_str::<Terms>("[]").is_err());
    assert_eq!(Terms::default().latest(), &Term::default());

    let mut written = serde_json::to_value(definition()).unwrap();
    written["created"] = serde_json::json!("");
    let read: TableDefinition = serde_json::from_value(written).unwrap();
    assert_eq!(read.created, None);
    assert_eq!(CreationId::parse(""), None);
}

#[test]
fn table_names_are_checked_where_they_are_read() {
    for name in ["payments", "a", "A-b_9", &"x".repeat(128)] {
        assert_eq!(TableName::new(name).unwrap().as_str(), name);
    }
    for name in ["", "../outside", "a/b", "a.log", "naïve", &"x".repeat(129)] {
        assert!(TableName::new(name).is_err(), "{name:?}");
    }
    let mut written = serde_json::to_value(definition()).unwrap();
    written["name"] = serde_json::json!("../outside");
    let read = serde_json::from_value::<TableDefinition>(written);
    assert!(read.unwrap_err().to_string().contains("table names are"));
}

#[test]
fn the_head_follows_syncs_every_so_often_and_on_request() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let head = directory.join("payments.head");
    let created = Instant::now();
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    let at_creation = std::fs::read(&head).unwrap();

    // A sync makes the records durable without writing the head each time,
    // which would cost three more syncs; the head still covers its creation.
    log.append(&records()[0]).unwrap();
    log.sync().unwrap();
    // A sync after `HEAD_INTERVAL` writes the head too, so it stays put
    // only while less time than that has passed since it was written.
    if created.elapsed() < HEAD_INTERVAL {
        assert_eq!(std::fs::read(&head).unwrap(), at_creation);
    }
    drop(log);
    let opened = Instant::now();
    let (_, replayed, mut log) = open(&path, &keys(7)).unwrap();
    assert!(same(&replayed, &records()[..1]));

    // After enough records the head follows on its own, and a save brings
    // it up to date at once.
    for _ in 0..64 {
        log.append(&records()[1]).unwrap();
        log.sync().unwrap();
    }
    let followed = std::fs::read(&head).unwrap();
    assert_ne!(followed, at_creation);
    log.append(&records()[2]).unwrap();
    log.sync().unwrap();
    if opened.elapsed() < HEAD_INTERVAL {
        assert_eq!(std::fs::read(&head).unwrap(), followed);
    }
    log.save_head().unwrap();
    assert_ne!(std::fs::read(&head).unwrap(), followed);
    drop(log);
    assert_eq!(open(&path, &keys(7)).unwrap().2.length(), 67);
}

#[test]
fn a_compactions_layout_comes_back_to_the_bit() {
    let directory = tempdir();
    let path = directory.join("payments.log");
    let mut log = Log::create(&path, &key(7), &definition(), Terms::default()).unwrap();
    // Values that only round-trip to the bit survive the log.
    let snapshot = Record::Snapshot(Rows {
        keys: vec![2],
        features: vec![3.0, 4.0],
        targets: Targets::Classes(vec![1]),
        times: Some(vec![Some(9)]),
        id: None,
        words: Vec::new(),
        writes: Vec::new(),
        layout: Some(RowLayout {
            ids: vec![RowId(5)],
            pages: vec![PageId(2)],
            new_pages: vec![PageLayout {
                id: PageId(2),
                bucket: Bucket(3),
                sum: vec![0.1 + 0.2, -1e-300],
                radius: 1.0 / 3.0,
            }],
        }),
        table: Some(Box::new(TableLayout {
            rows: 1,
            ids: 6,
            index: IndexLayout {
                cells: vec![CellLayout {
                    center: vec![0.3, f64::MIN_POSITIVE],
                    radius: 2.0 / 3.0,
                    reach: 0.7,
                    pages: vec![(Bucket(3), vec![PageId(2)])],
                }],
                next_page: 4,
                reach: 0.7,
                built_reach: 0.35,
            },
            scaling: Some(Scaling {
                center: vec![0.1, 0.2],
                scale: vec![1.5, f32::MIN_POSITIVE],
                lookups: vec![(vec![4.0], vec![0.25, 0.75, 0.5, 0.5])],
                rows: 6,
            }),
            column_sums: vec![3.0, 4.0],
            column_counts: vec![1, 1],
            fitted: Some(FittedColumns {
                rows: 6,
                means: vec![0.1, 0.2],
                deviations: vec![0.0, 1e-9],
            }),
            changed: 2,
        })),
    });

    log.append(&snapshot).unwrap();
    drop(log);
    let (_, replayed, _) = open(&path, &keys(7)).unwrap();

    assert_eq!(replayed, [snapshot]);
}

#[test]
fn series_definitions_name_their_columns_and_replay() {
    use gf_core::forecasting::GroupStrategy;
    use gf_server::wal::{GroupSpec, SeriesDefinition};

    let series = |known: Vec<&str>, step: u64| {
        Some(Box::new(SeriesDefinition {
            series: "item".to_string(),
            time: "week".to_string(),
            target: "units".to_string(),
            step,
            known: known.into_iter().map(str::to_string).collect(),
            group: GroupSpec::Random { size: 4, seed: 1 },
            horizon: Some(8),
        }))
    };
    // Heads are kept by table name, so this table's name is its own; a
    // table of series takes no strategy or time bucket.
    let mut definition = TableDefinition {
        name: TableName::new("sales").unwrap(),
        classes: 0,
        strategy: StrategySpec::default(),
        time_bucket: None,
        series: series(vec!["store"], 7),
        ..definition()
    };

    let settings = definition.series_settings().unwrap().unwrap();

    // The known column's model feature, after the numeric column's one.
    assert_eq!(settings.known, vec![1]);
    assert_eq!(settings.step.get(), 7);
    assert!(matches!(
        settings.group,
        GroupStrategy::Random { seed: 1, .. }
    ));
    definition.series = series(vec!["price"], 7);
    assert!(definition.series_settings().unwrap_err().contains("price"));
    definition.series = series(vec![], 0);
    assert!(definition.series_settings().unwrap_err().contains("step"));
    definition.series = series(vec![], 7);
    definition.columns[0].name = "item".to_string();
    assert!(
        definition
            .series_settings()
            .unwrap_err()
            .contains("series, time and target")
    );

    let directory = tempdir();
    let path = directory.join("sales.log");
    definition.columns[0].name = "amount".to_string();
    let log = Log::create(&path, &key(7), &definition, Terms::default()).unwrap();
    drop(log);
    let (replayed, _, _) = open(&path, &keys(7)).unwrap();
    assert_eq!(replayed, definition);
}
