//! A table's rows: identity, versions, the change log and snapshots.
//!
//! The table itself lives outside generalforecast. This is generalforecast's view of it, built from the
//! source's change stream, and what every cached state is keyed on.

use std::fmt;
use std::hash::{BuildHasher, RandomState};
use std::sync::Arc;

use roaring::RoaringBitmap;

use crate::context::{Context, Fingerprint};

/// How many row ids a table has: every `u32`.
pub const ROW_IDS: u64 = u32::MAX as u64 + 1;
/// A table served by a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableId(pub u32);

/// A row's key in the source table: 128 bits, as clients map their keys.
///
/// An integer key keeps its value, a negative one in two's complement; a UUID
/// keeps its 128 bits; a string or bytes key takes a 128-bit digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RowKey(pub u128);

impl fmt::Display for RowKey {
    /// An integer key as the integer, and any other as hex, as a digest reads.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let signed = self.0 as i128;
        if let Ok(value) = u64::try_from(self.0) {
            write!(formatter, "{value}")
        } else if let Ok(value) = i64::try_from(signed) {
            write!(formatter, "{value}")
        } else {
            write!(formatter, "0x{:032x}", self.0)
        }
    }
}

/// A row's dense id in generalforecast, assigned on insert and never reused.
///
/// So a table takes at most `2^32` inserts over its life, updates and deletes
/// not counting; after that it refuses inserts with `TableError::Full`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RowId(pub u32);

/// How many times a row has been updated since it was inserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Version(pub u64);

/// The number of changes a table has applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogPosition(pub u64);

/// One change from the source's change stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    Insert(RowKey),
    Update(RowKey),
    Delete(RowKey),
}

/// Why a change could not be applied. The table is left unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableError {
    DuplicateKey(RowKey),
    UnknownKey(RowKey),
    /// Every `u32` row id has been assigned.
    Full,
    /// A restored row whose id is live already or was never taken.
    Restore(RowId),
}

impl fmt::Display for TableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateKey(key) => write!(formatter, "row key {key} already exists"),
            Self::UnknownKey(key) => write!(formatter, "row key {key} does not exist"),
            Self::Full => write!(formatter, "every row id has been assigned"),
            Self::Restore(row) => write!(formatter, "row {} cannot be restored", row.0),
        }
    }
}

impl std::error::Error for TableError {}

/// Live keys and their rows, in about 16 bytes of key and 8 to 21 of table a
/// row, where a hash map of keys takes 50 to 70.
///
/// Each row id's key is kept by row id, and an open-addressed table holds the
/// live rows with 31 bits of their keys' hashes, so a probe reads a key only
/// when those bits match. Hashes are keyed per table, so chosen keys cannot
/// make probes long.
#[derive(Debug, Default)]
struct KeyMap {
    /// Per slot, 0 if empty, else the hash bits with the top bit set and the row.
    slots: Vec<u64>,
    /// Each row id's key, live or since deleted.
    keys: Vec<RowKey>,
    live: usize,
    hasher: RandomState,
}

impl KeyMap {
    fn hash(&self, key: RowKey) -> u64 {
        self.hasher.hash_one(key.0)
    }

    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    fn tag(hash: u64) -> u64 {
        (hash | 1 << 63) & !u64::from(u32::MAX)
    }

    /// The slot holding `key`, if it is live.
    fn find(&self, key: RowKey) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = self.hash(key);
        let mut slot = hash as usize & self.mask();
        loop {
            let held = self.slots[slot];
            if held == 0 {
                return None;
            }
            if held & !u64::from(u32::MAX) == Self::tag(hash)
                && self.keys[held as u32 as usize] == key
            {
                return Some(slot);
            }
            slot = (slot + 1) & self.mask();
        }
    }

    fn get(&self, key: RowKey) -> Option<RowId> {
        self.find(key).map(|slot| RowId(self.slots[slot] as u32))
    }

    /// Adds a key that is not live.
    fn insert(&mut self, key: RowKey, row: RowId) {
        if (self.live + 1) * 4 > self.slots.len() * 3 {
            self.grow();
        }
        let index = row.0 as usize;
        if index >= self.keys.len() {
            self.keys.resize(index + 1, RowKey(0));
        }
        self.keys[index] = key;
        self.place(self.hash(key), row);
        self.live += 1;
    }

    fn place(&mut self, hash: u64, row: RowId) {
        let mut slot = hash as usize & self.mask();
        while self.slots[slot] != 0 {
            slot = (slot + 1) & self.mask();
        }
        self.slots[slot] = Self::tag(hash) | u64::from(row.0);
    }

    fn grow(&mut self) {
        let size = (self.slots.len() * 2).max(16);
        let old = std::mem::replace(&mut self.slots, vec![0; size]);
        for held in old.into_iter().filter(|&held| held != 0) {
            let row = RowId(held as u32);
            self.place(self.hash(self.keys[row.0 as usize]), row);
        }
    }

    /// Removes a live key, shifting later slots of its run back, so no
    /// probe ever passes a hole.
    fn remove(&mut self, key: RowKey) {
        let Some(mut hole) = self.find(key) else {
            return;
        };
        let mask = self.mask();
        let mut next = (hole + 1) & mask;
        while self.slots[next] != 0 {
            let row = self.slots[next] as u32 as usize;
            let home = self.hash(self.keys[row]) as usize & mask;
            // The entry can fill the hole if its home is not between the hole
            // and itself, going around.
            if (next.wrapping_sub(home) & mask) >= (next.wrapping_sub(hole) & mask) {
                self.slots[hole] = self.slots[next];
                hole = next;
            }
            next = (next + 1) & mask;
        }
        self.slots[hole] = 0;
        self.live -= 1;
    }

    fn iter(&self) -> impl Iterator<Item = (RowKey, RowId)> + '_ {
        self.slots
            .iter()
            .filter(|&&held| held != 0)
            .map(|&held| (self.keys[held as u32 as usize], RowId(held as u32)))
    }
}

/// Each live row's key, which seeded samples hash and ties fall to.
pub trait Keys {
    fn key(&self, row: RowId) -> u128;
}

impl Keys for Table {
    fn key(&self, row: RowId) -> u128 {
        Table::key(self, row)
            .expect("samples and ties read live rows alone")
            .0
    }
}

/// generalforecast's view of a source table, kept current by applying its changes in order.
///
/// Versions and the live set are shared with snapshots and copied on the first
/// write after a snapshot, so taking a snapshot is cheap.
#[derive(Debug, Default)]
pub struct Table {
    keys: KeyMap,
    versions: Arc<Vec<Version>>,
    live: Arc<RoaringBitmap>,
    /// The fingerprint of every live row, kept as rows change.
    fingerprint: Fingerprint,
    position: LogPosition,
}

impl Table {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies the next change from the source and advances the log.
    ///
    /// # Errors
    ///
    /// If the change inserts an existing key, changes a missing one, or no row
    /// ids are left. The table is then unchanged.
    pub fn apply(&mut self, change: Change) -> Result<RowId, TableError> {
        let row = match change {
            Change::Insert(key) => self.insert(key)?,
            Change::Update(key) => {
                let row = self.existing(key)?;
                let version = &mut Arc::make_mut(&mut self.versions)[row.0 as usize];
                let old = Fingerprint::of(row, *version);
                version.0 += 1;
                self.fingerprint = self.fingerprint - old + Fingerprint::of(row, *version);
                row
            }
            Change::Delete(key) => {
                let row = self.existing(key)?;
                self.keys.remove(key);
                Arc::make_mut(&mut self.live).remove(row.0);
                self.fingerprint =
                    self.fingerprint - Fingerprint::of(row, self.versions[row.0 as usize]);
                row
            }
        };
        self.position.0 += 1;
        Ok(row)
    }

    /// An empty table whose first `ids` row ids are taken, as a restore from
    /// a compacted log begins: its rows come back under their own ids.
    #[must_use]
    pub fn with_ids(ids: u32) -> Self {
        Self {
            versions: Arc::new(vec![Version(0); ids as usize]),
            ..Self::default()
        }
    }

    /// Restores a live row under the id it had, at version 0.
    ///
    /// # Errors
    ///
    /// If the key or the row is live already, or the row id was never taken.
    pub fn restore(&mut self, key: RowKey, row: RowId) -> Result<(), TableError> {
        if self.keys.get(key).is_some() {
            return Err(TableError::DuplicateKey(key));
        }
        if row.0 as usize >= self.versions.len() || self.live.contains(row.0) {
            return Err(TableError::Restore(row));
        }
        Arc::make_mut(&mut self.live).insert(row.0);
        self.keys.insert(key, row);
        self.fingerprint = self.fingerprint + Fingerprint::of(row, Version(0));
        Ok(())
    }

    fn insert(&mut self, key: RowKey) -> Result<RowId, TableError> {
        if self.keys.get(key).is_some() {
            return Err(TableError::DuplicateKey(key));
        }
        let row = RowId(u32::try_from(self.versions.len()).map_err(|_| TableError::Full)?);
        Arc::make_mut(&mut self.versions).push(Version(0));
        Arc::make_mut(&mut self.live).insert(row.0);
        self.keys.insert(key, row);
        self.fingerprint = self.fingerprint + Fingerprint::of(row, Version(0));
        Ok(row)
    }

    fn existing(&self, key: RowKey) -> Result<RowId, TableError> {
        self.keys.get(key).ok_or(TableError::UnknownKey(key))
    }

    /// How many more rows can be inserted, since row ids are never reused.
    #[must_use]
    pub fn ids_left(&self) -> u64 {
        ROW_IDS.saturating_sub(self.ids_used())
    }

    /// How many row ids rows have taken, live or since deleted.
    #[must_use]
    pub fn ids_used(&self) -> u64 {
        self.versions.len() as u64
    }

    /// Every live key and its row, in no particular order.
    pub fn keys(&self) -> impl Iterator<Item = (RowKey, RowId)> + '_ {
        self.keys.iter()
    }

    /// The row a key currently maps to, if the key exists.
    #[must_use]
    pub fn row(&self, key: RowKey) -> Option<RowId> {
        self.keys.get(key)
    }

    /// A live row's key, or None if the row is not live. A deleted row's id
    /// may name no key: a table restored from a compacted log takes the ids
    /// of rows that had gone without their keys.
    #[must_use]
    pub fn key(&self, row: RowId) -> Option<RowKey> {
        self.live
            .contains(row.0)
            .then(|| self.keys.keys[row.0 as usize])
    }

    /// A live row's version, or None if the row is not live.
    #[must_use]
    pub fn version(&self, row: RowId) -> Option<Version> {
        self.live
            .contains(row.0)
            .then(|| self.versions[row.0 as usize])
    }

    /// How many changes the table has applied.
    #[must_use]
    pub fn position(&self) -> LogPosition {
        self.position
    }

    /// Number of live rows.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.live.len()
    }

    /// Whether the table has no live rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// The table as of now. Later changes do not affect it.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            position: self.position,
            versions: Arc::clone(&self.versions),
            live: Arc::clone(&self.live),
            fingerprint: self.fingerprint,
        }
    }
}

/// A table's live rows and their versions at one log position.
///
/// Every request reads one snapshot, so it never sees half of a change.
#[derive(Clone, Debug)]
pub struct Snapshot {
    position: LogPosition,
    versions: Arc<Vec<Version>>,
    live: Arc<RoaringBitmap>,
    fingerprint: Fingerprint,
}

impl Snapshot {
    /// The log position the snapshot was taken at.
    #[must_use]
    pub fn position(&self) -> LogPosition {
        self.position
    }

    /// Every live row.
    #[must_use]
    pub fn rows(&self) -> Context {
        Context::from_bitmap(self.live.as_ref().clone())
    }

    /// The snapshot with `rows` no longer live, such as rows a plan must
    /// not read.
    #[must_use]
    pub fn without(&self, rows: &RoaringBitmap) -> Self {
        let gone = self.live.as_ref() & rows;
        if gone.is_empty() {
            return self.clone();
        }
        let prints: Fingerprint = gone
            .iter()
            .map(|row| Fingerprint::of(RowId(row), self.versions[row as usize]))
            .sum();
        Self {
            position: self.position,
            versions: Arc::clone(&self.versions),
            live: Arc::new(self.live.as_ref() - &gone),
            fingerprint: self.fingerprint - prints,
        }
    }

    /// A live row's version, or None if the row is not live.
    #[must_use]
    pub fn version(&self, row: RowId) -> Option<Version> {
        self.live
            .contains(row.0)
            .then(|| self.versions[row.0 as usize])
    }

    /// The fingerprint of a context's live rows at their versions in this snapshot.
    ///
    /// Rows that are not live are ignored. Every live row's fingerprint is kept
    /// as the table changes, so a context of the whole table costs no hashing.
    #[must_use]
    pub fn fingerprint(&self, context: &Context) -> Fingerprint {
        if context.bitmap() == self.live.as_ref() {
            return self.fingerprint;
        }
        (context.bitmap() & self.live.as_ref())
            .iter()
            .map(|row| Fingerprint::of(RowId(row), self.versions[row as usize]))
            .sum()
    }
}
