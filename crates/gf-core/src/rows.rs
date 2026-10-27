// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rows by id, in small blocks that stay in memory while a budget allows and
//! spill to a file on disk when it does not.
//!
//! Each row is its features, then its target's bits, then its key, so a
//! block of a few kilobytes holds consecutive row ids. The node's tables share
//! one pool of memory for their blocks: once the pool passes its budget, the
//! least recently used blocks leave memory, written first, sealed, to the end
//! of their table's row file if they changed since they were last there. A
//! read brings a block back. A changed block is written again rather than in
//! place, and the copies it leaves behind go when the file is rewritten.
//!
//! Blocks never change once made: a write makes a new block, so a snapshot,
//! which holds the blocks it saw, reads the rows as they were while writes go
//! on, as a compaction in the background does.
//!
//! A pool without a directory keeps every block in memory, as a process that
//! owns its rows does; a server gives each table a file and a sealer, so rows
//! at rest are encrypted as its logs are.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::table::RowId;

/// Bytes of a block, about: a block holds as many whole rows as fit, at least one.
pub const BLOCK_BYTES: usize = 4096;

/// Seals blocks before they are written, and opens them when they are read.
pub trait Sealer: Send + Sync + fmt::Debug {
    /// The sealed bytes of a block's contents, bound to its place and version.
    fn seal(&self, block: u32, version: u32, plain: &[u8]) -> Vec<u8>;

    /// A block's contents, or None if the bytes are not this block's at this version.
    fn open(&self, block: u32, version: u32, sealed: &[u8]) -> Option<Vec<u8>>;
}

/// Why rows could not be read or written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowsError {
    /// The row file could not be read or written.
    Io(io::ErrorKind),
    /// A block on disk does not open: it was changed, swapped or torn.
    Damaged { block: u32 },
    /// A row id the store does not hold.
    Unknown(RowId),
}

impl fmt::Display for RowsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(kind) => write!(formatter, "the row file failed: {kind}"),
            Self::Damaged { block } => write!(formatter, "row block {block} is damaged"),
            Self::Unknown(row) => write!(formatter, "row {} is not stored", row.0),
        }
    }
}

impl std::error::Error for RowsError {}

impl From<io::Error> for RowsError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}

/// Where a block's last written copy is in its table's file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Location {
    offset: u64,
    length: u32,
    version: u32,
}

/// A block's bytes, shared with the snapshots that saw them.
type Bytes = Arc<Vec<u8>>;

/// One table's blocks: those in memory, and where each was last written.
#[derive(Debug)]
struct Store {
    row_bytes: usize,
    rows_per_block: usize,
    rows: u32,
    memory: HashMap<u32, Resident>,
    disk: Vec<Option<Location>>,
    file: Option<Arc<RowFile>>,
    /// Bytes of the file that no block points to any more.
    garbage: u64,
}

/// A block in memory: its bytes, whether its file copy is older, and when it was last read.
#[derive(Debug)]
struct Resident {
    bytes: Bytes,
    dirty: bool,
    used: u64,
}

/// A table's row file and the sealer its blocks go through.
#[derive(Debug)]
struct RowFile {
    file: File,
    path: PathBuf,
    sealer: Arc<dyn Sealer>,
    length: Mutex<u64>,
}

impl RowFile {
    /// A new, empty row file at `path`, sealed by `sealer`. A file there
    /// already is unlinked rather than emptied, so snapshots reading it keep
    /// the blocks they saw.
    fn create(path: &Path, sealer: Arc<dyn Sealer>) -> Result<Self, RowsError> {
        match std::fs::remove_file(path) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error.into()),
            _ => {}
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
            sealer,
            length: Mutex::new(0),
        })
    }

    fn read(&self, block: u32, location: Location) -> Result<Vec<u8>, RowsError> {
        let mut sealed = vec![0; location.length as usize];
        self.file.read_exact_at(&mut sealed, location.offset)?;
        self.sealer
            .open(block, location.version, &sealed)
            .ok_or(RowsError::Damaged { block })
    }

    fn write(&self, block: u32, version: u32, plain: &[u8]) -> Result<Location, RowsError> {
        let sealed = self.sealer.seal(block, version, plain);
        let mut length = self.length.lock().unwrap_or_else(PoisonError::into_inner);
        self.file.write_all_at(&sealed, *length)?;
        let location = Location {
            offset: *length,
            length: u32::try_from(sealed.len()).expect("a sealed block is small"),
            version,
        };
        *length += sealed.len() as u64;
        Ok(location)
    }
}

/// A table's place in the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StoreId(u64);

#[derive(Debug, Default)]
struct Pool {
    stores: HashMap<StoreId, Store>,
    next_store: u64,
    /// Blocks in memory by when they were last used, oldest first.
    order: BTreeMap<u64, (StoreId, u32)>,
    clock: u64,
    bytes: u64,
    budget: Option<u64>,
}

/// Memory the node's tables share for their rows, and its budget.
#[derive(Clone, Debug, Default)]
pub struct RowMemory(Arc<Mutex<Pool>>);

impl RowMemory {
    /// A pool that keeps every block in memory.
    #[must_use]
    pub fn unbounded() -> Self {
        Self::default()
    }

    /// A pool that keeps at most `bytes` of blocks in memory, as far as tables
    /// with files can spill; a table without one keeps its blocks regardless.
    #[must_use]
    pub fn bounded(bytes: u64) -> Self {
        let memory = Self::default();
        memory.lock().budget = Some(bytes);
        memory
    }

    /// Bytes of blocks in memory.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.lock().bytes
    }

    fn lock(&self) -> MutexGuard<'_, Pool> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One table's rows, each `width` features, a target and a key.
///
/// Targets are also kept in memory, four bytes a row, so planning can read a
/// row's label without its block.
#[derive(Debug)]
pub struct RowStore {
    memory: RowMemory,
    id: StoreId,
    width: usize,
    targets: Vec<u32>,
}

/// The rows a store held when the snapshot was taken, readable while it changes.
#[derive(Clone, Debug)]
pub struct RowSnapshot {
    width: usize,
    row_bytes: usize,
    rows_per_block: usize,
    rows: u32,
    memory: HashMap<u32, Bytes>,
    disk: Vec<Option<Location>>,
    file: Option<Arc<RowFile>>,
}

/// One row: its features, its target's bits and its key.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub features: Vec<f32>,
    pub target: u32,
    pub key: u128,
}

impl RowStore {
    /// An empty store in `memory` for rows of `width` features, kept in memory only.
    #[must_use]
    pub fn new(memory: &RowMemory, width: usize) -> Self {
        Self::make(memory, width, None)
    }

    /// An empty store that spills to a new file at `path`, sealed by `sealer`.
    /// A file there already is replaced.
    ///
    /// # Errors
    ///
    /// If the file cannot be made.
    pub fn on_disk(
        memory: &RowMemory,
        width: usize,
        path: &Path,
        sealer: Arc<dyn Sealer>,
    ) -> Result<Self, RowsError> {
        let file = RowFile::create(path, sealer)?;
        Ok(Self::make(memory, width, Some(Arc::new(file))))
    }

    fn make(memory: &RowMemory, width: usize, file: Option<Arc<RowFile>>) -> Self {
        let row_bytes = width * 4 + 4 + 16;
        let mut pool = memory.lock();
        let id = StoreId(pool.next_store);
        pool.next_store += 1;
        pool.stores.insert(
            id,
            Store {
                row_bytes,
                rows_per_block: (BLOCK_BYTES / row_bytes).max(1),
                rows: 0,
                memory: HashMap::new(),
                disk: Vec::new(),
                file,
                garbage: 0,
            },
        );
        Self {
            memory: memory.clone(),
            id,
            width,
            targets: Vec::new(),
        }
    }

    /// Features per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Row ids the store holds: every id below it.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.memory.lock().stores[&self.id].rows
    }

    /// Whether the store holds no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The path of the store's file, if it has one.
    #[must_use]
    pub fn path(&self) -> Option<PathBuf> {
        let pool = self.memory.lock();
        pool.stores[&self.id]
            .file
            .as_ref()
            .map(|file| file.path.clone())
    }

    /// Bytes of the store's blocks in memory.
    #[must_use]
    pub fn memory_bytes(&self) -> u64 {
        let pool = self.memory.lock();
        let store = &pool.stores[&self.id];
        (store.memory.len() * store.rows_per_block * store.row_bytes) as u64
    }

    /// Bytes the store's file holds that no block points to any more.
    #[must_use]
    pub fn garbage(&self) -> u64 {
        self.memory.lock().stores[&self.id].garbage
    }

    /// Adds rows after the last, with ids from `len()` on, each `width` features.
    ///
    /// # Errors
    ///
    /// If a block must be read from or written to disk and cannot be.
    pub fn push(
        &mut self,
        features: &[f32],
        targets: &[u32],
        keys: &[u128],
    ) -> Result<(), RowsError> {
        let mut pool = self.memory.lock();
        let first = pool.stores[&self.id].rows;
        let count = u32::try_from(keys.len()).map_err(|_| RowsError::Unknown(RowId(u32::MAX)))?;
        pool.stores
            .get_mut(&self.id)
            .expect("a store's pool holds it")
            .rows = first + count;
        let rows: Vec<u32> = (first..first + count).collect();
        let written = pool
            .write(self.id, &rows, features, targets, Some(keys), self.width)
            .and_then(|()| pool.trim());
        if written.is_err() {
            // The rows did not go in, so their ids stay free, as the table's do.
            pool.stores
                .get_mut(&self.id)
                .expect("a store's pool holds it")
                .rows = first;
        } else {
            self.targets.extend_from_slice(targets);
        }
        written
    }

    /// Takes as many row ids as `rows`, every row zero, for rows put later in any order.
    ///
    /// # Errors
    ///
    /// If blocks must be written to disk and cannot be.
    pub fn reserve(&mut self, rows: u32) -> Result<(), RowsError> {
        let mut pool = self.memory.lock();
        let store = pool
            .stores
            .get_mut(&self.id)
            .expect("a store's pool holds it");
        store.rows = store.rows.max(rows);
        self.targets.resize(store.rows as usize, 0);
        pool.trim()
    }

    /// Replaces rows' features and targets, keeping their keys; or, with
    /// `keys`, their keys too.
    ///
    /// # Errors
    ///
    /// If a row is not stored, or a block cannot be read or written.
    pub fn set(
        &mut self,
        rows: &[RowId],
        features: &[f32],
        targets: &[u32],
        keys: Option<&[u128]>,
    ) -> Result<(), RowsError> {
        let mut pool = self.memory.lock();
        let held = pool.stores[&self.id].rows;
        if let Some(row) = rows.iter().find(|row| row.0 >= held) {
            return Err(RowsError::Unknown(*row));
        }
        let ids: Vec<u32> = rows.iter().map(|row| row.0).collect();
        pool.write(self.id, &ids, features, targets, keys, self.width)?;
        for (&row, &target) in ids.iter().zip(targets) {
            self.targets[row as usize] = target;
        }
        pool.trim()
    }

    /// A row's target's bits, read from memory, or None if it is not stored.
    #[must_use]
    pub fn target(&self, row: RowId) -> Option<u32> {
        self.targets.get(row.0 as usize).copied()
    }

    /// The features of `rows`, row-major, and their targets' bits, in order.
    ///
    /// # Errors
    ///
    /// If a row is not stored, or a block cannot be read.
    pub fn gather(&self, rows: &[RowId]) -> Result<(Vec<f32>, Vec<u32>), RowsError> {
        let mut pool = self.memory.lock();
        let mut features = Vec::with_capacity(rows.len() * self.width);
        let mut targets = Vec::with_capacity(rows.len());
        for &row in rows {
            let bytes = pool.block_of(self.id, row)?;
            let store = &pool.stores[&self.id];
            let start = (row.0 as usize % store.rows_per_block) * store.row_bytes;
            decode_row(
                &bytes[start..start + store.row_bytes],
                self.width,
                &mut features,
                &mut targets,
            );
        }
        pool.trim()?;
        Ok((features, targets))
    }

    /// One row's features.
    ///
    /// # Errors
    ///
    /// If the row is not stored, or its block cannot be read.
    pub fn features(&self, row: RowId) -> Result<Vec<f32>, RowsError> {
        Ok(self.gather(&[row])?.0)
    }

    /// Rows with their keys, in order.
    ///
    /// # Errors
    ///
    /// If a row is not stored, or a block cannot be read.
    pub fn rows(&self, rows: &[RowId]) -> Result<Vec<Row>, RowsError> {
        self.snapshot().rows(rows)
    }

    /// The rows as they are now, readable as the store changes.
    #[must_use]
    pub fn snapshot(&self) -> RowSnapshot {
        let pool = self.memory.lock();
        let store = &pool.stores[&self.id];
        RowSnapshot {
            width: self.width,
            row_bytes: store.row_bytes,
            rows_per_block: store.rows_per_block,
            rows: store.rows,
            memory: store
                .memory
                .iter()
                .map(|(&block, resident)| (block, Arc::clone(&resident.bytes)))
                .collect(),
            disk: store.disk.clone(),
            file: store.file.clone(),
        }
    }

    /// Empties the store, keeping its width and its file's path. The rows go
    /// on in a new file, so snapshots taken before read the old one, as they
    /// read any block as it was.
    ///
    /// # Errors
    ///
    /// If the new file cannot be made. The store is then unchanged.
    pub fn clear(&mut self) -> Result<(), RowsError> {
        let mut pool = self.memory.lock();
        let file = match &pool.stores[&self.id].file {
            Some(file) => Some(Arc::new(RowFile::create(
                &file.path,
                Arc::clone(&file.sealer),
            )?)),
            None => None,
        };
        let blocks: Vec<u32> = pool.stores[&self.id].memory.keys().copied().collect();
        for block in blocks {
            pool.forget(self.id, block);
        }
        let store = pool
            .stores
            .get_mut(&self.id)
            .expect("a store's pool holds it");
        store.rows = 0;
        self.targets.clear();
        store.disk.clear();
        store.garbage = 0;
        store.file = file;
        Ok(())
    }
}

impl Drop for RowStore {
    fn drop(&mut self) {
        let mut pool = self.memory.lock();
        let blocks: Vec<u32> = pool
            .stores
            .get(&self.id)
            .map(|store| store.memory.keys().copied().collect())
            .unwrap_or_default();
        for block in blocks {
            pool.forget(self.id, block);
        }
        if let Some(store) = pool.stores.remove(&self.id)
            && let Some(file) = store.file
        {
            let _ = std::fs::remove_file(&file.path);
        }
    }
}

impl RowSnapshot {
    /// Rows in the store when the snapshot was taken.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.rows
    }

    /// Whether the snapshot holds no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Rows with their keys, in order, as they were when the snapshot was taken.
    ///
    /// # Errors
    ///
    /// If a row was not stored, or a block cannot be read.
    pub fn rows(&self, rows: &[RowId]) -> Result<Vec<Row>, RowsError> {
        let mut read: HashMap<u32, Bytes> = HashMap::new();
        rows.iter()
            .map(|&row| {
                if row.0 >= self.rows {
                    return Err(RowsError::Unknown(row));
                }
                let block = u32::try_from(row.0 as usize / self.rows_per_block)
                    .expect("blocks number fewer than rows");
                let bytes = match read.get(&block) {
                    Some(bytes) => Arc::clone(bytes),
                    None => {
                        let bytes = self.block(block)?;
                        read.insert(block, Arc::clone(&bytes));
                        bytes
                    }
                };
                let start = (row.0 as usize % self.rows_per_block) * self.row_bytes;
                let slice = &bytes[start..start + self.row_bytes];
                let (mut features, mut targets) = (Vec::new(), Vec::new());
                decode_row(slice, self.width, &mut features, &mut targets);
                let key = u128::from_le_bytes(
                    slice[self.row_bytes - 16..]
                        .try_into()
                        .expect("sixteen bytes"),
                );
                Ok(Row {
                    features,
                    target: targets[0],
                    key,
                })
            })
            .collect()
    }

    fn block(&self, block: u32) -> Result<Bytes, RowsError> {
        if let Some(bytes) = self.memory.get(&block) {
            return Ok(Arc::clone(bytes));
        }
        match (self.disk.get(block as usize).copied().flatten(), &self.file) {
            (Some(location), Some(file)) => Ok(Arc::new(file.read(block, location)?)),
            _ => Ok(Arc::new(vec![0; self.rows_per_block * self.row_bytes])),
        }
    }
}

impl Pool {
    /// Writes rows into their blocks, each block once, as new blocks.
    fn write(
        &mut self,
        id: StoreId,
        rows: &[u32],
        features: &[f32],
        targets: &[u32],
        keys: Option<&[u128]>,
        width: usize,
    ) -> Result<(), RowsError> {
        let (row_bytes, rows_per_block) = {
            let store = &self.stores[&id];
            (store.row_bytes, store.rows_per_block)
        };
        // In row order, so each block is copied once however the rows came;
        // a row written twice keeps its later values.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&index| rows[index]);
        let mut at = 0;
        while at < order.len() {
            let first = order[at];
            let block = u32::try_from(rows[first] as usize / rows_per_block)
                .expect("blocks number fewer than rows");
            let mut bytes = (*self.block_of(id, RowId(rows[first]))?).clone();
            while at < order.len() && rows[order[at]] as usize / rows_per_block == block as usize {
                let index = order[at];
                let start = (rows[index] as usize % rows_per_block) * row_bytes;
                let row = &mut bytes[start..start + row_bytes];
                for (column, value) in features[index * width..(index + 1) * width]
                    .iter()
                    .enumerate()
                {
                    row[column * 4..column * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
                row[width * 4..width * 4 + 4].copy_from_slice(&targets[index].to_le_bytes());
                if let Some(keys) = keys {
                    row[width * 4 + 4..].copy_from_slice(&keys[index].to_le_bytes());
                }
                at += 1;
            }
            self.keep(id, block, Arc::new(bytes), true);
            // A large write spills as it goes rather than all at its end.
            self.trim()?;
        }
        Ok(())
    }

    /// A row's block, from memory, from disk, or new and zero.
    fn block_of(&mut self, id: StoreId, row: RowId) -> Result<Bytes, RowsError> {
        let store = &self.stores[&id];
        if row.0 >= store.rows {
            return Err(RowsError::Unknown(row));
        }
        let block = u32::try_from(row.0 as usize / store.rows_per_block)
            .expect("blocks number fewer than rows");
        if let Some(resident) = store.memory.get(&block) {
            let bytes = Arc::clone(&resident.bytes);
            self.touch(id, block);
            return Ok(bytes);
        }
        let bytes = match (
            store.disk.get(block as usize).copied().flatten(),
            &store.file,
        ) {
            (Some(location), Some(file)) => file.read(block, location)?,
            _ => vec![0; store.rows_per_block * store.row_bytes],
        };
        let bytes = Arc::new(bytes);
        self.keep(id, block, Arc::clone(&bytes), false);
        Ok(bytes)
    }

    /// Keeps a block in memory, as the most recently used.
    fn keep(&mut self, id: StoreId, block: u32, bytes: Bytes, dirty: bool) {
        self.forget(id, block);
        self.clock += 1;
        self.order.insert(self.clock, (id, block));
        self.bytes += bytes.len() as u64;
        let store = self.stores.get_mut(&id).expect("a store's pool holds it");
        let dirty = dirty || store.disk.get(block as usize).copied().flatten().is_none();
        store.memory.insert(
            block,
            Resident {
                bytes,
                dirty,
                used: self.clock,
            },
        );
    }

    fn touch(&mut self, id: StoreId, block: u32) {
        let store = self.stores.get_mut(&id).expect("a store's pool holds it");
        let resident = store
            .memory
            .get_mut(&block)
            .expect("touched blocks are resident");
        self.order.remove(&resident.used);
        self.clock += 1;
        resident.used = self.clock;
        self.order.insert(self.clock, (id, block));
    }

    /// Drops a block from memory without writing it.
    fn forget(&mut self, id: StoreId, block: u32) {
        let Some(store) = self.stores.get_mut(&id) else {
            return;
        };
        if let Some(resident) = store.memory.remove(&block) {
            self.order.remove(&resident.used);
            self.bytes -= resident.bytes.len() as u64;
        }
    }

    /// Writes out and drops the least recently used blocks of stores with
    /// files until the pool is within its budget.
    fn trim(&mut self) -> Result<(), RowsError> {
        let Some(budget) = self.budget else {
            return Ok(());
        };
        let mut skipped = Vec::new();
        while self.bytes > budget {
            let Some((&used, &(id, block))) = self.order.iter().next() else {
                break;
            };
            self.order.remove(&used);
            let store = self
                .stores
                .get_mut(&id)
                .expect("ordered blocks' stores exist");
            let Some(file) = store.file.clone() else {
                skipped.push((used, (id, block)));
                continue;
            };
            let resident = store
                .memory
                .remove(&block)
                .expect("ordered blocks are resident");
            if resident.dirty {
                let old = store.disk.get(block as usize).copied().flatten();
                let version = old.map_or(0, |old| old.version + 1);
                let location = match file.write(block, version, &resident.bytes) {
                    Ok(location) => location,
                    Err(error) => {
                        // The block stays in memory, so no row is lost.
                        store.memory.insert(block, resident);
                        self.order.insert(used, (id, block));
                        for (used, entry) in skipped {
                            self.order.insert(used, entry);
                        }
                        return Err(error);
                    }
                };
                if store.disk.len() <= block as usize {
                    store.disk.resize(block as usize + 1, None);
                }
                store.disk[block as usize] = Some(location);
                store.garbage += old.map_or(0, |old| u64::from(old.length));
            }
            self.bytes -= resident.bytes.len() as u64;
        }
        // Blocks of stores without files stay; they wait at the front again.
        for (used, entry) in skipped {
            self.order.insert(used, entry);
        }
        Ok(())
    }
}

fn decode_row(slice: &[u8], width: usize, features: &mut Vec<f32>, targets: &mut Vec<u32>) {
    features.extend(
        slice[..width * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes"))),
    );
    targets.push(u32::from_le_bytes(
        slice[width * 4..width * 4 + 4]
            .try_into()
            .expect("four bytes"),
    ));
}
