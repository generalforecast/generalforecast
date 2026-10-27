// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The write-ahead log: every pushed change, encrypted and synced before it applies.
//!
//! Each table has its own log, `<table>.log`: a plain header naming the format
//! and the key, then the table's definition, then its records. A record is a
//! frame, sealed with XChaCha20-Poly1305 under a random nonce, with its
//! sequence number, the table's name and creation id, and the previous
//! record's tag as associated data, so changed, swapped, spliced or foreign
//! records fail to open, including records from another table's log or from
//! an earlier table of the same name. A crash can only leave a partial final
//! record, which replay drops, and a failed append is cut back off; a log that
//! cannot cut it off takes no more writes until the server restarts.
//!
//! A log sealed under a previous key still opens, and the server rewrites it
//! under the current key at startup, entry for entry, keeping every sequence
//! number, so it stays the same log as its copies elsewhere.
//!
//! Beside each log, `<table>.head` holds a length the log was synced through,
//! sealed with the table's name. A log shorter than its head, or without one,
//! lost whole records from its end and does not open. A record is
//! acknowledged once it is synced, so a crash can only tear an append that
//! was never acknowledged. Writing the head costs three more syncs, so while
//! a table is busy it trails the synced length by up to `HEAD_LAG` records or
//! `HEAD_INTERVAL`, and it catches up once the table's writes go idle and
//! when the log closes. A record before the head's length that does not open
//! is refused as damage; one at or past it is taken for a torn append, and
//! opening cuts it off with everything after it. So a crash loses no
//! acknowledged record, and only damage to one synced within the head's lag
//! before a crash could be taken for a torn append.
//! Restoring an older copy of both files together still opens; replicas are
//! what catch that.
//!
//! Opening a log replays it one record at a time, so a long log never sits in
//! memory at once.
//!
//! Records that add rows carry the words they added to the vocabularies of
//! the table's categorical columns, so replay codes categories as the primary did.
//! Records that add observations to a table of series carry the keys of the
//! series they first name, as the client gave them, so a client that opens
//! the table later can name its series again.
//!
//! Replicas copy a table by reading its log from a sequence number: entry 0 is
//! the definition, and each later entry is one record. Records that change
//! rows may carry the client's write id, so a write sent twice applies once.
//!
//! Compacting rewrites a log as its definition and a snapshot of the live rows,
//! which keeps the snapshot's sequence number: later records continue the same
//! numbering, so replicas can follow a compaction like any other record.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};

use gf_core::rows::Sealer;
use serde::{Deserialize, Serialize};
use serde_json::json;
use zeroize::Zeroize;

use crate::name::TableName;
use crate::protocol::{Frame, FrameError};

mod record;

/// The definition's types where the crates that embed this server found
/// them before they had a module of their own.
pub use crate::definition::{CreationId, PastCapSpec, RouteBy, RouteKind, TableDefinition};
pub use record::{GivenKey, LogEntry, PastWrite, Record, Rows, SeriesKey, Term, Terms, decode};
use record::{definition_from, record_frame, record_from, terms_from};

const NONCE: usize = 24;
/// Numbers each staged compaction file, unique in the process, so a dropped
/// table's late compaction never shares a file with its successor's.
static COMPACTIONS: AtomicU64 = AtomicU64::new(0);
/// A record's length, then its complement: a torn write leaves the pair short,
/// while a damaged length breaks the pair instead of hiding later records.
const PREFIX: usize = 8;
/// The authentication tag that ends each sealed record.
const TAG: usize = 16;
/// The file's plain header: a magic, the format, and the id of the key.
const MAGIC: &[u8; 6] = b"GF_LOG";
const FORMAT: u8 = 4;
const HEADER: usize = MAGIC.len() + 2 + KEY_ID;
/// Bytes of a key's id.
const KEY_ID: usize = 8;

/// The key that seals log records: 32 bytes, wiped when dropped.
#[derive(Clone)]
pub struct EncryptionKey(Key);

impl EncryptionKey {
    /// A key from 64 hex digits.
    ///
    /// # Errors
    ///
    /// If the string is not 32 bytes of hex.
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        if hex.len() != 64 || !hex.is_ascii() {
            return Err("an encryption key is 64 hex digits".to_string());
        }
        let mut bytes: Vec<u8> = (0..32)
            .map(|index| u8::from_str_radix(&hex[2 * index..2 * index + 2], 16))
            .collect::<Result<_, _>>()
            .map_err(|_| "an encryption key is 64 hex digits".to_string())?;
        let key = Self(*Key::from_slice(&bytes));
        bytes.zeroize();
        Ok(key)
    }

    /// An id of the key that reveals nothing about it: the tag of an empty
    /// message sealed under it, as hex.
    #[must_use]
    pub fn id(&self) -> String {
        crate::setup::hex(&self.id_bytes())
    }

    /// The key's id, as a log's header names it.
    fn id_bytes(&self) -> [u8; KEY_ID] {
        let tag = XChaCha20Poly1305::new(&self.0)
            .encrypt(
                XNonce::from_slice(&[0; NONCE]),
                Payload {
                    msg: &[],
                    aad: b"gf log key",
                },
            )
            .expect("sealing an empty message cannot fail");
        tag[..KEY_ID]
            .try_into()
            .expect("a tag is longer than an id")
    }
}

impl Drop for EncryptionKey {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

/// The key records are sealed under, and earlier keys logs may still be sealed under.
#[derive(Clone)]
pub struct Keys {
    pub current: EncryptionKey,
    pub previous: Vec<EncryptionKey>,
}

impl Keys {
    /// Only one key, current and past.
    #[must_use]
    pub fn only(current: EncryptionKey) -> Self {
        Self {
            current,
            previous: Vec::new(),
        }
    }

    fn find(&self, id: &[u8; KEY_ID]) -> Option<(&EncryptionKey, bool)> {
        if self.current.id_bytes() == *id {
            return Some((&self.current, true));
        }
        self.previous
            .iter()
            .find(|key| key.id_bytes() == *id)
            .map(|key| (key, false))
    }
}

/// Why a log could not be read or written.
#[derive(Debug)]
pub enum LogError {
    Io(std::io::Error),
    /// The file is not a log of a format this server reads.
    Format(String),
    /// The log is sealed under a key the server was not given.
    Key {
        id: String,
    },
    /// The log is sealed under a previous key; compact it before appending.
    Rotate,
    /// A complete record failed to open: the wrong key, or changed bytes.
    Decrypt {
        offset: u64,
    },
    /// A record's length prefix was changed.
    Length {
        offset: u64,
    },
    /// An earlier write failed in a way that left the file unknown.
    Poisoned(String),
    /// The log is shorter than its head says it was synced to, or its head is
    /// missing or does not open.
    Head(String),
    /// A read from a sequence number past the log's end.
    Beyond {
        from: u64,
        length: u64,
    },
    Malformed(String),
    /// The log ends inside its definition: a creation a crash cut short,
    /// which was never acknowledged.
    Unfinished,
    /// The table names a strategy this server no longer serves; its log is
    /// whole and stays where it is.
    RetiredStrategy(String),
}

impl LogError {
    /// Whether the file itself is damaged, rather than unreadable for now, sealed
    /// under a key the server lacks, or short of records its head expects.
    #[must_use]
    pub fn damaged(&self) -> bool {
        matches!(
            self,
            Self::Format(_)
                | Self::Decrypt { .. }
                | Self::Length { .. }
                | Self::Malformed(_)
                | Self::Unfinished
        )
    }
}

impl fmt::Display for LogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Format(reason) => write!(formatter, "not a log this server reads: {reason}"),
            Self::Key { id } => write!(
                formatter,
                "the log is sealed under key {id}, which the server was not given; \
                 name it in previous_key_envs to rotate it"
            ),
            Self::Rotate => write!(
                formatter,
                "the log is sealed under a previous key; compact it before appending"
            ),
            Self::Decrypt { offset } => {
                write!(formatter, "the log record at byte {offset} does not open")
            }
            Self::Length { offset } => {
                write!(
                    formatter,
                    "the log record at byte {offset} has a damaged length"
                )
            }
            Self::Poisoned(reason) => write!(
                formatter,
                "the log takes no more writes after a failed one ({reason}); \
                 restart the server to recover it"
            ),
            Self::Head(reason) => write!(formatter, "the log lost records from its end: {reason}"),
            Self::Beyond { from, length } => write!(
                formatter,
                "the log holds {length} entries, so it cannot be read from entry {from}"
            ),
            Self::Malformed(message) => formatter.write_str(message),
            Self::Unfinished => write!(
                formatter,
                "the log ends inside its definition: a creation a crash cut short"
            ),
            Self::RetiredStrategy(kind) => write!(
                formatter,
                "the table uses the retired {kind} strategy; recreate it with \
                 an exact or retrieved one"
            ),
        }
    }
}

impl std::error::Error for LogError {}

impl From<std::io::Error> for LogError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<FrameError> for LogError {
    fn from(error: FrameError) -> Self {
        Self::Malformed(error.0)
    }
}

/// An open log, positioned to append.
pub struct Log {
    path: PathBuf,
    /// The table's name and creation id, which every record is sealed with.
    table: TableName,
    created: Option<CreationId>,
    file: File,
    /// Seals new records and the head, under the current key, whose id
    /// heads the head file.
    cipher: XChaCha20Poly1305,
    key_id: [u8; KEY_ID],
    /// Opens the records on disk: the current key, or the previous key they
    /// were sealed under until the log is compacted.
    reader: XChaCha20Poly1305,
    rotate: bool,
    /// Where each entry starts, then where the next one will.
    offsets: Vec<u64>,
    /// How many of `offsets` are synced to disk, and the tag then.
    durable: usize,
    durable_tag: [u8; TAG],
    /// The tag of the last entry, which the next is sealed with.
    last_tag: [u8; TAG],
    /// The sequence number of the first record; later than 1 after compaction.
    base: u64,
    terms: Terms,
    /// The length the head file holds, and when it was written.
    head: u64,
    head_at: Instant,
    /// Why the file may hold more or less than `offsets` says, after a failed write.
    poisoned: Option<String>,
}

/// Records a sync may run past the head, and how long the head may stay
/// behind a sync, before a sync writes the head too. Writing it costs three
/// more syncs, so a busy table writes it every so often rather than with
/// every commit; a record past the head that does not open is cut off as a
/// torn append, rather than refused, until the head catches up, as it does
/// when the table's writes go idle.
const HEAD_LAG: u64 = 64;
/// How long the head may stay behind a sync before a sync writes it too.
pub const HEAD_INTERVAL: Duration = Duration::from_secs(1);

fn definition_frame(
    definition: &TableDefinition,
    base: u64,
    terms: &Terms,
) -> Result<Frame, LogError> {
    let header =
        serde_json::to_value(definition).map_err(|error| LogError::Malformed(error.to_string()))?;
    Ok(Frame::header(
        json!({"record": "table", "definition": header, "base": base, "terms": terms}),
    ))
}

/// A log file's plain header, naming the key its records are sealed under.
fn file_header(key_id: &[u8; KEY_ID]) -> [u8; HEADER] {
    let mut header = [0; HEADER];
    header[..MAGIC.len()].copy_from_slice(MAGIC);
    header[MAGIC.len()] = FORMAT;
    header[MAGIC.len() + 2..].copy_from_slice(key_id);
    header
}

impl Log {
    /// Creates a table's log, starting with its definition and terms: term 0
    /// for a new table, or its source's for a replica.
    ///
    /// # Errors
    ///
    /// If the file exists or cannot be written.
    pub fn create(
        path: &Path,
        key: &EncryptionKey,
        definition: &TableDefinition,
        terms: Terms,
    ) -> Result<Self, LogError> {
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .read(true)
            .open(path)?;
        remove_staged_compactions(path);
        let cipher = XChaCha20Poly1305::new(&key.0);
        let mut log = Self {
            path: path.to_path_buf(),
            table: definition.name.clone(),
            created: definition.created.clone(),
            file,
            reader: cipher.clone(),
            cipher,
            key_id: key.id_bytes(),
            rotate: false,
            offsets: vec![HEADER as u64],
            durable: 1,
            durable_tag: [0; TAG],
            last_tag: [0; TAG],
            base: 1,
            terms,
            head: 1,
            head_at: Instant::now(),
            poisoned: None,
        };
        // The head goes first, so a crash never leaves the log without one.
        let written = log
            .write_head(1)
            .map_err(LogError::from)
            .and_then(|()| log.seal(&definition_frame(definition, 1, &log.terms)?, 0))
            .and_then(|bytes| {
                log.file.write_all(&file_header(&log.key_id))?;
                log.file.write_all(&bytes)?;
                log.file.sync_data()?;
                sync_directory(path)?;
                Ok(bytes)
            });
        match written {
            Ok(bytes) => {
                log.offsets.push(HEADER as u64 + bytes.len() as u64);
                log.last_tag = tag_of(&bytes);
                log.durable = 2;
                log.durable_tag = log.last_tag;
                Ok(log)
            }
            Err(error) => {
                let _ = remove(path);
                Err(error)
            }
        }
    }

    /// Opens a log to replay: its definition is read now, and its records come
    /// from the returned `Replay`, which `Replay::finish` turns into the log.
    ///
    /// # Errors
    ///
    /// If the file is not a log, is sealed under a key not in `keys`, has no
    /// head that opens, or the file cannot be read.
    pub fn open(path: &Path, keys: &Keys) -> Result<Replay, LogError> {
        Replay::start(path, keys, true)
    }

    /// Opens a log to read only, as a check does: a torn final record is
    /// reported by `Replay::finish` but left on disk.
    ///
    /// # Errors
    ///
    /// As `open`.
    pub fn open_read_only(path: &Path, keys: &Keys) -> Result<Replay, LogError> {
        Replay::start(path, keys, false)
    }

    /// Writes the head, sealed and synced: staged beside it, then renamed over
    /// it, so a crash leaves the old head or the new one whole.
    fn write_head(&self, length: u64) -> std::io::Result<()> {
        let mut nonce = [0u8; NONCE];
        getrandom::fill(&mut nonce).map_err(|error| std::io::Error::other(error.to_string()))?;
        let body = serde_json::to_vec(&Head {
            length,
            created: self.created.clone(),
        })
        .map_err(std::io::Error::other)?;
        let sealed = self
            .cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: &body,
                    aad: &head_associated(&self.table),
                },
            )
            .map_err(|_| std::io::Error::other("sealing the head failed"))?;
        let mut bytes = self.key_id.to_vec();
        bytes.extend_from_slice(&nonce);
        bytes.extend_from_slice(&sealed);
        let path = head_path(&self.path);
        let staged = self.path.with_extension("head-new");
        let mut file = File::create(&staged)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&staged, &path)?;
        sync_directory(&path)
    }

    /// Writes the head if records were synced since it was last written.
    ///
    /// # Errors
    ///
    /// If the head cannot be written; it then still holds an earlier length.
    pub fn save_head(&mut self) -> std::io::Result<()> {
        let synced = self.durable_length();
        if self.head < synced {
            self.write_head(synced)?;
            self.head_written(synced);
        }
        Ok(())
    }

    /// Writes the head through every synced record, so a log closed cleanly
    /// checks all its acknowledged records.
    ///
    /// # Errors
    ///
    /// If the head cannot be written.
    pub fn close(mut self) -> std::io::Result<()> {
        self.save_head()
    }

    /// Notes a head written at `length`.
    fn head_written(&mut self, length: u64) {
        self.head = length;
        self.head_at = Instant::now();
    }

    /// The length the log had at its last sync.
    fn durable_length(&self) -> u64 {
        self.base + self.durable as u64 - 2
    }

    /// The sequence number the next record gets: entries so far, counting the
    /// definition and any records compaction folded into a snapshot.
    #[must_use]
    pub fn length(&self) -> u64 {
        self.base + self.offsets.len() as u64 - 2
    }

    /// The sequence number of the first record on disk; later than 1 after compaction.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Bytes the log takes on disk.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.offsets.last().copied().unwrap_or_default()
    }

    /// Whether the log is sealed under a previous key and awaits compaction.
    #[must_use]
    pub fn needs_rotation(&self) -> bool {
        self.rotate
    }

    /// The table's creation id, as its definition records it.
    #[must_use]
    pub fn created(&self) -> Option<&CreationId> {
        self.created.as_ref()
    }

    /// The creation id entry `sequence` is sealed with: none for the
    /// definition, which names the creation itself.
    fn sealed_creation(&self, sequence: u64) -> &str {
        sealed_creation(self.created.as_ref(), sequence)
    }

    /// The entry between `start` and `end` in the file, as sealed, with its
    /// length prefix.
    fn sealed_at(&self, start: u64, end: u64) -> std::io::Result<Vec<u8>> {
        let mut sealed = vec![0u8; usize::try_from(end - start).expect("an entry fits in memory")];
        self.file.read_exact_at(&mut sealed, start)?;
        Ok(sealed)
    }

    /// Opens entry `sequence`, sealed after `previous` and found at
    /// `offset`, under the key the log's records are sealed with; `sealed`
    /// is its nonce and ciphertext.
    fn open_sealed(
        &self,
        sealed: &[u8],
        sequence: u64,
        previous: &[u8; TAG],
        offset: u64,
    ) -> Result<Vec<u8>, LogError> {
        open_entry(
            &self.reader,
            &self.table,
            self.sealed_creation(sequence),
            sealed,
            sequence,
            previous,
            offset,
        )
    }

    /// Who takes the table's writes, as its records last said.
    #[must_use]
    pub fn term(&self) -> &Term {
        self.terms.latest()
    }

    /// Every term the table has had, oldest first.
    #[must_use]
    pub fn terms(&self) -> &Terms {
        &self.terms
    }

    /// Takes later terms from another copy's definition, as when that copy's
    /// log was compacted past the records that began them; the snapshot that
    /// follows makes them durable.
    pub fn adopt(&mut self, terms: Terms) {
        if terms.latest().term > self.term().term {
            self.terms = terms;
        }
    }

    /// The sequence number of the entry at `index` in the file.
    fn sequence(&self, index: usize) -> u64 {
        if index == 0 {
            0
        } else {
            self.base + index as u64 - 1
        }
    }

    /// Entries with sequence numbers from `from` on, decrypted, for a replica
    /// to apply: at most `limit` of them, and past the first, no more than
    /// `bytes` of them sealed, so a page of snapshot records never sits in
    /// memory at once. Records compaction folded away are not there: a
    /// replica that far behind starts from the definition, for its term, and
    /// the snapshot.
    ///
    /// # Errors
    ///
    /// `Beyond` if `from` is past the log's synced end, or if the file cannot
    /// be read or an entry does not open.
    pub fn read(
        &self,
        from: u64,
        limit: usize,
        bytes: usize,
    ) -> Result<Vec<(u64, Vec<u8>)>, LogError> {
        let length = self.durable_length();
        if from > length {
            return Err(LogError::Beyond { from, length });
        }
        let first = if from < self.base {
            0
        } else {
            usize::try_from(from - self.base + 1).unwrap_or(usize::MAX)
        };
        let mut entries = Vec::new();
        let mut previous = [0; TAG];
        if first > 0 {
            previous = tag_of(&self.sealed_at(self.offsets[first - 1], self.offsets[first])?);
        }
        let synced = &self.offsets[..self.durable];
        let mut read = 0usize;
        for (index, window) in synced.windows(2).enumerate().skip(first) {
            let size = usize::try_from(window[1] - window[0]).expect("fits");
            if entries.len() >= limit || (!entries.is_empty() && read + size > bytes) {
                break;
            }
            read += size;
            let sequence = self.sequence(index);
            let sealed = self.sealed_at(window[0], window[1])?;
            let plain = self.open_sealed(&sealed[PREFIX..], sequence, &previous, window[0])?;
            previous = tag_of(&sealed);
            entries.push((sequence, plain));
        }
        Ok(entries)
    }

    /// Refuses writes once a failed one left the file unknown, or the log
    /// awaits rewriting under the current key.
    /// Why the log takes no more writes until the server restarts, if it does not.
    #[must_use]
    pub fn poisoned(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    fn writable(&self) -> Result<(), LogError> {
        if self.rotate {
            return Err(LogError::Rotate);
        }
        match &self.poisoned {
            Some(reason) => Err(LogError::Poisoned(reason.clone())),
            None => Ok(()),
        }
    }

    /// Appends a record and syncs it to disk before returning.
    ///
    /// A failed write or sync is cut back off the file, so the log holds
    /// exactly its entries before it; if that fails too, the log takes no more
    /// writes.
    ///
    /// # Errors
    ///
    /// If the write or sync fails, or an earlier one left the log unusable.
    pub fn append(&mut self, record: &Record) -> Result<(), LogError> {
        self.stage(record)?;
        self.sync()
    }

    /// Writes a record without syncing it, for `sync` to make durable with
    /// others; until then it is not an entry a reader or replica may see.
    ///
    /// # Errors
    ///
    /// If the record cannot be sealed or written, which cuts every staged
    /// record back off, so a failed stage leaves none; or if an earlier
    /// failure left the log unusable.
    pub fn stage(&mut self, record: &Record) -> Result<(), LogError> {
        self.writable()?;
        let sequence = self.length();
        let bytes = match self.seal(&record_frame(record), sequence) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.discard_staged();
                return Err(error);
            }
        };
        let start = self.offsets.last().copied().unwrap_or_default();
        if let Err(error) = self.file.write_all(&bytes) {
            self.cut_back();
            return Err(error.into());
        }
        self.offsets.push(start + bytes.len() as u64);
        self.last_tag = tag_of(&bytes);
        if let Record::Term { term, primary } = record {
            self.terms.begin(Term {
                term: *term,
                primary: primary.clone(),
                start: sequence,
            });
        }
        Ok(())
    }

    /// Syncs the staged records to disk; a failure cuts them back off, so
    /// they were never written. The head follows every `HEAD_LAG` records or
    /// `HEAD_INTERVAL`, and the shard saves it when its writes go idle.
    ///
    /// # Errors
    ///
    /// If the sync fails, or an earlier failure left the log unusable.
    pub fn sync(&mut self) -> Result<(), LogError> {
        self.writable()?;
        if self.offsets.len() == self.durable {
            return Ok(());
        }
        if let Err(error) = self.file.sync_data() {
            self.cut_back();
            return Err(error.into());
        }
        self.durable = self.offsets.len();
        self.durable_tag = self.last_tag;
        let length = self.length();
        if length - self.head >= HEAD_LAG || self.head_at.elapsed() >= HEAD_INTERVAL {
            // The records are durable; a head that failed to write still
            // holds an earlier length, which only weakens the check until
            // the next save.
            match self.write_head(length) {
                Ok(()) => self.head_written(length),
                Err(error) => {
                    tracing::error!(log = %self.path.display(), "writing the head: {error}");
                }
            }
        }
        Ok(())
    }

    /// Whether records are staged but not yet synced.
    #[must_use]
    pub fn has_staged(&self) -> bool {
        self.offsets.len() > self.durable
    }

    /// Cuts staged records back off, for a caller that will not sync them.
    pub fn discard_staged(&mut self) {
        if self.has_staged() {
            self.cut_back();
        }
    }

    /// Cuts the file back to its last synced entry, and the log's state with it.
    fn cut_back(&mut self) {
        let start = self.offsets[self.durable - 1];
        self.offsets.truncate(self.durable);
        self.last_tag = self.durable_tag;
        let length = self.length();
        self.terms.cut(length);
        let cut = self
            .file
            .set_len(start)
            .and_then(|()| self.file.sync_data());
        if let Err(cut) = cut {
            self.poisoned = Some(format!("cutting a failed write off: {cut}"));
        }
    }

    /// Rewrites the log as its definition and `records`, a snapshot and any
    /// records that follow it, from sequence number `at` on; later records
    /// continue from there. The new log is sealed under the current key.
    ///
    /// The new log is written beside the old one, synced, and renamed over it,
    /// so a crash leaves one or the other whole.
    ///
    /// # Errors
    ///
    /// If the files cannot be written or renamed.
    pub fn compact(
        &mut self,
        definition: &TableDefinition,
        at: u64,
        records: &[Record],
    ) -> Result<(), LogError> {
        let mut compactor = self.begin_compaction(definition, at)?;
        for record in records {
            compactor.append(record)?;
        }
        self.install_compaction(compactor.finish()?)
    }

    /// Rewrites the log under the current key, entry for entry, keeping every
    /// sequence number, so it stays the same log as its copies elsewhere,
    /// where a compaction would add an entry this copy alone holds.
    ///
    /// # Errors
    ///
    /// If records are staged, an entry does not open, or the files cannot be
    /// written or renamed.
    pub fn rotate(&mut self) -> Result<(), LogError> {
        if !self.rotate {
            return Ok(());
        }
        if let Some(reason) = &self.poisoned {
            return Err(LogError::Poisoned(reason.clone()));
        }
        if self.has_staged() {
            return Err(LogError::Malformed(
                "a log with records staged cannot be rewritten".to_string(),
            ));
        }
        let staged = self.path.with_extension("rotating");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&staged)?;
        file.write_all(&file_header(&self.key_id))?;
        let mut offsets = vec![HEADER as u64];
        let (mut read, mut written) = ([0; TAG], [0; TAG]);
        for (index, window) in self.offsets.windows(2).enumerate() {
            let sequence = self.sequence(index);
            let sealed = self.sealed_at(window[0], window[1])?;
            let plain = self.open_sealed(&sealed[PREFIX..], sequence, &read, window[0])?;
            read = tag_of(&sealed);
            let bytes = self.seal_plain_after(&plain, sequence, &written)?;
            file.write_all(&bytes)?;
            written = tag_of(&bytes);
            offsets.push(offsets.last().copied().unwrap_or_default() + bytes.len() as u64);
        }
        file.sync_all()?;
        let base = self.base;
        self.install(&staged, offsets, written, base)
    }

    /// Renames a rewritten file over the log and takes up its entries; the
    /// head is written again, under the current key.
    fn install(
        &mut self,
        staged: &Path,
        offsets: Vec<u64>,
        last_tag: [u8; TAG],
        base: u64,
    ) -> Result<(), LogError> {
        std::fs::rename(staged, &self.path)?;
        // The old file is gone now, so a failure past here leaves the log unknown.
        let reopened = sync_directory(&self.path)
            .and_then(|()| OpenOptions::new().read(true).append(true).open(&self.path));
        match reopened {
            Ok(file) => {
                self.file = file;
                self.offsets = offsets;
                self.durable = self.offsets.len();
                self.last_tag = last_tag;
                self.durable_tag = last_tag;
                self.base = base;
                self.reader = self.cipher.clone();
                self.rotate = false;
                // A head that failed to write still holds an earlier length,
                // which only weakens the check until the next sync saves it.
                let length = self.durable_length();
                match self.write_head(length) {
                    Ok(()) => self.head_written(length),
                    Err(error) => {
                        tracing::error!(log = %self.path.display(), "writing the head: {error}");
                    }
                }
                Ok(())
            }
            Err(error) => {
                self.poisoned = Some(format!("reopening a rewritten log: {error}"));
                Err(error.into())
            }
        }
    }

    /// A frame sealed under its sequence number after the last entry, with its length prefix.
    fn seal(&self, frame: &Frame, sequence: u64) -> Result<Vec<u8>, LogError> {
        self.seal_after(frame, sequence, &self.last_tag)
    }

    fn seal_after(
        &self,
        frame: &Frame,
        sequence: u64,
        previous: &[u8; TAG],
    ) -> Result<Vec<u8>, LogError> {
        self.seal_plain_after(&frame.to_bytes(), sequence, previous)
    }

    /// An entry's plaintext sealed under its sequence number after `previous`,
    /// with its length prefix.
    fn seal_plain_after(
        &self,
        plain: &[u8],
        sequence: u64,
        previous: &[u8; TAG],
    ) -> Result<Vec<u8>, LogError> {
        seal_entry(
            &self.cipher,
            &self.table,
            self.sealed_creation(sequence),
            plain,
            sequence,
            previous,
        )
    }

    /// Starts rewriting the log as a compaction at `at`, its length now or, for
    /// a snapshot a replica skips ahead to, later: the
    /// definition sealed into a staged file, for `Compactor` to add the
    /// compaction's records to off the shard, and `install_compaction` to put
    /// in the log's place.
    ///
    /// # Errors
    ///
    /// If the log takes no writes, has records staged, or the staged file
    /// cannot be made.
    pub fn begin_compaction(
        &self,
        definition: &TableDefinition,
        at: u64,
    ) -> Result<Compactor, LogError> {
        if let Some(reason) = &self.poisoned {
            return Err(LogError::Poisoned(reason.clone()));
        }
        if self.has_staged() {
            return Err(LogError::Malformed(
                "a log with records staged cannot be compacted".to_string(),
            ));
        }
        let number = COMPACTIONS.fetch_add(1, Ordering::Relaxed);
        let staged = self.path.with_extension(format!("compacting.{number}"));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staged)?;
        file.write_all(&file_header(&self.key_id))?;
        let mut compactor = Compactor {
            staged,
            number,
            file,
            cipher: self.cipher.clone(),
            table: self.table.clone(),
            created: self.created.clone(),
            offsets: vec![HEADER as u64],
            previous: [0; TAG],
            next: at,
            at,
            begun: self.length(),
        };
        let frame = definition_frame(definition, at, &self.terms)?;
        compactor.seal(&frame, 0)?;
        Ok(compactor)
    }

    /// Puts a finished compaction in the log's place.
    ///
    /// # Errors
    ///
    /// If the log took records since the compaction began, which the
    /// compaction does not hold, or the file cannot be put in place.
    pub fn install_compaction(&mut self, compacted: Compacted) -> Result<(), LogError> {
        if self.length() != compacted.begun || self.has_staged() {
            let _ = std::fs::remove_file(&compacted.staged);
            return Err(LogError::Malformed(
                "the log took records while it was compacted".to_string(),
            ));
        }
        self.install(
            &compacted.staged,
            compacted.offsets,
            compacted.previous,
            compacted.at,
        )
    }
}

/// A log being opened, replayed one record at a time: iterate it for the
/// records, then `finish` it for the log.
pub struct Replay {
    entries: Entries,
    definition: TableDefinition,
    writable: bool,
}

/// Reads a log's entries in order, checking each as it goes.
struct Entries {
    log: Log,
    reader: BufReader<File>,
    /// Bytes of the file when it was opened.
    size: u64,
    /// The length the head says was synced: entries before it must open.
    synced: u64,
    /// Where a torn final append begins, if one was found.
    torn: Option<u64>,
    done: bool,
}

impl Replay {
    fn start(path: &Path, keys: &Keys, writable: bool) -> Result<Self, LogError> {
        let table = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| TableName::new(stem).ok())
            .ok_or_else(|| LogError::Malformed("a log is named <table>.log".to_string()))?;
        let file = if writable {
            OpenOptions::new().read(true).append(true).open(path)?
        } else {
            File::open(path)?
        };
        let size = file.metadata()?.len();
        let mut reader = BufReader::new(file.try_clone()?);
        if size < HEADER as u64 {
            return Err(LogError::Format("missing header".to_string()));
        }
        let mut header = [0; HEADER];
        reader.read_exact(&mut header)?;
        if &header[..MAGIC.len()] != MAGIC {
            return Err(LogError::Format("missing header".to_string()));
        }
        if header[MAGIC.len()] != FORMAT {
            return Err(LogError::Format(format!(
                "format {} rather than {FORMAT}",
                header[MAGIC.len()]
            )));
        }
        let id: [u8; KEY_ID] = header[MAGIC.len() + 2..].try_into().expect("sized");
        let (key, current) = keys.find(&id).ok_or_else(|| LogError::Key {
            id: crate::setup::hex(&id),
        })?;
        let head = read_head(path, &table, keys)?;
        let mut entries = Entries {
            log: Log {
                path: path.to_path_buf(),
                table,
                created: None,
                file,
                cipher: XChaCha20Poly1305::new(&keys.current.0),
                key_id: keys.current.id_bytes(),
                reader: XChaCha20Poly1305::new(&key.0),
                rotate: !current,
                offsets: vec![HEADER as u64],
                durable: 1,
                durable_tag: [0; TAG],
                last_tag: [0; TAG],
                base: 1,
                terms: Terms::default(),
                head: head.length,
                head_at: Instant::now(),
                poisoned: None,
            },
            reader,
            size,
            synced: head.length.max(1),
            torn: None,
            done: false,
        };
        // A creation writes its head through the definition alone, then the
        // definition, and is acknowledged once that is synced; so a log whose
        // head names no more ends inside it only when the creation never
        // finished. One whose head names more lost acknowledged entries.
        let unfinished = head.length <= 1;
        let frame = match entries.next_entry() {
            Ok(Some(plain)) => Frame::from_bytes(&plain)?,
            Ok(None) | Err(LogError::Head(_)) if unfinished => return Err(LogError::Unfinished),
            Ok(None) => return Err(LogError::Malformed("the log has no definition".to_string())),
            Err(error) => return Err(error),
        };
        let definition = definition_from(&frame)?;
        if definition.name != entries.log.table {
            return Err(LogError::Malformed(format!(
                "{} holds table {:?}",
                path.display(),
                definition.name
            )));
        }
        if head.created != definition.created {
            return Err(LogError::Head(
                "its head belongs to another creation of the table".to_string(),
            ));
        }
        entries.log.base = frame.header["base"].as_u64().unwrap_or(1);
        entries.log.created.clone_from(&definition.created);
        entries.log.terms = terms_from(&frame)?;
        if writable {
            remove_staged_compactions(path);
        }
        Ok(Self {
            entries,
            definition,
            writable,
        })
    }

    /// The table's definition, as the log's first entry records it.
    #[must_use]
    pub fn definition(&self) -> &TableDefinition {
        &self.definition
    }

    /// The log's length so far: the sequence number of the next record.
    #[must_use]
    pub fn length(&self) -> u64 {
        self.entries.log.length()
    }

    /// Whether the log is sealed under a previous key and awaits compaction.
    #[must_use]
    pub fn needs_rotation(&self) -> bool {
        self.entries.log.rotate
    }

    /// Reads the rest of the log, cuts off a torn final append, and returns
    /// the log, positioned to append.
    ///
    /// # Errors
    ///
    /// If a record before the head's length does not open, the log ends
    /// before it, or a torn append cannot be cut off.
    pub fn finish(mut self) -> Result<Log, LogError> {
        for record in self.by_ref() {
            record?;
        }
        let Entries {
            mut log,
            synced,
            torn,
            ..
        } = self.entries;
        if log.length() < synced {
            return Err(LogError::Head(format!(
                "it ends before entry {}, but it was synced through entry {}",
                log.length(),
                synced - 1
            )));
        }
        if let Some(offset) = torn {
            tracing::warn!(
                log = %log.path.display(),
                offset,
                "cutting off a record past the head that does not open, and everything after it: an append a crash tore before it was synced, or damage to a record synced since the head was last written"
            );
            if self.writable {
                log.file.set_len(offset)?;
                log.file.sync_data()?;
            }
        }
        log.durable = log.offsets.len();
        log.durable_tag = log.last_tag;
        Ok(log)
    }
}

impl Iterator for Replay {
    type Item = Result<Record, LogError>;

    fn next(&mut self) -> Option<Self::Item> {
        let entries = &mut self.entries;
        let plain = match entries.next_entry() {
            Ok(Some(plain)) => plain,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
        };
        let record = Frame::from_bytes(&plain)
            .map_err(LogError::from)
            .and_then(|frame| record_from(&frame));
        match &record {
            Ok(Record::Term { term, primary }) => {
                let start = entries.log.length() - 1;
                entries.log.terms.begin(Term {
                    term: *term,
                    primary: primary.clone(),
                    start,
                });
            }
            Ok(_) => {}
            Err(_) => entries.done = true,
        }
        Some(record)
    }
}

impl Entries {
    /// The next entry's plaintext, or None at the end. Damage at or past the
    /// head's length is a torn append, which ends the log.
    fn next_entry(&mut self) -> Result<Option<Vec<u8>>, LogError> {
        if self.done {
            return Ok(None);
        }
        let offset = *self.log.offsets.last().expect("the header at least");
        let sequence = self.log.sequence(self.log.offsets.len() - 1);
        match self.read_sealed(offset, sequence) {
            Ok(Some(opened)) => {
                self.log.offsets.push(opened.end);
                self.log.last_tag = opened.tag;
                Ok(Some(opened.plain))
            }
            Ok(None) => {
                self.done = true;
                Ok(None)
            }
            Err(error) => {
                self.done = true;
                if sequence >= self.synced && !matches!(error, LogError::Io(_)) {
                    self.torn = Some(offset);
                    return Ok(None);
                }
                Err(error)
            }
        }
    }

    /// The entry at `offset`, opened; None at the end of the file.
    fn read_sealed(&mut self, offset: u64, sequence: u64) -> Result<Option<Opened>, LogError> {
        let remaining = self.size - offset;
        if remaining == 0 {
            return Ok(None);
        }
        let synced = self.synced;
        let short = || {
            LogError::Head(format!(
                "it ends inside entry {sequence}, but it was synced through entry {}",
                synced - 1
            ))
        };
        if remaining < PREFIX as u64 {
            return Err(short());
        }
        let mut prefix = [0; PREFIX];
        self.reader.read_exact(&mut prefix)?;
        let length = u32::from_le_bytes(prefix[..4].try_into().expect("four bytes"));
        let complement = u32::from_le_bytes(prefix[4..].try_into().expect("four bytes"));
        if complement != !length || (length as usize) < NONCE + TAG {
            return Err(LogError::Length { offset });
        }
        let length = u64::from(length);
        if remaining - (PREFIX as u64) < length {
            return Err(short());
        }
        let mut sealed = vec![0; usize::try_from(length).expect("a u32 fits")];
        self.reader.read_exact(&mut sealed)?;
        let plain = self
            .log
            .open_sealed(&sealed, sequence, &self.log.last_tag, offset)?;
        Ok(Some(Opened {
            plain,
            tag: tag_of(&sealed),
            end: offset + PREFIX as u64 + length,
        }))
    }
}

/// An entry read and opened: its plaintext, its tag, and where it ends.
struct Opened {
    plain: Vec<u8>,
    tag: [u8; TAG],
    end: u64,
}

/// Reads a log's head, which must open under one of `keys`.
fn read_head(path: &Path, table: &str, keys: &Keys) -> Result<Head, LogError> {
    let bytes = match std::fs::read(head_path(path)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(LogError::Head("its head file is missing".to_string()));
        }
        Err(error) => return Err(error.into()),
    };
    let unreadable = || LogError::Head("its head file does not open".to_string());
    if bytes.len() < KEY_ID + NONCE {
        return Err(unreadable());
    }
    let (id, rest) = bytes.split_at(KEY_ID);
    let (nonce, ciphertext) = rest.split_at(NONCE);
    let (key, _) = keys
        .find(&id.try_into().expect("sized"))
        .ok_or_else(unreadable)?;
    let plain = XChaCha20Poly1305::new(&key.0)
        .decrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad: &head_associated(table),
            },
        )
        .map_err(|_| unreadable())?;
    serde_json::from_slice(&plain).map_err(|_| unreadable())
}

/// What a log's head file holds.
#[derive(Serialize, Deserialize)]
struct Head {
    length: u64,
    #[serde(default, deserialize_with = "CreationId::optional")]
    created: Option<CreationId>,
}

fn head_path(log: &Path) -> PathBuf {
    log.with_extension("head")
}

fn remove_staged_compactions(log: &Path) {
    let Some(stem) = log.file_stem().and_then(|stem| stem.to_str()) else {
        return;
    };
    let plain = format!("{stem}.compacting");
    let numbered = format!("{stem}.compacting.");
    let directory = log.parent().unwrap_or_else(|| Path::new("."));
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(directory = %directory.display(), "listing staged compactions: {error}");
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(directory = %directory.display(), "reading staged compaction entry: {error}");
                continue;
            }
        };
        let matches = entry.file_name().to_str().is_some_and(|name| {
            name == plain
                || name.strip_prefix(&numbered).is_some_and(|number| {
                    !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit())
                })
        });
        if !matches {
            continue;
        }
        if let Err(error) = std::fs::remove_file(entry.path())
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(staged = %entry.path().display(), "removing staged compaction: {error}");
        }
    }
}

/// What a head is sealed with besides its bytes.
fn head_associated(table: &str) -> Vec<u8> {
    let mut data = b"gf log head".to_vec();
    data.push(FORMAT);
    data.extend_from_slice(table.as_bytes());
    data
}

/// Removes a log and its head, and syncs their directory.
///
/// # Errors
///
/// If the log cannot be removed or the directory synced.
pub fn remove(path: &Path) -> std::io::Result<()> {
    std::fs::remove_file(path)?;
    match std::fs::remove_file(head_path(path)) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    sync_directory(path)
}

/// Renames a log and its head out of the way with `suffix` after their
/// extensions, so restarts leave them alone; returns where the log went.
///
/// # Errors
///
/// If the log cannot be renamed or the directory synced.
pub fn set_aside(path: &Path, suffix: &str) -> std::io::Result<PathBuf> {
    // A log set aside before under the same suffix stays, for whoever looks.
    let suffix = (0..)
        .map(|copy| match copy {
            0 => suffix.to_string(),
            copy => format!("{suffix}.{copy}"),
        })
        .find(|suffix| !path.with_extension(format!("log.{suffix}")).exists())
        .expect("some suffix is free");
    let aside = path.with_extension(format!("log.{suffix}"));
    std::fs::rename(path, &aside)?;
    let head = head_path(path);
    match std::fs::rename(&head, head.with_extension(format!("head.{suffix}"))) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    sync_directory(path)?;
    Ok(aside)
}

/// Sets a damaged log aside as `<table>.log.unreadable`, so the other tables
/// still start and a replica can be copied back, keeping it for inspection.
pub fn set_aside_damaged(path: &Path, error: &LogError) {
    match set_aside(path, "unreadable") {
        Ok(aside) => tracing::error!(
            log = %path.display(),
            aside = %aside.display(),
            "setting aside a log that does not open: {error}"
        ),
        Err(rename) => tracing::error!(
            log = %path.display(),
            "skipping a log that does not open ({error}) and cannot be moved: {rename}"
        ),
    }
}

/// The authentication tag ending a sealed entry, with or without its length prefix.
fn tag_of(sealed: &[u8]) -> [u8; TAG] {
    sealed[sealed.len() - TAG..].try_into().expect("sized")
}

/// What a record is sealed with besides its bytes: the format, its sequence
/// number, the previous entry's tag, and the table's creation id and name.
fn associated(table: &str, created: &str, sequence: u64, previous: &[u8; TAG]) -> Vec<u8> {
    let mut data = vec![FORMAT];
    data.extend_from_slice(&sequence.to_le_bytes());
    data.extend_from_slice(previous);
    data.extend_from_slice(created.as_bytes());
    data.push(0);
    data.extend_from_slice(table.as_bytes());
    data
}

/// Syncs the directory holding `path`, so a file created, renamed or removed there stays so.
///
/// # Errors
///
/// If the directory cannot be opened or synced.
pub fn sync_directory(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(directory) => File::open(directory)?.sync_all(),
        None => Ok(()),
    }
}

/// A log being rewritten as a compaction, off the shard that owns the log:
/// records sealed in order after the definition into a staged file.
pub struct Compactor {
    staged: PathBuf,
    number: u64,
    file: File,
    cipher: XChaCha20Poly1305,
    table: TableName,
    created: Option<CreationId>,
    offsets: Vec<u64>,
    previous: [u8; TAG],
    next: u64,
    at: u64,
    /// The log's length when the compaction began, which it must still have.
    begun: u64,
}

/// A compaction written and synced, for `Log::install_compaction`.
#[derive(Debug)]
pub struct Compacted {
    staged: PathBuf,
    offsets: Vec<u64>,
    previous: [u8; TAG],
    at: u64,
    begun: u64,
}

impl Compacted {
    /// Removes the staged file, for a compaction that will not be installed.
    pub fn discard(self) {
        let _ = std::fs::remove_file(&self.staged);
    }
}

impl Compactor {
    /// The process-wide number in this compaction's staged file name.
    #[must_use]
    pub fn number(&self) -> u64 {
        self.number
    }

    /// Seals the compaction's next record into the staged file.
    ///
    /// # Errors
    ///
    /// If the record is too large or cannot be written.
    pub fn append(&mut self, record: &Record) -> Result<(), LogError> {
        let sequence = self.next;
        self.seal(&record_frame(record), sequence)?;
        self.next += 1;
        Ok(())
    }

    /// Syncs the staged file, ready to be put in the log's place.
    ///
    /// # Errors
    ///
    /// If the file cannot be synced.
    pub fn finish(mut self) -> Result<Compacted, LogError> {
        self.file.sync_all()?;
        Ok(Compacted {
            staged: std::mem::take(&mut self.staged),
            offsets: std::mem::take(&mut self.offsets),
            previous: self.previous,
            at: self.at,
            begun: self.begun,
        })
    }

    fn seal(&mut self, frame: &Frame, sequence: u64) -> Result<(), LogError> {
        let bytes = seal_entry(
            &self.cipher,
            &self.table,
            sealed_creation(self.created.as_ref(), sequence),
            &frame.to_bytes(),
            sequence,
            &self.previous,
        )?;
        self.file.write_all(&bytes)?;
        self.previous = tag_of(&bytes);
        let end = self.offsets.last().copied().unwrap_or_default() + bytes.len() as u64;
        self.offsets.push(end);
        Ok(())
    }
}

impl Drop for Compactor {
    fn drop(&mut self) {
        if !self.staged.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.staged);
        }
    }
}

impl fmt::Debug for Compactor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Compactor({})", self.staged.display())
    }
}

/// The creation id entry `sequence` of a table created as `created` is
/// sealed with: none for the definition, which names the creation itself.
fn sealed_creation(created: Option<&CreationId>, sequence: u64) -> &str {
    match created {
        Some(created) if sequence != 0 => created.as_str(),
        _ => "",
    }
}

/// Opens an entry `seal_entry` sealed, given its nonce and ciphertext,
/// found at `offset` in its file.
fn open_entry(
    cipher: &XChaCha20Poly1305,
    table: &str,
    created: &str,
    sealed: &[u8],
    sequence: u64,
    previous: &[u8; TAG],
    offset: u64,
) -> Result<Vec<u8>, LogError> {
    let (nonce, ciphertext) = sealed.split_at(NONCE);
    cipher
        .decrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad: &associated(table, created, sequence, previous),
            },
        )
        .map_err(|_| LogError::Decrypt { offset })
}

/// An entry sealed for a log: its length twice, then the nonce and ciphertext.
fn seal_entry(
    cipher: &XChaCha20Poly1305,
    table: &str,
    created: &str,
    plain: &[u8],
    sequence: u64,
    previous: &[u8; TAG],
) -> Result<Vec<u8>, LogError> {
    let mut nonce = [0u8; NONCE];
    getrandom::fill(&mut nonce).map_err(|error| LogError::Malformed(error.to_string()))?;
    let ciphertext = cipher
        .encrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: plain,
                aad: &associated(table, created, sequence, previous),
            },
        )
        .map_err(|_| LogError::Malformed("encryption failed".to_string()))?;
    let length = u32::try_from(NONCE + ciphertext.len())
        .map_err(|_| LogError::Malformed("record too large".to_string()))?;
    let mut bytes = Vec::with_capacity(PREFIX + length as usize);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&(!length).to_le_bytes());
    bytes.extend_from_slice(&nonce);
    bytes.extend_from_slice(&ciphertext);
    Ok(bytes)
}

/// Seals a table's row blocks under the log key, as its log is sealed: each
/// bound to the table, its creation, and the block's place and version, so a
/// block from another table, an earlier table of the same name, another place
/// or an older write does not open.
pub struct RowSealer {
    cipher: XChaCha20Poly1305,
    binding: Vec<u8>,
}

impl RowSealer {
    /// A sealer for the rows of `table` as created at `creation`.
    #[must_use]
    pub fn new(key: &EncryptionKey, table: &TableName, creation: &str) -> Self {
        let mut binding = b"gf rows\0".to_vec();
        binding.extend_from_slice(table.as_str().as_bytes());
        binding.push(0);
        binding.extend_from_slice(creation.as_bytes());
        binding.push(0);
        Self {
            cipher: XChaCha20Poly1305::new(&key.0),
            binding,
        }
    }

    fn associated(&self, block: u32, version: u32) -> Vec<u8> {
        let mut associated = self.binding.clone();
        associated.extend_from_slice(&block.to_le_bytes());
        associated.extend_from_slice(&version.to_le_bytes());
        associated
    }
}

impl fmt::Debug for RowSealer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RowSealer")
    }
}

impl Sealer for RowSealer {
    fn seal(&self, block: u32, version: u32, plain: &[u8]) -> Vec<u8> {
        let mut nonce = [0u8; NONCE];
        getrandom::fill(&mut nonce).expect("the system has randomness");
        let sealed = self
            .cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: plain,
                    aad: &self.associated(block, version),
                },
            )
            .expect("sealing a block in memory does not fail");
        [nonce.as_slice(), &sealed].concat()
    }

    fn open(&self, block: u32, version: u32, sealed: &[u8]) -> Option<Vec<u8>> {
        let (nonce, sealed) = sealed.split_at_checked(NONCE)?;
        self.cipher
            .decrypt(
                XNonce::from_slice(nonce),
                Payload {
                    msg: sealed,
                    aad: &self.associated(block, version),
                },
            )
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::TimeKind;
    use crate::protocol::{StrategySpec, SummarySpec};
    use crate::setup;
    use gf_core::columns::{Column, ColumnKind};
    use serde_json::Value;

    /// Writes a log whose table names `strategy`, as an earlier server could.
    fn log_with_strategy(path: &Path, key: &EncryptionKey, strategy: Value) {
        let definition = TableDefinition {
            name: TableName::new("payments").unwrap(),
            model: "general-forecast/tabicl-v2-classifier".to_string(),
            columns: vec![Column {
                name: "amount".to_string(),
                kind: ColumnKind::Numeric,
            }],
            classes: 2,
            strategy: StrategySpec::Exact,
            page_size: 64,
            time_bucket: None,
            refit_after: None,
            refit_shift: None,
            fit_rows: None,
            past_cap: PastCapSpec::Warn,
            regions: None,
            series: None,
            refit_cutoffs: "each".to_string(),
            retrieval_categoricals: "codes".to_string(),
            scale_rows: None,
            summary: SummarySpec::Table,
            created: CreationId::parse("0123456789abcdef"),
            times: TimeKind::Integer,
            route_by: None,
            roles: None,
            durability: Default::default(),
        };
        let log = Log::create(path, key, &definition, Terms::default()).unwrap();
        let mut header = serde_json::to_value(&definition).unwrap();
        header["strategy"] = strategy;
        let frame = Frame::header(
            json!({"record": "table", "definition": header, "base": 1, "terms": Terms::default()}),
        );
        let sealed = log.seal_after(&frame, 0, &[0; TAG]).unwrap();
        drop(log);
        std::fs::write(
            path,
            [file_header(&key.id_bytes()).as_slice(), &sealed].concat(),
        )
        .unwrap();
    }

    #[test]
    fn a_table_with_a_retired_strategy_is_not_served_and_keeps_its_log() {
        let directory = std::env::temp_dir().join(format!("gf-retired-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("payments.log");
        let key = EncryptionKey::from_hex(&"07".repeat(32)).unwrap();
        log_with_strategy(
            &path,
            &key,
            json!({"kind": "pages", "budget": 4096, "stitch": "joint"}),
        );
        let log = std::fs::read(&path).unwrap();
        let head = std::fs::read(head_path(&path)).unwrap();
        let keys = Keys::only(key);

        let error = Log::open(&path, &keys).err().unwrap();
        assert!(matches!(&error, LogError::RetiredStrategy(kind) if kind == "pages"));
        assert!(!error.damaged());
        assert!(error.to_string().contains("retired pages strategy"));
        // The server starts without the table and leaves its log as it was, so
        // the name stays taken.
        assert!(setup::open_log(&path, &keys).unwrap().is_none());
        assert_eq!(std::fs::read(&path).unwrap(), log);
        assert_eq!(std::fs::read(head_path(&path)).unwrap(), head);
        assert!(!directory.join("payments.log.unreadable").exists());
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
