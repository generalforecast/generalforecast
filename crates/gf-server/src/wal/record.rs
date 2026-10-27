// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! What a log's entries hold: its definition and terms, and the records
//! that change its rows, as the frames they are sealed as.

use gf_core::columns::Words;
use gf_core::node::{FittedColumns, RowLayout, Scaling, TableLayout, Targets};
use gf_core::pages::{Bucket, CellLayout, IndexLayout, PageId, PageLayout};
use gf_core::table::RowId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::LogError;
use crate::definition::TableDefinition;
use crate::protocol::{
    Frame, StrategySpec, f32_bytes, f32_values, f64_bytes, f64_values, i64_bytes, i64_values,
    key_bytes, key_values, target_bytes, targets_from, u32_bytes, u32_values, u64_bytes,
    u64_values,
};

/// A series' key as a client gave it: user data a table of series keeps,
/// logs and replays, so a client that opens the table later can name its
/// series again. The text is the key's own form: a whole number's decimal
/// digits, which may take 128 bits, more than JSON numbers carry exactly; a
/// string as it is; bytes in lowercase hex; a UUID hyphenated.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum GivenKey {
    Integer(String),
    String(String),
    Bytes(String),
    Uuid(String),
}

impl GivenKey {
    /// The 64-bit digest a table knows the series by, computed as
    /// `generalforecast.keys` computes it: the low 64 bits of the key's 128,
    /// which are an integer's two's complement, a UUID's bits, and the
    /// little-endian 16-byte BLAKE2b digest of a string's UTF-8 or of bytes.
    /// Checks the text is of its kind's form on the way, so every client
    /// reads it back.
    ///
    /// # Errors
    ///
    /// Names what is wrong with the text.
    pub fn digest(&self) -> Result<u64, String> {
        use blake2::digest::consts::U16;
        use blake2::{Blake2b, Digest};

        let hashed = |bytes: &[u8]| {
            let digest = Blake2b::<U16>::digest(bytes);
            u64::from_le_bytes(digest[..8].try_into().expect("16 bytes"))
        };
        match self {
            Self::Integer(text) => text
                .parse::<i128>()
                .map(|number| number as u128)
                .or_else(|_| text.parse::<u128>())
                .map(|number| number as u64)
                .map_err(|_| format!("integer key {text:?} is not a whole number of 128 bits")),
            Self::String(text) => Ok(hashed(text.as_bytes())),
            Self::Bytes(text) => unhex(text)
                .ok_or_else(|| format!("bytes key {text:?} is not lowercase hex"))
                .map(|bytes| hashed(&bytes)),
            Self::Uuid(text) => {
                let hyphenated = text.len() == 36
                    && [8, 13, 18, 23]
                        .iter()
                        .all(|&index| text.as_bytes()[index] == b'-');
                let plain: String = text
                    .bytes()
                    .enumerate()
                    .filter(|&(index, _)| !matches!(index, 8 | 13 | 18 | 23))
                    .map(|(_, byte)| byte as char)
                    .collect();
                let bytes = hyphenated
                    .then(|| unhex(&plain))
                    .flatten()
                    .ok_or_else(|| format!("UUID key {text:?} is not hyphenated lowercase hex"))?;
                // A UUID's bytes are big-endian: its last 8 hold the low half.
                Ok(u64::from_be_bytes(bytes[8..].try_into().expect("16 bytes")))
            }
        }
    }
}

/// Bytes from lowercase hex of even length, or None for other text.
fn unhex(text: &str) -> Option<Vec<u8>> {
    let digit = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    if !text.len().is_multiple_of(2) {
        return None;
    }
    text.as_bytes()
        .chunks(2)
        .map(|pair| Some((digit(pair[0])? << 4) | digit(pair[1])?))
        .collect()
}

/// A series of a table of series: the 64-bit digest the table knows it by,
/// which clients pack into its observations' row keys, and its key as the
/// client gave it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesKey {
    pub digest: u64,
    pub key: GivenKey,
}

impl SeriesKey {
    /// Checks the key reads back and is the series' own: it digests to the
    /// digest the table knows the series by.
    ///
    /// # Errors
    ///
    /// Names what is wrong with the key.
    pub fn check(&self) -> Result<(), String> {
        let digest = self.key.digest()?;
        if digest != self.digest {
            return Err(format!(
                "series key {:?} digests to {digest:#x}, not to series {:#x}",
                self.key, self.digest
            ));
        }
        Ok(())
    }
}

/// Rows a record carries: keys, row-major features, targets, and times if any.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub keys: Vec<u128>,
    pub features: Vec<f32>,
    pub targets: Targets,
    /// Each row's time if it has one; None when the rows were written
    /// without times. Only a snapshot mixes rows with and without times.
    pub times: Option<Vec<Option<i64>>>,
    /// The client's id for the write, if it sent one, so a retry applies once.
    pub id: Option<String>,
    /// Words the record adds to columns' vocabularies, in code order; a
    /// snapshot's are every word.
    pub words: Vec<Words>,
    /// The series the record's rows first name in a table of series, each
    /// key as the client gave it by the digest the table knows it by; a
    /// snapshot's are every series the table holds.
    pub series: Vec<SeriesKey>,
    /// A snapshot's: the writes the table last applied, by client id, so a
    /// retry still applies once after the log is compacted.
    pub writes: Vec<PastWrite>,
    /// A compaction's: each row's id and page, rows in page order, and the
    /// pages that begin among them, so a copy rebuilt from the log keeps them.
    pub layout: Option<RowLayout>,
    /// A compaction's first record's: what the table holds besides its rows.
    pub table: Option<Box<TableLayout>>,
}

/// A write a table applied, as a snapshot remembers it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PastWrite {
    pub id: String,
    pub rows: u64,
    pub log_length: u64,
}

/// A change to a table.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Insert(Rows),
    Update(Rows),
    /// Rows replacing those of keys the table holds, and added otherwise.
    Upsert(Rows),
    Delete {
        keys: Vec<u128>,
        /// The client's id for the write, if it sent one.
        id: Option<String>,
    },
    Refit,
    /// Every row deleted, as when the source table is truncated.
    Truncate,
    /// From here on the table chooses contexts with this strategy.
    Strategy(StrategySpec),
    /// Every live row at this point: the table is exactly these rows.
    Snapshot(Rows),
    /// From here on the table's writes go to the node named `primary`, under
    /// fencing epoch `term`.
    Term {
        term: u64,
        primary: String,
    },
}

impl Record {
    /// The client's id for the write, if the record carries one.
    #[must_use]
    pub fn write_id(&self) -> Option<&str> {
        match self {
            Self::Insert(rows) | Self::Update(rows) | Self::Upsert(rows) => rows.id.as_deref(),
            Self::Delete { id, .. } => id.as_deref(),
            Self::Refit
            | Self::Truncate
            | Self::Strategy(_)
            | Self::Snapshot(_)
            | Self::Term { .. } => None,
        }
    }

    /// The keys of the rows the record changes; none for a refit or term.
    #[must_use]
    pub fn keys(&self) -> &[u128] {
        match self {
            Self::Insert(rows) | Self::Update(rows) | Self::Upsert(rows) | Self::Snapshot(rows) => {
                &rows.keys
            }
            Self::Delete { keys, .. } => keys,
            Self::Refit | Self::Truncate | Self::Strategy(_) | Self::Term { .. } => &[],
        }
    }

    /// The words the record adds to vocabularies, if it carries rows.
    #[must_use]
    pub fn words(&self) -> &[Words] {
        match self {
            Self::Insert(rows) | Self::Update(rows) | Self::Upsert(rows) | Self::Snapshot(rows) => {
                &rows.words
            }
            Self::Delete { .. }
            | Self::Refit
            | Self::Truncate
            | Self::Strategy(_)
            | Self::Term { .. } => &[],
        }
    }

    /// The series the record names, if it carries rows of a table of series.
    #[must_use]
    pub fn series(&self) -> &[SeriesKey] {
        match self {
            Self::Insert(rows) | Self::Update(rows) | Self::Upsert(rows) | Self::Snapshot(rows) => {
                &rows.series
            }
            Self::Delete { .. }
            | Self::Refit
            | Self::Truncate
            | Self::Strategy(_)
            | Self::Term { .. } => &[],
        }
    }
}

/// Who takes a table's writes: the epoch a promotion began, the node it named,
/// and the sequence number of the record that began it. Term 0, the table's
/// first, has no named primary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    pub term: u64,
    pub primary: String,
    pub start: u64,
}

/// Every term a table has had, oldest first. Never empty: a table begins at term 0.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<Term>", into = "Vec<Term>")]
pub struct Terms(Vec<Term>);

impl Default for Terms {
    /// A new table's: term 0 alone.
    fn default() -> Self {
        Self(vec![Term::default()])
    }
}

impl TryFrom<Vec<Term>> for Terms {
    type Error = String;

    fn try_from(terms: Vec<Term>) -> Result<Self, Self::Error> {
        if terms.is_empty() {
            return Err("a table has term 0 at least".to_string());
        }
        Ok(Self(terms))
    }
}

impl From<Terms> for Vec<Term> {
    fn from(terms: Terms) -> Self {
        terms.0
    }
}

impl std::ops::Deref for Terms {
    type Target = [Term];

    fn deref(&self) -> &[Term] {
        &self.0
    }
}

impl Terms {
    /// The latest term: who takes the table's writes now.
    #[must_use]
    pub fn latest(&self) -> &Term {
        self.0.last().expect("terms are never empty")
    }

    /// Begins a later term.
    pub fn begin(&mut self, term: Term) {
        self.0.push(term);
    }

    /// Forgets the terms begun at `length` or later, keeping term 0.
    pub(super) fn cut(&mut self, length: u64) {
        self.0.retain(|term| term.start < length || term.term == 0);
        if self.0.is_empty() {
            self.0.push(Term::default());
        }
    }
}

/// One entry of a log, as replicas receive it.
#[derive(Clone, Debug, PartialEq)]
pub enum LogEntry {
    /// The table's definition, and its terms when the log was last compacted.
    Definition(TableDefinition, Terms),
    Record(Record),
}

/// Decodes an entry read from a log.
///
/// # Errors
///
/// If the bytes are not a definition or a record.
pub fn decode(bytes: &[u8]) -> Result<LogEntry, LogError> {
    let frame = Frame::from_bytes(bytes)?;
    if frame.header.get("record").and_then(Value::as_str) == Some("table") {
        return Ok(LogEntry::Definition(
            definition_from(&frame)?,
            terms_from(&frame)?,
        ));
    }
    Ok(LogEntry::Record(record_from(&frame)?))
}

/// Strategies tables could once name, whose stitched contexts this server no
/// longer builds.
const RETIRED_STRATEGIES: [&str; 2] = ["pages", "global_local"];

pub(super) fn definition_from(frame: &Frame) -> Result<TableDefinition, LogError> {
    let definition = &frame.header["definition"];
    if let Some(kind) = definition.pointer("/strategy/kind").and_then(Value::as_str)
        && RETIRED_STRATEGIES.contains(&kind)
    {
        return Err(LogError::RetiredStrategy(kind.to_string()));
    }
    serde_json::from_value(definition.clone())
        .map_err(|error| LogError::Malformed(error.to_string()))
}

pub(super) fn terms_from(frame: &Frame) -> Result<Terms, LogError> {
    match frame.header.get("terms") {
        Some(terms) => serde_json::from_value(terms.clone())
            .map_err(|error| LogError::Malformed(error.to_string())),
        None => Ok(Terms::default()),
    }
}

pub(super) fn record_frame(record: &Record) -> Frame {
    let mut frame = Frame::header(Value::Null);
    frame.header = match record {
        Record::Insert(rows)
        | Record::Update(rows)
        | Record::Upsert(rows)
        | Record::Snapshot(rows) => {
            let kind = match record {
                Record::Insert(_) => "insert",
                Record::Update(_) => "update",
                Record::Upsert(_) => "upsert",
                _ => "snapshot",
            };
            let (target_buffer, target_type) = target_bytes(&rows.targets);
            let mut header = json!({
                "record": kind,
                "keys": frame.push(key_bytes(&rows.keys)),
                "features": frame.push(f32_bytes(&rows.features)),
                "targets": frame.push(target_buffer),
                "target_type": target_type,
            });
            if let Some(times) = &rows.times {
                let values: Vec<i64> = times.iter().map(|time| time.unwrap_or(0)).collect();
                header["times"] = json!(frame.push(i64_bytes(&values)));
                // Which rows have a time, when some do not.
                if times.iter().any(Option::is_none) {
                    let timed = times.iter().map(|time| u8::from(time.is_some())).collect();
                    header["timed"] = json!(frame.push(timed));
                }
            }
            if let Some(id) = &rows.id {
                header["id"] = json!(id);
            }
            if !rows.words.is_empty() {
                header["words"] = json!(rows.words);
            }
            if !rows.series.is_empty() {
                header["series"] = json!(rows.series);
            }
            if !rows.writes.is_empty() {
                header["writes"] = json!(rows.writes);
            }
            if let Some(layout) = &rows.layout {
                header["layout"] = row_layout_header(&mut frame, layout);
            }
            if let Some(table) = &rows.table {
                header["table_layout"] = table_layout_header(&mut frame, table);
            }
            header
        }
        Record::Delete { keys, id } => {
            let mut header = json!({"record": "delete", "keys": frame.push(key_bytes(keys))});
            if let Some(id) = id {
                header["id"] = json!(id);
            }
            header
        }
        Record::Refit => json!({"record": "refit"}),
        Record::Truncate => json!({"record": "truncate"}),
        Record::Strategy(strategy) => json!({"record": "strategy", "strategy": strategy}),
        Record::Term { term, primary } => {
            json!({"record": "term", "term": term, "primary": primary})
        }
    };
    frame
}

/// A compaction record's row layout, as frame buffers.
fn row_layout_header(frame: &mut Frame, layout: &RowLayout) -> Value {
    let ids: Vec<u32> = layout.ids.iter().map(|row| row.0).collect();
    let pages: Vec<u32> = layout.pages.iter().map(|page| page.0).collect();
    let new: Vec<u32> = layout.new_pages.iter().map(|page| page.id.0).collect();
    let buckets: Vec<u64> = layout.new_pages.iter().map(|page| page.bucket.0).collect();
    let sums: Vec<f64> = layout
        .new_pages
        .iter()
        .flat_map(|page| page.sum.iter().copied())
        .collect();
    let radii: Vec<f64> = layout.new_pages.iter().map(|page| page.radius).collect();
    json!({
        "ids": frame.push(u32_bytes(&ids)),
        "pages": frame.push(u32_bytes(&pages)),
        "new_pages": frame.push(u32_bytes(&new)),
        "buckets": frame.push(u64_bytes(&buckets)),
        "sums": frame.push(f64_bytes(&sums)),
        "radii": frame.push(f64_bytes(&radii)),
    })
}

/// A compaction record's row layout, from frame buffers.
fn row_layout_from(frame: &Frame, header: &Value) -> Result<RowLayout, LogError> {
    let new = u32_values(frame.buffer(header, "new_pages")?)?;
    let buckets = u64_values(frame.buffer(header, "buckets")?)?;
    let radii = f64_values(frame.buffer(header, "radii")?)?;
    let sums = f64_values(frame.buffer(header, "sums")?)?;
    if buckets.len() != new.len() || radii.len() != new.len() {
        return Err(malformed(
            "a layout's pages and their buckets or radii differ in number",
        ));
    }
    let width = per(sums.len(), new.len(), "a layout's page sums")?;
    let new_pages = new
        .iter()
        .zip(&buckets)
        .zip(&radii)
        .enumerate()
        .map(|(index, ((&id, &bucket), &radius))| PageLayout {
            id: PageId(id),
            bucket: Bucket(bucket),
            sum: sums[index * width..(index + 1) * width].to_vec(),
            radius,
        })
        .collect();
    Ok(RowLayout {
        ids: u32_values(frame.buffer(header, "ids")?)?
            .into_iter()
            .map(RowId)
            .collect(),
        pages: u32_values(frame.buffer(header, "pages")?)?
            .into_iter()
            .map(PageId)
            .collect(),
        new_pages,
    })
}

/// A compaction's table layout, as frame buffers.
fn table_layout_header(frame: &mut Frame, table: &TableLayout) -> Value {
    let cells = &table.index.cells;
    let centers: Vec<f64> = cells
        .iter()
        .flat_map(|cell| cell.center.iter().copied())
        .collect();
    let radii: Vec<f64> = cells.iter().map(|cell| cell.radius).collect();
    let reaches: Vec<f64> = cells.iter().map(|cell| cell.reach).collect();
    let groups: Vec<u32> = cells.iter().map(|cell| cell.pages.len() as u32).collect();
    let buckets: Vec<u64> = cells
        .iter()
        .flat_map(|cell| cell.pages.iter().map(|(bucket, _)| bucket.0))
        .collect();
    let sizes: Vec<u32> = cells
        .iter()
        .flat_map(|cell| cell.pages.iter().map(|(_, pages)| pages.len() as u32))
        .collect();
    let pages: Vec<u32> = cells
        .iter()
        .flat_map(|cell| {
            cell.pages
                .iter()
                .flat_map(|(_, pages)| pages.iter().map(|page| page.0))
        })
        .collect();
    let scaling = table.scaling.as_ref().map_or(Value::Null, |scaling| {
        let lookups: Vec<Value> = scaling
            .lookups
            .iter()
            .map(|(codes, vectors)| {
                json!({
                    "codes": frame.push(f32_bytes(codes)),
                    "vectors": frame.push(f32_bytes(vectors)),
                })
            })
            .collect();
        json!({
            "center": frame.push(f32_bytes(&scaling.center)),
            "scale": frame.push(f32_bytes(&scaling.scale)),
            "lookups": lookups,
            "rows": scaling.rows,
        })
    });
    let fitted = table.fitted.as_ref().map_or(Value::Null, |fitted| {
        json!({
            "rows": fitted.rows,
            "means": frame.push(f64_bytes(&fitted.means)),
            "deviations": frame.push(f64_bytes(&fitted.deviations)),
        })
    });
    json!({
        "rows": table.rows,
        "ids": table.ids,
        "next_page": table.index.next_page,
        "changed": table.changed,
        "reach": frame.push(f64_bytes(&[table.index.reach, table.index.built_reach])),
        "cells": {
            "centers": frame.push(f64_bytes(&centers)),
            "radii": frame.push(f64_bytes(&radii)),
            "reaches": frame.push(f64_bytes(&reaches)),
            "groups": frame.push(u32_bytes(&groups)),
            "buckets": frame.push(u64_bytes(&buckets)),
            "sizes": frame.push(u32_bytes(&sizes)),
            "pages": frame.push(u32_bytes(&pages)),
        },
        "scaling": scaling,
        "column_sums": frame.push(f64_bytes(&table.column_sums)),
        "column_counts": frame.push(u64_bytes(&table.column_counts)),
        "fitted": fitted,
        "strategy": table.strategy.map(StrategySpec::of),
    })
}

/// A compaction's table layout, from frame buffers.
fn table_layout_from(frame: &Frame, header: &Value) -> Result<TableLayout, LogError> {
    let number = |field: &str| {
        header
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed(&format!("a table layout needs {field}")))
    };
    let reach = f64_values(frame.buffer(header, "reach")?)?;
    let [reach, built_reach] = reach[..] else {
        return Err(malformed("a table layout's reach is two numbers"));
    };
    let cells = header.get("cells").unwrap_or(&Value::Null);
    let radii = f64_values(frame.buffer(cells, "radii")?)?;
    let reaches = f64_values(frame.buffer(cells, "reaches")?)?;
    let groups = u32_values(frame.buffer(cells, "groups")?)?;
    let buckets = u64_values(frame.buffer(cells, "buckets")?)?;
    let sizes = u32_values(frame.buffer(cells, "sizes")?)?;
    let pages = u32_values(frame.buffer(cells, "pages")?)?;
    let centers = f64_values(frame.buffer(cells, "centers")?)?;
    let count = radii.len();
    if reaches.len() != count
        || groups.len() != count
        || sizes.len() != buckets.len()
        || groups.iter().map(|&groups| groups as usize).sum::<usize>() != buckets.len()
        || sizes.iter().map(|&size| size as usize).sum::<usize>() != pages.len()
    {
        return Err(malformed("a table layout's cells do not add up"));
    }
    let width = per(centers.len(), count, "a table layout's cell centers")?;
    let (mut group, mut page) = (0, 0);
    let mut laid = Vec::with_capacity(count);
    for index in 0..count {
        let mut members = Vec::new();
        for _ in 0..groups[index] {
            let size = sizes[group] as usize;
            let ids = pages[page..page + size]
                .iter()
                .map(|&id| PageId(id))
                .collect();
            members.push((Bucket(buckets[group]), ids));
            group += 1;
            page += size;
        }
        laid.push(CellLayout {
            center: centers[index * width..(index + 1) * width].to_vec(),
            radius: radii[index],
            reach: reaches[index],
            pages: members,
        });
    }
    let scaling = match header.get("scaling") {
        Some(scaling) if !scaling.is_null() => Some(Scaling {
            center: f32_values(frame.buffer(scaling, "center")?)?,
            scale: f32_values(frame.buffer(scaling, "scale")?)?,
            lookups: scaling
                .get("lookups")
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .map(|lookup| {
                    Ok((
                        f32_values(frame.buffer(lookup, "codes")?)?,
                        f32_values(frame.buffer(lookup, "vectors")?)?,
                    ))
                })
                .collect::<Result<_, LogError>>()?,
            rows: scaling
                .get("rows")
                .and_then(Value::as_u64)
                .ok_or_else(|| malformed("a scaling needs its rows"))?,
        }),
        _ => None,
    };
    let fitted = match header.get("fitted") {
        Some(fitted) if !fitted.is_null() => Some(FittedColumns {
            rows: fitted
                .get("rows")
                .and_then(Value::as_u64)
                .ok_or_else(|| malformed("fitted columns need their rows"))?,
            means: f64_values(frame.buffer(fitted, "means")?)?,
            deviations: f64_values(frame.buffer(fitted, "deviations")?)?,
        }),
        _ => None,
    };
    Ok(TableLayout {
        rows: number("rows")?,
        ids: u32::try_from(number("ids")?).map_err(|_| malformed("too many row ids"))?,
        index: IndexLayout {
            cells: laid,
            next_page: u32::try_from(number("next_page")?)
                .map_err(|_| malformed("too many pages"))?,
            reach,
            built_reach,
        },
        scaling,
        column_sums: f64_values(frame.buffer(header, "column_sums")?)?,
        column_counts: u64_values(frame.buffer(header, "column_counts")?)?,
        fitted,
        changed: number("changed")?,
        strategy: match header.get("strategy") {
            Some(strategy) if !strategy.is_null() => Some(
                serde_json::from_value::<StrategySpec>(strategy.clone())
                    .map_err(|_| malformed("a layout's strategy is malformed"))?
                    .strategy(),
            ),
            _ => None,
        },
    })
}

/// How many values each of `count` items has, where `total` values hold them all.
fn per(total: usize, count: usize, what: &str) -> Result<usize, LogError> {
    match count {
        0 if total == 0 => Ok(0),
        0 => Err(malformed(&format!("{what} hold values for nothing"))),
        count if total.is_multiple_of(count) => Ok(total / count),
        _ => Err(malformed(&format!("{what} do not divide evenly"))),
    }
}

fn malformed(why: &str) -> LogError {
    LogError::Malformed(why.to_string())
}

/// A record's row times, each None where its `timed` mask says the row has none.
fn times_from(frame: &Frame, header: &Value) -> Result<Vec<Option<i64>>, LogError> {
    let values = i64_values(frame.buffer(header, "times")?)?;
    let Some(_) = header.get("timed") else {
        return Ok(values.into_iter().map(Some).collect());
    };
    let timed = frame.buffer(header, "timed")?;
    if timed.len() != values.len() || timed.iter().any(|&flag| flag > 1) {
        return Err(LogError::Malformed(
            "a record's time mask does not match its times".to_string(),
        ));
    }
    Ok(values
        .into_iter()
        .zip(timed)
        .map(|(time, &flag)| (flag == 1).then_some(time))
        .collect())
}

pub(super) fn record_from(frame: &Frame) -> Result<Record, LogError> {
    let header = &frame.header;
    let kind = header
        .get("record")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let id = header.get("id").and_then(Value::as_str).map(str::to_string);
    let keys = || -> Result<Vec<u128>, LogError> { Ok(key_values(frame.buffer(header, "keys")?)?) };
    let rows = || -> Result<Rows, LogError> {
        let target_type = header
            .get("target_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let times = match header.get("times") {
            Some(_) => Some(times_from(frame, header)?),
            None => None,
        };
        Ok(Rows {
            keys: keys()?,
            features: f32_values(frame.buffer(header, "features")?)?,
            targets: targets_from(frame.buffer(header, "targets")?, target_type)?,
            times,
            id: id.clone(),
            words: match header.get("words") {
                Some(words) => serde_json::from_value(words.clone())
                    .map_err(|error| LogError::Malformed(error.to_string()))?,
                None => Vec::new(),
            },
            series: match header.get("series") {
                Some(series) => serde_json::from_value(series.clone())
                    .map_err(|error| LogError::Malformed(error.to_string()))?,
                None => Vec::new(),
            },
            writes: match header.get("writes") {
                Some(writes) => serde_json::from_value(writes.clone())
                    .map_err(|error| LogError::Malformed(error.to_string()))?,
                None => Vec::new(),
            },
            layout: header
                .get("layout")
                .map(|layout| row_layout_from(frame, layout))
                .transpose()?,
            table: header
                .get("table_layout")
                .map(|table| table_layout_from(frame, table).map(Box::new))
                .transpose()?,
        })
    };
    Ok(match kind {
        "insert" => Record::Insert(rows()?),
        "update" => Record::Update(rows()?),
        "upsert" => Record::Upsert(rows()?),
        "snapshot" => Record::Snapshot(rows()?),
        "delete" => Record::Delete { keys: keys()?, id },
        "refit" => Record::Refit,
        "truncate" => Record::Truncate,
        "strategy" => Record::Strategy(
            header
                .get("strategy")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| LogError::Malformed(format!("a strategy record: {error}")))?
                .ok_or_else(|| {
                    LogError::Malformed("a strategy record needs a strategy".to_string())
                })?,
        ),
        "term" => Record::Term {
            term: header
                .get("term")
                .and_then(Value::as_u64)
                .ok_or_else(|| LogError::Malformed("a term record needs a term".to_string()))?,
            primary: header
                .get("primary")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        _ => return Err(LogError::Malformed(format!("unknown record {kind:?}"))),
    })
}
