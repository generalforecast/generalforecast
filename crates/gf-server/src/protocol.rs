// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Frames: a JSON header plus raw little-endian arrays.
//!
//! The server and its workers exchange frames over a Unix socket, and log
//! records are frames too. Arrays travel as bytes the receiver can view in
//! place, so a step's features are never parsed number by number.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use gf_core::cache::{CacheKey, Movement, SlotId};
use gf_core::forecasting::{Forecast, Grid, SeriesUnit};
use gf_core::node::{QueryBatch, Step, SummaryScope, Targets, Unit};
use gf_core::planner::{Core, CoreKind, Recency, Strategy};
use gf_core::table::{RowId, TableId};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The most bytes one frame may claim, so a damaged length never exhausts
/// memory. Large frames from the server travel through shared files instead.
const MAX_FRAME_BYTES: u64 = 1 << 33;

/// The version of the frames the server and its workers exchange, which a
/// worker reports when it connects: both sides must come from one generalforecast.
pub const PROTOCOL_VERSION: u32 = 13;

/// The most buffers one frame may claim.
const MAX_BUFFERS: u32 = 1 << 20;

/// A JSON header and the arrays it refers to by index.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub header: Value,
    pub buffers: Vec<Vec<u8>>,
}

/// A frame that could not be read.
#[derive(Debug)]
pub struct FrameError(pub String);

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FrameError {}

impl From<std::io::Error> for FrameError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

fn take<'a>(bytes: &mut &'a [u8], count: usize) -> Result<&'a [u8], FrameError> {
    if bytes.len() < count {
        return Err(FrameError("frame ends early".to_string()));
    }
    let (head, rest) = bytes.split_at(count);
    *bytes = rest;
    Ok(head)
}

fn take_u32(bytes: &mut &[u8]) -> Result<u32, FrameError> {
    Ok(u32::from_le_bytes(
        take(bytes, 4)?.try_into().expect("four bytes"),
    ))
}

fn take_u64(bytes: &mut &[u8]) -> Result<u64, FrameError> {
    Ok(u64::from_le_bytes(
        take(bytes, 8)?.try_into().expect("eight bytes"),
    ))
}

impl Frame {
    /// A frame with only a header.
    #[must_use]
    pub fn header(header: Value) -> Self {
        Self {
            header,
            buffers: Vec::new(),
        }
    }

    /// Adds a buffer and returns its index for the header to refer to.
    pub fn push(&mut self, buffer: Vec<u8>) -> usize {
        self.buffers.push(buffer);
        self.buffers.len() - 1
    }

    /// The header's `kind`, or an empty string.
    #[must_use]
    pub fn kind(&self) -> &str {
        self.header
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// The frame as bytes: header length, header, buffer count, then each buffer with its length.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let header = self.header.to_string().into_bytes();
        let mut bytes = Vec::with_capacity(
            header.len()
                + 8
                + self
                    .buffers
                    .iter()
                    .map(|buffer| buffer.len() + 8)
                    .sum::<usize>(),
        );
        bytes.extend_from_slice(
            &u32::try_from(header.len())
                .expect("headers are small")
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(
            &u32::try_from(self.buffers.len())
                .expect("frames have few buffers")
                .to_le_bytes(),
        );
        for buffer in &self.buffers {
            bytes.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
            bytes.extend_from_slice(buffer);
        }
        bytes
    }

    /// Reads a frame written by `to_bytes`.
    ///
    /// # Errors
    ///
    /// If the bytes end early, have bytes left over, or the header is not JSON.
    pub fn from_bytes(mut bytes: &[u8]) -> Result<Self, FrameError> {
        let length = take_u32(&mut bytes)? as usize;
        let header = serde_json::from_slice(take(&mut bytes, length)?)
            .map_err(|error| FrameError(format!("bad header: {error}")))?;
        let count = take_u32(&mut bytes)?;
        if count > MAX_BUFFERS {
            return Err(FrameError(format!("a frame claims {count} buffers")));
        }
        let mut buffers = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let length = usize::try_from(take_u64(&mut bytes)?)
                .map_err(|_| FrameError("buffer too large".to_string()))?;
            buffers.push(take(&mut bytes, length)?.to_vec());
        }
        if !bytes.is_empty() {
            return Err(FrameError("bytes after the frame".to_string()));
        }
        Ok(Self { header, buffers })
    }

    /// Writes the frame with a length prefix.
    ///
    /// # Errors
    ///
    /// If the stream fails.
    pub async fn write_to(
        &self,
        stream: &mut (impl AsyncWriteExt + Unpin),
    ) -> Result<(), FrameError> {
        let bytes = self.to_bytes();
        stream
            .write_all(&(bytes.len() as u64).to_le_bytes())
            .await?;
        stream.write_all(&bytes).await?;
        stream.flush().await?;
        Ok(())
    }

    /// The frame with its buffers moved into a new file at `path`, for a worker to map.
    ///
    /// The header keeps the path and each buffer's size, in order; the worker
    /// maps the file, reads the buffers in place, and removes it. The file is
    /// readable by this user alone, and an existing file or link there is an
    /// error rather than written through.
    ///
    /// # Errors
    ///
    /// If the file exists or cannot be written.
    pub fn shared(&self, path: &std::path::Path) -> Result<Self, FrameError> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?;
        let mut file = std::io::BufWriter::new(file);
        for buffer in &self.buffers {
            file.write_all(buffer)?;
        }
        file.flush()?;
        let mut header = self.header.clone();
        header["shared"] = json!({
            "path": path.to_string_lossy(),
            "sizes": self.buffers.iter().map(Vec::len).collect::<Vec<_>>(),
        });
        Ok(Self {
            header,
            buffers: Vec::new(),
        })
    }

    /// Reads a frame written by `write_to`.
    ///
    /// # Errors
    ///
    /// If the stream fails or the frame is malformed.
    pub async fn read_from(stream: &mut (impl AsyncReadExt + Unpin)) -> Result<Self, FrameError> {
        let length = stream.read_u64_le().await?;
        if length > MAX_FRAME_BYTES {
            return Err(FrameError(format!("a frame claims {length} bytes")));
        }
        let mut bytes = vec![
            0;
            usize::try_from(length)
                .map_err(|_| FrameError("frame too large".to_string()))?
        ];
        stream.read_exact(&mut bytes).await?;
        Self::from_bytes(&bytes)
    }

    /// A buffer the header refers to under `field`.
    ///
    /// # Errors
    ///
    /// If the field is missing or names no buffer.
    pub fn buffer(&self, header: &Value, field: &str) -> Result<&[u8], FrameError> {
        let index = header
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| FrameError(format!("missing buffer field {field}")))?;
        self.buffers
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .map(Vec::as_slice)
            .ok_or_else(|| FrameError(format!("no buffer {index}")))
    }
}

/// Little-endian bytes of `f32` values.
#[must_use]
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// `f32` values from little-endian bytes.
///
/// # Errors
///
/// If the length is not a multiple of four.
pub fn f32_values(bytes: &[u8]) -> Result<Vec<f32>, FrameError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(FrameError("float buffer has a partial value".to_string()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

/// Little-endian bytes of `u64` values.
#[must_use]
pub fn u64_bytes(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Little-endian bytes of `u32` values.
#[must_use]
pub fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// `u32` values from little-endian bytes.
///
/// # Errors
///
/// If the length is not a multiple of four.
pub fn u32_values(bytes: &[u8]) -> Result<Vec<u32>, FrameError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(FrameError("integer buffer has a partial value".to_string()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

/// Little-endian bytes of `f64` values, each kept to the bit.
#[must_use]
pub fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// `f64` values from little-endian bytes.
///
/// # Errors
///
/// If the length is not a multiple of eight.
pub fn f64_values(bytes: &[u8]) -> Result<Vec<f64>, FrameError> {
    Ok(u64_values(bytes)?.into_iter().map(f64::from_bits).collect())
}

/// 128-bit keys as little-endian bytes.
#[must_use]
pub fn key_bytes(keys: &[u128]) -> Vec<u8> {
    keys.iter().flat_map(|key| key.to_le_bytes()).collect()
}

/// 128-bit keys from little-endian bytes.
///
/// # Errors
///
/// If the length is not a multiple of sixteen.
pub fn key_values(bytes: &[u8]) -> Result<Vec<u128>, FrameError> {
    if !bytes.len().is_multiple_of(16) {
        return Err(FrameError("key buffer has a partial key".to_string()));
    }
    Ok(bytes
        .chunks_exact(16)
        .map(|chunk| u128::from_le_bytes(chunk.try_into().expect("sixteen bytes")))
        .collect())
}

/// `i64` values as little-endian bytes.
#[must_use]
pub fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// `i64` values from little-endian bytes.
///
/// # Errors
///
/// If the length is not a multiple of eight.
pub fn i64_values(bytes: &[u8]) -> Result<Vec<i64>, FrameError> {
    Ok(u64_values(bytes)?
        .into_iter()
        .map(|value| value as i64)
        .collect())
}

/// `u64` values from little-endian bytes.
///
/// # Errors
///
/// If the length is not a multiple of eight.
pub fn u64_values(bytes: &[u8]) -> Result<Vec<u64>, FrameError> {
    if !bytes.len().is_multiple_of(8) {
        return Err(FrameError("integer buffer has a partial value".to_string()));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect())
}

/// Targets as a buffer and its type name.
#[must_use]
pub fn target_bytes(targets: &Targets) -> (Vec<u8>, &'static str) {
    match targets {
        Targets::Classes(labels) => (
            labels
                .iter()
                .flat_map(|label| label.to_le_bytes())
                .collect(),
            "classes",
        ),
        Targets::Values(values) => (f32_bytes(values), "values"),
    }
}

/// Targets from a buffer and its type name.
///
/// # Errors
///
/// If the type is unknown or the buffer has a partial value.
pub fn targets_from(bytes: &[u8], kind: &str) -> Result<Targets, FrameError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(FrameError("target buffer has a partial value".to_string()));
    }
    match kind {
        "classes" => Ok(Targets::Classes(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four bytes")))
                .collect(),
        )),
        "values" => Ok(Targets::Values(f32_values(bytes)?)),
        _ => Err(FrameError(format!("unknown target type {kind}"))),
    }
}

fn row_bytes(rows: &[RowId]) -> Vec<u8> {
    rows.iter().flat_map(|row| row.0.to_le_bytes()).collect()
}

/// A cache key as JSON: fingerprints are strings, since JSON numbers lose 128-bit precision.
fn key_json(key: &CacheKey) -> Value {
    json!([
        key.model.0,
        key.table.0,
        key.epoch.0,
        key.kind.name(),
        key.rows.bits().to_string(),
    ])
}

/// Movements as JSON for a worker.
#[must_use]
pub fn movements_json(movements: &[Movement]) -> Value {
    Value::Array(
        movements
            .iter()
            .map(|movement| {
                json!({
                    "key": key_json(&movement.key),
                    "source": movement.from.name(),
                    "target": movement.to.as_ref().map(|placement| placement.tier.name()),
                    "slots": movement.to.as_ref().map_or_else(Vec::new, |placement| {
                        placement.slots.iter().map(|slot| slot.0).collect()
                    }),
                })
            })
            .collect(),
    )
}

/// A step as a frame for a worker.
#[must_use]
pub fn step_frame(step: &Step) -> Frame {
    let mut frame = Frame::header(Value::Null);
    let fits: Vec<Value> = step
        .fits()
        .iter()
        .map(|fit| {
            let (target_buffer, target_type) = target_bytes(&fit.targets);
            json!({
                "table": fit.table.0,
                "epoch": fit.epoch.0,
                "rows": frame.push(row_bytes(&fit.rows)),
                "count": fit.rows.len(),
                "features": frame.push(f32_bytes(&fit.features)),
                "targets": frame.push(target_buffer),
                "target_type": target_type,
                "pages": fit.pages.iter().map(|page| page.0).collect::<Vec<_>>(),
            })
        })
        .collect();
    let pages: Vec<Value> = step
        .pages()
        .iter()
        .map(|page| {
            let (features, targets, target_type) = match &page.embed {
                Some((features, targets)) => {
                    let (target_buffer, target_type) = target_bytes(targets);
                    (
                        Some(frame.push(f32_bytes(features))),
                        Some(frame.push(target_buffer)),
                        Some(target_type),
                    )
                }
                None => (None, None, None),
            };
            json!({
                "table": page.table.0,
                "key": key_json(&page.key),
                "rows": frame.push(row_bytes(&page.rows)),
                "count": page.rows.len(),
                "action": page.placed.action().name(),
                "slots": slot_numbers(page.placed.slots()),
                "features": features,
                "targets": targets,
                "target_type": target_type,
                "copies": page
                    .copies
                    .iter()
                    .map(|(source, positions)| {
                        json!({"source": key_json(source), "positions": positions})
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let units: Vec<Value> = step
        .units()
        .iter()
        .map(|unit| unit_json(&mut frame, unit))
        .collect();
    let queries: Vec<Value> = step
        .queries()
        .iter()
        .map(|batch| batch_json(&mut frame, batch))
        .collect();
    let series_units: Vec<Value> = step
        .series_units()
        .iter()
        .map(|unit| series_unit_json(&mut frame, unit))
        .collect();
    // Forecasts of one unit over one horizon share one known future.
    let mut futures: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let forecasts: Vec<Value> = step
        .forecasts()
        .iter()
        .map(|forecast| forecast_json(&mut frame, forecast, &mut futures))
        .collect();
    frame.header = json!({
        "kind": "step",
        "movements": movements_json(step.movements()),
        "fits": fits,
        "pages": pages,
        "units": units,
        "queries": queries,
        "series_units": series_units,
        "forecasts": forecasts,
    });
    frame
}

fn bool_bytes(values: &[bool]) -> Vec<u8> {
    values.iter().map(|&value| u8::from(value)).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A grid's values and mask as buffers; the worker reads them `rows` by the
/// steps the entry names.
fn grid_buffers(frame: &mut Frame, grid: &Grid) -> (usize, usize) {
    let values = frame.push(f32_bytes(&grid.values));
    let observed = frame.push(bool_bytes(&grid.observed));
    (values, observed)
}

/// A series unit as the worker reads it: its group's grid, with the members
/// as strings, since JSON numbers lose 64-bit precision.
fn series_unit_json(frame: &mut Frame, unit: &SeriesUnit) -> Value {
    let (values, observed) = grid_buffers(frame, &unit.grid);
    json!({
        "table": unit.table.0,
        "key": key_json(&unit.key),
        "members": unit.members.iter().map(|member| member.0.to_string()).collect::<Vec<_>>(),
        "cutoff": unit.cutoff,
        "rows": unit.grid.rows,
        "steps": unit.grid.steps,
        "values": values,
        "observed": observed,
        "roles": frame.push(i32_bytes(&unit.grid.roles)),
        "action": unit.placed.action().name(),
        "slots": slot_numbers(unit.placed.slots()),
        "source": unit.source.as_ref().map(key_json),
        "grown": unit.grown,
    })
}

/// A forecast as the worker reads it: the unit it reads, its target member,
/// and the group's known future over the horizon, which forecasts of one
/// unit over one horizon share.
fn forecast_json(
    frame: &mut Frame,
    forecast: &Forecast,
    futures: &mut HashMap<(usize, usize), (usize, usize)>,
) -> Value {
    let (values, observed) = *futures
        .entry((forecast.unit.0, forecast.horizon))
        .or_insert_with(|| grid_buffers(frame, &forecast.future));
    json!({
        "request": forecast.request.0,
        "table": forecast.table.0,
        "epoch": forecast.epoch.0,
        "unit": forecast.unit.0,
        "target": forecast.target,
        "horizon": forecast.horizon,
        "rows": forecast.future.rows,
        "values": values,
        "observed": observed,
    })
}

fn slot_numbers(slots: &[SlotId]) -> Vec<u32> {
    slots.iter().map(|slot| slot.0).collect()
}

fn unit_json(frame: &mut Frame, unit: &Unit) -> Value {
    let (features, targets, target_type) = match &unit.fit {
        Some((features, targets)) => {
            let (target_buffer, target_type) = target_bytes(targets);
            (
                Some(frame.push(f32_bytes(features))),
                Some(frame.push(target_buffer)),
                Some(target_type),
            )
        }
        None => (None, None, None),
    };
    json!({
        "table": unit.table.0,
        "key": key_json(&unit.key),
        "rows": frame.push(row_bytes(&unit.rows)),
        "count": unit.rows.len(),
        "action": unit.placed.action().name(),
        "slots": slot_numbers(unit.placed.slots()),
        "features": features,
        "targets": targets,
        "target_type": target_type,
    })
}

fn batch_json(frame: &mut Frame, batch: &QueryBatch) -> Value {
    json!({
        "request": batch.request.0,
        "table": batch.table.0,
        "epoch": batch.epoch.0,
        "count": batch.units.len(),
        "features": frame.push(f32_bytes(&batch.features)),
        "units": batch.units.iter().map(|unit| unit.0).collect::<Vec<_>>(),
    })
}

/// What the server asks a worker, beside steps, which `step_frame` builds
/// with their arrays. The worker ships in the same wheel, so both ends agree
/// on these exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerRequest {
    /// Sizes the worker's slot pools as the node's tiers.
    Allocate {
        device_slots: u32,
        host_slots: u32,
        disk_slots: u32,
    },
    /// What a model needs per context row, for a table of `features` columns.
    Profile {
        model: String,
        features: usize,
    },
    /// How many rows a model's fit takes within the worker's call budget,
    /// for a table of `features` columns.
    FitLimits {
        model: String,
        features: usize,
    },
    /// Loads a model the worker did not start with, under `name`, from a
    /// built-in model's repository id or a `module:function` spec: a control,
    /// sent in a `Controls` frame ahead of the first table naming the model.
    AddModel {
        name: String,
        spec: String,
    },
    /// Whether a model can fit a table of this task and width.
    Check {
        model: String,
        classes: u32,
        features: usize,
        /// Series a forecast's group holds at most; tabular tables send 1.
        series: usize,
    },
    AddTable {
        table: u32,
        model: String,
        classes: u32,
        features: usize,
        /// Columns holding category codes.
        categorical: Vec<usize>,
        /// Rows every page embed pads to at least.
        page_size: usize,
        /// What the table's column summary is fit on.
        summary: SummarySpec,
        /// Whether the table holds series, forecast by a forecaster.
        #[serde(default)]
        series: bool,
    },
    RemoveTable {
        table: u32,
    },
    /// Copies and drops between tiers, as `movements_json` writes them.
    Apply {
        movements: Value,
    },
    /// Models loaded, tables added and removed and tiers' copies and drops, carried out in
    /// order: what a shard tick asks of the worker besides its steps, in one
    /// frame. The worker answers `Ok`, or `Controlled` with those that failed.
    Controls {
        requests: Vec<WorkerRequest>,
    },
    /// Samples the worker's threads for `seconds`, their stacks written
    /// collapsed to `path`; the worker answers once the sampling starts,
    /// after the step in flight, and `path` appears when it ends. With
    /// `trace`, JAX's profiler traces the same window into that directory.
    Record {
        seconds: f64,
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace: Option<String>,
    },
}

impl WorkerRequest {
    /// The request as a frame.
    #[must_use]
    pub fn frame(&self) -> Frame {
        Frame::header(serde_json::to_value(self).expect("requests serialize"))
    }

    /// The request a frame holds.
    ///
    /// # Errors
    ///
    /// If the frame holds no request, such as a step.
    pub fn of(frame: &Frame) -> Result<Self, FrameError> {
        serde_json::from_value(frame.header.clone())
            .map_err(|error| FrameError(format!("not a worker request: {error}")))
    }
}

/// A request of a `Controls` frame that failed: its index there, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlFailure {
    pub index: usize,
    pub message: String,
}

/// A worker's reply.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerReply {
    /// Sent once on connecting: the models the worker loaded, in the order it
    /// numbers them, its device's memory, and the tiers its memory budget gives.
    Ready {
        /// Missing from workers before versions, which read as 0.
        #[serde(default)]
        protocol: u32,
        models: Vec<String>,
        device_memory: u64,
        device_bytes: u64,
        host_bytes: u64,
        row_memory_bytes: u64,
    },
    Profile(ModelProfile),
    /// How many rows a model's fit takes within the worker's call budget:
    /// the most a state alone is fit on, and the most a fit embeds in the
    /// call that fits its state; None for any.
    FitLimits {
        state_rows: Option<u64>,
        fused_rows: Option<u64>,
    },
    Ok,
    /// A check the model refuses.
    Invalid {
        message: String,
    },
    /// A request that failed.
    Error {
        message: String,
    },
    /// The requests of a `Controls` frame that failed; the rest were carried out.
    Controlled {
        failures: Vec<ControlFailure>,
    },
    /// An allocation carried out, after compiling the programs earlier
    /// workers recorded for its models, and the seconds they took.
    Warmed {
        compiles: u64,
        seconds: f64,
    },
    /// Sent while a step runs, each time the worker finishes compiling a
    /// program, so a long compile is not taken for a hung worker; it
    /// answers no request.
    Progress,
    /// A step's predictions, whose values are the frame's buffers, the
    /// tables whose part of the step failed, the seconds the step took the
    /// worker, its compiled calls and their seconds by stage, and its parts
    /// outside those calls, which split its seconds, and their seconds.
    Predictions {
        results: Vec<PredictionsHeader>,
        #[serde(default)]
        failures: Vec<Failure>,
        #[serde(default)]
        seconds: f64,
        #[serde(default)]
        stages: BTreeMap<String, StageTime>,
        #[serde(default)]
        parts: BTreeMap<String, StageTime>,
    },
}

impl WorkerReply {
    /// The reply as a frame, with no buffers.
    #[must_use]
    pub fn frame(&self) -> Frame {
        Frame::header(serde_json::to_value(self).expect("replies serialize"))
    }

    /// The reply a frame holds.
    ///
    /// # Errors
    ///
    /// If the frame is not a reply this server reads.
    pub fn of(frame: &Frame) -> Result<Self, FrameError> {
        serde_json::from_value(frame.header.clone())
            .map_err(|error| FrameError(format!("not a worker reply: {error}")))
    }
}

/// What a model needs per table, as a worker measures it for one feature count.
///
/// A forecaster's profile has `series` set and the series fields filled: a
/// patch of its cache takes `bytes_per_row` per padded row of a group and
/// `patch_bytes` beside them. A tabular model's profile leaves them out, and
/// they read as their defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    /// Context cache bytes per row.
    pub bytes_per_row: u64,
    /// One row's representation.
    pub representation_bytes: u64,
    /// Whether the model predicts values rather than classes.
    pub regression: bool,
    /// The most classes the model serves.
    pub class_slots: u64,
    /// Columns in the model's predictions: class slots, or quantiles.
    pub outputs: u64,
    /// The most context rows the model was trained on, if it says.
    pub context_rows: Option<u64>,
    /// The table state, which a table that summarizes each context keeps per
    /// unit, for a table without categorical columns.
    pub state_bytes: u64,
    /// The table state of a table with categorical columns, which some
    /// models keep more for; 0 from forecasters and workers that leave it out.
    #[serde(default)]
    pub categorical_state_bytes: u64,
    /// Whether the state holds a column summary pooled over context rows, so
    /// a table may summarize each context.
    pub column_summary: bool,
    #[serde(default)]
    pub series: bool,
    /// Cache bytes per patch beside those per row.
    #[serde(default)]
    pub patch_bytes: u64,
    /// Steps per input patch.
    #[serde(default)]
    pub patch: u64,
    /// Steps per output patch, which horizons pad to.
    #[serde(default)]
    pub output_patch: u64,
    /// Most context steps the forecaster reads.
    #[serde(default)]
    pub context_limit: u64,
    /// Most horizon steps one pass predicts, if the forecaster loops.
    #[serde(default)]
    pub pass_limit: Option<u64>,
    /// The quantile levels of a forecast's last axis, increasing.
    #[serde(default)]
    pub quantiles: Option<Vec<f64>>,
    /// Whether the forecaster reuses a context encoding across horizons.
    #[serde(default)]
    pub reusable: bool,
    /// Whether the forecaster grows an encoding by whole patches exactly.
    #[serde(default)]
    pub extendable: bool,
    /// Most rows in a group the forecaster reads, if it has a cap.
    #[serde(default)]
    pub max_rows: Option<u64>,
    /// Whether the model reads context rows whose target is not known yet.
    pub unlabeled_context: bool,
    /// The most rows a table's fit takes before the server asks the worker
    /// for its `FitLimits`, or None where fits are not limited.
    #[serde(default)]
    pub fit_bound: Option<u64>,
}

/// One request's predictions in a reply: which buffer holds them, and their shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionsHeader {
    pub request: u64,
    pub rows: usize,
    pub columns: usize,
    pub buffer: usize,
    /// Steps per row of a forecast, whose values are `rows` by `horizon` by
    /// `columns`; None for a table's predictions.
    #[serde(default)]
    pub horizon: Option<usize>,
}

/// A table whose part of a step failed, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub table: u32,
    pub message: String,
}

/// One stage's compiled calls in a step, and the seconds they took.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageTime {
    pub calls: u64,
    pub seconds: f64,
}

/// A request's predictions: `rows` by `columns`, row-major, or for a
/// forecast `rows` by `horizon` by `columns`.
#[derive(Clone, Debug, PartialEq)]
pub struct Predictions {
    pub rows: usize,
    pub columns: usize,
    pub horizon: Option<usize>,
    pub values: Vec<f32>,
    /// Where each series' forecast starts, which the shard sets for a
    /// forecast: its cutoff, the first step forecast.
    pub cutoffs: Vec<i64>,
    /// The request's share of its step's worker time, which the shard sets.
    pub gpu_seconds: f64,
    /// The epoch of the table's state that answered, which the shard sets: it
    /// rises with each refit.
    pub epoch: u64,
}

/// A worker's reply to a step: predictions per request, the tables whose
/// part of the step failed, with why, and what the step cost the worker.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    pub predictions: Vec<(u64, Predictions)>,
    pub failed: Vec<(TableId, String)>,
    pub seconds: f64,
    pub stages: BTreeMap<String, StageTime>,
    /// The step's parts, which split its seconds: how often each ran and
    /// its seconds, as the worker's `WorkerStats.parts` names them.
    pub parts: BTreeMap<String, StageTime>,
}

/// A worker's reply to a step.
///
/// # Errors
///
/// If the frame is not predictions, or their shapes do not match their buffers.
pub fn outcome_from(frame: &Frame) -> Result<Outcome, FrameError> {
    let WorkerReply::Predictions {
        results,
        failures,
        seconds,
        stages,
        parts,
    } = WorkerReply::of(frame)?
    else {
        return Err(FrameError(format!(
            "the worker answered a step with {}",
            frame.header
        )));
    };
    let predictions = results
        .into_iter()
        .map(|result| {
            let buffer = frame
                .buffers
                .get(result.buffer)
                .ok_or_else(|| FrameError(format!("no buffer {}", result.buffer)))?;
            let values = f32_values(buffer)?;
            let cells = result
                .rows
                .checked_mul(result.columns)
                .and_then(|cells| cells.checked_mul(result.horizon.unwrap_or(1)));
            if cells != Some(values.len()) {
                return Err(FrameError("predictions have the wrong size".to_string()));
            }
            Ok((
                result.request,
                Predictions {
                    rows: result.rows,
                    columns: result.columns,
                    horizon: result.horizon,
                    values,
                    cutoffs: Vec::new(),
                    gpu_seconds: 0.0,
                    epoch: 0,
                },
            ))
        })
        .collect::<Result<_, _>>()?;
    Ok(Outcome {
        predictions,
        failed: failures
            .into_iter()
            .map(|failure| (TableId(failure.table), failure.message))
            .collect(),
        seconds,
        stages,
        parts,
    })
}

/// What a table's column summary is fit on, as clients and logs spell it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SummarySpec {
    #[default]
    Table,
    Context,
}

impl SummarySpec {
    /// The node's scope this spec names.
    #[must_use]
    pub fn scope(self) -> SummaryScope {
        match self {
            Self::Table => SummaryScope::Table,
            Self::Context => SummaryScope::Context,
        }
    }
}

/// How a table chooses each query's context, as clients and logs spell it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StrategySpec {
    #[default]
    Exact,
    Retrieved {
        #[serde(default)]
        core: Option<CoreSpec>,
        #[serde(default)]
        neighbors: usize,
        #[serde(default)]
        among: Option<usize>,
        #[serde(default)]
        recency: Option<RecencySpec>,
    },
    /// A random core alone, as earlier clients spell it.
    Subsample {
        rows: u64,
        #[serde(default)]
        seed: u64,
    },
    /// Neighbors without a core, as earlier clients spell them.
    Neighbors { rows: usize },
}

/// A retrieved context's core, as clients and logs spell it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreSpec {
    pub rows: u64,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub kind: CoreKindSpec,
}

/// The recent rows a retrieved context keeps; see `Recency`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecencySpec {
    Last(u64),
    Span(u64),
}

/// How a core's rows are drawn, by `CoreKind`'s name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CoreKindSpec(pub CoreKind);

impl Default for CoreKindSpec {
    fn default() -> Self {
        Self(CoreKind::Random)
    }
}

impl TryFrom<String> for CoreKindSpec {
    type Error = String;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        name.parse().map(Self)
    }
}

impl From<CoreKindSpec> for String {
    fn from(kind: CoreKindSpec) -> Self {
        kind.0.name().to_string()
    }
}

impl StrategySpec {
    /// The planner strategy this spec names.
    #[must_use]
    pub fn strategy(&self) -> Strategy {
        match *self {
            Self::Exact => Strategy::Exact,
            Self::Retrieved {
                core,
                neighbors,
                among,
                recency,
            } => Strategy::Retrieved {
                core: core.map(|core| Core {
                    rows: core.rows,
                    seed: core.seed,
                    kind: core.kind.0,
                }),
                neighbors,
                among,
                recency: recency.map(|recency| match recency {
                    RecencySpec::Last(rows) => Recency::Last(rows),
                    RecencySpec::Span(span) => Recency::Span(span),
                }),
            },
            Self::Subsample { rows, seed } => Strategy::subsample(rows, seed, CoreKind::Random),
            Self::Neighbors { rows } => Strategy::neighbors(rows),
        }
    }
}
