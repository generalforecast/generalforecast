// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Series: each series' rows by time, laid onto regular grids before a cutoff.
//!
//! A series table's rows are observations, one per series and time, keyed by
//! the series and the time packed together. A forecast reads a series' last
//! `steps` grid points before a cutoff, so the index keeps each series' rows
//! ordered by time and lays them onto the grid as rows and an observed mask,
//! gaps unobserved. Grids are regular: the table declares a step, every time
//! of a series is a whole number of steps from its others, and a time already
//! taken is refused. Calendar frequencies that are not regular in the time
//! unit, such as months, use period numbers as times, with a step of 1.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::num::NonZeroU64;

use crate::context::{Context, Fingerprint};
use crate::table::{RowId, RowKey};

/// A series' key: a 64-bit digest of the client's key, which a row key holds
/// in its high half, with the row's time in its low half.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeriesKey(pub u64);

impl SeriesKey {
    /// The row key of this series' observation at `time`.
    #[must_use]
    pub fn row_key(self, time: i64) -> RowKey {
        RowKey((u128::from(self.0) << 64) | u128::from(time as u64))
    }

    /// The series and time a row key packs.
    #[must_use]
    pub fn unpack(key: RowKey) -> (Self, i64) {
        (Self((key.0 >> 64) as u64), key.0 as u64 as i64)
    }
}

/// Why a row could not be placed in a series, or a grid laid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeriesError {
    /// A step longer than any span of times.
    Step,
    /// A row key whose packed time is not the time the row was given.
    Mismatch { key: RowKey, time: i64 },
    /// A time that is not a whole number of steps from the series' other times.
    Irregular { series: SeriesKey, time: i64 },
    /// A time a row of the series already has.
    Taken { series: SeriesKey, time: i64 },
    /// A cutoff that is not on the series' grid.
    OffGrid { series: SeriesKey, cutoff: i64 },
}

impl fmt::Display for SeriesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Step => write!(formatter, "the step is longer than any span of times"),
            Self::Mismatch { key, time } => {
                write!(formatter, "row key {key} does not pack time {time}")
            }
            Self::Irregular { series, time } => write!(
                formatter,
                "time {time} is off the grid of series {}",
                series.0
            ),
            Self::Taken { series, time } => {
                write!(formatter, "series {} already has time {time}", series.0)
            }
            Self::OffGrid { series, cutoff } => write!(
                formatter,
                "cutoff {cutoff} is off the grid of series {}",
                series.0
            ),
        }
    }
}

impl std::error::Error for SeriesError {}

/// One series' rows by time, the residue its times share modulo the step,
/// and the times whose target is unobserved: a covariate's value stored ahead
/// of the series, or a gap.
#[derive(Debug)]
struct Series {
    residue: i64,
    rows: BTreeMap<i64, RowId>,
    blank: BTreeSet<i64>,
}

/// A series' rows on a grid of `steps` points, laid from a cutoff.
///
/// From `window`, `rows[k]` is the row at `cutoff - (steps - k) * step`: the
/// last entry is the step before the cutoff, and no row is at or after it.
/// From `after`, `rows[k]` is the row at `cutoff + k * step`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub cutoff: i64,
    pub rows: Vec<Option<RowId>>,
}

impl Window {
    /// The rows on the grid, as a context.
    #[must_use]
    pub fn context(&self) -> Context {
        self.rows.iter().flatten().copied().collect()
    }

    /// Which grid points hold a row.
    #[must_use]
    pub fn observed(&self) -> Vec<bool> {
        self.rows.iter().map(Option::is_some).collect()
    }

    /// How many grid points hold a row.
    #[must_use]
    pub fn count(&self) -> usize {
        self.rows.iter().flatten().count()
    }
}

/// Every series' rows by time, kept as rows come and go.
///
/// The time a row key packs is the row's time: `insert` refuses a row whose
/// key says otherwise, so the grids here and the cutoffs `TimeIndex` applies
/// never disagree.
#[derive(Debug)]
pub struct SeriesIndex {
    step: i64,
    series: HashMap<SeriesKey, Series>,
}

impl SeriesIndex {
    /// An empty index for grids of `step`, in the table's time unit.
    ///
    /// # Errors
    ///
    /// If the step is longer than any span of times.
    pub fn new(step: NonZeroU64) -> Result<Self, SeriesError> {
        let step = i64::try_from(step.get()).map_err(|_| SeriesError::Step)?;
        Ok(Self {
            step,
            series: HashMap::new(),
        })
    }

    /// The grid's step.
    #[must_use]
    pub fn step(&self) -> i64 {
        self.step
    }

    /// Checks that a row could be placed at the series and time its key packs.
    ///
    /// # Errors
    ///
    /// The error `insert` would return.
    pub fn check(&self, key: RowKey, time: i64) -> Result<SeriesKey, SeriesError> {
        let (series, packed) = SeriesKey::unpack(key);
        if packed != time {
            return Err(SeriesError::Mismatch { key, time });
        }
        if let Some(held) = self.series.get(&series) {
            if held.residue != time.rem_euclid(self.step) {
                return Err(SeriesError::Irregular { series, time });
            }
            if held.rows.contains_key(&time) {
                return Err(SeriesError::Taken { series, time });
            }
        }
        Ok(series)
    }

    /// Checks that a batch of rows could be placed, each at the series and
    /// time its key packs, rows earlier in the batch included: a new series'
    /// grid is set by its first row, and a time the batch places twice is
    /// taken. A time the index holds is taken unless `replaceable` says the
    /// row may be written over.
    ///
    /// # Errors
    ///
    /// The first error `insert` would meet applying the batch in order.
    pub fn check_batch(
        &self,
        rows: impl IntoIterator<Item = (RowKey, i64)>,
        replaceable: impl Fn(RowKey) -> bool,
    ) -> Result<(), SeriesError> {
        let mut grids: HashMap<SeriesKey, (i64, BTreeSet<i64>)> = HashMap::new();
        for (key, time) in rows {
            let (series, packed) = SeriesKey::unpack(key);
            if packed != time {
                return Err(SeriesError::Mismatch { key, time });
            }
            let held = self.series.get(&series);
            let (residue, placed) = grids.entry(series).or_insert_with(|| {
                let residue = held.map_or_else(|| time.rem_euclid(self.step), |held| held.residue);
                (residue, BTreeSet::new())
            });
            if *residue != time.rem_euclid(self.step) {
                return Err(SeriesError::Irregular { series, time });
            }
            let taken = held.is_some_and(|held| held.rows.contains_key(&time)) && !replaceable(key);
            if taken || !placed.insert(time) {
                return Err(SeriesError::Taken { series, time });
            }
        }
        Ok(())
    }

    /// Places a row at the series and time its key packs, `observed` when its
    /// target has a value.
    ///
    /// # Errors
    ///
    /// If the key does not pack `time`, the time is off the series' grid, or
    /// a row of the series has it. The index is then unchanged.
    pub fn insert(
        &mut self,
        row: RowId,
        key: RowKey,
        time: i64,
        observed: bool,
    ) -> Result<SeriesKey, SeriesError> {
        let series = self.check(key, time)?;
        let held = self.series.entry(series).or_insert_with(|| Series {
            residue: time.rem_euclid(self.step),
            rows: BTreeMap::new(),
            blank: BTreeSet::new(),
        });
        held.rows.insert(time, row);
        if !observed {
            held.blank.insert(time);
        }
        Ok(series)
    }

    /// Records whether the row at the series and time a key packs has a
    /// target, as an update changes it.
    pub fn observe(&mut self, key: RowKey, observed: bool) {
        let (series, time) = SeriesKey::unpack(key);
        if let Some(held) = self.series.get_mut(&series)
            && held.rows.contains_key(&time)
        {
            if observed {
                held.blank.remove(&time);
            } else {
                held.blank.insert(time);
            }
        }
    }

    /// Removes the row at the series and time a key packs, returning it if
    /// there was one. A series without rows forgets its grid.
    pub fn remove(&mut self, key: RowKey) -> Option<RowId> {
        let (series, time) = SeriesKey::unpack(key);
        let held = self.series.get_mut(&series)?;
        let row = held.rows.remove(&time)?;
        held.blank.remove(&time);
        if held.rows.is_empty() {
            self.series.remove(&series);
        }
        Some(row)
    }

    /// The row at the series and time a key packs, if there is one.
    #[must_use]
    pub fn row(&self, key: RowKey) -> Option<RowId> {
        let (series, time) = SeriesKey::unpack(key);
        self.series.get(&series)?.rows.get(&time).copied()
    }

    /// Every series with rows, in no particular order.
    pub fn series(&self) -> impl Iterator<Item = SeriesKey> + '_ {
        self.series.keys().copied()
    }

    /// How many rows a series has.
    #[must_use]
    pub fn count(&self, series: SeriesKey) -> usize {
        self.series.get(&series).map_or(0, |held| held.rows.len())
    }

    /// Whether no series has rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// The first point of a series' grid at or after `cutoff`: the cutoff
    /// itself for a series without rows, or on the same grid. A window before
    /// it holds every row of the series before the cutoff, since no row sits
    /// between them.
    ///
    /// # Errors
    ///
    /// If no point of the grid follows the cutoff within `i64`.
    pub fn align(&self, series: SeriesKey, cutoff: i64) -> Result<i64, SeriesError> {
        let Some(held) = self.series.get(&series) else {
            return Ok(cutoff);
        };
        let ahead =
            (i128::from(held.residue) - i128::from(cutoff)).rem_euclid(i128::from(self.step));
        cutoff
            .checked_add(ahead as i64)
            .ok_or(SeriesError::OffGrid { series, cutoff })
    }

    /// Whether the series has rows.
    #[must_use]
    pub fn has(&self, series: SeriesKey) -> bool {
        self.series.contains_key(&series)
    }

    /// Whether the series has an observed target before `cutoff`.
    #[must_use]
    pub fn observed_before(&self, series: SeriesKey, cutoff: i64) -> bool {
        self.series.get(&series).is_some_and(|held| {
            held.rows
                .range(..cutoff)
                .any(|(time, _)| !held.blank.contains(time))
        })
    }

    /// A series' earliest time, if it has rows.
    #[must_use]
    pub fn earliest(&self, series: SeriesKey) -> Option<i64> {
        self.series
            .get(&series)
            .and_then(|held| held.rows.keys().next().copied())
    }

    /// A series' latest time, if it has rows.
    #[must_use]
    pub fn latest(&self, series: SeriesKey) -> Option<i64> {
        self.series
            .get(&series)
            .and_then(|held| held.rows.keys().next_back().copied())
    }

    /// A series' latest time with an observed target, if it has one: where
    /// a forecast without a cutoff starts, a step later, so covariates stored
    /// ahead of the series are its known future.
    #[must_use]
    pub fn latest_observed(&self, series: SeriesKey) -> Option<i64> {
        self.series.get(&series).and_then(|held| {
            held.rows
                .keys()
                .rev()
                .find(|time| !held.blank.contains(time))
                .copied()
        })
    }

    /// A series' rows on the `steps` grid points before `cutoff`, none at or
    /// after it: the context of a forecast from the cutoff on.
    ///
    /// A series without rows lies on any grid, so a new series gets a window
    /// of unobserved points.
    ///
    /// # Errors
    ///
    /// If `cutoff` is off the series' grid.
    pub fn window(
        &self,
        series: SeriesKey,
        cutoff: i64,
        steps: usize,
    ) -> Result<Window, SeriesError> {
        let span = i128::from(self.step) * steps as i128;
        let first = (i128::from(cutoff) - span).max(i128::from(i64::MIN)) as i64;
        self.grid(series, cutoff, steps, first..cutoff, |time| {
            let back = (i128::from(cutoff) - i128::from(time)) / i128::from(self.step);
            steps - back as usize
        })
    }

    /// A series' rows on the `steps` grid points from `cutoff` on: the known
    /// future of a covariate, which only a known covariate's features may
    /// read, never a target's.
    ///
    /// # Errors
    ///
    /// If `cutoff` is off the series' grid.
    pub fn after(
        &self,
        series: SeriesKey,
        cutoff: i64,
        steps: usize,
    ) -> Result<Window, SeriesError> {
        if steps == 0 {
            return self.grid(series, cutoff, 0, cutoff..cutoff, |_| 0);
        }
        let span = i128::from(self.step) * (steps as i128 - 1);
        let last = (i128::from(cutoff) + span).min(i128::from(i64::MAX)) as i64;
        self.grid(series, cutoff, steps, cutoff..=last, |time| {
            let forward = (i128::from(time) - i128::from(cutoff)) / i128::from(self.step);
            forward as usize
        })
    }

    fn grid<R: std::ops::RangeBounds<i64>>(
        &self,
        series: SeriesKey,
        cutoff: i64,
        steps: usize,
        times: R,
        slot: impl Fn(i64) -> usize,
    ) -> Result<Window, SeriesError> {
        let mut rows = vec![None; steps];
        if let Some(held) = self.series.get(&series) {
            if held.residue != cutoff.rem_euclid(self.step) {
                return Err(SeriesError::OffGrid { series, cutoff });
            }
            for (&time, &row) in held.rows.range(times) {
                rows[slot(time)] = Some(row);
            }
        }
        Ok(Window { cutoff, rows })
    }
}

/// A unit's fingerprint with its window folded in: the same rows before
/// another cutoff, or on a grid of another length, are another context, with
/// other unobserved steps and other patch boundaries.
#[must_use]
pub fn with_window(print: Fingerprint, cutoff: i64, steps: usize) -> Fingerprint {
    print.salted(cutoff as u64).salted(steps as u64)
}
