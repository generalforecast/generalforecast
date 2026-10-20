//! Rows by timestamp, for time cutoffs and windows.

use std::collections::{BTreeMap, HashMap, btree_map};

use roaring::RoaringBitmap;

use crate::context::Context;
use crate::table::RowId;

/// Each row's timestamp, ordered, so time ranges come back as contexts.
///
/// Rows without a time are before every cutoff, held apart, so every `i64`
/// stays a time a row may have.
#[derive(Debug, Default)]
pub struct TimeIndex {
    by_time: BTreeMap<i64, RoaringBitmap>,
    times: HashMap<RowId, i64>,
    untimed: RoaringBitmap,
}

impl TimeIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a row's timestamp, moving it if it already has one.
    pub fn insert(&mut self, row: RowId, time: i64) {
        self.remove(row);
        self.by_time.entry(time).or_default().insert(row.0);
        self.times.insert(row, time);
    }

    /// Adds a row without a time, before every cutoff, moving it if it has one.
    pub fn insert_untimed(&mut self, row: RowId) {
        self.remove(row);
        self.untimed.insert(row.0);
    }

    /// Sets a row's time, or none.
    pub fn set(&mut self, row: RowId, time: Option<i64>) {
        match time {
            Some(time) => self.insert(row, time),
            None => self.insert_untimed(row),
        }
    }

    /// Whether a row is held without a time.
    #[must_use]
    pub fn is_untimed(&self, row: RowId) -> bool {
        self.untimed.contains(row.0)
    }

    /// Whether any row is held without a time.
    #[must_use]
    pub fn has_untimed(&self) -> bool {
        !self.untimed.is_empty()
    }

    /// Removes a row, returning its timestamp if it had one.
    pub fn remove(&mut self, row: RowId) -> Option<i64> {
        self.untimed.remove(row.0);
        let time = self.times.remove(&row)?;
        let rows = self
            .by_time
            .get_mut(&time)
            .expect("indexed times have rows");
        rows.remove(row.0);
        if rows.is_empty() {
            self.by_time.remove(&time);
        }
        Some(time)
    }

    /// Rows strictly before `time`, and rows without a time.
    #[must_use]
    pub fn before(&self, time: i64) -> Context {
        let mut rows = self.untimed.clone();
        for (_, bitmap) in self.by_time.range(..time) {
            rows |= bitmap;
        }
        Context::from_bitmap(rows)
    }

    /// The earliest row time at or after `time`, if any row has one.
    #[must_use]
    pub fn first_from(&self, time: i64) -> Option<i64> {
        self.by_time.range(time..).next().map(|(&time, _)| time)
    }

    /// The earliest timestamp of any row with one, if there are such rows.
    #[must_use]
    pub fn earliest(&self) -> Option<i64> {
        self.by_time.keys().next().copied()
    }

    /// The first time at or after `time` of a row not among `excluded`, if any.
    #[must_use]
    pub fn first_outside(&self, time: i64, excluded: &RoaringBitmap) -> Option<i64> {
        self.by_time
            .range(time..)
            .find(|(_, rows)| !rows.is_subset(excluded))
            .map(|(&time, _)| time)
    }

    /// Whether a row without a time is not among `excluded`.
    #[must_use]
    pub fn has_untimed_outside(&self, excluded: &RoaringBitmap) -> bool {
        !self.untimed.is_subset(excluded)
    }

    /// A row's timestamp, if it has one.
    #[must_use]
    pub fn time(&self, row: RowId) -> Option<i64> {
        self.times.get(&row).copied()
    }

    /// The rows before `cutoff` that are not before `from`, if given: the
    /// untimed rows and every row before `cutoff` without it, so windows
    /// before ascending cutoffs grow one into the next.
    #[must_use]
    pub fn added_before(&self, from: Option<i64>, cutoff: i64) -> Context {
        match from {
            None => self.before(cutoff),
            Some(from) => self.between(from, cutoff),
        }
    }

    /// The `count` latest rows of `within`, by time: untimed rows are the
    /// oldest, and at one time the higher rows count as later. With `before`,
    /// every row of `within` is before it, so the walk starts there.
    #[must_use]
    pub fn latest(&self, within: &Context, count: u64, before: Option<i64>) -> Context {
        let within = within.bitmap();
        let mut rows = RoaringBitmap::new();
        let groups = self
            .up_to(before)
            .map(|(_, rows)| rows)
            .rev()
            .chain(std::iter::once(&self.untimed));
        for group in groups {
            let left = count - rows.len();
            if left == 0 {
                break;
            }
            let held = group & within;
            if held.len() <= left {
                rows |= held;
            } else {
                rows.extend(held.iter().rev().take(left as usize));
            }
        }
        Context::from_bitmap(rows)
    }

    /// The rows of `within` whose times are at most `span` before the newest
    /// time among them; none if none of them has a time. With `before`, every
    /// row of `within` is before it, so the search starts there.
    #[must_use]
    pub fn recent(&self, within: &Context, span: u64, before: Option<i64>) -> Context {
        let Some(newest) = self
            .up_to(before)
            .rev()
            .find(|(_, rows)| !rows.is_disjoint(within.bitmap()))
            .map(|(&time, _)| time)
        else {
            return Context::new();
        };
        let start = newest.saturating_sub(i64::try_from(span).unwrap_or(i64::MAX));
        let rows = self.collect(self.by_time.range(start..=newest));
        &rows & within
    }

    /// Every time group before `before`, or every one.
    fn up_to(&self, before: Option<i64>) -> btree_map::Range<'_, i64, RoaringBitmap> {
        match before {
            Some(before) => self.by_time.range(..before),
            None => self.by_time.range(..),
        }
    }

    /// The `count` latest of a few `rows`, ordered as `latest` orders them, by
    /// their times one by one rather than a walk of every time.
    #[must_use]
    pub fn latest_of(&self, rows: &Context, count: u64) -> Context {
        let mut timed: Vec<(Option<i64>, u32)> = rows
            .iter()
            .map(|row| (self.times.get(&row).copied(), row.0))
            .collect();
        timed.sort_unstable_by(|left, right| right.cmp(left));
        timed
            .into_iter()
            .take(usize::try_from(count).unwrap_or(usize::MAX))
            .map(|(_, row)| RowId(row))
            .collect()
    }

    /// The times of `rows`, ascending, untimed rows first as `i64::MIN`.
    #[must_use]
    pub fn times_of(&self, rows: &Context) -> Vec<i64> {
        let mut times: Vec<i64> = rows
            .iter()
            .map(|row| self.times.get(&row).copied().unwrap_or(i64::MIN))
            .collect();
        times.sort_unstable();
        times
    }

    /// The time a window of the `count` latest rows of `within` starts at, or
    /// `i64::MIN` if it reaches untimed rows.
    #[must_use]
    pub fn latest_from(&self, within: &Context, count: u64) -> i64 {
        let mut held = 0;
        for (&time, rows) in self.by_time.iter().rev() {
            held += rows.intersection_len(within.bitmap());
            if held >= count {
                return time;
            }
        }
        i64::MIN
    }

    /// The newest time among `within`, if any of them has one.
    #[must_use]
    pub fn newest(&self, within: &Context) -> Option<i64> {
        self.by_time
            .iter()
            .rev()
            .find(|(_, rows)| !rows.is_disjoint(within.bitmap()))
            .map(|(&time, _)| time)
    }

    /// Rows with a time from `start` up to, not including, `end`.
    #[must_use]
    pub fn between(&self, start: i64, end: i64) -> Context {
        if start >= end {
            return Context::new();
        }
        self.collect(self.by_time.range(start..end))
    }

    fn collect<'a>(&self, ranges: impl Iterator<Item = (&'a i64, &'a RoaringBitmap)>) -> Context {
        let mut rows = RoaringBitmap::new();
        for (_, bitmap) in ranges {
            rows |= bitmap;
        }
        Context::from_bitmap(rows)
    }
}
