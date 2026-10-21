// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Contexts: sets of rows a model conditions on, and their fingerprints.

use std::iter::Sum;
use std::ops::{Add, BitAnd, BitOr, Sub};

use roaring::RoaringBitmap;

use crate::table::{RowId, Version};

/// A set of rows, stored as a compressed bitmap.
///
/// Retrieval, filters and time cutoffs all produce contexts, so combining them
/// is set algebra: `&a | &b`, `&a & &b` and `&a - &b`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context {
    rows: RoaringBitmap,
}

impl Context {
    /// An empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn from_bitmap(rows: RoaringBitmap) -> Self {
        Self { rows }
    }

    pub(crate) fn bitmap(&self) -> &RoaringBitmap {
        &self.rows
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.rows.len()
    }

    /// Whether the context has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Whether the context holds `row`.
    #[must_use]
    pub fn contains(&self, row: RowId) -> bool {
        self.rows.contains(row.0)
    }

    /// Whether the contexts share a row.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        !self.rows.is_disjoint(&other.rows)
    }

    /// The rows in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = RowId> + '_ {
        self.rows.iter().map(RowId)
    }

    /// The blocks of `1 << bits` row ids that hold rows, in increasing order.
    pub(crate) fn blocks(&self, bits: u32) -> impl Iterator<Item = u32> + '_ {
        let mut last = None;
        self.rows
            .iter()
            .map(move |row| row >> bits)
            .filter(move |&block| last.replace(block) != Some(block))
    }
}

impl FromIterator<RowId> for Context {
    fn from_iter<I: IntoIterator<Item = RowId>>(rows: I) -> Self {
        Self::from_bitmap(rows.into_iter().map(|row| row.0).collect())
    }
}

impl std::ops::BitOrAssign<&Context> for Context {
    fn bitor_assign(&mut self, other: &Context) {
        self.rows |= &other.rows;
    }
}

impl BitOr for &Context {
    type Output = Context;

    fn bitor(self, other: Self) -> Context {
        Context::from_bitmap(&self.rows | &other.rows)
    }
}

impl BitAnd for &Context {
    type Output = Context;

    fn bitand(self, other: Self) -> Context {
        Context::from_bitmap(&self.rows & &other.rows)
    }
}

impl Sub for &Context {
    type Output = Context;

    fn sub(self, other: Self) -> Context {
        Context::from_bitmap(&self.rows - &other.rows)
    }
}

/// An order-independent hash of a set of rows at their versions.
///
/// It is the wrapping sum of each row's hash, so it identifies a context for a
/// cache key, updates in O(1) when one row changes, and adds across disjoint
/// contexts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(u128);

impl Fingerprint {
    /// The fingerprint of no rows.
    pub const EMPTY: Self = Self(0);

    /// The fingerprint's bits, for keying state outside Rust.
    #[must_use]
    pub fn bits(self) -> u128 {
        self.0
    }

    /// The fingerprint mixed with a salt, for a context that differs by
    /// more than its rows, such as the origin a series window ends at.
    #[must_use]
    pub fn salted(self, salt: u64) -> Self {
        let high = mix((self.0 >> 64) as u64 ^ salt);
        let low = mix(self.0 as u64 ^ mix(salt));
        Self((u128::from(high) << 64) | u128::from(low))
    }

    /// The fingerprint of one row at one version.
    #[must_use]
    pub fn of(row: RowId, version: Version) -> Self {
        let lane = |seed: u64| mix(mix(u64::from(row.0) ^ seed) ^ version.0);
        let high = u128::from(lane(0x9e37_79b9_7f4a_7c15));
        let low = u128::from(lane(0xd1b5_4a32_d192_ed03));
        Self((high << 64) | low)
    }
}

/// The splitmix64 finalizer: a fast, well-mixed 64-bit hash.
pub(crate) fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl Add for Fingerprint {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }
}

impl Sub for Fingerprint {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0.wrapping_sub(other.0))
    }
}

impl Sum for Fingerprint {
    fn sum<I: Iterator<Item = Self>>(fingerprints: I) -> Self {
        fingerprints.fold(Self::EMPTY, Add::add)
    }
}
