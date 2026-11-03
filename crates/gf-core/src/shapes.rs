// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Padded sizes for compiled model calls.

use std::num::NonZeroUsize;

/// Past this many rows, padded sizes step by a quarter of the power of two
/// below them rather than doubling.
pub const FINE_BUCKETS_PAST: usize = 1 << 14;
/// Past this many rows, padded sizes step by an eighth of the power of two
/// below them.
pub const FINER_BUCKETS_PAST: usize = 1 << 18;

/// The padded size for `rows` rows: `minimum` doubled until it fits, and past
/// `FINE_BUCKETS_PAST` rows, the power of two below `rows` and quarters of
/// it, so a large context pads by at most a quarter rather than doubling;
/// past `FINER_BUCKETS_PAST`, eighths, so by at most an eighth.
///
/// Padding to a few sizes keeps the number of compiled programs small: four
/// to each doubling past `FINE_BUCKETS_PAST` and eight past
/// `FINER_BUCKETS_PAST`, where a context's padding costs more memory and work
/// than a program's compile.
///
/// # Panics
///
/// If the padded size would overflow `usize`.
#[must_use]
pub fn bucket_size(rows: usize, minimum: NonZeroUsize) -> usize {
    let mut size = minimum.get();
    while size < rows {
        size = size.checked_mul(2).expect("padded size overflows usize");
    }
    let below = size / 2;
    let step = below / if below >= FINER_BUCKETS_PAST { 8 } else { 4 };
    if below >= FINE_BUCKETS_PAST && step >= minimum.get() && rows > below {
        return below + (rows - below).div_ceil(step) * step;
    }
    size
}

/// The most slots a block of a context's rows takes, for rows no wider than
/// a slot.
pub const MAX_BLOCK_SLOTS: u64 = 16;
/// A block of rows no wider than a slot wastes at most this share of its
/// bytes, one in sixteen, where `MAX_BLOCK_SLOTS` allow it.
pub const BLOCK_WASTE: u64 = 16;

/// The rows a block of a context's cache holds and the slots it takes, for
/// rows of `row_bytes` bytes in slots of `slot_bytes`, as the worker packs
/// them.
///
/// A row wider than a slot takes whole slots of its own. Otherwise a block
/// is the fewest slots whose rows leave at most a `BLOCK_WASTE`th of them
/// empty, up to `MAX_BLOCK_SLOTS`, or else the block of those that leaves the
/// least empty: rows of a little more than half a slot would otherwise take
/// a slot each and leave almost half of it empty. A row may then cross a
/// slot's end within its block.
///
/// # Panics
///
/// If `slot_bytes` is zero.
#[must_use]
pub fn block_shape(row_bytes: u64, slot_bytes: u64) -> (u64, u64) {
    assert!(slot_bytes > 0, "a slot holds bytes");
    let row = row_bytes.max(1);
    if row > slot_bytes {
        return (1, row.div_ceil(slot_bytes));
    }
    let mut best = (slot_bytes / row, 1);
    let mut best_left = (slot_bytes % row, slot_bytes);
    for slots in 1..=MAX_BLOCK_SLOTS {
        let bytes = slots * slot_bytes;
        let rows = bytes / row;
        let left = bytes - rows * row;
        if left * BLOCK_WASTE <= bytes {
            return (rows, slots);
        }
        // Less left empty for its size than the best so far.
        if u128::from(left) * u128::from(best_left.1) < u128::from(best_left.0) * u128::from(bytes)
        {
            best = (rows, slots);
            best_left = (left, bytes);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_leave_little_of_their_slots_empty() {
        let slot = 1 << 20;
        // Narrow rows fill one slot, as many as fit.
        assert_eq!(block_shape(13_312, slot), (78, 1));
        assert_eq!(block_shape(1, slot), (slot, 1));
        // A row of a little more than half a slot shares a block of slots.
        assert_eq!(block_shape(540_672, slot), (11, 6));
        assert_eq!(block_shape(slot / 2, slot), (2, 1));
        // A row wider than a slot takes whole slots of its own.
        assert_eq!(block_shape(slot + 1, slot), (1, 2));
        for row in [1, 7, 1000, 300_000, 540_672, 700_001, slot] {
            let (rows, slots) = block_shape(row, slot);
            let bytes = slots * slot;
            assert!(
                rows >= 1 && rows * row <= bytes && slots <= MAX_BLOCK_SLOTS,
                "{row}"
            );
        }
    }

    fn bucket(rows: usize, minimum: usize) -> usize {
        bucket_size(rows, NonZeroUsize::new(minimum).unwrap())
    }

    #[test]
    fn large_tables_pad_by_at_most_a_quarter() {
        assert_eq!(bucket(16_384, 8), 16_384);
        assert_eq!(bucket(16_385, 8), 20_480);
        assert_eq!(bucket(19_451, 8), 20_480);
        assert_eq!(bucket(28_673, 8), 32_768);
        assert_eq!(bucket(32_769, 8), 40_960);
        assert_eq!(bucket(200_000, 8), 229_376);
        // Past 262,144 rows, by an eighth.
        assert_eq!(bucket(262_145, 8), 294_912);
        assert_eq!(bucket(793_076, 8), 851_968);
        for rows in [16_385, 19_451, 50_000, 131_073, 793_076] {
            let size = bucket(rows, 8);
            assert!(
                rows <= size && 4 * size <= 5 * rows + 4 * FINE_BUCKETS_PAST,
                "{rows}"
            );
            // A padded size pads to itself.
            assert_eq!(bucket(size, 8), size);
        }
    }

    #[test]
    fn tables_take_the_minimum_doubled_until_they_fit() {
        assert_eq!(bucket(0, 8), 8);
        assert_eq!(bucket(8, 8), 8);
        assert_eq!(bucket(9, 8), 16);
        assert_eq!(bucket(1000, 8), 1024);
        assert_eq!(bucket(7, 6), 12);
    }
}
