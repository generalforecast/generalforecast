// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Padded sizes for compiled model calls.

use std::num::NonZeroUsize;

/// The padded size for `rows` rows: `minimum` doubled until it fits.
///
/// Padding to a few sizes keeps the number of compiled programs small.
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
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bucket(rows: usize, minimum: usize) -> usize {
        bucket_size(rows, NonZeroUsize::new(minimum).unwrap())
    }

    #[test]
    fn small_tables_take_the_minimum() {
        assert_eq!(bucket(0, 8), 8);
        assert_eq!(bucket(8, 8), 8);
    }

    #[test]
    fn larger_tables_double_the_minimum_until_they_fit() {
        assert_eq!(bucket(9, 8), 16);
        assert_eq!(bucket(1000, 8), 1024);
        assert_eq!(bucket(7, 6), 12);
    }
}
