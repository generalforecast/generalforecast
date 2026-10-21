// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Key functions: where a row or query sits in a table's key space.

/// A point in a table's key space.
pub type Key = u64;

/// The splitmix64 finalizer.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The key of a value hashed into the key space, so nearby values scatter.
pub fn hashed(bytes: &[u8]) -> Key {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    mix(h)
}

/// The key of a number placed in order, so a range of numbers is a range of keys.
pub fn ordered(value: f64) -> Key {
    let bits = value.to_bits();
    if bits >> 63 == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_keys_keep_order() {
        let values = [-1e9, -3.5, -0.0, 0.0, 1e-9, 2.0, 7e12];
        let keys: Vec<Key> = values.iter().map(|v| ordered(*v)).collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]), "{keys:?}");
    }

    #[test]
    fn hashed_keys_scatter_neighbors() {
        let a = hashed(&1u64.to_le_bytes());
        let b = hashed(&2u64.to_le_bytes());
        assert!(a.abs_diff(b) > 1 << 40);
        assert_eq!(a, hashed(&1u64.to_le_bytes()));
    }
}
