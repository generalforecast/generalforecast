// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Which shard serves a context: the rules gf-server's front and the
//! local engine both route by, so a context's table lives on one shard.

/// The shard of `count` that serves the context `fingerprint` names: its
/// first eight bytes, little-endian, modulo the count.
///
/// # Panics
///
/// If `count` is zero.
#[must_use]
pub fn shard_of(fingerprint: &[u8; 32], count: usize) -> usize {
    assert!(count > 0, "a context needs a shard to route to");
    let mut first = [0u8; 8];
    first.copy_from_slice(&fingerprint[..8]);
    (u64::from_le_bytes(first) % count as u64) as usize
}

/// The shard a new context goes to: the least busy of `loads`, ties going
/// to the first from `shard_of` on, so idle shards place it by
/// `shard_of` alone.
///
/// # Panics
///
/// If `loads` is empty.
#[must_use]
pub fn least_busy(fingerprint: &[u8; 32], loads: &[usize]) -> usize {
    let first = shard_of(fingerprint, loads.len());
    (0..loads.len())
        .map(|offset| (first + offset) % loads.len())
        .min_by_key(|&shard| loads[shard])
        .expect("a context needs a shard to route to")
}

#[cfg(test)]
mod tests {
    use super::{least_busy, shard_of};

    #[test]
    fn contexts_route_by_their_fingerprints_first_eight_bytes() {
        let mut fingerprint = [0xff; 32];
        fingerprint[..8].copy_from_slice(&10u64.to_le_bytes());
        assert_eq!(shard_of(&fingerprint, 4), 2);
        assert_eq!(shard_of(&fingerprint, 1), 0);
        fingerprint[8] = 0;
        assert_eq!(shard_of(&fingerprint, 4), 2);
    }

    #[test]
    fn a_new_context_goes_to_the_least_busy_shard_ties_by_its_fingerprint() {
        let mut fingerprint = [0u8; 32];
        fingerprint[..8].copy_from_slice(&2u64.to_le_bytes());
        assert_eq!(least_busy(&fingerprint, &[0, 0, 0, 0]), 2);
        assert_eq!(least_busy(&fingerprint, &[1, 0, 1, 0]), 3);
        assert_eq!(least_busy(&fingerprint, &[0, 3, 2, 1]), 0);
        assert_eq!(least_busy(&fingerprint, &[5]), 0);
    }
}
