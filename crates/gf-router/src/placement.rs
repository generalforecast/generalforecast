//! Rendezvous hashing: every table ranks every node, and the top node serves it.
//!
//! Adding a node moves only the tables it now ranks first, and removing one
//! moves only the tables it served, so a node set change disturbs as little as
//! possible. The hash is fixed, so every router agrees on the placement.

/// The nodes tables are placed on, in configuration order.
#[derive(Clone, Debug)]
pub struct Placement {
    nodes: Vec<String>,
}

/// FNV-1a over bytes, then the splitmix64 finalizer for well-mixed bits.
fn hash(bytes: &[u8]) -> u64 {
    let mut value: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        value ^= u64::from(byte);
        value = value.wrapping_mul(0x0000_0100_0000_01b3);
    }
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl Placement {
    /// A placement over named nodes.
    #[must_use]
    pub fn new(nodes: Vec<String>) -> Self {
        Self { nodes }
    }

    /// Node indices for a table, from the one that serves it down.
    #[must_use]
    pub fn ranked(&self, table: &str) -> Vec<usize> {
        let mut scores: Vec<(u64, usize)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (hash(format!("{node}\0{table}").as_bytes()), index))
            .collect();
        scores.sort_unstable_by(|left, right| right.cmp(left));
        scores.into_iter().map(|(_, index)| index).collect()
    }

    /// The node that serves a table.
    ///
    /// # Panics
    ///
    /// If there are no nodes.
    #[must_use]
    pub fn primary(&self, table: &str) -> usize {
        self.ranked(table)[0]
    }
}
