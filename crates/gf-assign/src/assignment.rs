// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Slices of a key space and the GPUs that hold them.

use crate::key::Key;

/// The keys from `start` up to the next slice's start, and the GPUs holding them.
/// The first holder is the owner, which takes writes; the rest are copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice {
    pub start: Key,
    pub holders: Vec<usize>,
}

/// Slices covering the whole key space in order, the first starting at 0,
/// each held by at least one GPU; `check` says so for one built by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub generation: u64,
    pub slices: Vec<Slice>,
}

/// Keys `[start, end)` changing from one holder to another; `end` of None is the top of the space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transfer {
    pub start: Key,
    pub end: Option<Key>,
    pub from: Option<usize>,
    pub to: usize,
}

impl Assignment {
    /// `count` slices of equal key width, dealt in turn over `gpus` or, when
    /// `contiguous`, in `gpus` runs of neighbors.
    pub fn even(count: usize, gpus: usize, contiguous: bool) -> Self {
        assert!(count > 0 && gpus > 0, "an assignment needs slices and GPUs");
        let width = u64::MAX / count as u64;
        let slices = (0..count)
            .map(|i| Slice {
                start: width * i as u64,
                holders: vec![if contiguous {
                    i * gpus / count
                } else {
                    i % gpus
                }],
            })
            .collect();
        Self {
            generation: 0,
            slices,
        }
    }

    /// Checks the assignment covers the key space over `gpus` GPUs and, when
    /// `contiguous`, that each GPU's slices are one run of neighbors.
    ///
    /// # Errors
    ///
    /// With what is wrong, naming the first slice at fault.
    pub fn check(&self, gpus: usize, contiguous: bool) -> Result<(), String> {
        if self.slices.first().is_none_or(|s| s.start != 0) {
            return Err("an assignment's first slice starts at key 0".into());
        }
        for (i, s) in self.slices.iter().enumerate() {
            if i > 0 && s.start <= self.slices[i - 1].start {
                return Err(format!("slice {i} starts at or before slice {}", i - 1));
            }
            if s.holders.is_empty() {
                return Err(format!("slice {i} has no holder"));
            }
            if let Some(&g) = s.holders.iter().find(|&&g| g >= gpus) {
                return Err(format!("slice {i} is held by GPU {g} of {gpus}"));
            }
            if contiguous && i > 0 && s.holders[0] < self.slices[i - 1].holders[0] {
                return Err(format!("slice {i} breaks the run of GPU {}", s.holders[0]));
            }
        }
        Ok(())
    }

    /// The index of the slice holding `key`.
    ///
    /// # Panics
    ///
    /// If the assignment has no slices.
    pub fn slice_of(&self, key: Key) -> usize {
        assert!(!self.slices.is_empty(), "an assignment needs slices");
        self.slices.partition_point(|s| s.start <= key) - 1
    }

    /// The GPUs holding `key`, owner first.
    pub fn holders(&self, key: Key) -> &[usize] {
        &self.slices[self.slice_of(key)].holders
    }

    /// The key after slice `i`'s last, or None for the last slice.
    pub fn end(&self, i: usize) -> Option<Key> {
        self.slices.get(i + 1).map(|s| s.start)
    }

    /// What must move to go from this assignment to `next`: each range whose
    /// holders gain a GPU, with a GPU that held it before as the source.
    pub fn transfers(&self, next: &Assignment) -> Vec<Transfer> {
        let mut cuts: Vec<Key> = self
            .slices
            .iter()
            .chain(&next.slices)
            .map(|s| s.start)
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        let mut out = Vec::new();
        for (i, &start) in cuts.iter().enumerate() {
            let end = cuts.get(i + 1).copied();
            let before = self.holders(start);
            for &to in next.holders(start) {
                if !before.contains(&to) {
                    out.push(Transfer {
                        start,
                        end,
                        from: before.first().copied(),
                        to,
                    });
                }
            }
        }
        out
    }

    /// Each GPU's slice count.
    pub fn slices_per_gpu(&self, gpus: usize) -> Vec<usize> {
        let mut counts = vec![0; gpus];
        for s in &self.slices {
            for &g in &s.holders {
                counts[g] += 1;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_slices_cover_the_space() {
        let a = Assignment::even(8, 4, false);
        assert_eq!(a.slice_of(0), 0);
        assert_eq!(a.slice_of(u64::MAX), 7);
        assert_eq!(a.slice_of(a.slices[3].start), 3);
        assert_eq!(a.slice_of(a.slices[3].start - 1), 2);
        assert_eq!(a.slices_per_gpu(4), vec![2, 2, 2, 2]);
        let c = Assignment::even(8, 4, true);
        let holders: Vec<usize> = c.slices.iter().map(|s| s.holders[0]).collect();
        assert_eq!(holders, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn check_names_the_slice_at_fault() {
        let a = Assignment::even(4, 2, true);
        assert_eq!(a.check(2, true), Ok(()));
        let mut gone = a.clone();
        gone.slices[1].holders = vec![2];
        assert_eq!(
            gone.check(2, false).unwrap_err(),
            "slice 1 is held by GPU 2 of 2"
        );
        let mut broken = a.clone();
        broken.slices.swap(0, 3);
        assert!(broken.check(2, false).unwrap_err().contains("key 0"));
        let mut mixed = a.clone();
        mixed.slices[0].holders = vec![1];
        assert_eq!(mixed.check(2, false), Ok(()));
        assert_eq!(
            mixed.check(2, true).unwrap_err(),
            "slice 1 breaks the run of GPU 0"
        );
        let mut empty = a;
        empty.slices[2].holders.clear();
        assert_eq!(empty.check(2, false).unwrap_err(), "slice 2 has no holder");
    }

    #[test]
    fn transfers_name_only_what_changed() {
        let a = Assignment::even(4, 2, false);
        let mut b = a.clone();
        b.slices[1].holders = vec![0];
        b.slices[2].holders.push(1);
        let t = a.transfers(&b);
        assert_eq!(t.len(), 2);
        assert_eq!(
            t[0],
            Transfer {
                start: a.slices[1].start,
                end: Some(a.slices[2].start),
                from: Some(1),
                to: 0
            }
        );
        assert_eq!(t[1].from, Some(0));
        assert_eq!(t[1].to, 1);
    }
}
