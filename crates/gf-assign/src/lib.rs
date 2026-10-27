// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Assigns slices of a table's key space to GPUs by measured load.
//!
//! Every row and query maps to a 64-bit key (`key`), a slice is a range of
//! keys (`assignment`), and `rebalance` moves, splits, merges and copies
//! slices under a `Policy`, as Slicer does (Adya et al., OSDI 2016).

#![forbid(unsafe_code)]

pub mod assignment;
pub mod key;
pub mod policy;
pub mod rebalance;

pub use assignment::{Assignment, Slice, Transfer};
pub use key::Key;
pub use policy::{Mode, Policy};
pub use rebalance::{Rebalance, SliceLoad, imbalance, rebalance};
