// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! How a table's slices are balanced.

use serde::{Deserialize, Serialize};

/// Whether a GPU may hold slices from anywhere in the key space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Any slice on any GPU, moved by Slicer's weighted moves. Balances best.
    #[default]
    Free,
    /// Each GPU holds one range of keys, and balancing shifts the cuts between
    /// neighbors. Keeps nearby keys on one GPU.
    Contiguous,
}

/// The settings of a table's rebalancing, with Slicer's defaults where it has them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    pub mode: Mode,
    /// The share of a table's rows one round may move.
    pub churn_budget: f64,
    /// The share of rows merging cold slices may move.
    pub merge_budget: f64,
    /// Slices a GPU keeps before cold ones merge.
    pub min_slices_per_gpu: usize,
    /// Slices a GPU may reach by splitting hot ones.
    pub max_slices_per_gpu: usize,
    /// A slice at least this many times the mean slice load splits.
    pub hot_split: f64,
    /// An unsplittable slice above this share of a GPU's mean load gains a copy.
    pub copy_above: f64,
    /// Most GPUs one slice may be held on; 1 turns copies off.
    pub max_copies: usize,
    /// No moves while the busiest GPU's load over the mean is below this.
    pub imbalance_floor: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            mode: Mode::Free,
            churn_budget: 0.03,
            merge_budget: 0.01,
            min_slices_per_gpu: 50,
            max_slices_per_gpu: 150,
            hot_split: 2.0,
            copy_above: 0.25,
            max_copies: 8,
            imbalance_floor: 1.2,
        }
    }
}
