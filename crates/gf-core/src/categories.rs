// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! How a table's categorical features place rows for retrieval.
//!
//! By default a category's code is standardized like any number, so codes
//! that happen to be close count as similar. One-hot encoding gives each
//! column's `ONE_HOT_CATEGORIES` most frequent categories a dimension of
//! their own and every other category one more, with a missing value at
//! none. Target encoding places a category by the targets of its rows: its
//! mean target, standardized, for a regressor's table, or its rate of each
//! class for a classifier's, each pulled toward the table's by
//! `SMOOTHING_ROWS` rows of it, so a rare category is not placed by a label
//! or two; an unseen or missing category takes the table's. Both are fit on
//! the rows the retrieval scaling is fit on, which are table rows only, never
//! a query's, and only when the scaling is: as the table's rows double, up to
//! `scale_rows`. Labels that change in between move nothing until then.

use std::collections::HashMap;

use crate::node::{Targets, Task};

/// Categories a column's one-hot block names, its most frequent; the rest
/// share one more dimension.
pub const ONE_HOT_CATEGORIES: usize = 15;

/// Each one-hot dimension's value, so two different categories lie
/// sqrt(2) times it apart: one deviation, as far as a standardized number
/// one deviation from another.
const ONE_HOT_WEIGHT: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Rows of the table's own targets each category's target encoding is
/// averaged with, an m-estimate: a category of n rows takes
/// (n * its own + m * the table's) / (n + m).
pub const SMOOTHING_ROWS: f32 = 10.0;

/// How categorical features place rows for retrieval.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CategoryEncoding {
    /// As their codes, standardized.
    #[default]
    Codes,
    /// One-hot over each column's most frequent categories.
    OneHot,
    /// As the targets of each category's rows.
    Target,
}

impl CategoryEncoding {
    /// The encoding's name, as clients and logs spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::OneHot => "one_hot",
            Self::Target => "target",
        }
    }

    /// Dimensions one categorical feature takes for a table of `task`.
    #[must_use]
    pub fn width(self, task: Task) -> usize {
        match (self, task) {
            (Self::Codes, _) | (Self::Target, Task::Values) => 1,
            (Self::OneHot, _) => ONE_HOT_CATEGORIES + 1,
            (Self::Target, Task::Classes(classes)) => classes as usize,
        }
    }
}

impl std::str::FromStr for CategoryEncoding {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        [Self::Codes, Self::OneHot, Self::Target]
            .into_iter()
            .find(|encoding| encoding.name() == name)
            .ok_or_else(|| {
                format!("retrieval_categoricals is codes, one_hot or target, not {name:?}")
            })
    }
}

/// One categorical feature's vectors: one for each known code, then one for
/// any other code.
#[derive(Clone, Debug, PartialEq)]
pub struct Lookup {
    pub codes: Vec<f32>,
    /// Row-major, a block per code in order, then the block for other codes.
    pub vectors: Vec<f32>,
    /// The block a missing value takes.
    missing: Vec<f32>,
    index: HashMap<u32, usize>,
}

impl Lookup {
    /// A lookup of `codes` to `vectors`, as `encoding` fit or a layout keeps them.
    #[must_use]
    pub fn new(encoding: CategoryEncoding, codes: Vec<f32>, vectors: Vec<f32>) -> Self {
        let width = vectors.len() / (codes.len() + 1);
        let other = vectors[codes.len() * width..].to_vec();
        let missing = match encoding {
            CategoryEncoding::OneHot => vec![0.0; width],
            CategoryEncoding::Codes | CategoryEncoding::Target => other,
        };
        let index = codes
            .iter()
            .enumerate()
            .map(|(block, &code)| (code as u32, block))
            .collect();
        Self {
            codes,
            vectors,
            missing,
            index,
        }
    }

    /// The lookup `encoding` fits on a column's `values` and their rows'
    /// `targets`, for a table of `task`.
    ///
    /// # Panics
    ///
    /// For `Codes`, which looks nothing up.
    #[must_use]
    pub fn fit(encoding: CategoryEncoding, values: &[f32], targets: &Targets, task: Task) -> Self {
        let mut rows: HashMap<u32, Vec<usize>> = HashMap::new();
        for (row, &value) in values.iter().enumerate() {
            if value.is_finite() {
                rows.entry(value as u32).or_default().push(row);
            }
        }
        // By frequency, then code, so a fit does not depend on hashing.
        let mut codes: Vec<(u32, Vec<usize>)> = rows.into_iter().collect();
        codes.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        let (codes, vectors) = match encoding {
            CategoryEncoding::Codes => unreachable!("codes are standardized, not looked up"),
            CategoryEncoding::OneHot => {
                codes.truncate(ONE_HOT_CATEGORIES);
                let width = ONE_HOT_CATEGORIES + 1;
                let mut vectors = vec![0.0; (codes.len() + 1) * width];
                for block in 0..codes.len() {
                    vectors[block * width + block] = ONE_HOT_WEIGHT;
                }
                vectors[codes.len() * width + ONE_HOT_CATEGORIES] = ONE_HOT_WEIGHT;
                (codes, vectors)
            }
            CategoryEncoding::Target => {
                let all: Vec<usize> = (0..values.len()).collect();
                let encode = TargetEncoder::new(targets, task, &all);
                let (table, _) = encode.block(&all);
                let mut vectors: Vec<f32> = codes
                    .iter()
                    .flat_map(|(_, rows)| {
                        let (own, labeled) = encode.block(rows);
                        own.iter()
                            .zip(&table)
                            .map(|(own, table)| {
                                (labeled * own + SMOOTHING_ROWS * table)
                                    / (labeled + SMOOTHING_ROWS)
                            })
                            .collect::<Vec<f32>>()
                    })
                    .collect();
                vectors.extend(table);
                (codes, vectors)
            }
        };
        let codes = codes.into_iter().map(|(code, _)| code as f32).collect();
        Self::new(encoding, codes, vectors)
    }

    /// Appends the block `value` takes to `vector`.
    pub fn extend(&self, value: f32, vector: &mut Vec<f32>) {
        if !value.is_finite() {
            vector.extend_from_slice(&self.missing);
            return;
        }
        let width = self.missing.len();
        let block = self
            .index
            .get(&(value as u32))
            .copied()
            .unwrap_or(self.codes.len());
        vector.extend_from_slice(&self.vectors[block * width..(block + 1) * width]);
    }
}

/// Places a set of rows by their targets.
struct TargetEncoder<'a> {
    targets: &'a Targets,
    task: Task,
    mean: f64,
    deviation: f64,
}

impl<'a> TargetEncoder<'a> {
    fn new(targets: &'a Targets, task: Task, all: &[usize]) -> Self {
        let values: Vec<f64> = match targets {
            Targets::Values(values) => all
                .iter()
                .map(|&row| f64::from(values[row]))
                .filter(|value| value.is_finite())
                .collect(),
            Targets::Classes(_) => Vec::new(),
        };
        let count = values.len().max(1) as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / count;
        let deviation = if variance > 0.0 { variance.sqrt() } else { 1.0 };
        Self {
            targets,
            task,
            mean,
            deviation,
        }
    }

    /// The rows' mean target, standardized, or their rate of each class, and
    /// how many of them have a target.
    fn block(&self, rows: &[usize]) -> (Vec<f32>, f32) {
        match (self.targets, self.task) {
            (Targets::Classes(labels), Task::Classes(classes)) => {
                let mut counts = vec![0.0f32; classes as usize];
                let mut seen = 0.0f32;
                for &row in rows {
                    if let Some(count) = usize::try_from(labels[row])
                        .ok()
                        .and_then(|label| counts.get_mut(label))
                    {
                        *count += 1.0;
                        seen += 1.0;
                    }
                }
                let rates = counts.iter().map(|count| count / seen.max(1.0)).collect();
                (rates, seen)
            }
            (Targets::Values(values), _) => {
                let finite: Vec<f64> = rows
                    .iter()
                    .map(|&row| f64::from(values[row]))
                    .filter(|value| value.is_finite())
                    .collect();
                let mean = if finite.is_empty() {
                    self.mean
                } else {
                    finite.iter().sum::<f64>() / finite.len() as f64
                };
                let labeled = finite.len() as f32;
                (vec![((mean - self.mean) / self.deviation) as f32], labeled)
            }
            (Targets::Classes(_), Task::Values) => (vec![0.0], 0.0),
        }
    }
}
