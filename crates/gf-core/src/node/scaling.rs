// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The retrieval space: each row's standardized vector, its strata, and the
//! column moments drift is measured by.

use super::*;

/// Per-column standardization for retrieval vectors, fit on a table's first
/// batch and again each time its rows double and at each refit.
#[derive(Clone, Debug)]
pub(super) struct Standardizer {
    /// Leading values of a row it leaves out: the model's, when the table
    /// has retrieval values.
    pub(super) skip: usize,
    pub(super) center: Vec<f32>,
    pub(super) scale: Vec<f32>,
    /// Each encoded categorical feature's lookup, in feature order; none
    /// when categoricals keep their codes.
    pub(super) lookups: Vec<(usize, Lookup)>,
}

/// Each column's mean and standard deviation over its finite values, in one
/// pass over the rows, as Welford's update keeps them; 0 for a column without any.
pub(super) fn moments(features: &[f32], width: usize) -> (Vec<f64>, Vec<f64>) {
    let mut counts = vec![0u64; width];
    let mut means = vec![0.0; width];
    let mut squares = vec![0.0; width];
    for row in features.chunks_exact(width.max(1)) {
        for (column, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let value = f64::from(value);
            counts[column] += 1;
            let delta = value - means[column];
            means[column] += delta / counts[column] as f64;
            squares[column] += delta * (value - means[column]);
        }
    }
    let deviations = squares
        .iter()
        .zip(&counts)
        .map(|(&squares, &count)| {
            if count == 0 {
                0.0
            } else {
                (squares / count as f64).sqrt()
            }
        })
        .collect();
    (means, deviations)
}

impl Standardizer {
    /// Standardization of `spec`'s features over row-major `features`, with
    /// its categorical features encoded from them and their rows' `targets`,
    /// or for a table with retrieval values, those values as they are.
    pub(super) fn fit(features: &[f32], targets: &Targets, spec: &TableSpec) -> Self {
        if spec.retrieval > 0 {
            let ones = vec![1.0; spec.retrieval];
            return Self::of(spec, vec![0.0; spec.retrieval], ones, Vec::new());
        }
        let (means, deviations) = moments(features, spec.features);
        let lookups = match spec.categories {
            CategoryEncoding::Codes => Vec::new(),
            encoding => spec
                .categorical
                .iter()
                .map(|&feature| {
                    let values: Vec<f32> = features
                        .chunks(spec.features)
                        .map(|row| row[feature])
                        .collect();
                    (feature, Lookup::fit(encoding, &values, targets, spec.task))
                })
                .collect(),
        };
        Self {
            skip: 0,
            center: means.iter().map(|&mean| mean as f32).collect(),
            scale: deviations
                .iter()
                .map(|&deviation| {
                    if deviation > 0.0 {
                        deviation as f32
                    } else {
                        1.0
                    }
                })
                .collect(),
            lookups,
        }
    }

    /// A standardization of `spec`'s rows by `center`, `scale` and `lookups`.
    pub(super) fn of(
        spec: &TableSpec,
        center: Vec<f32>,
        scale: Vec<f32>,
        lookups: Vec<(usize, Lookup)>,
    ) -> Self {
        Self {
            skip: spec.model_features() * usize::from(spec.retrieval > 0),
            center,
            scale,
            lookups,
        }
    }

    /// A row's retrieval vector: standardized, with missing values at 0, and
    /// each encoded categorical feature's block in its place.
    pub(super) fn vector(&self, row: &[f32]) -> Vec<f32> {
        let mut lookups = self.lookups.iter().peekable();
        let mut vector = Vec::with_capacity(row.len() - self.skip);
        let values = row[self.skip..]
            .iter()
            .zip(self.center.iter().zip(&self.scale));
        for (feature, (&value, (center, scale))) in values.enumerate() {
            if let Some((_, lookup)) = lookups.next_if(|(at, _)| *at == feature) {
                lookup.extend(value, &mut vector);
                continue;
            }
            let standardized = (value - center) / scale;
            vector.push(if standardized.is_finite() {
                standardized
            } else {
                0.0
            });
        }
        vector
    }
}

/// A table's rows in its retrieval space, which its index reads its vectors from.
pub(super) struct Scaled<'a> {
    pub(super) rows: &'a RowStore,
    pub(super) standardizer: Option<&'a Standardizer>,
}

impl Vectors for Scaled<'_> {
    fn vector(&self, row: RowId) -> Result<Vec<f32>, RowsError> {
        let features = self.rows.features(row)?;
        Ok(self
            .standardizer
            .expect("rows are scaled before they are placed")
            .vector(&features))
    }
}

/// Rows' strata: a row's class, or the bin among `edges` its value falls in.
pub(super) struct Labels<'a> {
    pub(super) rows: &'a RowStore,
    pub(super) task: Task,
    pub(super) edges: &'a [f32],
}

impl Strata for Labels<'_> {
    fn stratum(&self, row: RowId) -> u32 {
        let bits = self.rows.target(row).unwrap_or(0);
        match self.task {
            Task::Classes(_) => bits,
            Task::Values => {
                let value = f32::from_bits(bits);
                self.edges.partition_point(|&edge| edge <= value) as u32
            }
        }
    }
}

/// Per-column sums over live rows' finite values, for each column's mean.
///
/// Recounted from the rows at each fit of every row, so the rounding of
/// adding and removing values never builds up past one such fit.
#[derive(Debug)]
pub(super) struct Columns {
    pub(super) sums: Vec<f64>,
    pub(super) counts: Vec<u64>,
}

impl Columns {
    pub(super) fn new(width: usize) -> Self {
        Self {
            sums: vec![0.0; width],
            counts: vec![0; width],
        }
    }

    /// The sums of row-major `features`.
    pub(super) fn of(features: &[f32], width: usize) -> Self {
        let mut columns = Self::new(width);
        for row in features.chunks(width) {
            columns.add(row, 1.0);
        }
        columns
    }

    /// Adds a row's model values, or with `sign` -1 removes them.
    pub(super) fn add(&mut self, row: &[f32], sign: f64) {
        for (column, &value) in row.iter().take(self.sums.len()).enumerate() {
            if value.is_finite() {
                self.sums[column] += sign * f64::from(value);
                self.counts[column] = if sign > 0.0 {
                    self.counts[column] + 1
                } else {
                    self.counts[column].saturating_sub(1)
                };
            }
        }
    }

    pub(super) fn mean(&self, column: usize) -> f64 {
        self.sums[column] / self.counts[column].max(1) as f64
    }
}

/// What drift is measured against: the state's rows and columns at its fit.
#[derive(Clone, Debug)]
pub(super) struct Fitted {
    /// The served rows the fit drew on: those it read, or for a fit on a
    /// sample, those the sample was drawn from; labeled ones, and unlabeled
    /// ones for a model that reads them.
    pub(super) rows: u64,
    pub(super) means: Vec<f64>,
    pub(super) deviations: Vec<f64>,
}
