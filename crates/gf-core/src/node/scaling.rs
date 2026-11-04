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
        let mut vector = Vec::with_capacity(row.len() - self.skip);
        self.extend(row, &mut vector);
        vector
    }

    /// Appends a row's retrieval vector, as `vector` gives it, to `vector`.
    pub(super) fn extend(&self, row: &[f32], vector: &mut Vec<f32>) {
        let mut lookups = self.lookups.iter().peekable();
        let values = row[self.skip..]
            .iter()
            .zip(self.center.iter().zip(&self.scale));
        for (feature, (&value, (center, scale))) in values.enumerate() {
            if let Some((_, lookup)) = lookups.next_if(|(at, _)| *at == feature) {
                lookup.extend(value, vector);
                continue;
            }
            let standardized = (value - center) / scale;
            vector.push(if standardized.is_finite() {
                standardized
            } else {
                0.0
            });
        }
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

    /// Many rows are read from a snapshot of the store and scaled on every
    /// core, each thread writing its share of the rows' places in `into`.
    fn extend_vectors(&self, rows: &[RowId], into: &mut Vec<f32>) -> Result<(), RowsError> {
        let standardizer = self
            .standardizer
            .expect("rows are scaled before they are placed");
        let threads = std::thread::available_parallelism()
            .map_or(1, NonZeroUsize::get)
            .min(rows.len() / PARALLEL_SCALED_ROWS);
        if threads <= 1 {
            let (features, _) = self.rows.gather(rows)?;
            let width = features.len() / rows.len().max(1);
            for row in features.chunks_exact(width.max(1)) {
                standardizer.extend(row, into);
            }
            return Ok(());
        }
        let snapshot = self.rows.snapshot();
        let dimensions = self.vector(rows[0])?.len();
        let start = into.len();
        into.resize(start + rows.len() * dimensions, 0.0);
        let share = rows.len().div_ceil(threads);
        let read: Vec<Result<(), RowsError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = rows
                .chunks(share)
                .zip(into[start..].chunks_mut(share * dimensions))
                .map(|(rows, place)| {
                    let snapshot = &snapshot;
                    scope.spawn(move || {
                        let (mut features, mut vectors) = (Vec::new(), Vec::new());
                        let batches = rows.chunks(SCALED_BATCH_ROWS);
                        for (batch, place) in
                            batches.zip(place.chunks_mut(SCALED_BATCH_ROWS * dimensions))
                        {
                            features.clear();
                            vectors.clear();
                            snapshot.extend_features(batch, &mut features)?;
                            let width = features.len() / batch.len();
                            for row in features.chunks_exact(width.max(1)) {
                                standardizer.extend(row, &mut vectors);
                            }
                            place.copy_from_slice(&vectors);
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("a scaling thread does not panic"))
                .collect()
        });
        read.into_iter().collect()
    }
}

/// Rows each thread scales at least, below which `Scaled::extend_vectors`
/// reads on the caller's thread alone.
const PARALLEL_SCALED_ROWS: usize = 8192;

/// Rows a scaling thread reads and scales at a time.
const SCALED_BATCH_ROWS: usize = 1024;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn many_rows_scale_on_many_threads_as_one_by_one() {
        let (count, width) = (40_000usize, 6usize);
        let memory = RowMemory::unbounded();
        let mut store = RowStore::new(&memory, width);
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let features: Vec<f32> = (0..count * width)
            .map(|at| match at % width {
                // A categorical column, and a column with missing values.
                2 => (next() * 5.0).floor(),
                4 if next() < 0.2 => f32::NAN,
                _ => next() * 8.0 - 4.0,
            })
            .collect();
        let keys: Vec<u128> = (0..count as u128).collect();
        store.push(&features, &vec![0; count], &keys).unwrap();
        let labels = Targets::Classes((0..count as i32).map(|row| row % 3).collect());
        let categories: Vec<f32> = features.chunks(width).map(|row| row[2]).collect();
        let standardizer = Standardizer {
            skip: 0,
            center: vec![0.5; width],
            scale: vec![2.0; width],
            lookups: vec![(
                2,
                Lookup::fit(
                    CategoryEncoding::OneHot,
                    &categories,
                    &labels,
                    Task::Classes(3),
                ),
            )],
        };
        let scaled = Scaled {
            rows: &store,
            standardizer: Some(&standardizer),
        };
        let rows: Vec<RowId> = (0..count as u32)
            .filter(|row| row % 7 != 3)
            .map(RowId)
            .collect();
        let mut many = vec![9.0];
        scaled.extend_vectors(&rows, &mut many).unwrap();
        let mut one_by_one = vec![9.0];
        for &row in &rows {
            one_by_one.extend(scaled.vector(row).unwrap());
        }
        assert!(one_by_one.len() > 1 + rows.len() * width);
        assert_eq!(
            many.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            one_by_one
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            scaled.extend_vectors(&[RowId(count as u32)], &mut many),
            Err(RowsError::Unknown(RowId(count as u32)))
        );
    }
}
