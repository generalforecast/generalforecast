// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Tables split into partitions, each an ordinary table on the nodes.
//!
//! A table made with `partitions` is that many tables,
//! `{name}--part-{index}-of-{count}`, each placed and replicated like any
//! other; each name says how many there are, so a router that finds any one
//! knows them all, and a missing one is missed rather than its table shrunk. A row goes to the partition its
//! key hashes to. A prediction gathers each query's context from every
//! partition: each offers the rows its strategy would choose, with their
//! ranks, and the router keeps, per query, what one table would have chosen
//! from all of them: every row, a sample split evenly between partitions, or
//! the nearest rows. One node then predicts each query against exactly its
//! context, encoded as one. Partitions standardize their rows apart, so ranks from
//! different partitions compare in spaces that differ a little; rows are
//! spread by a hash of their keys, so each partition is a sample of the
//! table's rows, and fit their own categorical encodings, whose most frequent
//! categories and target means differ a little too. A table with retrieval
//! columns ranks by them unscaled, so its partitions share one space exactly.
//!
//! A table made with `route_by` as well sends rows and queries to partitions
//! by one column's value instead, hashed or in order between bounds, so each
//! partition holds the rows its queries read and answers them as a table of
//! its own, with its own contexts, regions and cutoffs, and nothing is
//! gathered. A delete carries keys alone, so it goes to every partition.

use std::collections::{BTreeMap, HashMap};

use arrow_array::{Array, FixedSizeBinaryArray, Float64Array, RecordBatch, UInt32Array};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use gf_core::planner::Strategy;
use gf_server::partition::{Given, keys_of, read_offers};
use gf_server::wal::RouteBy;

/// What separates a table's name from a partition's index in the partition's name.
pub const MARK: &str = "--part-";

/// The most partitions a table may have.
pub const MOST: usize = 256;

/// The name of partition `index` of a table split into `count`.
#[must_use]
pub fn name(table: &str, index: usize, count: usize) -> String {
    format!("{table}{MARK}{index}-of-{count}")
}

/// The table, index and count a partition's name holds, if it is one.
#[must_use]
pub fn parse(name: &str) -> Option<(&str, usize, usize)> {
    let (table, rest) = name.rsplit_once(MARK)?;
    let (index, count) = rest.split_once("-of-")?;
    let (index, count): (usize, usize) = (index.parse().ok()?, count.parse().ok()?);
    (!table.is_empty() && index < count && name == self::name(table, index, count))
        .then_some((table, index, count))
}

/// The partition a key belongs to, by a hash that never changes.
#[must_use]
pub fn of(key: u128, count: usize) -> usize {
    (gf_assign::key::wide(key) % count as u64) as usize
}

/// Each partition's rows of the batches, by their keys' partitions.
///
/// # Errors
///
/// If a batch's keys cannot be read.
pub fn split(batches: &[RecordBatch], count: usize) -> Result<Vec<Vec<RecordBatch>>, String> {
    dealt(batches, count, |batch| {
        Ok(keys_of(batch)?
            .into_iter()
            .map(|key| of(key, count))
            .collect())
    })
}

/// Each partition's rows of the batches, by the route's column.
///
/// # Errors
///
/// If a batch's route column cannot be read.
pub fn split_by(
    route: &RouteBy,
    batches: &[RecordBatch],
    count: usize,
) -> Result<Vec<Vec<RecordBatch>>, String> {
    dealt(batches, count, |batch| routed(route, batch, count))
}

/// Each row's partition by the route's column, which must hold a number in
/// every row.
///
/// # Errors
///
/// If the route's bounds cut another number of partitions, or the batch
/// lacks the column, or a row's value is missing or not a number.
pub fn routed(route: &RouteBy, batch: &RecordBatch, count: usize) -> Result<Vec<usize>, String> {
    let assignment = route.assignment(count);
    if assignment.slices.len() != count {
        return Err(format!(
            "route_by bounds cut {} partitions, not {count}",
            assignment.slices.len()
        ));
    }
    let column = batch
        .column_by_name(&route.column)
        .ok_or_else(|| format!("missing column {:?}, which routes rows", route.column))?;
    let values = arrow_cast::cast(column, &arrow_schema::DataType::Float64)
        .map_err(|_| format!("column {:?} must hold numbers", route.column))?;
    let values = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("cast to float64");
    (0..values.len())
        .map(|row| {
            // A null or a NaN names no partition.
            (values.is_valid(row) && !values.value(row).is_nan())
                .then(|| assignment.slice_of(route.key(values.value(row))))
                .ok_or_else(|| {
                    format!(
                        "column {:?} routes rows, so every row needs a number in it, \
                         and row {row} has none",
                        route.column
                    )
                })
        })
        .collect()
}

/// Each partition's rows of the batches, by `partition_of` each batch's rows.
fn dealt(
    batches: &[RecordBatch],
    count: usize,
    partition_of: impl Fn(&RecordBatch) -> Result<Vec<usize>, String>,
) -> Result<Vec<Vec<RecordBatch>>, String> {
    let mut parts: Vec<Vec<RecordBatch>> = vec![Vec::new(); count];
    for batch in batches {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); count];
        for (row, partition) in partition_of(batch)?.into_iter().enumerate() {
            rows[partition].push(u32::try_from(row).map_err(|error| error.to_string())?);
        }
        for (part, rows) in parts.iter_mut().zip(rows) {
            if !rows.is_empty() {
                let taken = take_record_batch(batch, &UInt32Array::from(rows))
                    .map_err(|error| error.to_string())?;
                part.push(taken);
            }
        }
    }
    Ok(parts)
}

/// A partition's share of a sample of `rows` rows.
#[must_use]
pub fn share(rows: u64, count: usize) -> u64 {
    rows.div_ceil(count as u64)
}

/// The sample count a strategy splits between partitions, if any.
#[must_use]
pub fn sampled(strategy: &Strategy) -> Option<u64> {
    match *strategy {
        Strategy::Retrieved {
            core: Some(core), ..
        } => Some(core.rows),
        _ => None,
    }
}

/// Each query's cutoff, or None for a query without one; None for queries
/// without a `before` column.
///
/// # Errors
///
/// If the column holds neither integers nor timestamps.
pub fn cutoffs(queries: &RecordBatch) -> Result<Option<Vec<Option<i64>>>, String> {
    let Some(before) = queries.column_by_name("before") else {
        return Ok(None);
    };
    let numbers = arrow_cast::cast(before, &arrow_schema::DataType::Int64)
        .map_err(|_| "column \"before\" must hold integers or timestamps".to_string())?;
    let numbers = numbers
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("cast to int64");
    Ok(Some(
        (0..numbers.len())
            .map(|row| numbers.is_valid(row).then(|| numbers.value(row)))
            .collect(),
    ))
}

/// A row some partition offered, as the router ranks it.
#[derive(Clone, Copy, Debug)]
struct Offered {
    partition: usize,
    row: usize,
    key: u128,
    score: f64,
}

/// Rows picked for contexts, each once, in the order first picked.
#[derive(Default)]
struct Picked {
    positions: HashMap<u128, u32>,
    /// Each position's partition and row in it.
    rows: Vec<(usize, usize)>,
}

impl Picked {
    fn pick(&mut self, offered: &Offered) -> u32 {
        *self.positions.entry(offered.key).or_insert_with(|| {
            self.rows.push((offered.partition, offered.row));
            (self.rows.len() - 1) as u32
        })
    }
}

/// The rows of every query's context and the queries themselves, as one
/// batch for the node that predicts, with each query's context, from each
/// partition's offer.
///
/// # Errors
///
/// If an offer is malformed, or the partitions' rows differ in their columns.
pub fn assemble(
    strategy: &Strategy,
    offers: &[RecordBatch],
) -> Result<(RecordBatch, Given), String> {
    let first = offers.first().ok_or("no partition offered rows")?;
    let mut shared: Vec<Offered> = Vec::new();
    let mut asked: Vec<(u32, usize)> = Vec::new();
    let mut per_query: BTreeMap<u32, Vec<Offered>> = BTreeMap::new();
    for (partition, batch) in offers.iter().enumerate() {
        let ranks = read_offers(batch)?;
        let keys = batch
            .column_by_name("key")
            .and_then(|keys| keys.as_any().downcast_ref::<FixedSizeBinaryArray>())
            .ok_or("an offer's keys must be fixed_size_binary(16)")?;
        for (row, rank) in ranks.iter().enumerate() {
            if keys.is_null(row) {
                // The queries, once: as the first partition read them.
                if partition == 0 {
                    asked.push((rank.query.unwrap_or(u32::MAX), row));
                }
                continue;
            }
            let offered = Offered {
                partition,
                row,
                key: u128::from_le_bytes(keys.value(row).try_into().expect("16 bytes")),
                score: rank.score,
            };
            match rank.query {
                None => shared.push(offered),
                Some(query) => per_query.entry(query).or_default().push(offered),
            }
        }
    }
    asked.sort_unstable();
    let queries = asked.len();
    let mut picked = Picked::default();
    let common: Vec<u32> = shared.iter().map(|offered| picked.pick(offered)).collect();
    let empty = Vec::new();
    let contexts: Vec<Vec<u32>> = (0..queries as u32)
        .map(|query| {
            let offered = per_query.get(&query).unwrap_or(&empty);
            let Strategy::Retrieved { neighbors, .. } = *strategy else {
                return common.clone();
            };
            let mut nearest: Vec<&Offered> = offered.iter().collect();
            nearest.sort_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then(left.key.cmp(&right.key))
            });
            let mut rows = common.clone();
            rows.extend(
                nearest
                    .into_iter()
                    .take(neighbors)
                    .map(|candidate| picked.pick(candidate)),
            );
            rows
        })
        .collect();
    // Every partition's rows in one batch, without their ranks, then the
    // picked rows in position order and the queries after them.
    let keep: Vec<usize> = first
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| !["query", "score"].contains(&field.name().as_str()))
        .map(|(index, _)| index)
        .collect();
    let projected: Vec<RecordBatch> = offers
        .iter()
        .map(|batch| batch.project(&keep))
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())?;
    let mut offsets = Vec::with_capacity(projected.len());
    let mut total = 0;
    for batch in &projected {
        offsets.push(total);
        total += batch.num_rows();
    }
    let all =
        concat_batches(&projected[0].schema(), &projected).map_err(|error| error.to_string())?;
    let indices: Vec<u32> = picked
        .rows
        .iter()
        .map(|&(partition, row)| (offsets[partition] + row) as u32)
        .chain(asked.iter().map(|&(_, row)| row as u32))
        .collect();
    let rows =
        take_record_batch(&all, &UInt32Array::from(indices)).map_err(|error| error.to_string())?;
    Ok((rows, Given { contexts }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, Int64Array};
    use gf_core::columns::numeric;
    use gf_server::partition::{ModelRows, Offer, rows_batch};
    use gf_server::wal::RouteKind;

    use super::*;

    #[test]
    fn names_hold_their_table_and_index() {
        assert_eq!(parse(&name("sales", 3, 4)), Some(("sales", 3, 4)));
        assert_eq!(parse("sales"), None);
        assert_eq!(parse("sales--part-03-of-4"), None);
        assert_eq!(parse("sales--part-4-of-4"), None);
        assert_eq!(parse("--part-1-of-2"), None);
    }

    #[test]
    fn keys_spread_over_partitions_and_never_move() {
        let mut counts = [0usize; 4];
        for key in 0..4_000u128 {
            counts[of(key, 4)] += 1;
            assert_eq!(of(key, 4), of(key, 4));
        }
        assert!(
            counts.iter().all(|&count| (800..1_200).contains(&count)),
            "{counts:?}"
        );
    }

    #[test]
    fn routed_rows_go_by_their_column_and_keep_their_keys() {
        let route = RouteBy {
            column: "f0".to_string(),
            kind: RouteKind::Ordered,
            bounds: vec![5.0],
        };
        let batch = RecordBatch::try_from_iter([
            (
                "key",
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            ),
            (
                "f0",
                Arc::new(Int32Array::from(vec![0, 9, 5, -3])) as ArrayRef,
            ),
        ])
        .unwrap();

        assert_eq!(routed(&route, &batch, 2).unwrap(), [0, 1, 1, 0]);
        let parts = split_by(&route, std::slice::from_ref(&batch), 2).unwrap();
        let keys: Vec<Vec<i64>> = parts
            .iter()
            .map(|part| {
                part[0]
                    .column_by_name("key")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(keys, [vec![1, 4], vec![2, 3]]);
        // The bounds cut two partitions, so a count of three is a mismatch,
        // and a row without a value has no partition.
        assert!(routed(&route, &batch, 3).is_err());
        let missing = RecordBatch::try_from_iter([(
            "f0",
            Arc::new(Float64Array::from(vec![Some(1.0), None])) as ArrayRef,
        )])
        .unwrap();
        assert!(routed(&route, &missing, 2).is_err());
    }

    #[test]
    fn a_route_refuses_nan_and_sends_negative_zero_with_zero() {
        for kind in [RouteKind::Hashed, RouteKind::Ordered] {
            let route = RouteBy {
                column: "f0".to_string(),
                kind,
                bounds: if kind == RouteKind::Ordered {
                    vec![0.0]
                } else {
                    vec![]
                },
            };
            let zeros = RecordBatch::try_from_iter([(
                "f0",
                Arc::new(Float64Array::from(vec![0.0, -0.0])) as ArrayRef,
            )])
            .unwrap();
            let parts = routed(&route, &zeros, 2).unwrap();
            assert_eq!(parts[0], parts[1], "{kind:?}");
            let nan = RecordBatch::try_from_iter([(
                "f0",
                Arc::new(Float64Array::from(vec![1.0, f64::NAN])) as ArrayRef,
            )])
            .unwrap();
            let error = routed(&route, &nan, 2).unwrap_err();
            assert!(error.contains("row 1 has none"), "{error}");
        }
    }

    /// A partition's offer: rows `keys` for query 0 at `scores`, then one query.
    fn offer(keys: &[u128], scores: &[f64]) -> RecordBatch {
        let count = keys.len();
        let rows = ModelRows {
            keys: keys.iter().copied().map(Some).chain([None]).collect(),
            features: keys.iter().map(|&key| key as f32).chain([0.5]).collect(),
            strings: Vec::new(),
            targets: (0..count).map(|_| Some(1.0)).chain([None]).collect(),
        };
        let offers: Vec<Offer> = (0..count)
            .map(|row| Offer {
                query: Some(0),
                score: scores[row],
            })
            .chain([Offer {
                query: Some(0),
                score: 0.0,
            }])
            .collect();
        rows_batch(&rows, &numeric(1), false, Some(&offers)).unwrap()
    }

    fn keys_of_rows(batch: &RecordBatch, positions: &[u32]) -> Vec<u128> {
        let keys = batch
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        positions
            .iter()
            .map(|&row| u128::from_le_bytes(keys.value(row as usize).try_into().unwrap()))
            .collect()
    }

    #[test]
    fn a_querys_nearest_rows_come_from_every_partition() {
        let offers = [
            offer(&[1, 2, 3], &[0.1, 0.5, 0.9]),
            offer(&[4, 5], &[0.2, 0.3]),
        ];

        let (rows, given) = assemble(&Strategy::neighbors(3), &offers).unwrap();

        assert_eq!(given.contexts.len(), 1);
        let nearest = &given.contexts[0];
        assert_eq!(keys_of_rows(&rows, nearest), [1, 4, 5]);
        // The query follows the context's rows, without a key.
        assert_eq!(rows.num_rows(), 4);
        assert!(rows.column_by_name("key").unwrap().is_null(3));
    }
}
