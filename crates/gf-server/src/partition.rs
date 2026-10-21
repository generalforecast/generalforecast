// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rows that travel between the nodes of a table split into partitions.
//!
//! Each partition codes its categorical columns with a vocabulary of its own,
//! so rows travel as model features with each categorical feature's words in
//! place of its codes: column `key`, then `f0 ..` one per model feature,
//! float32 or, for a categorical one, strings, then `target`. A partition
//! offering rows for queries' contexts adds `query`, the query a row is
//! offered for (null for a row every query shares), and `score`, its rank
//! there (lower first). It also returns the queries themselves in the same form, without a key, so the
//! node that predicts reads them as it reads the rows.

use std::sync::Arc;

use arrow_array::builder::{FixedSizeBinaryBuilder, StringBuilder};
use arrow_array::{
    Array, ArrayRef, FixedSizeBinaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use gf_core::node::Targets;
use serde::{Deserialize, Serialize};

use crate::columns::{self, Column, ColumnKind, Strings};

/// Rows as model features, with categorical features' words apart.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelRows {
    /// Each row's key, or None for a query.
    pub keys: Vec<Option<u128>>,
    /// Row-major model features; NaN where `strings` holds a categorical
    /// feature's word.
    pub features: Vec<f32>,
    /// The words of each categorical column, by column, a row each.
    pub strings: Strings,
    /// Each row's target: a class index, or a value; None for a query.
    pub targets: Vec<Option<f32>>,
}

/// How a partition ranks a row it offers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Offer {
    /// The query the row is offered for, or None for a row every query shares.
    pub query: Option<u32>,
    /// Lower ranks first: a squared distance, or 0 for a shared row.
    pub score: f64,
}

/// Each query's context, as a predict over given rows carries it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Given {
    /// Each query's rows, as positions among the request's keyed rows.
    pub contexts: Vec<Vec<u32>>,
}

/// The categorical model features: each categorical column's feature, with the column.
fn categorical(definition: &[Column]) -> Vec<(usize, usize)> {
    definition
        .iter()
        .zip(columns::offsets(definition))
        .enumerate()
        .filter(|(_, (column, _))| column.kind == ColumnKind::Categorical)
        .map(|(index, (_, offset))| (index, offset))
        .collect()
}

/// A batch of rows as model features, with offers if a partition ranks them.
///
/// # Errors
///
/// If the rows do not match the columns, or Arrow refuses the batch.
pub fn rows_batch(
    rows: &ModelRows,
    definition: &[Column],
    regression: bool,
    offers: Option<&[Offer]>,
) -> Result<RecordBatch, ArrowError> {
    let width = columns::width(definition);
    let count = rows.keys.len();
    if rows.features.len() != count * width || rows.targets.len() != count {
        return Err(ArrowError::InvalidArgumentError(
            "rows, features and targets disagree".to_string(),
        ));
    }
    let words: Vec<(usize, &Vec<Option<String>>)> = categorical(definition)
        .into_iter()
        .filter_map(|(column, offset)| {
            rows.strings
                .iter()
                .find(|(held, _)| *held == column)
                .map(|(_, values)| (offset, values))
        })
        .collect();
    let mut fields = vec![Field::new("key", DataType::FixedSizeBinary(16), true)];
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(width + 5);
    let mut keys = FixedSizeBinaryBuilder::with_capacity(count, 16);
    for key in &rows.keys {
        match key {
            Some(key) => keys.append_value(key.to_le_bytes())?,
            None => keys.append_null(),
        }
    }
    arrays.push(Arc::new(keys.finish()));
    for feature in 0..width {
        let name = format!("f{feature}");
        if let Some((_, values)) = words.iter().find(|(offset, _)| *offset == feature) {
            let mut strings = StringBuilder::new();
            for value in values.iter() {
                strings.append_option(value.as_deref());
            }
            fields.push(Field::new(name, DataType::Utf8, true));
            arrays.push(Arc::new(strings.finish()));
        } else {
            fields.push(Field::new(name, DataType::Float32, true));
            arrays.push(Arc::new(Float32Array::from_iter_values(
                (0..count).map(|row| rows.features[row * width + feature]),
            )));
        }
    }
    if regression {
        fields.push(Field::new("target", DataType::Float32, true));
        arrays.push(Arc::new(Float32Array::from(rows.targets.clone())));
    } else {
        fields.push(Field::new("target", DataType::Int32, true));
        arrays.push(Arc::new(Int32Array::from(
            rows.targets
                .iter()
                .map(|target| target.map(|class| class as i32))
                .collect::<Vec<_>>(),
        )));
    }
    if let Some(offers) = offers {
        if offers.len() != count {
            return Err(ArrowError::InvalidArgumentError(
                "every row needs an offer".to_string(),
            ));
        }
        fields.push(Field::new("query", DataType::Int32, true));
        arrays.push(Arc::new(Int32Array::from(
            offers
                .iter()
                .map(|offer| offer.query.map(|query| query as i32))
                .collect::<Vec<_>>(),
        )));
        fields.push(Field::new("score", DataType::Float64, false));
        arrays.push(Arc::new(Float64Array::from_iter_values(
            offers.iter().map(|offer| offer.score),
        )));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
}

/// Rows read back from batches as `rows_batch` writes them.
///
/// # Errors
///
/// If a batch lacks a column, or holds one of the wrong type.
pub fn read_rows(batches: &[RecordBatch], definition: &[Column]) -> Result<ModelRows, String> {
    let width = columns::width(definition);
    let categorical = categorical(definition);
    let mut rows = ModelRows {
        strings: categorical
            .iter()
            .map(|&(column, _)| (column, Vec::new()))
            .collect(),
        ..ModelRows::default()
    };
    for batch in batches {
        let count = batch.num_rows();
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .ok_or_else(|| format!("missing column {name:?}"))
        };
        let keys = column("key")?
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .filter(|keys| keys.value_length() == 16)
            .ok_or("column \"key\" must be fixed_size_binary(16)")?;
        rows.keys.extend((0..count).map(|row| {
            keys.is_valid(row)
                .then(|| u128::from_le_bytes(keys.value(row).try_into().expect("16 bytes")))
        }));
        let start = rows.features.len();
        rows.features.resize(start + count * width, f32::NAN);
        for feature in 0..width {
            let name = format!("f{feature}");
            let values = column(&name)?;
            if let Some(position) = categorical
                .iter()
                .position(|&(_, offset)| offset == feature)
            {
                let words = values
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| format!("column {name:?} must hold strings"))?;
                rows.strings[position].1.extend(
                    (0..count).map(|row| words.is_valid(row).then(|| words.value(row).to_string())),
                );
            } else {
                let numbers = values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| format!("column {name:?} must be float32"))?;
                for row in 0..count {
                    if numbers.is_valid(row) {
                        rows.features[start + row * width + feature] = numbers.value(row);
                    }
                }
            }
        }
        let targets = column("target")?;
        if let Some(values) = targets.as_any().downcast_ref::<Float32Array>() {
            rows.targets
                .extend((0..count).map(|row| values.is_valid(row).then(|| values.value(row))));
        } else if let Some(classes) = targets.as_any().downcast_ref::<Int32Array>() {
            rows.targets.extend(
                (0..count).map(|row| classes.is_valid(row).then(|| classes.value(row) as f32)),
            );
        } else {
            return Err("column \"target\" must be float32 or int32".to_string());
        }
    }
    Ok(rows)
}

/// A batch's 128-bit keys: 16 little-endian bytes each, or 64-bit integers,
/// which keep their value, a negative one in two's complement, as clients map them.
///
/// # Errors
///
/// If the batch has no `key` column, or one of another type.
pub fn keys_of(batch: &RecordBatch) -> Result<Vec<u128>, String> {
    let key = batch
        .column_by_name("key")
        .ok_or("missing column \"key\"")?;
    let any = key.as_any();
    if let Some(wide) = any.downcast_ref::<FixedSizeBinaryArray>()
        && wide.value_length() == 16
    {
        return Ok((0..wide.len())
            .map(|index| u128::from_le_bytes(wide.value(index).try_into().expect("16 bytes")))
            .collect());
    }
    if let Some(unsigned) = any.downcast_ref::<arrow_array::UInt64Array>() {
        return Ok(unsigned
            .values()
            .iter()
            .map(|&key| u128::from(key))
            .collect());
    }
    if let Some(signed) = any.downcast_ref::<Int64Array>() {
        return Ok(signed
            .values()
            .iter()
            .map(|&key| i128::from(key) as u128)
            .collect());
    }
    Err("column \"key\" must be fixed_size_binary(16), uint64 or int64".to_string())
}

/// Each row's offer, from a batch a partition offered.
///
/// # Errors
///
/// If the batch lacks the offer columns.
pub fn read_offers(batch: &RecordBatch) -> Result<Vec<Offer>, String> {
    let typed = |name: &str| {
        batch
            .column_by_name(name)
            .ok_or_else(|| format!("missing column {name:?}"))
    };
    let queries = typed("query")?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or("column \"query\" must be int32")?;
    let scores = typed("score")?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or("column \"score\" must be float64")?;
    Ok((0..batch.num_rows())
        .map(|row| Offer {
            query: queries
                .is_valid(row)
                .then(|| u32::try_from(queries.value(row)).unwrap_or(u32::MAX)),
            score: scores.value(row),
        })
        .collect())
}

/// Targets of a table's task from rows' targets; keyless rows have none.
///
/// # Errors
///
/// If a keyed row has no target.
pub fn targets(rows: &ModelRows, regression: bool) -> Result<Targets, String> {
    let keyed = rows
        .keys
        .iter()
        .zip(&rows.targets)
        .filter(|(key, _)| key.is_some())
        .map(|(_, target)| target.ok_or_else(|| "a row with a key has no target".to_string()));
    if regression {
        keyed.collect::<Result<_, _>>().map(Targets::Values)
    } else {
        keyed
            .map(|target| target.map(|class| class as i32))
            .collect::<Result<_, _>>()
            .map(Targets::Classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_come_back_as_they_were_sent() {
        let definition = vec![
            Column {
                name: "age".to_string(),
                kind: ColumnKind::Numeric,
            },
            Column {
                name: "city".to_string(),
                kind: ColumnKind::Categorical,
            },
        ];
        let rows = ModelRows {
            keys: vec![Some(7), None],
            features: vec![1.5, f32::NAN, 2.5, f32::NAN],
            strings: vec![(1, vec![Some("oslo".to_string()), None])],
            targets: vec![Some(2.0), None],
        };
        let offers = [
            Offer {
                query: Some(0),
                score: 0.25,
            },
            Offer {
                query: None,
                score: 0.0,
            },
        ];

        let batch = rows_batch(&rows, &definition, false, Some(&offers)).unwrap();

        let read = read_rows(std::slice::from_ref(&batch), &definition).unwrap();
        assert_eq!(read.keys, rows.keys);
        assert_eq!(read.strings, rows.strings);
        assert_eq!(read.targets, rows.targets);
        assert_eq!(read.features[0], 1.5);
        assert_eq!(read.features[2], 2.5);
        assert_eq!(read_offers(&batch).unwrap(), offers);
        assert_eq!(targets(&read, false).unwrap(), Targets::Classes(vec![2]));
    }
}
