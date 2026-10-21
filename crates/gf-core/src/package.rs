// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The single binder of package tables: a lock's schema and Arrow rows as
//! the numbers the engine reads.
//!
//! A package's lock names each column's physical type, role and declared
//! semantic. Its input columns become feature columns, encoded as
//! [`crate::columns`] encodes every table, so a package context, its queries
//! and a server table answer the same. The labeled context rows are encoded
//! once, giving categorical columns their vocabularies in the order words
//! first appear; queries are encoded against them, a word the context never
//! held reading as missing. The Python package runtime and `gf-server` both
//! bind through here.

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_cast::cast;
use arrow_schema::DataType;
use serde::Deserialize;

use crate::catalog::Task;
use crate::columns::{self, Cells, Column, ColumnKind, Vocabulary};
use crate::node::Targets;
use crate::request::{Label, RequestError};

/// One column of a lock's schema, as `TableSchema.to_dict` writes it.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct SchemaColumn {
    pub name: String,
    #[serde(rename = "type")]
    pub physical: String,
    pub nullable: bool,
    pub role: String,
    pub semantic: Option<String>,
}

/// A lock's table schema, as `TableSchema.to_dict` writes it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Schema {
    pub columns: Vec<SchemaColumn>,
    pub rows: Option<u64>,
    pub classes: serde_json::Value,
    pub frequency: Option<f64>,
}

impl Schema {
    /// The feature columns its inputs become, in their locked order.
    ///
    /// A declared category or a string is categorical, a declared datetime,
    /// date or timestamp is a datetime, and every other input is numeric.
    #[must_use]
    pub fn features(&self) -> Vec<Column> {
        self.columns
            .iter()
            .filter(|column| column.role == "input")
            .map(|column| Column {
                name: column.name.clone(),
                kind: match (column.semantic.as_deref(), column.physical.as_str()) {
                    (Some("category"), _) | (_, "string") => ColumnKind::Categorical,
                    (Some("datetime"), _) | (_, "date" | "timestamp") => ColumnKind::Datetime,
                    _ => ColumnKind::Numeric,
                },
            })
            .collect()
    }
}

fn invalid(message: impl Into<String>) -> RequestError {
    RequestError::new("InvalidPackage", message, 422)
}

/// Feature columns and the vocabularies a context gave them.
#[derive(Clone, Debug)]
pub struct Encoder {
    columns: Vec<Column>,
    vocabularies: Vec<Vocabulary>,
}

impl Encoder {
    #[must_use]
    pub fn new(columns: Vec<Column>) -> Self {
        let vocabularies = vec![Vocabulary::default(); columns.len()];
        Self {
            columns,
            vocabularies,
        }
    }

    #[must_use]
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Model features per row.
    #[must_use]
    pub fn width(&self) -> usize {
        columns::width(&self.columns)
    }

    /// Each column's words, in code order; empty but for categorical columns.
    pub fn vocabularies(&self) -> impl Iterator<Item = &[String]> {
        self.vocabularies.iter().map(Vocabulary::words)
    }

    /// Encodes context rows, row-major, giving new words the next codes.
    ///
    /// # Errors
    ///
    /// `InvalidPackage` if a column is missing or holds values of another
    /// kind, or a column would hold too many or too long words; nothing is
    /// added to the vocabularies then.
    pub fn fit(&mut self, rows: &RecordBatch) -> Result<Vec<f32>, RequestError> {
        let mut added = Vec::with_capacity(self.columns.len());
        let matrix = self.matrix(rows, |index, column, strings| {
            let mut new = Vec::new();
            let codes = self.vocabularies[index]
                .assign(strings, &mut new)
                .map_err(|error| invalid(format!("column {:?}: {error}", column.name)));
            added.push((index, new));
            codes
        })?;
        for (index, words) in added {
            self.vocabularies[index].extend(words);
        }
        Ok(matrix)
    }

    /// Encodes query rows, row-major, against the vocabularies held: a word
    /// they do not hold is missing.
    ///
    /// # Errors
    ///
    /// `InvalidPackage` if a column is missing or holds values of another kind.
    pub fn encode(&self, rows: &RecordBatch) -> Result<Vec<f32>, RequestError> {
        self.matrix(rows, |index, _, strings| {
            Ok(self.vocabularies[index].codes(strings))
        })
    }

    fn matrix(
        &self,
        rows: &RecordBatch,
        mut code: impl FnMut(usize, &Column, &[Option<String>]) -> Result<Vec<f32>, RequestError>,
    ) -> Result<Vec<f32>, RequestError> {
        let count = rows.num_rows();
        let width = self.width();
        let mut matrix = vec![0.0_f32; count * width];
        for ((index, column), start) in self
            .columns
            .iter()
            .enumerate()
            .zip(columns::offsets(&self.columns))
        {
            let array = rows
                .column_by_name(&column.name)
                .ok_or_else(|| invalid(format!("rows lack the column {:?}", column.name)))?;
            match columns::decode(array, column).map_err(invalid)? {
                Cells::Numbers(values) => {
                    for (row, value) in values.into_iter().enumerate() {
                        matrix[row * width + start] = value;
                    }
                }
                Cells::Times(times) => {
                    for (row, time) in times.into_iter().enumerate() {
                        let parts = time
                            .map_or([f32::NAN; columns::DATETIME_PARTS], columns::datetime_parts);
                        matrix[row * width + start..row * width + start + parts.len()]
                            .copy_from_slice(&parts);
                    }
                }
                Cells::Strings(strings) => {
                    for (row, value) in code(index, column, &strings)?.into_iter().enumerate() {
                        matrix[row * width + start] = value;
                    }
                }
            }
        }
        Ok(matrix)
    }
}

/// A package's bound tabular context: its labeled rows encoded once.
#[derive(Clone, Debug)]
pub struct Context {
    pub encoder: Encoder,
    /// Labeled rows' features, row-major, `encoder.width()` wide.
    pub features: Vec<f32>,
    pub targets: Targets,
    /// The classes, in output order; empty for a regression.
    pub classes: Vec<Label>,
    pub rows: usize,
}

/// A target column's labels, None where unlabeled.
fn labels(array: &ArrayRef) -> Result<Vec<Option<Label>>, RequestError> {
    let refused = || invalid("class labels must be strings, booleans or integers");
    if let Some(values) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok((0..values.len())
            .map(|row| values.is_valid(row).then(|| Label::Bool(values.value(row))))
            .collect());
    }
    if matches!(
        array.data_type(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View | DataType::Dictionary(..)
    ) {
        let values = cast(array, &DataType::Utf8).map_err(|_| refused())?;
        let values = values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(refused)?;
        return Ok((0..values.len())
            .map(|row| {
                values
                    .is_valid(row)
                    .then(|| Label::Text(values.value(row).to_owned()))
            })
            .collect());
    }
    if array.data_type().is_integer() {
        let values = cast(array, &DataType::Int64).map_err(|_| refused())?;
        let values = values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(refused)?;
        return Ok((0..values.len())
            .map(|row| values.is_valid(row).then(|| Label::Int(values.value(row))))
            .collect());
    }
    Err(refused())
}

/// A target column's values as float32, None where unlabeled.
fn values(array: &ArrayRef) -> Result<Vec<Option<f32>>, RequestError> {
    let floats = cast(array, &DataType::Float32)
        .map_err(|_| invalid("regression targets must be numbers"))?;
    let floats = floats
        .as_any()
        .downcast_ref::<arrow_array::Float32Array>()
        .expect("cast to float32");
    Ok((0..floats.len())
        .map(|row| floats.is_valid(row).then(|| floats.value(row)))
        .collect())
}

/// Binds a tabular package's context: its labeled rows, encoded once.
///
/// # Errors
///
/// `EmptyContext` (503) for no labeled rows, `InvalidPackage` for a
/// classification context with fewer than two classes, labels of an
/// unsupported type, or columns `Encoder::fit` refuses.
pub fn bind(
    schema: &Schema,
    task: Task,
    target: &str,
    context: &RecordBatch,
) -> Result<Context, RequestError> {
    let column = context
        .column_by_name(target)
        .ok_or_else(|| invalid(format!("rows lack the column {target:?}")))?;
    let labeled: Vec<bool> = (0..column.len()).map(|row| column.is_valid(row)).collect();
    let mask = BooleanArray::from(labeled);
    let rows = arrow_select::filter::filter_record_batch(context, &mask)
        .map_err(|error| invalid(error.to_string()))?;
    if rows.num_rows() == 0 {
        return Err(RequestError::new(
            "EmptyContext",
            "context has no labeled rows",
            503,
        ));
    }
    let target_column = rows
        .column_by_name(target)
        .expect("the target column is kept");
    let (targets, classes) = if task == Task::Classification {
        let labels: Vec<Label> = labels(target_column)?.into_iter().flatten().collect();
        let mut classes: Vec<Label> = labels.clone();
        classes.sort();
        classes.dedup();
        if classes.len() < 2 {
            return Err(invalid("selected context needs at least two classes"));
        }
        if matches!(classes[0], Label::Bool(_)) {
            classes = vec![Label::Bool(false), Label::Bool(true)];
        }
        let codes = labels
            .iter()
            .map(|label| {
                i32::try_from(classes.binary_search(label).expect("a label is a class"))
                    .expect("fewer than 2^31 classes")
            })
            .collect();
        (Targets::Classes(codes), classes)
    } else {
        (
            Targets::Values(values(target_column)?.into_iter().flatten().collect()),
            Vec::new(),
        )
    };
    let mut encoder = Encoder::new(schema.features());
    let features = encoder.fit(&rows)?;
    Ok(Context {
        encoder,
        features,
        targets,
        classes,
        rows: rows.num_rows(),
    })
}
