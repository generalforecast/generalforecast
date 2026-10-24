// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Locked tabular predictors on the Models API: what `gf serve` says of one,
//! and its JSON rows as the Arrow columns `gf_core::predictor`, the one
//! binder, encodes.
//!
//! A snapshot predictor's rows are bound once, when the server starts, into a
//! durable table; a request-context predictor binds each request's rows into
//! the ephemeral table their encoding names, as a tabular request does.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type, TimestampMillisecondType};
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, RecordBatchOptions, StringArray,
};
use arrow_cast::cast;
use arrow_ipc::reader::StreamReader;
use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use gf_core::catalog::{self, Task};
use gf_core::predictor::{Schema, SchemaColumn};
use gf_core::request::RequestError;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Where a predictor's context comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContextMode {
    /// Rows locked with the predictor, bound once.
    Snapshot,
    /// Rows each request brings.
    Request,
}

impl ContextMode {
    /// The manifest's word for it.
    #[must_use]
    pub fn kind(self) -> &'static str {
        match self {
            Self::Snapshot => "snapshot",
            Self::Request => "request",
        }
    }
}

/// A locked tabular predictor, as `gf serve` writes it from the lock.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServedPredictor {
    /// The lock's table schema, as `TableSchema.to_dict` writes it; no
    /// columns for a request predictor checked without data.
    pub schema: Schema,
    /// `classification` or `regression`.
    pub task: String,
    /// The target column, if the lock names one.
    pub target: Option<String>,
    /// The input columns, in locked order.
    pub inputs: Vec<String>,
    /// The column naming each query, which predictions repeat.
    pub id: Option<String>,
    pub context: ContextMode,
    /// The quantile levels a regression reports.
    pub quantiles: Vec<f64>,
    /// The digest of a snapshot's rows, which predictions report as their
    /// context's digest.
    pub context_digest: Option<String>,
    /// The digest of the lock served.
    pub lock_digest: String,
    /// A snapshot predictor's context rows, as an Arrow IPC stream.
    pub snapshot: Option<PathBuf>,
    /// A forecast predictor's series layout, as its lock's spec gives it.
    #[serde(default)]
    pub forecast: Option<ForecastPredictor>,
}

/// How a forecast predictor's rows form its related series.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastPredictor {
    /// The columns naming each row's series.
    pub series: Vec<String>,
    /// The time column.
    pub time: String,
    pub horizon: usize,
    /// Covariates known only up to the origin.
    pub past_inputs: Vec<String>,
    /// Covariates whose future values each query brings.
    pub future_inputs: Vec<String>,
    /// The time between rows, as the lock's schema holds it; None for a
    /// request predictor, whose context shows it.
    pub frequency: Option<f64>,
    /// The time between rows the spec declares, in time units.
    pub declared: Option<f64>,
}

/// A forecast predictor's context, aligned as its lock lays it out: its
/// related series, their shared origin and step, and the forecast request
/// of the group, as `PredictorExecutor` builds it.
#[derive(Clone, Debug)]
pub struct ForecastContext {
    /// Each series' values of the lock's series columns, in context order.
    pub identities: Vec<Vec<Value>>,
    pub origin: f64,
    pub step: f64,
    /// The group's `gf_core::request` forecast payload.
    pub request: Value,
}

/// Where a forecast predictor's queries fall: each row's series and horizon
/// position, the series queried, and the known inputs' future values.
#[derive(Clone, Debug)]
pub struct ForecastQueries {
    pub positions: Vec<(usize, usize)>,
    /// The series queried, in context order.
    pub members: Vec<usize>,
    /// Each future input's values over the horizon, in the lock's order.
    pub future: Vec<Vec<f64>>,
}

fn refused(message: &str) -> RequestError {
    RequestError::new("InvalidPredictor", message, 422)
}

/// A column's cells as JSON values: numbers, booleans and strings as such,
/// and dates and times as their UTC seconds.
fn cells(array: &ArrayRef) -> Result<Vec<Value>, RequestError> {
    let failed = || refused("forecast columns must be numeric, boolean, string, date or timestamp");
    let data_type = array.data_type();
    let rows = 0..array.len();
    if data_type.is_integer() {
        let values = cast(array, &DataType::Int64).map_err(|_| failed())?;
        let values = values.as_primitive::<Int64Type>();
        Ok(rows
            .map(|row| {
                if values.is_null(row) {
                    Value::Null
                } else {
                    json!(values.value(row))
                }
            })
            .collect())
    } else if data_type.is_floating() {
        let values = cast(array, &DataType::Float64).map_err(|_| failed())?;
        let values = values.as_primitive::<Float64Type>();
        Ok(rows
            .map(|row| {
                if values.is_null(row) {
                    Value::Null
                } else {
                    json!(values.value(row))
                }
            })
            .collect())
    } else if *data_type == DataType::Boolean {
        let values = array.as_boolean();
        Ok(rows
            .map(|row| {
                if values.is_null(row) {
                    Value::Null
                } else {
                    json!(values.value(row))
                }
            })
            .collect())
    } else if matches!(data_type, DataType::Utf8 | DataType::LargeUtf8) {
        let values = cast(array, &DataType::Utf8).map_err(|_| failed())?;
        let values = values.as_string::<i32>();
        Ok(rows
            .map(|row| {
                if values.is_null(row) {
                    Value::Null
                } else {
                    json!(values.value(row))
                }
            })
            .collect())
    } else if data_type.is_temporal() {
        seconds(array)
    } else {
        Err(failed())
    }
}

/// Dates and times, or their ISO text, as UTC seconds.
fn seconds(array: &ArrayRef) -> Result<Vec<Value>, RequestError> {
    let zone = match array.data_type() {
        DataType::Timestamp(_, zone) => zone.clone(),
        _ => None,
    };
    let millis = cast(array, &DataType::Timestamp(TimeUnit::Millisecond, zone))
        .map_err(|_| refused("forecast time must be numeric, date or timestamp"))?;
    let millis = millis.as_primitive::<TimestampMillisecondType>();
    Ok((0..millis.len())
        .map(|row| {
            if millis.is_null(row) {
                Value::Null
            } else {
                json!(millis.value(row) as f64 / 1000.0)
            }
        })
        .collect())
}

/// A time column's instants: periods as numbers, dates and times as UTC
/// seconds, as `grouping.instant` reads them.
fn instants(array: &ArrayRef) -> Result<Vec<Value>, RequestError> {
    let data_type = array.data_type();
    if data_type.is_integer() || data_type.is_floating() {
        cells(array)
    } else if data_type.is_temporal() || matches!(data_type, DataType::Utf8 | DataType::LargeUtf8) {
        seconds(array)
    } else {
        Err(refused("forecast time must be numeric, date or timestamp"))
    }
}

impl ForecastPredictor {
    /// The columns a forecast query carries: its series, its time and its
    /// known future inputs.
    #[must_use]
    pub fn query_columns(&self) -> Vec<&str> {
        self.series
            .iter()
            .chain([&self.time])
            .chain(&self.future_inputs)
            .map(String::as_str)
            .collect()
    }

    /// The columns a forecast context carries besides its target.
    #[must_use]
    pub fn context_columns(&self) -> Vec<String> {
        self.series
            .iter()
            .chain([&self.time])
            .chain(&self.past_inputs)
            .chain(&self.future_inputs)
            .cloned()
            .collect()
    }

    /// Aligns a context's related series into one forecast request for
    /// `model`, with one query over the whole group and zeros for the
    /// known inputs' future, which each prediction fills in.
    ///
    /// # Errors
    ///
    /// The refusals `ForecastContext.build` makes: too many series, series
    /// ending at different origins, missing times, or an irregular step.
    pub fn context(
        &self,
        model: &str,
        target: &str,
        levels: &[f64],
        rows: &RecordBatch,
        frequency: Option<f64>,
    ) -> Result<ForecastContext, RequestError> {
        let column = |name: &str| {
            rows.column_by_name(name)
                .ok_or_else(|| refused(&format!("context has no column {name:?}")))
        };
        let series = self
            .series
            .iter()
            .map(|name| cells(column(name)?))
            .collect::<Result<Vec<_>, _>>()?;
        let times = instants(column(&self.time)?)?;
        let targets = cells(column(target)?)?;
        let inputs: Vec<&String> = self.past_inputs.iter().chain(&self.future_inputs).collect();
        let covariates = inputs
            .iter()
            .map(|name| cells(column(name)?))
            .collect::<Result<Vec<_>, _>>()?;
        let mut identities: Vec<Vec<Value>> = Vec::new();
        let mut ends: Vec<(f64, Value)> = Vec::new();
        let mut moments: Vec<Vec<f64>> = Vec::new();
        let mut context = Vec::with_capacity(rows.num_rows());
        for row in 0..rows.num_rows() {
            let identity: Vec<Value> = series.iter().map(|column| column[row].clone()).collect();
            let moment = &times[row];
            let at = moment
                .as_f64()
                .filter(|_| identity.iter().all(|value| !value.is_null()))
                .ok_or_else(|| refused("forecast time and series cannot be missing"))?;
            let index = if let Some(index) = identities.iter().position(|known| *known == identity)
            {
                index
            } else {
                identities.push(identity);
                ends.push((at, moment.clone()));
                moments.push(Vec::new());
                identities.len() - 1
            };
            if at > ends[index].0 {
                ends[index] = (at, moment.clone());
            }
            moments[index].push(at);
            let values: Map<String, Value> = inputs
                .iter()
                .zip(&covariates)
                .map(|(name, column)| ((*name).clone(), column[row].clone()))
                .collect();
            context.push(json!({
                "key": row.to_string(),
                "series": format!("gfseries:{index}"),
                "time": moment,
                "value": targets[row],
                "covariates": values,
            }));
        }
        if identities.is_empty() || identities.len() > catalog::catalog().policy.group_series {
            return Err(RequestError::new(
                "CapacityExceeded",
                "grouped forecast supports at most eight explicitly related series",
                422,
            ));
        }
        if ends
            .iter()
            .any(|(end, _)| end.total_cmp(&ends[0].0).is_ne())
        {
            return Err(refused(
                "related series must share the same forecast origin",
            ));
        }
        let frequency = match frequency {
            Some(frequency) => frequency,
            None => self.observed(&mut moments)?,
        };
        let origin = ends[0].1.clone();
        let step = if origin.is_i64() && frequency.fract() == 0.0 {
            json!(frequency as i64)
        } else {
            json!(frequency)
        };
        let group: Vec<String> = (0..identities.len())
            .map(|index| format!("gfseries:{index}"))
            .collect();
        let future: Map<String, Value> = self
            .future_inputs
            .iter()
            .map(|name| (name.clone(), json!(vec![0.0; self.horizon])))
            .collect();
        let request = json!({
            "model": model,
            "task": "forecast",
            "target": target,
            "features": [],
            "context": context,
            "queries": [{
                "key": "warm",
                "series": "gfseries:0",
                "origin": origin,
                "group": group,
                "future_covariates": future,
            }],
            "options": {
                "series": {
                    "step": step,
                    "horizon": self.horizon,
                    "quantiles": levels,
                    "known_covariates": self.future_inputs,
                }
            },
        });
        Ok(ForecastContext {
            identities,
            origin: ends[0].0,
            step: frequency,
            request,
        })
    }

    /// The one step between each series' times, as the lock's compiler
    /// verifies it.
    fn observed(&self, moments: &mut [Vec<f64>]) -> Result<f64, RequestError> {
        let mut steps: Vec<f64> = Vec::new();
        for times in moments.iter_mut() {
            if times.len() < 2 {
                return Err(refused(
                    "each forecast series needs two times to verify frequency",
                ));
            }
            times.sort_by(f64::total_cmp);
            for pair in times.windows(2) {
                if pair[1] <= pair[0] {
                    return Err(refused("duplicate forecast time within a series"));
                }
                let step = pair[1] - pair[0];
                if !steps.iter().any(|known| known.total_cmp(&step).is_eq()) {
                    steps.push(step);
                }
            }
        }
        if steps.len() != 1 {
            return Err(RequestError::new(
                "AmbiguousFrequency",
                "forecast frequency is irregular; resample explicitly",
                422,
            ));
        }
        if self
            .declared
            .is_some_and(|declared| declared.total_cmp(&steps[0]).is_ne())
        {
            return Err(refused("declared frequency differs from observed times"));
        }
        Ok(steps[0])
    }

    /// A forecast's query rows as the lock types them: exactly its series,
    /// time and future inputs, and its id if it declares one, with no
    /// missing values.
    ///
    /// # Errors
    ///
    /// The refusals `query_table` makes for forecast queries.
    pub fn query_rows(
        &self,
        schema: &Schema,
        id: Option<&str>,
        queries: &[Value],
    ) -> Result<RecordBatch, RequestError> {
        let required = self.query_columns();
        let allowed: Vec<&str> = required.iter().copied().chain(id).collect();
        for row in queries {
            let row = row
                .as_object()
                .ok_or_else(|| refused("rows must be objects"))?;
            if !required.iter().all(|name| row.contains_key(*name))
                || !row.keys().all(|name| allowed.contains(&name.as_str()))
            {
                return Err(refused("forecast query columns differ from locked roles"));
            }
            if row.values().any(Value::is_null) {
                return Err(refused("forecast queries cannot contain missing values"));
            }
        }
        let columns: Vec<SchemaColumn> = schema
            .columns
            .iter()
            .filter(|column| allowed.contains(&column.name.as_str()))
            .cloned()
            .collect();
        records(&columns, queries).map_err(|_| refused("query physical type differs from lock"))
    }

    /// Places each query row on its series and horizon step, with the known
    /// inputs' future values they bring.
    ///
    /// # Errors
    ///
    /// The refusals `ForecastContext.predict` makes: an unknown series, a
    /// time off the horizon or repeated, unbounded or conflicting future
    /// values, or a series queried over less than the whole horizon.
    pub fn queries(
        &self,
        context: &ForecastContext,
        rows: &RecordBatch,
    ) -> Result<ForecastQueries, RequestError> {
        let column = |name: &str| {
            rows.column_by_name(name)
                .ok_or_else(|| refused("forecast query columns differ from locked roles"))
        };
        let series = self
            .series
            .iter()
            .map(|name| cells(column(name)?))
            .collect::<Result<Vec<_>, _>>()?;
        let times = instants(column(&self.time)?)?;
        let inputs = self
            .future_inputs
            .iter()
            .map(|name| cells(column(name)?))
            .collect::<Result<Vec<_>, _>>()?;
        let mut positions = Vec::with_capacity(rows.num_rows());
        let mut seen = vec![vec![false; self.horizon]; context.identities.len()];
        let mut future: Vec<Vec<Option<f64>>> =
            vec![vec![None; self.horizon]; self.future_inputs.len()];
        for row in 0..rows.num_rows() {
            let identity: Vec<Value> = series.iter().map(|column| column[row].clone()).collect();
            let member = context
                .identities
                .iter()
                .position(|known| *known == identity)
                .ok_or_else(|| refused("query series is missing from context"))?;
            let position = times[row]
                .as_f64()
                .map(|at| (at - context.origin) / context.step - 1.0)
                .filter(|offset| {
                    (offset - offset.round()).abs() < 1e-6
                        && offset.round() >= 0.0
                        && offset.round() < self.horizon as f64
                })
                .map(|offset| offset.round() as usize)
                .ok_or_else(|| {
                    refused(
                        "query times must cover the locked future horizon after the context origin",
                    )
                })?;
            if std::mem::replace(&mut seen[member][position], true) {
                return Err(refused("duplicate forecast query time within a series"));
            }
            positions.push((member, position));
            for (values, column) in future.iter_mut().zip(&inputs) {
                let value = match &column[row] {
                    Value::Bool(value) => Some(f64::from(u8::from(*value))),
                    Value::Number(value) => value.as_f64(),
                    _ => None,
                }
                .filter(|value| value.is_finite() && value.abs() <= 1e12)
                .ok_or_else(|| refused("known future covariates must be bounded numeric values"))?;
                if values[position].is_some_and(|held| held.total_cmp(&value).is_ne()) {
                    return Err(RequestError::new(
                        "UnsupportedCovariates",
                        "grouped targets must share named future covariates",
                        422,
                    ));
                }
                values[position] = Some(value);
            }
        }
        let members: Vec<usize> = (0..seen.len())
            .filter(|&member| seen[member].iter().any(|&held| held))
            .collect();
        if members
            .iter()
            .any(|&member| !seen[member].iter().all(|&held| held))
        {
            return Err(refused(
                "queries must include the complete horizon for each requested series",
            ));
        }
        Ok(ForecastQueries {
            positions,
            members,
            future: future
                .into_iter()
                .map(|values| values.into_iter().map(Option::unwrap_or_default).collect())
                .collect(),
        })
    }
}

impl ServedPredictor {
    /// The task the binder fits.
    ///
    /// # Errors
    ///
    /// `InvalidPredictor` for a task a tabular predictor does not have.
    pub fn task(&self) -> Result<Task, RequestError> {
        match self.task.as_str() {
            "classification" => Ok(Task::Classification),
            "regression" => Ok(Task::Regression),
            other => Err(RequestError::new(
                "InvalidPredictor",
                format!("a tabular predictor has no task {other:?}"),
                422,
            )),
        }
    }

    /// Whether each request names its own features and target, because the
    /// lock holds no schema.
    #[must_use]
    pub fn schema_free(&self) -> bool {
        self.context == ContextMode::Request && self.schema.columns.is_empty()
    }
}

/// A schema for rows a lock holds none for: `features` as inputs and
/// `target`, each typed by its first value.
#[must_use]
pub fn inferred(features: &[String], target: &str, rows: &[Value]) -> Schema {
    let column = |name: &str, role: &str| {
        let physical = rows
            .iter()
            .filter_map(|row| row.get(name))
            .find(|value| !value.is_null())
            .map_or("float64", |value| match value {
                Value::Bool(_) => "boolean",
                Value::Number(number) if number.is_i64() || number.is_u64() => "int64",
                Value::String(_) => "string",
                _ => "float64",
            });
        SchemaColumn {
            name: name.to_string(),
            physical: physical.to_string(),
            nullable: true,
            role: role.to_string(),
            semantic: None,
        }
    };
    let mut columns: Vec<SchemaColumn> = features
        .iter()
        .map(|feature| column(feature, "input"))
        .collect();
    columns.push(column(target, "target"));
    Schema {
        columns,
        rows: None,
        classes: Value::Null,
        frequency: None,
    }
}

fn mistyped(name: &str, physical: &str) -> RequestError {
    RequestError::new(
        "ValidationError",
        format!("column {name:?} holds values that are not {physical}"),
        422,
    )
}

/// JSON rows as Arrow columns of the schema's types, null where a row
/// lacks the column; dates and timestamps stay ISO strings, which the
/// binder's datetime columns parse.
///
/// # Errors
///
/// `ValidationError` for a value of another type than its column's, or a
/// column of a type JSON does not carry.
pub fn records(columns: &[SchemaColumn], rows: &[Value]) -> Result<RecordBatch, RequestError> {
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for column in columns {
        let name = column.name.as_str();
        let physical = column.physical.as_str();
        let cells = rows
            .iter()
            .map(|row| row.get(name).filter(|value| !value.is_null()));
        let (data_type, array): (DataType, ArrayRef) = match physical {
            "boolean" => (
                DataType::Boolean,
                Arc::new(
                    cells
                        .map(|cell| cell.map(|value| value.as_bool().ok_or(())).transpose())
                        .collect::<Result<BooleanArray, ()>>()
                        .map_err(|()| mistyped(name, physical))?,
                ),
            ),
            "int64" => (
                DataType::Int64,
                Arc::new(
                    cells
                        .map(|cell| cell.map(|value| value.as_i64().ok_or(())).transpose())
                        .collect::<Result<Int64Array, ()>>()
                        .map_err(|()| mistyped(name, physical))?,
                ),
            ),
            "float64" => (
                DataType::Float64,
                Arc::new(
                    cells
                        .map(|cell| cell.map(|value| value.as_f64().ok_or(())).transpose())
                        .collect::<Result<Float64Array, ()>>()
                        .map_err(|()| mistyped(name, physical))?,
                ),
            ),
            "string" | "date" | "timestamp" => (
                DataType::Utf8,
                Arc::new(
                    cells
                        .map(|cell| cell.map(|value| value.as_str().ok_or(())).transpose())
                        .collect::<Result<StringArray, ()>>()
                        .map_err(|()| mistyped(name, physical))?,
                ),
            ),
            _ => {
                return Err(RequestError::new(
                    "ValidationError",
                    format!("column {name:?} is {physical}, which JSON rows do not carry"),
                    422,
                ));
            }
        };
        fields.push(Field::new(name, data_type, true));
        arrays.push(array);
    }
    RecordBatch::try_new_with_options(
        Arc::new(ArrowSchema::new(fields)),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )
    .map_err(|error| RequestError::invalid(error.to_string()))
}

/// A snapshot's rows, from the Arrow IPC stream `gf serve` wrote.
///
/// # Errors
///
/// The file's path and why it could not be read.
pub fn read_snapshot(path: &Path) -> Result<RecordBatch, String> {
    let failed = |error: &dyn std::fmt::Display| format!("{}: {error}", path.display());
    let file = File::open(path).map_err(|error| failed(&error))?;
    let reader = StreamReader::try_new(file, None).map_err(|error| failed(&error))?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| failed(&error))?;
    arrow_select::concat::concat_batches(&schema, &batches).map_err(|error| failed(&error))
}
