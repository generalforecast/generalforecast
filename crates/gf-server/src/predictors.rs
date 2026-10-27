// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Locked tabular predictors on the Models API: what `gf serve` says of one,
//! and its JSON rows as the Arrow columns `gf_core::predictor`, the one
//! binder, encodes.
//!
//! A snapshot predictor's rows are bound once, when the server starts, into a
//! durable table; a request-context predictor binds each request's rows into
//! the ephemeral table their encoding names, as a tabular request does.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type, TimestampMillisecondType};
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, RecordBatchOptions, StringArray,
    TimestampMicrosecondArray,
};
use arrow_cast::cast;
use arrow_ipc::reader::StreamReader;
use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use gf_core::catalog::{self, Task};
use gf_core::civil::{Grammar, civil, iso_micros};
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
    /// The engine strategy each query's context rows are drawn by, as
    /// `SelectionInfo.strategy_spec` spells the lock's selection; exact
    /// when None.
    #[serde(default)]
    pub strategy: Option<Value>,
    /// The most context rows one request may carry, for a request context.
    #[serde(default)]
    pub max_rows: Option<usize>,
    /// The kinds the lock declares for its columns, by name: `int`,
    /// `float`, `0/1`, `category`, `text`, `timestamp` or `bool`. A context
    /// the lock holds no schema for is typed by them, as the binder types it
    /// in process; a column without one is typed by every value it holds.
    #[serde(default)]
    pub types: std::collections::BTreeMap<String, String>,
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
    /// A calendar frequency's alias, such as `MS`, `2QE` or `W-MON`, as the
    /// lock spells it. Times are then the ordinals of their UTC periods and
    /// `frequency` a count of periods; context times must sit on the alias's
    /// anchors at one time of day, as `frequency.check` holds them.
    #[serde(default)]
    pub calendar: Option<String>,
    /// The series forecast jointly, as the lock's `related` declares them:
    /// None for every series on its own, an empty list for one group of
    /// every series, else the series columns whose values name each group.
    #[serde(default)]
    pub related: Option<Vec<String>>,
}

/// A forecast predictor's context, aligned as `ForecastContext.build` aligns
/// it in process: each series' context rows and origin, and the groups its
/// lock declares related.
#[derive(Clone, Debug)]
pub struct ForecastContext {
    /// Each series' values of the lock's series columns, in context order.
    pub identities: Vec<Vec<Value>>,
    /// Each series' index, by its values as JSON text.
    pub index: HashMap<String, usize>,
    /// Each series' last time on the frequency's axis, and as JSON.
    pub origins: Vec<(f64, Value)>,
    pub step: f64,
    /// Each series' group, and each group's series.
    pub group_of: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
    /// Each series' context rows, as the forecast request takes them.
    pub rows: Vec<Vec<Value>>,
    /// The request's model, target and series options.
    pub options: Map<String, Value>,
}

/// Where a forecast predictor's queries fall: each row's series and horizon
/// position, and each queried series' known-future values.
#[derive(Clone, Debug)]
pub struct ForecastQueries {
    pub positions: Vec<(usize, usize)>,
    /// Each queried series' future values by input, in ascending series
    /// order, as its query carries them.
    pub futures: std::collections::BTreeMap<usize, Map<String, Value>>,
}

impl ForecastContext {
    /// The forecast request for queried series and their known-future
    /// values: the context rows of every group a queried series is in, and
    /// a query per queried series in ascending order, as
    /// `ForecastContext.request` writes it.
    #[must_use]
    pub fn request(
        &self,
        futures: &std::collections::BTreeMap<usize, Map<String, Value>>,
    ) -> Value {
        let mut groups: Vec<usize> = futures
            .keys()
            .map(|&series| self.group_of[series])
            .collect();
        groups.sort_unstable();
        groups.dedup();
        let context: Vec<Value> = groups
            .iter()
            .flat_map(|&group| self.groups[group].iter())
            .flat_map(|&series| self.rows[series].iter().cloned())
            .collect();
        let queries: Vec<Value> = futures
            .iter()
            .map(|(&series, future)| {
                let group: Vec<String> = self.groups[self.group_of[series]]
                    .iter()
                    .map(|member| format!("gfseries:{member}"))
                    .collect();
                json!({
                    "key": format!("q{series}"),
                    "series": format!("gfseries:{series}"),
                    "origin": self.origins[series].1,
                    "group": group,
                    "future_covariates": future,
                })
            })
            .collect();
        let mut request = self.options.clone();
        request.insert("context".to_string(), Value::Array(context));
        request.insert("queries".to_string(), Value::Array(queries));
        Value::Object(request)
    }
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

/// A time column's places on its frequency's axis: periods as numbers,
/// dates and times as UTC seconds, as `frequency.instant` reads them, or as
/// the ordinals of their UTC calendar periods, as `frequency.ordinal`
/// counts them, when the frequency steps by the calendar.
fn instants(array: &ArrayRef, calendar: Option<&str>) -> Result<Vec<Value>, RequestError> {
    let calendar = calendar.map(unit);
    let data_type = array.data_type();
    if data_type.is_integer() || data_type.is_floating() {
        if calendar.is_some() {
            return Err(refused("a calendar frequency needs dates or timestamps"));
        }
        cells(array)
    } else if data_type.is_temporal() || matches!(data_type, DataType::Utf8 | DataType::LargeUtf8) {
        let times = seconds(array)?;
        match calendar {
            None => Ok(times),
            Some(unit) => times
                .into_iter()
                .map(|time| match time.as_f64() {
                    Some(at) => ordinal(at, unit).map(|period| json!(period)),
                    None => Ok(Value::Null),
                })
                .collect(),
        }
    } else {
        Err(refused("forecast time must be numeric, date or timestamp"))
    }
}

/// The period a calendar alias counts: `M`, `Q`, `Y` or `W-<DAY>`, as
/// `frequency.calendar_unit` names it.
fn unit(alias: &str) -> &str {
    let alias = alias.trim_start_matches(|c: char| c.is_ascii_digit());
    if alias.starts_with("W-") {
        alias
    } else {
        alias.get(..1).unwrap_or(alias)
    }
}

const WEEKDAYS: [&str; 7] = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"];

/// Whether a UTC instant falls on one of a calendar alias's anchors: a
/// month, quarter or year start or end, or an anchored week's day.
fn anchored(seconds: f64, alias: &str) -> bool {
    #[allow(clippy::cast_possible_truncation)]
    let days = (seconds / 86_400.0).floor() as i64;
    let (_, month, day) = civil(days);
    let starts = day == 1;
    let ends = civil(days + 1).2 == 1;
    let alias = alias.trim_start_matches(|c: char| c.is_ascii_digit());
    match alias {
        "MS" => starts,
        "ME" => ends,
        "QS" => starts && month % 3 == 1,
        "QE" => ends && month % 3 == 0,
        "YS" => starts && month == 1,
        "YE" => ends && month == 12,
        _ => alias.strip_prefix("W-").is_some_and(|anchor| {
            // 1970-01-01 was a Thursday, index 3.
            let weekday = usize::try_from((days + 3).rem_euclid(7)).unwrap_or_default();
            WEEKDAYS[weekday] == anchor
        }),
    }
}

/// Which UTC calendar period an instant falls in, counted from 1970, as
/// `frequency.ordinal` counts it: months, quarters, years, or weeks
/// anchored on a day from that day's first occurrence on or after
/// 1970-01-01.
fn ordinal(seconds: f64, unit: &str) -> Result<i64, RequestError> {
    #[allow(clippy::cast_possible_truncation)]
    let days = (seconds / 86_400.0).floor() as i64;
    let (year, month, _) = civil(days);
    match unit {
        "M" => Ok((year - 1970) * 12 + month - 1),
        "Q" => Ok((year - 1970) * 4 + (month - 1) / 3),
        "Y" => Ok(year - 1970),
        _ => {
            let day = unit
                .strip_prefix("W-")
                .and_then(|day| WEEKDAYS.iter().position(|known| *known == day))
                .ok_or_else(|| refused("unknown calendar frequency"))?;
            // 1970-01-01 was a Thursday, index 3.
            let anchor = (i64::try_from(day).unwrap_or_default() - 3).rem_euclid(7);
            Ok((days - anchor).div_euclid(7))
        }
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

    /// The past and known future inputs, the columns the lock lists as a
    /// forecaster's inputs.
    #[must_use]
    pub fn covariates(&self) -> Vec<String> {
        [self.past_inputs.as_slice(), self.future_inputs.as_slice()].concat()
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
        let times = instants(column(&self.time)?, self.calendar.as_deref())?;
        let clock = match &self.calendar {
            Some(_) => Some(seconds(column(&self.time)?)?),
            None => None,
        };
        let targets = cells(column(target)?)?;
        let inputs: Vec<&String> = self.past_inputs.iter().chain(&self.future_inputs).collect();
        let covariates = inputs
            .iter()
            .map(|name| cells(column(name)?))
            .collect::<Result<Vec<_>, _>>()?;
        let mut identities: Vec<Vec<Value>> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut ends: Vec<(f64, Value)> = Vec::new();
        let mut moments: Vec<Vec<f64>> = Vec::new();
        let mut days: Vec<Option<f64>> = Vec::new();
        let mut series_rows: Vec<Vec<Value>> = Vec::new();
        for row in 0..rows.num_rows() {
            let identity: Vec<Value> = series.iter().map(|column| column[row].clone()).collect();
            let moment = &times[row];
            let at = moment
                .as_f64()
                .filter(|_| identity.iter().all(|value| !value.is_null()))
                .ok_or_else(|| refused("forecast time and series cannot be missing"))?;
            let key = Value::Array(identity.clone()).to_string();
            let at_series = if let Some(&known) = index.get(&key) {
                known
            } else {
                index.insert(key, identities.len());
                identities.push(identity);
                ends.push((at, moment.clone()));
                moments.push(Vec::new());
                days.push(None);
                series_rows.push(Vec::new());
                identities.len() - 1
            };
            if let (Some(alias), Some(clock)) = (&self.calendar, &clock) {
                // One rule with `frequency.check`: each time on an anchor,
                // every time of a series at the same time of day.
                let moment = clock[row].as_f64().unwrap_or_default();
                let of_day = moment.rem_euclid(86_400.0);
                if !anchored(moment, alias)
                    || days[at_series].is_some_and(|known| known.total_cmp(&of_day).is_ne())
                {
                    return Err(refused("declared frequency differs from observed times"));
                }
                days[at_series] = Some(of_day);
            }
            if at > ends[at_series].0 {
                ends[at_series] = (at, moment.clone());
            }
            moments[at_series].push(at);
            let values: Map<String, Value> = inputs
                .iter()
                .zip(&covariates)
                .map(|(name, column)| ((*name).clone(), column[row].clone()))
                .collect();
            series_rows[at_series].push(json!({
                "key": row.to_string(),
                "series": format!("gfseries:{at_series}"),
                "time": moment,
                "value": targets[row],
                "covariates": values,
            }));
        }
        if identities.is_empty() {
            return Err(RequestError::new(
                "EmptyContext",
                "a forecast context has no series",
                422,
            ));
        }
        let (group_of, groups) = self.groups(&identities)?;
        for members in &groups {
            if members
                .iter()
                .any(|&member| ends[member].0.total_cmp(&ends[members[0]].0).is_ne())
            {
                return Err(refused(
                    "related series must share the same forecast origin",
                ));
            }
        }
        let frequency = match frequency {
            Some(frequency) => frequency,
            None => self.observed(&mut moments)?,
        };
        let step = if ends[0].1.is_i64() && frequency.fract() == 0.0 {
            json!(frequency as i64)
        } else {
            json!(frequency)
        };
        let Value::Object(options) = json!({
            "model": model,
            "task": "forecast",
            "target": target,
            "features": [],
            "options": {
                "series": {
                    "step": step,
                    "horizon": self.horizon,
                    "quantiles": levels,
                    "known_covariates": self.future_inputs,
                }
            },
        }) else {
            unreachable!("a JSON object literal")
        };
        Ok(ForecastContext {
            identities,
            index,
            origins: ends,
            step: frequency,
            group_of,
            groups,
            rows: series_rows,
            options,
        })
    }

    /// Each series' group, and each group's series, by the lock's `related`,
    /// as `forecast.series_groups` makes them.
    fn groups(
        &self,
        identities: &[Vec<Value>],
    ) -> Result<(Vec<usize>, Vec<Vec<usize>>), RequestError> {
        let Some(related) = &self.related else {
            return Ok((
                (0..identities.len()).collect(),
                (0..identities.len()).map(|series| vec![series]).collect(),
            ));
        };
        let positions: Vec<usize> = related
            .iter()
            .map(|column| {
                self.series
                    .iter()
                    .position(|name| name == column)
                    .ok_or_else(|| refused("related groups are named by series columns"))
            })
            .collect::<Result<_, _>>()?;
        let mut keys: HashMap<String, usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut group_of = Vec::with_capacity(identities.len());
        for (series, identity) in identities.iter().enumerate() {
            let key = Value::Array(positions.iter().map(|&at| identity[at].clone()).collect())
                .to_string();
            let group = *keys.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[group].push(series);
            group_of.push(group);
        }
        let most = catalog::catalog().policy.group_series;
        if let Some(largest) = groups
            .iter()
            .map(Vec::len)
            .max()
            .filter(|&largest| largest > most)
        {
            return Err(RequestError::new(
                "CapacityExceeded",
                format!("a related group holds {largest} series; at most {most} forecast together"),
                422,
            ));
        }
        Ok((group_of, groups))
    }

    /// The one step between each series' times, as the lock's compiler
    /// verifies it.
    fn observed(&self, moments: &mut [Vec<f64>]) -> Result<f64, RequestError> {
        let mut steps: Vec<f64> = Vec::new();
        for times in moments.iter_mut() {
            if times.len() < 2 {
                // A series of one time follows a declared frequency, as
                // `frequency.check` holds it.
                if self.declared.is_some() {
                    continue;
                }
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
        if steps.is_empty()
            && let Some(declared) = self.declared
        {
            return Ok(declared);
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
        records(&columns, queries).map_err(|error| {
            refused(&format!(
                "query physical type differs from lock: {}",
                error.message
            ))
        })
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
        let times = instants(column(&self.time)?, self.calendar.as_deref())?;
        let inputs = self
            .future_inputs
            .iter()
            .map(|name| cells(column(name)?))
            .collect::<Result<Vec<_>, _>>()?;
        let mut positions = Vec::with_capacity(rows.num_rows());
        let mut seen: HashMap<usize, Vec<bool>> = HashMap::new();
        // Known-future values by owner: a series on its own, or its related
        // group (as `usize::MAX - group`), which shares them.
        let mut future: HashMap<(usize, usize, usize), f64> = HashMap::new();
        let owner = |member: usize| {
            let group = context.group_of[member];
            if context.groups[group].len() > 1 {
                usize::MAX - group
            } else {
                member
            }
        };
        for row in 0..rows.num_rows() {
            let identity: Vec<Value> = series.iter().map(|column| column[row].clone()).collect();
            let member = *context
                .index
                .get(&Value::Array(identity).to_string())
                .ok_or_else(|| refused("query series is missing from context"))?;
            let origin = context.origins[member].0;
            let position = times[row]
                .as_f64()
                .map(|at| (at - origin) / context.step - 1.0)
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
            let held = seen
                .entry(member)
                .or_insert_with(|| vec![false; self.horizon]);
            if std::mem::replace(&mut held[position], true) {
                return Err(refused("duplicate forecast query time within a series"));
            }
            positions.push((member, position));
            for (input, column) in inputs.iter().enumerate() {
                let value = match &column[row] {
                    Value::Bool(value) => Some(f64::from(u8::from(*value))),
                    Value::Number(value) => value.as_f64(),
                    _ => None,
                }
                .filter(|value| value.is_finite() && value.abs() <= 1e12)
                .ok_or_else(|| refused("known future covariates must be bounded numeric values"))?;
                let key = (owner(member), input, position);
                if future
                    .get(&key)
                    .is_some_and(|held| held.total_cmp(&value).is_ne())
                {
                    return Err(RequestError::new(
                        "UnsupportedCovariates",
                        "related series must share named future covariates",
                        422,
                    ));
                }
                future.insert(key, value);
            }
        }
        if seen.values().any(|held| !held.iter().all(|&held| held)) {
            return Err(refused(
                "queries must include the complete horizon for each requested series",
            ));
        }
        let futures = seen
            .keys()
            .map(|&member| {
                let values: Map<String, Value> = self
                    .future_inputs
                    .iter()
                    .enumerate()
                    .map(|(input, name)| {
                        let steps: Vec<Value> = (0..self.horizon)
                            .map(|step| json!(future[&(owner(member), input, step)]))
                            .collect();
                        (name.clone(), Value::Array(steps))
                    })
                    .collect();
                (member, values)
            })
            .collect();
        Ok(ForecastQueries { positions, futures })
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

    /// The lock's inputs as a predictor resource lists them: each column's
    /// name, its type (`number`, `category`, `text`, `boolean`, `date` or
    /// `timestamp`), and its role: `input` for a tabular predictor, `past`
    /// or `future` for a forecaster's covariates.
    #[must_use]
    pub fn described_inputs(&self) -> Vec<Value> {
        let roles: Vec<(&str, &str)> = match &self.forecast {
            Some(forecast) => forecast
                .past_inputs
                .iter()
                .map(|name| (name.as_str(), "past"))
                .chain(
                    forecast
                        .future_inputs
                        .iter()
                        .map(|name| (name.as_str(), "future")),
                )
                .collect(),
            None => self
                .inputs
                .iter()
                .map(|name| (name.as_str(), "input"))
                .collect(),
        };
        roles
            .into_iter()
            .map(|(name, role)| json!({"name": name, "type": self.column_type(name), "role": role}))
            .collect()
    }

    /// A column's kind, in the one vocabulary `gf.lock` and the API's `Kind`
    /// use: the kind the lock declares, else the kind its locked storage
    /// holds, else null for a column a lock checked without rows never saw.
    fn column_type(&self, name: &str) -> Option<&str> {
        if let Some(kind) = self.types.get(name) {
            return Some(kind.as_str());
        }
        self.schema
            .columns
            .iter()
            .find(|column| column.name == name)
            .map(
                |column| match (column.physical.as_str(), column.semantic.as_deref()) {
                    (_, Some("category")) | ("string", _) => "category",
                    ("boolean", _) => "bool",
                    ("date" | "timestamp", _) => "timestamp",
                    ("int64", _) => "int",
                    _ => "float",
                },
            )
    }

    /// Whether each request names its own features and target, because the
    /// lock holds no schema.
    #[must_use]
    pub fn schema_free(&self) -> bool {
        self.context == ContextMode::Request && self.schema.columns.is_empty()
    }
}

/// The Arrow storage a declared kind takes, as `tables.PHYSICAL` has it.
fn declared_physical(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "int" | "0/1" => "int64",
        "float" => "float64",
        "category" | "text" => "string",
        "timestamp" => "timestamp",
        "bool" => "boolean",
        _ => return None,
    })
}

/// The storage a column of JSON values takes, judged by all of them, as
/// Arrow types a list of them: text if any is text, else a float if any
/// number has a fraction, else whole numbers, else booleans; numbers if it
/// holds no value at all.
fn observed_physical<'a>(values: impl Iterator<Item = &'a Value>) -> &'static str {
    let (mut text, mut fraction, mut whole, mut boolean) = (false, false, false, false);
    for value in values {
        match value {
            Value::String(_) => text = true,
            Value::Number(number) if number.is_i64() || number.is_u64() => whole = true,
            Value::Number(_) => fraction = true,
            Value::Bool(_) => boolean = true,
            _ => {}
        }
    }
    if text {
        "string"
    } else if fraction {
        "float64"
    } else if whole {
        "int64"
    } else if boolean {
        "boolean"
    } else {
        "float64"
    }
}

/// A column's storage and the binder's override for it, from the kind the
/// lock declares and the storage its values take, as the binder types it in
/// process (`tables.semantic`): a time the lock calls a timestamp is one, a
/// float column of whole numbers widens, and a category or boolean stored
/// otherwise is read as categories.
///
/// # Errors
///
/// `ValidationError` naming `inputs.types` for a kind the values cannot
/// hold, such as a float stored as text.
fn kind_storage(
    name: &str,
    kind: Option<&str>,
    observed: Option<&'static str>,
) -> Result<(&'static str, Option<&'static str>), RequestError> {
    let Some(kind) = kind else {
        return Ok((observed.unwrap_or("float64"), None));
    };
    let Some(observed) = observed else {
        return Ok((declared_physical(kind).unwrap_or("float64"), None));
    };
    let categorical = matches!(kind, "category" | "text");
    Ok(match (kind, observed) {
        ("timestamp", "string" | "int64" | "timestamp") => ("timestamp", None),
        ("float", "int64") => ("float64", None),
        ("int" | "float" | "0/1", "int64" | "float64" | "boolean") => (observed, None),
        (_, "int64" | "boolean") if categorical => (observed, Some("category")),
        (_, "string") if categorical => ("string", None),
        ("bool", "boolean") => ("boolean", None),
        ("bool", "string" | "int64") => (observed, Some("category")),
        _ => {
            return Err(RequestError::new(
                "ValidationError",
                format!("column {name:?} holds {observed} values, which cannot be read as {kind}"),
                422,
            )
            .param("inputs.types"));
        }
    })
}

/// A schema for rows a lock holds none for: `features` as inputs and
/// `target`, each typed by the kind `types` declares for it and the storage
/// every value the rows hold takes, so a whole first value of a float
/// column, or a time the lock calls a timestamp, types the column as the
/// binder does in process. One of `inputs` whose kind the lock does not
/// record reads text that is all ISO 8601 times as UTC timestamps, as
/// `runtime.context.typed` does (see `iso_times`).
///
/// # Errors
///
/// As `kind_storage`, for a kind a column's values cannot hold.
pub fn inferred(
    features: &[String],
    target: &str,
    rows: &[Value],
    types: &std::collections::BTreeMap<String, String>,
    inputs: &[String],
) -> Result<Schema, RequestError> {
    let column = |name: &str, role: &str| -> Result<SchemaColumn, RequestError> {
        let values: Vec<&Value> = rows
            .iter()
            .filter_map(|row| row.get(name))
            .filter(|value| !value.is_null())
            .collect();
        let observed = (!values.is_empty()).then(|| observed_physical(values.iter().copied()));
        let kind = types.get(name).map(String::as_str);
        let unrecorded = kind.is_none() && inputs.iter().any(|input| input == name);
        let (physical, semantic) =
            if unrecorded && observed == Some("string") && iso_times(values.iter().copied()) {
                ("timestamp", None)
            } else {
                kind_storage(name, kind, observed)?
            };
        Ok(SchemaColumn {
            name: name.to_string(),
            physical: physical.to_string(),
            nullable: true,
            role: role.to_string(),
            semantic: semantic.map(str::to_string),
        })
    };
    let mut columns = features
        .iter()
        .map(|feature| column(feature, "input"))
        .collect::<Result<Vec<_>, _>>()?;
    columns.push(column(target, "target")?);
    Ok(Schema {
        columns,
        rows: None,
        classes: Value::Null,
        frequency: None,
    })
}

/// Refuses a column whose row holds `value`, which its declared type does
/// not take, naming both.
fn mistyped(name: &str, physical: &str, value: &Value) -> RequestError {
    let held = match value {
        Value::Bool(_) => "a boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "a whole number",
        Value::Number(_) => "a number with a fraction",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
        Value::Null => "null",
    };
    let takes = match physical {
        "boolean" => "true or false",
        "int64" => "whole numbers",
        "float64" => "numbers",
        "timestamp" => "ISO 8601 strings or whole microseconds since 1970 in UTC",
        _ => "strings",
    };
    RequestError::new(
        "ValidationError",
        format!(
            "column {name:?} is declared {physical}, which takes {takes}, but a row holds {held}"
        ),
        422,
    )
}

/// A column's cells, each its row's value or None where the row lacks it,
/// as `read` takes them; the first a column's type does not take is refused.
fn typed<T>(
    name: &str,
    physical: &str,
    cells: &[Option<&Value>],
    read: impl Fn(&Value) -> Option<T>,
) -> Result<Vec<Option<T>>, RequestError> {
    cells
        .iter()
        .map(|cell| {
            cell.map(|value| read(value).ok_or_else(|| mistyped(name, physical, value)))
                .transpose()
        })
        .collect()
}

/// The ISO 8601 times a context or query cell holds, as UTC microseconds
/// and whether the text names a zone: `civil::iso_micros`'s value grammar,
/// the one Python's `_timestamps` reads.
fn iso_time(text: &str) -> Option<(i64, bool)> {
    iso_micros(text, Grammar::Value)
}

/// Whether JSON values are all ISO 8601 times by `iso_time`, every one
/// naming a zone or none: what Python's `_timestamps` reads as a column of
/// UTC timestamps, trying a naive cast and then a UTC one.
fn iso_times<'a>(values: impl Iterator<Item = &'a Value>) -> bool {
    let mut zoned = None;
    for value in values {
        let Some((_, named)) = value.as_str().and_then(iso_time) else {
            return false;
        };
        if *zoned.get_or_insert(named) != named {
            return false;
        }
    }
    zoned.is_some()
}

/// A timestamp column's cells as ISO 8601 strings, the binder's datetime
/// columns parse: an ISO time `iso_time` reads, and whole numbers, which
/// the declared type counts in microseconds since 1970 in UTC, written out
/// in full; other strings as given.
fn timestamps(name: &str, cells: &[Option<&Value>]) -> Result<StringArray, RequestError> {
    let failed = |error: arrow_schema::ArrowError| {
        RequestError::new(
            "ValidationError",
            format!("column {name:?} holds a time that cannot be written out: {error}"),
            422,
        )
    };
    let micros = TimestampMicrosecondArray::from(
        cells
            .iter()
            .map(|cell| match cell {
                Some(Value::String(text)) => iso_time(text).map(|(micros, _)| micros),
                Some(value) => value.as_i64(),
                None => None,
            })
            .collect::<Vec<_>>(),
    )
    .with_timezone("+00:00");
    let written = cast(&micros, &DataType::Utf8).map_err(failed)?;
    let written = written.as_string::<i32>();
    cells
        .iter()
        .enumerate()
        .map(|(row, cell)| match cell {
            None => Ok(None),
            Some(Value::String(_)) if written.is_valid(row) => {
                Ok(Some(written.value(row).to_string()))
            }
            Some(Value::String(text)) => Ok(Some(text.clone())),
            Some(value) if value.as_i64().is_some() => Ok(Some(written.value(row).to_string())),
            Some(value) => Err(mistyped(name, "timestamp", value)),
        })
        .collect()
}

/// JSON rows as Arrow columns of the schema's types, null where a row
/// lacks the column; dates and timestamps stay ISO strings, which the
/// binder's datetime columns parse, and a timestamp may be given as the
/// whole microseconds since 1970 its declared type counts.
///
/// # Errors
///
/// `ValidationError` naming the column, its declared type and what a row
/// holds, for a value its type does not take, or a column of a type JSON
/// does not carry.
pub fn records(columns: &[SchemaColumn], rows: &[Value]) -> Result<RecordBatch, RequestError> {
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for column in columns {
        let name = column.name.as_str();
        let physical = column.physical.as_str();
        let cells: Vec<Option<&Value>> = rows
            .iter()
            .map(|row| row.get(name).filter(|value| !value.is_null()))
            .collect();
        let (data_type, array): (DataType, ArrayRef) = match physical {
            "boolean" => (
                DataType::Boolean,
                Arc::new(BooleanArray::from(typed(
                    name,
                    physical,
                    &cells,
                    Value::as_bool,
                )?)),
            ),
            "int64" => (
                DataType::Int64,
                Arc::new(Int64Array::from(typed(
                    name,
                    physical,
                    &cells,
                    Value::as_i64,
                )?)),
            ),
            "float64" => (
                DataType::Float64,
                Arc::new(Float64Array::from(typed(
                    name,
                    physical,
                    &cells,
                    Value::as_f64,
                )?)),
            ),
            "timestamp" => (DataType::Utf8, Arc::new(timestamps(name, &cells)?)),
            "string" | "date" => (
                DataType::Utf8,
                Arc::new(StringArray::from(typed(name, physical, &cells, |value| {
                    value.as_str().map(str::to_string)
                })?)),
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

#[cfg(test)]
mod tests {
    use super::{civil, ordinal};
    use serde_json::{Value, json};

    #[test]
    fn iso_text_reads_as_python_reads_it() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/iso_times.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let text = case[0].as_str().unwrap();
            let read = super::iso_time(text).map(|(micros, _)| micros);
            assert_eq!(read, case[1].as_i64(), "{text:?}");
        }
        // Python reads a column naive or in UTC, never a mix of the two.
        let times = |values: Value| super::iso_times(values.as_array().unwrap().iter());
        assert!(times(json!(["2025-01-01", "2025-01-02 06:00"])));
        assert!(times(json!([
            "2025-01-01T00:00Z",
            "2025-01-02T06:00+01:00"
        ])));
        assert!(!times(json!([
            "2025-01-01T00:00",
            "2025-01-02T06:00+01:00"
        ])));
        assert!(!times(json!(["2025-01-01", "gold"])));
        assert!(!times(json!([])));
    }

    #[test]
    fn an_input_of_no_recorded_kind_reads_iso_text_as_utc_timestamps() {
        let rows = json!([
            {"seen": "2025-01-01T00:00:00Z", "tier": "gold", "day": "2025-01-01", "y": "2025-01-01"},
            {"seen": "2025-01-02T06:00:00+01:00", "tier": "2025-01-01", "day": "2025-01-02", "y": "2025-01-02"},
        ]);
        let rows = rows.as_array().unwrap();
        let features: Vec<String> = ["seen", "tier", "day"].map(String::from).to_vec();
        let types = [("day".to_string(), "category".to_string())].into();
        let schema = super::inferred(&features, "y", rows, &types, &features).unwrap();
        let physical: Vec<(&str, &str)> = schema
            .columns
            .iter()
            .map(|column| (column.name.as_str(), column.physical.as_str()))
            .collect();
        // A declared kind and the target keep their text; so does an input
        // whose text is not all times.
        assert_eq!(
            physical,
            [
                ("seen", "timestamp"),
                ("tier", "string"),
                ("day", "string"),
                ("y", "string")
            ]
        );
        let batch = super::records(&schema.columns, rows).unwrap();
        let seen = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        assert_eq!(seen.value(0), "2025-01-01T00:00:00Z");
        assert_eq!(seen.value(1), "2025-01-02T05:00:00Z");
        let unlisted = super::inferred(&features, "y", rows, &types, &[]).unwrap();
        assert_eq!(unlisted.columns[0].physical, "string");
    }

    #[test]
    fn civil_dates_follow_the_proleptic_calendar() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(-1), (1969, 12, 31));
        assert_eq!(civil(19_782), (2024, 2, 29));
        assert_eq!(civil(19_783), (2024, 3, 1));
    }

    #[test]
    fn anchors_and_units_follow_the_alias() {
        let day = 86_400.0;
        assert_eq!(super::unit("2QE"), "Q");
        assert_eq!(super::unit("W-MON"), "W-MON");
        assert_eq!(super::unit("MS"), "M");
        // 2024-01-01 starts a month, quarter and year; 2024-03-31 ends a quarter.
        for alias in ["MS", "QS", "YS", "W-MON"] {
            assert!(super::anchored(19_723.0 * day, alias), "{alias}");
        }
        assert!(super::anchored(19_813.0 * day, "QE"));
        assert!(super::anchored(19_813.0 * day, "ME"));
        assert!(!super::anchored(19_737.0 * day, "MS"));
        assert!(!super::anchored(19_723.0 * day, "W-TUE"));
    }

    #[test]
    fn ordinals_count_utc_calendar_periods_as_frequency_ordinal_does() {
        let day = 86_400.0;
        // 2024-01-01 and 2024-02-01, then 2024-04-01 and 2024-12-31.
        assert_eq!(ordinal(19_723.0 * day, "M").unwrap(), 648);
        assert_eq!(ordinal(19_754.0 * day, "M").unwrap(), 649);
        assert_eq!(ordinal(19_814.0 * day, "Q").unwrap(), 217);
        assert_eq!(ordinal(20_088.0 * day, "Y").unwrap(), 54);
        // Mondays: 1970-01-05 is the first, 1969-12-29 the one before.
        assert_eq!(ordinal(4.0 * day, "W-MON").unwrap(), 0);
        assert_eq!(ordinal(11.0 * day, "W-MON").unwrap(), 1);
        assert_eq!(ordinal(-3.0 * day, "W-MON").unwrap(), -1);
        // A time of day keeps its day.
        assert_eq!(ordinal(19_754.0 * day + 3_600.0, "M").unwrap(), 649);
        assert!(ordinal(0.0, "W-XYZ").is_err());
    }
}
