// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The ad hoc route: a manifest fragment and its rows, predicted as the
//! request-context predictor `hosting.ad_hoc_lock` resolves in process.
//!
//! The fragment becomes a [`ServedPredictor`] whose context comes with the
//! request and whose lock holds no schema, so the binder types its columns
//! from the rows as it does any request context: text and categories as the
//! in-process path encodes them, and a forecaster's series, time and
//! covariates on the series path every forecast predictor takes.

use std::collections::BTreeMap;

use super::incoming::fields;
use super::predictions::{declared_call, served_predict};
use super::*;

/// The fields a fragment may carry besides `model`, `target`, `context`
/// and `queries`.
const OPTIONAL: [&str; 9] = [
    "inputs",
    "id",
    "time",
    "series",
    "frequency",
    "horizon",
    "quantiles",
    "outputs",
    "selection",
];

/// The ad hoc route's body: a manifest fragment (`model`, `target`, and
/// optionally `inputs`, `id`, `time`, `series`, `frequency`, `horizon`,
/// `quantiles`, `outputs`; `selection` is refused) with the `context` and
/// `queries` rows, for a catalog model served here, predicted on `engine`.
///
/// # Errors
///
/// `ValidationError` for a fragment `gf check` would refuse, naming the
/// field; `CapacityExceeded` past the model's per-call bounds; and the
/// refusals of the predictor path it runs on.
pub(super) async fn adhoc_predict_with(
    front: &Front,
    engine: &Engine,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    refuse(
        payload,
        "task",
        "is not a field: the model determines the task",
    )?;
    refuse(
        payload,
        "options",
        "is not a field: id, quantiles and outputs are top-level fields",
    )?;
    refuse(payload, "features", "is spelled inputs")?;
    let body = fields(
        payload,
        &["model", "target", "context", "queries"],
        &OPTIONAL,
    )?;
    let model = catalog::canonical(body["model"].as_str(), front.surface)?;
    let served = front.serving(&model)?;
    let forecast = model.entry.task == catalog::Task::Forecast;
    let task = match model.entry.task {
        catalog::Task::Classification => "classification",
        catalog::Task::Regression => "regression",
        catalog::Task::Forecast => "forecast",
    };
    if !forecast {
        for field in ["time", "series", "frequency", "horizon"] {
            if body.contains_key(field) {
                return Err(ApiError::validation(format!(
                    "{field} is a forecast field; {} is a {task} model",
                    model.canonical_id()
                ))
                .param(field));
            }
        }
    }
    if body.contains_key("selection") {
        // A selection resolves against a model's row cap as `gf check`
        // resolves it; a predictor's lock carries its resolved selection.
        return Err(ApiError::validation(
            "selection is resolved into a predictor's gf.lock; serve a predictor to select rows",
        )
        .param("selection"));
    }
    let target = body["target"]
        .as_str()
        .filter(|target| !target.is_empty())
        .ok_or_else(|| ApiError::validation("target must name a column").param("target"))?;
    let (Some(rows), Some(queries)) = (body["context"].as_array(), body["queries"].as_array())
    else {
        return Err(ApiError::validation(
            "context and queries must be lists of rows",
        ));
    };
    require(
        rows.iter().chain(queries).all(Value::is_object),
        "rows must be objects",
    )?;
    bounded(&model, rows.len(), queries.len())?;
    let id = optional_text(body, "id")?;
    let layout = if forecast {
        Some(series_layout(body, rows)?)
    } else {
        None
    };
    let roles: Vec<&str> = [Some(target), id]
        .into_iter()
        .flatten()
        .chain(layout.iter().flat_map(|layout| layout.roles()))
        .collect();
    let (past, future) = inputs(body, rows, &roles, forecast)?;
    if past.len() + future.len() > catalog::catalog().policy.features {
        return Err(ApiError::validation("invalid feature list").param("inputs"));
    }
    let outputs = outputs(body, task, served)?;
    let quantiles = match body.get("quantiles") {
        None | Some(Value::Null) => DEFAULT_QUANTILES.to_vec(),
        Some(levels) => {
            require(
                outputs.iter().any(|output| output == "quantiles"),
                "quantiles require the quantiles output",
            )?;
            levels
                .as_array()
                .and_then(|levels| levels.iter().map(Value::as_f64).collect::<Option<_>>())
                .ok_or_else(|| ApiError::validation("invalid quantile levels").param("quantiles"))?
        }
    };
    let mut types = BTreeMap::new();
    let forecast = layout.map(|layout| {
        if layout.timed {
            types.insert(layout.time.clone(), "timestamp".to_string());
        }
        ForecastPredictor {
            series: layout.series,
            time: layout.time,
            horizon: layout.horizon,
            past_inputs: past.clone(),
            future_inputs: future.clone(),
            frequency: None,
            declared: layout.declared,
            calendar: layout.calendar,
            related: None,
        }
    });
    let predictor = ServedPredictor {
        schema: gf_core::predictor::Schema {
            columns: Vec::new(),
            rows: None,
            classes: Value::Null,
            frequency: None,
        },
        task: task.to_string(),
        target: Some(target.to_string()),
        inputs: past.iter().chain(&future).cloned().collect(),
        id: id.map(str::to_string),
        context: ContextMode::Request,
        quantiles: if task == "classification" {
            Vec::new()
        } else {
            quantiles
        },
        context_digest: None,
        lock_digest: String::new(),
        snapshot: None,
        // A request context's size is only known per call, so a tabular
        // fragment's selection is adaptive, as `ad_hoc_lock` resolves it; a
        // forecaster reads each series' whole history.
        strategy: (task != "forecast").then(|| json!({"kind": "adaptive"})),
        max_rows: None,
        forecast,
        types,
    };
    let mut adhoc = served.clone();
    adhoc.outputs = outputs;
    adhoc.predictor = Some(predictor.clone());
    let call = declared_call(
        &predictor,
        &json!({"queries": queries, "context": rows}),
        || Ok(json!([])),
    )?;
    served_predict(front, engine, &adhoc, &predictor, &call, charge, false).await
}

/// Refuses a field the fragment must not carry, saying why.
fn refuse(payload: &Value, field: &str, reason: &str) -> Result<(), ApiError> {
    if payload.get(field).is_some() {
        return Err(ApiError::validation(format!("{field} {reason}")).param(field));
    }
    Ok(())
}

/// A field naming a column, or None when the fragment leaves it out.
fn optional_text<'a>(
    body: &'a Map<String, Value>,
    field: &str,
) -> Result<Option<&'a str>, ApiError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if !text.is_empty() => Ok(Some(text.as_str())),
        Some(_) => Err(ApiError::validation(format!("{field} must name a column")).param(field)),
    }
}

/// The model's per-call bounds on a request's rows, as a tabular request
/// is held to them.
fn bounded(model: &catalog::Model, context: usize, queries: usize) -> Result<(), ApiError> {
    let policy = &catalog::catalog().policy;
    if context > policy.context_rows {
        return Err(RequestError::capacity(
            &model.to_string(),
            "context rows",
            policy.context_rows,
            context,
            "sample the context, or serve a predictor whose context is pinned",
        )
        .param("context")
        .into());
    }
    if queries > policy.query_rows {
        return Err(RequestError::capacity(
            &model.to_string(),
            "queries",
            policy.query_rows,
            queries,
            "split the queries, or push a predictor and call it in batches",
        )
        .param("queries")
        .into());
    }
    require(!queries.eq(&0), "queries must not be empty")?;
    Ok(())
}

/// The outputs asked for, every one the model has; by default a
/// classifier's label and probabilities, a regressor's value, and a
/// forecaster's value and quantiles.
fn outputs(
    body: &Map<String, Value>,
    task: &str,
    served: &ServedModel,
) -> Result<Vec<String>, ApiError> {
    let refused = || ApiError::validation("model does not support requested outputs");
    let outputs: Vec<String> = match body.get("outputs") {
        Some(Value::Array(outputs)) if !outputs.is_empty() => outputs
            .iter()
            .map(|output| output.as_str().map(str::to_string))
            .collect::<Option<_>>()
            .ok_or_else(refused)?,
        Some(Value::Array(_)) | None | Some(Value::Null) => match task {
            "classification" => vec!["label".into(), "probabilities".into()],
            "forecast" => vec!["value".into(), "quantiles".into()],
            _ if body
                .get("quantiles")
                .is_some_and(|levels| !levels.is_null()) =>
            {
                vec!["value".into(), "quantiles".into()]
            }
            _ => vec!["value".into()],
        },
        Some(_) => return Err(refused()),
    };
    let unique: std::collections::HashSet<&String> = outputs.iter().collect();
    if unique.len() != outputs.len()
        || !outputs.iter().all(|output| served.outputs.contains(output))
    {
        return Err(refused());
    }
    Ok(outputs)
}

/// A forecaster fragment's series, time, horizon and frequency.
struct Layout {
    series: Vec<String>,
    time: String,
    horizon: usize,
    /// The step a declared frequency takes on the time axis: seconds for a
    /// span, periods for a calendar alias or a number.
    declared: Option<f64>,
    /// A declared calendar alias, such as `MS` or `W-MON`.
    calendar: Option<String>,
    /// Whether the times are dates or timestamps, not numbered periods.
    timed: bool,
}

impl Layout {
    fn roles(&self) -> impl Iterator<Item = &str> {
        self.series
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(self.time.as_str()))
    }
}

fn series_layout(body: &Map<String, Value>, rows: &[Value]) -> Result<Layout, ApiError> {
    let time = optional_text(body, "time")?
        .ok_or_else(|| ApiError::validation("a forecast names its time column").param("time"))?;
    let series = match body.get("series") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(column)) => vec![column.clone()],
        Some(Value::Array(columns)) => columns
            .iter()
            .map(|column| column.as_str().map(str::to_string))
            .collect::<Option<_>>()
            .ok_or_else(|| ApiError::validation("series must name columns").param("series"))?,
        Some(_) => return Err(ApiError::validation("series must name columns").param("series")),
    };
    let horizon = body
        .get("horizon")
        .and_then(Value::as_u64)
        .and_then(|horizon| usize::try_from(horizon).ok())
        .filter(|horizon| *horizon >= 1)
        .ok_or_else(|| {
            ApiError::validation("a forecast's horizon is a positive number of steps")
                .param("horizon")
        })?;
    let (declared, calendar, timed) = match body.get("frequency") {
        // Without a declared frequency the context's times show the step;
        // ISO text steps through times, numbers through periods.
        None | Some(Value::Null) => (
            None,
            None,
            rows.iter()
                .any(|row| row.get(time).is_some_and(Value::is_string)),
        ),
        Some(value) => {
            let (step, calendar, timed) = frequency(value)?;
            (Some(step), calendar, timed)
        }
    };
    Ok(Layout {
        series,
        time: time.to_string(),
        horizon,
        declared,
        calendar,
        timed,
    })
}

/// A declared frequency as `frequency.normalize` reads it: a positive
/// number of periods; a span such as `1d`, `15min` or pandas' `H`, in
/// seconds; or a calendar alias such as `MS`, `2QE` or `W-MON`, in periods.
/// Returns the step, the alias, and whether it steps through times.
fn frequency(value: &Value) -> Result<(f64, Option<String>, bool), ApiError> {
    let refused = || {
        ApiError::validation("frequency must be a positive number, a span such as 1d, or a calendar alias such as MS")
            .param("frequency")
    };
    if let Some(number) = value.as_f64().filter(|_| !value.is_boolean()) {
        return (number.is_finite() && number > 0.0)
            .then_some((number, None, false))
            .ok_or_else(refused);
    }
    let text = value.as_str().ok_or_else(refused)?.trim();
    let digits = text.chars().take_while(char::is_ascii_digit).count();
    let (count, unit) = text.split_at(digits);
    let count: u32 = if count.is_empty() {
        1
    } else {
        count
            .parse()
            .ok()
            .filter(|count| *count > 0)
            .ok_or_else(refused)?
    };
    let unit = unit.trim();
    let span = match unit {
        "ms" | "L" | "l" => Some(0.001),
        "s" | "S" | "sec" => Some(1.0),
        "min" | "T" | "t" | "m" => Some(60.0),
        "h" | "H" | "hr" => Some(3600.0),
        "d" | "D" => Some(86400.0),
        "w" | "W" => Some(604_800.0),
        _ => None,
    };
    if let Some(seconds) = span {
        return Ok((f64::from(count) * seconds, None, true));
    }
    let upper = unit.to_ascii_uppercase();
    let weekdays = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"];
    let alias = match unit {
        "MS" => "MS",
        "M" | "ME" => "ME",
        "QS" => "QS",
        "Q" | "QE" => "QE",
        "YS" | "AS" => "YS",
        "Y" | "A" | "YE" => "YE",
        _ if upper
            .strip_prefix("W-")
            .is_some_and(|day| weekdays.contains(&day)) =>
        {
            upper.as_str()
        }
        _ => return Err(refused()),
    };
    let spelled = if count == 1 {
        alias.to_string()
    } else {
        format!("{count}{alias}")
    };
    Ok((f64::from(count), Some(spelled), true))
}

/// The input columns, split into past and future covariates for a
/// forecaster: a list, every one of them past for a forecaster; or a
/// forecaster's `{past, future}`; or, when left out, every column of the
/// first context row that has no role.
fn inputs(
    body: &Map<String, Value>,
    rows: &[Value],
    roles: &[&str],
    forecast: bool,
) -> Result<(Vec<String>, Vec<String>), ApiError> {
    let names = |value: Option<&Value>| -> Result<Vec<String>, ApiError> {
        match value {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(columns)) => columns
                .iter()
                .map(|column| column.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| {
                    ApiError::validation("inputs must be a list of column names").param("inputs")
                }),
            Some(_) => {
                Err(ApiError::validation("inputs must be a list of column names").param("inputs"))
            }
        }
    };
    let (past, future) = match body.get("inputs") {
        None | Some(Value::Null) => {
            let inferred: Vec<String> = rows
                .first()
                .and_then(Value::as_object)
                .map(|row| {
                    row.keys()
                        .filter(|column| !roles.contains(&column.as_str()))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            (inferred, Vec::new())
        }
        Some(Value::Object(split)) if forecast => {
            for key in split.keys() {
                if key != "past" && key != "future" {
                    return Err(ApiError::validation(format!(
                        "inputs.{key} is not a field; a forecast's inputs are past and future"
                    ))
                    .param("inputs"));
                }
            }
            (names(split.get("past"))?, names(split.get("future"))?)
        }
        Some(listed) => (names(Some(listed))?, Vec::new()),
    };
    if !forecast && past.is_empty() {
        return Err(ApiError::validation(
            "no input columns: the context rows need a column besides the target",
        )
        .param("inputs"));
    }
    for column in past.iter().chain(&future) {
        if !rows.iter().any(|row| row.get(column).is_some()) {
            return Err(
                ApiError::validation(format!("input `{column}` is not in the context"))
                    .param(format!("inputs.{column}")),
            );
        }
        if roles.contains(&column.as_str()) {
            return Err(ApiError::validation(format!(
                "{column} has a role, so it is not an input"
            ))
            .param("inputs"));
        }
    }
    Ok((past, future))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequencies_read_as_frequency_normalize_reads_them() {
        let read = |value: Value| frequency(&value).map_err(|error| error.message);
        assert_eq!(read(json!("1d")), Ok((86400.0, None, true)));
        assert_eq!(read(json!("15min")), Ok((900.0, None, true)));
        assert_eq!(read(json!("H")), Ok((3600.0, None, true)));
        assert_eq!(read(json!("MS")), Ok((1.0, Some("MS".into()), true)));
        assert_eq!(read(json!("2Q")), Ok((2.0, Some("2QE".into()), true)));
        assert_eq!(read(json!("w-mon")), Ok((1.0, Some("W-MON".into()), true)));
        assert_eq!(read(json!(7)), Ok((7.0, None, false)));
        for refused in [
            json!("1.5h"),
            json!("B"),
            json!(0),
            json!(-1),
            json!(true),
            json!("0d"),
        ] {
            assert!(read(refused.clone()).is_err(), "{refused}");
        }
    }
}
