// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Answering the Models API's actions: the models and predictors served, and
//! predictions from catalog models, locked predictors and hosted revisions.

use super::incoming::fields;
use super::*;

/// A class's key in a probabilities mapping: `true`/`false`, or the label as text.
fn class_key(label: &Label) -> String {
    match label {
        Label::Bool(value) => value.to_string(),
        Label::Int(value) => value.to_string(),
        Label::Text(value) => value.clone(),
    }
}

fn label_json(label: &Label) -> Value {
    match label {
        Label::Bool(value) => json!(value),
        Label::Int(value) => json!(value),
        Label::Text(value) => json!(value),
    }
}

/// A quantile level as Python prints a float, which is how outputs key it.
fn level_key(level: f64) -> String {
    let text = format!("{level}");
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

/// Each query's class probabilities, keyed by class.
fn probabilities(classes: &[Label], inferred: &Inferred) -> Vec<Map<String, Value>> {
    let predictions = &inferred.predictions;
    (0..predictions.rows)
        .map(|row| {
            classes
                .iter()
                .enumerate()
                .map(|(class, label)| {
                    let value = predictions.values[row * predictions.columns + class];
                    (class_key(label), json!(f64::from(value)))
                })
                .collect()
        })
        .collect()
}

/// Each query's values at `levels`, keyed by level, from the table's own levels.
pub(super) fn quantiles(
    inferred: &Inferred,
    levels: &[f64],
) -> Result<Vec<Map<String, Value>>, ApiError> {
    let known = inferred.levels.as_deref().ok_or_else(|| {
        ApiError::new(
            "ModelOutputError",
            "native model returned invalid outputs",
            503,
        )
    })?;
    let columns =
        quantile_values(&inferred.predictions.values, known, levels).ok_or_else(|| {
            ApiError::new(
                "ModelOutputError",
                "native model returned invalid outputs",
                503,
            )
        })?;
    Ok((0..inferred.predictions.rows)
        .map(|row| {
            levels
                .iter()
                .zip(&columns)
                .map(|(&level, column)| (level_key(level), json!(f64::from(column[row]))))
                .collect()
        })
        .collect())
}

pub(super) async fn dispatch(
    front: &Front,
    project: &str,
    action: Action,
    resource: &str,
    environment: Option<&str>,
    payload: Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    match action {
        Action::List => Ok(json!({
            "models": front
                .models
                .iter()
                .filter(|model| model.predictor.is_none())
                .map(|model| front.model_description(model))
                .collect::<Vec<_>>()
        })),
        Action::Get => Ok(front.model_description(front.served(resource)?)),
        Action::Predictors => Ok(json!({
            "predictors": front
                .predictors(project)
                .map(|(served, predictor)| front.predictor_description(served, predictor))
                .collect::<Vec<_>>()
        })),
        Action::Predictor => {
            let (served, predictor) = front.predictor(project, resource)?;
            Ok(front.predictor_description(served, predictor))
        }
        Action::PredictorPredict => match front.predictor(project, resource) {
            Ok(_) if environment.is_some() => Err(ApiError::validation(format!(
                "predictor {resource} is served from its lock alone, with no environments"
            ))
            .param(ENVIRONMENT_PARAMETER)),
            Ok((served, predictor)) => {
                predictor_predict(front, served, predictor, &payload, charge).await
            }
            Err(error) => match front.revisions.as_ref() {
                Some(revisions) => {
                    let environment = environment.unwrap_or(DEFAULT_ENVIRONMENT);
                    revision_predict(
                        front,
                        revisions,
                        project,
                        resource,
                        environment,
                        &payload,
                        charge,
                    )
                    .await
                }
                None => Err(error),
            },
        },
        Action::AdHoc => adhoc_predict(front, &payload, charge).await,
    }
}

fn forecasting_unserved() -> ApiError {
    ApiError::new(
        "ValidationError",
        "this server answers tabular models; forecasters are served by the Python engine",
        422,
    )
}

/// Refuses the fields of `payload` named in `fields`, each with `reason`
/// after its name, so a caller learns which field to drop.
fn refuse_fields(payload: &Value, fields: &[&str], reason: &str) -> Result<(), ApiError> {
    if let Some(object) = payload.as_object() {
        for field in fields {
            if object.contains_key(*field) {
                return Err(ApiError::validation(format!("{field} {reason}")).param(*field));
            }
        }
    }
    Ok(())
}

fn context_declared() -> ApiError {
    ApiError::validation("context is declared by the predictor")
        .param("context")
        .hint("drop context: this predictor's context is pinned in its lock")
}

/// What a predictor's lock declares, which its predict body may not give.
const DECLARED: [&str; 9] = [
    "target",
    "inputs",
    "model",
    "task",
    "features",
    "options",
    "quantiles",
    "outputs",
    "id",
];

/// The `served_predict` call a declared predictor's body makes: its
/// `queries`, with `context` only when the lock says each request brings
/// it, and `snapshot`'s rows when the lock pins them; what the lock
/// declares is refused by name.
fn declared_call(
    predictor: &ServedPredictor,
    payload: &Value,
    snapshot: impl FnOnce() -> Result<Value, ApiError>,
) -> Result<Value, ApiError> {
    refuse_fields(payload, &DECLARED, "is declared by the predictor")?;
    let body = fields(payload, &["queries"], &["context", "as_of"])?;
    let context = match predictor.context {
        ContextMode::Request => body.get("context").cloned().ok_or_else(|| {
            ApiError::validation(
                "context is required: this predictor takes its context with each request",
            )
            .param("context")
        })?,
        ContextMode::Snapshot => {
            if body.contains_key("context") {
                return Err(context_declared());
            }
            snapshot()?
        }
    };
    let inputs: Vec<&str> = match &predictor.forecast {
        // A lock checked without rows holds no schema to read the columns
        // from; the request's own context shows them.
        Some(forecast) if predictor.schema_free() => forecast.query_columns(),
        Some(forecast) => {
            let queried = forecast.query_columns();
            predictor
                .schema
                .columns
                .iter()
                .map(|column| column.name.as_str())
                .filter(|name| queried.contains(name))
                .collect()
        }
        None => predictor.inputs.iter().map(String::as_str).collect(),
    };
    Ok(json!({
        "target": predictor.target,
        "inputs": inputs,
        "context": context,
        "queries": body["queries"],
    }))
}

/// Refuses `as_of` for a context that is not live: only a live source's
/// versions stand at different times.
fn not_live(predictor: &str, context: &str) -> ApiError {
    ApiError::validation(format!(
        "as_of selects a live context's version; predictor {predictor}'s context is a {context} one"
    ))
    .param(as_of::FIELD)
}

/// A predictor served here from its lock: a snapshot predictor's rows are
/// bound at startup, so its body brings none. Its context is never live, so
/// it takes no `as_of`.
async fn predictor_predict(
    front: &Front,
    served: &ServedModel,
    predictor: &ServedPredictor,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    if AsOf::parse(payload.get(as_of::FIELD))?.is_some() {
        return Err(not_live(&served.name, predictor.context.kind()));
    }
    let call = declared_call(predictor, payload, || Ok(json!([])))?;
    served_predict(front, served, predictor, &call, charge).await
}

/// The ad hoc route's body: a manifest fragment (`model`, `target`, and
/// optionally `inputs`, `id`, `quantiles`, `outputs`; `selection` is refused) with the
/// `context` and `queries` rows, for a catalog model served here.
async fn adhoc_predict(front: &Front, payload: &Value, charge: Charge) -> Result<Value, ApiError> {
    adhoc_predict_with(front, front.engine()?, payload, charge).await
}

pub(super) async fn adhoc_predict_with(
    front: &Front,
    engine: &Engine,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    refuse_fields(
        payload,
        &["task"],
        "is not a field: the model determines the task",
    )?;
    refuse_fields(
        payload,
        &["options"],
        "is not a field: id, quantiles and outputs are top-level fields",
    )?;
    refuse_fields(payload, &["features"], "is spelled inputs")?;
    let body = fields(
        payload,
        &["model", "target", "context", "queries"],
        &[
            "inputs",
            "id",
            "time",
            "series",
            "frequency",
            "horizon",
            "quantiles",
            "outputs",
            "selection",
        ],
    )?;
    let model = catalog::canonical(body["model"].as_str(), front.surface)?;
    if model.entry.task == catalog::Task::Forecast {
        return Err(forecasting_unserved());
    }
    let served = front.serving(&model)?;
    let task = match model.entry.task {
        catalog::Task::Classification => "classification",
        _ => "regression",
    };
    for field in ["time", "series", "frequency", "horizon"] {
        if body.contains_key(field) {
            return Err(ApiError::validation(format!(
                "{field} is a forecast field; {} is a {task} model",
                model.canonical_id()
            ))
            .param(field));
        }
    }
    if body.contains_key("selection") {
        // A selection resolves against a model's row cap as `gf check`
        // resolves it; this route binds every row, so it refuses one rather
        // than drop it. A predictor's lock carries its resolved selection.
        return Err(ApiError::validation(
            "selection is resolved into a predictor's gf.lock; serve a predictor to select rows",
        )
        .param("selection"));
    }
    let target = &body["target"];
    let target_name = target
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
    let id = match body.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) => Some(id.as_str()),
        Some(_) => return Err(ApiError::validation("id must name a column").param("id")),
    };
    let inputs: Vec<Value> = match body.get("inputs") {
        None | Some(Value::Null) => rows
            .first()
            .and_then(Value::as_object)
            .map(|row| {
                row.keys()
                    .filter(|column| column.as_str() != target_name && Some(column.as_str()) != id)
                    .map(|column| json!(column))
                    .collect()
            })
            .unwrap_or_default(),
        Some(Value::Array(inputs)) if inputs.iter().all(Value::is_string) => inputs.clone(),
        Some(_) => {
            return Err(
                ApiError::validation("inputs must be a list of column names").param("inputs"),
            );
        }
    };
    if inputs.is_empty() {
        return Err(ApiError::validation(
            "no input columns: the context rows need a column besides the target",
        )
        .param("inputs"));
    }
    let features = &inputs;
    let outputs: Vec<String> = match body.get("outputs") {
        Some(Value::Array(outputs)) if !outputs.is_empty() => outputs
            .iter()
            .map(|output| output.as_str().map(str::to_string))
            .collect::<Option<_>>()
            .ok_or_else(|| ApiError::validation("model does not support requested outputs"))?,
        Some(Value::Array(_)) | None => {
            if task == "classification" {
                vec!["label".into(), "probabilities".into()]
            } else {
                vec!["value".into()]
            }
        }
        Some(_) => {
            return Err(ApiError::validation(
                "model does not support requested outputs",
            ));
        }
    };
    let unique: std::collections::HashSet<&String> = outputs.iter().collect();
    require(
        unique.len() == outputs.len()
            && outputs.iter().all(|output| served.outputs.contains(output)),
        "model does not support requested outputs",
    )?;
    let levels = body.get("quantiles");
    require(
        levels.is_none() || outputs.iter().any(|output| output == "quantiles"),
        "quantiles require the quantiles output",
    )?;
    let select = |row: &Value| -> Value {
        Value::Object(
            features
                .iter()
                .filter_map(Value::as_str)
                .map(|feature| {
                    (
                        feature.to_string(),
                        row.get(feature).cloned().unwrap_or(Value::Null),
                    )
                })
                .collect(),
        )
    };
    let context: Vec<Value> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.get(target_name).is_some_and(|label| !label.is_null()))
        .map(|(index, row)| json!({"key": index.to_string(), "features": select(row), "label": row[target_name]}))
        .collect();
    let keyed: Vec<Value> = queries
        .iter()
        .enumerate()
        .map(|(index, row)| json!({"key": index.to_string(), "features": select(row)}))
        .collect();
    let mut request_options = Map::new();
    if task == "regression" {
        request_options.insert(
            "quantiles".into(),
            levels.cloned().unwrap_or_else(|| json!(DEFAULT_QUANTILES)),
        );
    }
    let request = json!({
        "model": body["model"],
        "task": task,
        "target": target,
        "features": features,
        "context": context,
        "queries": keyed,
        "options": request_options,
    });
    let bytes = serde_json::to_vec(&request)
        .map_err(|_| ApiError::new("InternalError", "request failed safely", 500))?;
    let Inference::Tabular(parsed) = request::parse(&bytes, &Options::request(front.surface))?
    else {
        return Err(forecasting_unserved());
    };
    let inferred = engine.tabular(&served.worker, &parsed, charge).await?;
    let predictions = prediction_rows(
        &parsed.classes,
        &parsed.levels,
        &inferred,
        &outputs,
        queries,
        id,
    )?;
    Ok(prediction_result(
        &served.resolved,
        task,
        &Used::request(parsed.usable_rows()),
        predictions,
    ))
}

/// The `/predictions` rows: the id column, if every query has it, then
/// each output asked for, from the classes or quantile levels predicted.
fn prediction_rows(
    classes: &[Label],
    levels: &[f64],
    inferred: &Inferred,
    outputs: &[String],
    queries: &[Value],
    id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let mut values: HashMap<&str, Vec<Value>> = HashMap::new();
    if classes.is_empty() {
        let median = quantiles(inferred, &[0.5])?;
        values.insert(
            "value",
            median.into_iter().map(|row| row["0.5"].clone()).collect(),
        );
        values.insert(
            "quantiles",
            quantiles(inferred, levels)?
                .into_iter()
                .map(Value::Object)
                .collect(),
        );
    } else {
        let scored = probabilities(classes, inferred);
        let labels = scored
            .iter()
            .map(|probabilities| {
                let mut best = 0;
                for (class, label) in classes.iter().enumerate() {
                    let at = |label: &Label| {
                        probabilities[&class_key(label)]
                            .as_f64()
                            .unwrap_or(f64::NAN)
                    };
                    if at(label) > at(&classes[best]) {
                        best = class;
                    }
                }
                label_json(&classes[best])
            })
            .collect();
        values.insert("label", labels);
        values.insert(
            "probabilities",
            scored.into_iter().map(Value::Object).collect(),
        );
    }
    Ok(projected(
        values,
        outputs,
        queries,
        id,
        inferred.predictions.rows,
    ))
}

/// Prediction rows: each query's declared id, when every query has one,
/// then the lock's outputs, in their order.
fn projected(
    mut values: HashMap<&str, Vec<Value>>,
    outputs: &[String],
    queries: &[Value],
    id: Option<&str>,
    rows: usize,
) -> Vec<Value> {
    let mut columns: Vec<(String, Vec<Value>)> = Vec::new();
    if let Some(id) = id
        && queries.iter().all(|row| row.get(id).is_some())
    {
        columns.push((
            id.to_string(),
            queries.iter().map(|row| row[id].clone()).collect(),
        ));
    }
    for output in outputs {
        columns.push((
            output.clone(),
            values.remove(output.as_str()).unwrap_or_default(),
        ));
    }
    (0..rows)
        .map(|row| {
            Value::Object(
                columns
                    .iter()
                    .map(|(name, column)| (name.clone(), column[row].clone()))
                    .collect(),
            )
        })
        .collect()
}

/// The context a prediction used, as every reply reports it: where it came
/// from (`snapshot`, `live` or `request`), the live source's name, the
/// context's version and digest where it has them, the rows it held, and
/// the time it stands as of. gf serve and the hosted runtime report it
/// alike; a field a context has no value for is null.
pub(super) struct Used {
    source: &'static str,
    name: Option<String>,
    version: Option<i64>,
    digest: Option<String>,
    rows: usize,
    as_of: Option<AsOf>,
}

impl Used {
    /// The context a request brought.
    fn request(rows: usize) -> Self {
        Self {
            source: "request",
            name: None,
            version: None,
            digest: None,
            rows,
            as_of: None,
        }
    }

    /// The rows a lock pins, by their digest.
    fn snapshot(rows: usize, digest: Option<&str>) -> Self {
        Self {
            source: "snapshot",
            digest: digest.map(str::to_string),
            ..Self::request(rows)
        }
    }

    /// The context a hosted revision was published with, as the control
    /// plane describes it: its source as `ContextSource` reads it, then its
    /// version, digest and as-of time. A legacy description gives its digest
    /// as a textual `version`.
    fn published(context: &Value, rows: usize) -> Self {
        let source = ContextSource::of(context);
        let kind = match source.as_ref().map(|source| source.kind) {
            Some(SourceKind::Live) => "live",
            _ => "snapshot",
        };
        let name = source
            .and_then(|source| source.name)
            .or_else(|| context["name"].as_str().map(str::to_string));
        let digest = context["digest"]
            .as_str()
            .or_else(|| context["version"].as_str())
            .map(str::to_string);
        Self {
            source: kind,
            name,
            version: context["version"].as_i64(),
            digest,
            rows,
            as_of: AsOf::read(context.get(as_of::FIELD)),
        }
    }

    fn json(&self) -> Value {
        json!({
            "source": self.source,
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "rows": self.rows,
            "as_of": AsOf::json(self.as_of),
        })
    }
}

/// The shared `PredictionResult` wire value, with the context it used.
fn prediction_result(model: &str, task: &str, used: &Used, predictions: Vec<Value>) -> Value {
    json!({
        "model": model,
        "task": task,
        "context": used.json(),
        "predictions": predictions,
    })
}

/// A locked predictor's `/predictions`: the body checked against the lock,
/// then the queries encoded by the binder and predicted from the
/// snapshot's durable table, or from the ephemeral table of the request's
/// own bound context.
async fn served_predict(
    front: &Front,
    served: &ServedModel,
    predictor: &ServedPredictor,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let body = fields(payload, &["target", "inputs", "context", "queries"], &[])?;
    let features: Vec<String> = body["inputs"]
        .as_array()
        .ok_or_else(|| ApiError::validation("inputs must be a list"))?
        .iter()
        .map(|input| input.as_str().map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| ApiError::validation("inputs differ from lock"))?;
    let queries = body["queries"]
        .as_array()
        .filter(|queries| (1..=MAX_QUERY_ROWS).contains(&queries.len()))
        .ok_or_else(|| ApiError::validation("queries must contain 1..10000 rows"))?;
    let context = body["context"]
        .as_array()
        .ok_or_else(|| ApiError::validation("rows must be lists"))?;
    require(
        queries.iter().chain(context).all(Value::is_object),
        "rows must be objects",
    )?;
    if let Some(forecast) = &predictor.forecast {
        return series_predict(
            front, served, predictor, forecast, body, &features, queries, context, charge,
        )
        .await;
    }
    let schema_free = predictor.schema_free();
    let target = if schema_free {
        let target = body["target"]
            .as_str()
            .ok_or_else(|| ApiError::validation("request target is required"))?;
        require(
            predictor
                .target
                .as_deref()
                .is_none_or(|locked| locked == target),
            "target differs from lock",
        )?;
        target.to_string()
    } else {
        require(
            body["target"].as_str().is_some()
                && body["target"].as_str() == predictor.target.as_deref(),
            "target differs from lock",
        )?;
        require(features == predictor.inputs, "inputs differ from lock")?;
        predictor.target.clone().unwrap_or_default()
    };
    let (inferred, classes, used) = match predictor.context {
        ContextMode::Snapshot => {
            if !context.is_empty() {
                return Err(context_declared());
            }
            let bound = front
                .bound
                .get()
                .and_then(|bound| bound.get(&served.name))
                .ok_or_else(not_ready)?;
            let encoded = bound
                .encoder
                .encode(&predictors::records(&predictor.schema.columns, queries)?)?;
            let inferred = front.engine()?.predicted(bound, encoded).await?;
            (
                inferred,
                bound.classes.clone(),
                Used::snapshot(bound.rows, predictor.context_digest.as_deref()),
            )
        }
        ContextMode::Request => {
            require(
                !context.is_empty() && context.len() <= MAX_CONTEXT_ROWS,
                "context must contain 1..100000 rows",
            )?;
            require(
                predictor.max_rows.is_none_or(|most| context.len() <= most),
                "context exceeds the predictor's max_rows",
            )?;
            let schema = if schema_free {
                predictors::inferred(&features, &target, context, &predictor.types)?
            } else {
                predictor.schema.clone()
            };
            let bound = predictor::bind(
                &schema,
                predictor.task()?,
                &target,
                &predictors::records(&schema.columns, context)?,
            )?;
            let encoded = bound
                .encoder
                .encode(&predictors::records(&schema.columns, queries)?)?;
            let fingerprint = bound_fingerprint(served, &bound);
            let context = engine::rows_context(
                &served.worker,
                &fingerprint,
                bound.encoder.width(),
                bound.classes.len(),
                bound.features,
                bound.targets,
                bound.rows,
                predictor.strategy.as_ref(),
            )?;
            let inferred = front
                .engine()?
                .ask(context, Asked::Rows(encoded), &fingerprint)
                .await?;
            (inferred, bound.classes, Used::request(bound.rows))
        }
    };
    drop(charge);
    let predictions = prediction_rows(
        &classes,
        &predictor.quantiles,
        &inferred,
        &served.outputs,
        queries,
        predictor.id.as_deref(),
    )?;
    Ok(prediction_result(
        &served.resolved,
        &predictor.task,
        &used,
        predictions,
    ))
}

/// A snapshot predictor's context rows, from the file `gf serve` wrote.
pub(crate) fn snapshot_rows(predictor: &ServedPredictor) -> Result<RecordBatch, ApiError> {
    let path = predictor.snapshot.as_deref().ok_or_else(|| {
        ApiError::new(
            "ContextRequired",
            "snapshot predictors are served from their predictor directory",
            422,
        )
    })?;
    predictors::read_snapshot(path).map_err(|error| ApiError::new("MissingFile", error, 503))
}

/// A forecast predictor's `/predictions`: the queries placed on the bound
/// group's series and horizon, the known inputs' future filled in, and the
/// group forecast on the shard path catalog forecasts take, from the table
/// its context and future values name.
#[allow(clippy::too_many_arguments)]
async fn series_predict(
    front: &Front,
    served: &ServedModel,
    predictor: &ServedPredictor,
    forecast: &ForecastPredictor,
    body: &Map<String, Value>,
    features: &[String],
    queries: &[Value],
    context: &[Value],
    charge: Charge,
) -> Result<Value, ApiError> {
    let failed = || ApiError::new("InternalError", "request failed safely", 500);
    let target = body["target"]
        .as_str()
        .ok_or_else(|| ApiError::validation("target differs from lock"))?;
    require(
        predictor
            .target
            .as_deref()
            .is_none_or(|locked| locked == target),
        "target differs from lock",
    )?;
    let aligned;
    let (bound, schema, snapshot) = match predictor.context {
        ContextMode::Snapshot => {
            if !context.is_empty() {
                return Err(context_declared());
            }
            let bound = front
                .series
                .get()
                .and_then(|series| series.get(&served.name))
                .ok_or_else(not_ready)?;
            (bound, predictor.schema.clone(), true)
        }
        ContextMode::Request => {
            require(
                !context.is_empty() && context.len() <= MAX_CONTEXT_ROWS,
                "context must contain 1..100000 rows",
            )?;
            require(
                predictor.max_rows.is_none_or(|most| context.len() <= most),
                "context exceeds the predictor's max_rows",
            )?;
            let schema = if predictor.schema_free() {
                // A forecaster's times arrive as ISO text, which a lock that
                // declares no kind for them still reads as times.
                let mut types = predictor.types.clone();
                let textual = context
                    .iter()
                    .filter_map(|row| row.get(&forecast.time))
                    .all(Value::is_string);
                if textual {
                    types
                        .entry(forecast.time.clone())
                        .or_insert_with(|| "timestamp".to_string());
                }
                predictors::inferred(&forecast.context_columns(), target, context, &types)?
            } else {
                predictor.schema.clone()
            };
            let rows = predictors::records(&schema.columns, context)?;
            aligned = front.bind_series(served, predictor, forecast, &rows, None)?;
            (&aligned, schema, false)
        }
    };
    let expected: Vec<&str> = schema
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .filter(|name| forecast.query_columns().contains(name))
        .collect();
    require(features == expected.as_slice(), "features differ from lock")?;
    let rows = forecast.query_rows(&schema, predictor.id.as_deref(), queries)?;
    let queried = forecast.queries(&bound.context, &rows)?;
    let horizon = forecast.horizon;
    // One forecast request of every queried series, as
    // `ForecastContext.predict` sends it; the engine forecasts its groups at
    // once, each from the table its fingerprint names.
    let bytes =
        serde_json::to_vec(&bound.context.request(&queried.futures)).map_err(|_| failed())?;
    let Inference::Forecast(parsed) = request::parse(&bytes, &front.predictor_options())? else {
        return Err(failed());
    };
    let answers = front
        .engine()?
        .forecast_request(&served.worker, &parsed)
        .await?;
    drop(charge);
    let invalid = || {
        ApiError::new(
            "ModelOutputError",
            "native forecaster returned invalid forecasts",
            503,
        )
    };
    // Each queried series' query, in ascending order, and where its answer is.
    let order: HashMap<usize, usize> = queried
        .futures
        .keys()
        .enumerate()
        .map(|(query, &series)| (series, query))
        .collect();
    let mut placed: HashMap<usize, (usize, usize)> = HashMap::new();
    for (answer, (indices, _)) in answers.iter().enumerate() {
        for (row, &query) in indices.iter().enumerate() {
            placed.insert(query, (answer, row));
        }
    }
    let levels = answers
        .first()
        .and_then(|(_, inferred)| inferred.levels.as_deref())
        .ok_or_else(invalid)?;
    let column = |level: f64| levels.iter().position(|known| *known == level);
    let median = column(0.5).ok_or_else(invalid)?;
    let columns: Vec<usize> = predictor
        .quantiles
        .iter()
        .map(|&level| column(level))
        .collect::<Option<_>>()
        .ok_or_else(|| {
            ApiError::new(
                "UnsupportedQuantiles",
                "the model only returns its own published quantile levels",
                422,
            )
        })?;
    for (indices, inferred) in &answers {
        let predicted = &inferred.predictions;
        if inferred.levels.as_deref() != Some(levels)
            || (predicted.rows, predicted.horizon, predicted.columns)
                != (indices.len(), Some(horizon), levels.len())
            || !predicted.values.iter().all(|value| value.is_finite())
        {
            return Err(invalid());
        }
    }
    let at = |member: usize, position: usize, column: usize| {
        let (answer, row) = placed[&order[&member]];
        let predicted = &answers[answer].1.predictions;
        f64::from(predicted.values[(row * horizon + position) * predicted.columns + column])
    };
    let mut values: HashMap<&str, Vec<Value>> = HashMap::new();
    values.insert(
        "value",
        queried
            .positions
            .iter()
            .map(|&(member, position)| json!(at(member, position, median)))
            .collect(),
    );
    values.insert(
        "quantiles",
        queried
            .positions
            .iter()
            .map(|&(member, position)| {
                Value::Object(
                    predictor
                        .quantiles
                        .iter()
                        .zip(&columns)
                        .map(|(&level, &column)| {
                            (level_key(level), json!(at(member, position, column)))
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    let predictions = projected(
        values,
        &served.outputs,
        queries,
        predictor.id.as_deref(),
        queried.positions.len(),
    );
    let used = if snapshot {
        Used::snapshot(bound.rows, predictor.context_digest.as_deref())
    } else {
        Used::request(bound.rows)
    };
    Ok(prediction_result(
        &served.resolved,
        &predictor.task,
        &used,
        predictions,
    ))
}

/// A bound request context's fingerprint: its model and its rows as the
/// binder encoded them, which is all its table holds, so contexts that
/// encode alike share one table and one fit.
fn bound_fingerprint(served: &ServedModel, bound: &predictor::Context) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"predictor\0");
    hasher.update(served.worker.as_bytes());
    hasher.update([0]);
    hasher.update((bound.encoder.width() as u64).to_le_bytes());
    hasher.update((bound.classes.len() as u64).to_le_bytes());
    let features: Vec<u8> = bound
        .features
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    hasher.update(&features);
    let targets: Vec<u8> = match &bound.targets {
        Targets::Classes(codes) => codes.iter().flat_map(|code| code.to_le_bytes()).collect(),
        Targets::Values(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    };
    hasher.update(&targets);
    hasher.finalize().into()
}

/// A hosted predictor: the revision `environment` serves, loaded from the
/// control plane on demand and predicted as a native predictor. With
/// `as_of`, a live context is the version that stood then, as the control
/// plane keeps its history; any other context refuses it.
async fn revision_predict(
    front: &Front,
    revisions: &Revisions,
    project: &str,
    predictor: &str,
    environment: &str,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let as_of = AsOf::parse(payload.get(as_of::FIELD))?;
    let resolved = revisions.resolve(project, predictor, environment).await?;
    let revision = resolved.revision;
    let version = resolved.context_version.unwrap_or(Value::Null);
    let started = std::time::Instant::now();
    let scored = async {
        let mut published = revisions
            .load(project, predictor, revision, Some(&version))
            .await?;
        if let Some(as_of) = as_of {
            if !published.live() {
                let Published::Native { served, .. } = &*published;
                let kind = served
                    .predictor
                    .as_ref()
                    .map_or("request", |predictor| predictor.context.kind());
                return Err(not_live(predictor, kind));
            }
            published = revisions
                .load_as_of(project, predictor, revision, as_of)
                .await?;
        }
        let Published::Native {
            served,
            context,
            rows,
        } = &*published;
        native_predict(
            front,
            served,
            context.as_ref(),
            rows.as_deref(),
            payload,
            charge,
        )
        .await
    }
    .await;
    revisions.record(Record {
        project: project.to_string(),
        predictor: predictor.to_string(),
        environment: Some(environment.to_string()),
        revision,
        ok: scored.is_ok(),
        latency_seconds: started.elapsed().as_secs_f64(),
    });
    let mut result = scored?;
    result["predictor"] = json!(predictor);
    result["revision"] = json!(revision);
    result["environment"] = json!(environment);
    Ok(result)
}

async fn native_predict(
    front: &Front,
    served: &ServedModel,
    captured: Option<&Value>,
    rows: Option<&[Value]>,
    payload: &Value,
    charge: Charge,
) -> Result<Value, ApiError> {
    let failed = || ApiError::new("InternalError", "request failed safely", 500);
    let predictor = served.predictor.as_ref().ok_or_else(failed)?;
    let base = front
        .models
        .iter()
        .find(|model| model.predictor.is_none() && model.resolved == served.resolved)
        .ok_or_else(|| {
            ApiError::new(
                "ModelUnavailable",
                "the revision's model is not served here",
                503,
            )
        })?;
    // A revision's snapshot rows come with it, so they bind with each request.
    let call = declared_call(predictor, payload, || {
        rows.map(|rows| Value::Array(rows.to_vec()))
            .ok_or_else(failed)
    })?;
    let mut ephemeral = served.clone();
    ephemeral.worker.clone_from(&base.worker);
    let mut bound = predictor.clone();
    bound.context = ContextMode::Request;
    bound.snapshot = None;
    ephemeral.predictor = Some(bound.clone());
    let mut result = served_predict(front, &ephemeral, &bound, &call, charge).await?;
    if predictor.context == ContextMode::Snapshot
        && let Some(captured) = captured
    {
        let rows = result["context"]["rows"].as_u64().unwrap_or_default();
        let rows = usize::try_from(rows).unwrap_or(usize::MAX);
        result["context"] = Used::published(captured, rows).json();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request-context forecaster whose lock was checked without rows,
    /// so it holds no column kinds.
    fn schema_free_forecaster() -> ServedPredictor {
        serde_json::from_value(json!({
            "schema": {"columns": [], "rows": null, "classes": null, "frequency": null},
            "task": "forecast",
            "target": "sales",
            "inputs": ["promo"],
            "id": null,
            "context": "request",
            "quantiles": [0.5],
            "context_digest": null,
            "lock_digest": "sha256:forecaster",
            "snapshot": null,
            "forecast": {
                "series": ["store"],
                "time": "day",
                "horizon": 2,
                "past_inputs": ["visits"],
                "future_inputs": ["promo"],
                "frequency": null,
                "declared": null,
            },
        }))
        .unwrap()
    }

    #[test]
    fn a_schema_free_forecasters_call_names_the_columns_its_context_gives() {
        let predictor = schema_free_forecaster();
        let forecast = predictor.forecast.as_ref().unwrap();
        let context = json!([
            {"store": "a", "day": "2026-10-01T00:00:00Z", "visits": 3, "promo": 0, "sales": 1.5},
            {"store": "a", "day": "2026-10-02T00:00:00Z", "visits": 4, "promo": 1, "sales": 2.5},
        ]);
        let payload = json!({"queries": [{"store": "a", "day": "2026-10-03T00:00:00Z", "promo": 1}], "context": context});
        let call = declared_call(&predictor, &payload, || Ok(json!([]))).unwrap();
        let inputs: Vec<&str> = call["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        // What `series_predict` holds the call's inputs to, from the context.
        let schema = predictors::inferred(
            &forecast.context_columns(),
            "sales",
            context.as_array().unwrap(),
            &predictor.types,
        )
        .unwrap();
        let expected: Vec<&str> = schema
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .filter(|name| forecast.query_columns().contains(name))
            .collect();
        assert_eq!(inputs, expected);
        assert_eq!(inputs, ["store", "day", "promo"]);
    }
}
