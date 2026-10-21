// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The single validator of inference requests, before any weights load.
//!
//! Every surface that takes an `InferenceRequest` (`model, task, target,
//! features, context, queries, options`) parses it here into a typed
//! [`Tabular`] or [`Forecast`] request: the hosted Models API, `gf serve`,
//! the in-process catalog service and the package runtime's forecast
//! contexts. The checks, their order, error codes, messages and HTTP
//! statuses are those the Python catalog service answered, so a client sees
//! the same refusal whichever surface it calls. What only loaded weights
//! know, a forecaster's patch, row limit and undeclared quantile levels, is
//! checked by execution after the load.
//!
//! Payloads are bytes, not parsed values: JSON keeps integers apart from
//! floats, as integer periods and integer steps need, and an object naming a
//! field twice is refused rather than silently taking the last.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use sha2::{Digest, Sha256};

use crate::catalog::{self, Admission, Model, Source, Surface, Task};
use crate::forecasting::{KNOWN, PAST, TARGET};
use crate::node::Targets;

/// Most bytes in one request payload.
pub const MAX_BYTES: usize = 1_048_576;
/// Most values in one request payload, counting every object and list.
pub const MAX_NODES: usize = 65_536;
/// Deepest nesting of one request payload.
pub const MAX_DEPTH: usize = 16;
/// The largest integer every JSON implementation reads exactly.
pub const MAX_INTEGER: i64 = (1 << 53) - 1;
/// Quantiles a regression returns when a caller asks for none.
pub const DEFAULT_QUANTILES: [f64; 3] = [0.1, 0.5, 0.9];
/// Most quantile levels one request selects.
pub const MAX_LEVELS: usize = 99;

/// A refusal: a stable code, a safe message and the HTTP status it answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestError {
    pub code: &'static str,
    pub message: String,
    pub status: u16,
}

impl RequestError {
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>, status: u16) -> Self {
        Self {
            code,
            message: message.into(),
            status,
        }
    }

    /// A malformed request: `ValidationError`, 422.
    #[must_use]
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new("ValidationError", message, 422)
    }

    fn empty(message: &str) -> Self {
        Self::new("EmptyContext", message, 503)
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RequestError {}

type Checked<T> = Result<T, RequestError>;

fn require(condition: bool, message: &str) -> Checked<()> {
    if condition {
        Ok(())
    } else {
        Err(RequestError::invalid(message))
    }
}

/// A string as Python's `repr` spells it, as messages that quote input do.
#[must_use]
pub fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut spelled = String::with_capacity(text.len() + 2);
    spelled.push(quote);
    for character in text.chars() {
        match character {
            '\\' => spelled.push_str("\\\\"),
            '\t' => spelled.push_str("\\t"),
            '\n' => spelled.push_str("\\n"),
            '\r' => spelled.push_str("\\r"),
            _ if character == quote => {
                spelled.push('\\');
                spelled.push(quote);
            }
            _ if printable(character) => spelled.push(character),
            _ => {
                let code = u32::from(character);
                if code <= 0xff {
                    spelled.push_str(&format!("\\x{code:02x}"));
                } else if code <= 0xffff {
                    spelled.push_str(&format!("\\u{code:04x}"));
                } else {
                    spelled.push_str(&format!("\\U{code:08x}"));
                }
            }
        }
    }
    spelled.push(quote);
    spelled
}

/// Python's `str.isprintable` for the characters requests carry: not a
/// control, separator other than the space, format or private-use character.
fn printable(character: char) -> bool {
    if character == ' ' {
        return true;
    }
    !(character.is_control()
        || character.is_whitespace()
        || matches!(
            character,
            '\u{ad}'
                | '\u{600}'..='\u{605}'
                | '\u{61c}'
                | '\u{6dd}'
                | '\u{70f}'
                | '\u{180e}'
                | '\u{200b}'..='\u{200f}'
                | '\u{202a}'..='\u{202e}'
                | '\u{2060}'..='\u{2064}'
                | '\u{2066}'..='\u{206f}'
                | '\u{e000}'..='\u{f8ff}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
                | '\u{f0000}'..='\u{10ffff}'
        ))
}

/// A JSON number, an integer or a float as the payload spells it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    fn is_int(self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// `self + step * count`, with Python's arithmetic: integers stay exact.
    #[allow(clippy::cast_precision_loss)]
    fn advanced(self, step: Self, count: i64) -> Self {
        let offset = match step {
            Self::Int(step) => Self::Int(step.saturating_mul(count)),
            Self::Float(step) => Self::Float(step * count as f64),
        };
        match (self, offset) {
            (Self::Int(origin), Self::Int(offset)) => Self::Int(origin.saturating_add(offset)),
            (origin, offset) => Self::Float(origin.as_f64() + offset.as_f64()),
        }
    }

    /// `(self - earlier) / step`, as Python's true division.
    #[allow(clippy::cast_precision_loss)]
    fn steps_after(self, earlier: Self, step: Self) -> f64 {
        let span = match (self, earlier) {
            (Self::Int(later), Self::Int(earlier)) => (later - earlier) as f64,
            (later, earlier) => later.as_f64() - earlier.as_f64(),
        };
        span / step.as_f64()
    }

    /// The same number as a set or map key; `-0.0` is `0.0`, as in Python.
    fn key(self) -> u64 {
        (self.as_f64() + 0.0).to_bits()
    }
}

/// The time axis of a forecast: integer periods or floating UTC seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clock {
    Periods,
    Seconds,
}

impl Clock {
    fn of(time: Num) -> Self {
        if time.is_int() {
            Self::Periods
        } else {
            Self::Seconds
        }
    }

    /// The axis as results name it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Periods => "integer-periods",
            Self::Seconds => "utc-seconds",
        }
    }
}

/// A parsed payload value. Objects are sorted by key; a key named twice is refused.
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(Num),
    Text(String),
    List(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

const DUPLICATE: &str = "duplicate JSON field";

impl<'de> de::Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Json, E> {
        Ok(Json::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Json, E> {
        Ok(Json::Num(Num::Int(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Json, E> {
        // Past `i64`, an integer is past the portable range too.
        Ok(Json::Num(Num::Int(
            i64::try_from(value).unwrap_or(i64::MAX),
        )))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Json, E> {
        Ok(Json::Num(Num::Float(value)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Json, E> {
        Ok(Json::Text(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Json, E> {
        Ok(Json::Text(value))
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut items: A) -> Result<Json, A::Error> {
        let mut list = Vec::new();
        while let Some(item) = items.next_element()? {
            list.push(item);
        }
        Ok(Json::List(list))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut entries: A) -> Result<Json, A::Error> {
        let mut object = BTreeMap::new();
        while let Some(key) = entries.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(de::Error::custom(DUPLICATE));
            }
            let value = entries.next_value()?;
            object.insert(key, value);
        }
        Ok(Json::Object(object))
    }
}

type Object = BTreeMap<String, Json>;

impl Json {
    fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    fn list(&self) -> Option<&[Self]> {
        match self {
            Self::List(list) => Some(list),
            _ => None,
        }
    }
}

/// Which bounds a payload is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limits {
    /// One Models API request's: bytes, values and nesting, and the serving
    /// policy's context and query rows.
    Request,
    /// A package's bound context, which its lock bounds instead.
    Package,
}

/// Parses a payload into a JSON object, held to the request bounds if asked.
fn document(payload: &[u8], limits: Limits) -> Checked<Object> {
    let value: Json = serde_json::from_slice(payload).map_err(|error| {
        RequestError::invalid(if error.classify() == serde_json::error::Category::Data {
            DUPLICATE
        } else {
            "invalid JSON body"
        })
    })?;
    let Json::Object(object) = value else {
        return Err(RequestError::invalid("expected JSON object"));
    };
    if limits == Limits::Request {
        let mut count = 0;
        for child in object.values() {
            portable(child, 1, &mut count)?;
        }
        require(payload.len() <= MAX_BYTES, "JSON body too large")?;
    }
    Ok(object)
}

/// The bounds `generalforecast.deployments.boundary.json_object` holds a value to.
fn portable(value: &Json, depth: usize, count: &mut usize) -> Checked<()> {
    *count += 1;
    // The root object is the first value.
    require(
        depth <= MAX_DEPTH && *count < MAX_NODES,
        "JSON structure too large",
    )?;
    match value {
        Json::Object(object) => {
            for child in object.values() {
                portable(child, depth + 1, count)?;
            }
        }
        Json::List(list) => {
            for child in list {
                portable(child, depth + 1, count)?;
            }
        }
        Json::Num(Num::Int(integer)) => require(
            integer.unsigned_abs() <= MAX_INTEGER.unsigned_abs(),
            "integer exceeds portable JSON range",
        )?,
        _ => {}
    }
    Ok(())
}

/// Checks an object's shape: every required field, and no unknown one.
fn fields<'a, S: AsRef<str>>(
    value: &'a Json,
    required: &[S],
    optional: &[&str],
) -> Checked<&'a Object> {
    let Json::Object(object) = value else {
        return Err(RequestError::invalid("expected an object"));
    };
    let required: BTreeSet<&str> = required.iter().map(AsRef::as_ref).collect();
    if !required.iter().all(|field| object.contains_key(*field)) {
        let names: Vec<String> = required.iter().map(|field| python_repr(field)).collect();
        return Err(RequestError::invalid(format!(
            "required fields: [{}]",
            names.join(", ")
        )));
    }
    require(
        object
            .keys()
            .all(|key| required.contains(key.as_str()) || optional.contains(&key.as_str())),
        "unknown fields",
    )?;
    Ok(object)
}

/// A bounded, nonempty identity string.
fn text<'a>(value: &'a Json, field: &str) -> Checked<&'a str> {
    value
        .text()
        .filter(|text| (1..=catalog::MAX_TEXT).contains(&text.chars().count()))
        .ok_or_else(|| RequestError::invalid(format!("invalid {field}")))
}

/// The closed range a number must fall in, and how Python prints its ends.
struct Range {
    low: f64,
    high: f64,
    spelled: &'static str,
}

const VALUES: Range = Range {
    low: -1e12,
    high: 1e12,
    spelled: "between -1000000000000.0 and 1000000000000.0",
};
const STEPS: Range = Range {
    low: 0.0,
    high: 1e12,
    spelled: "between 0 and 1000000000000.0",
};
const LEVELS: Range = Range {
    low: 0.0,
    high: 1.0,
    spelled: "between 0 and 1",
};

/// A finite number in range; booleans are not numbers.
fn number(value: &Json, range: &Range, field: &str) -> Checked<Num> {
    match value {
        Json::Num(number) if (range.low..=range.high).contains(&number.as_f64()) => Ok(*number),
        _ => Err(RequestError::invalid(format!(
            "{field} must be {}",
            range.spelled
        ))),
    }
}

/// Validates requested quantile levels: strictly increasing, inside (0, 1).
fn levels(value: &Json) -> Checked<Vec<f64>> {
    let list = value
        .list()
        .filter(|list| (1..=MAX_LEVELS).contains(&list.len()))
        .ok_or_else(|| RequestError::invalid("invalid quantile levels"))?;
    let levels = list
        .iter()
        .map(|level| number(level, &LEVELS, "quantile").map(Num::as_f64))
        .collect::<Checked<Vec<f64>>>()?;
    require(
        levels.iter().all(|level| 0.0 < *level && *level < 1.0),
        "quantiles must be within (0,1)",
    )?;
    require(
        levels.windows(2).all(|pair| pair[0] < pair[1]),
        "quantiles must strictly increase",
    )?;
    Ok(levels)
}

/// Validates a JSON list of quantile levels, as a request's options carry them.
///
/// # Errors
///
/// `ValidationError` for malformed JSON or levels outside (0, 1) that do not
/// strictly increase.
pub fn quantile_levels(payload: &[u8]) -> Result<Vec<f64>, RequestError> {
    let value: Json =
        serde_json::from_slice(payload).map_err(|_| RequestError::invalid("invalid JSON body"))?;
    levels(&value)
}

/// A class label: one type per request.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Label {
    Bool(bool),
    Int(i64),
    Text(String),
}

/// A classification or regression request: labeled context and queries as matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct Tabular {
    pub model: Model,
    pub task: Task,
    pub target: String,
    pub features: Vec<String>,
    /// The labeled context rows' features, row-major, `features.len()` wide.
    pub context: Vec<f32>,
    /// Class codes into `classes`, or regression values, per labeled row.
    pub targets: Targets,
    /// The classes, in output order; empty for a regression.
    pub classes: Vec<Label>,
    /// The queries' features, row-major.
    pub queries: Vec<f32>,
    pub query_keys: Vec<String>,
    /// The quantile levels a regression returns; empty for a classification.
    pub levels: Vec<f64>,
    /// Context rows supplied, labeled or not.
    pub supplied_rows: usize,
    pub fingerprint: [u8; 32],
}

impl Tabular {
    /// The labeled context rows the model reads.
    #[must_use]
    pub fn usable_rows(&self) -> usize {
        self.context.len() / self.features.len()
    }
}

/// Related series and their covariates on one time axis, as a forecaster reads them.
///
/// Rows are the group's targets in query order, then its covariates sorted
/// by name. Time is not padded: the last step is the origin and the first
/// the earliest observation, and execution left-pads to the model's patch.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignedGroup {
    pub rows: usize,
    pub steps: usize,
    pub horizon: usize,
    /// `(rows, steps)` row-major, zero where unobserved.
    pub values: Vec<f32>,
    pub observed: Vec<bool>,
    /// Per row, `TARGET`, `PAST` or `KNOWN`.
    pub roles: Vec<i32>,
    /// Known covariates' values over the horizon, `(rows, horizon)`.
    pub future: Vec<f32>,
    pub future_observed: Vec<bool>,
}

/// One forecast query: a series forecast with its group from an origin.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesQuery {
    pub key: String,
    pub series: String,
    pub origin: Num,
    /// The group's target series, in row order.
    pub members: Vec<String>,
    /// The row of `series` among `members`.
    pub member: usize,
    /// The forecast's times, one step after another past the origin.
    pub times: Vec<Num>,
    pub group: AlignedGroup,
    /// SHA-256 of the context fingerprint and this group: its members in row
    /// order, origin, horizon and known-future covariates. Queries with the
    /// same key forecast the same aligned group.
    pub fingerprint: [u8; 32],
}

/// A forecast request: aligned groups per query, and what results report.
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast {
    pub model: Source,
    pub target: String,
    pub step: Num,
    pub horizon: usize,
    pub levels: Vec<f64>,
    pub known: Vec<String>,
    pub queries: Vec<SeriesQuery>,
    pub clock: Clock,
    /// Context rows supplied, observed or not.
    pub supplied_rows: usize,
    /// Context rows with an observed value.
    pub usable_observations: usize,
    pub fingerprint: [u8; 32],
}

/// A parsed inference request.
#[derive(Clone, Debug, PartialEq)]
pub enum Inference {
    Tabular(Tabular),
    Forecast(Forecast),
}

impl Inference {
    /// The model the request resolved to.
    #[must_use]
    pub fn model(&self) -> Source {
        match self {
            Self::Tabular(request) => Source::Catalog(request.model.clone()),
            Self::Forecast(request) => request.model.clone(),
        }
    }

    /// SHA-256 of what the model's context is: same context, same table.
    #[must_use]
    pub fn fingerprint(&self) -> [u8; 32] {
        match self {
            Self::Tabular(request) => request.fingerprint,
            Self::Forecast(request) => request.fingerprint,
        }
    }
}

/// A fingerprint as lowercase hex.
#[must_use]
pub fn hex(fingerprint: &[u8; 32]) -> String {
    fingerprint
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// How a request is parsed: the surface it is served on, its bounds, and
/// a forecaster's levels when the caller has loaded its weights.
#[derive(Clone, Copy, Debug)]
pub struct Options<'a> {
    pub surface: Surface,
    pub limits: Limits,
    /// The levels the loaded model outputs, or None for the catalog's.
    pub levels: Option<&'a [f64]>,
    /// The models it may name.
    pub admission: Admission<'a>,
}

impl Options<'_> {
    /// A Models API request on a surface, before any weights load.
    #[must_use]
    pub fn request(surface: Surface) -> Self {
        Self {
            surface,
            limits: Limits::Request,
            levels: None,
            admission: Admission::Catalog,
        }
    }
}

/// Parses and validates one inference request.
///
/// # Errors
///
/// The refusal the Python catalog service answered for the same request,
/// such as `ValidationError` (422), `ModelNotFound` (404),
/// `UnsupportedTask`, `UnsupportedClassCount`, `UnsupportedQuantiles`,
/// `UnsupportedCovariates` (422) or `EmptyContext` (503).
pub fn parse(payload: &[u8], options: &Options<'_>) -> Result<Inference, RequestError> {
    let request = Json::Object(document(payload, options.limits)?);
    let request = fields(
        &request,
        &[
            "model", "task", "target", "features", "context", "queries", "options",
        ],
        &[],
    )?;
    let source = catalog::admit(request["model"].text(), options.surface, options.admission)?;
    let task = match request["task"].text() {
        Some("classification") => Task::Classification,
        Some("regression") => Task::Regression,
        Some("forecast") => Task::Forecast,
        _ => return Err(RequestError::invalid("invalid task")),
    };
    if task != source.task() {
        return Err(unsupported_task());
    }
    let policy = &catalog::catalog().policy;
    let bounded = options.limits == Limits::Request;
    let target = text(&request["target"], "target")?.to_owned();
    let features = request["features"]
        .list()
        .filter(|features| features.len() <= policy.features)
        .ok_or_else(|| RequestError::invalid("invalid feature list"))?
        .iter()
        .map(|feature| text(feature, "feature").map(str::to_owned))
        .collect::<Checked<Vec<String>>>()?;
    require(
        features.iter().collect::<HashSet<_>>().len() == features.len(),
        "features must be unique",
    )?;
    require(!features.contains(&target), "target cannot be a feature")?;
    let context = request["context"]
        .list()
        .ok_or_else(|| RequestError::invalid("context must be a list"))?;
    require(
        !bounded || context.len() <= policy.context_rows,
        "context capacity exceeded",
    )?;
    let queries = request["queries"]
        .list()
        .ok_or_else(|| RequestError::invalid("queries must be a list"))?;
    require(
        !queries.is_empty() && (!bounded || queries.len() <= policy.query_rows),
        "query capacity exceeded",
    )?;
    let settings = fields(
        &request["options"],
        &[] as &[&str],
        &[if task == Task::Forecast {
            "series"
        } else {
            "quantiles"
        }],
    )?;
    if task == Task::Forecast {
        let levels = match (&source, options.levels) {
            (_, Some(levels)) => levels,
            (Source::Catalog(model), None) => {
                let entry: &'static catalog::Entry = model.entry;
                &entry.bounds.quantiles
            }
            (Source::Local { .. }, None) => {
                return Err(RequestError::invalid(
                    "a trusted model's request needs the levels it outputs",
                ));
            }
        };
        let parts = Parts {
            model: source,
            target,
            context,
            queries,
        };
        return forecast(parts, &features, &request["options"], levels).map(Inference::Forecast);
    }
    let Source::Catalog(model) = source else {
        return Err(unsupported_task());
    };
    tabular(
        Parts {
            model,
            target,
            context,
            queries,
        },
        task,
        features,
        settings,
    )
    .map(Inference::Tabular)
}

fn unsupported_task() -> RequestError {
    RequestError::new(
        "UnsupportedTask",
        "selected model does not support this task",
        422,
    )
}

/// What every request names, checked before its task's own checks.
struct Parts<'a, M = Model> {
    model: M,
    target: String,
    context: &'a [Json],
    queries: &'a [Json],
}

/// Context or query rows: unique keys and exactly the request's features.
fn rows<'a>(rows: &'a [Json], features: &[String], scoring: bool) -> Checked<Vec<&'a Object>> {
    let mut keys = HashSet::new();
    let optional: &[&str] = if scoring { &[] } else { &["label", "event_at"] };
    rows.iter()
        .map(|row| {
            let row = fields(row, &["key", "features"], optional)?;
            let key = text(&row["key"], "key")?;
            require(keys.insert(key), "duplicate row key")?;
            for value in fields(&row["features"], features, &[])?.values() {
                number(value, &VALUES, "feature")?;
            }
            if let Some(time) = row.get("event_at") {
                number(time, &VALUES, "event_at")?;
            }
            Ok(row)
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn matrix(rows: &[&Object], features: &[String]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| {
            let Json::Object(values) = &row["features"] else {
                unreachable!("rows checked their features");
            };
            features.iter().map(move |feature| match values[feature] {
                Json::Num(value) => value.as_f64() as f32,
                _ => unreachable!("rows checked their features"),
            })
        })
        .collect()
}

fn tabular(
    parts: Parts<'_>,
    task: Task,
    features: Vec<String>,
    settings: &Object,
) -> Checked<Tabular> {
    require(!features.is_empty(), "tabular inference requires features")?;
    let context = rows(parts.context, &features, false)?;
    let queries = rows(parts.queries, &features, true)?;
    let labeled: Vec<&Object> = context
        .iter()
        .copied()
        .filter(|row| row.contains_key("label"))
        .collect();
    if labeled.is_empty() {
        return Err(RequestError::empty("no usable labeled rows; no score"));
    }
    let labels: Vec<&Json> = labeled.iter().map(|row| &row["label"]).collect();
    let (targets, classes, levels) = if task == Task::Classification {
        require(
            settings.is_empty(),
            "classification has no quantile options",
        )?;
        let classes = class_labels(&labels)?;
        let capacity = parts.model.capacity();
        if classes.len() > capacity {
            return Err(RequestError::new(
                "UnsupportedClassCount",
                format!(
                    "this model serves at most {capacity} classes; @K error-correcting codes serve more"
                ),
                422,
            ));
        }
        let codes = class_codes(&labels, &classes);
        (Targets::Classes(codes), classes, Vec::new())
    } else {
        #[allow(clippy::cast_possible_truncation)]
        let values = labels
            .iter()
            .map(|label| number(label, &VALUES, "label").map(|value| value.as_f64() as f32))
            .collect::<Checked<Vec<f32>>>()?;
        let default = Json::List(
            DEFAULT_QUANTILES
                .iter()
                .map(|level| Json::Num(Num::Float(*level)))
                .collect(),
        );
        let levels = levels(settings.get("quantiles").unwrap_or(&default))?;
        let step = parts.model.entry.bounds.quantile_step.unwrap_or(0.001);
        let grid = (1.0 / step).round_ties_even();
        for level in &levels {
            let scaled = level / step;
            let index = scaled.round_ties_even();
            require(
                (scaled - index).abs() <= 1e-9 && 1.0 <= index && index < grid,
                &format!("quantiles use levels k/{grid} for k=1..{}", grid - 1.0),
            )?;
        }
        (Targets::Values(values), Vec::new(), levels)
    };
    let context = matrix(&labeled, &features);
    let mut hash = Fingerprint::new(
        "tabular",
        parts.model.identity(),
        parts.model.entry.task,
        &parts.target,
    );
    hash.strings(&features);
    hash.count(classes.len());
    for class in &classes {
        hash.label(class);
    }
    hash.count(labeled.len());
    hash.floats(&context);
    match &targets {
        Targets::Classes(codes) => codes
            .iter()
            .for_each(|code| hash.bytes(&code.to_le_bytes())),
        Targets::Values(values) => hash.floats(values),
    }
    Ok(Tabular {
        fingerprint: hash.finish(),
        model: parts.model,
        task,
        target: parts.target,
        queries: matrix(&queries, &features),
        query_keys: queries
            .iter()
            .map(|row| row["key"].text().unwrap_or_default().to_owned())
            .collect(),
        features,
        context,
        targets,
        classes,
        levels,
        supplied_rows: parts.context.len(),
    })
}

/// The labels' classes, in order: one type of string, boolean or integer.
fn class_labels(labels: &[&Json]) -> Checked<Vec<Label>> {
    let label = |value: &Json| match value {
        Json::Bool(value) => Some(Label::Bool(*value)),
        Json::Num(Num::Int(value)) => Some(Label::Int(*value)),
        Json::Text(value) => Some(Label::Text(value.clone())),
        _ => None,
    };
    let kind = std::mem::discriminant(&label(labels[0]).ok_or_else(|| {
        RequestError::invalid("classification labels must be strings, booleans or integers")
    })?);
    let labels = labels
        .iter()
        .map(|value| label(value).filter(|label| std::mem::discriminant(label) == kind))
        .collect::<Option<Vec<Label>>>()
        .ok_or_else(|| RequestError::invalid("class labels must have one type"))?;
    for value in &labels {
        if let Label::Text(value) = value {
            require(
                (1..=catalog::MAX_TEXT).contains(&value.chars().count()),
                "invalid label",
            )?;
        }
    }
    if matches!(labels[0], Label::Bool(_)) {
        return Ok(vec![Label::Bool(false), Label::Bool(true)]);
    }
    Ok(labels
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn class_codes(labels: &[&Json], order: &[Label]) -> Vec<i32> {
    labels
        .iter()
        .map(|value| {
            let label = match value {
                Json::Bool(value) => Label::Bool(*value),
                Json::Num(Num::Int(value)) => Label::Int(*value),
                Json::Text(value) => Label::Text(value.clone()),
                _ => unreachable!("labels were checked"),
            };
            order
                .binary_search(&label)
                .expect("a label is one of its classes") as i32
        })
        .collect()
}

/// One series' observation at a time.
struct Point {
    time: Num,
    value: Option<f64>,
    covariates: BTreeMap<String, Option<f64>>,
}

fn forecast(
    parts: Parts<'_, Source>,
    features: &[String],
    options: &Json,
    declared: &[f64],
) -> Checked<Forecast> {
    let policy = &catalog::catalog().policy;
    require(
        features.is_empty(),
        "forecast uses series covariates, not tabular features",
    )?;
    let options = fields(options, &["series"], &[])?;
    let spec = fields(
        &options["series"],
        &["step", "horizon", "quantiles", "known_covariates"],
        &[],
    )?;
    let step = number(&spec["step"], &STEPS, "step")?;
    require(step.as_f64() > 0.0, "step must be positive")?;
    let horizon = match spec["horizon"] {
        Json::Num(Num::Int(horizon)) if (1..=policy.horizon as i64).contains(&horizon) => {
            usize::try_from(horizon).expect("a horizon is positive")
        }
        _ => {
            return Err(RequestError::invalid(format!(
                "horizon must be 1..{}",
                policy.horizon
            )));
        }
    };
    let levels = levels(&spec["quantiles"])?;
    if !declared.is_empty() && !levels.iter().all(|level| declared.contains(level)) {
        return Err(RequestError::new(
            "UnsupportedQuantiles",
            "the model only returns its own published quantile levels",
            422,
        ));
    }
    let known = spec["known_covariates"]
        .list()
        .filter(|known| known.len() <= policy.covariates)
        .ok_or_else(|| RequestError::invalid("invalid known_covariates"))?
        .iter()
        .map(|name| text(name, "known covariate").map(str::to_owned))
        .collect::<Checked<Vec<String>>>()?;
    let known_set: BTreeSet<&str> = known.iter().map(String::as_str).collect();
    require(
        known_set.len() == known.len(),
        "known covariates must be unique",
    )?;
    require(
        !known_set.contains(parts.target.as_str()),
        "target cannot be a known covariate",
    )?;

    let mut series: BTreeMap<String, (Vec<Point>, HashSet<u64>)> = BTreeMap::new();
    let mut context_keys = HashSet::new();
    let mut clock = None;
    for row in parts.context {
        let row = fields(row, &["key", "series", "time", "value", "covariates"], &[])?;
        let key = text(&row["key"], "key")?;
        require(context_keys.insert(key), "duplicate context key")?;
        let identity = text(&row["series"], "series")?;
        let time = number(&row["time"], &VALUES, "time")?;
        let axis = *clock.get_or_insert(Clock::of(time));
        require(
            Clock::of(time) == axis,
            "cannot mix integer periods and floating UTC seconds",
        )?;
        let (points, times) = series.entry(identity.to_owned()).or_default();
        require(times.insert(time.key()), "duplicate series time")?;
        let value = match &row["value"] {
            Json::Null => None,
            value => Some(number(value, &VALUES, "value")?.as_f64()),
        };
        let Json::Object(named) = &row["covariates"] else {
            return Err(RequestError::invalid("invalid covariates"));
        };
        require(named.len() <= policy.covariates, "invalid covariates")?;
        let mut covariates = BTreeMap::new();
        require(
            named
                .keys()
                .all(|name| (1..=catalog::MAX_TEXT).contains(&name.chars().count())),
            "invalid covariate",
        )?;
        for (name, value) in named {
            let value = match value {
                Json::Null => None,
                value => Some(number(value, &VALUES, "covariate")?.as_f64()),
            };
            covariates.insert(name.clone(), value);
        }
        points.push(Point {
            time,
            value,
            covariates,
        });
    }
    let mut covariates: BTreeSet<&str> = known_set.clone();
    for (points, _) in series.values() {
        for point in points {
            covariates.extend(point.covariates.keys().map(String::as_str));
        }
    }
    require(
        covariates.len() <= policy.covariates,
        &format!("at most {} covariates", policy.covariates),
    )?;
    require(
        !covariates
            .iter()
            .any(|name| series.contains_key(*name) || *name == parts.target),
        "target series cannot be covariates",
    )?;
    require(
        clock != Some(Clock::Periods) || step.is_int(),
        "integer periods require an integer step",
    )?;
    let observed = |points: &[Point]| points.iter().filter(|point| point.value.is_some()).count();
    let usable_observations: usize = series.values().map(|(points, _)| observed(points)).sum();
    if usable_observations == 0 {
        return Err(RequestError::empty(
            "no usable series observations; no forecast",
        ));
    }
    let clock = clock.expect("observations have a time axis");

    struct Query<'a> {
        key: &'a str,
        series: &'a str,
        origin: Num,
        group: Vec<&'a str>,
        times: Vec<Num>,
        future: &'a Object,
    }
    let mut checked = Vec::with_capacity(parts.queries.len());
    let mut query_keys = HashSet::new();
    for query in parts.queries {
        let query = fields(
            query,
            &["key", "series", "origin", "group", "future_covariates"],
            &[],
        )?;
        let key = text(&query["key"], "key")?;
        require(query_keys.insert(key), "duplicate query key")?;
        let identity = text(&query["series"], "series")?;
        let origin = number(&query["origin"], &VALUES, "origin")?;
        require(
            Clock::of(origin) == clock,
            "query and context must use the same time axis",
        )?;
        let times: Vec<Num> = (0..=horizon as i64)
            .map(|index| origin.advanced(step, index))
            .collect();
        require(
            times
                .windows(2)
                .all(|pair| pair[1].as_f64() > pair[0].as_f64()),
            "step is below the time axis precision",
        )?;
        let group = query["group"]
            .list()
            .filter(|group| (1..=policy.group_series).contains(&group.len()))
            .ok_or_else(|| {
                RequestError::invalid(format!(
                    "group must contain 1..{} series",
                    policy.group_series
                ))
            })?
            .iter()
            .map(|member| text(member, "group series"))
            .collect::<Checked<Vec<&str>>>()?;
        require(
            group.iter().collect::<HashSet<_>>().len() == group.len() && group.contains(&identity),
            "invalid query group",
        )?;
        require(
            group.iter().all(|member| series.contains_key(*member)),
            "group series missing from context",
        )?;
        for member in &group {
            let points = &series[*member].0;
            require(
                points
                    .iter()
                    .all(|point| point.time.as_f64() <= origin.as_f64()),
                "future target leakage",
            )?;
            let positions: Vec<f64> = points
                .iter()
                .map(|point| origin.steps_after(point.time, step))
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let bound = policy.context_rows as f64;
            require(
                positions
                    .iter()
                    .all(|position| position.is_finite() && (0.0..bound).contains(position)),
                "series span exceeds context bound",
            )?;
            require(
                positions
                    .iter()
                    .all(|position| (position - position.round_ties_even()).abs() < 1e-6),
                "series times must align to step and origin",
            )?;
            if observed(points) == 0 {
                return Err(RequestError::empty(
                    "query group has no usable series observations",
                ));
            }
        }
        let future = fields(&query["future_covariates"], &known, &[])?;
        for values in future.values() {
            let values = values
                .list()
                .filter(|values| values.len() == horizon)
                .ok_or_else(|| {
                    RequestError::invalid("known-future values must cover the complete horizon")
                })?;
            for value in values {
                number(value, &VALUES, "future covariate")?;
            }
        }
        checked.push(Query {
            key,
            series: identity,
            origin,
            group,
            times,
            future,
        });
    }

    let mut hash = Fingerprint::new(
        "forecast",
        parts.model.identity(),
        Task::Forecast,
        &parts.target,
    );
    hash.num(step);
    hash.strings(
        &known_set
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>(),
    );
    hash.count(series.len());
    for (name, (points, _)) in &series {
        hash.bytes(name.as_bytes());
        let mut points: Vec<&Point> = points.iter().collect();
        points.sort_by(|a, b| a.time.as_f64().total_cmp(&b.time.as_f64()));
        hash.count(points.len());
        for point in points {
            hash.num(point.time);
            hash.observation(point.value);
            hash.count(point.covariates.len());
            for (covariate, value) in &point.covariates {
                hash.bytes(covariate.as_bytes());
                hash.observation(*value);
            }
        }
    }
    let fingerprint = hash.finish();

    let queries = checked
        .into_iter()
        .map(|query| {
            let points: Vec<&[Point]> = query
                .group
                .iter()
                .map(|member| series[*member].0.as_slice())
                .collect();
            let group = aligned(
                &points,
                &known_set,
                query.origin,
                step,
                horizon,
                query.future,
            )?;
            let mut hash = Fingerprint(Sha256::new());
            hash.bytes(b"gf-group-v1");
            hash.bytes(&fingerprint);
            hash.strings(
                &query
                    .group
                    .iter()
                    .map(|member| (*member).to_owned())
                    .collect::<Vec<_>>(),
            );
            hash.num(query.origin);
            hash.count(horizon);
            hash.count(query.future.len());
            for (name, values) in query.future {
                hash.bytes(name.as_bytes());
                for value in values.list().unwrap_or_default() {
                    if let Json::Num(value) = value {
                        hash.observation(Some(value.as_f64()));
                    }
                }
            }
            Ok(SeriesQuery {
                fingerprint: hash.finish(),
                key: query.key.to_owned(),
                series: query.series.to_owned(),
                origin: query.origin,
                member: query
                    .group
                    .iter()
                    .position(|member| *member == query.series)
                    .expect("a query's series is in its group"),
                members: query
                    .group
                    .iter()
                    .map(|member| (*member).to_owned())
                    .collect(),
                times: query.times[1..].to_vec(),
                group,
            })
        })
        .collect::<Checked<Vec<SeriesQuery>>>()?;

    Ok(Forecast {
        fingerprint,
        model: parts.model,
        target: parts.target,
        step,
        horizon,
        levels,
        known,
        queries,
        clock,
        supplied_rows: parts.context.len(),
        usable_observations,
    })
}

/// Aligns a query's group: its target rows, then its covariates sorted by name.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn aligned(
    members: &[&[Point]],
    known: &BTreeSet<&str>,
    origin: Num,
    step: Num,
    horizon: usize,
    future: &Object,
) -> Checked<AlignedGroup> {
    let mut names: BTreeSet<&str> = known.clone();
    for points in members {
        for point in *points {
            names.extend(point.covariates.keys().map(String::as_str));
        }
    }
    let covariates: Vec<&str> = names.into_iter().collect();
    let offset = |time: Num| origin.steps_after(time, step).round_ties_even() as usize;
    let steps = members
        .iter()
        .flat_map(|points| points.iter().map(|point| offset(point.time)))
        .max()
        .expect("a group has observations")
        + 1;
    let rows = members.len() + covariates.len();
    let mut values = vec![0.0_f32; rows * steps];
    let mut observed = vec![false; rows * steps];
    let mut roles = vec![TARGET; rows];
    let mut ahead = vec![0.0_f32; rows * horizon];
    let mut ahead_observed = vec![false; rows * horizon];
    for (index, points) in members.iter().enumerate() {
        for point in *points {
            let position = steps - 1 - offset(point.time);
            if let Some(value) = point.value {
                values[index * steps + position] = value as f32;
                observed[index * steps + position] = true;
            }
            for (name, value) in &point.covariates {
                let Some(value) = value else { continue };
                let row = members.len()
                    + covariates
                        .binary_search(&name.as_str())
                        .expect("a named covariate");
                let at = row * steps + position;
                if observed[at] && values[at] != *value as f32 {
                    return Err(RequestError::new(
                        "UnsupportedCovariates",
                        "grouped targets must share named covariate values at each time",
                        422,
                    ));
                }
                values[at] = *value as f32;
                observed[at] = true;
            }
        }
    }
    for (offset, name) in covariates.iter().enumerate() {
        let row = members.len() + offset;
        if known.contains(name) {
            roles[row] = KNOWN;
            let Some(Json::List(series)) = future.get(*name) else {
                unreachable!("queries carry every known covariate");
            };
            for (step, value) in series.iter().enumerate() {
                if let Json::Num(value) = value {
                    ahead[row * horizon + step] = value.as_f64() as f32;
                }
                ahead_observed[row * horizon + step] = true;
            }
        } else {
            roles[row] = PAST;
        }
    }
    Ok(AlignedGroup {
        rows,
        steps,
        horizon,
        values,
        observed,
        roles,
        future: ahead,
        future_observed: ahead_observed,
    })
}

/// SHA-256 over a length-prefixed canonical encoding of a request's context.
struct Fingerprint(Sha256);

impl Fingerprint {
    fn new(kind: &str, (name, digest): (String, &str), task: Task, target: &str) -> Self {
        let mut hash = Self(Sha256::new());
        hash.bytes(b"gf-context-v1");
        hash.bytes(kind.as_bytes());
        hash.bytes(name.as_bytes());
        hash.bytes(digest.as_bytes());
        hash.bytes(task.as_str().as_bytes());
        hash.bytes(target.as_bytes());
        hash
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }

    fn count(&mut self, count: usize) {
        self.0.update((count as u64).to_le_bytes());
    }

    fn strings(&mut self, strings: &[String]) {
        self.count(strings.len());
        for string in strings {
            self.bytes(string.as_bytes());
        }
    }

    fn label(&mut self, label: &Label) {
        match label {
            Label::Bool(value) => self.bytes(&[0, u8::from(*value)]),
            Label::Int(value) => {
                self.0.update([1]);
                self.bytes(&value.to_le_bytes());
            }
            Label::Text(value) => {
                self.0.update([2]);
                self.bytes(value.as_bytes());
            }
        }
    }

    fn num(&mut self, number: Num) {
        match number {
            Num::Int(value) => {
                self.0.update([0]);
                self.0.update(value.to_le_bytes());
            }
            Num::Float(value) => {
                self.0.update([1]);
                self.0.update((value + 0.0).to_bits().to_le_bytes());
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn observation(&mut self, value: Option<f64>) {
        match value {
            None => self.0.update([0]),
            Some(value) => {
                self.0.update([1]);
                self.0
                    .update(((value as f32) + 0.0).to_bits().to_le_bytes());
            }
        }
    }

    fn floats(&mut self, values: &[f32]) {
        self.count(values.len());
        for value in values {
            self.0.update((value + 0.0).to_bits().to_le_bytes());
        }
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}
