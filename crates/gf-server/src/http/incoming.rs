// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reading a request: which route it names and whether the front answers it,
//! its query, its headers, and its envelope, parsed strictly and bounded.

use super::*;

/// The deployment environment a predictor's `predict?env=` names, if it
/// names one: the route's only query parameter, given once, a resource name.
pub(super) fn environment(
    query: Option<&str>,
    action: &Action,
) -> Result<Option<String>, ApiError> {
    let Some(query) = query else {
        return Ok(None);
    };
    require(
        *action == Action::PredictorPredict,
        "query parameters are not supported",
    )?;
    let mut named = None;
    for pair in query.split('&') {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        require(
            key == ENVIRONMENT_PARAMETER && named.is_none(),
            "env, once, is the only query parameter a prediction takes",
        )?;
        named = Some(name(value).map_err(|error| error.param(ENVIRONMENT_PARAMETER))?);
    }
    Ok(named)
}

/// Refuses repeated headers and header values that are too long or split lines.
pub(super) fn check_headers(headers: &HeaderMap) -> Result<(), ApiError> {
    for name in headers.keys() {
        require(
            headers.get_all(name).iter().count() == 1,
            "duplicate HTTP header",
        )?;
    }
    for value in headers.values() {
        require(value.len() <= 8192, "invalid HTTP header")?;
    }
    Ok(())
}

/// On loopback, refuses DNS-rebound hosts and cross-origin browser calls.
pub(super) fn check_origin(headers: &HeaderMap) -> Result<(), ApiError> {
    let host = headers
        .get(header::HOST)
        .and_then(|host| host.to_str().ok())
        .unwrap_or_default();
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split_once(']').map(|(name, _)| name),
        None => Some(host.rsplit_once(':').map_or(host, |(name, port)| {
            if port.bytes().all(|byte| byte.is_ascii_digit()) {
                name
            } else {
                host
            }
        })),
    };
    let local = name.is_some_and(|name| {
        ["127.0.0.1", "localhost", "::1"].contains(&name)
            || name.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
    });
    require(
        local && !host.contains('@') && !host.contains('/'),
        "untrusted Host",
    )?;
    if let Some(origin) = headers.get(header::ORIGIN) {
        require(
            origin.to_str().ok() == Some(&format!("http://{host}")),
            "untrusted Origin",
        )?;
    }
    Ok(())
}

/// Behind one endpoint, refuses other hosts and every browser call.
pub(super) fn check_endpoint(headers: &HeaderMap, hosts: &[String]) -> Result<(), ApiError> {
    let host = headers
        .get(header::HOST)
        .and_then(|host| host.to_str().ok());
    if !host.is_some_and(|host| hosts.iter().any(|allowed| allowed == host)) {
        return Err(ApiError::new(
            "Forbidden",
            "unrecognized endpoint Host",
            403,
        ));
    }
    if headers.contains_key(header::ORIGIN) {
        return Err(ApiError::new(
            "Forbidden",
            "browser origins are not supported",
            403,
        ));
    }
    Ok(())
}

/// An envelope's resource name: a plain name, or for a model its canonical
/// id `general-forecast/<name>`, kept as sent.
fn resource_name(kind: &str, value: &str) -> Result<String, ApiError> {
    match value.strip_prefix(&format!("{}/", catalog::ORG)) {
        Some(bare) if kind == "model" => {
            name(bare)?;
            Ok(value.to_string())
        }
        _ => name(value),
    }
}

/// Whether an envelope's resource name names the route's resource; a model's
/// canonical id names its bare route segment.
pub(super) fn names_match(envelope: &str, resource: &str) -> bool {
    envelope == resource
        || envelope
            .strip_prefix(&format!("{}/", catalog::ORG))
            .is_some_and(|bare| bare == resource)
}

/// A resource name: a lowercase letter, then up to 62 letters, digits or `-`.
pub(super) fn name(value: &str) -> Result<String, ApiError> {
    let mut bytes = value.bytes();
    let valid = value.len() <= 63
        && bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
    require(valid, "invalid name")?;
    Ok(value.to_string())
}

/// Whether `method` on `path` is the front's to answer even with an upstream
/// behind it: health, the catalog models it serves, and predictions,
/// however their segments are cased or split, so no inference route, nor a
/// misspelling of one, is ever proxied. Every other route streams upstream:
/// the predictor registry and each predictor's lifecycle, its creation,
/// revisions, environments, promotions, rollbacks, metrics, logs and
/// deletion, are the upstream's, which `resolve` answers only from the
/// front's own predictors when nothing stands behind it.
#[must_use]
pub fn owns(method: &Method, path: &str) -> bool {
    if matches!(path, "/livez" | "/healthz" | "/readyz") {
        return true;
    }
    // Segments are read as an upstream would read them, escapes decoded; a
    // path that had any is the front's, which `resolve` refuses, rather than
    // a spelling of one of its routes an upstream might take for it.
    let escaped = path.contains('%');
    let parts: Vec<String> = path
        .split('/')
        .filter(|part| !part.is_empty())
        .map(|part| percent_decoded(part).to_ascii_lowercase())
        .collect();
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    let [api, projects, _, tail @ ..] = parts.as_slice() else {
        return false;
    };
    if *api != "v1" || *projects != "projects" {
        return false;
    }
    if escaped && !tail.is_empty() {
        return true;
    }
    route(method, tail)
        .is_some_and(|action| action != Action::Predictors && action != Action::Predictor)
}

/// The action a route's segments after its project name, if it is one the
/// front serves.
pub(super) fn route(method: &Method, tail: &[&str]) -> Option<Action> {
    match (method, tail) {
        (&Method::GET, ["models"]) => Some(Action::List),
        (&Method::GET, ["models", _]) => Some(Action::Get),
        (&Method::GET, ["predictors"]) => Some(Action::Predictors),
        (&Method::GET, ["predictors", _]) => Some(Action::Predictor),
        (&Method::POST, ["predictors", _, "predict"]) => Some(Action::PredictorPredict),
        (&Method::POST, ["predict"]) => Some(Action::AdHoc),
        _ => None,
    }
}

/// Whether `path` is one of the control plane's service-to-service routes,
/// under `/internal`, which only this server's own calls may reach; a
/// client's request for one is refused rather than proxied.
#[must_use]
pub fn internal(path: &str) -> bool {
    path.split('/')
        .find(|part| !part.is_empty())
        .is_some_and(|first| percent_decoded(first).eq_ignore_ascii_case(INTERNAL))
}

/// Whether `path` could reach an upstream as another route than the front
/// reads it as: a `.` or `..` segment, plain or escaped, which the
/// upstream's URL parser resolves away, or a separator, `/` or `\`, escaped
/// inside a segment, or a backslash, which it may take for one. Such a path
/// is refused before it is routed, so neither `/internal` nor a route the
/// front owns reaches the upstream under another spelling.
#[must_use]
pub fn ambiguous(path: &str) -> bool {
    path.contains('\\')
        || path
            .split('/')
            .map(percent_decoded)
            .any(|part| part == "." || part == ".." || part.contains('/') || part.contains('\\'))
}

/// `segment` with its `%XX` escapes decoded; an escape that is not one is
/// kept as it is.
fn percent_decoded(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let escaped = (bytes[index] == b'%')
            .then(|| bytes.get(index + 1..index + 3))
            .flatten()
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match escaped {
            Some(byte) => {
                decoded.push(byte);
                index += 3;
            }
            None => {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

/// Maps the Models API paths to a project, action and resource; every other
/// path is NotFound.
pub(super) fn resolve(method: &Method, path: &str) -> Result<(String, Action, String), ApiError> {
    let not_found = || ApiError::new("NotFound", "route not found", 404);
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 5 || parts[..3] != ["", "v1", "projects"] {
        return Err(not_found());
    }
    if parts[3..].iter().any(|part| part.is_empty()) {
        return Err(not_found());
    }
    let project = name(parts[3])?;
    let tail = &parts[4..];
    let action = route(method, tail).ok_or_else(not_found)?;
    let resource = match action {
        Action::List => "models".to_string(),
        Action::Predictors => "predictors".to_string(),
        Action::AdHoc => "ad-hoc".to_string(),
        Action::Get | Action::Predictor | Action::PredictorPredict => name(tail[1])?,
    };
    Ok((project, action, resource))
}

/// JSON parsed without accepting ambiguous objects, which repeat a key.
struct Unique(Value);

impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueVisitor).map(Unique)
    }
}

struct UniqueVisitor;

impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(Unique(item)) = seq.next_element()? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let Unique(value) = map.next_value()?;
            if object.insert(key, value).is_some() {
                return Err(de::Error::custom(DUPLICATE));
            }
        }
        Ok(Value::Object(object))
    }
}

const DUPLICATE: &str = "duplicate JSON field";

/// Checks JSON is portable and bounded: nested at most 16 deep, at most
/// 65,536 values, integers exact in a double.
fn bounded(value: &Value) -> Result<(), ApiError> {
    fn visit(value: &Value, depth: usize, count: &mut usize) -> Result<(), ApiError> {
        *count += 1;
        require(
            depth <= MAX_DEPTH && *count <= MAX_NODES,
            "JSON structure too large",
        )?;
        match value {
            Value::Object(object) => object
                .values()
                .try_for_each(|child| visit(child, depth + 1, count)),
            Value::Array(items) => items
                .iter()
                .try_for_each(|child| visit(child, depth + 1, count)),
            Value::Number(number) => require(
                number.as_f64().is_some()
                    && (number.is_f64()
                        || number
                            .as_i64()
                            .is_some_and(|integer| integer.abs() <= MAX_INTEGER)),
                "integer exceeds portable JSON range",
            ),
            _ => Ok(()),
        }
    }
    let mut count = 0;
    visit(value, 0, &mut count)
}

/// Checks an object has exactly the `required` keys and perhaps the `optional` ones.
pub(super) fn fields<'a>(
    value: &'a Value,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a Map<String, Value>, ApiError> {
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::validation("expected an object"))?;
    let mut sorted = required.to_vec();
    sorted.sort_unstable();
    let listed: Vec<String> = sorted.iter().map(|key| format!("'{key}'")).collect();
    require(
        required.iter().all(|key| object.contains_key(*key)),
        &format!("required fields: [{}]", listed.join(", ")),
    )?;
    require(
        object
            .keys()
            .all(|key| required.contains(&key.as_str()) || optional.contains(&key.as_str())),
        "unknown fields",
    )?;
    Ok(object)
}

/// A request envelope: its scope and resource, and its payload.
pub(super) fn parse_envelope(body: &[u8]) -> Result<(Scope, Value), ApiError> {
    require(
        !body.is_empty() && body.len() <= MAX_BYTES,
        "bounded JSON body required",
    )?;
    let value = match serde_json::from_slice::<Unique>(body) {
        Ok(Unique(value)) => value,
        Err(error) if error.to_string().starts_with(DUPLICATE) => {
            return Err(ApiError::validation(DUPLICATE));
        }
        Err(_) => return Err(ApiError::validation("invalid JSON body")),
    };
    require(value.is_object(), "expected JSON object")?;
    bounded(&value)?;
    let object = fields(
        &value,
        &[
            "api_version",
            "project",
            "request_id",
            "resource",
            "payload",
        ],
        &[],
    )?;
    require(object["api_version"] == VERSION, "unsupported api_version")?;
    let resource = fields(&object["resource"], &["kind", "name"], &[])?;
    let project = name(object["project"].as_str().unwrap_or_default())?;
    let request_id = object["request_id"]
        .as_str()
        .filter(|id| !id.is_empty() && id.chars().count() <= 256)
        .ok_or_else(|| ApiError::validation("invalid request_id"))?;
    let kind = resource["kind"]
        .as_str()
        .filter(|kind| RESOURCE_KINDS.contains(kind))
        .ok_or_else(|| ApiError::validation("invalid resource kind"))?;
    require(object["payload"].is_object(), "expected JSON object")?;
    let resource_name = resource_name(kind, resource["name"].as_str().unwrap_or_default())?;
    let scope = Scope {
        project,
        request_id: request_id.to_string(),
        kind: kind.to_string(),
        name: resource_name,
    };
    Ok((scope, object["payload"].clone()))
}
