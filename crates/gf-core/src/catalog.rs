// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The model catalog as request validation reads it, and model references.
//!
//! `generalforecast.models.catalog` is the single source: it emits the
//! manifest checked in beside this crate, which is compiled in, and a Python
//! test fails when the two drift. A reference names a built-in model by any
//! of its spellings, optionally pinned to its content digest, then optionally
//! `@K` error-correcting codes and `*N` ensemble estimators; resolving it for
//! a serving surface applies that surface's gates and the v1 serving policy,
//! with the same codes, messages and statuses the Python catalog service
//! answers.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::request::{RequestError, python_repr};

/// The manifest `python -m generalforecast.models.catalog` prints.
const MANIFEST: &str = include_str!("../catalog.json");

/// Most characters in a model reference, as in any request identity string.
pub const MAX_TEXT: usize = 256;

/// What a model's weights do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Classification,
    Regression,
    Forecast,
}

impl Task {
    /// The task's name on the wire.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Classification => "classification",
            Self::Regression => "regression",
            Self::Forecast => "forecast",
        }
    }

    /// Whether it is a classifier or regressor rather than a forecaster.
    #[must_use]
    pub fn tabular(self) -> bool {
        self != Self::Forecast
    }
}

/// The bounds of what a service allows, frozen as v1 so locks keep their identity.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub context_rows: usize,
    pub query_rows: usize,
    pub features: usize,
    pub horizon: usize,
    pub group_series: usize,
    pub covariates: usize,
    pub max_codes: u64,
    pub max_estimators: u64,
    pub max_members: u64,
}

impl Policy {
    /// Why the policy refuses a variant's cost, or None if it runs.
    #[must_use]
    pub fn refuses(&self, codes: Option<&Count>, estimators: Option<&Count>) -> Option<String> {
        let codes = codes.map_or(1, |count| count.value);
        let estimators = estimators.map_or(1, |count| count.value);
        if codes > self.max_codes {
            return Some(format!(
                "at most @{} error-correcting codes",
                self.max_codes
            ));
        }
        if estimators > self.max_estimators {
            return Some(format!("at most *{} estimators", self.max_estimators));
        }
        if codes.saturating_mul(estimators) > self.max_members {
            return Some(format!(
                "at most {} codes times estimators",
                self.max_members
            ));
        }
        None
    }
}

/// What a model serves before its weights are read.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    /// Classes a classifier serves natively, without `@K`.
    pub classes: Option<usize>,
    /// A forecaster's output levels, when known without its weights.
    pub quantiles: Vec<f64>,
    /// The spacing of a regressor's output quantile grid.
    pub quantile_step: Option<f64>,
}

/// Which surfaces beyond trusted local use the evidence qualifies a model for.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub predictor: bool,
    pub remote: bool,
    /// Surfaces beyond local that run its `@K` and `*N` variants.
    pub variants: Vec<String>,
}

impl Gates {
    /// Whether remote serving may run the model, or one of its variants.
    #[must_use]
    pub fn remote(&self, variant: bool) -> bool {
        self.remote && (!variant || self.variants.iter().any(|surface| surface == "remote"))
    }
}

/// What one call may send a model, as the registry's `Limits` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub context_rows: usize,
    pub queries: usize,
    pub inputs: usize,
}

/// One built-in model.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// The canonical id, such as `general-forecast/tabicl-v2-classifier`.
    pub id: String,
    /// The short name, such as `tabicl-v2-classifier`: the resource routes name.
    pub name: String,
    pub task: Task,
    pub limits: Limits,
    /// The `sha256:` digest of its pinned file manifest, if it has one.
    pub digest: Option<String>,
    /// Whether its file hashes are recorded.
    pub pinned: bool,
    /// Whether its license permits serving it on General Forecast infrastructure.
    pub hostable: bool,
    pub bounds: Bounds,
    pub gates: Gates,
}

impl Entry {
    /// Most classes a variant of this classifier serves, or 0 for a regressor
    /// or forecaster.
    #[must_use]
    pub fn capacity(&self, codes: bool, max_classes: usize) -> usize {
        match self.bounds.classes {
            None => 0,
            Some(_) if codes => max_classes,
            Some(slots) => slots,
        }
    }
}

/// Every built-in model, its spellings and the serving policy.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    /// The most classes a classifier serves through `@K` codes.
    pub max_classes: usize,
    pub policy: Policy,
    /// Each canonical id, to the short name of its model.
    pub names: BTreeMap<String, String>,
    /// Old spellings (bare names, upstream ids), to the canonical id to use instead.
    pub aliases: BTreeMap<String, String>,
    pub entries: Vec<Entry>,
}

impl Catalog {
    /// The built-in model with this canonical name.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// The model a spelling resolves to.
    #[must_use]
    pub fn named(&self, spelling: &str) -> Option<&Entry> {
        self.names.get(spelling).and_then(|name| self.entry(name))
    }
}

/// The compiled-in catalog.
///
/// # Panics
///
/// If the manifest checked in beside the crate is not a catalog; its tests
/// parse it, so a build that passed them never does.
#[must_use]
pub fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| serde_json::from_str(MANIFEST).expect("the checked-in catalog parses"))
}

/// Where a model is served, which decides the gates a reference passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// A caller that has already admitted the model, such as a predictor lock:
    /// no gates, only the serving policy.
    Local,
    /// General Forecast's hosted Models API: remote-qualified, hostable models.
    Remote,
    /// A server its operator runs, such as `gf serve`: also pinned base
    /// models whose license forbids hosting them for others.
    SelfHosted,
}

impl std::str::FromStr for Surface {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "local" => Ok(Self::Local),
            "remote" => Ok(Self::Remote),
            "self-hosted" => Ok(Self::SelfHosted),
            _ => Err(format!("unknown serving surface {text:?}")),
        }
    }
}

/// A positive `@K` or `*N` count as spelled, saturating past `u64`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Count {
    pub value: u64,
    /// The decimal spelling without leading zeros, as Python prints the count.
    digits: String,
}

impl Count {
    fn parse(digits: &str) -> Self {
        let trimmed = digits.trim_start_matches('0');
        let digits = if trimmed.is_empty() { "0" } else { trimmed };
        Self {
            value: digits.parse().unwrap_or(u64::MAX),
            digits: digits.to_owned(),
        }
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.digits)
    }
}

/// A resolved reference: a built-in model, its optional pin and its variant.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub entry: &'static Entry,
    /// The `sha256:` pin the reference names, which equals the entry's digest.
    pub digest: Option<String>,
    pub codes: Option<Count>,
    pub estimators: Option<Count>,
}

impl Model {
    /// Its public id: `general-forecast/<name>`, with any variant suffix.
    #[must_use]
    pub fn canonical_id(&self) -> String {
        self.to_string()
    }

    /// Its public id pinned to the catalog's digest, as results report it.
    #[must_use]
    pub fn pinned_id(&self) -> String {
        match &self.entry.digest {
            Some(digest) if self.digest.is_none() => {
                let mut id = format!("{}@{digest}", self.entry.id);
                if let Some(codes) = &self.codes {
                    id.push_str(&format!("@{codes}"));
                }
                if let Some(estimators) = &self.estimators {
                    id.push_str(&format!("*{estimators}"));
                }
                id
            }
            _ => self.canonical_id(),
        }
    }

    /// Whether it wraps the model in codes or an ensemble.
    #[must_use]
    pub fn variant(&self) -> bool {
        self.codes.is_some() || self.estimators.is_some()
    }

    /// Its spelling and pinned digest, which a context's fingerprint covers.
    #[must_use]
    pub fn identity(&self) -> (String, &str) {
        (
            self.to_string(),
            self.entry.digest.as_deref().unwrap_or_default(),
        )
    }

    /// Most classes it serves: `@K` codes serve up to the catalog's maximum.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.entry
            .capacity(self.codes.is_some(), catalog().max_classes)
    }
}

impl fmt::Display for Model {
    /// The canonical spelling, which `generalforecast.models.catalog.parse`
    /// parses back to the same reference: the id, then the pin and variant.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.entry.id)?;
        if let Some(digest) = &self.digest {
            write!(f, "@{digest}")?;
        }
        if let Some(codes) = &self.codes {
            write!(f, "@{codes}")?;
        }
        if let Some(estimators) = &self.estimators {
            write!(f, "*{estimators}")?;
        }
        Ok(())
    }
}

/// Why a reference names no runnable model, as `ModelRefError` codes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefError {
    pub code: &'static str,
    pub message: String,
}

/// The parts of a reference's spelling: name, then the optional pin and counts.
struct Spelled<'a> {
    name: &'a str,
    digest: Option<&'a str>,
    codes: Option<&'a str>,
    estimators: Option<&'a str>,
}

fn digits(text: &str) -> usize {
    text.bytes().take_while(u8::is_ascii_digit).count()
}

/// `name(@sha256:<64 hex>)?(@digits)?(\*digits)?` exactly, or None.
fn spelled(text: &str) -> Option<Spelled<'_>> {
    let end = text.find(['@', '*']).unwrap_or(text.len());
    if end == 0 {
        return None;
    }
    let (name, mut rest) = text.split_at(end);
    let mut digest = None;
    if let Some(pin) = rest.strip_prefix("@sha256:") {
        let hex = pin
            .bytes()
            .take_while(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
            .count();
        let after = pin.get(64..).unwrap_or("");
        if hex >= 64 && (after.is_empty() || after.starts_with(['@', '*'])) {
            digest = Some(&rest[1..1 + 7 + 64]);
            rest = after;
        }
    }
    let mut codes = None;
    if let Some(count) = rest.strip_prefix('@') {
        let length = digits(count);
        if length == 0 {
            return None;
        }
        codes = Some(&count[..length]);
        rest = &count[length..];
    }
    let mut estimators = None;
    if let Some(count) = rest.strip_prefix('*') {
        let length = digits(count);
        if length == 0 {
            return None;
        }
        estimators = Some(&count[..length]);
        rest = &count[length..];
    }
    rest.is_empty().then_some(Spelled {
        name,
        digest,
        codes,
        estimators,
    })
}

/// Every canonical name, sorted, as Python prints a list of them.
fn sorted_names() -> String {
    let mut names: Vec<&str> = catalog()
        .entries
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    names.sort_unstable();
    names
        .iter()
        .map(|name| python_repr(name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether a name is a custom model's `module:attribute`: dotted identifiers, a colon, an identifier.
fn custom_shaped(name: &str) -> bool {
    let Some((module, attribute)) = name.split_once(':') else {
        return false;
    };
    let identifier = |part: &str| {
        let mut chars = part.chars();
        chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
            && chars.all(|c| c.is_alphanumeric() || c == '_')
    };
    module.split('.').all(identifier) && identifier(attribute)
}

/// The one message a custom model is refused with off the local surface.
fn custom_refused(spec: &str) -> String {
    format!(
        "custom model `{spec}` runs locally only; predictors, gf push and serving take a \
         catalog model id, such as {ORG}/tabicl-v2-classifier"
    )
}

/// Parses an untrusted reference as `generalforecast.models.catalog.parse` does.
///
/// # Errors
///
/// `InvalidModel` for a malformed spelling or a zero count, `UnsupportedModel`
/// for a custom model's `module:attribute`, which runs locally only (as
/// `generalforecast.models.catalog.custom_refused` says), `UnknownModel` for no built-in model,
/// `ModelMismatch` for a pin to other content, and `UnsupportedCapability`
/// for a variant the model cannot take.
pub fn parse(spec: &str) -> Result<Model, RefError> {
    let invalid = || RefError {
        code: "InvalidModel",
        message: format!(
            "a model spec is a name, then optionally @K, then optionally *N: {}",
            python_repr(spec)
        ),
    };
    let prefixed = spec
        .strip_prefix(&format!("{ORG}/"))
        .or_else(|| spec.strip_prefix("catalog/"));
    let parts = prefixed
        .and_then(spelled)
        .or_else(|| spelled(spec))
        .ok_or_else(invalid)?;
    let codes = parts.codes.map(Count::parse);
    let estimators = parts.estimators.map(Count::parse);
    if [&codes, &estimators]
        .into_iter()
        .flatten()
        .any(|count| count.value == 0)
    {
        return Err(RefError {
            code: "InvalidModel",
            message: format!("counts in a model spec are positive: {}", python_repr(spec)),
        });
    }
    // A bare name or an upstream id that names a model is refused with the
    // canonical id, as the Python catalog refuses it; an unknown one falls
    // through to the catalog lookup and its `UnknownModel`.
    if prefixed.is_none() && !parts.name.contains(':') {
        let canonical = catalog().aliases.get(parts.name).cloned().or_else(|| {
            catalog()
                .named(&format!("{ORG}/{}", parts.name))
                .map(|entry| entry.id.clone())
        });
        if let Some(canonical) = canonical {
            return Err(RefError {
                code: "InvalidModel",
                message: format!("`{spec}` is not a catalog model id; use {canonical}"),
            });
        }
    }
    if parts.name.contains(':') {
        if custom_shaped(parts.name) && spec.starts_with(parts.name) {
            if codes.is_some() || estimators.is_some() || parts.digest.is_some() {
                return Err(RefError {
                    code: "UnsupportedCapability",
                    message: format!(
                        "a custom model takes no pin, codes or estimators: {}",
                        python_repr(spec)
                    ),
                });
            }
            return Err(RefError {
                code: "UnsupportedModel",
                message: custom_refused(parts.name),
            });
        }
        return Err(RefError {
            code: "InvalidModel",
            message: format!(
                "`{}` is neither a catalog model id, {ORG}/<name>, nor a custom model's \
                 \"module:attribute\"",
                parts.name
            ),
        });
    }
    let entry = catalog()
        .named(&format!("{ORG}/{}", parts.name))
        .ok_or_else(|| RefError {
            code: "UnknownModel",
            message: format!(
                "unknown model {}; built-in models are [{}]",
                python_repr(parts.name),
                sorted_names()
            ),
        })?;
    if parts.digest.is_some() && parts.digest != entry.digest.as_deref() {
        return Err(RefError {
            code: "ModelMismatch",
            message: "requested model pin differs from catalog".into(),
        });
    }
    let model = Model {
        entry,
        digest: parts.digest.map(str::to_owned),
        codes,
        estimators,
    };
    if model.variant() && !entry.task.tabular() {
        return Err(RefError {
            code: "UnsupportedCapability",
            message: format!(
                "a forecaster takes no codes or estimators: {}",
                python_repr(spec)
            ),
        });
    }
    if model.codes.is_some() && entry.task != Task::Classification {
        return Err(RefError {
            code: "UnsupportedCapability",
            message: format!(
                "error-correcting codes need a classifier: {}",
                python_repr(spec)
            ),
        });
    }
    Ok(model)
}

/// What a request's model names: a built-in model, gated for `surface`.
///
/// # Errors
///
/// What `resolve` refuses.
pub fn admit(spec: Option<&str>, surface: Surface) -> Result<Model, RequestError> {
    resolve(spec, surface)
}

/// The model a request names, if the surface may serve it.
///
/// `spec` is None for a value that is not a string. Hosted serving takes
/// remote-qualified, hostable models; a self-hosted server also takes a
/// pinned base model whose license only its operator can grant; a local
/// caller has admitted the model already. Every surface applies the serving
/// policy's variant bounds.
///
/// # Errors
///
/// The `ServiceError` the Python catalog service raises for the same
/// reference: `ValidationError`, `ModelNotFound` (404), `ModelMismatch`,
/// `UnsupportedModel`, `UnsupportedCapability` or `CapacityExceeded`.
/// The organisation every catalog model id starts with.
pub const ORG: &str = "general-forecast";

/// A request's model id, which is `general-forecast/<name>` and nothing
/// else: a bare or `catalog/` spelling is refused with the id to use.
///
/// # Errors
///
/// `ValidationError` (422) with `param: "model"` and a `hint` for a known
/// model spelled another way; otherwise as `resolve`.
pub fn canonical(spec: Option<&str>, surface: Surface) -> Result<Model, RequestError> {
    let spec = spec
        .filter(|spec| (1..=MAX_TEXT).contains(&spec.chars().count()))
        .ok_or_else(|| RequestError::invalid("invalid model").param("model"))?;
    if let Some(name) = spec.strip_prefix(&format!("{ORG}/"))
        && !name.is_empty()
        && !name.starts_with("catalog/")
    {
        return resolve(Some(spec), surface).map_err(|error| error.param("model"));
    }
    let mut error = RequestError::invalid(format!(
        "model ids are spelled {ORG}/<name>; {} is not one",
        python_repr(spec)
    ))
    .param("model");
    let bare = spelled(spec).map_or(spec, |parts| parts.name);
    if let Some(canonical) = catalog().aliases.get(bare) {
        error = error.hint(format!("use {canonical}"));
    } else if let Ok(model) = resolve(Some(&format!("{ORG}/{spec}")), surface) {
        error = error.hint(format!("use {}", model.canonical_id()));
    }
    Err(error)
}

pub fn resolve(spec: Option<&str>, surface: Surface) -> Result<Model, RequestError> {
    let spec = spec
        .filter(|spec| (1..=MAX_TEXT).contains(&spec.chars().count()))
        .ok_or_else(|| RequestError::invalid("invalid model"))?;
    let model = parse(spec).map_err(|error| match error.code {
        "UnknownModel" | "InvalidModel" => {
            RequestError::new("ModelNotFound", "model is not in the curated catalog", 404)
        }
        code => RequestError::new(code, error.message, 422),
    })?;
    let entry = model.entry;
    match surface {
        Surface::Local => {}
        Surface::SelfHosted if !entry.hostable => {
            if model.variant() {
                return Err(RequestError::new(
                    "UnsupportedCapability",
                    format!("{model} variants are not qualified"),
                    422,
                ));
            }
            if !entry.pinned {
                return Err(RequestError::new(
                    "UnsupportedModel",
                    format!("{} has no pinned files", entry.name),
                    422,
                ));
            }
        }
        Surface::Remote | Surface::SelfHosted => {
            if !entry.hostable {
                return Err(RequestError::new(
                    "UnsupportedModel",
                    format!(
                        "{} is local-only: its license does not permit hosted serving",
                        entry.name
                    ),
                    422,
                ));
            }
            if !entry.gates.remote(false) {
                return Err(RequestError::new(
                    "ModelNotFound",
                    "model is not qualified for remote serving",
                    404,
                ));
            }
            if !entry.gates.remote(model.variant()) {
                return Err(RequestError::new(
                    "UnsupportedCapability",
                    format!("{model} variants are not qualified remotely"),
                    422,
                ));
            }
        }
    }
    if let Some(refused) = catalog()
        .policy
        .refuses(model.codes.as_ref(), model.estimators.as_ref())
    {
        return Err(RequestError::new(
            "CapacityExceeded",
            format!("serving policy allows {refused}"),
            422,
        ));
    }
    Ok(model)
}
