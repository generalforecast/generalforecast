// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The single request validator: parsing, the Models API's refusals and fingerprints.

use gf_core::catalog::{self, Admission, Source, Surface};
use gf_core::forecasting::{KNOWN, PAST, TARGET};
use gf_core::node::Targets;
use gf_core::request::{
    self, Clock, Inference, Label, Limits, Num, Options, RequestError, python_repr,
};
use serde_json::{Value, json};

fn tabular() -> Value {
    json!({
        "model": "tabicl-v2-classifier",
        "task": "classification",
        "target": "outcome",
        "features": ["x", "z"],
        "context": [
            {"key": "c1", "features": {"x": 1, "z": 2}, "label": false},
            {"key": "c2", "features": {"x": 3, "z": 4.5}, "label": true},
        ],
        "queries": [{"key": "q", "features": {"x": 2, "z": 3}}],
        "options": {},
    })
}

fn forecast() -> Value {
    let context: Vec<Value> = [("a", 10), ("b", 30)]
        .iter()
        .flat_map(|(name, base)| {
            (0..4).map(move |i| {
                json!({
                    "key": format!("{name}-{i}"),
                    "series": name,
                    "time": i,
                    "value": base + i,
                    "covariates": {"promo": i % 2, "past": f64::from(i) / 10.0},
                })
            })
        })
        .collect();
    let queries: Vec<Value> = ["a", "b"]
        .iter()
        .map(|name| {
            json!({
                "key": format!("q-{name}"),
                "series": name,
                "origin": 3,
                "group": ["a", "b"],
                "future_covariates": {"promo": [0, 1, 0]},
            })
        })
        .collect();
    json!({
        "model": "chronos-2",
        "task": "forecast",
        "target": "sales",
        "features": [],
        "context": context,
        "queries": queries,
        "options": {"series": {
            "step": 1,
            "horizon": 3,
            "quantiles": [0.1, 0.5, 0.9],
            "known_covariates": ["promo"],
        }},
    })
}

fn parse(value: &Value) -> Result<Inference, RequestError> {
    request::parse(
        &serde_json::to_vec(value).unwrap(),
        &Options::request(Surface::Remote),
    )
}

fn refusal(value: &Value) -> (&'static str, String, u16) {
    let error = parse(value).expect_err("the request is refused");
    (error.code, error.message, error.status)
}

fn invalid(message: &str) -> (&'static str, String, u16) {
    ("ValidationError", message.to_owned(), 422)
}

#[test]
fn tabular_requests_parse_into_matrices_and_class_codes() {
    let Inference::Tabular(parsed) = parse(&tabular()).unwrap() else {
        panic!("a classification request is tabular");
    };
    assert_eq!(parsed.model.to_string(), "tabicl-v2-classifier");
    assert_eq!(parsed.context, [1.0, 2.0, 3.0, 4.5]);
    assert_eq!(parsed.queries, [2.0, 3.0]);
    assert_eq!(parsed.targets, Targets::Classes(vec![0, 1]));
    assert_eq!(parsed.classes, [Label::Bool(false), Label::Bool(true)]);
    assert_eq!((parsed.usable_rows(), parsed.supplied_rows), (2, 2));
    assert!(parsed.levels.is_empty());
}

#[test]
fn classes_sort_and_a_boolean_task_always_has_both() {
    let mut value = tabular();
    value["context"][0]["label"] = json!(23);
    value["context"][1]["label"] = json!(-2);
    let Inference::Tabular(parsed) = parse(&value).unwrap() else {
        panic!("tabular");
    };
    assert_eq!(parsed.classes, [Label::Int(-2), Label::Int(23)]);
    assert_eq!(parsed.targets, Targets::Classes(vec![1, 0]));
    value["context"] = json!([{"key": "c", "features": {"x": 1, "z": 2}, "label": true}]);
    let Inference::Tabular(parsed) = parse(&value).unwrap() else {
        panic!("tabular");
    };
    assert_eq!(parsed.classes, [Label::Bool(false), Label::Bool(true)]);
    assert_eq!(parsed.targets, Targets::Classes(vec![1]));
}

#[test]
fn regressions_select_levels_on_the_models_grid() {
    let mut value = tabular();
    value["model"] = json!("tabicl-v2-regressor");
    value["task"] = json!("regression");
    value["context"][0]["label"] = json!(10.0);
    value["context"][1]["label"] = json!(11);
    let Inference::Tabular(parsed) = parse(&value).unwrap() else {
        panic!("tabular");
    };
    assert_eq!(parsed.levels, request::DEFAULT_QUANTILES);
    assert_eq!(parsed.targets, Targets::Values(vec![10.0, 11.0]));
    value["options"] = json!({"quantiles": [0.1234]});
    assert_eq!(
        refusal(&value),
        invalid("quantiles use levels k/1000 for k=1..999")
    );
    for (levels, message) in [
        (json!([]), "invalid quantile levels"),
        (json!([0.5, 0.1]), "quantiles must strictly increase"),
        (json!([0]), "quantiles must be within (0,1)"),
        (json!([true]), "quantile must be between 0 and 1"),
    ] {
        value["options"] = json!({"quantiles": levels});
        assert_eq!(refusal(&value), invalid(message));
    }
}

#[test]
fn requests_are_refused_as_the_models_api_refuses_them() {
    let cases: Vec<(Value, (&str, &str, u16))> = vec![
        (
            json!({"model": "custom.py:run"}),
            ("ModelNotFound", "model is not in the curated catalog", 404),
        ),
        (
            json!({"task": "forecast"}),
            (
                "UnsupportedTask",
                "selected model does not support this task",
                422,
            ),
        ),
        (
            json!({"task": "other"}),
            ("ValidationError", "invalid task", 422),
        ),
        (
            json!({"features": []}),
            (
                "ValidationError",
                "tabular inference requires features",
                422,
            ),
        ),
        (
            json!({"features": "x"}),
            ("ValidationError", "invalid feature list", 422),
        ),
        (
            json!({"features": ["x", "x"]}),
            ("ValidationError", "features must be unique", 422),
        ),
        (
            json!({"features": ["outcome"]}),
            ("ValidationError", "target cannot be a feature", 422),
        ),
        (
            json!({"queries": []}),
            ("ValidationError", "query capacity exceeded", 422),
        ),
        (
            json!({"queries": {}}),
            ("ValidationError", "queries must be a list", 422),
        ),
        (
            json!({"queries": [{"key": "q", "features": {"x": true, "z": 2}}]}),
            (
                "ValidationError",
                "feature must be between -1000000000000.0 and 1000000000000.0",
                422,
            ),
        ),
        (
            json!({"queries": [{"key": "q", "features": {"x": 1}}]}),
            ("ValidationError", "required fields: ['x', 'z']", 422),
        ),
        (
            json!({"queries": [{"key": "q", "features": {"x": 1, "z": 2, "y": 3}}]}),
            ("ValidationError", "unknown fields", 422),
        ),
        (
            json!({"options": {"quantiles": [0.5]}}),
            (
                "ValidationError",
                "classification has no quantile options",
                422,
            ),
        ),
        (
            json!({"target": ""}),
            ("ValidationError", "invalid target", 422),
        ),
        (
            json!({"unexpected": 1}),
            ("ValidationError", "unknown fields", 422),
        ),
        (
            json!({"model": "tabicl-classifier@9"}),
            (
                "CapacityExceeded",
                "serving policy allows at most @8 error-correcting codes",
                422,
            ),
        ),
        (
            json!({"model": "tabpfn-3-classifier"}),
            (
                "UnsupportedModel",
                "tabpfn-3-classifier is local-only: its license does not permit hosted serving",
                422,
            ),
        ),
        (
            json!({"model": "limix-2-classifier"}),
            (
                "UnsupportedModel",
                "limix-2-classifier is local-only: its license does not permit hosted serving",
                422,
            ),
        ),
        (
            json!({"context": []}),
            ("EmptyContext", "no usable labeled rows; no score", 503),
        ),
    ];
    for (change, (code, message, status)) in cases {
        let mut value = tabular();
        for (key, field) in change.as_object().unwrap() {
            value[key] = field.clone();
        }
        assert_eq!(
            refusal(&value),
            (code, message.to_owned(), status),
            "{change}"
        );
    }
    let mut value = tabular();
    value["context"] = json!(
        (0..11)
            .map(|i| json!({"key": i.to_string(), "features": {"x": i, "z": 0}, "label": i}))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        refusal(&value).0,
        "UnsupportedClassCount",
        "ten class slots without @K codes"
    );
    value["model"] = json!("tabicl-classifier@4");
    assert!(parse(&value).is_ok());
}

#[test]
fn payloads_keep_json_bounds_and_refuse_duplicate_fields() {
    let options = Options::request(Surface::Remote);
    let refused = |payload: &[u8]| request::parse(payload, &options).unwrap_err().message;
    assert_eq!(refused(br#"{"a": 1, "a": 2}"#), "duplicate JSON field");
    assert_eq!(refused(b"[1]"), "expected JSON object");
    assert_eq!(refused(b"{"), "invalid JSON body");
    let mut value = tabular();
    value["queries"][0]["features"]["x"] = json!(1_i64 << 53);
    assert_eq!(
        refusal(&value),
        invalid("integer exceeds portable JSON range")
    );
    let mut deep = json!(0);
    for _ in 0..17 {
        deep = json!([deep]);
    }
    let mut value = tabular();
    value["options"] = json!({"quantiles": deep});
    assert_eq!(refusal(&value), invalid("JSON structure too large"));
}

#[test]
fn forecasts_align_related_series_and_their_covariates() {
    let mut value = forecast();
    value["context"][1]["value"] = Value::Null;
    value["context"].as_array_mut().unwrap().remove(6);
    let Inference::Forecast(parsed) = parse(&value).unwrap() else {
        panic!("a forecast request");
    };
    assert_eq!(
        (parsed.step, parsed.horizon, parsed.clock),
        (Num::Int(1), 3, Clock::Periods)
    );
    assert_eq!((parsed.supplied_rows, parsed.usable_observations), (7, 6));
    let query = &parsed.queries[1];
    assert_eq!((query.series.as_str(), query.member), ("b", 1));
    assert_eq!(query.times, [Num::Int(4), Num::Int(5), Num::Int(6)]);
    let group = &query.group;
    assert_eq!((group.rows, group.steps, group.horizon), (4, 4, 3));
    assert_eq!(group.roles, [TARGET, TARGET, PAST, KNOWN]);
    assert_eq!(
        &group.observed[..8],
        [true, false, true, true, true, true, false, true]
    );
    assert_eq!(&group.values[4..8], [30.0, 31.0, 0.0, 33.0]);
    assert_eq!(&group.future[9..], [0.0, 1.0, 0.0]);
    assert_eq!(
        group.future_observed,
        [
            false, false, false, false, false, false, false, false, false, true, true, true
        ]
    );
}

#[test]
fn forecasts_are_refused_as_the_models_api_refuses_them() {
    let cases: Vec<(&[&str], Value, &str)> = vec![
        (
            &["options", "series", "step"],
            json!(1.0),
            "integer periods require an integer step",
        ),
        (
            &["options", "series", "step"],
            json!(0),
            "step must be positive",
        ),
        (
            &["options", "series", "horizon"],
            json!(33),
            "horizon must be 1..32",
        ),
        (
            &["options", "series", "known_covariates"],
            json!(["a"]),
            "target series cannot be covariates",
        ),
        (
            &["context", "0", "time"],
            json!(0.0),
            "cannot mix integer periods and floating UTC seconds",
        ),
        (&["context", "0", "time"], json!(4), "future target leakage"),
        (&["context", "0", "time"], json!(1), "duplicate series time"),
        (
            &["queries", "0", "origin"],
            json!(100_030),
            "series span exceeds context bound",
        ),
        (
            &["queries", "0", "group"],
            json!(["b"]),
            "invalid query group",
        ),
        (
            &["queries", "0", "future_covariates"],
            json!({"promo": [0]}),
            "known-future values must cover the complete horizon",
        ),
    ];
    for (path, change, message) in cases {
        let mut value = forecast();
        let mut cursor = &mut value;
        for key in &path[..path.len() - 1] {
            cursor = match key.parse::<usize>() {
                Ok(index) => &mut cursor[index],
                Err(_) => &mut cursor[*key],
            };
        }
        cursor[path[path.len() - 1]] = change;
        assert_eq!(refusal(&value), invalid(message), "{path:?}");
    }
    let mut value = forecast();
    value["options"]["series"]["quantiles"] = json!([0.123]);
    assert_eq!(refusal(&value).0, "UnsupportedQuantiles");
    let mut value = forecast();
    value["context"][0]["covariates"]["promo"] = json!(99);
    assert_eq!(refusal(&value).0, "UnsupportedCovariates");
}

#[test]
fn package_contexts_take_loaded_levels_and_skip_request_bounds() {
    let mut value = forecast();
    value["model"] = json!("timesfm-3");
    value["options"]["series"]["quantiles"] = json!([0.25]);
    let payload = serde_json::to_vec(&value).unwrap();
    let refused = request::parse(&payload, &Options::request(Surface::Remote)).unwrap_err();
    assert_eq!(refused.code, "UnsupportedModel");
    let levels = [0.1, 0.5, 0.9];
    let options = Options {
        surface: Surface::Local,
        limits: Limits::Package,
        levels: Some(&levels),
        admission: Admission::Catalog,
    };
    assert_eq!(
        request::parse(&payload, &options).unwrap_err().code,
        "UnsupportedQuantiles"
    );
    let options = Options {
        levels: Some(&[0.25]),
        ..options
    };
    assert!(request::parse(&payload, &options).is_ok());
}

#[test]
fn fingerprints_follow_the_context_and_model_only() {
    let print = |value: &Value| parse(value).unwrap().fingerprint();
    let base = print(&tabular());
    let mut queries = tabular();
    queries["queries"][0]["features"]["x"] = json!(9);
    let mut keys = tabular();
    keys["context"][0]["key"] = json!("renamed");
    let mut context = tabular();
    context["context"][0]["features"]["x"] = json!(-1);
    let mut model = tabular();
    model["model"] = json!("tabicl-v2-classifier@2");
    assert_eq!(base, print(&queries));
    assert_eq!(base, print(&keys));
    assert_ne!(base, print(&context));
    assert_ne!(base, print(&model));
    let mut horizon = forecast();
    horizon["options"]["series"]["horizon"] = json!(2);
    horizon["queries"][0]["future_covariates"]["promo"] = json!([0, 1]);
    horizon["queries"][1]["future_covariates"]["promo"] = json!([0, 1]);
    assert_eq!(print(&forecast()), print(&horizon));
    let groups = |value: &Value| match parse(value).unwrap() {
        Inference::Forecast(parsed) => parsed
            .queries
            .iter()
            .map(|query| query.fingerprint)
            .collect::<Vec<_>>(),
        Inference::Tabular(_) => unreachable!("a forecast"),
    };
    let base_groups = groups(&forecast());
    assert_eq!(
        base_groups[0], base_groups[1],
        "both queries forecast one group"
    );
    let mut keyed = forecast();
    keyed["queries"][0]["key"] = json!("renamed");
    assert_eq!(groups(&keyed), base_groups);
    let mut future = forecast();
    future["queries"][0]["future_covariates"]["promo"] = json!([1, 1, 1]);
    let changed = groups(&future);
    assert_ne!(changed[0], base_groups[0]);
    assert_eq!(changed[1], base_groups[1]);
    let mut ordered = forecast();
    ordered["queries"][0]["group"] = json!(["b", "a"]);
    assert_ne!(groups(&ordered)[0], base_groups[0]);
    assert_eq!(request::hex(&base).len(), 64);
}

#[test]
fn references_resolve_with_their_surfaces_gates() {
    let model = catalog::resolve(
        Some("general-forecast/tabicl-v2-classifier@04*2"),
        Surface::Remote,
    )
    .unwrap();
    assert_eq!(model.to_string(), "tabicl-v2-classifier@4*2");
    assert_eq!(model.capacity(), catalog::catalog().max_classes);
    let refused = |spec: &str, surface| catalog::resolve(Some(spec), surface).unwrap_err();
    assert_eq!(
        refused("tabpfn-3-classifier@4", Surface::SelfHosted).code,
        "UnsupportedCapability"
    );
    assert_eq!(
        refused("limix-2-classifier", Surface::SelfHosted).message,
        "limix-2-classifier has no pinned files"
    );
    assert!(catalog::resolve(Some("tabpfn-3-classifier"), Surface::SelfHosted).is_ok());
    assert!(catalog::resolve(Some("timesfm-3"), Surface::Local).is_ok());
    assert_eq!(
        refused("tabicl-classifier@4*8", Surface::Local).code,
        "CapacityExceeded"
    );
    assert_eq!(
        catalog::resolve(None, Surface::Remote).unwrap_err().message,
        "invalid model"
    );
}

#[test]
fn quoted_input_is_spelled_as_python_repr_spells_it() {
    assert_eq!(python_repr("plain"), "'plain'");
    assert_eq!(python_repr("it's"), "\"it's\"");
    assert_eq!(python_repr("both '\""), "'both \\'\"'");
    assert_eq!(
        python_repr("a\nb\\\u{0}\u{a0}é\u{1f600}"),
        "'a\\nb\\\\\\x00\\xa0é\u{1f600}'"
    );
}

const CODE: &str = "sha256:code";

#[test]
fn fingerprints_are_pinned_so_changing_them_is_deliberate() {
    let tabular = request::hex(&parse(&tabular()).unwrap().fingerprint());
    let forecast = request::hex(&parse(&forecast()).unwrap().fingerprint());
    assert_eq!(
        tabular,
        "f1da5df46a674260e0d1aba77d2185e7f9e8c90bd71b1972041cf4aa22e817f3"
    );
    assert_eq!(
        forecast,
        "e67deb0b628634811d1c1d7cfa643e33ac8f4b61c40e2eecf1d2599d9318a0fb"
    );
}

#[test]
fn only_a_local_caller_admits_trusted_forecasters() {
    let spec = "tests.models.chronos2.tiny:forecaster";
    let mut value = forecast();
    value["model"] = json!(spec);
    let levels: Vec<f64> = value["options"]["series"]["quantiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|level| level.as_f64().unwrap())
        .collect();
    let payload = serde_json::to_vec(&value).unwrap();
    let local = Options {
        surface: Surface::Local,
        limits: Limits::Package,
        levels: Some(&levels),
        admission: Admission::Catalog,
    };
    for surface in [Surface::Remote, Surface::SelfHosted] {
        let refused = |admission| {
            let options = Options {
                surface,
                admission,
                ..local
            };
            request::parse(&payload, &options).unwrap_err().code
        };
        assert_eq!(refused(Admission::Catalog), "ModelNotFound");
        assert_eq!(
            refused(Admission::Trusted { digest: CODE }),
            "UntrustedModel"
        );
        let served = request::parse(&payload, &Options::request(surface));
        assert_eq!(served.unwrap_err().code, "ModelNotFound");
    }
    assert_eq!(
        request::parse(&payload, &local).unwrap_err().code,
        "ModelNotFound"
    );
    let trusted = Options {
        admission: Admission::Trusted { digest: CODE },
        ..local
    };
    let Inference::Forecast(parsed) = request::parse(&payload, &trusted).unwrap() else {
        panic!("expected a forecast");
    };
    assert_eq!(
        parsed.model,
        Source::Local {
            spec: spec.into(),
            digest: CODE.into(),
        }
    );
    let edited = Options {
        admission: Admission::Trusted {
            digest: "sha256:edited",
        },
        ..trusted
    };
    let edited = request::parse(&payload, &edited).unwrap();
    assert_ne!(edited.fingerprint(), parsed.fingerprint);
    let undigested = Options {
        admission: Admission::Trusted { digest: "" },
        ..trusted
    };
    assert_eq!(
        request::parse(&payload, &undigested).unwrap_err().code,
        "ValidationError"
    );
    let unleveled = Options {
        levels: None,
        ..trusted
    };
    assert_eq!(
        request::parse(&payload, &unleveled).unwrap_err().code,
        "ValidationError"
    );
    let mut value = tabular();
    value["model"] = json!("tests.models.tabicl.tiny:classifier");
    let payload = serde_json::to_vec(&value).unwrap();
    assert_eq!(
        request::parse(&payload, &trusted).unwrap_err().code,
        "UnsupportedTask"
    );
}
