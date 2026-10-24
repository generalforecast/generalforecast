// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The predictor binder: a lock's schema and rows as the numbers the engine reads.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, TimestampMillisecondArray};
use gf_core::catalog::Task;
use gf_core::node::Targets;
use gf_core::predictor::{self, Schema};
use gf_core::request::Label;
use serde_json::json;

fn schema() -> Schema {
    serde_json::from_value(json!({
        "columns": [
            {"name": "x", "type": "float64", "nullable": true, "role": "input", "semantic": null},
            {"name": "color", "type": "string", "nullable": true, "role": "input", "semantic": null},
            {"name": "at", "type": "timestamp", "nullable": false, "role": "input", "semantic": null},
            {"name": "id", "type": "string", "nullable": false, "role": "id", "semantic": null},
            {"name": "y", "type": "string", "nullable": true, "role": "target", "semantic": null},
        ],
        "rows": 3,
        "classes": ["a", "b"],
        "frequency": null,
    }))
    .unwrap()
}

fn rows(x: Vec<f64>, color: Vec<Option<&str>>, labels: Vec<Option<&str>>) -> RecordBatch {
    let count = x.len();
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("x", Arc::new(Float64Array::from(x))),
        ("color", Arc::new(StringArray::from(color))),
        (
            "at",
            Arc::new(TimestampMillisecondArray::from(vec![86_400_000; count])),
        ),
        ("id", Arc::new(StringArray::from(vec!["i"; count]))),
        ("y", Arc::new(StringArray::from(labels))),
    ];
    RecordBatch::try_from_iter(columns).unwrap()
}

#[test]
fn contexts_keep_labeled_rows_and_code_words_as_they_first_appear() {
    let context = rows(
        vec![1.0, 2.0, 3.0],
        vec![Some("red"), Some("blue"), Some("red")],
        vec![Some("b"), None, Some("a")],
    );
    let bound = predictor::bind(&schema(), Task::Classification, "y", &context).unwrap();
    assert_eq!(bound.rows, 2);
    assert_eq!(bound.encoder.width(), 7);
    assert_eq!(
        bound.classes,
        [Label::Text("a".into()), Label::Text("b".into())]
    );
    assert_eq!(bound.targets, Targets::Classes(vec![1, 0]));
    assert_eq!(bound.features[..2], [1.0, 0.0]);
    assert_eq!(bound.features[7..9], [3.0, 0.0]);
    assert_eq!(bound.features[2..7], [1.0, 0.0, 4.0, 2.0, 1.0]);
    let vocabularies: Vec<Vec<String>> = bound
        .encoder
        .vocabularies()
        .map(<[String]>::to_vec)
        .collect();
    assert_eq!(
        vocabularies[1],
        ["red"],
        "the unlabeled row's word is never coded"
    );
    let queries = bound
        .encoder
        .encode(&rows(vec![5.0], vec![Some("blue")], vec![None]))
        .unwrap();
    assert_eq!(queries[0], 5.0);
    assert!(
        queries[1].is_nan(),
        "a word the context never held is missing"
    );
}

#[test]
fn contexts_are_refused_as_packages_refuse_them() {
    let one = rows(vec![1.0], vec![None], vec![Some("a")]);
    assert_eq!(
        predictor::bind(&schema(), Task::Classification, "y", &one)
            .unwrap_err()
            .message,
        "selected context needs at least two classes"
    );
    let none = rows(vec![1.0], vec![None], vec![None]);
    assert_eq!(
        predictor::bind(&schema(), Task::Classification, "y", &none)
            .unwrap_err()
            .code,
        "EmptyContext"
    );
    let bound = predictor::bind(
        &schema(),
        Task::Classification,
        "y",
        &rows(vec![1.0, 2.0], vec![None, None], vec![Some("a"), Some("b")]),
    )
    .unwrap();
    let missing = RecordBatch::try_from_iter(vec![(
        "x",
        Arc::new(Float64Array::from(vec![1.0])) as ArrayRef,
    )])
    .unwrap();
    assert_eq!(
        bound.encoder.encode(&missing).unwrap_err().code,
        "InvalidPredictor"
    );
}
