// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A locked predictor's JSON rows, read as the Arrow columns its schema declares.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::TimestampMillisecondType;
use arrow_schema::{DataType, TimeUnit};
use gf_core::predictor::SchemaColumn;
use gf_server::predictors::records;
use serde_json::json;

fn column(name: &str, physical: &str) -> SchemaColumn {
    SchemaColumn {
        name: name.to_string(),
        physical: physical.to_string(),
        nullable: true,
        role: "input".to_string(),
        semantic: None,
    }
}

#[test]
fn a_timestamp_takes_iso_strings_and_the_microseconds_its_type_counts() {
    let rows = [
        json!({"at": "2023-11-14T22:13:20Z"}),
        json!({"at": 1_700_000_000_000_000_i64}),
        json!({}),
    ];
    let batch = records(&[column("at", "timestamp")], &rows).unwrap();
    // Read back as the binder reads a datetime column's text.
    let read = arrow_cast::cast(
        batch.column(0),
        &DataType::Timestamp(TimeUnit::Millisecond, None),
    )
    .unwrap();
    let read = read.as_primitive::<TimestampMillisecondType>();
    assert_eq!(read.value(0), 1_700_000_000_000);
    assert_eq!(read.value(1), 1_700_000_000_000);
    assert!(read.is_null(2));
}

#[test]
fn a_value_its_column_does_not_take_is_refused_by_column_and_both_types() {
    for (physical, value, held) in [
        ("int64", json!("12"), "a string"),
        ("int64", json!(1.5), "a number with a fraction"),
        ("timestamp", json!(1.5), "a number with a fraction"),
        ("timestamp", json!(true), "a boolean"),
        ("float64", json!([1]), "a list"),
        ("boolean", json!(1), "a whole number"),
        ("string", json!({"a": 1}), "an object"),
    ] {
        let refused = records(&[column("when", physical)], &[json!({"when": value})]).unwrap_err();
        assert_eq!(refused.status, 422);
        assert!(
            refused
                .message
                .starts_with(&format!("column \"when\" is declared {physical}"))
                && refused
                    .message
                    .ends_with(&format!("but a row holds {held}")),
            "{}",
            refused.message
        );
    }
}

#[test]
fn a_context_the_lock_holds_no_schema_for_is_typed_by_its_kinds_and_every_value() {
    use gf_server::predictors::inferred;
    let rows = [
        json!({"a": 1, "b": 2, "c": "2026-10-27T08:00:00Z", "d": 3, "y": true}),
        json!({"a": 1.5, "b": 4, "c": "2026-10-28T08:00:00Z", "d": 4, "y": false}),
    ];
    let features = ["a", "b", "c", "d"].map(str::to_string);
    let types = [("c", "timestamp"), ("d", "category")]
        .into_iter()
        .map(|(name, kind)| (name.to_string(), kind.to_string()))
        .collect();
    let schema = inferred(&features, "y", &rows, &types, &features).unwrap();
    let physical: Vec<(&str, &str, Option<&str>)> = schema
        .columns
        .iter()
        .map(|column| {
            (
                column.name.as_str(),
                column.physical.as_str(),
                column.semantic.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        physical,
        [
            // A whole first value of a column of fractions is a float.
            ("a", "float64", None),
            ("b", "int64", None),
            // A time the lock calls a timestamp is one, not text.
            ("c", "timestamp", None),
            // Whole numbers the lock calls categories are read as them.
            ("d", "int64", Some("category")),
            ("y", "boolean", None),
        ]
    );
    // Its rows read as those types, whole numbers widening to floats.
    let batch = records(&schema.columns, &rows).unwrap();
    assert_eq!(batch.column(0).data_type(), &DataType::Float64);
}

#[test]
fn a_kind_a_columns_values_cannot_hold_is_refused_naming_the_types() {
    use gf_server::predictors::inferred;
    let rows = [json!({"a": "high", "y": true})];
    let types = [("a".to_string(), "float".to_string())]
        .into_iter()
        .collect();
    let inputs = ["a".to_string()];
    let refused = inferred(&inputs, "y", &rows, &types, &inputs).unwrap_err();
    assert_eq!(refused.status, 422);
    assert_eq!(
        refused
            .detail
            .as_ref()
            .and_then(|detail| detail.param.as_deref()),
        Some("inputs.types")
    );
}
