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
