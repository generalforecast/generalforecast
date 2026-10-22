// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

use gf_core::categories::{CategoryEncoding, Lookup, ONE_HOT_CATEGORIES};
use gf_core::node::{Targets, Task};

fn vector(lookup: &Lookup, value: f32) -> Vec<f32> {
    let mut vector = Vec::new();
    lookup.extend(value, &mut vector);
    vector
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn one_hot_names_the_most_frequent_categories_and_pools_the_rest() {
    // Code 16 is the rarest and code 15 the next, so both fall to "other".
    let values: Vec<f32> = (0..17u16)
        .flat_map(|code| std::iter::repeat_n(f32::from(code), 20 - usize::from(code)))
        .collect();
    let targets = Targets::Classes(vec![0; values.len()]);

    let lookup = Lookup::fit(
        CategoryEncoding::OneHot,
        &values,
        &targets,
        Task::Classes(2),
    );

    assert_eq!(lookup.codes.len(), ONE_HOT_CATEGORIES);
    assert_eq!(vector(&lookup, 15.0), vector(&lookup, 16.0));
    assert_eq!(vector(&lookup, 15.0), vector(&lookup, 99.0));
    assert!((distance(&vector(&lookup, 0.0), &vector(&lookup, 1.0)) - 1.0).abs() < 1e-6);
    assert_eq!(vector(&lookup, f32::NAN), vec![0.0; ONE_HOT_CATEGORIES + 1]);
    // A layout's codes and vectors give back the same lookup.
    let kept = Lookup::new(
        CategoryEncoding::OneHot,
        lookup.codes.clone(),
        lookup.vectors.clone(),
    );
    assert_eq!(kept, lookup);
}

#[test]
fn target_encoding_places_categories_by_their_rows_targets() {
    let values = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let classes = Targets::Classes(vec![0, 0, 0, 1, 1, 1]);

    let lookup = Lookup::fit(
        CategoryEncoding::Target,
        &values,
        &classes,
        Task::Classes(2),
    );

    // Category 0's 2 rows, all class 0, averaged with 10 of the table's
    // even rates.
    let smoothed = vector(&lookup, 0.0);
    assert!((smoothed[0] - 7.0 / 12.0).abs() < 1e-6 && (smoothed[1] - 5.0 / 12.0).abs() < 1e-6);
    assert_eq!(vector(&lookup, 1.0), [0.5, 0.5]);
    // Unseen and missing categories take the table's rates.
    assert_eq!(vector(&lookup, 7.0), [0.5, 0.5]);
    assert_eq!(vector(&lookup, f32::NAN), [0.5, 0.5]);

    let values = Targets::Values(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let lookup = Lookup::fit(
        CategoryEncoding::Target,
        &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        &values,
        Task::Values,
    );
    // Means 1, 2 and 3 about the table's 2, in its deviation of sqrt(2/3),
    // each category's 2 rows averaged with 10 at the table's mean.
    let deviation = (2.0f32 / 3.0).sqrt();
    assert!((vector(&lookup, 0.0)[0] + 2.0 / 12.0 / deviation).abs() < 1e-6);
    assert_eq!(vector(&lookup, 1.0), [0.0]);
    assert_eq!(vector(&lookup, 9.0), [0.0]);
}
