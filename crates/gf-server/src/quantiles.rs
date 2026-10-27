// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Quantile levels a caller selects, and the predictions at them, which
//! Flight calls and Models API requests compute alike.

use gf_core::request::MAX_LEVELS;

/// The levels a comma-separated list names: at most `MAX_LEVELS` of them,
/// each strictly between 0 and 1 and above the one before, as the Models
/// API takes them too.
///
/// # Errors
///
/// What the list breaks.
pub fn parse_levels(text: &str) -> Result<Vec<f64>, String> {
    let levels = text
        .split(',')
        .map(|level| level.trim().parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()
        .map_err(|_| "quantile levels are numbers".to_string())?;
    if levels.is_empty() || levels.len() > MAX_LEVELS {
        return Err(format!("name 1 to {MAX_LEVELS} quantile levels"));
    }
    if levels.iter().any(|level| !(*level > 0.0 && *level < 1.0)) {
        return Err("quantile levels are strictly between 0 and 1".to_string());
    }
    if levels.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("quantile levels strictly increase".to_string());
    }
    Ok(levels)
}

/// Predictions at `levels`, a column per level, each linearly between the
/// two of the table's `known` levels around it; one beyond the outermost
/// known level takes that level. Clients compute the same. `values` are
/// row-major with `known.len()` columns; None if they are not.
#[must_use]
pub fn quantile_values(values: &[f32], known: &[f64], levels: &[f64]) -> Option<Vec<Vec<f32>>> {
    let columns = known.len();
    if columns == 0 || !values.len().is_multiple_of(columns) {
        return None;
    }
    let rows = values.len() / columns;
    Some(
        levels
            .iter()
            .map(|&level| {
                let upper = known.partition_point(|&at| at < level).min(columns - 1);
                let lower = upper.saturating_sub(1);
                let span = known[upper] - known[lower];
                let weight = if span > 0.0 {
                    ((level - known[lower]) / span).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (0..rows)
                    .map(|row| {
                        let at = |column: usize| f64::from(values[row * columns + column]);
                        (at(lower) * (1.0 - weight) + at(upper) * weight) as f32
                    })
                    .collect()
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::{MAX_LEVELS, parse_levels};

    #[test]
    fn levels_are_bounded_in_count_and_strictly_increase() {
        assert_eq!(parse_levels("0.1, 0.5,0.9"), Ok(vec![0.1, 0.5, 0.9]));
        let most: Vec<String> = (1..=MAX_LEVELS)
            .map(|k| format!("{}", k as f64 / 100.0))
            .collect();
        assert_eq!(
            parse_levels(&most.join(",")).map(|levels| levels.len()),
            Ok(MAX_LEVELS)
        );
        let many: Vec<String> = (1..=4000)
            .map(|k| format!("{}", k as f64 / 4001.0))
            .collect();
        assert!(
            parse_levels(&many.join(",")).is_err(),
            "past the most levels"
        );
        for refused in ["", "0.5,0.5", "0.9,0.1", "0", "1", "x", "0.5,", "nan"] {
            assert!(parse_levels(refused).is_err(), "{refused:?}");
        }
    }
}
