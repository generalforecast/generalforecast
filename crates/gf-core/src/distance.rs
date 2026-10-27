// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Squared distances between points, the one measure retrieval, pages,
//! cores and regions place rows by.

/// The squared distance between two points, in float64 whatever their
/// coordinates' type, summed in order.
pub(crate) fn squared<L, R>(left: &[L], right: &[R]) -> f64
where
    L: Copy + Into<f64>,
    R: Copy + Into<f64>,
{
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (left.into() - right.into()).powi(2))
        .sum()
}

/// The squared distance between two float32 points, eight lanes at a time:
/// faster than `squared`, for choosing among near candidates, where the last
/// bits of rounding matter little.
pub(crate) fn squared32(point: &[f32], vector: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut pairs = point.chunks_exact(8).zip(vector.chunks_exact(8));
    for (left, right) in &mut pairs {
        for lane in 0..8 {
            let difference = left[lane] - right[lane];
            lanes[lane] += difference * difference;
        }
    }
    let whole = point.len() / 8 * 8;
    let tail: f32 = point[whole..]
        .iter()
        .zip(&vector[whole..])
        .map(|(left, right)| (left - right) * (left - right))
        .sum();
    lanes.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::{squared, squared32};

    #[test]
    fn both_measures_agree_on_whole_numbers() {
        let left: Vec<f32> = (0..11).map(|value| value as f32).collect();
        let right: Vec<f32> = (0..11).map(|value| (2 * value) as f32).collect();
        let expected: f64 = (0..11).map(|value| f64::from(value * value)).sum();
        assert_eq!(squared(&left, &right), expected);
        assert_eq!(f64::from(squared32(&left, &right)), expected);
        let center: Vec<f64> = left.iter().map(|&value| f64::from(value)).collect();
        assert_eq!(squared(&center, &right), expected);
    }
}
