// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `as_of` on the wire: the one place gf-server reads and writes it.
//!
//! A response carries `as_of` as an RFC 3339 UTC string, such as
//! `2026-10-27T08:00:00Z` (six fractional digits when it has a fraction), or
//! null. A request may give an RFC 3339 date-time, a naive one read in UTC,
//! or a number of seconds since 1970, at least 0; anything else is refused as
//! a `ValidationError` naming `as_of`. Instants are kept to the microsecond,
//! and digits past it are dropped, never rounded up, so an `as_of` never
//! reads a write after the time it names. Python's `generalforecast.as_of`
//! reads and writes it by the same rules; the fixture
//! `tests/fixtures/as_of.json` pins both.

use gf_core::civil::{Grammar, iso_micros, rfc3339};
use serde_json::Value;

use crate::http::ApiError;

/// The request and response field this module reads and writes.
pub const FIELD: &str = "as_of";

/// An instant, to the microsecond, in UTC, at or after 1970.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AsOf {
    micros: i64,
}

/// The latest instant a four-digit year writes: 9999-12-31T23:59:59.999999
/// UTC, in microseconds since 1970.
const LATEST: i64 = 253_402_300_799_999_999;

impl AsOf {
    /// The instant `micros` microseconds after 1970 began, if it is at or
    /// after 1970 and a four-digit year writes it.
    #[must_use]
    pub fn from_micros(micros: i64) -> Option<Self> {
        (0..=LATEST).contains(&micros).then_some(Self { micros })
    }

    /// Microseconds since 1970 began.
    #[must_use]
    pub fn micros(self) -> i64 {
        self.micros
    }

    /// The instant a request gives: null or absent for none.
    ///
    /// # Errors
    ///
    /// A `ValidationError` naming `as_of` for anything but an RFC 3339
    /// date-time or a number of seconds since 1970, at least 0.
    pub fn parse(value: Option<&Value>) -> Result<Option<Self>, ApiError> {
        let refused = || {
            ApiError::validation(
                "as_of is an RFC 3339 time, such as 2026-10-27T08:00:00Z, or seconds since 1970",
            )
            .param(FIELD)
        };
        match value {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(seconds)) => decimal_micros(&seconds.to_string())
                .and_then(Self::from_micros)
                .map(Some)
                .ok_or_else(refused),
            Some(Value::String(text)) => iso_micros(text, Grammar::Instant)
                .and_then(|(micros, _)| Self::from_micros(micros))
                .map(Some)
                .ok_or_else(refused),
            Some(_) => Err(refused()),
        }
    }

    /// The instant an answer from elsewhere gives, as `parse` reads a
    /// request's, or None where it gives none it can read.
    #[must_use]
    pub fn read(value: Option<&Value>) -> Option<Self> {
        Self::parse(value).ok().flatten()
    }

    /// The instant as RFC 3339 in UTC: whole seconds when it has no
    /// fraction, else six fractional digits.
    #[must_use]
    pub fn rfc3339(self) -> String {
        rfc3339(self.micros)
    }

    /// The instant as the wire carries it, or null.
    #[must_use]
    pub fn json(value: Option<Self>) -> Value {
        value.map_or(Value::Null, |instant| Value::String(instant.rfc3339()))
    }
}

/// A JSON number of seconds, as its shortest decimal text spells it, in
/// whole microseconds: digits past the microsecond are dropped, so the
/// instant is never later than the number says. None for a negative or
/// unreadable number.
fn decimal_micros(text: &str) -> Option<i64> {
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()?),
        None => (text, 0),
    };
    if mantissa.starts_with('-') || mantissa.is_empty() {
        return None;
    }
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if !(whole
        .bytes()
        .chain(fraction.bytes())
        .all(|byte| byte.is_ascii_digit()))
    {
        return None;
    }
    // The digits, and where the point falls among them once scaled to
    // microseconds.
    let digits = format!("{whole}{fraction}");
    let point = i64::try_from(whole.len()).ok()? + i64::from(exponent) + 6;
    if point <= 0 {
        return Some(0);
    }
    let point = usize::try_from(point).ok()?;
    let kept: String = if point <= digits.len() {
        digits[..point].to_string()
    } else {
        format!("{digits}{}", "0".repeat(point - digits.len()))
    };
    let kept = kept.trim_start_matches('0');
    if kept.len() > 18 {
        return None;
    }
    if kept.is_empty() {
        return Some(0);
    }
    kept.parse().ok()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::AsOf;

    /// The cases Python's `generalforecast.as_of` reads the same way.
    fn fixture() -> Vec<(Value, Option<i64>)> {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/as_of.json")).unwrap();
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| (case[0].clone(), case[1].as_i64()))
            .collect()
    }

    #[test]
    fn requests_read_as_the_shared_fixture_says() {
        for (value, micros) in fixture() {
            let read = AsOf::parse(Some(&value)).ok().flatten().map(AsOf::micros);
            assert_eq!(read, micros, "{value}");
        }
    }

    #[test]
    fn replies_write_rfc3339_utc_to_the_microsecond() {
        let written = |micros| AsOf::from_micros(micros).unwrap().rfc3339();
        assert_eq!(written(1_793_088_000_000_000), "2026-10-27T08:00:00Z");
        assert_eq!(
            written(1_793_088_000_866_500),
            "2026-10-27T08:00:00.866500Z"
        );
        assert_eq!(
            written(1_793_088_000_000_001),
            "2026-10-27T08:00:00.000001Z"
        );
        for (value, micros) in fixture() {
            let Some(micros) = micros else { continue };
            let again = json!(written(micros));
            assert_eq!(
                AsOf::read(Some(&again)).map(AsOf::micros),
                Some(micros),
                "{value}"
            );
        }
    }

    #[test]
    fn anything_else_is_refused_naming_as_of() {
        for value in [json!("yesterday"), json!(true), json!([1]), json!({"t": 1})] {
            let refused = AsOf::parse(Some(&value)).unwrap_err();
            assert_eq!(
                (refused.status, refused.code.as_str()),
                (422, "ValidationError"),
                "{value}"
            );
            assert_eq!(
                refused
                    .detail
                    .as_ref()
                    .and_then(|detail| detail.param.as_deref()),
                Some("as_of"),
                "{value}"
            );
        }
        assert_eq!(AsOf::parse(None).unwrap(), None);
        assert_eq!(AsOf::parse(Some(&Value::Null)).unwrap(), None);
    }
}
