// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `as_of` on the wire: the one place gf-server reads and writes it.
//!
//! A response carries `as_of` as an RFC 3339 UTC string, such as
//! `2026-10-27T08:00:00Z`, or null. A request may give an RFC 3339 string,
//! a naive one read in UTC, or a number of seconds since 1970; anything else
//! is refused as a `ValidationError` naming `as_of`. gf's Python surfaces
//! and the control plane read and write it the same way.

use serde_json::Value;

use crate::http::ApiError;

/// The request and response field this module reads and writes.
pub const FIELD: &str = "as_of";

/// An instant, to the millisecond, in UTC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AsOf {
    millis: i64,
}

/// The earliest and latest instants a four-digit year writes: 0001-01-01
/// and 9999-12-31T23:59:59.999 UTC, in milliseconds since 1970.
const EARLIEST: i64 = -62_135_596_800_000;
const LATEST: i64 = 253_402_300_799_999;

impl AsOf {
    /// The instant `millis` milliseconds after 1970 began, if a four-digit
    /// year writes it.
    #[must_use]
    pub fn from_millis(millis: i64) -> Option<Self> {
        (EARLIEST..=LATEST)
            .contains(&millis)
            .then_some(Self { millis })
    }

    /// Milliseconds since 1970 began.
    #[must_use]
    pub fn millis(self) -> i64 {
        self.millis
    }

    /// The instant a request gives: null or absent for none.
    ///
    /// # Errors
    ///
    /// A `ValidationError` naming `as_of` for anything but an RFC 3339
    /// string or a number of seconds since 1970.
    pub fn parse(value: Option<&Value>) -> Result<Option<Self>, ApiError> {
        let refused = || {
            ApiError::validation(
                "as_of is an RFC 3339 time, such as 2026-10-27T08:00:00Z, or seconds since 1970",
            )
            .param(FIELD)
        };
        match value {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(seconds)) => seconds
                .as_f64()
                .filter(|seconds| seconds.is_finite())
                .and_then(Self::from_seconds)
                .map(Some)
                .ok_or_else(refused),
            Some(Value::String(text)) => Self::from_text(text).map(Some).ok_or_else(refused),
            Some(_) => Err(refused()),
        }
    }

    /// The instant an answer from elsewhere gives, as `parse` reads a
    /// request's, or None where it gives none it can read.
    #[must_use]
    pub fn read(value: Option<&Value>) -> Option<Self> {
        Self::parse(value).ok().flatten()
    }

    fn from_seconds(seconds: f64) -> Option<Self> {
        let millis = (seconds * 1000.0).round();
        (millis >= EARLIEST as f64 && millis <= LATEST as f64)
            .then(|| Self::from_millis(millis as i64))
            .flatten()
    }

    fn from_text(text: &str) -> Option<Self> {
        if !text.contains(['T', 't', ' ']) {
            return None;
        }
        let nanos = arrow_cast::parse::string_to_timestamp_nanos(text.trim()).ok()?;
        Self::from_millis(nanos.div_euclid(1_000_000))
    }

    /// The instant as RFC 3339 in UTC: whole seconds when it has no
    /// milliseconds, else to the millisecond.
    #[must_use]
    pub fn rfc3339(self) -> String {
        let seconds = self.millis.div_euclid(1000);
        let millis = self.millis.rem_euclid(1000);
        let days = seconds.div_euclid(86_400);
        let of_day = seconds.rem_euclid(86_400);
        let (year, month, day) = civil(days);
        let (hour, minute, second) = (of_day / 3600, of_day % 3600 / 60, of_day % 60);
        let fraction = if millis == 0 {
            String::new()
        } else {
            format!(".{millis:03}")
        };
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{fraction}Z")
    }

    /// The instant as the wire carries it, or null.
    #[must_use]
    pub fn json(value: Option<Self>) -> Value {
        value.map_or(Value::Null, |instant| Value::String(instant.rfc3339()))
    }
}

/// The proleptic Gregorian date `days` days after 1970-01-01.
fn civil(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let of_era = shifted.rem_euclid(146_097);
    let year_of_era = (of_era - of_era / 1460 + of_era / 36_524 - of_era / 146_096) / 365;
    let of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * of_year + 2) / 153;
    let day = of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::AsOf;

    fn parsed(value: serde_json::Value) -> Option<String> {
        AsOf::parse(Some(&value)).unwrap().map(AsOf::rfc3339)
    }

    #[test]
    fn requests_give_rfc3339_naive_utc_or_epoch_seconds() {
        let at = Some("2026-10-27T08:00:00Z".to_string());
        assert_eq!(parsed(json!("2026-10-27T08:00:00Z")), at);
        assert_eq!(parsed(json!("2026-10-27T10:00:00+02:00")), at);
        assert_eq!(parsed(json!("2026-10-27T08:00:00")), at, "naive is UTC");
        assert_eq!(parsed(json!(1_793_088_000)), at);
        assert_eq!(
            parsed(json!(1_793_088_000.25)),
            Some("2026-10-27T08:00:00.250Z".into())
        );
        assert_eq!(parsed(json!(0)), Some("1970-01-01T00:00:00Z".into()));
        assert_eq!(parsed(json!(-86_400)), Some("1969-12-31T00:00:00Z".into()));
        assert_eq!(
            parsed(json!("2024-02-29T23:59:59.5Z")),
            Some("2024-02-29T23:59:59.500Z".into())
        );
        assert_eq!(parsed(json!(null)), None);
        assert_eq!(AsOf::parse(None).unwrap(), None);
    }

    #[test]
    fn anything_else_is_refused_naming_as_of() {
        for value in [
            json!("yesterday"),
            json!("2026-10-27"),
            json!(""),
            json!(true),
            json!([1]),
            json!({"t": 1}),
            json!(1e300),
        ] {
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
    }
}
