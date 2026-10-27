// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Civil dates and ISO 8601 times on the UTC axis: the one reader every
//! Rust path uses.
//!
//! Dates are the proleptic Gregorian calendar's (Howard Hinnant's
//! `civil_from_days` and `days_from_civil`). Times are read to the
//! microsecond, by one of two grammars:
//!
//! - [`Grammar::Value`], a context or query cell, as Python's
//!   `runtime.context._timestamps` reads it: a date `YYYY-MM-DD`, then
//!   optionally `T` or a space and `HH`, `HH:MM`, `HH:MM:SS` or
//!   `HH:MM:SS.ffffff` (one to six digits), then after a time optionally
//!   `Z`, `+HH`, `+HHMM` or `+HH:MM` (or `-`). The fixture
//!   `crates/gf-server/tests/fixtures/iso_times.json` pins it.
//! - [`Grammar::Instant`], an `as_of`, as `generalforecast.as_of` reads it:
//!   an RFC 3339 date-time, `YYYY-MM-DD`, `T`, `t` or a space,
//!   `HH:MM:SS`, optionally `.` and one to nine digits (truncated to the
//!   microsecond, never rounded up), then optionally `Z`, `z` or `+HH:MM`
//!   (or `-`). A date alone, an hour alone and ISO weeks are refused. The
//!   fixture `crates/gf-server/tests/fixtures/as_of.json` pins it.
//!
//! Text without a zone is UTC in both.

/// Microseconds in a day.
pub const MICROS_PER_DAY: i64 = 86_400_000_000;

/// How strictly [`iso_micros`] reads text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grammar {
    /// A cell of a context or query column.
    Value,
    /// A request's `as_of`: an RFC 3339 date-time.
    Instant,
}

/// The civil date `(year, month, day)` of a day counted from 1970-01-01.
#[must_use]
pub fn civil(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let of_era = shifted.rem_euclid(146_097);
    let year_of_era = (of_era - of_era / 1460 + of_era / 36_524 - of_era / 146_096) / 365;
    let of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * of_year + 2) / 153;
    let day = of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// The days since 1970-01-01 of a civil date, or None for a month or day
/// the calendar does not have.
#[must_use]
pub fn epoch_day(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) || day < 1 {
        return None;
    }
    let shifted = if month <= 2 { year - 1 } else { year };
    let era = shifted.div_euclid(400);
    let of_era = shifted.rem_euclid(400);
    let of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let of_era_days = of_era * 365 + of_era / 4 - of_era / 100 + of_year;
    let days = era * 146_097 + of_era_days - 719_468;
    (civil(days) == (year, month, day)).then_some(days)
}

/// ISO 8601 text as UTC microseconds since 1970, and whether it named a
/// zone; None for text the grammar does not read.
#[must_use]
pub fn iso_micros(text: &str, grammar: Grammar) -> Option<(i64, bool)> {
    let bytes = text.as_bytes();
    let number = |range: std::ops::Range<usize>| -> Option<i64> {
        let digits = bytes.get(range)?;
        digits.iter().try_fold(0_i64, |total, digit| {
            digit
                .is_ascii_digit()
                .then(|| total * 10 + i64::from(digit - b'0'))
        })
    };
    if bytes.len() < 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let days = epoch_day(number(0..4)?, number(5..7)?, number(8..10)?)?;
    let mut micros = days * MICROS_PER_DAY;
    let instant = grammar == Grammar::Instant;
    if bytes.len() == 10 {
        return (!instant).then_some((micros, false));
    }
    let separators: &[u8] = if instant { b"Tt " } else { b"T " };
    if !separators.contains(&bytes[10]) {
        return None;
    }
    let hour = number(11..13).filter(|hour| *hour < 24)?;
    let mut seconds = hour * 3600;
    let mut at = 13;
    let mut whole = false;
    if bytes.get(at) == Some(&b':') {
        seconds += number(at + 1..at + 3).filter(|minute| *minute < 60)? * 60;
        at += 3;
        if bytes.get(at) == Some(&b':') {
            seconds += number(at + 1..at + 3).filter(|second| *second < 60)?;
            at += 3;
            whole = true;
            if bytes.get(at) == Some(&b'.') {
                let digits = bytes[at + 1..]
                    .iter()
                    .take_while(|byte| byte.is_ascii_digit())
                    .count();
                let most = if instant { 9 } else { 6 };
                if !(1..=most).contains(&digits) {
                    return None;
                }
                // Past the microsecond, digits are dropped: an instant is
                // never read as later than the text says.
                let kept = digits.min(6);
                let fraction = number(at + 1..at + 1 + kept)?;
                micros += fraction * 10_i64.pow(u32::try_from(6 - kept).ok()?);
                at += 1 + digits;
            }
        }
    }
    if instant && !whole {
        return None;
    }
    micros += seconds * 1_000_000;
    let offset = match &bytes[at..] {
        [] => return Some((micros, false)),
        [b'Z'] => 0,
        [b'z'] if instant => 0,
        [sign @ (b'+' | b'-'), rest @ ..] => {
            let (hours, minutes) = match rest.len() {
                2 if !instant => (number(at + 1..at + 3)?, 0),
                4 if !instant => (number(at + 1..at + 3)?, number(at + 3..at + 5)?),
                5 if rest[2] == b':' => (number(at + 1..at + 3)?, number(at + 4..at + 6)?),
                _ => return None,
            };
            if hours >= 24 || minutes >= 60 {
                return None;
            }
            let offset = (hours * 60 + minutes) * 60_000_000;
            if *sign == b'+' { offset } else { -offset }
        }
        _ => return None,
    };
    Some((micros - offset, true))
}

/// UTC microseconds since 1970 as RFC 3339, `2026-10-27T08:00:00Z`: whole
/// seconds when the instant has no fraction, else six fractional digits, as
/// Python's `generalforecast.as_of.format` writes it.
#[must_use]
pub fn rfc3339(micros: i64) -> String {
    let seconds = micros.div_euclid(1_000_000);
    let fraction = micros.rem_euclid(1_000_000);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil(days);
    let (hour, minute, second) = (of_day / 3600, of_day % 3600 / 60, of_day % 60);
    let fraction = if fraction == 0 {
        String::new()
    } else {
        format!(".{fraction:06}")
    };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{fraction}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_and_dates_are_inverse() {
        for days in [-719_468, -1, 0, 1, 10_957, 19_782, 2_932_896] {
            let (year, month, day) = civil(days);
            assert_eq!(epoch_day(year, month, day), Some(days));
        }
        assert_eq!(epoch_day(2023, 2, 29), None);
        assert_eq!(epoch_day(2024, 2, 29), Some(19_782));
    }

    #[test]
    fn instants_are_rfc3339_date_times_read_to_the_microsecond() {
        let read = |text| iso_micros(text, Grammar::Instant).map(|(micros, _)| micros);
        assert_eq!(
            read("2026-10-27T08:00:00.866999999Z"),
            read("2026-10-27T08:00:00.866999Z")
        );
        assert_eq!(read("2026-10-27t08:00:00z"), read("2026-10-27T08:00:00Z"));
        for refused in [
            "2026-10-27",
            "2026-10-27T08",
            "2026-10-27T08:00",
            "2026-W44-2",
        ] {
            assert_eq!(read(refused), None, "{refused}");
        }
        assert_eq!(
            rfc3339(1_793_088_000_866_500),
            "2026-10-27T08:00:00.866500Z"
        );
        assert_eq!(rfc3339(1_793_088_000_000_000), "2026-10-27T08:00:00Z");
    }
}
