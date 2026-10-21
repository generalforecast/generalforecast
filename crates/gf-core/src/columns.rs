//! Feature columns: their kinds, the numbers typed Arrow columns become, and
//! the vocabularies that give categories their codes.
//!
//! Each of a table's columns has a kind. Numeric columns are read as float32,
//! with NaN for nulls. Boolean columns become 0 or 1. Datetime columns take
//! dates, timestamps or ISO 8601 strings, never plain numbers, read in UTC to
//! the millisecond, and each becomes five model features, as tabular
//! libraries split dates: days since 1970-01-01 for the trend, then the hour
//! of the day with its fraction, the day of the week from Monday as 0, the day
//! of the month and the month. One float32 of days alone cannot tell apart
//! times minutes apart today; the hour of the day keeps them to well under a
//! second. Categorical columns hold strings, which the table's vocabulary
//! codes in the order it first saw them, or numbers that are codes already.
//! A column holds at most `MAX_WORDS` categories, each word at most
//! `MAX_WORD_BYTES` long, so codes stay exact in float32 and a vocabulary
//! cannot grow without bound. Retrieval columns are numbers that place a row
//! for retrieval and that the model never reads; they follow every other
//! column. The Python client encodes local tables the same way, so a table
//! answers the same wherever it lives.

use std::collections::HashMap;

use arrow_array::{Array, ArrayRef, BooleanArray, Float32Array, Int64Array, StringArray};
use arrow_cast::cast;
use arrow_schema::{DataType, TimeUnit};
use serde::{Deserialize, Serialize};

/// What a column holds, and so how its values become numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColumnKind {
    #[default]
    Numeric,
    Categorical,
    Boolean,
    Datetime,
    Retrieval,
}

/// The model features a datetime column becomes: days since 1970-01-01, hour
/// of the day, day of the week, day of the month, and month.
pub const DATETIME_PARTS: usize = 5;

/// The most categories one column holds: codes run below it, and float32
/// holds every whole number up to 2^24 exactly.
pub const MAX_WORDS: usize = 1 << 24;

/// The most UTF-8 bytes in one category's word.
pub const MAX_WORD_BYTES: usize = 1024;

impl ColumnKind {
    /// How many values of a row a column of this kind becomes.
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            Self::Datetime => DATETIME_PARTS,
            Self::Numeric | Self::Categorical | Self::Boolean | Self::Retrieval => 1,
        }
    }
}

/// A feature column of a table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    #[serde(default)]
    pub kind: ColumnKind,
}

/// Names the fixed columns of writes and queries take, so no feature column may.
pub const RESERVED: [&str; 4] = ["key", "target", "time", "before"];

/// Numeric columns `f0 ..`, for tables made with a feature count alone.
#[must_use]
pub fn numeric(count: usize) -> Vec<Column> {
    (0..count)
        .map(|index| Column {
            name: format!("f{index}"),
            kind: ColumnKind::Numeric,
        })
        .collect()
}

/// Checks a table's columns: one to `most`, named apart from each other and
/// from the fixed columns, with any retrieval columns after one the model
/// reads and every other.
///
/// # Errors
///
/// With what is wrong.
pub fn check(columns: &[Column], most: usize) -> Result<(), String> {
    if columns.is_empty() || columns.len() > most {
        return Err(format!("a table has 1 to {most} feature columns"));
    }
    let mut seen = HashMap::new();
    for column in columns {
        let name = &column.name;
        if name.is_empty() || name.len() > 128 {
            return Err("column names are 1 to 128 characters".to_string());
        }
        if RESERVED.contains(&name.as_str()) {
            return Err(format!(
                "{name:?} names a fixed column; feature columns need other names"
            ));
        }
        if seen.insert(name, ()).is_some() {
            return Err(format!("two columns are named {name:?}"));
        }
    }
    let read = columns.len() - retrieval(columns);
    if read == 0
        || columns[read..]
            .iter()
            .any(|column| column.kind != ColumnKind::Retrieval)
    {
        return Err(
            "retrieval columns come after every other column, and a table needs one \
             other"
                .to_string(),
        );
    }
    Ok(())
}

/// How many retrieval columns there are, which end the columns.
#[must_use]
pub fn retrieval(columns: &[Column]) -> usize {
    columns
        .iter()
        .filter(|column| column.kind == ColumnKind::Retrieval)
        .count()
}

/// How many values of a row the columns become, in all.
#[must_use]
pub fn width(columns: &[Column]) -> usize {
    columns.iter().map(|column| column.kind.width()).sum()
}

/// The model feature each column's features start at.
#[must_use]
pub fn offsets(columns: &[Column]) -> Vec<usize> {
    columns
        .iter()
        .scan(0, |next, column| {
            let start = *next;
            *next += column.kind.width();
            Some(start)
        })
        .collect()
}

/// The model features that hold category codes.
#[must_use]
pub fn categorical(columns: &[Column]) -> Vec<usize> {
    columns
        .iter()
        .zip(offsets(columns))
        .filter(|(column, _)| column.kind == ColumnKind::Categorical)
        .map(|(_, offset)| offset)
        .collect()
}

/// Words a record appends to a column's vocabulary, in code order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Words {
    pub column: usize,
    pub values: Vec<String>,
}

/// The strings of categorical columns as a client sent them, by column.
pub type Strings = Vec<(usize, Vec<Option<String>>)>;

/// A categorical column's words, each with the code of its position.
#[derive(Clone, Debug, Default)]
pub struct Vocabulary {
    words: Vec<String>,
    codes: HashMap<String, u32>,
}

impl Vocabulary {
    /// Every word, in code order.
    #[must_use]
    pub fn words(&self) -> &[String] {
        &self.words
    }

    /// Adds words after the ones held; a word held already keeps its code.
    pub fn extend(&mut self, words: impl IntoIterator<Item = String>) {
        for word in words {
            if !self.codes.contains_key(&word) {
                let code = u32::try_from(self.words.len()).expect("fewer than 2^32 words");
                self.codes.insert(word.clone(), code);
                self.words.push(word);
            }
        }
    }

    pub fn clear(&mut self) {
        self.words.clear();
        self.codes.clear();
    }

    /// The codes of strings, with NaN for nulls and for words not held.
    #[must_use]
    pub fn codes(&self, strings: &[Option<String>]) -> Vec<f32> {
        strings
            .iter()
            .map(|word| {
                word.as_ref()
                    .and_then(|word| self.codes.get(word))
                    .map_or(f32::NAN, |&code| code as f32)
            })
            .collect()
    }

    /// The codes of strings, with NaN for nulls; words not held yet take the
    /// next codes and go into `new`, in that order.
    ///
    /// # Errors
    ///
    /// If a new word is longer than `MAX_WORD_BYTES`, or the column would
    /// hold more than `MAX_WORDS`.
    pub fn assign(
        &self,
        strings: &[Option<String>],
        new: &mut Vec<String>,
    ) -> Result<Vec<f32>, String> {
        self.assign_within(strings, new, MAX_WORDS)
    }

    /// Assigns codes as `assign` does, refusing a word whose code would reach `limit`.
    fn assign_within(
        &self,
        strings: &[Option<String>],
        new: &mut Vec<String>,
        limit: usize,
    ) -> Result<Vec<f32>, String> {
        let mut added: HashMap<&str, usize> = HashMap::new();
        strings
            .iter()
            .map(|word| {
                let Some(word) = word else {
                    return Ok(f32::NAN);
                };
                if let Some(&code) = self.codes.get(word) {
                    return Ok(code as f32);
                }
                if let Some(&code) = added.get(word.as_str()) {
                    return Ok(code as f32);
                }
                if word.len() > MAX_WORD_BYTES {
                    return Err(format!(
                        "a category is at most {MAX_WORD_BYTES} bytes, not {}",
                        word.len()
                    ));
                }
                let code = self.words.len() + new.len();
                if code >= limit {
                    return Err(format!("a column holds at most {limit} categories"));
                }
                new.push(word.clone());
                added.insert(word.as_str(), code);
                Ok(code as f32)
            })
            .collect()
    }
}

/// One decoded column: numbers, times, or the strings of a categorical column
/// for the table's vocabulary to code.
#[derive(Clone, Debug, PartialEq)]
pub enum Cells {
    Numbers(Vec<f32>),
    /// Milliseconds since 1970-01-01 in UTC, None for nulls.
    Times(Vec<Option<i64>>),
    Strings(Vec<Option<String>>),
}

impl Cells {
    /// Row `row`'s model features, appended to `features`; NaN for strings,
    /// which the table's vocabulary codes later.
    pub fn extend_row(&self, row: usize, features: &mut Vec<f32>) {
        match self {
            Self::Numbers(values) => features.push(values[row]),
            Self::Times(times) => {
                features.extend(times[row].map_or([f32::NAN; DATETIME_PARTS], datetime_parts))
            }
            Self::Strings(_) => features.push(f32::NAN),
        }
    }
}

const MILLIS_PER_DAY: i64 = 86_400_000;
const MILLIS_PER_HOUR: f64 = 3_600_000.0;

/// A time's model features: days since 1970-01-01, hour of the day with its
/// fraction, day of the week from Monday as 0, day of the month, and month.
#[must_use]
pub fn datetime_parts(millis: i64) -> [f32; DATETIME_PARTS] {
    let day = millis.div_euclid(MILLIS_PER_DAY);
    let of_day = millis.rem_euclid(MILLIS_PER_DAY);
    let (month, day_of_month) = month_and_day(day);
    [
        (millis as f64 / MILLIS_PER_DAY as f64) as f32,
        (of_day as f64 / MILLIS_PER_HOUR) as f32,
        // 1970-01-01 was a Thursday.
        (day + 3).rem_euclid(7) as f32,
        day_of_month as f32,
        month as f32,
    ]
}

/// The month and day of the month of a day counted from 1970-01-01, in the
/// proleptic Gregorian calendar (Howard Hinnant's `civil_from_days`).
fn month_and_day(day: i64) -> (i64, i64) {
    let shifted = day + 719_468;
    let of_era = shifted.rem_euclid(146_097);
    let year_of_era = (of_era - of_era / 1460 + of_era / 36_524 - of_era / 146_096) / 365;
    let day_of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day_of_month = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    (month, day_of_month)
}

fn is_text(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => true,
        DataType::Dictionary(_, values) => is_text(values),
        _ => false,
    }
}

fn floats(array: &ArrayRef, name: &str) -> Result<Vec<f32>, String> {
    let values = cast(array, &DataType::Float32)
        .map_err(|_| format!("column {name:?} must hold numbers"))?;
    let values = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("cast to float32");
    Ok((0..values.len())
        .map(|index| {
            if values.is_null(index) {
                f32::NAN
            } else {
                values.value(index)
            }
        })
        .collect())
}

/// Whole numbers of an integer array as milliseconds, each counting
/// `per_milli` milliseconds if positive, or a `-per_milli`th of one, rounded down.
fn millis(array: &ArrayRef, name: &str, per_milli: i64) -> Result<Vec<Option<i64>>, String> {
    let values = cast(array, &DataType::Int64)
        .map_err(|_| format!("column {name:?} must hold dates or times"))?;
    let values = values
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("cast to int64");
    Ok((0..values.len())
        .map(|index| {
            (!values.is_null(index)).then(|| {
                let value = values.value(index);
                if per_milli > 0 {
                    value.saturating_mul(per_milli)
                } else {
                    value.div_euclid(-per_milli)
                }
            })
        })
        .collect())
}

/// A column's values as the numbers a model reads, by the column's kind.
///
/// # Errors
///
/// If the Arrow type cannot hold values of the column's kind, or a number in
/// a categorical column is not a code.
pub fn decode(array: &ArrayRef, column: &Column) -> Result<Cells, String> {
    let name = &column.name;
    let data_type = array.data_type();
    if *data_type == DataType::Null {
        return Ok(Cells::Numbers(vec![f32::NAN; array.len()]));
    }
    match column.kind {
        ColumnKind::Numeric | ColumnKind::Retrieval => {
            if is_text(data_type) {
                return Err(format!(
                    "column {name:?} is numeric; it cannot hold strings"
                ));
            }
            floats(array, name).map(Cells::Numbers)
        }
        ColumnKind::Boolean => {
            if let Some(values) = array.as_any().downcast_ref::<BooleanArray>() {
                return Ok(Cells::Numbers(
                    (0..values.len())
                        .map(|index| {
                            if values.is_null(index) {
                                f32::NAN
                            } else {
                                f32::from(u8::from(values.value(index)))
                            }
                        })
                        .collect(),
                ));
            }
            if is_text(data_type) {
                return Err(format!(
                    "column {name:?} is boolean; it holds true, false or null"
                ));
            }
            floats(array, name).map(Cells::Numbers)
        }
        // Times are kept to the millisecond, as clients keep them.
        ColumnKind::Datetime => match data_type {
            DataType::Date32 => millis(array, name, MILLIS_PER_DAY).map(Cells::Times),
            DataType::Date64 => millis(array, name, 1).map(Cells::Times),
            DataType::Timestamp(unit, _) => {
                let per_milli = match unit {
                    TimeUnit::Second => 1_000,
                    TimeUnit::Millisecond => 1,
                    TimeUnit::Microsecond => -1_000,
                    TimeUnit::Nanosecond => -1_000_000,
                };
                millis(array, name, per_milli).map(Cells::Times)
            }
            kind if is_text(kind) => {
                let parsed = cast(array, &DataType::Timestamp(TimeUnit::Millisecond, None))
                    .map_err(|_| format!("column {name:?} holds a time that does not parse"))?;
                millis(&parsed, name, 1).map(Cells::Times)
            }
            // Nothing says whether a number counts days, seconds or milliseconds.
            _ => Err(format!(
                "column {name:?} is datetime; it takes dates, timestamps or ISO 8601 \
                 strings, not plain numbers"
            )),
        },
        ColumnKind::Categorical => {
            if is_text(data_type) {
                let values = cast(array, &DataType::Utf8).map_err(|error| error.to_string())?;
                let values = values
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("cast to utf8");
                return Ok(Cells::Strings(
                    (0..values.len())
                        .map(|index| {
                            (!values.is_null(index)).then(|| values.value(index).to_string())
                        })
                        .collect(),
                ));
            }
            let codes = floats(array, name)?;
            if let Some(code) = codes.iter().find(|code| {
                !code.is_nan()
                    && (code.fract() != 0.0 || **code < 0.0 || **code >= MAX_WORDS as f32)
            }) {
                return Err(format!(
                    "column {name:?} holds {code}, which is not a category code; \
                     codes are whole numbers from 0 below {MAX_WORDS}, or strings"
                ));
            }
            Ok(Cells::Numbers(codes))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Date32Array, Float64Array, TimestampMicrosecondArray, TimestampSecondArray};

    use super::*;

    fn column(kind: ColumnKind) -> Column {
        Column {
            name: "c".to_string(),
            kind,
        }
    }

    #[test]
    fn vocabularies_code_words_in_the_order_they_arrive() {
        let mut vocabulary = Vocabulary::default();
        let mut new = Vec::new();
        let strings = [
            Some("b".to_string()),
            None,
            Some("a".to_string()),
            Some("b".to_string()),
        ];
        let codes = vocabulary.assign(&strings, &mut new).unwrap();
        assert_eq!(codes[0], 0.0);
        assert!(codes[1].is_nan());
        assert_eq!(&codes[2..], &[1.0, 0.0]);
        assert_eq!(new, ["b", "a"]);
        vocabulary.extend(new);
        let later = vocabulary.codes(&[Some("a".to_string()), Some("c".to_string())]);
        assert_eq!(later[0], 1.0);
        assert!(later[1].is_nan());
        assert_eq!(vocabulary.words(), ["b", "a"]);
    }

    #[test]
    fn vocabularies_refuse_long_words_and_more_categories_than_codes_hold() {
        let mut vocabulary = Vocabulary::default();
        let long = Some("x".repeat(MAX_WORD_BYTES + 1));
        assert!(vocabulary.assign(&[long], &mut Vec::new()).is_err());

        let limit = 8;
        vocabulary.extend((0..limit - 1).map(|index| index.to_string()));
        let mut new = Vec::new();
        let last = vocabulary.assign_within(&[Some("last".to_string())], &mut new, limit);
        assert_eq!(last.unwrap(), [(limit - 1) as f32]);
        vocabulary.extend(new);
        let past = vocabulary.assign_within(&[Some("past".to_string())], &mut Vec::new(), limit);
        assert!(past.unwrap_err().contains("at most"));
        // The last code a full column takes is still exact in float32.
        assert_eq!(((MAX_WORDS - 1) as f32) as usize, MAX_WORDS - 1);
        // Queries code unknown words as missing, however many there are.
        assert!(vocabulary.codes(&[Some("past".to_string())])[0].is_nan());
    }

    #[test]
    fn each_kind_reads_its_arrow_types() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![Some("x"), None]));
        assert_eq!(
            decode(&strings, &column(ColumnKind::Categorical)).unwrap(),
            Cells::Strings(vec![Some("x".to_string()), None])
        );
        assert!(decode(&strings, &column(ColumnKind::Numeric)).is_err());
        let codes: ArrayRef = Arc::new(Float64Array::from(vec![Some(2.0), None]));
        let Cells::Numbers(numbers) = decode(&codes, &column(ColumnKind::Categorical)).unwrap()
        else {
            panic!("numbers")
        };
        assert_eq!(numbers[0], 2.0);
        assert!(numbers[1].is_nan());
        let fractions: ArrayRef = Arc::new(Float64Array::from(vec![1.5]));
        assert!(decode(&fractions, &column(ColumnKind::Categorical)).is_err());
        let flags: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        assert_eq!(
            decode(&flags, &column(ColumnKind::Boolean)).unwrap(),
            Cells::Numbers(vec![1.0, 0.0])
        );
        let seconds: ArrayRef = Arc::new(TimestampSecondArray::from(vec![86_400 * 3 / 2]));
        assert_eq!(
            decode(&seconds, &column(ColumnKind::Datetime)).unwrap(),
            Cells::Times(vec![Some(129_600_000)])
        );
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![7]));
        assert_eq!(
            decode(&dates, &column(ColumnKind::Datetime)).unwrap(),
            Cells::Times(vec![Some(7 * 86_400_000)])
        );
        let text: ArrayRef = Arc::new(StringArray::from(vec![Some("1970-01-03T12:00:00"), None]));
        assert_eq!(
            decode(&text, &column(ColumnKind::Datetime)).unwrap(),
            Cells::Times(vec![Some(216_000_000), None])
        );
        // Before 1970, sub-millisecond times round down, as datetime64 does.
        let micros: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![-1]));
        assert_eq!(
            decode(&micros, &column(ColumnKind::Datetime)).unwrap(),
            Cells::Times(vec![Some(-1)])
        );
        // Epoch seconds as plain numbers would be read as days, so they are refused.
        let epoch: ArrayRef = Arc::new(Float64Array::from(vec![1_700_000_000.0]));
        assert!(decode(&epoch, &column(ColumnKind::Datetime)).is_err());
    }

    #[test]
    fn a_time_splits_into_trend_and_calendar_features() {
        // 2026-10-13 15:30:45.250 UTC, a Tuesday.
        let millis = 1_791_905_445_250;
        let [days, hour, weekday, day, month] = datetime_parts(millis);
        assert_eq!(days, (millis as f64 / 86_400_000.0) as f32);
        assert!((hour - (15.0 + 30.0 / 60.0 + 45.25 / 3600.0)).abs() < 1e-5);
        assert_eq!([weekday, day, month], [1.0, 13.0, 10.0]);
        // The last millisecond of 1969 was a Wednesday, December 31st.
        assert_eq!(datetime_parts(-1)[2..], [2.0, 31.0, 12.0]);
        assert_eq!(datetime_parts(-1)[1], (86_399_999.0 / 3_600_000.0) as f32);
        // Leap days count.
        let leap = 19_782 * 86_400_000; // 2024-02-29
        assert_eq!(datetime_parts(leap)[3..], [29.0, 2.0]);
        let columns = [
            column(ColumnKind::Numeric),
            column(ColumnKind::Datetime),
            column(ColumnKind::Categorical),
        ];
        assert_eq!(width(&columns), 7);
        assert_eq!(offsets(&columns), [0, 1, 6]);
        assert_eq!(categorical(&columns), [6]);
    }

    #[test]
    fn columns_need_distinct_names_apart_from_the_fixed_ones() {
        assert!(check(&numeric(3), 10).is_ok());
        assert!(check(&[], 10).is_err());
        assert!(check(&numeric(11), 10).is_err());
        let mut twice = numeric(2);
        twice[1].name = "f0".to_string();
        assert!(check(&twice, 10).is_err());
        let mut fixed = numeric(1);
        fixed[0].name = "key".to_string();
        assert!(check(&fixed, 10).is_err());
    }
}
